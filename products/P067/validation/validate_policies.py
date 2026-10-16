"""The head-to-head: fixed patterns, Bayesian-greedy and the learned policy.

    python validation/validate_policies.py > validation/validate_policies_output.txt

Protocol. 300 scenarios are generated from ``data/scenarios.json``'s declared
ranges with seed 20261016 and split by id into 195 training and 105 held-out
scenarios. Training rows come only from the training split, under an
epsilon-greedy Bayesian-greedy behaviour policy restricted to the same
12-cell shortlist the learned policy re-ranks at decision time. Every policy
is then run on the same held-out scenarios, the same drawn target cells and
the same per-episode seeds, so the comparison is paired.

The headline statistic is the mean episode time with non-acquisition charged
at the scenario deadline (three nominal pass periods).

The expected result, stated before the run: Bayesian-greedy is hard to beat.
Whatever the numbers say is what this file says.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from _common import rule

from acqsearch import (
    BayesianGreedyPolicy,
    FixedPatternPolicy,
    LearnedScanPolicy,
    build_environment,
    build_training_set,
    evaluate_policies,
    generate_scenarios,
    paired_difference,
    run_episode,
    sample_target_cells,
    score_policy,
    split_scenarios,
    summarise,
    winner,
)
from acqsearch.policy import FEATURE_NAMES, build_features, initial_posterior

N_SCENARIOS = 300
SEED = 20261016
TRAIN_FRACTION = 0.65
TRAIN_EPISODES = 2
TARGETS = 8
EVAL_SEED = 20261017
ABLATION_SCENARIOS = 40
CALIBRATION_SCENARIOS = 25


def _run_learned(scenarios, policy, targets_per_scenario, seed):
    """Run one policy over the held-out scenarios with the benchmark's seeds."""
    results = []
    for scenario in scenarios:
        environment, _orders = build_environment(scenario)
        targets = sample_target_cells(
            environment.grid,
            targets_per_scenario,
            np.random.default_rng(seed + 104729 * scenario.scenario_id),
        )
        for index, target in enumerate(targets):
            rng = np.random.default_rng(seed + 15485863 * scenario.scenario_id + 31 * index)
            results.append(run_episode(environment, policy, int(target), rng))
    return results


def main() -> int:
    """Write the policy evidence."""
    out = sys.stdout
    started = time.time()
    scenarios = generate_scenarios(N_SCENARIOS, SEED)
    train, test = split_scenarios(scenarios, TRAIN_FRACTION)

    rule(out, "acqsearch policy head-to-head on held-out scenarios")
    out.write(f"scenarios {N_SCENARIOS}   train {len(train)}   held out {len(test)}\n")
    out.write(f"scenario seed {SEED}   evaluation seed {EVAL_SEED}   ")
    out.write(f"target draws per held-out scenario {TARGETS}\n")

    build_started = time.time()
    features, target = build_training_set(train, episodes_per_scenario=TRAIN_EPISODES)
    build_seconds = time.time() - build_started
    fit_started = time.time()
    learned = LearnedScanPolicy().fit(features, target)
    fit_seconds = time.time() - fit_started
    out.write(f"training rows {features.shape[0]} x {features.shape[1]} features ")
    out.write(f"(rollout {build_seconds:.1f} s, fit {fit_seconds:.1f} s)\n")
    out.write(f"target: remaining time / nominal period, mean {target.mean():.4f}, ")
    out.write(f"max {target.max():.4f}\n")

    evaluation_started = time.time()
    results = evaluate_policies(
        test, learned, targets_per_scenario=TARGETS, seed=EVAL_SEED, include_all_patterns=True
    )
    scores = summarise(results)
    out.write(f"evaluation {time.time() - evaluation_started:.1f} s, ")
    out.write(f"{len(results['learned'])} episodes per policy\n")

    out.write("\n1. Held-out performance, every policy on identical episodes\n")
    header = f"{'policy':<22}{'mean_s':>10}{'s.e._s':>9}{'median_s':>10}"
    out.write(header + f"{'acquired':>10}{'dwells':>9}\n")
    for score in sorted(scores.values(), key=lambda s: s.mean_time_s):
        out.write(
            f"{score.name:<22}{score.mean_time_s:>10.4f}{score.standard_error_s:>9.4f}"
            f"{score.median_time_s:>10.4f}{score.acquisition_rate:>10.4f}"
            f"{score.mean_dwells:>9.1f}\n"
        )
    out.write(f"\nwinner by mean censored time: {winner(scores)}\n")

    out.write("\n2. Paired differences on identical episodes (negative favours the first)\n")
    out.write(f"{'comparison':<44}{'mean diff_s':>13}{'s.e._s':>10}{'z':>8}\n")
    pairs = [
        ("learned", "bayesian_greedy"),
        ("learned", "fixed_spiral"),
        ("bayesian_greedy", "fixed_spiral"),
        ("bayesian_greedy", "fixed_raster"),
        ("fixed_spiral", "fixed_raster"),
        ("fixed_spiral", "fixed_random_restart"),
    ]
    for a, b in pairs:
        difference = paired_difference(a, results[a], b, results[b])
        out.write(
            f"{a + ' - ' + b:<44}{difference.mean_difference_s:>+13.4f}"
            f"{difference.standard_error_s:>10.4f}{difference.z_score:>+8.2f}\n"
        )

    out.write("\n3. Shortlist ablation for the learned policy\n")
    out.write(
        "The learned policy re-ranks the top-k cells by Bayesian-greedy score. k = 89\n"
        "is the whole grid, which is the unconstrained learned policy. The model is\n"
        "the same fit in every row; only the candidate set at decision time changes,\n"
        "so k = 12 matches the behaviour policy that produced the training rows and\n"
        "the others do not. That mismatch is part of what the larger k values cost.\n"
    )
    out.write(
        f"Run on the first {ABLATION_SCENARIOS} held-out scenarios rather than all "
        f"{len(test)}, to keep\n"
        "the whole script inside the three-minute compute budget. The two reference\n"
        "policies are re-run on the same subset so the column is internally\n"
        "comparable; the k = 12 row is therefore not identical to the table above,\n"
        "and both are reported rather than one being substituted for the other.\n"
        "k = 1 leaves the learned model exactly one candidate, which is the cell\n"
        "Bayesian-greedy would have chosen, so that row IS the baseline.\n"
    )
    subset = test[:ABLATION_SCENARIOS]
    out.write(f"{'policy':>22}{'mean_s':>10}{'s.e._s':>9}{'median_s':>10}{'acquired':>10}\n")
    for k in (1, 4, 8, 12, 24, 89):
        learned.shortlist = k
        score = score_policy(
            f"learned k={k}", _run_learned(subset, learned, TARGETS, EVAL_SEED)
        )
        out.write(
            f"{score.name:>22}{score.mean_time_s:>10.4f}{score.standard_error_s:>9.4f}"
            f"{score.median_time_s:>10.4f}{score.acquisition_rate:>10.4f}\n"
        )
    learned.shortlist = 12
    for name, policy in (
        ("bayesian_greedy", BayesianGreedyPolicy()),
        ("fixed_spiral", None),
    ):
        if policy is None:
            episodes = []
            for scenario in subset:
                environment, orders = build_environment(scenario)
                targets = sample_target_cells(
                    environment.grid,
                    TARGETS,
                    np.random.default_rng(EVAL_SEED + 104729 * scenario.scenario_id),
                )
                fixed = FixedPatternPolicy(orders["spiral"], name="fixed_spiral")
                for index, cell in enumerate(targets):
                    episodes.append(
                        run_episode(
                            environment,
                            fixed,
                            int(cell),
                            np.random.default_rng(
                                EVAL_SEED + 15485863 * scenario.scenario_id + 31 * index
                            ),
                        )
                    )
        else:
            episodes = _run_learned(subset, policy, TARGETS, EVAL_SEED)
        score = score_policy(name, episodes)
        out.write(
            f"{score.name:>22}{score.mean_time_s:>10.4f}{score.standard_error_s:>9.4f}"
            f"{score.median_time_s:>10.4f}{score.acquisition_rate:>10.4f}\n"
        )

    out.write("\n4. Calibration of the learned policy's prediction interval\n")
    out.write(
        "Each held-out decision taken by the learned policy carries a (0.1, 0.9)\n"
        "pinball-quantile interval on the remaining time. A calibrated interval\n"
        "covers the realised remaining time 80 percent of the time.\n"
    )
    covered = 0
    total = 0
    widths = []
    confidences = []
    for scenario in test[:CALIBRATION_SCENARIOS]:
        environment, _orders = build_environment(scenario)
        targets = sample_target_cells(
            environment.grid, 2, np.random.default_rng(EVAL_SEED + 104729 * scenario.scenario_id)
        )
        for index, cell in enumerate(targets):
            record: list = []
            outcome = run_episode(
                environment,
                learned,
                int(cell),
                np.random.default_rng(
                    EVAL_SEED + 15485863 * scenario.scenario_id + 31 * index
                ),
                record=record,
            )
            posterior = initial_posterior(environment.grid)
            visits = np.zeros(environment.grid.n_cells, dtype=np.int64)
            current = np.zeros(2)
            for step, (_features, chosen, elapsed) in enumerate(record):
                decision = learned.decide(
                    environment, posterior, current, visits, step, elapsed
                )
                realised = max(0.0, outcome.time_s - elapsed)
                low, high = decision.interval_s
                covered += int(low <= realised <= high)
                total += 1
                widths.append(high - low)
                confidences.append(decision.confidence)
                posterior_cell = int(chosen)
                visits[posterior_cell] += 1
                current = environment.grid.centres_rad[posterior_cell]
                q = float(environment.cell_detection[posterior_cell])
                from acqsearch import update_posterior

                posterior = update_posterior(posterior, posterior_cell, q)
    out.write(f"decisions scored                 {total}\n")
    out.write(f"empirical interval coverage      {covered / total:.4f}  (nominal 0.8000)\n")
    out.write(f"mean interval width              {float(np.mean(widths)):.4f} s\n")
    out.write(f"mean reported confidence         {float(np.mean(confidences)):.4f}\n")

    out.write("\n5. Feature columns used by the learned policy\n")
    for index, name in enumerate(FEATURE_NAMES):
        out.write(f"{index:>4d}  {name}\n")
    sample_environment, _ = build_environment(test[0])
    sample = build_features(
        sample_environment,
        initial_posterior(sample_environment.grid),
        np.zeros(2),
        np.zeros(sample_environment.grid.n_cells, dtype=np.int64),
        0,
        0.0,
    )
    out.write("feature matrix shape at the first decision of the first held-out scenario: ")
    out.write(f"{sample.shape}\n")

    rule(out)
    out.write(f"total wall clock {time.time() - started:.1f} s on 2 cores\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
