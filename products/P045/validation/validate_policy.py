"""Validation: the learned redundancy policy against three baselines.

Discipline, stated before any number
------------------------------------
Three disjoint seed sets (``arqlonghaul.datasets.SEED_SPLIT``):

    fit     trains the learned model's cost-to-go regressor,
    tune    selects every free parameter of every policy -- the learned
            model's forest depth and leaf size, the fixed schedule's two
            actions, the escalating heuristic's base action,
    report  scores all four policies, once.

A baseline whose parameter was chosen on the reporting seeds is not a baseline.
The tune set exists so that the fixed-rate baseline receives exactly the
advantage the learned policy receives, and the reporting seeds are touched once.

The four policies
-----------------
    analytic-fixed   the throughput-optimal fixed (first, later) schedule,
                     computed from the closed form at the stationary mixture
                     per-symbol error probability.  Uses no data at all: no
                     episode is simulated and no seed is read.
    tuned-fixed      the best fixed (first, later) schedule by grid search on
                     the tune seeds.
    escalating       send the next larger increment after each failure, base
                     action chosen on the tune seeds.
    learned-forest   random-forest cost-to-go, trained on the fit seeds,
                     hyperparameters chosen on the tune seeds.

Two channel regimes, both reported whatever the answer
------------------------------------------------------
The structural reason a learned policy might or might not help here: on a long
link the feedback that would reveal the fade state is stale by at least one
round trip, so the policy can only use the channel's statistics plus whatever
correlation survives that staleness.

    long-burst   mean faded sojourn 25 rounds, state autocorrelation 1/e time
                 19.5 rounds.  Stale feedback is still informative.
    short-burst  mean faded sojourn 1.5 rounds, 1/e time 0.56 rounds.  Stale
                 feedback is nearly worthless.

Both have the same stationary marginal per-symbol error probability, so the
analytic baseline is identical in the two and any difference is correlation.

Runtime: about 300 s on one core.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
)

from arqlonghaul.datasets import (  # noqa: E402
    SEED_SPLIT,
    FadeHarqEnv,
    effective_memory_rounds,
    long_burst_env,
    short_burst_env,
)
from arqlonghaul.policy import (  # noqa: E402
    FEATURE_NAMES,
    LearnedRedundancyPolicy,
    Policy,
    analytic_fixed,
    collect_transitions,
    evaluate,
    run_episode,
    tune_escalating,
    tune_fixed,
)

FIT_SEEDS = SEED_SPLIT.fit[:120]
TUNE_SEEDS = SEED_SPLIT.tune[:40]
REPORT_SEEDS = SEED_SPLIT.report[:300]
FIT_FRAMES = 150
TUNE_FRAMES = 120
REPORT_FRAMES = 150
HYPER_GRID = ((6, 40), (10, 20), (14, 10), (None, 5))


def run_env(env: FadeHarqEnv) -> dict[str, dict[str, float]]:
    """Fit, tune and report every policy on one environment."""
    print("=" * 100)
    print(f"ENVIRONMENT: {env.name}")
    print("=" * 100)
    d = env.describe()
    for key in (
        "k",
        "alpha",
        "esn0_good_db",
        "esn0_bad_db",
        "rtt_symbols",
        "max_rounds",
        "drop_penalty",
        "pi_bad",
        "mean_burst_rounds",
        "ber_good",
        "ber_bad",
        "ber_mixture",
    ):
        print(f"  {key:<20}{d[key]}")
    print(f"  {'memory_rounds':<20}{effective_memory_rounds(env):.4f}")
    print(f"  actions (symbols)   {env.actions}")
    print(f"  first-round code rates {[round(env.k / (env.k + a), 4) for a in env.actions]}")
    print()

    results: dict[str, dict[str, float]] = {}
    policies: dict[str, Policy] = {}

    print("-- step 1: analytic baseline, no data used ------------------------------")
    analytic, detail = analytic_fixed(env)
    for key, value in detail.items():
        print(f"  {key:<26}{value:.6g}")
    policies["analytic-fixed"] = analytic

    print()
    print("-- step 2: baselines tuned on the tune seeds ----------------------------")
    print(f"  tune set: {len(TUNE_SEEDS)} seeds x {TUNE_FRAMES} frames")
    tuned, tdetail = tune_fixed(env, TUNE_SEEDS, TUNE_FRAMES)
    for key, value in tdetail.items():
        print(f"  tuned-fixed {key:<20}{value:.6g}")
    policies["tuned-fixed"] = tuned
    esc, edetail = tune_escalating(env, TUNE_SEEDS, TUNE_FRAMES)
    for key, value in edetail.items():
        print(f"  escalating  {key:<20}{value:.6g}")
    policies["escalating"] = esc

    print()
    print("-- step 3: learned policy, fit on the fit seeds -------------------------")
    transitions = collect_transitions(env, FIT_SEEDS, FIT_FRAMES, rng_seed=7)
    print(f"  fit set: {len(FIT_SEEDS)} seeds x {FIT_FRAMES} frames")
    print(f"  transitions collected: {transitions['features'].shape[0]} rows, "
          f"{transitions['features'].shape[1]} features + 1 action column")
    print(f"  features: {', '.join(FEATURE_NAMES)}")
    print(f"  cost target: mean {transitions['cost'].mean():.2f}, "
          f"sd {transitions['cost'].std():.2f} symbol times")
    print()
    print("  hyperparameters selected on the TUNE seeds, never the report seeds:")
    print(f"  {'max_depth':>10}{'min_leaf':>10}{'tune cost/frame':>18}")
    best: tuple[float, LearnedRedundancyPolicy, tuple] | None = None
    for depth, leaf in HYPER_GRID:
        pol = LearnedRedundancyPolicy(
            len(env.actions),
            n_estimators=140,
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=0,
        )
        pol.fit(transitions["features"], transitions["actions"], transitions["cost"])
        score = evaluate(env, pol, TUNE_SEEDS, TUNE_FRAMES)["cost_per_frame"]
        print(f"  {str(depth):>10}{leaf:>10}{score:>18.3f}")
        if best is None or score < best[0]:
            best = (score, pol, (depth, leaf))
    assert best is not None
    print(f"  chosen: max_depth={best[2][0]}, min_samples_leaf={best[2][1]}")
    learned = best[1]
    policies["learned-forest"] = learned

    print()
    print("-- step 4: report, on seeds no policy has seen --------------------------")
    print(f"  report set: {len(REPORT_SEEDS)} seeds x {REPORT_FRAMES} frames "
          f"= {len(REPORT_SEEDS) * REPORT_FRAMES} frames")
    print()
    print(f"{'policy':<17}{'cost/frame':>12}{'stderr':>9}{'goodput':>10}"
          f"{'stderr':>9}{'residual':>11}{'rounds':>8}{'symbols':>9}")
    for name, pol in policies.items():
        r = evaluate(env, pol, REPORT_SEEDS, REPORT_FRAMES)
        results[name] = r
        print(
            f"{name:<17}{r['cost_per_frame']:>12.3f}{r['cost_stderr']:>9.3f}"
            f"{r['goodput']:>10.5f}{r['goodput_stderr']:>9.5f}"
            f"{r['residual_fer']:>11.3e}{r['mean_rounds']:>8.4f}"
            f"{r['symbols_per_frame']:>9.2f}"
        )

    print()
    print("  learned policy against each baseline, paired over the report seeds")
    print(f"  {'baseline':<17}{'goodput diff %':>16}{'cost diff %':>14}"
          f"{'z on cost':>12}{'verdict':>12}")
    lr = results["learned-forest"]
    for name in ("analytic-fixed", "tuned-fixed", "escalating"):
        b = results[name]
        gp = 100 * (lr["goodput"] / b["goodput"] - 1)
        cost = 100 * (lr["cost_per_frame"] / b["cost_per_frame"] - 1)
        se = math.sqrt(lr["cost_stderr"] ** 2 + b["cost_stderr"] ** 2)
        z = (lr["cost_per_frame"] - b["cost_per_frame"]) / se if se > 0 else 0.0
        if abs(z) < 2.0:
            verdict = "tie"
        elif z < 0:
            verdict = "learned"
        else:
            verdict = name
        print(f"  {name:<17}{gp:>16.3f}{cost:>14.3f}{z:>12.2f}{verdict:>12}")
    print()
    print("  z is on cost per frame, which is the quantity the learned policy")
    print("  minimises; the standard errors are across report seeds. |z| < 2 is")
    print("  reported as a tie, because at this episode length a difference")
    print("  smaller than two standard errors is not a result.")

    print()
    print("-- step 5: uncertainty output -------------------------------------------")
    _, tr = run_episode(env, learned, 400, int(REPORT_SEEDS[0]), record=True)
    feats = tr["features"]
    uniq = np.unique(feats, axis=0)
    margins = []
    stds = []
    for row in uniq:
        conf = learned.decision_confidence(row)
        margins.append(conf["margin_sigma"])
        stds.append(conf["std"])
    margins_arr = np.asarray(margins)
    stds_arr = np.asarray(stds)
    print(f"  distinct observed states on one report seed: {uniq.shape[0]}")
    print(f"  memoisation cache entries after the report run: {learned.cache_size}")
    print("  across-tree sd of the chosen action's predicted cost:")
    print(f"    median {np.median(stds_arr):.2f}, 10th pct {np.percentile(stds_arr, 10):.2f}, "
          f"90th pct {np.percentile(stds_arr, 90):.2f} symbol times")
    print("  decision margin to the runner-up action, in pooled sd:")
    print(f"    median {np.median(margins_arr):.2f}, "
          f"fraction below 1 sd: {np.mean(margins_arr < 1.0):.3f}")
    print("  A margin below about one standard deviation means the model cannot")
    print("  distinguish its two best actions at that state; the fraction above")
    print("  is how often that happens, and it is the honest reading of how much")
    print("  of the policy is a real decision.")
    print()
    print("  worked example of the uncertainty output at one state:")
    row = uniq[uniq.shape[0] // 2]
    mean, std = learned.predict_with_uncertainty(row)
    print(f"    state: {dict(zip(FEATURE_NAMES, [round(float(v), 4) for v in row], strict=True))}")
    print(f"    {'action symbols':>16}{'predicted cost':>16}{'across-tree sd':>16}")
    for a, mu, sd in zip(env.actions, mean, std, strict=True):
        print(f"    {a:>16}{mu:>16.2f}{sd:>16.2f}")
    conf = learned.decision_confidence(row)
    print(f"    chosen action {env.actions[int(conf['action'])]} symbols, "
          f"margin {conf['margin']:.2f} symbol times "
          f"= {conf['margin_sigma']:.2f} pooled sd")

    print()
    print("-- step 6: action usage -------------------------------------------------")
    print(f"  {'policy':<17}" + "".join(f"{f'a={a}':>10}" for a in env.actions))
    for name, pol in policies.items():
        stats, _ = run_episode(env, pol, 2000, int(REPORT_SEEDS[1]))
        total = stats.action_counts.sum()
        print(
            f"  {name:<17}"
            + "".join(f"{c / total:>10.4f}" for c in stats.action_counts)
        )
    print("  (fraction of rounds at each increment, one report seed, 2000 frames)")
    print()
    return results


def main() -> None:
    print("seed discipline:", SEED_SPLIT.summary())
    print(f"fit seeds used {len(FIT_SEEDS)}, tune {len(TUNE_SEEDS)}, "
          f"report {len(REPORT_SEEDS)}")
    print("the three sets are disjoint by construction; SeedSplit raises if not")
    print()
    all_results = {}
    for env in (long_burst_env(), short_burst_env()):
        all_results[env.name] = run_env(env)
        print()

    print("=" * 100)
    print("OVERALL, STATED WITHOUT SPIN")
    print("=" * 100)
    print(f"{'environment':<36}{'best policy by cost':<20}{'learned vs best %':>19}"
          f"{'|z|':>8}")
    for env_name, res in all_results.items():
        best_name = min(res, key=lambda k: res[k]["cost_per_frame"])
        lr = res["learned-forest"]
        b = res[best_name]
        pct = 100 * (lr["cost_per_frame"] / b["cost_per_frame"] - 1)
        se = math.sqrt(lr["cost_stderr"] ** 2 + b["cost_stderr"] ** 2)
        z = abs(lr["cost_per_frame"] - b["cost_per_frame"]) / se if se > 0 else 0.0
        print(f"{env_name:<36}{best_name:<20}{pct:>19.3f}{z:>8.2f}")
    print()
    print("Lower cost per frame is better. A positive 'learned vs best' means")
    print("the learned policy is worse than the best policy in that row.")


if __name__ == "__main__":
    main()
