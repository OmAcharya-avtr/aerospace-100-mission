"""Planning under predicted-duration uncertainty: what robustness buys and costs.

Four estimators plan the same instances and every plan is then scored against
the same realised durations and energies:

``oracle``      plans with the realised values. Not achievable; an upper bound.
``analytic``    plans with the declared analytic model.
``ml-point``    plans with the learned point predictions.
``ml-robust``   plans with the upper ends of the learned conformal intervals.

Scoring rule, from :mod:`activityplan.robust`: a realised plan that violates any
declared constraint scores zero, not partial credit. The same rule is applied to
all four modes, so the comparison between them is unaffected by the choice; only
the absolute levels are.

What makes this measurement worth having is that it scores the *plan*, not the
predictor. A predictor with a lower MAE can still produce worse plans if its
errors land on the activities that matter, and a predictor with a wider interval
can produce better plans by refusing to pack the timeline tight.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from _bootstrap import Tee, add_src_to_path, environment_lines

add_src_to_path()

from activityplan.features import (  # noqa: E402
    analytic_duration_s,
    feature_matrix,
    realise,
    sample_requests,
)
from activityplan.instances import instance_family  # noqa: E402
from activityplan.predictor import DurationEnergyPredictor  # noqa: E402
from activityplan.robust import (  # noqa: E402
    PLANNING_MODES,
    compare_modes,
    estimate_for_mode,
    realise_spec,
)

OUTPUT = Path(__file__).with_name("validate_robust_output.txt")
N_TRAIN = 1000
PREDICTOR_SEED = 61
ALPHA = 0.10
FAMILY_SEED = 9000
FAMILY_SIZE = 12
REALISATIONS = 3


def main() -> int:
    say = Tee(OUTPUT)
    started = time.perf_counter()
    say("activityplan -- validate_robust.py")
    say("four planning modes under realised durations, on identical instances")
    for line in environment_lines():
        say(line)
    say(f"predictor        : {N_TRAIN} training activities, seed {PREDICTOR_SEED}, "
        f"normalised conformal, alpha {ALPHA}")
    say(f"instances        : {FAMILY_SIZE} from seed {FAMILY_SEED}, family defaults")
    say(f"realisations     : {REALISATIONS} per instance, so "
        f"{FAMILY_SIZE * REALISATIONS} scored plans per mode")
    say("planner          : greedy, so that all four modes pay the same planning cost")

    rng = np.random.default_rng(PREDICTOR_SEED)
    requests = sample_requests(N_TRAIN, rng)
    realised = [realise(request, rng) for request in requests]
    t0 = time.perf_counter()
    predictor = DurationEnergyPredictor(
        alpha=ALPHA, mode="normalised", n_estimators=160, random_state=PREDICTOR_SEED
    ).fit(
        feature_matrix(list(requests)),
        np.array([item.duration_s for item in realised], dtype=float),
        np.array([item.energy_wh for item in realised], dtype=float),
    )
    say(f"predictor fit    : {time.perf_counter() - t0:.2f} s")

    say.rule("check 1: how far each estimator's duration sits from the realised one")
    say("signed error = estimate - realised, in seconds. A negative mean is optimism,")
    say("which is what makes a plan fail; ml-robust is deliberately pessimistic.")
    say("")
    say(
        f"{'mode':11s} {'mean err/s':>11s} {'median err/s':>13s} {'p05 err/s':>10s} "
        f"{'p95 err/s':>10s} {'frac optimistic':>16s}"
    )
    errors: dict[str, list[float]] = {mode: [] for mode in PLANNING_MODES}
    for spec in instance_family(FAMILY_SIZE, FAMILY_SEED):
        truth = realise_spec(spec, np.random.default_rng(spec.seed + 500))
        for mode in PLANNING_MODES:
            estimate = estimate_for_mode(spec, mode, predictor=predictor, truth=truth)
            errors[mode].extend((estimate.durations_s - truth[0]).tolist())
    for mode in PLANNING_MODES:
        array = np.array(errors[mode], dtype=float)
        say(
            f"{mode:11s} {float(np.mean(array)):11.3f} {float(np.median(array)):13.3f} "
            f"{float(np.percentile(array, 5)):10.3f} "
            f"{float(np.percentile(array, 95)):10.3f} "
            f"{float(np.mean(array < 0.0)):16.4f}"
        )

    say.rule("check 2: realised feasibility and realised objective, per mode")
    say(
        f"{'mode':11s} {'plans':>6s} {'feasible':>9s} {'rate':>7s} {'mean placed':>12s} "
        f"{'mean planned':>13s} {'mean realised':>14s} {'realised/oracle':>16s}"
    )
    tally: dict[str, dict[str, float]] = {
        mode: {"n": 0.0, "feasible": 0.0, "placed": 0.0, "planned": 0.0, "realised": 0.0}
        for mode in PLANNING_MODES
    }
    failure_reasons: dict[str, Counter[str]] = {mode: Counter() for mode in PLANNING_MODES}
    per_instance: list[tuple[str, int, dict[str, tuple[bool, float, int]]]] = []
    for spec in instance_family(FAMILY_SIZE, FAMILY_SEED):
        for repeat in range(REALISATIONS):
            draw_seed = spec.seed * 100 + repeat
            outcomes = compare_modes(
                spec, np.random.default_rng(draw_seed), predictor=predictor, planner="greedy"
            )
            row: dict[str, tuple[bool, float, int]] = {}
            for outcome in outcomes:
                entry = tally[outcome.mode]
                entry["n"] += 1.0
                entry["feasible"] += 1.0 if outcome.feasible else 0.0
                entry["placed"] += float(outcome.n_placed)
                entry["planned"] += (
                    outcome.planned_value if math.isfinite(outcome.planned_value) else 0.0
                )
                entry["realised"] += outcome.realised_value
                if not outcome.feasible:
                    failure_reasons[outcome.mode][outcome.binding] += 1
                row[outcome.mode] = (outcome.feasible, outcome.realised_value, outcome.n_placed)
            per_instance.append((spec.name, repeat, row))
    oracle_realised = tally["oracle"]["realised"]
    for mode in PLANNING_MODES:
        entry = tally[mode]
        n = entry["n"]
        say(
            f"{mode:11s} {int(n):6d} {int(entry['feasible']):9d} "
            f"{entry['feasible'] / n:7.4f} {entry['placed'] / n:12.3f} "
            f"{entry['planned'] / n:13.3f} {entry['realised'] / n:14.3f} "
            f"{entry['realised'] / max(oracle_realised, 1e-12):16.4f}"
        )
    say("")
    say("why the infeasible plans failed, by binding constraint")
    for mode in PLANNING_MODES:
        if not failure_reasons[mode]:
            say(f"  {mode:11s} none")
            continue
        parts = ", ".join(f"{name} x{count}" for name, count in failure_reasons[mode].most_common())
        say(f"  {mode:11s} {parts}")

    say.rule("check 3: per-instance detail, so the aggregate is not the only view")
    say(
        f"{'instance':14s} {'rep':>4s} "
        + " ".join(f"{mode:>22s}" for mode in PLANNING_MODES)
    )
    say(f"{'':14s} {'':>4s} " + " ".join(f"{'feas/value/placed':>22s}" for _ in PLANNING_MODES))
    for name, repeat, row in per_instance:
        cells = []
        for mode in PLANNING_MODES:
            feasible, value, placed = row[mode]
            cells.append(f"{('y' if feasible else 'n'):>4s} {value:9.1f} {placed:7d}")
        say(f"{name:14s} {repeat:4d} " + " ".join(cells))

    say.rule("check 4: what the robust plan costs when nothing goes wrong")
    say("the robust plan books the conformal upper bound on every duration and energy, so")
    say("it places fewer activities. The cost is the planned objective it gives up")
    say("against the point plan; the benefit is in the feasibility rate above.")
    say("")
    point = tally["ml-point"]
    robust = tally["ml-robust"]
    oracle = tally["oracle"]
    analytic = tally["analytic"]
    say(
        f"mean activities placed    : oracle {oracle['placed'] / oracle['n']:.3f}, "
        f"analytic {analytic['placed'] / analytic['n']:.3f}, "
        f"ml-point {point['placed'] / point['n']:.3f}, "
        f"ml-robust {robust['placed'] / robust['n']:.3f}"
    )
    say(
        f"mean realised objective   : oracle {oracle['realised'] / oracle['n']:.3f}, "
        f"analytic {analytic['realised'] / analytic['n']:.3f}, "
        f"ml-point {point['realised'] / point['n']:.3f}, "
        f"ml-robust {robust['realised'] / robust['n']:.3f}"
    )
    say(
        f"feasibility rate          : oracle {oracle['feasible'] / oracle['n']:.4f}, "
        f"analytic {analytic['feasible'] / analytic['n']:.4f}, "
        f"ml-point {point['feasible'] / point['n']:.4f}, "
        f"ml-robust {robust['feasible'] / robust['n']:.4f}"
    )
    say("")
    best = max(
        ("analytic", analytic["realised"]),
        ("ml-point", point["realised"]),
        ("ml-robust", robust["realised"]),
        key=lambda pair: pair[1],
    )
    say(f"best achievable mode on realised objective : {best[0]}")
    say(
        "oracle is unachievable by construction and is reported only as the ceiling."
    )

    say.rule("check 5: why duration robustness is ANTI-robust for the recorder")
    say("The result above is not what a reader would expect, so here is the mechanism,")
    say("measured. An activity's data volume is DECLARED, not predicted. Its production")
    say("rate is therefore data_mbit / duration_s, which is INVERSELY proportional to the")
    say("duration. Booking the conformal upper bound on the duration makes the plan")
    say("assume a LOW production rate; the realised duration is shorter, so the real rate")
    say("is HIGHER, and the recorder fills faster than the plan allowed for.")
    say("")
    say("Planning against an upper bound on duration is conservative for time and for")
    say("energy and anti-conservative for the recorder. A duration-robust plan is not a")
    say("resource-robust plan.")
    say("")
    say(
        f"{'mode':11s} {'planned rate':>13s} {'realised rate':>14s} {'rate ratio':>11s} "
        f"{'frac rate underbooked':>22s}"
    )
    for mode in PLANNING_MODES:
        planned_rates: list[float] = []
        realised_rates: list[float] = []
        for spec in instance_family(FAMILY_SIZE, FAMILY_SEED):
            truth = realise_spec(spec, np.random.default_rng(spec.seed + 500))
            estimate = estimate_for_mode(spec, mode, predictor=predictor, truth=truth)
            volumes = np.array([request.data_mbit for request in spec.requests], dtype=float)
            planned_rates.extend((volumes / estimate.durations_s).tolist())
            realised_rates.extend((volumes / truth[0]).tolist())
        planned = np.array(planned_rates, dtype=float)
        actual = np.array(realised_rates, dtype=float)
        say(
            f"{mode:11s} {float(np.mean(planned)):13.5f} {float(np.mean(actual)):14.5f} "
            f"{float(np.mean(actual / planned)):11.4f} "
            f"{float(np.mean(actual > planned)):22.4f}"
        )
    say("")
    say("rates are Mbit/s. A rate ratio above 1 means the realised production outran the")
    say("plan's assumption. ml-robust has the largest ratio of the three achievable modes,")
    say("which is the quantitative statement of the mechanism above, and it is why")
    say("ml-robust fails recorder-capacity more often than ml-point does.")

    say.rule("check 6: the analytic duration model's own bias, for reference")
    reference_rng = np.random.default_rng(4242)
    sample = sample_requests(600, reference_rng)
    truth = np.array([realise(r, reference_rng).duration_s for r in sample], dtype=float)
    analytic_duration = np.array([analytic_duration_s(r) for r in sample], dtype=float)
    signed = analytic_duration - truth
    say(f"analytic duration mean signed error : {float(np.mean(signed)):.4f} s")
    say(f"analytic duration fraction optimistic: {float(np.mean(signed < 0.0)):.4f}")
    say("the analytic model is optimistic on average because the generator's two omitted")
    say("terms are both non-negative, which is why its plans fail more often than the")
    say("learned ones when they fail.")
    say("")
    say("Taken with check 5, the two failure mechanisms are opposite: the analytic model")
    say("underbooks DURATION and breaks the battery, while the duration-robust plan")
    say("underbooks the data RATE and breaks the recorder. Neither is fixed by a better")
    say("duration predictor alone, and this package does not claim to fix either.")

    say.rule("summary")
    for mode in PLANNING_MODES:
        entry = tally[mode]
        say(
            f"{mode:11s} feasible {entry['feasible'] / entry['n']:.4f}  "
            f"realised {entry['realised'] / entry['n']:.3f}  "
            f"placed {entry['placed'] / entry['n']:.3f}"
        )
    say(f"wall clock  {time.perf_counter() - started:.1f} s")
    say("this script has no pass criterion: every number is a measurement of the")
    say("robustness trade and all of them are reported, including the ones that favour")
    say("the analytic baseline.")
    say(f"raw output written to {OUTPUT.name}")
    say.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
