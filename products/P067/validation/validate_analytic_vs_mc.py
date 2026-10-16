"""Expected acquisition time: the analytic model against seeded Monte Carlo.

    python validation/validate_analytic_vs_mc.py > validation/validate_analytic_vs_mc_output.txt

Three comparisons, in increasing order of what they put at risk.

1. The analytic expectation against a Monte Carlo that samples exactly the
   assumptions the analytic model integrates (A1-A4 of
   :mod:`acqsearch.acquisition`). Any disagreement beyond the Monte Carlo
   standard error is an error in the formula.
2. The analytic deadline probability against the same Monte Carlo, and against
   a Monte Carlo in which the false-alarm penalty is drawn rather than charged
   as an expectation. The mean is unchanged by that; the deadline probability
   is not, and the size of the change is reported.
3. The analytic expectation against a Monte Carlo that drops the cell
   abstraction entirely: a continuous target position and the pointwise
   detection probability at the true offset. The difference is the cost of the
   abstraction and it is reported, not absorbed.
"""

from __future__ import annotations

import sys

import numpy as np
from _common import DWELL_S, SEED, VERIFICATION_TIME_S, reference, rule

from acqsearch import (
    build_schedule,
    cell_detection_probabilities,
    cell_visit_order,
    expected_acquisition_time,
    monte_carlo_acquisition,
    monte_carlo_acquisition_continuous,
    probability_acquired_by,
    standard_paths,
)

TRIALS = 200000
CONTINUOUS_TRIALS = 40000
SEEDS = (20261016, 20261017, 20261018, 20261019, 20261020)


def main() -> int:
    """Write the analytic / Monte Carlo evidence."""
    out = sys.stdout
    failures = 0
    uncertainty, geometry, detector, grid = reference()
    q = cell_detection_probabilities(grid, geometry, detector, DWELL_S, uncertainty)
    f = detector.false_alarm_probability(DWELL_S)
    orders = {
        name: cell_visit_order(path, grid) for name, path in standard_paths(grid).items()
    }

    rule(out, "acqsearch expected acquisition time: analytic against Monte Carlo")
    out.write(f"cells {grid.n_cells}   prior mass in grid {grid.prior_mass.sum():.6f}   ")
    out.write(f"outside {grid.mass_outside:.6f}\n")
    out.write(f"dwell {DWELL_S:.6e} s   P_fa {f:.9f}   verification {VERIFICATION_TIME_S} s\n")
    mean_q = float(np.average(q, weights=grid.prior_mass))
    out.write(f"cell detection probability: mean {mean_q:.6f}, ")
    out.write(f"min {q.min():.6f}, max {q.max():.6f}\n")

    out.write("\n1. E[T | target in grid]: analytic against Monte Carlo over five seeds\n")
    header = f"{'pattern':<16}{'period_s':>11}{'analytic_s':>12}{'MC mean_s':>12}"
    out.write(header + f"{'MC s.e.':>10}{'difference':>12}{'|d|/s.e.':>10}\n")
    for name, order in orders.items():
        schedule = build_schedule(grid, order, geometry, DWELL_S, f, VERIFICATION_TIME_S)
        analytic = expected_acquisition_time(grid, schedule, q).expected_time_s
        means = []
        for seed in SEEDS:
            means.append(
                monte_carlo_acquisition(grid, schedule, q, n_trials=TRIALS, seed=seed).mean_time_s
            )
        pooled = float(np.mean(means))
        se = float(np.std(means, ddof=1) / np.sqrt(len(means)))
        difference = pooled - analytic
        ratio = abs(difference) / se if se > 0 else float("inf")
        failures += ratio > 4.0
        out.write(
            f"{name:<16}{schedule.period_s:>11.6f}{analytic:>12.6f}{pooled:>12.6f}"
            f"{se:>10.6f}{difference:>+12.6f}{ratio:>10.2f}\n"
        )

    out.write("\n2. P(acquired by a deadline of two pass periods), spiral order\n")
    schedule = build_schedule(grid, orders["spiral"], geometry, DWELL_S, f, VERIFICATION_TIME_S)
    deadline = 2.0 * schedule.period_s
    analytic_p = probability_acquired_by(grid, schedule, q, deadline)
    deterministic = monte_carlo_acquisition(
        grid, schedule, q, n_trials=TRIALS, seed=SEED, deadline_s=deadline
    )
    stochastic = monte_carlo_acquisition(
        grid,
        schedule,
        q,
        n_trials=TRIALS,
        seed=SEED,
        deadline_s=deadline,
        stochastic_false_alarms=True,
        false_alarm_probability=f,
        verification_time_s=VERIFICATION_TIME_S,
    )
    rate_se = float(np.sqrt(analytic_p * (1.0 - analytic_p) / TRIALS))
    out.write(f"deadline                                   {deadline:.6f} s\n")
    out.write(f"analytic P(acquired)                       {analytic_p:.6f}\n")
    out.write("Monte Carlo, expected false-alarm charge   ")
    out.write(f"{deterministic.probability_by_deadline:.6f}  (s.e. {rate_se:.6f})\n")
    out.write("Monte Carlo, drawn false alarms            ")
    out.write(f"{stochastic.probability_by_deadline:.6f}\n")
    out.write(
        f"effect of making the penalty stochastic    "
        f"{stochastic.probability_by_deadline - deterministic.probability_by_deadline:+.6f}\n"
    )
    out.write(f"mean time, expected charge                 {deterministic.mean_time_s:.6f} s\n")
    out.write(f"mean time, drawn false alarms              {stochastic.mean_time_s:.6f} s\n")
    failures += abs(deterministic.probability_by_deadline - analytic_p) > 4.0 * rate_se

    out.write("\n3. Cost of the cell abstraction, spiral order\n")
    continuous = monte_carlo_acquisition_continuous(
        grid,
        schedule,
        geometry,
        detector,
        uncertainty,
        DWELL_S,
        n_trials=CONTINUOUS_TRIALS,
        seed=SEED,
        false_alarm_probability=f,
        verification_time_s=VERIFICATION_TIME_S,
        restrict_to_grid=True,
    )
    free = monte_carlo_acquisition_continuous(
        grid,
        schedule,
        geometry,
        detector,
        uncertainty,
        DWELL_S,
        n_trials=CONTINUOUS_TRIALS,
        seed=SEED,
        false_alarm_probability=f,
        verification_time_s=VERIFICATION_TIME_S,
        restrict_to_grid=False,
    )
    analytic = expected_acquisition_time(grid, schedule, q).expected_time_s
    out.write(f"analytic cell model, E[T | in grid]        {analytic:.6f} s\n")
    out.write(
        f"continuous geometry, targets in grid       {continuous.mean_time_s:.6f} s "
        f"+- {continuous.standard_error_s:.6f} s  ({continuous.n_trials} trials)\n"
    )
    out.write(
        f"continuous geometry, whole prior           {free.mean_time_s:.6f} s "
        f"+- {free.standard_error_s:.6f} s  (acquired {free.probability_by_deadline:.6f})\n"
    )
    relative = (continuous.mean_time_s - analytic) / analytic
    out.write(f"cell model is pessimistic by               {-relative * 100.0:.2f} percent\n")
    out.write(
        "The cell model forbids a dwell from acquiring a target in a neighbouring\n"
        "cell. The continuous geometry allows it, so the abstraction is a bound in a\n"
        "known direction rather than an unquantified error.\n"
    )

    rule(out)
    out.write(f"checks failed: {failures}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
