"""A compact numeric fingerprint, for diffing between the ends of every
declared dependency range.

    python validation/dependency_range_check.py > validation/dependency_range_check_output.txt

A suite that passes at both ends of a range while the numbers underneath move
is the dangerous case, and no test catches it. This script prints the numbers
that would move. It is run at both ends of every declared range and the two
outputs are diffed; the diff is recorded in VALIDATION.md.

It is deliberately cheap (about twenty seconds) so it can be run in a fresh
virtual environment without the full benchmark.
"""

from __future__ import annotations

import sys

import numpy as np
from _common import DWELL_S, VERIFICATION_TIME_S, reference

from acqsearch import (
    BayesianGreedyPolicy,
    FixedPatternPolicy,
    LearnedScanPolicy,
    build_environment,
    build_schedule,
    build_training_set,
    cell_detection_probabilities,
    cell_visit_order,
    coverage_curve,
    evaluate_policies,
    expected_acquisition_time,
    generate_scenarios,
    mean_censored_time,
    monte_carlo_acquisition,
    optimal_dwell_time,
    probability_acquired_by,
    run_episode,
    sample_target_cells,
    spiral_closed_form_coverage,
    split_scenarios,
    standard_paths,
    summarise,
)


def main() -> int:
    """Print the fingerprint."""
    out = sys.stdout
    uncertainty, geometry, detector, grid = reference()
    q = cell_detection_probabilities(grid, geometry, detector, DWELL_S, uncertainty)
    f = detector.false_alarm_probability(DWELL_S)
    orders = {name: cell_visit_order(p, grid) for name, p in standard_paths(grid).items()}
    schedule = build_schedule(grid, orders["spiral"], geometry, DWELL_S, f, VERIFICATION_TIME_S)

    out.write("acqsearch numeric fingerprint\n")
    out.write(f"n_cells                        {grid.n_cells}\n")
    out.write(f"prior_mass_in_grid             {grid.prior_mass.sum():.12f}\n")
    out.write(f"cell_side_rad                  {grid.side_rad:.12e}\n")
    out.write(f"false_alarm_probability        {f:.12f}\n")
    mean_q = float(np.average(q, weights=grid.prior_mass))
    out.write(f"mean_cell_detection            {mean_q:.12f}\n")
    out.write(f"min_cell_detection             {q.min():.12f}\n")
    for name in ("spiral", "raster", "lissajous", "random_restart"):
        local = build_schedule(grid, orders[name], geometry, DWELL_S, f, VERIFICATION_TIME_S)
        estimate = expected_acquisition_time(grid, local, q)
        out.write(f"expected_time_{name:<18}{estimate.expected_time_s:.12f}\n")
    out.write(f"period_spiral_s                {schedule.period_s:.12f}\n")
    out.write(
        f"p_acquired_by_2s               "
        f"{probability_acquired_by(grid, schedule, q, 2.0):.12f}\n"
    )
    mc = monte_carlo_acquisition(grid, schedule, q, n_trials=50000, seed=20261016)
    out.write(f"monte_carlo_mean_s             {mc.mean_time_s:.12f}\n")
    out.write(f"monte_carlo_se_s               {mc.standard_error_s:.12f}\n")
    curve = coverage_curve(
        standard_paths(grid)["spiral"], uncertainty, geometry.footprint_radius_rad, 40000, seed=7
    )
    out.write(f"spiral_coverage_mc             {curve.final_coverage():.12f}\n")
    closed = spiral_closed_form_coverage(
        grid.side_rad, 6, geometry.footprint_radius_rad, uncertainty
    )
    out.write(f"spiral_coverage_closed_form    {closed:.12f}\n")
    optimum = optimal_dwell_time(
        grid,
        orders["spiral"],
        geometry,
        detector,
        uncertainty,
        np.geomspace(2.0e-4, 2.0e-2, 41),
        VERIFICATION_TIME_S,
    )
    out.write(f"best_dwell_s                   {optimum.best_dwell_s:.12e}\n")
    out.write(f"best_expected_time_s           {optimum.best_expected_time_s:.12f}\n")

    scenarios = generate_scenarios(60, 20261016)
    train, test = split_scenarios(scenarios, 0.65)
    features, target = build_training_set(train, episodes_per_scenario=1)
    out.write(f"training_rows                  {features.shape[0]}\n")
    out.write(f"training_target_mean           {float(target.mean()):.12f}\n")
    learned = LearnedScanPolicy(n_estimators=60, max_depth=4).fit(features, target)
    results = evaluate_policies(test, learned, targets_per_scenario=3, include_all_patterns=False)
    scores = summarise(results)
    for name in sorted(scores):
        out.write(f"policy_mean_{name:<18}{scores[name].mean_time_s:.12f}\n")

    environment, pattern_orders = build_environment(test[0])
    targets = sample_target_cells(environment.grid, 5, np.random.default_rng(1))
    for label, policy in (
        ("fixed", FixedPatternPolicy(pattern_orders["spiral"])),
        ("greedy", BayesianGreedyPolicy()),
        ("learned", learned),
    ):
        episodes = [
            run_episode(environment, policy, int(c), np.random.default_rng(100 + i))
            for i, c in enumerate(targets)
        ]
        out.write(f"single_scenario_{label:<14}{mean_censored_time(episodes):.12f}\n")
    decision = learned.decide(
        environment,
        environment.grid.prior_mass / environment.grid.prior_mass.sum(),
        np.zeros(2),
        np.zeros(environment.grid.n_cells, dtype=np.int64),
        0,
        0.0,
    )
    out.write(f"decision_cell                  {decision.cell}\n")
    out.write(f"decision_point_s               {decision.predicted_remaining_s:.12f}\n")
    out.write(f"decision_interval_low_s        {decision.interval_s[0]:.12f}\n")
    out.write(f"decision_interval_high_s       {decision.interval_s[1]:.12f}\n")
    out.write(f"decision_confidence            {decision.confidence:.12f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
