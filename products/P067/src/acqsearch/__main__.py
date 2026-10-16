"""Command line interface: ``python -m acqsearch <command>``.

Commands
--------
``patterns``   expected acquisition time and pass period for all four patterns
``expected``   one pattern, with the analytic / Monte Carlo cross-check
``dwell``      sweep the dwell time and print the trade
``coverage``   spiral coverage, closed form against Monte Carlo
``benchmark``  held-out head-to-head of fixed pattern, Bayesian-greedy, learned
``scenarios``  regenerate the committed scenario file

Angles are entered in microradians and times in milliseconds where that is the
natural unit for an optical terminal; every printed quantity states its unit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .acquisition import (
    build_schedule,
    cell_detection_probabilities,
    expected_acquisition_time,
    monte_carlo_acquisition,
    monte_carlo_acquisition_continuous,
    optimal_dwell_time,
    probability_acquired_by,
)
from .benchmark import build_training_set, evaluate_policies, paired_difference, summarise, winner
from .coverage import coverage_curve, spiral_closed_form_coverage, spiral_closed_form_valid
from .detection import PoissonDwellDetector
from .geometry import PointingUncertainty, TerminalGeometry, build_cell_grid
from .patterns import cell_visit_order, spiral_path, standard_paths
from .policy import LearnedScanPolicy
from .scenarios import generate_scenarios, save_scenarios, split_scenarios

URAD = 1e-6
MS = 1e-3


def _add_physics_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sigma-urad", type=float, default=150.0, help="per-axis 1-sigma, urad")
    parser.add_argument(
        "--divergence-urad", type=float, default=100.0, help="full beam divergence, urad"
    )
    parser.add_argument("--fov-urad", type=float, default=200.0, help="full detector FOV, urad")
    parser.add_argument("--slew-mrad-s", type=float, default=10.0, help="slew rate, mrad/s")
    parser.add_argument("--settle-ms", type=float, default=2.0, help="settle time, ms")
    parser.add_argument("--signal-hz", type=float, default=2.0e4, help="on-axis signal rate, 1/s")
    parser.add_argument("--background-hz", type=float, default=2.0e3, help="background rate, 1/s")
    parser.add_argument("--threshold", type=int, default=8, help="count threshold, counts")
    parser.add_argument("--dwell-ms", type=float, default=1.4, help="dwell time, ms")
    parser.add_argument("--verify-s", type=float, default=0.5, help="false-alarm verification, s")
    parser.add_argument("--rytov", type=float, default=0.2, help="Rytov variance, dimensionless")
    parser.add_argument("--extent-sigma", type=float, default=2.5, help="grid radius, in sigma")


def _assemble(args: argparse.Namespace):
    uncertainty = PointingUncertainty(args.sigma_urad * URAD)
    geometry = TerminalGeometry(
        beam_divergence_rad=args.divergence_urad * URAD,
        detector_fov_rad=args.fov_urad * URAD,
        slew_rate_rad_s=args.slew_mrad_s * 1e-3,
        settle_time_s=args.settle_ms * MS,
    )
    detector = PoissonDwellDetector(
        signal_rate_hz=args.signal_hz,
        background_rate_hz=args.background_hz,
        threshold_counts=args.threshold,
        rytov_variance=args.rytov,
    )
    grid = build_cell_grid(uncertainty, geometry, extent_sigma=args.extent_sigma)
    return uncertainty, geometry, detector, grid


def _command_patterns(args: argparse.Namespace, out) -> int:
    uncertainty, geometry, detector, grid = _assemble(args)
    dwell = args.dwell_ms * MS
    q = cell_detection_probabilities(grid, geometry, detector, dwell, uncertainty)
    f = detector.false_alarm_probability(dwell)
    out.write(f"cells {grid.n_cells}  prior mass in grid {grid.prior_mass.sum():.6f}\n")
    mean_q = float(np.average(q, weights=grid.prior_mass))
    out.write(f"mean cell detection probability {mean_q:.6f}")
    out.write(f"  per-dwell false alarm probability {f:.6e}\n")
    header = f"{'pattern':<16}{'period_s':>12}{'E[T]_s':>12}{'slew_rad':>12}{'P(by 3T)':>12}"
    out.write(header + "\n")
    for name, path in standard_paths(grid).items():
        order = cell_visit_order(path, grid)
        schedule = build_schedule(grid, order, geometry, dwell, f, args.verify_s)
        estimate = expected_acquisition_time(grid, schedule, q)
        probability = probability_acquired_by(grid, schedule, q, 3.0 * schedule.period_s)
        out.write(
            f"{name:<16}{schedule.period_s:>12.6f}{estimate.expected_time_s:>12.6f}"
            f"{schedule.total_slew_rad(grid):>12.6f}{probability:>12.6f}\n"
        )
    return 0


def _command_expected(args: argparse.Namespace, out) -> int:
    uncertainty, geometry, detector, grid = _assemble(args)
    dwell = args.dwell_ms * MS
    q = cell_detection_probabilities(grid, geometry, detector, dwell, uncertainty)
    f = detector.false_alarm_probability(dwell)
    order = cell_visit_order(standard_paths(grid)[args.pattern], grid)
    schedule = build_schedule(grid, order, geometry, dwell, f, args.verify_s)
    estimate = expected_acquisition_time(grid, schedule, q)
    mc = monte_carlo_acquisition(grid, schedule, q, n_trials=args.trials, seed=args.seed)
    continuous = monte_carlo_acquisition_continuous(
        grid,
        schedule,
        geometry,
        detector,
        uncertainty,
        dwell,
        n_trials=max(2000, args.trials // 5),
        seed=args.seed,
        false_alarm_probability=f,
        verification_time_s=args.verify_s,
        restrict_to_grid=True,
    )
    out.write(f"pattern {args.pattern}  cells {grid.n_cells}  period {schedule.period_s:.6f} s\n")
    out.write(f"analytic E[T | in grid]            {estimate.expected_time_s:.6f} s\n")
    out.write(
        f"Monte Carlo, same assumptions      {mc.mean_time_s:.6f} s "
        f"+- {mc.standard_error_s:.6f} s ({mc.n_inside} trials in grid)\n"
    )
    out.write(
        f"Monte Carlo, continuous geometry   {continuous.mean_time_s:.6f} s "
        f"+- {continuous.standard_error_s:.6f} s ({continuous.n_inside} acquired)\n"
    )
    out.write(f"prior mass outside the grid        {estimate.mass_outside:.6f}\n")
    return 0


def _command_dwell(args: argparse.Namespace, out) -> int:
    uncertainty, geometry, detector, grid = _assemble(args)
    order = cell_visit_order(standard_paths(grid)[args.pattern], grid)
    dwells = np.geomspace(args.dwell_min_ms * MS, args.dwell_max_ms * MS, args.points)
    optimum = optimal_dwell_time(
        grid, order, geometry, detector, uncertainty, dwells, args.verify_s
    )
    out.write(f"{'dwell_ms':>10}{'E[T]_s':>14}{'mean_q':>12}{'P_fa':>16}\n")
    for tau, expected, q, f in zip(
        optimum.dwell_grid_s,
        optimum.expected_time_s,
        optimum.detection_probability,
        optimum.false_alarm_probability,
        strict=True,
    ):
        out.write(f"{tau / MS:>10.4f}{expected:>14.6f}{q:>12.6f}{f:>16.6e}\n")
    out.write(
        f"grid minimum {optimum.best_expected_time_s:.6f} s at "
        f"{optimum.best_dwell_s / MS:.4f} ms\n"
    )
    return 0


def _command_coverage(args: argparse.Namespace, out) -> int:
    uncertainty, geometry, _detector, grid = _assemble(args)
    a = geometry.footprint_radius_rad
    pitch = args.pitch_urad * URAD if args.pitch_urad > 0.0 else grid.side_rad
    spacing = args.spacing_urad * URAD if args.spacing_urad > 0.0 else 0.5 * grid.side_rad
    valid = spiral_closed_form_valid(pitch, spacing, a)
    out.write(f"footprint radius {a / URAD:.4f} urad  pitch {pitch / URAD:.4f} urad\n")
    out.write(f"along-track spacing {spacing / URAD:.4f} urad  closed form applies: {valid}\n")
    header = f"{'turns':>7}{'closed_form':>14}{'monte_carlo':>14}{'difference':>13}{'se':>11}"
    out.write(header + "\n")
    for turns in range(1, args.turns + 1):
        closed = spiral_closed_form_coverage(pitch, turns, a, uncertainty)
        path = spiral_path(pitch, float(turns), spacing)
        curve = coverage_curve(path, uncertainty, a, n_samples=args.samples, seed=args.seed)
        measured = curve.final_coverage()
        out.write(
            f"{turns:>7}{closed:>14.6f}{measured:>14.6f}"
            f"{measured - closed:>+13.6f}{curve.standard_error[-1]:>11.6f}\n"
        )
    return 0


def _command_benchmark(args: argparse.Namespace, out) -> int:
    scenarios = generate_scenarios(args.scenarios, args.seed)
    train, test = split_scenarios(scenarios, args.train_fraction)
    features, target = build_training_set(train, episodes_per_scenario=args.train_episodes)
    learned = LearnedScanPolicy().fit(features, target)
    results = evaluate_policies(
        test, learned, targets_per_scenario=args.targets, include_all_patterns=True
    )
    scores = summarise(results)
    out.write(
        f"train scenarios {len(train)}  test scenarios {len(test)}  "
        f"rows {features.shape[0]}\n"
    )
    out.write(f"{'policy':<22}{'mean_s':>10}{'se_s':>9}{'median_s':>10}{'acq':>8}{'dwells':>9}\n")
    for score in sorted(scores.values(), key=lambda s: s.mean_time_s):
        out.write(
            f"{score.name:<22}{score.mean_time_s:>10.4f}{score.standard_error_s:>9.4f}"
            f"{score.median_time_s:>10.4f}{score.acquisition_rate:>8.4f}{score.mean_dwells:>9.1f}\n"
        )
    out.write(f"winner {winner(scores)}\n")
    for other in ("bayesian_greedy", "fixed_spiral"):
        if "learned" in results and other in results:
            difference = paired_difference("learned", results["learned"], other, results[other])
            out.write(
                f"paired learned - {other}: {difference.mean_difference_s:+.4f} s "
                f"+- {difference.standard_error_s:.4f} s  z = {difference.z_score:+.2f}\n"
            )
    return 0


def _command_scenarios(args: argparse.Namespace, out) -> int:
    scenarios = generate_scenarios(args.scenarios, args.seed)
    save_scenarios(scenarios, args.output)
    train, test = split_scenarios(scenarios, args.train_fraction)
    out.write(f"wrote {len(scenarios)} scenarios to {Path(args.output)}\n")
    out.write(f"train {len(train)}  held out {len(test)}  seed {args.seed}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``python -m acqsearch``."""
    parser = argparse.ArgumentParser(prog="acqsearch", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    patterns = sub.add_parser("patterns", help="compare all four scan patterns")
    _add_physics_arguments(patterns)
    patterns.set_defaults(handler=_command_patterns)

    expected = sub.add_parser("expected", help="expected acquisition time for one pattern")
    _add_physics_arguments(expected)
    expected.add_argument("--pattern", default="spiral")
    expected.add_argument("--trials", type=int, default=40000)
    expected.add_argument("--seed", type=int, default=20261016)
    expected.set_defaults(handler=_command_expected)

    dwell = sub.add_parser("dwell", help="sweep the dwell time")
    _add_physics_arguments(dwell)
    dwell.add_argument("--pattern", default="spiral")
    dwell.add_argument("--dwell-min-ms", type=float, default=0.2)
    dwell.add_argument("--dwell-max-ms", type=float, default=8.0)
    dwell.add_argument("--points", type=int, default=25)
    dwell.set_defaults(handler=_command_dwell)

    coverage = sub.add_parser("coverage", help="spiral coverage, closed form against Monte Carlo")
    _add_physics_arguments(coverage)
    coverage.add_argument("--pitch-urad", type=float, default=0.0, help="0 means the cell side")
    coverage.add_argument("--spacing-urad", type=float, default=0.0, help="0 means half the side")
    coverage.add_argument("--turns", type=int, default=6)
    coverage.add_argument("--samples", type=int, default=100000)
    coverage.add_argument("--seed", type=int, default=20261016)
    coverage.set_defaults(handler=_command_coverage)

    benchmark = sub.add_parser("benchmark", help="held-out policy head-to-head")
    benchmark.add_argument("--scenarios", type=int, default=120)
    benchmark.add_argument("--train-fraction", type=float, default=0.65)
    benchmark.add_argument("--train-episodes", type=int, default=2)
    benchmark.add_argument("--targets", type=int, default=4)
    benchmark.add_argument("--seed", type=int, default=20261016)
    benchmark.set_defaults(handler=_command_benchmark)

    scenarios = sub.add_parser("scenarios", help="regenerate the scenario file")
    scenarios.add_argument("--scenarios", type=int, default=300)
    scenarios.add_argument("--train-fraction", type=float, default=0.65)
    scenarios.add_argument("--seed", type=int, default=20261016)
    scenarios.add_argument("--output", default="data/scenarios.json")
    scenarios.set_defaults(handler=_command_scenarios)
    return parser


def main(argv: list[str] | None = None, out=None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    stream = sys.stdout if out is None else out
    return int(args.handler(args, stream))


if __name__ == "__main__":
    raise SystemExit(main())
