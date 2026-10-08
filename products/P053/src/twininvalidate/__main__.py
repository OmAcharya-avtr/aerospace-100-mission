"""Command-line interface: ``python -m twininvalidate``.

Subcommands
-----------
``twin``
    Print the declared twin, its steady-state gain and innovation variance.
``scenarios``
    Print the three declared change scenarios.
``calibrate``
    Set each baseline's threshold from a declared false-alarm target and print
    the achieved in-control ARL0.
``curve``
    Print the detection-delay against false-alarm-rate curve for one change
    type.
``benchmark``
    Run the learned drift classifier against the three analytic baselines at a
    matched in-control ARL0, in distribution and out of it.
``ambiguity``
    Demonstrate that an asset fault and a twin error can produce the same
    residual.

This module is the only place in the package that writes to stdout. Every
number it prints comes from a computation run at the moment of printing; none
is cached or hard-coded.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .ambiguity import max_absolute_difference, paired_streams
from .arl import delay_after_onset, delay_curve, threshold_grid
from .asset import AssetChange, StreamSpec, simulate_residuals
from .classifier import DriftClassifier
from .datasets import (
    SCENARIO_LABELS,
    SCENARIOS,
    SEED_THRESHOLD_CALIBRATION,
    calibration_set,
    changed_streams,
    in_control_streams,
    training_set,
)
from .detectors import BASELINES, DetectorSpec
from .thresholds import bracket_threshold, calibrate_threshold, rate_from_arl0
from .twin import REFERENCE_DT, reference_twin

DEFAULT_TARGET_ARL0 = 1000.0
SAMPLE_RATE_HZ = 1.0 / REFERENCE_DT


def _baseline_specs() -> list[DetectorSpec]:
    return [DetectorSpec(name) for name in BASELINES]


def _cmd_twin(_: argparse.Namespace) -> int:
    twin = reference_twin()
    filt = twin.steady_state()
    print("declared reference twin (single-axis attitude channel)")
    print(f"  dt                      {twin.dt:.4f} s  ({SAMPLE_RATE_HZ:.1f} Hz)")
    print(f"  A                       {np.array2string(twin.A, precision=6)}")
    print(f"  B                       {np.array2string(twin.B.ravel(), precision=6)}")
    print(f"  C                       {np.array2string(twin.C.ravel(), precision=6)}")
    print(f"  Q                       {np.array2string(twin.Q, precision=9)}")
    print(f"  R                       {twin.R[0, 0]:.6e} rad^2")
    print(f"  predictor gain K        {np.array2string(filt.K.ravel(), precision=6)}")
    print(f"  innovation variance S   {filt.S:.6e} rad^2")
    print(f"  sqrt(S)                 {np.sqrt(filt.S):.6e} rad")
    print(f"  spectral radius A - KC  {filt.spectral_radius():.6f}  (must be < 1)")
    print(f"  Riccati residual        {filt.riccati_residual():.3e}")
    return 0


def _cmd_scenarios(_: argparse.Namespace) -> int:
    print("declared change scenarios (change present from sample 0)")
    for name, change in SCENARIOS.items():
        print(f"  {name:<16} {SCENARIO_LABELS[name]}")
        print(
            f"  {'':<16} kind={change.kind} magnitude={change.magnitude:g} "
            f"ramp_samples={change.ramp_samples}"
        )
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    ic = in_control_streams(
        n_runs=args.runs, n_samples=args.samples, seed=SEED_THRESHOLD_CALIBRATION
    )
    print(
        f"threshold calibration on {args.runs} x {args.samples} in-control samples, "
        f"target ARL0 = {args.target:g} samples"
    )
    print(
        f"  equivalent false-alarm rate "
        f"{rate_from_arl0(args.target, SAMPLE_RATE_HZ):.1f} per 1000 h at "
        f"{SAMPLE_RATE_HZ:.0f} Hz"
    )
    print(f"  {'detector':<30}{'threshold':>12}{'ARL0':>10}{'stderr':>9}{'alarms':>8}  method")
    for spec in [*_baseline_specs(), DetectorSpec("varcusum")]:
        cal = calibrate_threshold(spec, ic, args.target)
        print(
            f"  {spec.label():<30}{cal.threshold:>12.4f}{cal.achieved_arl0:>10.1f}"
            f"{cal.achieved_stderr:>9.1f}{cal.n_alarms:>8d}  {cal.method}"
        )
    return 0


def _cmd_curve(args: argparse.Namespace) -> int:
    if args.scenario not in SCENARIOS:
        raise SystemExit(f"unknown scenario {args.scenario!r}; choose from {sorted(SCENARIOS)}")
    ic = in_control_streams(
        n_runs=args.runs, n_samples=args.samples, seed=SEED_THRESHOLD_CALIBRATION
    )
    oc = changed_streams(args.scenario, n_runs=args.runs, n_samples=args.samples)
    print(f"detection-delay against false-alarm curve: {SCENARIO_LABELS[args.scenario]}")
    print(f"  zero-state convention, {args.runs} runs x {args.samples} samples per point")
    for spec in [*_baseline_specs(), DetectorSpec("varcusum")]:
        grid = threshold_grid(spec.statistic(ic), n_points=args.points)
        print(f"  {spec.label()}")
        print(
            f"    {'threshold':>10}{'ARL0':>10}{'FA/1000h':>11}{'ARL1':>9}"
            f"{'stderr':>8}{'det':>7}"
        )
        for point in delay_curve(spec, ic, oc, grid):
            arl0 = point.arl0.value
            fa = rate_from_arl0(arl0, SAMPLE_RATE_HZ) if np.isfinite(arl0) else float("nan")
            print(
                f"    {point.threshold:>10.4f}{arl0:>10.1f}{fa:>11.1f}"
                f"{point.arl1.value:>9.1f}{point.arl1.stderr:>8.1f}"
                f"{point.arl1.detection_fraction:>7.2f}"
            )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    ic = in_control_streams(
        n_runs=args.runs, n_samples=args.samples, seed=SEED_THRESHOLD_CALIBRATION
    )
    train = training_set()
    cal = calibration_set()
    clf = DriftClassifier().fit(train.x, train.y, cal.x, cal.y)
    specs = [*_baseline_specs(), DetectorSpec("varcusum")]
    thresholds = {s.name: calibrate_threshold(s, ic, args.target).threshold for s in specs}
    bracket = bracket_threshold(clf.statistic(ic), args.target)
    onset = clf.window
    print(
        f"learned classifier against analytic baselines, matched in-control "
        f"ARL0 = {args.target:g} samples"
    )
    print(
        f"  steady-state protocol: {onset} in-control samples before the change, "
        f"so the windowed classifier has a full feature window at onset"
    )
    print(
        f"  classifier achievable ARL0 {bracket.conservative_arl0:.1f} at "
        f"confidence {bracket.conservative:.6f} "
        f"({bracket.n_levels} distinct confidence levels)"
    )
    header = f"  {'scenario':<34}" + "".join(f"{s.name:>12}" for s in specs) + f"{'learned':>12}"
    print(header)
    rows: list[tuple[str, AssetChange, int]] = [
        (SCENARIO_LABELS[name], SCENARIOS[name], 53200 + i)
        for i, name in enumerate(SCENARIOS)
    ]
    rows.append(
        ("OUT OF DIST: gain +1 % (sign flipped)", AssetChange("parameter_step", 0, 0.01), 53300)
    )
    rows.append(
        ("OUT OF DIST: gain -0.4 % (smaller)", AssetChange("parameter_step", 0, -0.004), 53301)
    )
    rows.append(("OUT OF DIST: gain -3 % (larger)", AssetChange("parameter_step", 0, -0.03), 53302))
    for label, change, seed in rows:
        shifted = AssetChange(
            kind=change.kind,
            onset=onset,
            magnitude=change.magnitude,
            ramp_samples=change.ramp_samples,
        )
        oc = simulate_residuals(
            StreamSpec(change=shifted, n_runs=args.runs, n_samples=args.samples, seed=seed)
        )
        cells = []
        for spec in specs:
            est, _ = delay_after_onset(spec.statistic(oc), thresholds[spec.name], onset)
            cells.append(f"{est.value:.0f}/{est.detection_fraction:.2f}")
        est, _ = delay_after_onset(clf.statistic(oc), bracket.conservative, onset)
        cells.append(f"{est.value:.0f}/{est.detection_fraction:.2f}")
        print(f"  {label:<34}" + "".join(f"{c:>12}" for c in cells))
    print("  cells are mean delay in samples / fraction of runs detected")
    print(f"  wall clock {time.perf_counter() - started:.1f} s on this container")
    return 0


def _cmd_ambiguity(args: argparse.Namespace) -> int:
    pair = paired_streams(offset=args.offset, n_runs=args.runs, n_samples=args.samples)
    diff = max_absolute_difference(pair)
    pre = pair.asset_fault[:, : pair.onset]
    post = pair.asset_fault[:, pair.onset :]
    print("twin invalidation against asset fault, on one shared noise realisation")
    print(f"  measurement offset        {pair.offset:g} rad from sample {pair.onset}")
    print("  world A, asset sensor bias, twin declares 0")
    print(f"  world B, asset clean, twin declares -{pair.offset:g}")
    print(f"  residual mean before      {pre.mean():+.4f}")
    print(f"  residual mean after       {post.mean():+.4f}")
    print(f"  max |A - B| over streams  {diff:.3e}  (algebraically 0)")
    print("  the two worlds are indistinguishable from the residual alone;")
    print("  this package reports disagreement, never its cause")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``python -m twininvalidate``."""
    parser = argparse.ArgumentParser(
        prog="python -m twininvalidate",
        description=(
            "Model-invalidation monitoring for a linear-Gaussian digital twin. "
            "Research-grade: not flight-qualified, not certified, not approved "
            "for operational aerospace use. An alarm means the twin and the "
            "asset disagree, not that either one is at fault."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_twin = sub.add_parser("twin", help="print the declared twin and its steady-state gain")
    p_twin.set_defaults(func=_cmd_twin)

    p_scen = sub.add_parser("scenarios", help="print the declared change scenarios")
    p_scen.set_defaults(func=_cmd_scenarios)

    p_cal = sub.add_parser("calibrate", help="set thresholds from a declared ARL0 target")
    p_cal.add_argument(
        "--target", type=float, default=DEFAULT_TARGET_ARL0, help="target ARL0, samples"
    )
    p_cal.add_argument("--runs", type=int, default=200, help="in-control runs")
    p_cal.add_argument("--samples", type=int, default=2000, help="samples per run")
    p_cal.set_defaults(func=_cmd_calibrate)

    p_curve = sub.add_parser("curve", help="detection-delay against false-alarm-rate curve")
    p_curve.add_argument("scenario", choices=sorted(SCENARIOS), help="change scenario")
    p_curve.add_argument("--runs", type=int, default=200, help="runs per point")
    p_curve.add_argument("--samples", type=int, default=2000, help="samples per run")
    p_curve.add_argument("--points", type=int, default=10, help="thresholds on the curve")
    p_curve.set_defaults(func=_cmd_curve)

    p_bench = sub.add_parser("benchmark", help="learned classifier against the baselines")
    p_bench.add_argument(
        "--target", type=float, default=DEFAULT_TARGET_ARL0, help="target ARL0, samples"
    )
    p_bench.add_argument("--runs", type=int, default=150, help="runs per scenario")
    p_bench.add_argument("--samples", type=int, default=2000, help="samples per run")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_amb = sub.add_parser("ambiguity", help="asset fault against twin error, same residual")
    p_amb.add_argument("--offset", type=float, default=0.02, help="measurement offset, rad")
    p_amb.add_argument("--runs", type=int, default=40, help="runs")
    p_amb.add_argument("--samples", type=int, default=1200, help="samples per run")
    p_amb.set_defaults(func=_cmd_ambiguity)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
