"""Command-line interface: ``python -m telemetryool``.

Subcommands
-----------
``design``
    Solve for the CUSUM, EWMA and limit-check thresholds that deliver a target
    window false-alarm probability, and print the resulting ARL0.
``arl``
    Average run length for a named chart at a given shift, by both the
    Markov-chain and (for CUSUM) the Siegmund closed form.
``check``
    Run the operational out-of-limit checker over a CSV of one channel and print
    the alarm transitions.
``far``
    Monte-Carlo measurement of a chart's window false-alarm rate, with its
    binomial standard error, beside the design value.
``compare``
    The matched-false-alarm-rate comparison, at a size the caller chooses.
``changepoint``
    Calibrate a change-point threshold and report the detected change points in
    a CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .arl import (
    cusum_arl_markov,
    cusum_arl_siegmund_two_sided,
    cusum_window_false_alarm,
    design_cusum_h,
    design_ewma_L,
    design_ool_limit,
    ewma_arl_markov,
    ewma_window_false_alarm,
)
from .calibration import estimate_rate, windows_for_precision
from .changepoint import calibrate_change_point_threshold, detect_change_points
from .charts import CusumChart, EwmaChart
from .detectors import build_detector_suite
from .harness import ScenarioSpec, run_comparison
from .limits import ChannelSpec, LimitSet, OolChecker
from .runs import any_run
from .synthetic import Anomaly, NominalModel, equicorrelation


def _add_design(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("design", help="thresholds that deliver a target window FAR")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="target window false-alarm probability (default 0.05)")
    p.add_argument("--window", type=int, default=100,
                   help="monitoring window length W in samples (default 100)")
    p.add_argument("--cusum-k", type=float, default=0.5,
                   help="CUSUM reference value in sigma units (default 0.5)")
    p.add_argument("--ewma-lam", type=float, default=0.2,
                   help="EWMA smoothing weight (default 0.2)")
    p.add_argument("--persistence", type=int, default=3,
                   help="limit-check debounce count (default 3)")


def _add_arl(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("arl", help="average run length for a chart")
    p.add_argument("chart", choices=["cusum", "ewma"])
    p.add_argument("--delta", type=float, default=0.0, help="mean shift in sigma (default 0)")
    p.add_argument("--k", type=float, default=0.5, help="CUSUM reference value (default 0.5)")
    p.add_argument("--h", type=float, default=5.0, help="CUSUM decision interval (default 5)")
    p.add_argument("--lam", type=float, default=0.2, help="EWMA weight (default 0.2)")
    p.add_argument("--limit", type=float, default=3.0, help="EWMA limit multiplier (default 3)")
    p.add_argument("--states", type=int, default=401, help="Markov states (default 401)")


def _add_check(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("check", help="run the OOL checker over a CSV column")
    p.add_argument("csvfile", help="CSV file with a header row")
    p.add_argument("--column", required=True, help="column name holding the value")
    p.add_argument("--valid-column", default=None, help="column holding a 0/1 validity flag")
    p.add_argument("--mode-column", default=None, help="column holding the mode name")
    p.add_argument("--soft-low", type=float, default=None)
    p.add_argument("--soft-high", type=float, default=None)
    p.add_argument("--hard-low", type=float, default=None)
    p.add_argument("--hard-high", type=float, default=None)
    p.add_argument("--persistence-soft", type=int, default=1)
    p.add_argument("--persistence-hard", type=int, default=1)
    p.add_argument("--clear-persistence", type=int, default=1)
    p.add_argument("--units", default="")
    p.add_argument("--all-samples", action="store_true",
                   help="print every sample, not just the transitions")


def _add_far(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("far", help="Monte-Carlo window false-alarm rate for a chart")
    p.add_argument("chart", choices=["cusum", "ewma"])
    p.add_argument("--alpha", type=float, default=0.05, help="design target (default 0.05)")
    p.add_argument("--window", type=int, default=100, help="window length (default 100)")
    p.add_argument("--windows", type=int, default=20000,
                   help="Monte-Carlo windows (default 20000)")
    p.add_argument("--k", type=float, default=0.5)
    p.add_argument("--lam", type=float, default=0.2)
    p.add_argument("--rho", type=float, default=0.0,
                   help="AR(1) coefficient of the nominal data (default 0, the design "
                        "hypothesis)")
    p.add_argument("--seed", type=int, default=20261005)


def _add_compare(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("compare", help="matched-false-alarm-rate comparison")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--cross-correlation", type=float, default=0.6)
    p.add_argument("--train-windows", type=int, default=1000)
    p.add_argument("--cal-windows", type=int, default=4000)
    p.add_argument("--measure-windows", type=int, default=4000)
    p.add_argument("--scenario-windows", type=int, default=1000)
    p.add_argument("--persistence", type=int, default=1)
    p.add_argument("--step", type=float, default=1.0, help="step magnitude in sigma")
    p.add_argument("--onset", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--confusion", action="store_true", help="also print every confusion matrix")


def _add_changepoint(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("changepoint", help="change-point detection over a CSV column")
    p.add_argument("csvfile")
    p.add_argument("--column", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--min-segment", type=int, default=20)
    p.add_argument("--simulations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``python -m telemetryool``."""
    parser = argparse.ArgumentParser(
        prog="python -m telemetryool",
        description=(
            "Operational out-of-limit checking, designed-false-alarm-rate drift "
            "detection, and a matched-false-alarm-rate detector comparison. "
            "Research-grade; not flight-qualified, not certified, not approved for "
            "operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"telemetryool {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_design(sub)
    _add_arl(sub)
    _add_check(sub)
    _add_far(sub)
    _add_compare(sub)
    _add_changepoint(sub)
    return parser


def _read_csv_column(path: str, column: str) -> list[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    if column not in rows[0]:
        raise SystemExit(f"{path}: no column {column!r}; columns are {sorted(rows[0])}")
    return [r[column] for r in rows]


def _cmd_design(args: argparse.Namespace) -> int:
    print(f"Target window false-alarm probability alpha_W = {args.alpha:.6f}")
    print(f"Window length W = {args.window} samples")
    print(f"Precision note: measuring this rate to 10 % relative standard error needs "
          f"{windows_for_precision(args.alpha, 0.10)} independent windows.")
    print()
    cu = design_cusum_h(args.alpha, args.cusum_k, args.window)
    ew = design_ewma_L(args.alpha, args.ewma_lam, args.window)
    ool = design_ool_limit(args.alpha, args.persistence, args.window)
    rows = [
        ("cusum", f"k={args.cusum_k}", "h", cu),
        ("ewma", f"lam={args.ewma_lam}", "L", ew),
        ("ool", f"persistence={args.persistence}", "L", ool),
    ]
    print(f"{'method':8s} {'config':20s} {'param':6s} {'value':>12s} {'alpha_W':>12s} "
          f"{'ARL0':>14s}")
    for name, cfg, param, d in rows:
        print(f"{name:8s} {cfg:20s} {param:6s} {d.threshold:12.6f} "
              f"{d.achieved_alpha_w:12.8f} {d.arl0:14.2f}")
    print()
    print("Every value above is analytic under iid normal data with known sigma. "
          "Measure it before trusting it: see the `far` subcommand.")
    return 0


def _cmd_arl(args: argparse.Namespace) -> int:
    if args.chart == "cusum":
        markov = cusum_arl_markov(args.delta, args.k, args.h, max(args.states, 400))
        sieg = cusum_arl_siegmund_two_sided(args.delta, args.k, args.h)
        print(f"two-sided CUSUM  k={args.k}  h={args.h}  delta={args.delta} sigma")
        print(f"  ARL (Brook & Evans 1972 Markov chain, {max(args.states, 400)} states) "
              f"= {markov:.4f} samples")
        print(f"  ARL (Siegmund 1985 closed form, b = h + 1.166)                 "
              f"= {sieg:.4f} samples")
        rel = abs(sieg - markov) / markov
        print(f"  relative difference = {rel:.3e}")
    else:
        markov = ewma_arl_markov(args.delta, args.lam, args.limit, args.states)
        print(f"two-sided EWMA  lam={args.lam}  L={args.limit}  delta={args.delta} sigma")
        print(f"  ARL (Lucas & Saccucci 1990 Markov chain, {args.states} states) "
              f"= {markov:.4f} samples")
        print("  no closed form is used; EWMA ARL has no standard closed form.")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    values = [float(v) for v in _read_csv_column(args.csvfile, args.column)]
    valid = (
        [bool(int(float(v))) for v in _read_csv_column(args.csvfile, args.valid_column)]
        if args.valid_column
        else [True] * len(values)
    )
    modes = (
        _read_csv_column(args.csvfile, args.mode_column) if args.mode_column else None
    )
    if len(valid) != len(values):
        raise SystemExit("validity column has a different length from the value column")
    limit_set = LimitSet(args.soft_low, args.soft_high, args.hard_low, args.hard_high)
    spec = ChannelSpec(
        name=args.column,
        units=args.units,
        limits={"*": limit_set},
        persistence_soft=args.persistence_soft,
        persistence_hard=args.persistence_hard,
        clear_persistence=args.clear_persistence,
    )
    checker = OolChecker(spec)
    samples = checker.update_series(values, valid, modes)
    print(f"channel {args.column!r} units={args.units!r} n={len(samples)}")
    print(f"limits soft=({args.soft_low}, {args.soft_high}) "
          f"hard=({args.hard_low}, {args.hard_high})")
    print(f"persistence soft={args.persistence_soft} hard={args.persistence_hard} "
          f"clear={args.clear_persistence}")
    print()
    print(f"{'idx':>6s} {'value':>12s} {'valid':>6s} {'mode':>10s} {'level':>9s} "
          f"{'state':>12s} {'soft':>5s} {'hard':>5s} {'below':>6s} {'event':>8s}")
    shown = 0
    for s in samples:
        if not (args.all_samples or s.raised or s.cleared):
            continue
        event = "RAISE" if s.raised else ("CLEAR" if s.cleared else "")
        level = "-" if s.level is None else s.level.name
        print(f"{s.index:6d} {s.value:12.6g} {int(s.valid):6d} {s.mode:>10s} {level:>9s} "
              f"{s.state.name:>12s} {s.soft_count:5d} {s.hard_count:5d} {s.below_count:6d} "
              f"{event:>8s}")
        shown += 1
    if shown == 0:
        print("(no alarm transitions; pass --all-samples to see every sample)")
    print()
    print(f"final state: {checker.state.name}")
    return 0


def _cmd_far(args: argparse.Namespace) -> int:
    if args.chart == "cusum":
        design = design_cusum_h(args.alpha, args.k, args.window)
        chart = CusumChart(k=args.k, h=design.threshold)
        label = f"cusum k={args.k} h={design.threshold:.6f}"
        analytic = cusum_window_false_alarm(args.k, design.threshold, args.window)
    else:
        design = design_ewma_L(args.alpha, args.lam, args.window)
        chart = EwmaChart(lam=args.lam, limit_mult=design.threshold)
        label = f"ewma lam={args.lam} L={design.threshold:.6f}"
        analytic = ewma_window_false_alarm(args.lam, design.threshold, args.window)
    model = NominalModel(1, rho_time=args.rho)
    block = _single_channel(model, args.windows, args.window, args.seed)
    alarms = any_run(chart.breach_mask(block), chart.persistence)
    est = estimate_rate(int(alarms.sum()), int(alarms.size))
    print(label)
    print(f"  window length W   = {args.window} samples")
    print(f"  Monte-Carlo windows = {args.windows}")
    print(f"  nominal AR(1) rho = {args.rho}")
    print(f"  design  alpha_W   = {analytic:.6f}   (ARL0 = {design.arl0:.2f} samples)")
    print(f"  measured alpha_W  = {est.rate:.6f} +/- {est.standard_error:.6f} (binomial SE)")
    print(f"  95 % Wilson       = [{est.wilson_low:.6f}, {est.wilson_high:.6f}]")
    print(f"  95 % Clopper-Pearson = [{est.exact_low:.6f}, {est.exact_high:.6f}]")
    print(f"  discrepancy       = {est.z_against(analytic):+.2f} design standard errors")
    if args.rho != 0.0:
        print("  NOTE: rho != 0 violates the design hypothesis; the design value above is "
              "not expected to hold.")
    return 0


def _single_channel(model: NominalModel, n_windows: int, window: int, seed: int):
    from .synthetic import generate_nominal

    return generate_nominal(model, n_windows, window, np.random.default_rng(seed))[:, :, 0]


def _cmd_compare(args: argparse.Namespace) -> int:
    corr = equicorrelation(args.channels, args.cross_correlation) if args.channels > 1 else None
    model = NominalModel(args.channels, correlation=corr)
    scenarios = [
        ScenarioSpec(f"step{args.step}", Anomaly("step", args.onset, args.step),
                     args.scenario_windows),
        ScenarioSpec("stuck", Anomaly("stuck", args.onset), args.scenario_windows),
    ]
    if args.channels > 1:
        scenarios.append(
            ScenarioSpec(
                "decorrelate",
                Anomaly("decorrelate", args.onset, channels=list(range(args.channels // 2 or 1))),
                args.scenario_windows,
            )
        )
    result = run_comparison(
        build_detector_suite(persistence=args.persistence),
        model,
        scenarios,
        target_alpha_w=args.alpha,
        window_length=args.window,
        n_train_windows=args.train_windows,
        n_cal_windows=args.cal_windows,
        n_measure_windows=args.measure_windows,
        seed=args.seed,
    )
    print(result.far_table())
    print()
    print(result.delay_table())
    if args.confusion:
        print(result.confusion_report())
    return 0


def _cmd_changepoint(args: argparse.Namespace) -> int:
    values = np.array([float(v) for v in _read_csv_column(args.csvfile, args.column)])
    th = calibrate_change_point_threshold(
        values.size, args.alpha, args.simulations, np.random.default_rng(args.seed)
    )
    points = detect_change_points(values, th.threshold, args.sigma, args.min_segment)
    print(f"column {args.column!r} n={values.size} sigma={args.sigma}")
    print(f"per-segment alpha target={th.alpha} threshold={th.threshold:.6f} "
          f"(Monte Carlo, {th.n_simulations} replicates, achieved "
          f"{th.achieved_alpha:.6f} +/- {th.standard_error:.6f})")
    print(f"detected change points: {points if points else '(none)'}")
    print("NOTE: the threshold is calibrated per segment; binary segmentation applies it at "
          "every recursion level, so the series-wide false-alarm rate is at least as high "
          "(measured 1.03x at n=200, alpha=0.05; see validation/validate_changepoint.py).")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns a process exit status."""
    args = build_parser().parse_args(argv)
    handlers = {
        "design": _cmd_design,
        "arl": _cmd_arl,
        "check": _cmd_check,
        "far": _cmd_far,
        "compare": _cmd_compare,
        "changepoint": _cmd_changepoint,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
