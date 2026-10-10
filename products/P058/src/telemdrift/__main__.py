"""Command-line interface: ``python -m telemdrift``.

Subcommands
-----------
``detectors``   list the detectors, their calibrated scalar and their default.
``defaults``    measure ARL0 at every detector's own default threshold.
``calibrate``   calibrate the analytic detectors to a target ARL0.
``arl``         ARL0 and ARL1 at equal ARL0 for one change type.
``tradeoff``    the delay-versus-false-alarm curve for one detector.
``transient``   firing rate on the transient negative control.
``trace``       alarm-ratio traces for one seeded episode.

Compute budget
--------------
Every subcommand that calibrates measures ARL0 over seeded stationary streams,
which is the expensive part. ``--budget full`` (the default) uses the published
configuration: 180000 calibration samples and 400000 held-out samples per
detector, about a minute of wall clock for the five analytic detectors on two
cores. ``--budget quick`` cuts both by more than an order of magnitude so a
reader can see the shape of the output in a few seconds. **Quick-budget numbers
have much wider error bars and are not the numbers in README.md**; every
subcommand says which budget it ran under, and the standard errors it prints are
the honest ones for that budget.

Exit status
-----------
0 on success. **2** when a run produced a finding the caller should not ignore:
a calibration that failed to bracket its target, or an ARL1 that is censored and
therefore only a lower bound. The non-zero status is a signal, not an error --
the output is still complete and still correct, and ``tests/test_cli.py``
asserts on the status for both cases.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from . import __version__
from .benchmark import (
    DETECTOR_LABELS,
    STANDARD,
    BenchmarkConfig,
    analytic_factory,
    calibrate_all_analytic,
    default_threshold_operating_points,
    measure_change_response,
    tradeoff_curve,
)
from .detectors import (
    ADWIN,
    ANALYTIC_DETECTORS,
    CUSUM,
    EWMA,
    PageHinkley,
    WindowedKS,
    alarm_ratio_trace,
    first_alarm_at_or_after,
)
from .scoring import wilson_interval
from .streams import CHANGE_TYPES, ChangeSpec, change_stream

_CLASSES = {
    "cusum": CUSUM,
    "page_hinkley": PageHinkley,
    "ewma": EWMA,
    "ks": WindowedKS,
    "adwin": ADWIN,
}

EXIT_OK = 0
EXIT_FINDING = 2

#: A reduced configuration for a quick look. Same seeds, far fewer samples, so
#: the shape of every table is identical and the error bars are much wider. Never
#: used for a published number.
QUICK = BenchmarkConfig(
    cal_seeds=(58_101, 58_102),
    eval_seeds=(58_201, 58_202),
    cal_length=8_000,
    eval_length=8_000,
    sweep_seeds=(58_301, 58_302),
    sweep_length=8_000,
    arl1_replicates=60,
)


def _config(args: argparse.Namespace) -> BenchmarkConfig:
    """The benchmark configuration this invocation uses."""
    return QUICK if getattr(args, "budget", "full") == "quick" else STANDARD


def _budget_line(args: argparse.Namespace) -> str:
    cfg = _config(args)
    tag = "quick" if cfg is QUICK else "full"
    note = ("  REDUCED BUDGET: wider error bars, not the README numbers."
            if cfg is QUICK else "")
    return (f"budget={tag}: calibration {len(cfg.cal_seeds)}x{cfg.cal_length}, "
            f"held-out {len(cfg.eval_seeds)}x{cfg.eval_length} samples.{note}")


def _spec(kind: str, magnitude: float, duration: int) -> ChangeSpec:
    return ChangeSpec(kind, magnitude, duration)


def _emit(payload: dict[str, Any], as_json: bool, lines: list[str]) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable))
    else:
        for line in lines:
            print(line)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def cmd_detectors(args: argparse.Namespace) -> int:
    rows = []
    lines = [
        "detector       scalar  default      note",
        "-------------- ------- ------------ ----------------------------------------",
    ]
    for key in ANALYTIC_DETECTORS:
        cls = _CLASSES[key]
        det = cls()
        default = cls.default_threshold()
        rows.append(
            {"key": key, "label": DETECTOR_LABELS[key],
             "scalar": det.threshold_name, "default_threshold": default}
        )
        lines.append(
            f"{key:14s} {det.threshold_name:7s} {default:<12.6g} {cls.__doc__.splitlines()[0]}"
        )
    rows.append({"key": "learned", "label": DETECTOR_LABELS["learned"],
                 "scalar": "p*", "default_threshold": 0.5})
    lines.append(f"{'learned':14s} {'p*':7s} {0.5:<12.6g} "
                 "Random-forest score on eight windowed features.")
    lines.append("")
    lines.append("Default thresholds are shipped in order to be measured, not used:")
    lines.append("run `python -m telemdrift defaults` to see how far apart they are.")
    _emit({"detectors": rows}, args.json, lines)
    return EXIT_OK


def cmd_defaults(args: argparse.Namespace) -> int:
    cfg = _config(args)
    res = default_threshold_operating_points(cfg)
    payload: dict[str, Any] = {"target_arl0": cfg.target_arl0, "detectors": {}}
    lines = [
        f"Measured ARL0 at each detector's own default threshold "
        f"({len(cfg.eval_seeds)} seeds x {cfg.eval_length} samples).",
        _budget_line(args),
        "",
        "detector        default      ARL0        SEM      rel.SEM   runs",
        "--------------- ------------ ----------- -------- --------- -----",
    ]
    for key, r in res.items():
        d = _CLASSES[key].default_threshold()
        payload["detectors"][key] = {
            "default_threshold": d, "arl0": r.arl0, "sem": r.sem, "n_runs": r.n_runs
        }
        lines.append(
            f"{DETECTOR_LABELS[key]:15s} {d:<12.6g} {r.arl0:11.1f} {r.sem:8.1f} "
            f"{100 * r.relative_sem:8.2f}% {r.n_runs:5d}"
        )
    arl0s = [r.arl0 for r in res.values()]
    spread = max(arl0s) / min(arl0s)
    payload["spread_factor"] = spread
    lines.append("")
    lines.append(f"Spread between the widest and narrowest default: {spread:.1f}x.")
    lines.append("Any detector comparison made at these thresholds compares "
                 "operating points, not detectors.")
    _emit(payload, args.json, lines)
    return EXIT_OK


def cmd_calibrate(args: argparse.Namespace) -> int:
    cfg = _config(args)
    cals = calibrate_all_analytic(cfg, target_arl0=args.target_arl0)
    payload: dict[str, Any] = {"target_arl0": args.target_arl0, "detectors": {}}
    lines = [f"Calibrating to target ARL0 = {args.target_arl0:.0f} samples.",
             f"calibration seeds {list(cfg.cal_seeds)} x {cfg.cal_length} samples; "
             f"held-out seeds {list(cfg.eval_seeds)} x {cfg.eval_length} samples.",
             _budget_line(args),
             ""]
    finding = False
    for key, c in cals.items():
        payload["detectors"][key] = {
            "threshold": c.threshold,
            "calibration_arl0": c.calibration_arl0,
            "achieved_arl0": c.achieved.arl0,
            "achieved_sem": c.achieved.sem,
            "n_runs": c.achieved.n_runs,
            "target_error": c.target_error,
            "bracketing_failed": c.bracketing_failed,
        }
        lines.append(c.summary())
        finding = finding or c.bracketing_failed
    if finding:
        lines.append("")
        lines.append("FINDING: at least one detector could not bracket the target ARL0. "
                     "Exit status 2.")
    _emit(payload, args.json, lines)
    return EXIT_FINDING if finding else EXIT_OK


def cmd_arl(args: argparse.Namespace) -> int:
    cfg = _config(args)
    spec = _spec(args.change, args.magnitude, args.duration)
    cals = calibrate_all_analytic(cfg, target_arl0=args.target_arl0)
    payload: dict[str, Any] = {
        "target_arl0": args.target_arl0,
        "change": {"kind": spec.kind, "magnitude": spec.magnitude,
                   "duration": spec.duration},
        "replicates": args.replicates,
        "detectors": {},
    }
    lines = [
        f"Change: {spec.kind} magnitude {spec.magnitude:g}"
        + (f" duration {spec.duration}" if spec.kind == "transient" else ""),
        f"All detectors at measured ARL0 ~ {args.target_arl0:.0f} samples. "
        f"{args.replicates} seeded replicates, delay censored at "
        f"{cfg.arl1_budget} samples.",
        _budget_line(args),
        "",
        "detector        threshold    ARL0(held-out)  ARL1          SEM   censored  pre-alarm",
        "--------------- ------------ --------------- ------------ ------ --------- ---------",
    ]
    censored_any = False
    for key, c in cals.items():
        r = measure_change_response(key, c.threshold, spec, cfg,
                                    replicates=args.replicates)
        payload["detectors"][key] = {
            "threshold": c.threshold, "arl0": c.achieved.arl0,
            "arl1": r.arl1, "arl1_sem": r.sem, "n_used": r.n_used,
            "censored_rate": r.censored_rate,
            "pre_change_alarm_rate": r.pre_change_alarm_rate,
            "is_lower_bound": r.is_lower_bound,
        }
        flag = ">=" if r.is_lower_bound else "  "
        lines.append(
            f"{DETECTOR_LABELS[key]:15s} {c.threshold:<12.6g} {c.achieved.arl0:15.1f} "
            f"{flag}{r.arl1:10.1f} {r.sem:6.1f} {100 * r.censored_rate:8.1f}% "
            f"{100 * r.pre_change_alarm_rate:8.1f}%"
        )
        censored_any = censored_any or r.is_lower_bound
    if spec.kind == "transient":
        lines.append("")
        lines.append("NOTE: the transient is a negative control. Every alarm counted "
                     "above is a FALSE alarm, and a large 'ARL1' here is the good "
                     "outcome, not a slow detector.")
    if censored_any:
        lines.append("")
        lines.append("FINDING: at least one ARL1 is right-censored and is a LOWER "
                     "BOUND only (marked >=). Exit status 2.")
    _emit(payload, args.json, lines)
    return EXIT_FINDING if censored_any else EXIT_OK


def cmd_tradeoff(args: argparse.Namespace) -> int:
    cfg = _config(args)
    cls = _CLASSES[args.detector]
    base = cls.default_threshold()
    factors = np.geomspace(1.0 / args.span, args.span, args.points)
    thresholds = [base * f for f in factors]
    if args.detector == "adwin":
        thresholds = [min(max(t, 1e-12), 0.999_999) for t in thresholds]
    if args.detector == "ks":
        thresholds = [min(max(t, 1e-6), 1.0) for t in thresholds]
    pts = tradeoff_curve(args.detector, thresholds,
                         _spec(args.change, args.magnitude, args.duration),
                         cfg, replicates=args.replicates)
    payload: dict[str, Any] = {"detector": args.detector, "points": []}
    lines = [
        f"{DETECTOR_LABELS[args.detector]} delay-versus-false-alarm curve on "
        f"{args.change} magnitude {args.magnitude:g}.",
        f"{len(cfg.sweep_seeds)} x {cfg.sweep_length} stationary samples per "
        f"ARL0 point; {args.replicates} replicates per ARL1 point.",
        _budget_line(args),
        "",
        "threshold      ARL0     ARL0 SEM     ARL1   ARL1 SEM  censored",
        "-------------- -------- -------- -------- --------- ---------",
    ]
    censored_any = False
    for p in pts:
        payload["points"].append({
            "threshold": p.threshold, "arl0": p.arl0.arl0, "arl0_sem": p.arl0.sem,
            "arl1": p.arl1.arl1, "arl1_sem": p.arl1.sem,
            "censored_rate": p.arl1.censored_rate,
        })
        lines.append(
            f"{p.threshold:<14.6g} {p.arl0.arl0:8.1f} {p.arl0.sem:8.1f} "
            f"{p.arl1.arl1:8.1f} {p.arl1.sem:9.1f} {100 * p.arl1.censored_rate:8.1f}%"
        )
        censored_any = censored_any or p.arl1.is_lower_bound
    if censored_any:
        lines.append("")
        lines.append("FINDING: at least one point is right-censored; its ARL1 is a "
                     "LOWER BOUND. Exit status 2.")
    _emit(payload, args.json, lines)
    return EXIT_FINDING if censored_any else EXIT_OK


def cmd_transient(args: argparse.Namespace) -> int:
    cfg = _config(args)
    spec = _spec("transient", args.magnitude, args.duration)
    cals = calibrate_all_analytic(cfg, target_arl0=args.target_arl0)
    payload: dict[str, Any] = {
        "amplitude": args.magnitude, "duration": args.duration,
        "target_arl0": args.target_arl0, "replicates": args.replicates,
        "detectors": {},
    }
    lines = [
        f"Transient negative control: amplitude {args.magnitude:g} sigma for "
        f"{args.duration} samples, then the channel recovers.",
        "There is no change to detect. Every alarm below is a false alarm.",
        f"All detectors at measured ARL0 ~ {args.target_arl0:.0f}; "
        f"{args.replicates} seeded replicates.",
        _budget_line(args),
        "",
        "detector        fired   rate    Wilson 95 % CI",
        "--------------- ------- ------- --------------------",
    ]
    for key, c in cals.items():
        fired = 0
        for s in cfg.arl1_seeds(2)[: args.replicates]:
            stream, _ = change_stream(cfg.pre_length, cfg.arl1_budget, spec, s)
            det = analytic_factory(key, c.threshold)()
            det.reset()
            for value in stream:
                if det.update(value):
                    fired += 1
                    break
        rate = fired / args.replicates
        lo, hi = wilson_interval(fired, args.replicates)
        payload["detectors"][key] = {"threshold": c.threshold, "fired": fired,
                                     "rate": rate, "wilson_ci": [lo, hi]}
        lines.append(f"{DETECTOR_LABELS[key]:15s} {fired:7d} {100 * rate:6.1f}% "
                     f"[{100 * lo:5.1f}%, {100 * hi:5.1f}%]")
    _emit(payload, args.json, lines)
    return EXIT_OK


def cmd_trace(args: argparse.Namespace) -> int:
    cfg = _config(args)
    spec = _spec(args.change, args.magnitude, args.duration)
    stream, idx = change_stream(cfg.pre_length, args.post, spec, args.seed)
    cals = calibrate_all_analytic(cfg, target_arl0=args.target_arl0)
    payload: dict[str, Any] = {"seed": args.seed, "change_index": idx, "detectors": {}}
    lines = [
        f"Single episode, seed {args.seed}: {spec.kind} magnitude {spec.magnitude:g} "
        f"at sample {idx}.",
        f"All detectors at measured ARL0 ~ {args.target_arl0:.0f}.",
        _budget_line(args),
        "",
        "Alarms before the change index are false alarms, not detections, and the",
        "detector is reset on each one, which is what it does in service.",
        "",
        "detector        first alarm   delay  pre-alarms   peak ratio",
        "--------------- ------------ ------- ----------- -----------",
    ]
    for key, c in cals.items():
        ratios, alarms = alarm_ratio_trace(analytic_factory(key, c.threshold)(), stream)
        first = first_alarm_at_or_after(alarms, idx)
        delay = None if first < 0 else first - idx
        pre = int((alarms < idx).sum())
        payload["detectors"][key] = {
            "threshold": c.threshold, "first_alarm_after_change": first,
            "delay": delay, "pre_change_alarms": pre,
            "peak_ratio": float(np.max(ratios)),
        }
        shown = "none" if first < 0 else str(first)
        dtxt = "n/a" if delay is None else str(delay)
        lines.append(f"{DETECTOR_LABELS[key]:15s} {shown:>12s} {dtxt:>7s} {pre:>11d} "
                     f"{float(np.max(ratios)):11.3f}")
    _emit(payload, args.json, lines)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m telemdrift",
        description=(
            "Streaming change-detection benchmark for a univariate telemetry "
            "channel, scored on detection delay against false-alarm rate. "
            "A benchmark harness, not a streaming framework: use river for "
            "production streaming. Research-grade; not flight-qualified, not "
            "certified, not approved for operational aerospace use."
        ),
        epilog="Exit status 2 means a finding (failed bracketing, or a censored "
               "ARL1 that is only a lower bound), not an error.",
    )
    p.add_argument("--version", action="version", version=f"telemdrift {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--json", action="store_true", help="emit JSON instead of text")
        sp.add_argument(
            "--budget", choices=("full", "quick"), default="full",
            help="Monte Carlo budget. 'full' is the published configuration; "
                 "'quick' is more than an order of magnitude smaller, for a look "
                 "at the shape of the output, and its error bars are much wider.",
        )

    def add_change(sp, default_kind="mean_step", default_mag=1.0):
        sp.add_argument("--change", choices=CHANGE_TYPES, default=default_kind)
        sp.add_argument("--magnitude", type=float, default=default_mag,
                        help="in pre-change sigma units (sigma multiplier for "
                             "variance_step, sigma/sample for drift_ramp)")
        sp.add_argument("--duration", type=int, default=20,
                        help="transient duration in samples; ignored otherwise")

    sp = sub.add_parser("detectors", help="list detectors and their calibrated scalar")
    add_common(sp)
    sp.set_defaults(func=cmd_detectors)

    sp = sub.add_parser("defaults", help="measure ARL0 at each default threshold")
    add_common(sp)
    sp.set_defaults(func=cmd_defaults)

    sp = sub.add_parser("calibrate", help="calibrate analytic detectors to a target ARL0")
    sp.add_argument("--target-arl0", type=float, default=STANDARD.target_arl0)
    add_common(sp)
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("arl", help="ARL0 and ARL1 at equal ARL0 for one change type")
    sp.add_argument("--target-arl0", type=float, default=STANDARD.target_arl0)
    sp.add_argument("--replicates", type=int, default=120)
    add_change(sp)
    add_common(sp)
    sp.set_defaults(func=cmd_arl)

    sp = sub.add_parser("tradeoff", help="delay-versus-false-alarm curve for one detector")
    sp.add_argument("--detector", choices=ANALYTIC_DETECTORS, default="cusum")
    sp.add_argument("--points", type=int, default=5)
    sp.add_argument("--span", type=float, default=2.0,
                    help="threshold sweep spans default/span to default*span")
    sp.add_argument("--replicates", type=int, default=120)
    add_change(sp)
    add_common(sp)
    sp.set_defaults(func=cmd_tradeoff)

    sp = sub.add_parser("transient", help="firing rate on the transient negative control")
    sp.add_argument("--target-arl0", type=float, default=STANDARD.target_arl0)
    sp.add_argument("--magnitude", type=float, default=4.0)
    sp.add_argument("--duration", type=int, default=20)
    sp.add_argument("--replicates", type=int, default=120)
    add_common(sp)
    sp.set_defaults(func=cmd_transient)

    sp = sub.add_parser("trace", help="alarm-ratio traces for one seeded episode")
    sp.add_argument("--target-arl0", type=float, default=STANDARD.target_arl0)
    sp.add_argument("--seed", type=int, default=58_901)
    sp.add_argument("--post", type=int, default=600)
    add_change(sp)
    add_common(sp)
    sp.set_defaults(func=cmd_trace)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
