"""Command line interface: ``python -m safemodetree``.

Exit codes:

- ``0`` success.
- ``1`` the requested check produced a finding and ``--fail-on-findings`` was
  given.
- ``2`` a refused request: an unreadable or invalid declaration, an unknown
  mode, or an argument outside its valid range.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .automaton import DetectionAssumption, build_automaton
from .diagnostics import analyse
from .evaluate import compare_detectors, compare_policies, conservatism_sweep
from .reach import reachability_report
from .spec import SafeModeSpec, load_spec, packaged_spec_names, packaged_spec_path

_DEFAULT_SPEC = "reaction_wheel_fmea"


def _resolve_spec(reference: str) -> SafeModeSpec:
    """Load a declaration from a packaged name or a filesystem path."""
    candidate = Path(reference)
    if candidate.exists():
        return load_spec(candidate)
    return load_spec(packaged_spec_path(reference))


def _assumption(args: argparse.Namespace) -> DetectionAssumption:
    budget = math.inf if args.latency_budget is None else float(args.latency_budget)
    return DetectionAssumption(
        latency_budget_s=budget, require_detection=not args.assume_perfect_detection
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m safemodetree",
        description=(
            "From an FMEA-shaped declaration to a checked fault-response policy. "
            "Research-grade; not flight-qualified, not certified, not approved for "
            "operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"safemodetree {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--spec",
            default=_DEFAULT_SPEC,
            help=(
                "a packaged declaration name or a path to a YAML file; packaged: "
                + ", ".join(packaged_spec_names())
            ),
        )
        target.add_argument("--json", action="store_true", help="machine-readable output")

    def add_assumption(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--latency-budget",
            type=float,
            default=None,
            help="detection latency budget in seconds; unbounded when omitted",
        )
        target.add_argument(
            "--assume-perfect-detection",
            action="store_true",
            help="treat every fault as actionable regardless of its declared detectability",
        )

    info = subparsers.add_parser("info", help="summarise a declaration")
    add_common(info)

    check = subparsers.add_parser("check", help="run every deterministic check")
    add_common(check)
    add_assumption(check)
    check.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="exit 1 when any check produces a finding",
    )

    reach = subparsers.add_parser("reach", help="safe-mode reachability per fault state")
    add_common(reach)
    add_assumption(reach)
    reach.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="exit 1 when a fault state cannot reach a declared safe mode",
    )

    ambiguity = subparsers.add_parser("ambiguity", help="ambiguity classes and their costs")
    add_common(ambiguity)
    add_assumption(ambiguity)

    benchmark = subparsers.add_parser(
        "benchmark", help="learned classifier against threshold logic at equal false-alarm rate"
    )
    add_common(benchmark)
    benchmark.add_argument("--noise-scale", type=float, default=2.0)
    benchmark.add_argument("--n-per-fault", type=int, default=400)
    benchmark.add_argument("--seed", type=int, default=63001)
    benchmark.add_argument(
        "--calibration", default="isotonic", choices=("none", "isotonic", "sigmoid")
    )

    policy = subparsers.add_parser("policy", help="policy cost and the cost of conservatism")
    add_common(policy)
    add_assumption(policy)
    policy.add_argument("--noise-scale", type=float, default=2.0)
    policy.add_argument("--n-per-fault", type=int, default=400)
    policy.add_argument("--seed", type=int, default=63001)
    policy.add_argument("--mode", default=None, help="mode the events occur in")
    policy.add_argument("--confidence-threshold", type=float, default=0.7)
    policy.add_argument("--detector", default="learned", choices=("learned", "baseline"))
    policy.add_argument("--sweep", action="store_true", help="also print the threshold sweep")
    return parser


def _dump(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _cmd_info(args: argparse.Namespace, spec: SafeModeSpec) -> int:
    automaton = build_automaton(spec)
    if args.json:
        _dump(
            {
                "name": spec.name,
                "modes": list(spec.mode_ids),
                "safe_modes": list(spec.safe_mode_ids),
                "faults": list(spec.fault_ids),
                "responses": [r.id for r in spec.responses],
                "channels": list(spec.observability.channel_ids),
                "ambiguity_groups": [
                    list(g) for g in spec.observability.indistinguishable_groups()
                ],
                "n_states": len(automaton),
                "n_transitions": len(automaton.transitions),
            }
        )
        return 0
    sys.stdout.write(spec.summary() + "\n")
    sys.stdout.write(automaton.summary() + "\n")
    for group in spec.observability.indistinguishable_groups():
        sys.stdout.write(f"indistinguishable: {', '.join(group)}\n")
    return 0


def _cmd_check(args: argparse.Namespace, spec: SafeModeSpec) -> int:
    report = analyse(spec, _assumption(args))
    if args.json:
        _dump(
            {
                "spec": report.spec_name,
                "assumption": report.assumption.describe(),
                "findings": report.finding_counts,
                "total_findings": report.total_findings,
                "ok": report.ok,
                "vacuous": [v.response_id for v in report.vacuous],
                "unreachable_fault_states": [
                    str(r.state) for r in report.reachability.unreachable_rows
                ],
                "unreachable_safe_modes": list(report.reachability.unreachable_safe_modes),
                "conflicting_classes": [
                    {"faults": list(c.faults), "mode": c.mode,
                     "conservative": c.conservative_response,
                     "conservative_cost_min": c.conservative_cost_min}
                    for c in report.conflicting_ambiguity
                ],
            }
        )
    else:
        sys.stdout.write(report.report() + "\n")
    return 1 if (args.fail_on_findings and not report.ok) else 0


def _cmd_reach(args: argparse.Namespace, spec: SafeModeSpec) -> int:
    report = reachability_report(spec, _assumption(args))
    if args.json:
        _dump(
            {
                "assumption": report.assumption.describe(),
                "all_reachable": report.all_reachable,
                "unreachable_safe_modes": list(report.unreachable_safe_modes),
                "rows": [
                    {
                        "mode": r.state.mode,
                        "fault": r.state.fault,
                        "reachable": r.reachable,
                        "cost_min": r.cost_min,
                        "hops": r.hops,
                        "path": list(r.path),
                        "safe_mode": r.safe_mode,
                    }
                    for r in report.rows
                ],
            }
        )
    else:
        sys.stdout.write(report.table() + "\n")
        sys.stdout.write(
            f"all fault states reach a declared safe mode: {report.all_reachable}\n"
        )
        if report.unreachable_safe_modes:
            sys.stdout.write(
                "unreachable declared safe modes: "
                + ", ".join(report.unreachable_safe_modes)
                + "\n"
            )
    return 1 if (args.fail_on_findings and not report.all_reachable) else 0


def _cmd_ambiguity(args: argparse.Namespace, spec: SafeModeSpec) -> int:
    report = analyse(spec, _assumption(args))
    if args.json:
        _dump(
            {
                "assumption": report.assumption.describe(),
                "classes": [
                    {
                        "faults": list(c.faults),
                        "mode": c.mode,
                        "signature": list(c.signature),
                        "conflicting": c.conflicting,
                        "preferred": c.preferred,
                        "conservative": c.conservative_response,
                        "conservative_cost_min": c.conservative_cost_min,
                        "specific_cost_min": c.best_specific_cost_min,
                        "conservatism_cost_min": c.conservatism_cost_min,
                    }
                    for c in report.ambiguity
                ],
            }
        )
        return 0
    if not report.ambiguity:
        sys.stdout.write("no declared ambiguity class in this declaration\n")
        return 0
    header = (
        f"{'faults':<36} {'mode':<16} {'conflict':<9} {'conservative':<18} "
        f"{'cons_min':>9} {'cost_of_cons':>13}"
    )
    sys.stdout.write(header + "\n" + "-" * len(header) + "\n")
    for item in report.ambiguity:
        sys.stdout.write(
            f"{'+'.join(item.faults):<36} {item.mode:<16} "
            f"{'YES' if item.conflicting else 'no':<9} "
            f"{item.conservative_response or 'NONE':<18} "
            f"{item.conservative_cost_min:>9.3f} {item.conservatism_cost_min:>13.3f}\n"
        )
    return 0


def _cmd_benchmark(args: argparse.Namespace, spec: SafeModeSpec) -> int:
    result = compare_detectors(
        spec,
        n_per_fault=args.n_per_fault,
        noise_scale=args.noise_scale,
        seed=args.seed,
        calibration_method=args.calibration,
    )
    scores = {k: result[k] for k in ("baseline", "learned", "learned_argmax")}
    if args.json:
        _dump(
            {
                "noise_scale": result["noise_scale"],
                "seed": result["seed"],
                "tau": result["tau"],
                "target_false_alarm_rate": result["target_far"],
                "accuracy_ceiling": result["ceiling"],
                "sizes": list(result["sizes"]),  # type: ignore[arg-type]
                "scores": {
                    name: {
                        "operating_point": s.operating_point,
                        "false_alarm_rate": s.false_alarm_rate,
                        "detection_rate": s.detection_rate,
                        "identification_accuracy": s.identification_accuracy,
                        "overall_accuracy": s.overall_accuracy,
                        "ece": s.calibration.ece,
                        "mce": s.calibration.mce,
                        "brier": s.brier,
                    }
                    for name, s in scores.items()
                },
            }
        )
        return 0
    sys.stdout.write(
        f"noise_scale {result['noise_scale']:g}  seed {result['seed']}  "
        f"train/cal/test {result['sizes']}\n"
        f"baseline false-alarm rate on the calibration split: "
        f"{float(result['target_far']):.6f}  matched tau {float(result['tau']):.6f}\n"
        f"accuracy ceiling from the declared observability model: "
        f"{float(result['ceiling']):.6f}\n\n"
    )
    header = (
        f"{'detector':<22} {'op_point':>8} {'FAR':>9} {'detect':>9} {'ident_acc':>9} "
        f"{'overall':>9} {'ECE':>8} {'MCE':>8} {'brier':>9}"
    )
    sys.stdout.write(header + "\n" + "-" * len(header) + "\n")
    for score in scores.values():
        sys.stdout.write(score.row() + "\n")
    return 0


def _cmd_policy(args: argparse.Namespace, spec: SafeModeSpec) -> int:
    comparison = compare_detectors(
        spec,
        n_per_fault=args.n_per_fault,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )
    scores = compare_policies(
        spec,
        comparison,
        mode_id=args.mode,
        confidence_threshold=args.confidence_threshold,
        assumption=_assumption(args),
        detector=args.detector,
    )
    names = ("oracle", "always_specific", "confidence_routed", "always_conservative")
    if args.json:
        _dump(
            {
                "mode": scores["mode"],
                "detector": scores["detector"],
                "noise_scale": args.noise_scale,
                "confidence_threshold": args.confidence_threshold,
                "cost_of_conservatism_vs_oracle": scores["cost_of_conservatism_vs_oracle"],
                "cost_of_conservatism_vs_specific": scores["cost_of_conservatism_vs_specific"],
                "misroutes_avoided": scores["misroutes_avoided"],
                "policies": {
                    name: {
                        "confidence_threshold": scores[name].confidence_threshold,
                        "n_fault_events": scores[name].n_fault_events,
                        "n_commanded": scores[name].n_commanded,
                        "n_missed": scores[name].n_missed,
                        "mean_cost_min": scores[name].mean_cost_min,
                        "n_misrouted": scores[name].n_misrouted,
                        "n_unsafe": scores[name].n_unsafe,
                        "conservative_fraction": scores[name].conservative_fraction,
                        "false_alarm_cost_min": scores[name].false_alarm_cost_min,
                    }
                    for name in names
                },
            }
        )
        return 0
    sys.stdout.write(
        f"mode {scores['mode']}  detector {scores['detector']}  "
        f"noise_scale {args.noise_scale:g}  threshold {args.confidence_threshold:g}\n\n"
    )
    header = (
        f"{'policy':<22} {'thr':>6} {'cmded':>7} {'missed':>7} {'cost_min':>10} "
        f"{'misrout':>7} {'unsafe':>7} {'cons_fr':>7} {'fa_cost':>10}"
    )
    sys.stdout.write(header + "\n" + "-" * len(header) + "\n")
    for name in names:
        sys.stdout.write(scores[name].row() + "\n")
    sys.stdout.write(
        f"\ncost of conservatism against the oracle:   "
        f"{float(scores['cost_of_conservatism_vs_oracle']):+.4f} min per commanded event\n"
        f"cost of conservatism against always-specific: "
        f"{float(scores['cost_of_conservatism_vs_specific']):+.4f} min per commanded event\n"
        f"mis-routes avoided: {scores['misroutes_avoided']}\n"
    )
    if args.sweep:
        sys.stdout.write("\nrouting threshold sweep\n")
        sweep_header = (
            f"{'threshold':<22} {'thr':>6} {'cmded':>7} {'missed':>7} {'cost_min':>10} "
            f"{'misrout':>7} {'unsafe':>7} {'cons_fr':>7} {'fa_cost':>10}"
        )
        sys.stdout.write(sweep_header + "\n" + "-" * len(sweep_header) + "\n")
        for row in conservatism_sweep(
            spec, comparison, mode_id=args.mode, detector=args.detector
        ):
            sys.stdout.write(row.row() + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        spec = _resolve_spec(args.spec)
    except (KeyError, OSError, ValueError, TypeError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2
    handlers = {
        "info": _cmd_info,
        "check": _cmd_check,
        "reach": _cmd_reach,
        "ambiguity": _cmd_ambiguity,
        "benchmark": _cmd_benchmark,
        "policy": _cmd_policy,
    }
    try:
        return handlers[args.command](args, spec)
    except (KeyError, ValueError, TypeError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())
