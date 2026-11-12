"""Command-line interface: ``python -m thermalbudget``.

Every subcommand that loads a trace requires an explicit ``--origin``. There is
no default and there is no inference from the filename, because a filename is
not evidence and a default would be the one place a synthetic figure could
acquire a measured label by accident.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import (
    VALIDATION_BASIS,
    VALIDATION_LEVEL,
    __version__,
    limits,
    measured_evidence,
    schema,
)
from .equilibrium import (
    DEFAULT_SLOPE_THRESHOLD_C_PER_S,
    DEFAULT_WINDOW_S,
    equilibrium_sensitivity,
    equilibrium_test,
)
from .provenance import Origin
from .rc_model import (
    RelaxationFitError,
    fit_exponential_relaxation,
    tau_from_integral,
    tau_from_log_differences,
)
from .report import analyse, render
from .trace import TraceDataAbsentError, load_campaign, load_trace

ORIGINS = {"measured": Origin.MEASURED, "synthetic": Origin.SYNTHETIC}


def _add_origin(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--origin",
        required=True,
        choices=sorted(ORIGINS),
        help="where the data came from; required, never defaulted",
    )


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, built separately so a test can introspect it."""
    parser = argparse.ArgumentParser(
        prog="thermalbudget",
        description=(
            "Sustained on-board throughput after thermal equilibrium, against the "
            "short-burst figure, with the gap as the headline. Research-grade; not "
            "flight-qualified, not certified, not approved for operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"thermalbudget {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("schema", help="print the binding trace schema")
    sub.add_parser("limits", help="print the declared campaign limits and unmeasured claims")
    sub.add_parser("validation-level", help="print the claimed validation level and its basis")

    equilibrium = sub.add_parser(
        "equilibrium", help="test one trace for thermal equilibrium"
    )
    equilibrium.add_argument("csv", help="path to a trace CSV")
    _add_origin(equilibrium)
    equilibrium.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    equilibrium.add_argument(
        "--threshold-c-per-s", type=float, default=DEFAULT_SLOPE_THRESHOLD_C_PER_S
    )

    tau = sub.add_parser("tau", help="the three independent tau estimates for one trace")
    tau.add_argument("csv", help="path to a trace CSV")
    _add_origin(tau)

    report = sub.add_parser("report", help="the full analysis over a trace directory")
    report.add_argument("directory", help="directory of trace CSVs plus manifest.json")
    _add_origin(report)
    report.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    report.add_argument(
        "--threshold-c-per-s", type=float, default=DEFAULT_SLOPE_THRESHOLD_C_PER_S
    )
    report.add_argument("--resamples", type=int, default=20000)
    report.add_argument(
        "--no-learned", action="store_true", help="classical baselines only, skip step 5"
    )
    report.add_argument("--json", action="store_true", help="emit the headline as JSON")
    return parser


def _print_schema() -> int:
    print("Binding trace schema (docs/SPEC.md):")
    print("  columns: " + ",".join(schema.expected_columns()))
    print(f"  die-temperature channel : {schema.DIE_COLUMN}")
    print(f"  missing sentinel        : {schema.MISSING_SENTINEL:g} -- never a value")
    print(f"  readable zones          : {', '.join(schema.READABLE_ZONES)}")
    print(
        "  present but unreadable  : "
        + ", ".join(schema.UNREADABLE_ZONES)
        + " (declared limit 4)"
    )
    print(
        "  temperature scale       : inferred from the data, not assumed; "
        f"degree band {schema.DEGREE_BAND}, milli-degree band {schema.MILLIDEGREE_BAND}"
    )
    return 0


def _print_limits() -> int:
    print(f"Board      : {limits.BOARD}")
    print(f"Power mode : {limits.POWER_MODE} (the only one reachable)")
    print(f"Platform   : {limits.PLATFORM}")
    print("")
    print("Declared limits of the campaign, recorded before it ran:")
    for item in limits.DECLARED_LIMITS:
        print(f"  [{item.key}] {item.headline}")
        print(f"      {item.detail}")
        print(f"      consequence: {item.consequence}")
    print("")
    print("UNMEASURED, by name:")
    for claim in limits.UNMEASURED_BY_NAME:
        print(f"  [{claim.key}] {claim.claim}")
        print(f"      reason: {claim.reason}")
    print("")
    print("STRUCTURALLY UNMEASURABLE evaluation axes:")
    for claim in limits.structurally_unmeasurable_axes():
        print(f"  [{claim.key}] {claim.claim}")
    return 0


def _print_validation_level() -> int:
    print(f"thermalbudget.VALIDATION_LEVEL = {VALIDATION_LEVEL}")
    print("")
    print(VALIDATION_BASIS)
    print("")
    evidence = measured_evidence()
    print("Measured evidence on disk right now:")
    print(f"  directory : {evidence['traces_directory']}")
    print(f"  CSV traces: {evidence['csv_count']}")
    print(f"  manifest  : {'present' if evidence['manifest'] else 'absent'}")
    if evidence["runs"]:
        print(f"  runs      : {', '.join(evidence['runs'])}")
    print(f"  {evidence['statement']}")
    return 0


def _equilibrium(arguments: argparse.Namespace) -> int:
    trace = load_trace(arguments.csv, origin=ORIGINS[arguments.origin])
    verdict = equilibrium_test(
        trace,
        window_s=arguments.window_s,
        threshold_c_per_s=arguments.threshold_c_per_s,
    )
    print(f"run     : {verdict.label}  (origin {verdict.origin})")
    print(f"status  : {verdict.status}")
    print(f"slope   : {verdict.slope_c_per_s:+.6f} degC/s")
    print(
        f"interval: [{verdict.slope_ci_c_per_s[0]:+.6f}, {verdict.slope_ci_c_per_s[1]:+.6f}] "
        "degC/s (AR(1)-corrected)"
    )
    print(f"samples : {verdict.n_samples} finite, {verdict.n_missing} missing")
    print(f"mean die: {verdict.mean_die_c:.3f} degC")
    print(f"reason  : {verdict.reason}")
    print("threshold ladder:")
    for threshold, status in equilibrium_sensitivity(
        trace, window_s=arguments.window_s
    ).items():
        print(f"  {threshold:g} degC/s -> {status}")
    return 0


def _tau(arguments: argparse.Namespace) -> int:
    trace = load_trace(arguments.csv, origin=ORIGINS[arguments.origin])
    print(f"run: {trace.label}  (origin {trace.origin})")
    try:
        fit = fit_exponential_relaxation(trace.t_rel, trace.die_temperature_c)
    except RelaxationFitError as error:
        print(f"least-squares route REFUSED: {error}")
        return 0
    print(
        f"  least squares : tau {fit.tau_s:.3f} s, asymptote {fit.asymptote_c:.3f} degC, "
        f"rms {fit.rms_residual_c:.4f} degC, rho {fit.residual_autocorrelation:+.3f}, "
        f"span {fit.span_in_tau:.2f} tau"
    )
    for estimator, keywords in (
        (tau_from_log_differences, {}),
        (tau_from_integral, {"asymptote_c": fit.asymptote_c}),
    ):
        try:
            estimate = estimator(trace.t_rel, trace.die_temperature_c, **keywords)
        except RelaxationFitError as error:
            print(f"  {estimator.__name__} REFUSED: {error}")
            continue
        print(
            f"  {estimate.method:<14s}: tau {estimate.tau_s:.3f} s +/- {estimate.stderr_s:.3f} "
            f"s, conditioning {estimate.conditioning:.4g}, n={estimate.n}"
        )
    return 0


def _report(arguments: argparse.Namespace) -> int:
    campaign = load_campaign(arguments.directory, origin=ORIGINS[arguments.origin])
    analysis = analyse(
        campaign,
        window_s=arguments.window_s,
        threshold_c_per_s=arguments.threshold_c_per_s,
        n_resamples=arguments.resamples,
        run_learned=not arguments.no_learned,
    )
    if arguments.json:
        head = analysis.headline
        print(
            json.dumps(
                {
                    "origin": str(analysis.origin),
                    "trace_directory": Path(arguments.directory).name,
                    "burst_iters_per_s": head.burst_iters_per_s,
                    "sustained_iters_per_s": head.sustained_iters_per_s,
                    "sustained_fraction": head.sustained_fraction.point,
                    "sustained_fraction_ci": [
                        head.sustained_fraction.ci_low,
                        head.sustained_fraction.ci_high,
                    ],
                    "overstatement": head.overstatement.point,
                    "overstatement_ci": [
                        head.overstatement.ci_low,
                        head.overstatement.ci_high,
                    ],
                    "verdict": head.verdict,
                    "validation_level": VALIDATION_LEVEL,
                },
                indent=2,
            )
        )
        return 0
    print(render(analysis), end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    arguments = build_parser().parse_args(argv)
    handlers = {
        "schema": lambda _: _print_schema(),
        "limits": lambda _: _print_limits(),
        "validation-level": lambda _: _print_validation_level(),
        "equilibrium": _equilibrium,
        "tau": _tau,
        "report": _report,
    }
    try:
        return handlers[arguments.command](arguments)
    except TraceDataAbsentError as error:
        print(f"DATA ABSENT: {error}", file=sys.stderr)
        return 2
    except (schema.TraceSchemaError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
