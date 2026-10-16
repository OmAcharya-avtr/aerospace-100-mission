"""Command line interface: ``python -m watchdogspec``.

Exit codes
----------
``0``
    The specification is internally consistent (no error finding), or the
    subcommand is purely informational.
``1``
    At least one error finding, or with ``--strict`` at least one finding of
    any severity. For ``simulate``, the simulation contradicted the closed
    form.
``2``
    The specification could not be read or parsed, or the arguments were
    wrong.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .checks import Severity, run_checks
from .errors import SpecFormatError
from .latency import DEFAULT_TOL_MS
from .parse import load_spec
from .report import (
    format_codes,
    format_json,
    format_markdown,
    format_report,
    format_timing_table,
)
from .simulate import simulate_detection


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m watchdogspec",
        description=(
            "Check a declared watchdog and liveness specification for internal "
            "consistency. Research-grade; measures no running system."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="run every check over a specification")
    check.add_argument("spec", help="path to the specification YAML")
    check.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="output format (default: text)",
    )
    check.add_argument(
        "--tol-ms",
        type=float,
        default=DEFAULT_TOL_MS,
        help=f"absolute tolerance in ms for deadline comparisons (default: {DEFAULT_TOL_MS:g})",
    )
    check.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on warnings as well as errors",
    )

    latency = sub.add_parser("latency", help="print the worst-case timing table only")
    latency.add_argument("spec", help="path to the specification YAML")
    latency.add_argument("--tol-ms", type=float, default=DEFAULT_TOL_MS)

    sim = sub.add_parser(
        "simulate",
        help="cross-check the closed-form worst case against a seeded simulation",
    )
    sim.add_argument("spec", help="path to the specification YAML")
    sim.add_argument("--task", default=None, help="task id (default: every timed task)")
    sim.add_argument("--seeds", type=int, default=8, help="number of seeds (default: 8)")
    sim.add_argument("--trials", type=int, default=1000, help="trials per seed (default: 1000)")

    mermaid = sub.add_parser("mermaid", help="print a mermaid flowchart of the specification")
    mermaid.add_argument("spec", help="path to the specification YAML")

    sub.add_parser("codes", help="print the check catalogue")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code; never raises on bad input."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "codes":
        sys.stdout.write(format_codes())
        return 0

    try:
        spec = load_spec(args.spec)
    except SpecFormatError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    if args.command == "mermaid":
        from .diagram import render_mermaid

        sys.stdout.write(render_mermaid(spec))
        return 0

    if args.command == "latency":
        report = run_checks(spec, tol_ms=args.tol_ms)
        sys.stdout.write(format_timing_table(report) + "\n")
        return 0

    if args.command == "simulate":
        report = run_checks(spec)
        task_ids = sorted(report.timings) if args.task is None else [args.task]
        if args.seeds < 1 or args.trials < 1:
            sys.stderr.write("error: --seeds and --trials must both be at least 1\n")
            return 2
        failures = 0
        for task_id in task_ids:
            task = spec.task(task_id)
            if task is None or task_id not in report.timings:
                sys.stderr.write(f"error: no timed task '{task_id}' in {args.spec}\n")
                return 2
            closed = report.timings[task_id].detection_latency_ms
            adversarial = simulate_detection(
                task, spec.supervisor, seed=0, trials=1, adversarial=True
            )
            worst = -float("inf")
            violations = 0
            for seed in range(args.seeds):
                result = simulate_detection(
                    task, spec.supervisor, seed=seed, trials=args.trials
                )
                worst = max(worst, result.max_latency_from_receipt_ms)
                violations += result.violations
            gap = adversarial.max_latency_from_receipt_ms - closed
            sys.stdout.write(
                f"{task_id}\n"
                f"  closed form WCDL        : {closed:.9f} ms\n"
                f"  adversarial simulation  : "
                f"{adversarial.max_latency_from_receipt_ms:.9f} ms\n"
                f"  difference              : {gap:+.3e} ms\n"
                f"  stochastic max over "
                f"{args.seeds * args.trials} trials: {worst:.9f} ms\n"
                f"  bound violations        : {violations}\n"
            )
            if violations or abs(gap) > 1e-6:
                failures += 1
        if failures:
            sys.stderr.write(f"error: {failures} task(s) disagreed with the closed form\n")
            return 1
        return 0

    report = run_checks(spec, tol_ms=args.tol_ms)
    if args.format == "json":
        sys.stdout.write(format_json(report))
    elif args.format == "markdown":
        sys.stdout.write(format_markdown(report))
    else:
        sys.stdout.write(format_report(report))
    if args.strict and report.findings:
        return 1
    return 1 if any(f.severity is Severity.ERROR for f in report.findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
