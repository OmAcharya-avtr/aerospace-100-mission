"""Command line interface: ``python -m hazardtrace``.

Exit codes
----------
0
    The command succeeded and, for ``check``, no blocking finding was raised.
1
    ``check`` raised at least one blocking finding.
2
    The command could not run: unreadable or malformed input.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .checks import Severity, run_checks
from .crosslinks import DEFAULT_REQUIREMENT_PATTERN, load_junit, load_requirements
from .diagram import render_control_structure_png, render_mermaid
from .errors import HazardTraceError
from .parse import load_analysis
from .report import (
    format_codes,
    format_coverage_table,
    format_grid,
    format_json,
    format_markdown,
    format_report,
)

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_CANNOT_RUN = 2


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so tests can exercise it directly."""
    parser = argparse.ArgumentParser(
        prog="python -m hazardtrace",
        description=(
            "Check a declared STPA hazard analysis for completeness. "
            "Research-grade; a clean run discharges no safety obligation."
        ),
    )
    parser.add_argument("--version", action="version", version=f"hazardtrace {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="run every check and print a report")
    check.add_argument("analysis", help="path to the STPA YAML document")
    check.add_argument(
        "-r", "--requirements", default=None,
        help="markdown requirements document, enabling HT012",
    )
    check.add_argument(
        "-t", "--tests", default=None, help="junit XML report, enabling HT013 and HT014"
    )
    check.add_argument(
        "--requirement-pattern", default=DEFAULT_REQUIREMENT_PATTERN,
        help="regular expression matching a requirement identifier",
    )
    check.add_argument(
        "-f", "--format", choices=("text", "json", "markdown"), default="text",
        help="output format",
    )
    check.add_argument(
        "--strict", action="store_true",
        help="treat warnings as blocking as well as errors",
    )

    grid = sub.add_parser("grid", help="print the UCA enumeration grid only")
    grid.add_argument("analysis")

    coverage = sub.add_parser("coverage", help="print the coverage figures only")
    coverage.add_argument("analysis")
    coverage.add_argument("-r", "--requirements", default=None)
    coverage.add_argument("-t", "--tests", default=None)
    coverage.add_argument("--requirement-pattern", default=DEFAULT_REQUIREMENT_PATTERN)

    diagram = sub.add_parser("diagram", help="render the control structure")
    diagram.add_argument("analysis")
    diagram.add_argument(
        "--mermaid", default=None, help="write Mermaid source to this path (or - for stdout)"
    )
    diagram.add_argument("--png", default=None, help="write a PNG to this path")

    sub.add_parser("codes", help="list every check code")
    return parser


def _load(args: argparse.Namespace):
    analysis = load_analysis(args.analysis)
    requirements = None
    tests = None
    if getattr(args, "requirements", None):
        requirements = load_requirements(args.requirements, args.requirement_pattern)
    if getattr(args, "tests", None):
        tests = load_junit(args.tests)
    return analysis, requirements, tests


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    out = sys.stdout

    if args.command == "codes":
        out.write(format_codes())
        return EXIT_OK

    try:
        analysis, requirements, tests = _load(args)
    except HazardTraceError as exc:
        sys.stderr.write(f"hazardtrace: {exc}\n")
        return EXIT_CANNOT_RUN

    if args.command == "diagram":
        if not args.mermaid and not args.png:
            sys.stderr.write("hazardtrace: diagram needs --mermaid or --png\n")
            return EXIT_CANNOT_RUN
        report = run_checks(analysis)
        if args.mermaid:
            text = render_mermaid(analysis, report=report)
            if args.mermaid == "-":
                out.write(text)
            else:
                with open(args.mermaid, "w", encoding="utf-8") as handle:
                    handle.write(text)
                out.write(f"wrote {args.mermaid}\n")
        if args.png:
            written = render_control_structure_png(report, args.png)
            out.write(f"wrote {written}\n")
        return EXIT_OK

    report = run_checks(analysis, requirements=requirements, tests=tests)

    if args.command == "grid":
        out.write(format_grid(report))
        return EXIT_OK
    if args.command == "coverage":
        out.write(format_coverage_table(report.coverage))
        return EXIT_OK

    if args.strict:
        report = type(report)(
            analysis=report.analysis,
            grid=report.grid,
            findings=report.findings,
            coverage=report.coverage,
            requirements_source=report.requirements_source,
            tests_source=report.tests_source,
            blocking_severities=frozenset({Severity.ERROR, Severity.WARNING}),
        )
    if args.format == "json":
        out.write(format_json(report))
    elif args.format == "markdown":
        out.write(format_markdown(report))
    else:
        out.write(format_report(report))
    return EXIT_FINDINGS if report.blocking else EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess tests
    raise SystemExit(main())
