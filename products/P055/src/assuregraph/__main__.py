"""Command-line interface: ``python -m assuregraph``.

This is the only module in the package that writes to a stream. Exit codes:

==  ==========================================================================
0   the requested operation succeeded and, for ``check``, the case is
    structurally complete
1   ``check`` only: the case is structurally incomplete. This is the documented
    nonzero exit a continuous-integration job keys on
2   usage error, unreadable file, or a document that is not a well-formed GSN
    case. Distinct from 1 on purpose: a malformed case and an incomplete case
    are different problems
==  ==========================================================================

``mermaid`` and ``summary`` exit 0 for any well-formed case regardless of
findings, because rendering a case with gaps is the point of rendering it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import TextIO

from . import __version__
from .checks import run_checks
from .errors import AssureGraphError
from .mermaid import render_mermaid
from .parse import load_case
from .report import SCOPE_STATEMENT, format_coverage_table, format_report, report_as_dict

EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_USAGE = 2

_EPILOG = (
    "Scope: " + SCOPE_STATEMENT + " Research-grade software: not flight-qualified, "
    "not certified, not approved for operational aerospace use. "
    "Exit codes: 0 complete, 1 incomplete (check only), 2 malformed input or usage error."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Separated out so tests can introspect it."""
    parser = argparse.ArgumentParser(
        prog="python -m assuregraph",
        description=(
            "Check the structure and evidence freshness of a Goal Structuring Notation "
            "assurance case, and render it as Mermaid."
        ),
        epilog=_EPILOG,
    )
    parser.add_argument("--version", action="version", version=f"assuregraph {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    check = sub.add_parser(
        "check",
        help="run all six checks; exit 1 when the case is structurally incomplete",
        epilog=_EPILOG,
    )
    check.add_argument("case", help="path to the YAML assurance case")
    check.add_argument(
        "--strict",
        action="store_true",
        help="count warnings as blocking, so an artifact with no recorded sha256 fails",
    )
    group = check.add_mutually_exclusive_group()
    group.add_argument("--json", action="store_true", help="emit the report as JSON")
    group.add_argument("--markdown", action="store_true", help="emit the report as Markdown")

    mermaid = sub.add_parser(
        "mermaid", help="render the case as a Mermaid flowchart", epilog=_EPILOG
    )
    mermaid.add_argument("case", help="path to the YAML assurance case")
    mermaid.add_argument(
        "--direction",
        default="TD",
        choices=["TD", "TB", "BT", "LR", "RL"],
        help="Mermaid flowchart direction (default: TD)",
    )
    mermaid.add_argument(
        "--max-label-chars",
        type=int,
        default=70,
        metavar="N",
        help="truncate statements longer than N characters; 0 disables (default: 70)",
    )
    mermaid.add_argument(
        "--fence", action="store_true", help="wrap the output in a ```mermaid fence"
    )
    mermaid.add_argument(
        "--highlight",
        action="store_true",
        help="style nodes named by an error finding with the gsnFinding class",
    )
    mermaid.add_argument(
        "-o", "--output", metavar="PATH", help="write to PATH instead of standard output"
    )

    summary = sub.add_parser(
        "summary", help="print the coverage summary table only", epilog=_EPILOG
    )
    summary.add_argument("case", help="path to the YAML assurance case")
    summary.add_argument("--markdown", action="store_true", help="emit the table as Markdown")

    return parser


def _run_check(args: argparse.Namespace, out: TextIO) -> int:
    case = load_case(args.case)
    report = run_checks(case, strict=args.strict)
    if args.json:
        out.write(json.dumps(report_as_dict(report), indent=2, sort_keys=False) + "\n")
    else:
        out.write(format_report(report, markdown=args.markdown))
    return report.exit_code


def _run_mermaid(args: argparse.Namespace, out: TextIO) -> int:
    case = load_case(args.case)
    report = run_checks(case) if args.highlight else None
    text = render_mermaid(
        case,
        report=report,
        direction=args.direction,
        max_label_chars=args.max_label_chars,
        fence=args.fence,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        out.write(text)
    return EXIT_OK


def _run_summary(args: argparse.Namespace, out: TextIO) -> int:
    case = load_case(args.case)
    report = run_checks(case)
    out.write(format_coverage_table(report.coverage, markdown=args.markdown))
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    """Entry point. Returns the process exit code rather than calling ``sys.exit``.

    Args:
        argv: Argument vector without the program name. ``None`` uses
            ``sys.argv[1:]``.
        out: Stream for the report. ``None`` uses ``sys.stdout``.
        err: Stream for errors. ``None`` uses ``sys.stderr``.

    Returns:
        0, 1 or 2; see the module docstring.
    """
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {"check": _run_check, "mermaid": _run_mermaid, "summary": _run_summary}
    try:
        return handlers[args.command](args, out)
    except AssureGraphError as exc:
        err.write(f"assuregraph: {exc}\n")
        return EXIT_USAGE
    except OSError as exc:
        err.write(f"assuregraph: {exc}\n")
        return EXIT_USAGE
    except ValueError as exc:
        err.write(f"assuregraph: {exc}\n")
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
