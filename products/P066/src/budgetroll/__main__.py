"""Command line interface: ``python -m budgetroll``.

Subcommands
-----------
``check``        run the ten checks and print the findings
``rollup``       print the rollup table, or JSON or markdown
``sensitivity``  rank the leaves by their effect on the top-level margin
``compare``      linear against root-sum-square against the exact form
``codes``        list the check codes and what each one means

Exit status
-----------
``0``  the command ran and raised no finding of severity ``error``
``1``  the command ran and raised at least one ``error`` finding
``2``  the command could not run: a malformed declaration, a missing file,
       or an invalid argument
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .aggregate import linear_rollup_uncertainty, rollup, rss_rollup_uncertainty
from .checks import CODES, has_blocking, run_checks, severity_counts
from .errors import BudgetRollError
from .parse import load
from .report import (
    SCOPE,
    findings_text,
    json_report,
    markdown_report,
    rollup_table,
    sensitivity_table,
    text_report,
)
from .sensitivity import sensitivity

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``python -m budgetroll``."""
    parser = argparse.ArgumentParser(
        prog="python -m budgetroll",
        description=(
            "Hierarchical budget rollup with explicit linear and root-sum-square "
            "combination and assumption checking. Educational-grade; not "
            "flight-qualified, not certified, not approved for operational "
            "aerospace use."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("check", "run the ten checks against a declaration"),
        ("rollup", "print the rollup"),
        ("sensitivity", "rank leaves by effect on the top-level margin"),
        ("compare", "compare the combination rules at the root"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("declaration", help="path to a budget declaration in YAML")
        if name in ("rollup",):
            p.add_argument(
                "--format",
                choices=("text", "json", "markdown", "full"),
                default="text",
                help="output format (default: text)",
            )
        if name in ("sensitivity", "rollup"):
            p.add_argument(
                "--step",
                type=float,
                default=0.10,
                help="relative perturbation for the margin elasticity (default: 0.10)",
            )

    sub.add_parser("codes", help="list the check codes")
    return parser


def _cmd_codes() -> int:
    print("budgetroll check codes")
    print(SCOPE)
    print()
    for code, description in sorted(CODES.items()):
        print(f"  {code}  {description}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "codes":
        return _cmd_codes()

    try:
        budget = load(args.declaration)
    except BudgetRollError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        if args.command == "check":
            findings = run_checks(budget)
            counts = severity_counts(findings)
            print(SCOPE)
            print()
            print(findings_text(findings))
            print()
            print(
                f"{counts['error']} error, {counts['warning']} warning, "
                f"{counts['info']} info"
            )
            return EXIT_FINDINGS if has_blocking(findings) else EXIT_OK

        if args.command == "rollup":
            if args.format == "json":
                print(json_report(budget, step=args.step), end="")
            elif args.format == "markdown":
                print(markdown_report(budget, step=args.step), end="")
            elif args.format == "full":
                print(text_report(budget, step=args.step))
            else:
                print(SCOPE)
                print()
                print(rollup_table(rollup(budget)))
            findings = run_checks(budget)
            return EXIT_FINDINGS if has_blocking(findings) else EXIT_OK

        if args.command == "sensitivity":
            print(SCOPE)
            print()
            print(sensitivity_table(sensitivity(budget, step=args.step), budget.unit))
            findings = run_checks(budget)
            return EXIT_FINDINGS if has_blocking(findings) else EXIT_OK

        if args.command == "compare":
            roll = rollup(budget)
            print(SCOPE)
            print()
            print(f"Combination of the {len(roll.leaf_results())} leaf uncertainties "
                  f"of {budget.id}, in {budget.unit}:")
            print(f"  linear (fully correlated upper bound)  "
                  f"{linear_rollup_uncertainty(budget):.6f}")
            print(f"  root-sum-square (independent)          "
                  f"{rss_rollup_uncertainty(budget):.6f}")
            print(f"  declared hierarchy at the root         {roll.root.uncertainty:.6f}")
            print(f"  exact form with declared correlations  "
                  f"{roll.root.exact_uncertainty:.6f}")
            findings = run_checks(budget, roll)
            return EXIT_FINDINGS if has_blocking(findings) else EXIT_OK
    except BudgetRollError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    parser.error(f"unknown command {args.command!r}")
    return EXIT_ERROR  # pragma: no cover - argparse exits first


if __name__ == "__main__":
    raise SystemExit(main())
