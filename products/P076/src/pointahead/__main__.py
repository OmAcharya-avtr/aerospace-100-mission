"""Command-line interface: ``python -m pointahead``.

Subcommands
-----------
``info``
    The package version, the measured environment and the named scenarios.
``budget CASE``
    The full attributed budget for a YAML case, with the analytic lines checked
    against a numerical propagation.
``compare CASE``
    The closed-form baseline scored against the iterated light-time reference.
``lighttime CASE``
    The two light-time solutions with their iteration histories and residuals.
``scenario NAME``
    The same as ``budget`` but for one of the built-in named geometries, so a
    reader can get a number without writing a file first.

Every subcommand writes to stdout and returns 0 on success. A bad case file
exits 2 with the offending key named, which is the convention for a usage
error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .angle import decompose_point_ahead, light_time_pair, score_closed_form
from .budget import BudgetInputs, build_budget, verify_sensitivities
from .cases import Case, load_case
from .report import (
    format_budget,
    format_decomposition,
    format_light_time,
    format_score,
    format_sensitivity_checks,
)
from .scenario import DEFAULT_BEAM, NAMED_SCENARIOS


def _case_from_scenario(name: str) -> Case:
    try:
        builder = NAMED_SCENARIOS[name]
    except KeyError:
        raise SystemExit(
            f"unknown scenario {name!r}; available: {sorted(NAMED_SCENARIOS)}"
        ) from None
    link = builder()
    inputs = BudgetInputs(
        sigma_velocity_per_axis_mps=0.05,
        ephemeris_age_s=30.0,
        light_time_iterations=1,
        beam=DEFAULT_BEAM,
        aberration_route="momentum_boost",
        angle_method="closed_form",
        loss_budget_db=1.0,
    )
    return Case(name=name, link=link, inputs=inputs, source=f"built-in scenario {name}")


def _load(path: str) -> Case:
    try:
        return load_case(Path(path))
    except FileNotFoundError:
        raise SystemExit(f"case file not found: {path}") from None
    except ValueError as exc:
        raise SystemExit(f"invalid case file {path}: {exc}") from None


def _budget_report(case: Case) -> str:
    budget = build_budget(case.link, case.inputs)
    checks = verify_sensitivities(case.link, case.inputs)
    return "\n\n".join(
        (
            f"case: {case.name}  (source: {case.source})",
            format_decomposition(decompose_point_ahead(case.link)),
            format_budget(budget),
            format_sensitivity_checks(checks),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so a test can exercise it without a shell."""
    parser = argparse.ArgumentParser(
        prog="pointahead",
        description=(
            "Point-ahead angle for an optical crosslink with every error term "
            "attributed. Research grade; not for operational pointing decisions."
        ),
    )
    parser.add_argument("--version", action="version", version=f"pointahead {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="version, environment and named scenarios")

    for name, help_text in (
        ("budget", "attributed error budget for a YAML case"),
        ("compare", "closed-form baseline scored against the iterated reference"),
        ("lighttime", "light-time solutions with iteration histories"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("case", help="path to a YAML case file")

    scenario = subparsers.add_parser("scenario", help="attributed budget for a built-in geometry")
    scenario.add_argument("name", choices=sorted(NAMED_SCENARIOS), help="named geometry")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "info":
        import numpy
        import scipy
        import sgp4
        import yaml

        print(f"pointahead {__version__}")
        print(f"python   {sys.version.split()[0]}")
        print(f"numpy    {numpy.__version__}")
        print(f"scipy    {scipy.__version__}")
        print(f"pyyaml   {yaml.__version__}")
        print(f"sgp4     {sgp4.__version__ if hasattr(sgp4, '__version__') else 'installed'}")
        print(f"scenarios {sorted(NAMED_SCENARIOS)}")
        print("research grade; not flight-qualified, not certified, not approved for")
        print("operational aerospace use")
        return 0

    case = (
        _case_from_scenario(arguments.name)
        if arguments.command == "scenario"
        else _load(arguments.case)
    )

    if arguments.command in ("budget", "scenario"):
        print(_budget_report(case))
        return 0
    if arguments.command == "compare":
        print(f"case: {case.name}  (source: {case.source})")
        print()
        print(format_score(score_closed_form(case.link)))
        print()
        print(format_decomposition(decompose_point_ahead(case.link)))
        return 0
    receive, transmit = light_time_pair(case.link)
    print(format_light_time(case.name, receive, transmit))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())
