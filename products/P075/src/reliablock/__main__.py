"""Command-line interface: ``python -m reliablock``.

Subcommands, each of which prints a report assembled by
:mod:`reliablock.report` so that the CLI, the examples and the validation
scripts cannot disagree about a number:

``info``
    Package version, the built-in cases and missions, and the statement that
    there is no learned component.
``show CASE``
    The structural summary and minimal cut sets of a built-in case.
``quantify CASE``
    Closed-form system reliability over a mission, with the per-node tree.
``compare CASE``
    The headline: reliability with and without the declared common-cause beta.
``importance CASE``
    Birnbaum, criticality and Fussell-Vesely for every basic event.
``mission CASE``
    The per-phase effective-age breakdown.
``simulate CASE``
    Monte Carlo against the closed form, with the interval and the budget.
``bridge``
    The five-edge bridge network computed four independent ways.

A ``--diagram PATH`` option replaces the built-in case with a declared YAML
diagram, and ``--mission PATH`` with a declared YAML mission profile.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .bridge import (
    bridge_diagram,
    bridge_reliability_equal,
    bridge_reliability_inclusion_exclusion,
    bridge_reliability_pivotal,
)
from .cases import CASES, MISSIONS, case, mission
from .diagram import Diagram, load_diagram
from .mission import MissionProfile, load_mission
from .montecarlo import simulate, trials_for_half_width
from .quantify import quantify
from .report import (
    common_cause_comparison,
    cut_set_report,
    diagram_report,
    importance_report,
    mission_report,
    quantity_tree,
)
from .structure import enumerate_states, minimal_cut_sets


def _resolve_diagram(args: argparse.Namespace) -> Diagram:
    """The declared diagram named by ``--diagram`` or by the positional case."""
    if getattr(args, "diagram", None):
        return load_diagram(Path(args.diagram))
    return case(args.case)


def _resolve_mission(args: argparse.Namespace) -> MissionProfile:
    """The mission profile named by ``--mission``, ``--hours`` or the default."""
    if getattr(args, "mission_file", None):
        return load_mission(Path(args.mission_file))
    if getattr(args, "hours", None) is not None:
        return MissionProfile.from_hours(args.hours)
    return mission(args.mission)


def _add_common(parser: argparse.ArgumentParser, *, with_mission: bool = True) -> None:
    """Add the diagram and mission selectors to a subparser."""
    parser.add_argument(
        "case",
        nargs="?",
        default="smallsat_bus",
        choices=sorted(CASES),
        help="built-in declared diagram (default: smallsat_bus)",
    )
    parser.add_argument("--diagram", help="path to a declared YAML diagram, replacing the case")
    if with_mission:
        parser.add_argument(
            "--mission",
            default="three_year",
            choices=sorted(MISSIONS),
            help="built-in mission profile (default: three_year)",
        )
        parser.add_argument(
            "--mission-file",
            dest="mission_file",
            help="path to a declared YAML mission profile",
        )
        parser.add_argument("--hours", type=float, help="single-phase mission duration in hours")


def build_parser() -> argparse.ArgumentParser:
    """The argument parser for ``python -m reliablock``."""
    parser = argparse.ArgumentParser(
        prog="reliablock",
        description=(
            "Redundancy arithmetic that is explicit about the assumption doing "
            "the work. Research-grade; not flight-qualified, not certified, not "
            "approved for operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"reliablock {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="package version, built-in cases, and what is not here")

    show = subparsers.add_parser("show", help="structure and minimal cut sets of a diagram")
    _add_common(show, with_mission=False)
    show.add_argument(
        "--enumerate",
        action="store_true",
        help="cross-check the cut sets against exhaustive state enumeration",
    )

    quantify_parser = subparsers.add_parser("quantify", help="closed-form system reliability")
    _add_common(quantify_parser)

    compare = subparsers.add_parser(
        "compare", help="reliability with and without the declared common cause"
    )
    _add_common(compare)

    importance = subparsers.add_parser("importance", help="importance measures per basic event")
    _add_common(importance)

    mission_parser = subparsers.add_parser("mission", help="per-phase effective-age breakdown")
    _add_common(mission_parser)

    simulate_parser = subparsers.add_parser(
        "simulate", help="Monte Carlo, scored against the closed form"
    )
    _add_common(simulate_parser)
    simulate_parser.add_argument("--trials", type=int, help="trials; default sizes from --target")
    simulate_parser.add_argument(
        "--target",
        type=float,
        default=0.02,
        help="target interval half-width as a fraction of the system unreliability",
    )
    simulate_parser.add_argument("--seed", type=int, default=20261025, help="RNG seed")

    subparsers.add_parser("bridge", help="the five-edge bridge network, four independent ways")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    lines: list[str] = []

    if args.command == "info":
        lines = [
            f"reliablock {__version__}",
            "",
            "Redundancy arithmetic that is explicit about the assumption doing the work.",
            "",
            "There is no learned component in this package, deliberately. For a declared",
            "diagram with declared distributions the reliability has a correct answer, so a",
            "fitted model could only approximate the arithmetic, and there is nothing in the",
            "problem to learn. Monte Carlo is scored against the closed forms, never the",
            "other way round.",
            "",
            "For distribution fitting, accelerated life testing or repairable-system",
            "analysis, use the 'reliability' package instead. This one takes a declared",
            "diagram as a checkable artifact and reports what rests on the independence",
            "assumption.",
            "",
            "built-in diagrams:",
        ]
        for name in sorted(CASES):
            diagram = CASES[name]
            lines.append(f"  {name:<20s}{len(diagram.blocks()):>3d} blocks  {diagram.name}")
        lines.append("")
        lines.append("built-in mission profiles:")
        for name in sorted(MISSIONS):
            profile = MISSIONS[name]
            lines.append(
                f"  {name:<20s}{profile.total_hours:>12.6g} h  "
                f"{len(profile.phases)} phase(s)"
            )
        lines.append("")
        lines.append(
            "Research-grade. Not flight-qualified, not certified, not approved for"
        )
        lines.append("operational aerospace use. A reliability number is not a safety case.")

    elif args.command == "show":
        diagram = _resolve_diagram(args)
        lines = diagram_report(diagram) + [""] + cut_set_report(diagram)
        if args.enumerate:
            structural = minimal_cut_sets(diagram)
            exhaustive = enumerate_states(diagram)
            agree = set(structural) == set(exhaustive["minimal_cut_sets"])
            lines += [
                "",
                f"exhaustive enumeration over {exhaustive['states']} states:",
                f"  failed states                : {exhaustive['failed_states']}",
                f"  minimal cut sets, structural : {len(structural)}",
                f"  minimal cut sets, exhaustive : {exhaustive['n_minimal']}",
                f"  identical                    : {agree}",
            ]

    elif args.command == "quantify":
        diagram = _resolve_diagram(args)
        profile = _resolve_mission(args)
        result = quantify(diagram, profile)
        lines = diagram_report(diagram) + [
            "",
            f"mission            : {profile.name}, {profile.total_hours:.6g} h",
            f"system reliability : {result.reliability:.12f}",
            f"system unreliability: {result.unreliability:.12e}",
            f"R + Q - 1          : {result.complement_residual:.3e}",
            "",
            "per-node breakdown:",
        ] + quantity_tree(result)

    elif args.command == "compare":
        diagram = _resolve_diagram(args)
        lines = common_cause_comparison(diagram, _resolve_mission(args))

    elif args.command == "importance":
        diagram = _resolve_diagram(args)
        try:
            lines = importance_report(diagram, _resolve_mission(args))
        except ValueError as error:
            lines = [f"importance measures refused for {diagram.name!r}:", "", f"  {error}"]

    elif args.command == "mission":
        diagram = _resolve_diagram(args)
        lines = mission_report(diagram, _resolve_mission(args))

    elif args.command == "simulate":
        diagram = _resolve_diagram(args)
        profile = _resolve_mission(args)
        closed = quantify(diagram, profile)
        trials = args.trials
        if trials is None:
            pilot = simulate(diagram, profile, 20_000, seed=args.seed + 1)
            reference = max(pilot.unreliability, closed.unreliability, 1e-9)
            trials = min(
                trials_for_half_width(reference, args.target * reference), 8_000_000
            )
        result = simulate(diagram, profile, trials, seed=args.seed)
        lines = [
            f"diagram            : {diagram.name}",
            f"mission            : {profile.name}, {profile.total_hours:.6g} h",
            "",
            f"closed form R      : {closed.reliability:.12f}",
            f"closed form Q      : {closed.unreliability:.6e}",
            f"simulated R        : {result.reliability:.12f}",
            f"trials             : {result.trials} (seed {result.seed})",
            f"Wilson {result.confidence:.0%} interval : "
            f"[{result.lower:.12f}, {result.upper:.12f}]",
            f"interval half-width: {result.half_width:.6e}",
            f"closed form inside : {result.brackets(closed.reliability)}",
            f"standard errors    : {result.standard_errors_from(closed.reliability):.3f}",
            "",
            "The closed form is the reference. The simulation is scored against it.",
            "",
            "| quantity | value | stability |",
            "|---|---|---|",
            f"| wall clock | {result.elapsed_s:.3f} s | UNSTABLE, shared hardware |",
        ]

    else:  # bridge
        p = 0.9
        declared = bridge_diagram(p)
        package = quantify(declared, 1.0)
        exhaustive = enumerate_states(declared)
        lines = [
            "five-edge bridge network, equal edge reliability p = 0.9",
            "",
            "The closed form R = 2p^5 - 5p^4 + 2p^3 + 2p^2 is DERIVED in",
            "reliablock.bridge by pivotal decomposition on the bridge edge. No page",
            "number and no published figure is quoted for it. It is verified here",
            "against three further independent routes.",
            "",
            f"  {'route':<44s}{'reliability':>22s}",
            f"  {'-' * 66}",
            f"  {'derived closed form 2p^5-5p^4+2p^3+2p^2':<44s}"
            f"{bridge_reliability_equal(p):>22.15f}",
            f"  {'pivotal decomposition, non-identical form':<44s}"
            f"{bridge_reliability_pivotal(p, p, p, p, p):>22.15f}",
            f"  {'inclusion-exclusion over 4 minimal paths':<44s}"
            f"{bridge_reliability_inclusion_exclusion({f'e{i}': p for i in range(1, 6)}):>22.15f}",
            f"  {'reliablock.quantify over declared network':<44s}"
            f"{package.reliability:>22.15f}",
            "",
            f"exhaustive enumeration over {exhaustive['states']} states: "
            f"{exhaustive['failed_states']} failed, "
            f"{exhaustive['n_minimal']} minimal cut sets",
            "minimal cut sets: "
            + ", ".join("{" + ", ".join(sorted(c)) + "}" for c in minimal_cut_sets(declared)),
        ]

    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
