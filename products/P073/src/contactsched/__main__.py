"""Command line interface: ``python -m contactsched <subcommand>``.

This is the only module in the package that writes to a stream. Every
subcommand works on a seeded synthetic instance from
:func:`contactsched.dataset.generate_instance`, so every command in README.md is
reproducible by anyone who clones the repository, with no data files and no
network.

Subcommands
-----------
``info``        versions and sizes, for a bug report.
``backends``    which solver backends this interpreter can actually run.
``windows``     the visibility windows of one instance.
``schedule``    one method's schedule.
``fairness``    the contention and fairness accounting for one method.
``contention``  the antenna-time contention map.
``compare``     every method side by side, with the price of the fairness
                constraints.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .benchmark import compare_methods, fairness_cost, optimality_gap, run_method
from .contention import contention_map
from .dataset import generate_instance
from .report import (
    format_backends,
    format_contention,
    format_fairness,
    format_schedule,
    format_windows,
)

METHODS = ("greedy", "greedy-demand", "exact", "exact-fair")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser. Separate from :func:`main` so it is testable."""
    parser = argparse.ArgumentParser(
        prog="contactsched",
        description=(
            "Ground-station contact scheduling with the contention and fairness cost "
            "measured. Research grade; not flight-qualified and not certified."
        ),
    )
    parser.add_argument("--version", action="version", version=f"contactsched {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="versions and instance sizes")
    subparsers.add_parser("backends", help="solver backends detected in this interpreter")

    def add_instance_arguments(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--seed", type=int, default=7, help="instance seed (default 7)")
        sub.add_argument(
            "--horizon-h", type=float, default=12.0, help="horizon in hours (default 12)"
        )

    windows = subparsers.add_parser("windows", help="visibility windows of one instance")
    add_instance_arguments(windows)
    windows.add_argument("--limit", type=int, default=25, help="rows to print (default 25)")

    schedule = subparsers.add_parser("schedule", help="one method's schedule")
    add_instance_arguments(schedule)
    schedule.add_argument("--method", choices=METHODS, default="exact")

    fairness = subparsers.add_parser("fairness", help="contention and fairness accounting")
    add_instance_arguments(fairness)
    fairness.add_argument("--method", choices=METHODS, default="exact")

    contention = subparsers.add_parser("contention", help="antenna-time contention map")
    add_instance_arguments(contention)
    contention.add_argument("--bin-s", type=float, default=1800.0, help="bin width in seconds")
    contention.add_argument("--limit", type=int, default=15, help="rows to print (default 15)")

    compare = subparsers.add_parser("compare", help="every method, with the fairness price")
    add_instance_arguments(compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit status.

    Returns 0 on success and 2 on an input the parser accepted but the package
    cannot act on, for example an instance whose candidate set is empty.
    """
    args = build_parser().parse_args(argv)

    if args.command == "backends":
        print(format_backends())
        return 0

    if args.command == "info":
        import numpy
        import scipy
        import sgp4
        import sklearn

        instance = generate_instance(7, horizon_s=43200.0)
        print(f"contactsched     {__version__}")
        print(f"python           {sys.version.split()[0]}")
        print(f"numpy            {numpy.__version__}")
        print(f"scipy            {scipy.__version__}")
        print(f"scikit-learn     {sklearn.__version__}")
        print(f"sgp4             {sgp4.__version__}")
        print("")
        print("default instance (seed 7, 12 h horizon):")
        print(f"  antennas       {len(instance.stations)}")
        print(f"  spacecraft     {len(instance.fleet)}")
        print(f"  requests       {len(instance.problem.requests)}")
        print(f"  windows        {len(instance.problem.windows)}")
        print(f"  candidates     {len(instance.problem.candidates)}")
        print("")
        print("Research grade. Not flight-qualified, not certified, not approved for")
        print("operational aerospace use.")
        return 0

    instance = generate_instance(int(args.seed), horizon_s=float(args.horizon_h) * 3600.0)
    problem = instance.problem
    if not problem.candidates:
        print(
            f"instance seed {args.seed} has no candidate placements over a "
            f"{args.horizon_h} h horizon; nothing to schedule",
            file=sys.stderr,
        )
        return 2

    if args.command == "windows":
        print(format_windows(list(problem.windows), limit=int(args.limit)))
        return 0

    if args.command == "contention":
        cells = contention_map(problem, instance.scenario.duration_s, float(args.bin_s))
        print(format_contention(cells, limit=int(args.limit)))
        return 0

    if args.command == "schedule":
        result = run_method(problem, args.method)
        print(format_schedule(problem, result.assignment))
        if result.violations:
            for violation in result.violations:
                print(f"VIOLATION: {violation}", file=sys.stderr)
            return 1
        return 0

    if args.command == "fairness":
        result = run_method(problem, args.method)
        print(format_fairness(result.report))
        return 0

    if args.command == "compare":
        results = compare_methods(problem)
        print(
            f"{'method':<24} {'status':<11} {'objective':>10} {'contacts':>9} "
            f"{'starved':>8} {'jain':>8} {'maxmin':>8} {'inv':>5}"
        )
        print("-" * 90)
        for name in METHODS:
            result = results[name]
            print(
                f"{result.assignment.method:<24} {result.assignment.status:<11} "
                f"{result.assignment.objective:10.6f} "
                f"{result.assignment.contact_count:9d} "
                f"{len(result.report.starved):8d} {result.report.jain:8.4f} "
                f"{result.report.max_min:8.4f} {result.report.inversion_count:5d}"
            )
        gap = optimality_gap(results["greedy"].assignment, results["exact"].assignment)
        print("")
        print(f"greedy optimality gap against the exact optimum: {gap:.6f}")
        price = fairness_cost(results["exact"], results["exact-fair"], "weighted_contacts")
        print("")
        print("price of enforcing the declared minimum contact counts:")
        for key in (
            "objective_unconstrained",
            "objective_constrained",
            "objective_loss",
            "objective_loss_fraction",
            "jain_gain",
            "max_min_gain",
            "starved_before",
            "starved_after",
            "seconds_short_before",
            "seconds_short_after",
        ):
            print(f"  {key:<26} {price[key]}")
        if results["exact-fair"].assignment.status == "infeasible":
            print("")
            print("the fairness-constrained program is infeasible on this instance:")
            for note in results["exact-fair"].assignment.notes:
                print(f"  {note}")
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
