"""Command-line interface: ``python -m downlinkpick``.

Subcommands
-----------
``select``
    Select from a case file with one selector and print the full report:
    the plan, the selection, the left-behind report, the marginal value of
    contact time and the invariant check.
``compare``
    Run greedy and exact on the same case and print the optimality gap, the
    bounds, and the first product the two selectors disagree about.
``marginal``
    Price extra contact time on each window under both selectors.
``stability``
    Measure selection stability under perturbed value estimates.
``lp``
    Write the PuLP LP file for a case, which is the portable statement of the
    MILP for anyone who has a solver this environment does not.
``info``
    Print the package version and the solver backends actually available.

This module is the only place in the package that writes to a stream.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .cases import load_case
from .exact import exact_select, lp_relaxation_bound, pulp_available, write_lp
from .greedy import dantzig_bound, density_order, greedy_select
from .marginal import left_behind_report, marginal_value_of_contact
from .report import format_full_report, format_marginals
from .selection import SelectionProblem
from .stability import stability_curve

SELECTORS = {"greedy": greedy_select, "exact": exact_select}


def _problem(path: str) -> SelectionProblem:
    case = load_case(path)
    return SelectionProblem.from_declared(case.inventory, case.plan)


def _cmd_select(args: argparse.Namespace, stream) -> int:
    problem = _problem(args.case)
    selector = SELECTORS[args.method]
    selection = selector(problem)
    marginals = marginal_value_of_contact(
        problem, selector=selector, delta_s=args.delta_s, base=selection
    )
    rows = left_behind_report(problem, selection)
    print(format_full_report(problem, selection, marginals, rows), file=stream)
    return 0


def _cmd_compare(args: argparse.Namespace, stream) -> int:
    problem = _problem(args.case)
    greedy = greedy_select(problem)
    exact = exact_select(problem)
    gap = (
        (exact.total_value - greedy.total_value) / exact.total_value
        if exact.total_value > 0.0
        else 0.0
    )
    print(f"case                  : {args.case}", file=stream)
    print(f"products / windows    : {len(problem.inventory)} / {len(problem.plan)}", file=stream)
    print(f"greedy value          : {greedy.total_value:.6f}", file=stream)
    print(f"exact value           : {exact.total_value:.6f}", file=stream)
    print(f"exact status          : {exact.status}", file=stream)
    print(f"exact method          : {exact.method}", file=stream)
    print(f"relative gap          : {gap:.6e}", file=stream)
    print(f"LP relaxation bound   : {lp_relaxation_bound(problem):.6f}", file=stream)
    print(f"Dantzig bound         : {dantzig_bound(problem):.6f}", file=stream)
    only_exact = sorted(exact.selected_set - greedy.selected_set)
    only_greedy = sorted(greedy.selected_set - exact.selected_set)
    print(f"in exact not greedy   : {', '.join(only_exact) or '-'}", file=stream)
    print(f"in greedy not exact   : {', '.join(only_greedy) or '-'}", file=stream)
    print(f"density order (head)  : {', '.join(density_order(problem)[:6])}", file=stream)
    return 0


def _cmd_marginal(args: argparse.Namespace, stream) -> int:
    problem = _problem(args.case)
    for label, selector in SELECTORS.items():
        print(f"selector: {label}", file=stream)
        print(
            format_marginals(
                marginal_value_of_contact(problem, selector=selector, delta_s=args.delta_s)
            ),
            file=stream,
        )
        print("", file=stream)
    return 0


def _cmd_stability(args: argparse.Namespace, stream) -> int:
    problem = _problem(args.case)
    sigmas = tuple(float(value) for value in args.sigmas)
    print(
        f"{'sigma':>8s} {'mean J':>8s} {'p05 J':>8s} {'min J':>8s} "
        f"{'mean kept':>10s} {'p05 kept':>9s} {'flips':>7s}",
        file=stream,
    )
    for point in stability_curve(
        problem, sigmas, selector=SELECTORS[args.method], n_trials=args.trials, seed=args.seed
    ):
        print(
            f"{point.sigma:8.3f} {point.mean_jaccard:8.4f} {point.p05_jaccard:8.4f} "
            f"{point.min_jaccard:8.4f} {point.mean_value_retained:10.5f} "
            f"{point.p05_value_retained:9.5f} {point.mean_flips:7.2f}",
            file=stream,
        )
    return 0


def _cmd_lp(args: argparse.Namespace, stream) -> int:
    problem = _problem(args.case)
    written = write_lp(problem, args.output)
    print(f"wrote {written}", file=stream)
    return 0


def _cmd_info(_args: argparse.Namespace, stream) -> int:
    import numpy
    import pulp
    import scipy
    import sklearn

    print(f"downlinkpick     : {__version__}", file=stream)
    print(f"numpy            : {numpy.__version__}", file=stream)
    print(f"scipy            : {scipy.__version__}", file=stream)
    print(f"scikit-learn     : {sklearn.__version__}", file=stream)
    print(f"pulp             : {pulp.__version__}", file=stream)
    print(f"pulp solvers     : {pulp.listSolvers(onlyAvailable=True) or 'none'}", file=stream)
    print(f"pulp usable      : {pulp_available()}", file=stream)
    print("exact backend    : scipy.optimize.milp (HiGHS) unless pulp is usable", file=stream)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so the tests can exercise it directly."""
    parser = argparse.ArgumentParser(
        prog="downlinkpick",
        description=(
            "Choose what to downlink when the contact is shorter than the queue, and "
            "report what the choice cost. Research-grade; not flight-qualified, not "
            "certified, not approved for operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"downlinkpick {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select", help="select from a case file")
    select.add_argument("case", help="path to a YAML case file")
    select.add_argument(
        "--method", choices=sorted(SELECTORS), default="exact", help="selector to use"
    )
    select.add_argument(
        "--delta-s",
        dest="delta_s",
        type=float,
        default=1.0,
        help="finite-difference step for the marginal value, in seconds",
    )
    select.set_defaults(handler=_cmd_select)

    compare = subparsers.add_parser("compare", help="greedy against exact on one case")
    compare.add_argument("case", help="path to a YAML case file")
    compare.set_defaults(handler=_cmd_compare)

    marginal = subparsers.add_parser("marginal", help="price extra contact time")
    marginal.add_argument("case", help="path to a YAML case file")
    marginal.add_argument(
        "--delta-s", dest="delta_s", type=float, default=1.0, help="step in seconds"
    )
    marginal.set_defaults(handler=_cmd_marginal)

    stability = subparsers.add_parser("stability", help="stability under perturbed values")
    stability.add_argument("case", help="path to a YAML case file")
    stability.add_argument(
        "--method", choices=sorted(SELECTORS), default="exact", help="selector to use"
    )
    stability.add_argument(
        "--sigmas",
        nargs="+",
        default=["0.0", "0.02", "0.05", "0.1", "0.2"],
        help="relative value-noise levels",
    )
    stability.add_argument("--trials", type=int, default=20, help="draws per noise level")
    stability.add_argument("--seed", type=int, default=20261021, help="perturbation seed")
    stability.set_defaults(handler=_cmd_stability)

    lp = subparsers.add_parser("lp", help="write the PuLP LP file for a case")
    lp.add_argument("case", help="path to a YAML case file")
    lp.add_argument("output", help="path of the LP file to write")
    lp.set_defaults(handler=_cmd_lp)

    info = subparsers.add_parser("info", help="versions and available solver backends")
    info.set_defaults(handler=_cmd_info)
    return parser


def main(argv: Sequence[str] | None = None, stream=None) -> int:
    """Entry point. Returns a process exit status.

    Parameters
    ----------
    argv:
        Argument list excluding the program name. ``None`` uses ``sys.argv``.
    stream:
        Output stream. ``None`` uses ``sys.stdout``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    target = sys.stdout if stream is None else stream
    try:
        return int(args.handler(args, target))
    except BrokenPipeError:
        # The reader closed the pipe, as `| head` does. Exit quietly: a tool
        # that prints a traceback when piped into head is a tool nobody pipes
        # into head twice. When this function owns the process's stdout, that
        # descriptor is also redirected to the null device so the interpreter's
        # own flush at exit does not raise again; when the caller supplied a
        # stream, the caller owns it and nothing is touched.
        if stream is None:
            import os

            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            os.close(devnull)
        return 0
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        print(f"downlinkpick: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
