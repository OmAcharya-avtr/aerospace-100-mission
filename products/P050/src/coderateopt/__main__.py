"""Command-line interface: ``python -m coderateopt``.

Subcommands
-----------
``solve``
    Solve one instance and print the mix, the goodput and the availability
    actually achieved.
``sensitivity``
    Solve, then report the stability interval of the decision in one
    parameter, with the knife-edge verdict.
``compare``
    Solve, then print the two rules of thumb beside the optimum.
``table``
    Print the illustrative MODCOD table with its availabilities at a stated
    margin.
``fade``
    Print fade quantiles and availabilities for a fade model.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .fade import AvailabilityModel, GammaGammaFade, LognormalFade
from .heuristics import compare_to_optimum
from .milp import MILP_OPTIONS
from .modcod import ModcodSet, illustrative_modcod_table
from .problem import InfeasibleProblem, RateProblem
from .select import select_rate
from .sensitivity import margin_sensitivity, scintillation_sensitivity, target_sensitivity

_SAFETY = (
    "research-grade; not flight-qualified, not certified, "
    "not approved for operational aerospace use"
)


def _build_fade(args: argparse.Namespace) -> AvailabilityModel:
    if args.fade_model == "lognormal":
        return LognormalFade(args.scintillation_index, median_preserving=args.median_preserving)
    return GammaGammaFade.from_scintillation(args.scintillation_index, ratio=args.gg_ratio)


def _load_table(path: str | None) -> ModcodSet:
    """Read a MODCOD table from CSV, or return the illustrative one.

    CSV columns: ``name,rate_bits_per_symbol,threshold_db``, with a header.
    """
    if path is None:
        return illustrative_modcod_table()
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    row["name"],
                    float(row["rate_bits_per_symbol"]),
                    float(row["threshold_db"]),
                )
            )
    if not rows:
        raise ValueError(f"no MODCOD rows found in {path!r}")
    return ModcodSet.from_rows(rows)


def _build_problem(args: argparse.Namespace) -> RateProblem:
    return RateProblem(
        modcods=_load_table(args.modcod_csv),
        fade=_build_fade(args),
        margin_db=args.margin_db,
        availability_target=args.target,
        max_entries=args.max_entries,
        mode=args.mode,
        min_dwell_fraction=args.min_dwell,
    )


def _print_solution(problem: RateProblem, solution) -> None:
    names = problem.modcods.names
    print(f"method                     {solution.method} (canonical={solution.canonical})")
    print(f"mode                       {problem.mode}")
    print(f"availability target        {problem.availability_target:.6f}")
    print(f"expected goodput           {solution.expected_goodput:.6f} bits/symbol")
    print(f"achieved availability      {solution.achieved_availability:.6f}")
    print(f"worst-interval availability {solution.worst_interval_availability:.6f}")
    print("mix:")
    for i in solution.support:
        print(
            f"  {names[i]:<14s} x={solution.time_fractions[i]:.6f}  "
            f"threshold={problem.modcods.thresholds_db[i]:6.2f} dB  "
            f"A={problem.availabilities()[i]:.6f}"
        )
    if solution.canonical and solution.tied_supports:
        tied = "; ".join(
            "+".join(names[i] for i in support) for support in solution.tied_supports
        )
        print(f"tied alternatives          {tied}")


def _cmd_solve(args: argparse.Namespace) -> int:
    problem = _build_problem(args)
    _print_solution(problem, select_rate(problem, method=args.method))
    return 0


def _cmd_sensitivity(args: argparse.Namespace) -> int:
    problem = _build_problem(args)
    _print_solution(problem, select_rate(problem))
    if args.parameter == "scintillation_index":
        report = scintillation_sensitivity(problem, knife_edge_at=args.knife_edge_at)
    elif args.parameter == "margin_db":
        report = margin_sensitivity(problem, knife_edge_at=args.knife_edge_at)
    else:
        report = target_sensitivity(problem, knife_edge_at=args.knife_edge_at)
    print()
    print(f"parameter                  {report.parameter}")
    print(f"nominal                    {report.nominal:.6f}")
    print(f"stable over                [{report.lower:.6f}, {report.upper:.6f}]")
    print(f"relative width             {report.relative_width:.6f}")
    print(
        f"headroom down / up         {report.downward_headroom:.4f} / "
        f"{report.upward_headroom:.4f}"
    )
    print(f"knife-edge at +/-{report.knife_edge_at:.0%}        {report.is_knife_edge}")
    print(f"flat over scanned range    {report.is_flat}")
    if report.bounded_below_by_scan or report.bounded_above_by_scan:
        print(
            "note                       an endpoint is the scan bound, so the width shown "
            "is a lower bound"
        )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    problem = _build_problem(args)
    solution = select_rate(problem)
    _print_solution(problem, solution)
    print()
    print(f"{'heuristic':<30s} {'goodput':>10s} {'rel loss':>10s} {'meets target':>13s}")
    for row in compare_to_optimum(problem, reserve_db=args.reserve_db):
        print(
            f"{row.name:<30s} {row.heuristic_goodput:>10.6f} {row.relative_loss:>10.6f} "
            f"{str(row.heuristic_meets_target):>13s}"
        )
    return 0


def _cmd_table(args: argparse.Namespace) -> int:
    table = _load_table(args.modcod_csv)
    fade = _build_fade(args)
    avail = np.asarray(fade.availability(args.margin_db, table.thresholds_db)).reshape(-1)
    print(f"fade model {fade.name}, margin {args.margin_db:.2f} dB")
    print(f"{'name':<14s} {'rate':>8s} {'thr_dB':>8s} {'A':>12s} {'R*A':>10s}")
    for i, entry in enumerate(table):
        print(
            f"{entry.name:<14s} {entry.net_rate:>8.4f} {entry.threshold_db:>8.2f} "
            f"{avail[i]:>12.8f} {entry.net_rate * avail[i]:>10.6f}"
        )
    return 0


def _cmd_fade(args: argparse.Namespace) -> int:
    fade = _build_fade(args)
    print(f"model {fade.name}")
    if isinstance(fade, LognormalFade):
        print(f"sigma_lnI {fade.sigma_ln_i:.6f}  sigma_dB {fade.sigma_db:.6f}")
        print(f"mean of 10log10(I) {fade.mean_db:.6f} dB")
    if isinstance(fade, GammaGammaFade):
        print(f"alpha {fade.alpha:.6f}  beta {fade.beta:.6f}")
    print(f"scintillation index {args.scintillation_index:.6f}")
    print(f"{'exceeded with prob':>20s} {'fade level dB':>14s}")
    for prob in (0.5, 0.9, 0.99, 0.999, 0.9999):
        print(f"{prob:>20.4f} {fade.quantile_db(prob):>14.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (exposed so tests can inspect it)."""
    parser = argparse.ArgumentParser(
        prog="coderateopt",
        description=(
            "Availability-constrained code-rate selection on a fading optical link. "
            f"{_SAFETY}."
        ),
        epilog=(
            "Every scipy.optimize.milp call here sets mip_rel_gap="
            f"{MILP_OPTIONS['mip_rel_gap']}; see coderateopt.milp.MILP_OPTIONS for why."
        ),
    )
    parser.add_argument("--version", action="version", version=f"coderateopt {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_fade_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--fade-model",
            choices=("lognormal", "gamma-gamma"),
            default="lognormal",
            help="marginal fade model (default: lognormal)",
        )
        sub.add_argument(
            "--scintillation-index",
            type=float,
            default=0.2,
            help="sigma_I^2, dimensionless, > 0 (default: 0.2)",
        )
        sub.add_argument(
            "--median-preserving",
            action="store_true",
            help="lognormal only: normalise median(I)=1 instead of E[I]=1",
        )
        sub.add_argument(
            "--gg-ratio",
            type=float,
            default=1.0,
            help="gamma-gamma only: beta/alpha in (0, 1] (default: 1.0)",
        )

    def add_problem_args(sub: argparse.ArgumentParser) -> None:
        add_fade_args(sub)
        sub.add_argument("--margin-db", type=float, default=12.0, help="clear-sky margin, dB")
        sub.add_argument(
            "--target", type=float, default=0.99, help="availability target in (0, 1)"
        )
        sub.add_argument(
            "--max-entries", type=int, default=1, help="table cardinality limit K (default: 1)"
        )
        sub.add_argument(
            "--mode",
            choices=("per_interval", "long_run"),
            default="per_interval",
            help="availability constraint form (default: per_interval)",
        )
        sub.add_argument(
            "--min-dwell",
            type=float,
            default=0.0,
            help="minimum time fraction for a used entry, in [0, 1)",
        )
        sub.add_argument(
            "--modcod-csv",
            default=None,
            help="CSV with name,rate_bits_per_symbol,threshold_db (default: illustrative table)",
        )

    solve = subparsers.add_parser("solve", help="solve one instance")
    add_problem_args(solve)
    solve.add_argument(
        "--method",
        choices=("auto", "milp", "exhaustive", "closed_form"),
        default="auto",
        help="solver path (default: auto)",
    )
    solve.set_defaults(func=_cmd_solve)

    sens = subparsers.add_parser("sensitivity", help="stability interval of the decision")
    add_problem_args(sens)
    sens.add_argument(
        "--parameter",
        choices=("scintillation_index", "margin_db", "availability_target"),
        default="scintillation_index",
    )
    sens.add_argument("--knife-edge-at", type=float, default=0.10)
    sens.set_defaults(func=_cmd_sensitivity)

    comp = subparsers.add_parser("compare", help="optimum beside the rules of thumb")
    add_problem_args(comp)
    comp.add_argument("--reserve-db", type=float, default=3.0)
    comp.set_defaults(func=_cmd_compare)

    tab = subparsers.add_parser("table", help="print the MODCOD table with availabilities")
    add_fade_args(tab)
    tab.add_argument("--margin-db", type=float, default=12.0)
    tab.add_argument("--modcod-csv", default=None)
    tab.set_defaults(func=_cmd_table)

    fade = subparsers.add_parser("fade", help="print fade quantiles")
    add_fade_args(fade)
    fade.set_defaults(func=_cmd_fade)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except InfeasibleProblem as exc:
        print(f"infeasible: {exc}", file=sys.stderr)
        return 3
    except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
