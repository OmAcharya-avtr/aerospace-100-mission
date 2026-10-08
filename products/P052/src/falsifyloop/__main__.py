"""Command-line interface: ``python -m falsifyloop``.

Four subcommands:

``instances``
    List the benchmark suite with its declared difficulty tiers.
``falsify``
    Run one strategy on one instance at one budget and seed.
``benchmark``
    Run the instance-by-strategy grid and print the per-instance table first,
    the aggregate second.
``evaluate``
    Simulate one decision vector and report the requirement robustness.

Vocabulary
----------
**Falsification is one-sided: finding no violation is not evidence of
correctness.** Nothing this CLI prints reads as a verdict of correctness. A
search that found nothing prints "NO VIOLATION FOUND" and restates the
one-sidedness; the words "pass", "safe", "verified" and their relatives never
appear. ``tests/test_vocabulary.py`` asserts that over the rendered output.

Exit status
-----------
``0`` means the command ran, **not** that anything was or was not found, so that
no CI system can read a zero as a clean result. ``2`` is a usage or validation
error. Pass ``--exit-on-violation`` to make a *found* violation exit ``1``; the
absence of a violation is never encoded in the exit status, in either mode.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .benchmark import run_benchmark, run_cell
from .curves import bootstrap_band, clopper_pearson, efficiency_curve
from .instances import SUITE_ORDER, instance, suite
from .report import (
    ONE_SIDED_NOTE,
    render_aggregate_table,
    render_cell_detail,
    render_cell_table,
    render_curve_points,
    render_difficulty_table,
    render_instances,
    render_search_result,
)
from .requirements import robustness
from .search import STRATEGIES
from .systems import LoopInput

_EPILOG = (
    "Falsification is one-sided: finding no violation is not evidence of correctness. "
    "Research-grade software; not flight-qualified, not certified, not approved for "
    "operational aerospace use."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="falsifyloop",
        description=(
            "Requirement falsification for autonomous control loops: find the setting "
            "that violates a requirement, and measure how many simulations it took."
        ),
        epilog=_EPILOG,
    )
    parser.add_argument("--version", action="version", version=f"falsifyloop {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "instances",
        help="list the benchmark suite with declared difficulty tiers",
        epilog=_EPILOG,
    )

    falsify = subparsers.add_parser(
        "falsify", help="run one strategy on one instance", epilog=_EPILOG
    )
    falsify.add_argument("--instance", required=True, choices=sorted(SUITE_ORDER))
    falsify.add_argument("--strategy", default="uniform-random", choices=sorted(STRATEGIES))
    falsify.add_argument("--budget", type=int, default=100, help="simulations (default 100)")
    falsify.add_argument("--seed", type=int, default=0)
    falsify.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="seeded runs; above 1 also prints the efficiency curve at selected counts",
    )
    falsify.add_argument(
        "--exit-on-violation",
        action="store_true",
        help="exit 1 when a violation is found; the absence of one is never an exit status",
    )

    bench = subparsers.add_parser(
        "benchmark", help="run the instance-by-strategy grid", epilog=_EPILOG
    )
    bench.add_argument(
        "--instances",
        nargs="+",
        default=None,
        choices=sorted(SUITE_ORDER),
        help="default: the whole suite",
    )
    bench.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        choices=sorted(STRATEGIES),
        help="default: all five, baseline first",
    )
    bench.add_argument("--budget", type=int, default=100)
    bench.add_argument("--repeats", type=int, default=10)
    bench.add_argument("--base-seed", type=int, default=1000)

    evaluate = subparsers.add_parser(
        "evaluate", help="simulate one decision vector and report robustness", epilog=_EPILOG
    )
    evaluate.add_argument("--instance", required=True, choices=sorted(SUITE_ORDER))
    evaluate.add_argument(
        "--input",
        required=True,
        help=(
            "six comma-separated decision variables in order: "
            + ", ".join(LoopInput.FIELDS)
        ),
    )

    difficulty = subparsers.add_parser(
        "difficulty",
        help="estimate instance difficulty by uniform sampling",
        epilog=_EPILOG,
    )
    difficulty.add_argument(
        "--instances", nargs="+", default=None, choices=sorted(SUITE_ORDER)
    )
    difficulty.add_argument("--draws", type=int, default=2000)
    difficulty.add_argument("--seed", type=int, default=0)

    return parser


def _cmd_instances(_: argparse.Namespace, out) -> int:
    print(render_instances(suite()), file=out)
    print(f"\n{ONE_SIDED_NOTE}", file=out)
    return 0


def _cmd_falsify(args: argparse.Namespace, out) -> int:
    inst = instance(args.instance)
    if args.budget < 1:
        raise ValueError(f"--budget must be at least 1, got {args.budget}")
    if args.repeats < 1:
        raise ValueError(f"--repeats must be at least 1, got {args.repeats}")
    if args.repeats == 1:
        result = STRATEGIES[args.strategy](inst, args.budget, args.seed)
        print(render_search_result(result, inst), file=out)
        return 1 if (args.exit_on_violation and result.found) else 0

    cell, results = run_cell(inst, args.strategy, args.budget, args.repeats, args.seed)
    print(render_cell_detail(cell), file=out)
    curve = efficiency_curve(cell.first_violations, cell.budget)
    lower, upper = bootstrap_band(cell.first_violations, cell.budget, seed=args.seed)
    marks = tuple(sorted({1, max(1, args.budget // 10), args.budget // 2, args.budget}))
    print("", file=out)
    print(render_curve_points(curve, lower, upper, marks), file=out)
    print("", file=out)
    print(
        f"violation-found rate {cell.success_rate:.3f} over {cell.repeats} seeds; "
        f"mean curve probability {cell.mean_curve_probability:.4f}",
        file=out,
    )
    print(ONE_SIDED_NOTE, file=out)
    any_found = any(r.found for r in results)
    return 1 if (args.exit_on_violation and any_found) else 0


def _cmd_benchmark(args: argparse.Namespace, out) -> int:
    chosen = (
        tuple(instance(i) for i in args.instances) if args.instances is not None else None
    )
    report = run_benchmark(
        instances=chosen,
        strategy_names=args.strategies,
        budget=args.budget,
        repeats=args.repeats,
        base_seed=args.base_seed,
    )
    print(render_cell_table(report), file=out)
    print("", file=out)
    print(render_aggregate_table(report), file=out)
    print("", file=out)
    print(
        f"wall clock {report.wall_clock_seconds:.1f} s on this machine; a shared-container "
        "wall clock, not a hardware characteristic",
        file=out,
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace, out) -> int:
    parts = [p for p in args.input.replace(" ", "").split(",") if p]
    if len(parts) != len(LoopInput.FIELDS):
        raise ValueError(
            f"--input needs {len(LoopInput.FIELDS)} comma-separated values "
            f"({', '.join(LoopInput.FIELDS)}), got {len(parts)}"
        )
    try:
        vector = np.array([float(p) for p in parts])
    except ValueError as exc:
        raise ValueError(f"--input must be numeric: {exc}") from None
    inst = instance(args.instance)
    clipped = inst.clip(vector)
    if not np.allclose(clipped, vector):
        raise ValueError(
            "input lies outside the declared search box; box rows are\n"
            + "\n".join(
                f"  {n}: [{lo:g}, {hi:g}]"
                for n, (lo, hi) in zip(LoopInput.FIELDS, inst.box, strict=True)
            )
        )
    trace = inst.simulate(vector)
    rho = robustness(inst.requirement, trace)
    print(f"instance    : {inst.identifier}  [{inst.tier}]", file=out)
    print(f"requirement : {inst.requirement}", file=out)
    for name, value in zip(LoopInput.FIELDS, vector, strict=True):
        print(f"  {name:<16s} {value:+.6f}", file=out)
    print(f"robustness  : {rho:+.6f} (dimensionless)", file=out)
    if rho < 0.0:
        print("VIOLATION: robustness is negative, the requirement is violated.", file=out)
    else:
        print(
            "NO VIOLATION at this single point. One point says nothing about the box.",
            file=out,
        )
        print(ONE_SIDED_NOTE, file=out)
    return 0


def _cmd_difficulty(args: argparse.Namespace, out) -> int:
    if args.draws < 1:
        raise ValueError(f"--draws must be at least 1, got {args.draws}")
    chosen = (
        tuple(instance(i) for i in args.instances) if args.instances is not None else suite()
    )
    rows = []
    for inst in chosen:
        rng = np.random.default_rng(args.seed)
        points = inst.sample(rng, args.draws)
        violations = sum(1 for p in points if inst.evaluate(p) < 0.0)
        rows.append(
            (inst.identifier, inst.tier, violations, args.draws, inst.design_target_probability)
        )
    print(render_difficulty_table(tuple(rows)), file=out)
    print("", file=out)
    print(
        "p is the fraction of uniform draws from the declared box that violate the "
        "requirement. The interval is Clopper-Pearson exact. A zero count gives a lower "
        "limit of exactly zero and says only that p is below the upper limit.",
        file=out,
    )
    def _width(row: tuple) -> float:
        lo, hi = clopper_pearson(row[2], row[3])
        return hi - lo

    widest = max(rows, key=_width)
    print(f"widest interval: {widest[0]}", file=out)
    print(ONE_SIDED_NOTE, file=out)
    return 0


_COMMANDS = {
    "instances": _cmd_instances,
    "falsify": _cmd_falsify,
    "benchmark": _cmd_benchmark,
    "evaluate": _cmd_evaluate,
    "difficulty": _cmd_difficulty,
}


def main(argv: Sequence[str] | None = None, out=None) -> int:
    """Entry point. Returns the process exit status; see the module docstring."""
    stream = sys.stdout if out is None else out
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _COMMANDS[args.command](args, stream)
    except (ValueError, TypeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
