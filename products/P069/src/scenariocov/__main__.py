"""Command line interface: ``python -m scenariocov <command>``.

Commands:

* ``report``    full deterministic coverage report for a finished campaign;
* ``uncovered`` just the enumerated uncovered t-way tuples, one per line;
* ``occupancy`` grid occupancy and the empty blocks;
* ``propose``   the learned gap finder's next batch, with its uncertainty;
* ``benchmark`` the seeded head-to-head of the three strategies.

Every command takes a space document and a campaign document; see
:mod:`scenariocov.io` for their shapes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from .benchmark import coverage_benchmark
from .io import load_campaign, load_class_document, load_space
from .occupancy import grid_occupancy
from .report import build_report
from .space import ScenarioSpace
from .strategies import (
    CoverageGreedyReference,
    GreedyMaxMinStrategy,
    LearnedGapFinderStrategy,
    UniformRandomStrategy,
)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--space", required=True, help="path to the space JSON document")
    parser.add_argument("--campaign", required=True, help="path to the executed campaign JSON")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (exposed so the tests can drive it)."""
    parser = argparse.ArgumentParser(
        prog="python -m scenariocov",
        description=(
            "Report which regions of a declared scenario space a finished verification "
            "campaign did not touch. Research-grade; not flight-qualified."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="full deterministic coverage report")
    _add_common(report)
    report.add_argument("--classes", help="path to the classes/boundaries JSON document")
    report.add_argument("--strength", type=int, default=2, help="interaction strength t")
    report.add_argument("--max-uncovered", type=int, default=20, help="uncovered tuples to print")

    uncovered = sub.add_parser("uncovered", help="enumerate uncovered t-way tuples")
    _add_common(uncovered)
    uncovered.add_argument("--strength", type=int, default=2, help="interaction strength t")

    occupancy = sub.add_parser("occupancy", help="grid occupancy over continuous parameters")
    _add_common(occupancy)
    occupancy.add_argument("--blocks", type=int, default=3, help="empty blocks to list")

    propose = sub.add_parser("propose", help="learned gap finder: propose the next batch")
    _add_common(propose)
    propose.add_argument("--pool", required=True, help="path to the candidate pool JSON")
    propose.add_argument("--count", type=int, default=5, help="batch size")
    propose.add_argument("--bootstrap", type=int, default=16, help="bootstrap replicates")
    propose.add_argument("--seed", type=int, default=0, help="bootstrap seed")

    benchmark = sub.add_parser("benchmark", help="seeded head-to-head of the three strategies")
    _add_common(benchmark)
    benchmark.add_argument("--pool", required=True, help="path to the candidate pool JSON")
    benchmark.add_argument("--start", type=int, default=40, help="starting campaign size")
    benchmark.add_argument("--steps", type=int, default=40, help="scenarios added per seed")
    benchmark.add_argument("--pool-size", type=int, default=200, help="candidates per seed")
    benchmark.add_argument("--seeds", type=int, default=12, help="number of seeds")
    benchmark.add_argument("--strength", type=int, default=2, help="interaction strength t")
    benchmark.add_argument(
        "--with-reference", action="store_true", help="include the coverage-greedy ceiling"
    )
    return parser


def _load(args: argparse.Namespace) -> tuple[ScenarioSpace, tuple[dict[str, object], ...]]:
    return load_space(args.space), load_campaign(args.campaign)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    args = build_parser().parse_args(argv)
    out = sys.stdout

    if args.command == "report":
        space, records = _load(args)
        classes, boundaries = (None, None)
        if args.classes:
            classes, boundaries = load_class_document(args.classes)
        report = build_report(
            space, records, strength=args.strength, classes=classes, boundaries=boundaries
        )
        out.write(report.to_text(max_uncovered=args.max_uncovered) + "\n")
        return 0

    if args.command == "uncovered":
        space, records = _load(args)
        report = build_report(space, records, strength=args.strength)
        for tup in report.tway.uncovered:
            out.write(tup.as_text() + "\n")
        out.write(
            f"# {report.tway.n_uncovered} uncovered of {report.tway.total} "
            f"{args.strength}-way tuples\n"
        )
        return 0

    if args.command == "occupancy":
        space, records = _load(args)
        occ = grid_occupancy(space, space.encode_many(records))
        out.write(
            f"grid over ({', '.join(occ.parameters)}) shape {occ.shape}: "
            f"{occ.occupied}/{occ.n_cells} occupied = {occ.fraction:.6f}\n"
        )
        out.write(
            f"uniform-random reference (Feller eq. 3, equiprobable cells): "
            f"{occ.expected_fraction_uniform:.6f}\n"
        )
        for block in occ.empty_blocks()[: args.blocks]:
            out.write(f"empty block of {block.size} cells, first cell {block.cells[0]}\n")
        return 0

    if args.command == "propose":
        space, records = _load(args)
        pool = load_campaign(args.pool)
        from .gapfinder import GapFinder

        finder = GapFinder(n_bootstrap=args.bootstrap, random_state=args.seed)
        finder.fit(space.normalise(records))
        proposals = finder.propose(space.normalise(pool), args.count)
        out.write(
            f"{'#':>3}  {'log_density':>12}{'sd':>8}{'confidence':>12}  scenario\n"
        )
        for p in proposals:
            record = pool[p.candidate_index]
            rendered = " ".join(f"{k}={record[k]}" for k in space.names)
            out.write(
                f"{p.order:>3}  {p.log_density:>12.4f}{p.log_density_sd:>8.4f}"
                f"{p.sparsity_agreement:>12.4f}  {rendered}\n"
            )
        return 0

    space, records = _load(args)
    pool = load_campaign(args.pool)
    strategies = [UniformRandomStrategy(), GreedyMaxMinStrategy(), LearnedGapFinderStrategy()]
    if args.with_reference:
        strategies.append(CoverageGreedyReference(t=args.strength))
    result = coverage_benchmark(
        space,
        records,
        pool,
        strategies,
        n_start=args.start,
        n_steps=args.steps,
        pool_size=args.pool_size,
        seeds=tuple(range(args.seeds)),
        t=args.strength,
    )
    out.write(
        f"{args.strength}-way coverage, {len(result.seeds)} seeds, start {result.n_start}, "
        f"+{result.n_steps} scenarios from a pool of {result.pool_size}\n"
    )
    header = f"{'strategy':<28}{'final mean':>12}{'sd':>10}{'min':>10}{'max':>10}{'gain/sc':>12}"
    out.write(header + "\n")
    for name, curves in result.by_strategy.items():
        s = curves.summary()
        out.write(
            f"{name:<28}{s['final_mean']:>12.6f}{s['final_sd']:>10.6f}"
            f"{s['final_min']:>10.6f}{s['final_max']:>10.6f}{s['gain_mean']:>12.6f}\n"
        )
    ranking = result.ranking()
    out.write(f"best mean gain per scenario: {ranking[0][0]} ({ranking[0][1]:.6f})\n")
    if np.isfinite(ranking[0][1]):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
