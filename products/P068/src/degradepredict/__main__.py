"""Command line interface: ``python -m degradepredict``.

Three subcommands:

``generate``
    Write a seeded synthetic population to CSV, one row per sample, so the
    shipped data set can be regenerated and diffed.
``fit``
    Fit the three analytic baselines to one trajectory's window and print the
    crossing estimate with its Fieller and delta intervals.
``benchmark``
    Run the full head-to-head and print the coverage and score table, with
    complete and right-censored trajectories in separate columns.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from .baselines import DegradationFitError, exponential_baseline, linear_baseline
from .benchmark import run_head_to_head
from .censoring import censoring_summary
from .generate import DEFAULT_LIMIT, GeneratorConfig, generate_population
from .physics import powerlaw_fade_baseline
from .report import format_fit, format_head_to_head


def _population(n: int, seed: int):
    return generate_population(n, seed=seed, config=GeneratorConfig(), limit=DEFAULT_LIMIT)


def _cmd_generate(args: argparse.Namespace) -> int:
    pop = _population(args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["traj_id", "family", "noise_sd_percent", "crossing_time_days", "origin_days",
             "followup_days", "remaining_true_days", "remaining_observed_days", "event",
             "time_days", "capacity_percent_bol"]
        )
        for tr in pop:
            for t, y in zip(tr.times_days, tr.values_percent, strict=True):
                writer.writerow(
                    [tr.traj_id, tr.family, f"{tr.noise_sd:.6f}",
                     f"{tr.crossing_time_days:.6f}", f"{tr.origin_days:.6f}",
                     f"{tr.followup_days:.6f}", f"{tr.remaining_true_days:.6f}",
                     f"{tr.remaining_observed_days:.6f}", int(tr.event),
                     f"{t:.6f}", f"{y:.6f}"]
                )
    summary = censoring_summary(
        np.array([tr.remaining_observed_days for tr in pop]),
        np.array([tr.event for tr in pop]),
    )
    print(f"wrote {out} : {len(pop)} trajectories, "
          f"{sum(len(tr.times_days) for tr in pop)} samples")
    print(f"censored fraction {summary['censored_fraction']:.4f} "
          f"({int(summary['n_censored'])} of {int(summary['n'])})")
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    pop = _population(max(args.index + 1, 1), args.seed)
    tr = pop[args.index]
    print(f"trajectory {tr.traj_id}: generated family {tr.family}, "
          f"true crossing {tr.crossing_time_days:.2f} d, origin {tr.origin_days:.2f} d, "
          f"remaining {tr.remaining_true_days:.2f} d, "
          f"{'complete' if tr.event else 'right-censored'}")
    print(f"limit: {DEFAULT_LIMIT.value} {DEFAULT_LIMIT.unit} ({DEFAULT_LIMIT.direction})")
    for fitter in (linear_baseline, exponential_baseline, powerlaw_fade_baseline):
        print()
        try:
            fit = fitter(tr.times_days, tr.values_percent, DEFAULT_LIMIT, level=args.level)
        except DegradationFitError as exc:
            print(f"model {fitter.__name__}: no crossing reported ({exc})")
            continue
        print(format_fit(fit, unit=DEFAULT_LIMIT.unit))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    train = _population(args.n_train, args.seed)
    test = _population(args.n_test, args.seed + 1)
    results = run_head_to_head(train, test, DEFAULT_LIMIT, level=args.level)
    print(f"train {len(train)} trajectories (seed {args.seed}), "
          f"test {len(test)} (seed {args.seed + 1})")
    print(format_head_to_head(results, level=args.level))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``python -m degradepredict``."""
    parser = argparse.ArgumentParser(
        prog="degradepredict",
        description=(
            "Limit-crossing forecasting for a degrading subsystem, with right "
            "censoring handled explicitly. Research-grade; not flight-qualified, "
            "not certified, not approved for operational aerospace use."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write a seeded synthetic population to CSV")
    gen.add_argument("--n", type=int, default=200, help="number of trajectories")
    gen.add_argument("--seed", type=int, default=20261016, help="generator seed")
    gen.add_argument("--out", default="population.csv", help="output CSV path")
    gen.set_defaults(func=_cmd_generate)

    fit = sub.add_parser("fit", help="fit the three analytic baselines to one trajectory")
    fit.add_argument("--index", type=int, default=0, help="trajectory index in the population")
    fit.add_argument("--seed", type=int, default=20261016, help="generator seed")
    fit.add_argument("--level", type=float, default=0.90, help="confidence level")
    fit.set_defaults(func=_cmd_fit)

    bench = sub.add_parser("benchmark", help="head-to-head over a seeded train/test split")
    bench.add_argument("--n-train", type=int, default=1200)
    bench.add_argument("--n-test", type=int, default=400)
    bench.add_argument("--seed", type=int, default=20261016)
    bench.add_argument("--level", type=float, default=0.90)
    bench.set_defaults(func=_cmd_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    level = getattr(args, "level", None)
    if level is not None and (level <= 0 or level >= 1):
        parser.error(f"--level must be in (0, 1), got {level}")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
