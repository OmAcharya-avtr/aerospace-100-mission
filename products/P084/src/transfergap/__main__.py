"""Command line interface: ``python -m transfergap <command>``.

Every command prints a table and nothing else. No command writes a file, and
no command has a hidden default: the sample size, campaign shape, confidence
and seed are all arguments with declared values, printed back in the header of
every run so that an output pasted into a report carries its own provenance.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .bounds import (
    empirical_bernstein_mean_upper,
    hoeffding_mean_upper,
    normal_tolerance_upper,
    wilks_tolerance_upper,
)
from .exchangeability import STATISTICS, permutation_test
from .learned import ConformalisedDiscrepancyModel, DiscrepancyModel
from .plan import plan_existence, plan_for_slack
from .report import (
    format_table,
    render_bounds,
    render_exchangeability,
    render_plan,
)
from .resample import bootstrap_mean_upper, split_conformal_upper
from .synthetic import BOUND_RANGE_KNM, CAMPAIGNS, UNIT, generate, population_truth


def _header(arguments: argparse.Namespace) -> str:
    fields = [f"transfergap {__version__}", f"command {arguments.command}"]
    for key in ("n", "campaign", "seed", "confidence", "proportion", "target_slack"):
        value = getattr(arguments, key, None)
        if value is not None:
            fields.append(f"{key} {value}")
    return " | ".join(fields)


def _bounds(arguments: argparse.Namespace) -> int:
    sample = generate(arguments.n, arguments.campaign, seed=arguments.seed)
    bounds = [
        hoeffding_mean_upper(sample.gaps, sample.bound_range, arguments.confidence, UNIT),
        empirical_bernstein_mean_upper(
            sample.gaps, sample.bound_range, arguments.confidence, UNIT
        ),
        bootstrap_mean_upper(
            sample.gaps, sample.bound_range, arguments.confidence, seed=arguments.seed,
            unit=UNIT,
        ),
    ]
    for builder in (wilks_tolerance_upper, normal_tolerance_upper):
        try:
            bounds.append(
                builder(
                    sample.gaps,
                    sample.bound_range,
                    arguments.proportion,
                    arguments.confidence,
                    UNIT,
                )
            )
        except ValueError as error:
            print(f"[not available] {builder.__name__}: {error}")
    try:
        bounds.append(
            split_conformal_upper(
                sample.gaps, sample.bound_range, arguments.proportion, UNIT
            )
        )
    except ValueError as error:
        print(f"[not available] split_conformal_upper: {error}")
    print(render_bounds(bounds, UNIT))
    return 0


def _plan(arguments: argparse.Namespace) -> int:
    sample = generate(arguments.n, arguments.campaign, seed=arguments.seed)
    plans = plan_for_slack(sample, arguments.target_slack, arguments.confidence)
    plans += plan_existence(sample, arguments.proportion, arguments.confidence)
    print(render_plan(plans))
    return 0


def _exchangeability(arguments: argparse.Namespace) -> int:
    sample = generate(arguments.n, arguments.campaign, seed=arguments.seed)
    tests = [
        permutation_test(
            sample.gaps,
            statistic=name,
            permutations=arguments.permutations,
            alpha=1.0 - arguments.confidence,
            seed=arguments.seed,
        )
        for name in sorted(STATISTICS)
    ]
    print(render_exchangeability(tests))
    return 0


def _learned(arguments: argparse.Namespace) -> int:
    sample = generate(arguments.n, arguments.campaign, seed=arguments.seed)
    pool = generate(2000, "random", seed=arguments.seed + 90001)
    model = DiscrepancyModel(
        upper_level=arguments.proportion,
        lower_level=1.0 - arguments.proportion,
        seed=arguments.seed,
    ).fit(sample.features, sample.gaps)
    prediction = model.predict_with_uncertainty(pool.features)
    raw_coverage = float(np.mean(pool.gaps <= prediction.upper))
    rows = [
        [
            "learned quantile (raw)",
            f"{arguments.proportion:.3f}",
            f"{raw_coverage:.4f}",
            f"{float(np.mean(prediction.upper)):.3f}",
            f"{prediction.crossing_rate:.4f}",
        ]
    ]
    try:
        calibrated = ConformalisedDiscrepancyModel(
            confidence=arguments.proportion, seed=arguments.seed
        ).fit(sample.features, sample.gaps)
        band = calibrated.predict_upper(pool.features)
        rows.append(
            [
                "learned quantile (conformal)",
                f"{arguments.proportion:.3f}",
                f"{float(np.mean(pool.gaps <= band)):.4f}",
                f"{float(np.mean(band)):.3f}",
                "-",
            ]
        )
    except ValueError as error:
        print(f"[not available] conformalised model: {error}")
    print(
        format_table(
            ["model", "declared", "achieved coverage", f"mean band [{UNIT}]", "crossing"],
            rows,
        )
    )
    print("")
    print("Coverage is measured on a 2000-point held-out pool from the same envelope.")
    print("A single run is one draw; the repeated-draw measurement is in")
    print("validation/validate_learned_model_output.txt and is what this product is")
    print("scored on. This model is not certified for operational flight use.")
    return 0


def _truth(arguments: argparse.Namespace) -> int:
    truth = population_truth(draws=arguments.draws)
    rows = [["mean", f"{truth.mean:.4f}", f"+/- {truth.standard_error:.4f}"]]
    for proportion, value in sorted(truth.quantiles.items()):
        rows.append([f"quantile {proportion:g}", f"{value:.4f}", "-"])
    rows.append(["clip fraction at R", f"{truth.clip_fraction:.6f}", "-"])
    rows.append(["declared R", f"{BOUND_RANGE_KNM:.4f}", "-"])
    print(format_table([f"population truth [{UNIT}]", "value", "MC standard error"], rows))
    print("")
    print(f"Monte Carlo draws {truth.draws}, seed {truth.seed}. This reference exists")
    print("only because the data is synthetic; a real campaign has no such reference,")
    print("which is why the bounds in this package have to be distribution-free.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser. Exposed so the test suite can exercise it directly."""
    parser = argparse.ArgumentParser(
        prog="transfergap",
        description=(
            "Bound the simulation-to-field discrepancy from the few paired field "
            "measurements available, and invert the bound into a sample count."
        ),
    )
    parser.add_argument("--version", action="version", version=f"transfergap {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--n", type=int, default=40, help="paired field measurements")
        sub.add_argument(
            "--campaign", choices=list(CAMPAIGNS), default="random",
            help="'random' is exchangeable; 'swept' is deliberately not",
        )
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--confidence", type=float, default=0.95)
        sub.add_argument("--proportion", type=float, default=0.95)

    bounds = subparsers.add_parser("bounds", help="every bound on one synthetic sample")
    common(bounds)
    bounds.set_defaults(handler=_bounds)

    plan = subparsers.add_parser("plan", help="how many MORE measurements are needed")
    common(plan)
    plan.add_argument(
        "--target-slack", dest="target_slack", type=float, default=2.0,
        help=f"slack to reach, {UNIT}",
    )
    plan.set_defaults(handler=_plan)

    exchange = subparsers.add_parser(
        "exchangeability", help="permutation test of the assumption baseline 2 rests on"
    )
    common(exchange)
    exchange.add_argument("--permutations", type=int, default=2000)
    exchange.set_defaults(handler=_exchangeability)

    learned = subparsers.add_parser("learned", help="the learned model on one sample")
    common(learned)
    learned.set_defaults(handler=_learned)

    truth = subparsers.add_parser("truth", help="the synthetic population truth")
    truth.add_argument("--draws", type=int, default=200_000)
    truth.set_defaults(handler=_truth)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    print(_header(arguments))
    print("")
    try:
        return int(arguments.handler(arguments))
    except (ValueError, TypeError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
