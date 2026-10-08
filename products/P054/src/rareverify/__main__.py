"""Command-line interface: ``python -m rareverify``.

Subcommands mirror the library: ``plan``, ``interval``, ``coverage``, ``mc``,
``is``, ``tilt-sweep``, ``subset``, ``surrogate``, ``benchmark``,
``known-answer``, ``campaign``.

Exit statuses
-------------
0
    Success.
2
    Invalid input (argparse default, and any ``ValueError`` or ``TypeError``
    raised by the library).
3
    A known-answer check failed, or a campaign did not meet its target.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from .benchmark import replicate, summarise, variance_reduction
from .campaign import run_campaign
from .estimate import interval_for
from .intervals import METHODS, SIDES, exact_coverage, proportion_interval, rule_of_three_upper
from .knownanswer import check_against_reference
from .limitstates import limit_state_from_name
from .montecarlo import crude_monte_carlo, required_samples_for_cov
from .planner import detection_probability, plan_campaign
from .subset import subset_simulation
from .surrogate import (
    fit_surrogate,
    surrogate_guided_importance_sampling,
    surrogate_probability,
)
from .tilting import (
    analytic_mean_shift,
    importance_sampling,
    oracle_mean_shift,
    orthogonal_mean_shift,
    scaled_mean_shift,
)

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_CHECK_FAILED = 3

LIMIT_STATES = ("linear", "lognormal", "rippled")


def _build_limit_state(args: argparse.Namespace):
    kwargs: dict[str, float] = {}
    if args.limit_state in ("linear", "rippled"):
        kwargs["beta"] = args.beta
        kwargs["dimension"] = args.dimension
    if args.limit_state == "rippled":
        kwargs["amplitude"] = args.amplitude
        kwargs["frequency"] = args.frequency
    return limit_state_from_name(args.limit_state, **kwargs)


def _add_limit_state_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--limit-state",
        choices=LIMIT_STATES,
        default="linear",
        help="which reference limit state to use (default: linear)",
    )
    parser.add_argument(
        "--beta", type=float, default=3.719, help="reliability index (default: 3.719)"
    )
    parser.add_argument(
        "--dimension", type=int, default=2, help="input dimension (default: 2)"
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.8,
        help="ripple amplitude for --limit-state rippled (default: 0.8)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=1.5,
        help="ripple frequency for --limit-state rippled (default: 1.5)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``python -m rareverify``."""
    parser = argparse.ArgumentParser(
        prog="rareverify",
        description=(
            "Plan and execute a Monte-Carlo verification campaign for a rare "
            "requirement violation. Every probability reported is the "
            "probability that the SIMULATED MODEL violates the requirement, "
            "not that any real vehicle does. Research-grade software: not "
            "flight-qualified, not certified, not approved for operational "
            "aerospace use."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="sample-size plan for a target probability")
    plan.add_argument("--target-probability", type=float, required=True)
    plan.add_argument("--confidence", type=float, default=0.95)
    plan.add_argument("--relative-width", type=float, default=0.5)
    plan.add_argument("--method", choices=METHODS, default="clopper-pearson")

    interval = subparsers.add_parser("interval", help="confidence interval for k/n")
    interval.add_argument("--failures", type=int, required=True)
    interval.add_argument("--samples", type=int, required=True)
    interval.add_argument("--confidence", type=float, default=0.95)
    interval.add_argument("--method", choices=METHODS, default="clopper-pearson")
    interval.add_argument("--side", choices=SIDES, default="two-sided")

    coverage = subparsers.add_parser(
        "coverage", help="exact coverage of an interval method at a given p"
    )
    coverage.add_argument("--samples", type=int, required=True)
    coverage.add_argument("--probability", type=float, required=True)
    coverage.add_argument("--confidence", type=float, default=0.95)
    coverage.add_argument("--method", choices=METHODS, default="clopper-pearson")
    coverage.add_argument("--side", choices=SIDES, default="two-sided")

    mc = subparsers.add_parser("mc", help="crude Monte-Carlo run")
    _add_limit_state_args(mc)
    mc.add_argument("--samples", type=int, default=200_000)
    mc.add_argument("--confidence", type=float, default=0.95)

    importance = subparsers.add_parser("is", help="importance-sampling run")
    _add_limit_state_args(importance)
    importance.add_argument("--samples", type=int, default=200_000)
    importance.add_argument(
        "--tilt",
        choices=("analytic", "oracle", "scaled", "orthogonal"),
        default="analytic",
    )
    importance.add_argument("--tilt-scale", type=float, default=1.0)
    importance.add_argument("--confidence", type=float, default=0.95)

    sweep = subparsers.add_parser(
        "tilt-sweep", help="sweep the tilt scale and report where IS is worse than crude"
    )
    _add_limit_state_args(sweep)
    sweep.add_argument("--samples", type=int, default=50_000)
    sweep.add_argument("--replications", type=int, default=20)

    sub = subparsers.add_parser("subset", help="subset-simulation run")
    _add_limit_state_args(sub)
    sub.add_argument("--per-level", type=int, default=2000)
    sub.add_argument("--p0", type=float, default=0.1)

    surrogate = subparsers.add_parser(
        "surrogate", help="fit the learned surrogate and benchmark it against the baseline"
    )
    _add_limit_state_args(surrogate)
    surrogate.add_argument("--n-train", type=int, default=300)
    surrogate.add_argument("--samples", type=int, default=100_000)

    bench = subparsers.add_parser(
        "benchmark", help="replicated comparison of crude, IS and subset simulation"
    )
    _add_limit_state_args(bench)
    bench.add_argument("--samples", type=int, default=100_000)
    bench.add_argument("--replications", type=int, default=20)

    known = subparsers.add_parser(
        "known-answer", help="run the known-answer checks; exit 3 if any fails"
    )
    known.add_argument("--samples", type=int, default=200_000)
    known.add_argument("--tolerance", type=float, default=4.0)
    known.add_argument("--seed", type=int, default=0)

    campaign = subparsers.add_parser("campaign", help="plan, run and bound a campaign")
    _add_limit_state_args(campaign)
    campaign.add_argument("--target-probability", type=float, required=True)
    campaign.add_argument("--confidence", type=float, default=0.95)
    campaign.add_argument(
        "--estimator", choices=("crude", "importance-sampling"), default="crude"
    )
    campaign.add_argument("--samples", type=int, default=None)

    return parser


def _cmd_plan(args: argparse.Namespace, out: list[str]) -> int:
    plan = plan_campaign(
        args.target_probability,
        confidence=args.confidence,
        relative_width=args.relative_width,
        method=args.method,
    )
    out.append(plan.describe())
    out.append(
        "crude runs for a 10 % coefficient of variation: "
        f"{required_samples_for_cov(args.target_probability, 0.1)}"
    )
    out.append(
        "probability of seeing at least one violation in the demonstration run: "
        f"{detection_probability(plan.n_demonstration, args.target_probability):.6f}"
    )
    return EXIT_OK


def _cmd_interval(args: argparse.Namespace, out: list[str]) -> int:
    result = proportion_interval(
        args.failures,
        args.samples,
        confidence=args.confidence,
        method=args.method,
        side=args.side,
    )
    out.append(result.describe())
    if args.failures == 0:
        out.append(f"rule-of-three approximation 3/n: {rule_of_three_upper(args.samples):.6e}")
    return EXIT_OK


def _cmd_coverage(args: argparse.Namespace, out: list[str]) -> int:
    value = exact_coverage(
        args.samples,
        args.probability,
        confidence=args.confidence,
        method=args.method,
        side=args.side,
    )
    out.append(
        f"exact coverage of {args.method} {args.side} at n={args.samples}, "
        f"p={args.probability:.6e}, nominal {args.confidence:.4g}: {value:.6f}"
    )
    out.append("under-covers" if value < args.confidence else "covers at or above nominal")
    return EXIT_OK


def _cmd_mc(args: argparse.Namespace, out: list[str]) -> int:
    limit_state = _build_limit_state(args)
    rng = np.random.default_rng(args.seed)
    estimate = crude_monte_carlo(limit_state, args.samples, rng=rng)
    out.append(limit_state.describe())
    out.append(estimate.describe())
    out.append(interval_for(estimate, confidence=args.confidence).describe())
    return EXIT_OK


def _cmd_is(args: argparse.Namespace, out: list[str]) -> int:
    limit_state = _build_limit_state(args)
    rng = np.random.default_rng(args.seed)
    if args.tilt == "analytic":
        tilt = analytic_mean_shift(limit_state)
    elif args.tilt == "oracle":
        tilt = oracle_mean_shift(limit_state, rng=rng)
    elif args.tilt == "scaled":
        tilt = scaled_mean_shift(limit_state, args.tilt_scale)
    else:
        tilt = orthogonal_mean_shift(limit_state, args.tilt_scale)
    estimate = importance_sampling(limit_state, tilt, args.samples, rng=rng)
    out.append(limit_state.describe())
    out.append(f"tilt {tilt.label}: theta={np.round(tilt.theta, 6).tolist()}")
    out.append(estimate.describe())
    out.append(interval_for(estimate, confidence=args.confidence).describe())
    return EXIT_OK


def _cmd_tilt_sweep(args: argparse.Namespace, out: list[str]) -> int:
    limit_state = _build_limit_state(args)
    reference = limit_state.analytic_probability()
    crude = summarise(
        "crude",
        replicate(
            lambda rng: crude_monte_carlo(limit_state, args.samples, rng=rng),
            args.replications,
            seed=args.seed,
        ),
        reference,
    )
    out.append(limit_state.describe())
    out.append(crude.describe())
    header = (
        f"{'tilt':>16} {'mean':>13} {'std':>12} {'relbias':>10} "
        f"{'VRF':>11} {'MSERF':>11} {'verdict':>10}"
    )
    out.append(header)
    scales = [-1.0, -0.1, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    tilts = [("scaled", s) for s in scales] + [
        ("orthogonal", s) for s in (0.25, 0.5, 0.75, 1.0)
    ]
    for kind, scale in tilts:
        tilt = (
            scaled_mean_shift(limit_state, scale)
            if kind == "scaled"
            else orthogonal_mean_shift(limit_state, scale)
        )
        summary = summarise(
            tilt.label,
            replicate(
                lambda rng, t=tilt: importance_sampling(
                    limit_state, t, args.samples, rng=rng
                ),
                args.replications,
                seed=args.seed + 1,
            ),
            reference,
        )
        if summary.empirical_std <= 0.0:
            out.append(
                f"{tilt.label:>16} {summary.mean_estimate:13.5e} "
                f"{0.0:12.4e} {summary.relative_bias:+10.4f} "
                f"{'n/a':>11} {'n/a':>11} {'DEGENERATE':>10}"
            )
            continue
        reduction = variance_reduction(summary, crude, reference)
        out.append(
            f"{tilt.label:>16} {summary.mean_estimate:13.5e} "
            f"{summary.empirical_std:12.4e} {summary.relative_bias:+10.4f} "
            f"{reduction.vrf_measured:11.4g} {reduction.mse_reduction_factor:11.4g} "
            f"{'WORSE' if reduction.worse_in_mse else 'better':>10}"
        )
    out.append(
        "verdict column is the mean-squared-error comparison, not the variance "
        "comparison: an over-tilted sampler can have tiny variance and be wrong "
        "by orders of magnitude"
    )
    return EXIT_OK


def _cmd_subset(args: argparse.Namespace, out: list[str]) -> int:
    limit_state = _build_limit_state(args)
    rng = np.random.default_rng(args.seed)
    estimate = subset_simulation(
        limit_state, n_per_level=args.per_level, p0=args.p0, rng=rng
    )
    out.append(limit_state.describe())
    out.append(estimate.describe())
    out.append(
        "levels={levels:.0f} mean acceptance rate={mean_acceptance_rate:.4f} "
        "independence-lower-bound cov={cov_independence_lower_bound:.4f}".format(
            **estimate.diagnostics
        )
    )
    out.append(interval_for(estimate).describe())
    return EXIT_OK


def _cmd_surrogate(args: argparse.Namespace, out: list[str]) -> int:
    limit_state = _build_limit_state(args)
    reference = limit_state.analytic_probability()
    rng = np.random.default_rng(args.seed)
    fit = fit_surrogate(limit_state, n_train=args.n_train, rng=rng)
    guided, design = surrogate_guided_importance_sampling(
        limit_state, fit, args.samples, rng=rng
    )
    baseline = importance_sampling(
        limit_state,
        analytic_mean_shift(limit_state),
        args.samples + args.n_train,
        rng=np.random.default_rng(args.seed + 1),
    )
    only = surrogate_probability(fit, 20_000, rng=rng)
    out.append(limit_state.describe())
    out.append(
        f"surrogate: n_train={fit.n_train} fit_seconds={fit.fit_seconds:.3f} "
        f"train_rmse={fit.diagnostics['train_rmse']:.4e}"
    )
    out.append(design.describe())
    out.append(f"analytic-IS baseline (equal evaluations): {baseline.describe()}")
    out.append(f"surrogate-guided IS                     : {guided.describe()}")
    out.append(
        f"surrogate-only (biased, for the record) : {only.describe()} "
        f"relative error {only.estimate / reference - 1:+.4f}"
    )
    verdict = (
        "surrogate-guided IS has the SMALLER standard error"
        if guided.standard_error < baseline.standard_error
        else "analytic-IS baseline has the SMALLER standard error"
    )
    out.append(f"verdict at equal true-evaluation budget: {verdict}")
    return EXIT_OK


def _cmd_benchmark(args: argparse.Namespace, out: list[str]) -> int:
    limit_state = _build_limit_state(args)
    reference = limit_state.analytic_probability()
    crude = summarise(
        "crude",
        replicate(
            lambda rng: crude_monte_carlo(limit_state, args.samples, rng=rng),
            args.replications,
            seed=args.seed,
        ),
        reference,
    )
    analytic = summarise(
        "analytic-IS",
        replicate(
            lambda rng: importance_sampling(
                limit_state, analytic_mean_shift(limit_state), args.samples, rng=rng
            ),
            args.replications,
            seed=args.seed + 1,
        ),
        reference,
    )
    per_level = max(100, args.samples // 50)
    subset = summarise(
        "subset-simulation",
        replicate(
            lambda rng: subset_simulation(limit_state, n_per_level=per_level, rng=rng),
            args.replications,
            seed=args.seed + 2,
        ),
        reference,
    )
    out.append(limit_state.describe())
    for summary in (crude, analytic, subset):
        out.append(summary.describe())
    for summary in (analytic, subset):
        out.append(variance_reduction(summary, crude, reference).describe())
    return EXIT_OK


def _cmd_known_answer(args: argparse.Namespace, out: list[str]) -> int:
    checks = []
    for name in LIMIT_STATES:
        limit_state = limit_state_from_name(name)
        reference = limit_state.analytic_probability()
        rng = np.random.default_rng(args.seed)
        crude = crude_monte_carlo(limit_state, args.samples, rng=rng)
        checks.append(
            check_against_reference(
                f"{limit_state.name} crude",
                crude,
                reference,
                limit_state.reference_kind,
                args.tolerance,
            )
        )
        tilted = importance_sampling(
            limit_state, analytic_mean_shift(limit_state), args.samples, rng=rng
        )
        checks.append(
            check_against_reference(
                f"{limit_state.name} analytic-IS",
                tilted,
                reference,
                limit_state.reference_kind,
                args.tolerance,
            )
        )
    for check in checks:
        out.append(check.describe())
    failed = [c for c in checks if not c.passed]
    out.append(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    return EXIT_CHECK_FAILED if failed else EXIT_OK


def _cmd_campaign(args: argparse.Namespace, out: list[str]) -> int:
    limit_state = _build_limit_state(args)
    report = run_campaign(
        limit_state,
        args.target_probability,
        confidence=args.confidence,
        estimator=args.estimator,
        n_samples=args.samples,
        rng=np.random.default_rng(args.seed),
    )
    out.append(limit_state.describe())
    out.append(report.describe())
    return EXIT_OK if report.meets_target else EXIT_CHECK_FAILED


_COMMANDS = {
    "plan": _cmd_plan,
    "interval": _cmd_interval,
    "coverage": _cmd_coverage,
    "mc": _cmd_mc,
    "is": _cmd_is,
    "tilt-sweep": _cmd_tilt_sweep,
    "subset": _cmd_subset,
    "surrogate": _cmd_surrogate,
    "benchmark": _cmd_benchmark,
    "known-answer": _cmd_known_answer,
    "campaign": _cmd_campaign,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status; writes to stdout/stderr."""
    parser = build_parser()
    args = parser.parse_args(argv)
    lines: list[str] = []
    try:
        status = _COMMANDS[args.command](args, lines)
    except (ValueError, TypeError) as error:
        sys.stderr.write(f"rareverify: {error}\n")
        return EXIT_BAD_INPUT
    except RuntimeError as error:
        sys.stderr.write(f"rareverify: {error}\n")
        return EXIT_CHECK_FAILED
    sys.stdout.write("\n".join(lines) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
