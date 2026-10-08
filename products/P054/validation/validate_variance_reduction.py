"""Measured variance reduction of importance sampling and subset simulation.

Every figure is an empirical variance over independent replications, put on an
equal true-evaluation footing. Nothing here is a formula prediction.

Replication noise: with R replications the empirical standard deviation has a
relative uncertainty of about 1/sqrt(2(R-1)), so at R = 40 a variance ratio is
good to roughly a factor of 1.25 and smaller differences are not resolved.

Run from the product directory:

    python validation/validate_variance_reduction.py
"""

from __future__ import annotations

import math

import numpy as np
from _reporting import Report  # noqa: E402

from rareverify.benchmark import replicate, summarise, variance_reduction  # noqa: E402
from rareverify.limitstates import (  # noqa: E402
    LinearGaussianLimitState,
    LognormalRatioLimitState,
    RippledLimitState,
)
from rareverify.montecarlo import crude_monte_carlo  # noqa: E402
from rareverify.subset import subset_simulation  # noqa: E402
from rareverify.tilting import (  # noqa: E402
    analytic_mean_shift,
    importance_sampling,
    oracle_mean_shift,
)

BUDGET = 100_000
REPLICATIONS = 40
PER_LEVEL = 2000


def main() -> int:
    report = Report("validate_variance_reduction")
    report.line(
        f"budget = {BUDGET} true evaluations, replications = {REPLICATIONS}, "
        f"subset n_per_level = {PER_LEVEL}"
    )
    report.line(
        f"replication noise on a standard deviation: "
        f"{1 / math.sqrt(2 * (REPLICATIONS - 1)):.3f} relative"
    )
    report.line("")

    instances = [
        ("linear beta=3.719 d=2", LinearGaussianLimitState(beta=3.719, dimension=2)),
        ("linear beta=4.753 d=2", LinearGaussianLimitState(beta=4.753, dimension=2)),
        ("linear beta=3.719 d=6", LinearGaussianLimitState(beta=3.719, dimension=6)),
        ("lognormal default", LognormalRatioLimitState()),
        ("rippled A=0.8 w=1.5", RippledLimitState()),
        (
            "rippled beta=5.5 A=2.5 w=2.0",
            RippledLimitState(beta=5.5, amplitude=2.5, frequency=2.0),
        ),
    ]

    for label, state in instances:
        reference = state.analytic_probability()
        report.line(f"## {label}   reference p = {reference:.8e}")

        crude = summarise(
            "crude",
            replicate(
                lambda rng, s=state: crude_monte_carlo(s, BUDGET, rng=rng),
                REPLICATIONS,
                seed=1001,
            ),
            reference,
        )
        analytic = summarise(
            "analytic-IS",
            replicate(
                lambda rng, s=state: importance_sampling(
                    s, analytic_mean_shift(s), BUDGET, rng=rng
                ),
                REPLICATIONS,
                seed=1002,
            ),
            reference,
        )
        oracle_tilt = oracle_mean_shift(state, rng=np.random.default_rng(7))
        oracle = summarise(
            "oracle-IS",
            replicate(
                lambda rng, s=state, t=oracle_tilt: importance_sampling(
                    s, t, BUDGET, rng=rng
                ),
                REPLICATIONS,
                seed=1003,
            ),
            reference,
        )
        subset = summarise(
            "subset-simulation",
            replicate(
                lambda rng, s=state: subset_simulation(
                    s, n_per_level=PER_LEVEL, rng=rng
                ),
                REPLICATIONS,
                seed=1004,
            ),
            reference,
        )
        for summary in (crude, analytic, oracle, subset):
            report.line("  " + summary.describe())
        for summary in (analytic, oracle, subset):
            reduction = variance_reduction(summary, crude, reference)
            report.line("  " + reduction.describe())
        report.check(
            f"{label}: analytic-IS beats crude in variance and in MSE",
            variance_reduction(analytic, crude, reference).vrf_measured > 1.0
            and not variance_reduction(analytic, crude, reference).worse_in_mse,
        )
        report.check(
            f"{label}: subset simulation beats crude in variance per evaluation",
            variance_reduction(subset, crude, reference).vrf_measured > 1.0,
        )
        report.check(
            f"{label}: every method's mean is within 4 replication standard "
            "errors of the reference",
            all(
                abs(s.mean_estimate - reference)
                <= 4.0 * s.empirical_std / math.sqrt(s.n_replications)
                for s in (crude, analytic, oracle, subset)
            ),
        )
        report.line("")

    report.line("## Reported standard error against the measured spread")
    report.line(
        "se_ratio = mean reported standard error / empirical standard "
        "deviation. A value near 1 means the estimator's own error bar is "
        "honest. Subset simulation's reported standard error assumes "
        "independence within a level and is therefore expected to be below 1."
    )
    report.line(f"{'instance':>30} {'crude':>8} {'analytic-IS':>12} {'subset':>8}")
    subset_ratios = []
    for label, state in instances:
        reference = state.analytic_probability()
        crude = summarise(
            "crude",
            replicate(
                lambda rng, s=state: crude_monte_carlo(s, BUDGET, rng=rng),
                REPLICATIONS,
                seed=2001,
            ),
            reference,
        )
        analytic = summarise(
            "analytic-IS",
            replicate(
                lambda rng, s=state: importance_sampling(
                    s, analytic_mean_shift(s), BUDGET, rng=rng
                ),
                REPLICATIONS,
                seed=2002,
            ),
            reference,
        )
        subset = summarise(
            "subset-simulation",
            replicate(
                lambda rng, s=state: subset_simulation(
                    s, n_per_level=PER_LEVEL, rng=rng
                ),
                REPLICATIONS,
                seed=2003,
            ),
            reference,
        )
        subset_ratios.append(subset.se_ratio)
        report.line(
            f"{label:>30} {crude.se_ratio:>8.3f} {analytic.se_ratio:>12.3f} "
            f"{subset.se_ratio:>8.3f}"
        )
    report.check(
        "subset simulation's reported standard error understates its measured "
        "spread on every instance, as the module docstring states",
        all(ratio < 1.0 for ratio in subset_ratios),
        f"(ratios {[round(r, 3) for r in subset_ratios]}, "
        f"mean understatement factor {1 / np.mean(subset_ratios):.2f})",
    )

    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
