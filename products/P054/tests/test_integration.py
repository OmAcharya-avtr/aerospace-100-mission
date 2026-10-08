"""Integration test: a whole campaign from plan to bound, through every module.

This exercises the real path a user takes: size the campaign from a target
probability, discover that the crude estimator cannot afford it, spend the
budget on importance sampling instead, cross-check against subset simulation
and the analytic reference, and quote the interval that is valid for the
estimator that produced the number.
"""

from __future__ import annotations

import numpy as np
import pytest

from rareverify import (
    analytic_mean_shift,
    binomial_interval,
    check_against_reference,
    counting_noise_floor,
    crude_monte_carlo,
    importance_sampling,
    interval_for,
    plan_campaign,
    replicate,
    required_samples_for_cov,
    run_campaign,
    subset_simulation,
    summarise,
    variance_reduction,
)
from rareverify.limitstates import LinearGaussianLimitState


def test_full_campaign_from_plan_to_bound():
    target = 1e-6
    state = LinearGaussianLimitState(beta=4.753, dimension=2)
    reference = state.analytic_probability()
    assert 0.9e-6 < reference < 1.1e-6

    # 1. Plan. The crude demonstration size is unaffordable here, which is the
    #    finding the planner exists to deliver.
    plan = plan_campaign(target, confidence=0.95)
    assert plan.n_demonstration == 2_995_731
    assert required_samples_for_cov(target, 0.1) == 99_999_900

    # 2. Run what the budget actually allows with the crude estimator, and
    #    establish that it cannot resolve the probability.
    budget = 200_000
    crude = crude_monte_carlo(state, budget, rng=np.random.default_rng(1))
    assert crude.n_failures <= 3
    crude_bound = binomial_interval(crude, side="upper").upper
    assert crude_bound > reference

    # 3. Spend the same budget on importance sampling with the declared tilt.
    tilted = importance_sampling(
        state, analytic_mean_shift(state), budget, rng=np.random.default_rng(2)
    )
    assert tilted.true_evaluations == budget
    floor = counting_noise_floor(reference, budget)
    assert tilted.standard_error < 0.05 * floor

    # 4. The interval quoted must match the estimator.
    crude_interval = interval_for(crude)
    tilted_interval = interval_for(tilted)
    assert crude_interval.kind == "clopper-pearson"
    assert tilted_interval.kind == "normal-weighted"
    assert tilted_interval.lower <= reference <= tilted_interval.upper

    # 5. Known-answer check against the closed form.
    check = check_against_reference(
        "integration linear beta=4.753",
        tilted,
        reference,
        state.reference_kind,
        tolerance_multiples=4.0,
    )
    assert check.passed, check.describe()

    # 6. An independent method must agree to within its own spread.
    subset_estimates = np.array(
        [
            subset_simulation(
                state, n_per_level=2000, rng=np.random.default_rng([7, i])
            ).estimate
            for i in range(10)
        ]
    )
    spread = subset_estimates.std(ddof=1) / np.sqrt(subset_estimates.size)
    assert abs(subset_estimates.mean() - reference) < 4.0 * spread

    # 7. The measured variance reduction, on an equal evaluation basis.
    crude_summary = summarise(
        "crude",
        replicate(
            lambda rng: crude_monte_carlo(state, budget, rng=rng), 10, seed=11
        ),
        reference,
    )
    tilted_summary = summarise(
        "analytic-IS",
        replicate(
            lambda rng: importance_sampling(
                state, analytic_mean_shift(state), budget, rng=rng
            ),
            10,
            seed=12,
        ),
        reference,
    )
    reduction = variance_reduction(tilted_summary, crude_summary, reference)
    assert reduction.vrf_measured > 100.0
    assert reduction.worse_in_mse is False


def test_campaign_report_and_plan_agree_on_the_bound_they_use():
    """The plan sizes from the one-sided limit; the verdict must use the same one."""
    state = LinearGaussianLimitState(beta=5.2, dimension=2)
    report = run_campaign(state, 1e-4, estimator="crude", rng=np.random.default_rng(1))
    assert report.estimate.n_failures == 0
    # The Beta-quantile path and the closed form agree to 13 significant
    # figures, not bit for bit; the closed form is the exact one.
    assert report.one_sided_upper == pytest.approx(
        report.zero_failure_bound, rel=1e-12
    )
    assert report.zero_failure_bound == report.plan.zero_failure_bound
    assert report.meets_target is True
