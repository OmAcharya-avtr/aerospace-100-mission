"""Tests for the replication-based benchmark harness."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rareverify.benchmark import replicate, summarise, variance_reduction
from rareverify.limitstates import LinearGaussianLimitState
from rareverify.montecarlo import crude_monte_carlo
from rareverify.tilting import (
    analytic_mean_shift,
    importance_sampling,
    orthogonal_mean_shift,
    scaled_mean_shift,
)

STATE = LinearGaussianLimitState(beta=3.719, dimension=2)
REFERENCE = STATE.analytic_probability()


def _crude(n: int, replications: int, seed: int):
    return summarise(
        "crude",
        replicate(
            lambda rng: crude_monte_carlo(STATE, n, rng=rng), replications, seed=seed
        ),
        REFERENCE,
    )


def test_replicate_is_reproducible_and_independent():
    first = replicate(
        lambda rng: crude_monte_carlo(STATE, 5000, rng=rng), 5, seed=1
    )
    again = replicate(
        lambda rng: crude_monte_carlo(STATE, 5000, rng=rng), 5, seed=1
    )
    assert [r.estimate for r in first] == [r.estimate for r in again]
    different = replicate(
        lambda rng: crude_monte_carlo(STATE, 5000, rng=rng), 5, seed=2
    )
    assert [r.estimate for r in first] != [r.estimate for r in different]


def test_replicate_requires_at_least_two_replications():
    with pytest.raises(ValueError, match="at least 2"):
        replicate(lambda rng: crude_monte_carlo(STATE, 100, rng=rng), 1)
    with pytest.raises(ValueError, match="at least 2"):
        summarise("one", [crude_monte_carlo(STATE, 100, rng=np.random.default_rng(1))])


def test_summary_fields_are_consistent():
    summary = _crude(20_000, 12, seed=7)
    assert summary.n_replications == 12
    assert summary.estimates.size == 12
    assert summary.mean_estimate == pytest.approx(float(summary.estimates.mean()))
    assert summary.empirical_std == pytest.approx(
        float(summary.estimates.std(ddof=1))
    )
    assert summary.mean_true_evaluations == 20_000
    assert "crude" in summary.describe()


def test_analytic_importance_sampling_beats_crude_in_both_metrics():
    crude = _crude(20_000, 20, seed=30)
    tilted = summarise(
        "analytic-IS",
        replicate(
            lambda rng: importance_sampling(
                STATE, analytic_mean_shift(STATE), 20_000, rng=rng
            ),
            20,
            seed=31,
        ),
        REFERENCE,
    )
    reduction = variance_reduction(tilted, crude, REFERENCE)
    assert reduction.vrf_measured > 100.0
    assert reduction.mse_reduction_factor > 100.0
    assert reduction.worse_than_reference is False
    assert reduction.worse_in_mse is False
    assert reduction.equal_evaluation_basis == 20_000


def test_an_orthogonal_tilt_is_worse_than_crude_in_variance_and_in_mse():
    """A required deliverable: importance sampling made worse on purpose."""
    crude = _crude(20_000, 30, seed=40)
    bad = summarise(
        "orthogonal=0.5",
        replicate(
            lambda rng: importance_sampling(
                STATE, orthogonal_mean_shift(STATE, 0.5), 20_000, rng=rng
            ),
            30,
            seed=41,
        ),
        REFERENCE,
    )
    reduction = variance_reduction(bad, crude, REFERENCE)
    assert reduction.worse_than_reference is True
    assert reduction.worse_in_mse is True
    assert reduction.vrf_measured < 1.0


def test_variance_and_mse_verdicts_disagree_for_an_over_tilted_sampler():
    """The regression that motivated the MSE metric.

    An over-tilted sampler at scale 3 collapses to an estimate many orders of
    magnitude below the truth with an almost zero variance, so the variance
    reduction factor is enormous while the mean-squared-error reduction factor
    is below 1. If this test ever fails because both verdicts agree, the
    reported variance reduction factor has stopped being a trap and the
    documentation in rareverify.benchmark should be revisited.
    """
    crude = _crude(20_000, 20, seed=50)
    over = summarise(
        "scale=3",
        replicate(
            lambda rng: importance_sampling(
                STATE, scaled_mean_shift(STATE, 3.0), 20_000, rng=rng
            ),
            20,
            seed=51,
        ),
        REFERENCE,
    )
    reduction = variance_reduction(over, crude, REFERENCE)
    assert reduction.vrf_measured > 1e6
    assert reduction.worse_than_reference is False
    assert reduction.mse_reduction_factor < 1.0
    assert reduction.worse_in_mse is True
    assert over.relative_bias == pytest.approx(-1.0, abs=1e-4)


def test_variance_reduction_refuses_a_degenerate_candidate():
    crude = _crude(5000, 5, seed=60)
    degenerate = summarise(
        "scale=-1",
        replicate(
            lambda rng: importance_sampling(
                STATE, scaled_mean_shift(STATE, -1.0), 5000, rng=rng
            ),
            5,
            seed=61,
        ),
        REFERENCE,
    )
    assert degenerate.empirical_std == 0.0
    assert degenerate.zero_estimate_count == 5
    with pytest.raises(ValueError, match="zero empirical spread"):
        variance_reduction(degenerate, crude, REFERENCE)


def test_variance_reduction_without_a_reference_probability_reports_nan():
    crude = _crude(5000, 5, seed=70)
    tilted = summarise(
        "analytic-IS",
        replicate(
            lambda rng: importance_sampling(
                STATE, analytic_mean_shift(STATE), 5000, rng=rng
            ),
            5,
            seed=71,
        ),
    )
    reduction = variance_reduction(tilted, crude)
    assert math.isnan(reduction.vrf_against_exact_crude)
    assert math.isnan(reduction.mse_reduction_factor)
    assert math.isnan(tilted.relative_bias)
    assert reduction.worse_in_mse is False
    assert "VRF" in reduction.describe()


def test_evaluation_rescaling_puts_methods_on_a_common_basis():
    """Candidate at half the evaluations must be compared at half the budget."""
    crude = _crude(20_000, 12, seed=80)
    tilted = summarise(
        "analytic-IS-10k",
        replicate(
            lambda rng: importance_sampling(
                STATE, analytic_mean_shift(STATE), 10_000, rng=rng
            ),
            12,
            seed=81,
        ),
        REFERENCE,
    )
    reduction = variance_reduction(tilted, crude, REFERENCE)
    assert reduction.equal_evaluation_basis == 10_000
    expected = (crude.empirical_std**2 * 2.0) / tilted.empirical_std**2
    assert reduction.vrf_measured == pytest.approx(expected, rel=1e-12)
