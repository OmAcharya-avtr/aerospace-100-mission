"""Tests for the estimate container and its intervals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rareverify.estimate import (
    RareEventEstimate,
    binomial_interval,
    bootstrap_interval,
    interval_for,
    normal_interval,
)


def _weighted(contributions: np.ndarray, n: int) -> RareEventEstimate:
    total = float(contributions.sum())
    estimate = total / n
    sum_sq = float((contributions**2).sum())
    variance = max(sum_sq / n - estimate**2, 0.0)
    ess = (total**2 / sum_sq) if sum_sq > 0.0 else 0.0
    return RareEventEstimate(
        method="importance-sampling",
        estimate=estimate,
        standard_error=math.sqrt(variance / n),
        n_samples=n,
        true_evaluations=n,
        n_failures=int(contributions.size),
        is_binomial=False,
        effective_sample_size=float(ess),
        wall_seconds=0.0,
        contributions=contributions,
    )


def test_binomial_interval_refuses_a_weighted_estimator():
    estimate = _weighted(np.full(50, 2.0e-3), 100_000)
    with pytest.raises(ValueError, match="not a binomial proportion"):
        binomial_interval(estimate)
    with pytest.raises(ValueError, match="not a binomial proportion"):
        binomial_interval(estimate, method="wilson")


def test_binomial_interval_rejects_an_unknown_method():
    estimate = RareEventEstimate(
        method="crude",
        estimate=1e-4,
        standard_error=1e-5,
        n_samples=10_000,
        true_evaluations=10_000,
        n_failures=1,
        is_binomial=True,
        effective_sample_size=1.0,
        wall_seconds=0.0,
    )
    with pytest.raises(ValueError, match="method must be one of"):
        binomial_interval(estimate, method="wald")


def test_normal_interval_hand_calculation():
    """estimate 1e-4, se 1e-5, 95 %: 1e-4 +- 1.959963985e-5."""
    lower, upper = normal_interval(1e-4, 1e-5, 0.95)
    assert lower == pytest.approx(1e-4 - 1.959963985e-5, rel=1e-9)
    assert upper == pytest.approx(1e-4 + 1.959963985e-5, rel=1e-9)


def test_normal_interval_clips_at_zero():
    lower, upper = normal_interval(1e-6, 1e-5, 0.95)
    assert lower == 0.0
    assert upper > 0.0


def test_normal_interval_validation():
    with pytest.raises(ValueError):
        normal_interval(1e-4, -1.0)
    with pytest.raises(ValueError):
        normal_interval(1e-4, 1e-5, confidence=1.0)


def test_bootstrap_interval_brackets_the_estimate():
    contributions = np.full(200, 5.0e-4)
    estimate = _weighted(contributions, 100_000)
    interval = bootstrap_interval(
        estimate, n_bootstrap=1000, rng=np.random.default_rng(1)
    )
    assert interval.kind == "bootstrap-weighted"
    assert interval.lower <= estimate.estimate <= interval.upper
    assert "percentile bootstrap" in interval.rationale


def test_bootstrap_interval_agrees_with_the_normal_interval_in_order_of_magnitude():
    rng = np.random.default_rng(2)
    contributions = rng.lognormal(mean=-8.0, sigma=0.5, size=500)
    estimate = _weighted(contributions, 200_000)
    boot = bootstrap_interval(estimate, n_bootstrap=2000, rng=np.random.default_rng(3))
    lower, upper = normal_interval(estimate.estimate, estimate.standard_error)
    assert 0.5 < boot.width / (upper - lower) < 2.0


def test_bootstrap_interval_on_an_empty_failure_set_is_degenerate_and_says_so():
    estimate = _weighted(np.zeros(0), 1000)
    interval = bootstrap_interval(estimate, n_bootstrap=200)
    assert interval.lower == interval.upper == 0.0
    assert "degenerate" in interval.rationale


def test_bootstrap_validation():
    estimate = _weighted(np.full(10, 1e-3), 1000)
    with pytest.raises(ValueError):
        bootstrap_interval(estimate, n_bootstrap=10)
    with pytest.raises(ValueError):
        bootstrap_interval(estimate, confidence=1.5)


def test_subset_standard_error_kind_is_flagged_in_the_rationale():
    estimate = RareEventEstimate(
        method="subset-simulation",
        estimate=1e-4,
        standard_error=1e-5,
        n_samples=8000,
        true_evaluations=7400,
        n_failures=200,
        is_binomial=False,
        effective_sample_size=math.nan,
        wall_seconds=0.0,
        standard_error_kind="subset-independence-lower-bound",
    )
    interval = interval_for(estimate)
    assert "LOWER BOUND" in interval.rationale
    assert interval.width > 0.0


def test_describe_and_coefficient_of_variation():
    estimate = _weighted(np.full(10, 1e-3), 1000)
    assert "importance-sampling" in estimate.describe()
    assert estimate.coefficient_of_variation > 0.0
    zero = _weighted(np.zeros(0), 1000)
    assert math.isinf(zero.coefficient_of_variation)
