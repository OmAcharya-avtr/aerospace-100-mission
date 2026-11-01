"""Unit, known-answer, input-validation and edge-case tests for baseline 2."""

from __future__ import annotations

import math

import numpy as np
import pytest

from transfergap.resample import (
    bootstrap_mean_upper,
    conformal_order_index,
    conformal_required_n,
    split_conformal_quantile,
    split_conformal_upper,
)


@pytest.mark.parametrize(
    ("n", "alpha", "expected"),
    [
        (19, 0.05, 19),  # ceil(20 * 0.95) = ceil(19.0)  = 19
        (18, 0.05, 19),  # ceil(19 * 0.95) = ceil(18.05) = 19  > n: no limit exists
        (99, 0.05, 95),  # ceil(100 * 0.95) = 95
        (9, 0.10, 9),  # ceil(10 * 0.90) = 9
        (20, 0.50, 11),  # ceil(21 * 0.50) = ceil(10.5) = 11
    ],
)
def test_conformal_order_index_known_answers(n, alpha, expected):
    """Hand calculation of k = ceil((n+1)(1-alpha)) from equation (12)."""
    assert conformal_order_index(n, alpha) == expected


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [(0.05, 19), (0.10, 9), (0.01, 99), (0.20, 4)],
)
def test_conformal_required_n_known_answers(alpha, expected):
    """ceil(1/alpha) - 1: at alpha = 0.05 that is 19, so 15 field points is not enough."""
    assert conformal_required_n(alpha) == expected
    assert conformal_order_index(expected, alpha) <= expected
    assert conformal_order_index(expected - 1, alpha) > expected - 1


def test_split_conformal_quantile_is_an_order_statistic():
    """With n = 19 and alpha = 0.05, k = 19: the largest of the 19 scores.

    scores = [0, 1, ..., 18], so the 19th smallest is 18.0 exactly.
    """
    scores = np.arange(19.0)
    assert split_conformal_quantile(scores, 0.05) == 18.0


def test_split_conformal_quantile_picks_the_kth_smallest():
    """n = 99, alpha = 0.05 -> k = 95, so the 95th smallest of 0..98 is 94.0."""
    scores = np.arange(99.0)
    assert split_conformal_quantile(scores, 0.05) == 94.0


def test_split_conformal_quantile_is_permutation_invariant():
    """The order statistic does not depend on the order the scores arrive in."""
    rng = np.random.default_rng(7)
    scores = rng.normal(size=60)
    first = split_conformal_quantile(scores, 0.05)
    second = split_conformal_quantile(rng.permutation(scores), 0.05)
    assert first == second


def test_split_conformal_refuses_below_the_required_n():
    with pytest.raises(ValueError, match="needs n >= 19"):
        split_conformal_quantile(np.arange(15.0), 0.05)


def test_split_conformal_rejects_a_nan_score():
    with pytest.raises(ValueError, match="non-finite"):
        split_conformal_quantile(np.array([1.0, np.nan, 3.0] * 10), 0.05)


def test_split_conformal_upper_matches_the_order_statistic(field_sample):
    bound = split_conformal_upper(field_sample.gaps, field_sample.bound_range, 0.95, "kN.m")
    k = conformal_order_index(field_sample.size, 0.05)
    assert bound.upper == float(np.sort(field_sample.gaps)[k - 1])
    assert bound.target == "quantile"
    assert bound.distribution_free is True


def test_bootstrap_percentile_on_a_constant_sample_has_zero_slack():
    """Edge case: every resample of a constant sample has the same mean."""
    bound = bootstrap_mean_upper(np.full(30, 4.0), 10.0, 0.95, resamples=200, seed=1)
    assert bound.slack == pytest.approx(0.0, rel=0.0, abs=1e-14)
    assert bound.upper == pytest.approx(4.0, rel=0.0, abs=1e-14)


def test_bootstrap_is_reproducible_from_its_seed(field_sample):
    first = bootstrap_mean_upper(
        field_sample.gaps, field_sample.bound_range, 0.95, resamples=500, seed=42
    )
    second = bootstrap_mean_upper(
        field_sample.gaps, field_sample.bound_range, 0.95, resamples=500, seed=42
    )
    assert first.upper == second.upper


def test_bootstrap_basic_and_percentile_differ_on_a_skewed_sample(field_sample):
    """The two reflections of equation (14) and (15) are not the same number.

    A discrepancy magnitude is right-skewed, so the bootstrap distribution of
    its mean is too, and the reflected limit sits on the other side of the
    estimate. They agree only for a symmetric bootstrap distribution.
    """
    percentile = bootstrap_mean_upper(
        field_sample.gaps, field_sample.bound_range, 0.95, resamples=4000, seed=3,
        kind="percentile",
    )
    basic = bootstrap_mean_upper(
        field_sample.gaps, field_sample.bound_range, 0.95, resamples=4000, seed=3,
        kind="basic",
    )
    assert abs(percentile.upper - basic.upper) > 1e-3
    assert percentile.distribution_free is False
    assert basic.distribution_free is False


def test_bootstrap_rejects_too_few_resamples(field_sample):
    with pytest.raises(ValueError, match="at least 100"):
        bootstrap_mean_upper(field_sample.gaps, field_sample.bound_range, resamples=10)


def test_bootstrap_rejects_an_unknown_kind(field_sample):
    with pytest.raises(ValueError, match="percentile"):
        bootstrap_mean_upper(field_sample.gaps, field_sample.bound_range, kind="bca")


def test_bootstrap_slack_shrinks_like_one_over_root_n():
    """A scaling check against the asymptotic rate, not against this code.

    The bootstrap standard error of a mean is s / sqrt(n), so the one-sided
    percentile slack should fall by about sqrt(4) = 2 between n and 4n. The
    tolerance is wide because the comparison is asymptotic and the samples are
    different draws, not a resampling of one.
    """
    rng = np.random.default_rng(11)
    pool = np.abs(rng.normal(5.0, 2.0, size=4000))
    small = bootstrap_mean_upper(pool[:250], 40.0, 0.95, resamples=2000, seed=5)
    large = bootstrap_mean_upper(pool[:1000], 40.0, 0.95, resamples=2000, seed=5)
    ratio = small.slack / large.slack
    assert 1.5 < ratio < 2.7, f"ratio {ratio} is not near sqrt(4) = 2"


def test_conformal_order_index_rejects_n_below_one():
    with pytest.raises(ValueError, match="at least 1"):
        conformal_order_index(0, 0.05)


def test_conformal_coverage_bracket_holds_on_uniform_scores():
    """Equation (13) checked by Monte Carlo on a case where the truth is known.

    With n exchangeable Uniform(0, 1) calibration scores and one more draw,
    the coverage probability must lie in [1-alpha, 1-alpha+1/(n+1)]. The
    bracket is derived from the order-statistic argument of Lei et al. (2018),
    Theorem 2.2 -- mathematics this implementation does not use, since it only
    takes the k-th smallest value by integer index.

    n = 39, alpha = 0.05: the bracket is [0.95, 0.975]. At 20000 repetitions
    the binomial standard error of the measured rate is at most 0.0016, and
    the allowance below is four of those, 0.0064, which follows the Monte
    Carlo conditioning of the estimate rather than being an absolute constant.
    """
    rng = np.random.default_rng(2026)
    repetitions = 20000
    n = 39
    scores = rng.uniform(size=(repetitions, n))
    fresh = rng.uniform(size=repetitions)
    k = conformal_order_index(n, 0.05)
    limits = np.sort(scores, axis=1)[:, k - 1]
    rate = float(np.mean(fresh <= limits))
    allowance = 4.0 * math.sqrt(0.95 * 0.05 / repetitions)
    assert 0.95 - allowance <= rate <= 0.975 + allowance, rate
