"""Bootstrap intervals, including the granularity of a three-run cluster bootstrap."""

from __future__ import annotations

import math

import numpy as np
import pytest

from thermalbudget.bootstrap import (
    bootstrap_cluster_mean,
    bootstrap_cluster_ratio,
    bootstrap_mean,
    cluster_resampled_means,
    observed_cluster_mean,
    percentile_interval,
    t_interval_of_group_means,
)


def test_bootstrap_mean_point_is_the_observed_mean_not_the_resample_mean():
    sample = np.array([1.0, 2.0, 3.0, 4.0])
    result = bootstrap_mean(sample, n_resamples=2000, seed=1)
    assert result.point == pytest.approx(2.5, rel=0.0, abs=1e-12)
    assert result.ci_low < 2.5 < result.ci_high
    assert abs(result.bias) < 0.2


def test_bootstrap_of_an_identical_sample_has_zero_width():
    result = bootstrap_mean(np.full(50, 7.0), n_resamples=500, seed=2)
    assert result.ci_low == pytest.approx(7.0, rel=0.0, abs=1e-12)
    assert result.ci_high == pytest.approx(7.0, rel=0.0, abs=1e-12)
    assert result.standard_error == pytest.approx(0.0, rel=0.0, abs=1e-12)


def test_bootstrap_mean_is_reproducible_from_the_seed():
    sample = np.linspace(0.0, 1.0, 40)
    first = bootstrap_mean(sample, n_resamples=1000, seed=11)
    second = bootstrap_mean(sample, n_resamples=1000, seed=11)
    assert first.ci_low == second.ci_low
    assert first.ci_high == second.ci_high


def test_bootstrap_mean_refuses_fewer_than_two_samples():
    with pytest.raises(ValueError, match="at least 2 finite samples"):
        bootstrap_mean([1.0])
    with pytest.raises(ValueError, match="at least 2 finite samples"):
        bootstrap_mean([math.nan, math.nan, 1.0])


def test_cluster_bootstrap_reports_its_own_granularity():
    rng = np.random.default_rng(3)
    groups = [rng.normal(loc, 1.0, size=100) for loc in (10.0, 10.5, 9.5)]
    result = bootstrap_cluster_mean(groups, n_resamples=4000, seed=5)
    # Three groups means at most 3**3 = 27 distinct multisets at the run level,
    # however many resamples are drawn. That number is reported.
    assert result.n_distinct_group_draws == 27
    assert result.n_groups == 3
    assert result.n_samples == 300
    assert result.point == pytest.approx(10.0, rel=0.0, abs=0.3)


def test_cluster_bootstrap_is_wider_than_the_pooled_single_level_one():
    # Between-group spread is the point: pooling it away understates the interval.
    rng = np.random.default_rng(7)
    groups = [rng.normal(loc, 0.2, size=200) for loc in (10.0, 12.0, 8.0)]
    clustered = bootstrap_cluster_mean(groups, n_resamples=4000, seed=5)
    pooled = bootstrap_mean(np.concatenate(groups), n_resamples=4000, seed=5)
    assert clustered.half_width > 3.0 * pooled.half_width


def test_cluster_bootstrap_refuses_one_group():
    with pytest.raises(ValueError, match="at least 2 non-empty groups"):
        bootstrap_cluster_mean([np.arange(10.0)])
    with pytest.raises(ValueError, match="at least 2 non-empty groups"):
        bootstrap_cluster_mean([np.arange(10.0), np.array([])])


def test_t_interval_is_an_independent_route_and_is_wider_for_three_groups():
    groups = [np.full(100, value) for value in (10.0, 12.0, 8.0)]
    point, low, high = t_interval_of_group_means(groups)
    assert point == pytest.approx(10.0, rel=0.0, abs=1e-12)
    # s = 2.0 over three means, t(0.975, 2) = 4.302652729911275,
    # half width = 4.302652729911275 * 2 / sqrt(3) = 4.968...
    half = 4.302652729911275 * 2.0 / math.sqrt(3.0)
    assert high - point == pytest.approx(half, rel=1e-9, abs=0.0)
    assert point - low == pytest.approx(half, rel=1e-9, abs=0.0)


def test_t_interval_refuses_one_group():
    with pytest.raises(ValueError, match="at least 2 groups"):
        t_interval_of_group_means([np.arange(5.0)])


def test_cluster_ratio_of_identical_sides_is_one():
    rng = np.random.default_rng(9)
    groups = [rng.normal(100.0, 1.0, size=80) for _ in range(3)]
    result = bootstrap_cluster_ratio(groups, groups, n_resamples=3000, seed=13)
    assert result.point == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert result.contains(1.0)


def test_cluster_ratio_refuses_a_single_group_side():
    groups = [np.full(10, 1.0), np.full(10, 1.0)]
    with pytest.raises(ValueError, match="numerator side"):
        bootstrap_cluster_ratio([np.full(10, 1.0)], groups)
    with pytest.raises(ValueError, match="denominator side"):
        bootstrap_cluster_ratio(groups, [np.full(10, 1.0)])


def test_resampled_means_and_observed_mean_agree_in_expectation():
    rng = np.random.default_rng(17)
    groups = [rng.normal(5.0, 0.5, size=120) for _ in range(3)]
    draws = cluster_resampled_means(groups, n_resamples=4000, seed=23)
    assert draws.size == 4000
    assert float(draws.mean()) == pytest.approx(
        observed_cluster_mean(groups), rel=0.0, abs=0.05
    )
    low, high = percentile_interval(draws)
    assert low < observed_cluster_mean(groups) < high


def test_observed_cluster_mean_is_unweighted_over_group_means():
    # Unweighted by construction: a 1000-sample group and a 2-sample group count
    # the same, because the quantity is the mean of per-run means.
    groups = [np.full(1000, 0.0), np.full(2, 10.0)]
    assert observed_cluster_mean(groups) == pytest.approx(5.0, rel=0.0, abs=1e-12)
    with pytest.raises(ValueError, match="empty input set"):
        observed_cluster_mean([])
