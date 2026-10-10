"""Edge cases: degenerate inputs, boundary sizes and numerical extremes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conformalband.baseline import GaussianResidualInterval
from conformalband.bounds import effective_sample_size, split_conformal_coverage_bound
from conformalband.conformal import (
    MondrianConformal,
    SplitConformal,
    WeightedConformal,
    assign_bins,
    weighted_quantile,
)
from conformalband.data import make_dataset
from conformalband.shift import CovariateShift


def test_split_with_identical_scores_gives_that_score():
    model = SplitConformal(0.1).calibrate(np.full(50, 2.0), np.full(50, 2.0))
    assert model.quantile == 0.0


def test_split_with_zero_residuals_gives_a_degenerate_interval():
    model = SplitConformal(0.1).calibrate(np.zeros(50), np.zeros(50))
    interval = model.interval(np.array([1.0]))
    assert float(interval.width[0]) == 0.0
    assert bool(interval.covers(np.array([1.0]))[0])


def test_split_at_the_minimum_calibration_size():
    rng = np.random.default_rng(58001)
    y = rng.normal(0.0, 1.0, 9)
    model = SplitConformal(0.1).calibrate(y, np.zeros(9))
    # k = ceil(10 * 0.9) = 9, the largest of the nine scores.
    assert model.quantile == pytest.approx(float(np.max(np.abs(y))), rel=0.0)


def test_split_with_alpha_one_half():
    rng = np.random.default_rng(58002)
    y = rng.normal(0.0, 1.0, 100)
    model = SplitConformal(0.5).calibrate(y, np.zeros(100))
    # k = ceil(101 * 0.5) = 51, so just over the median absolute residual.
    assert model.quantile == pytest.approx(float(np.sort(np.abs(y))[50]), rel=0.0)


def test_mondrian_with_many_small_bins_is_refused():
    rng = np.random.default_rng(58003)
    y = rng.normal(0.0, 1.0, 100)
    bins = np.arange(100) % 20  # five points per bin
    with pytest.raises(ValueError, match="too few"):
        MondrianConformal(0.1).calibrate(y, np.zeros(100), bins)


def test_mondrian_with_exactly_nine_points_per_bin_is_accepted():
    rng = np.random.default_rng(58004)
    y = rng.normal(0.0, 1.0, 27)
    bins = np.repeat(np.arange(3), 9)
    model = MondrianConformal(0.1).calibrate(y, np.zeros(27), bins)
    assert model.counts == {0: 9, 1: 9, 2: 9}


def test_weighted_with_one_dominant_calibration_weight():
    rng = np.random.default_rng(58005)
    y = rng.normal(0.0, 1.0, 200)
    weights = np.full(200, 1e-9)
    weights[0] = 1.0
    model = WeightedConformal(0.1).calibrate(y, np.zeros(200), weights)
    assert model.effective_sample_size == pytest.approx(1.0, abs=1e-3)


def test_weighted_quantile_with_a_single_point():
    assert weighted_quantile(np.array([4.0]), np.array([1.0]), 0.5) == 4.0


def test_weighted_quantile_with_a_zero_weight_point_skips_it():
    # The heavy small value carries all the mass, so the quantile stays low.
    values = np.array([1.0, 99.0])
    weights = np.array([1.0, 0.0])
    assert weighted_quantile(values, weights, 0.9) == 1.0


def test_weighted_all_test_weights_huge_gives_all_infinite():
    rng = np.random.default_rng(58006)
    y = rng.normal(0.0, 1.0, 100)
    model = WeightedConformal(0.1).calibrate(y, np.zeros(100), np.ones(100))
    got = model.quantiles(np.full(5, 1e6))
    assert np.all(np.isinf(got))


def test_bound_at_the_exact_minimum_size():
    bound = split_conformal_coverage_bound(9, 0.1)
    assert bound.rank == 9
    assert bound.exact == pytest.approx(0.9, rel=1e-15)
    assert bound.conservatism == pytest.approx(0.0, abs=1e-15)


def test_bound_for_a_very_large_calibration_size():
    bound = split_conformal_coverage_bound(1_000_000, 0.1)
    assert abs(bound.exact - 0.9) < 1e-6


def test_effective_sample_size_of_a_single_weight():
    assert effective_sample_size(np.array([5.0])) == pytest.approx(1.0, rel=1e-15)


def test_assign_bins_handles_values_below_and_above_all_edges():
    got = assign_bins(np.array([-1e9, 1e9]), np.array([0.0, 1.0]))
    assert int(got[0]) == 0
    assert int(got[1]) == 2


def test_gaussian_interval_with_a_single_residual():
    model = GaussianResidualInterval(0.1).fit(np.array([2.0]), np.array([0.0]))
    assert model.sigma == pytest.approx(2.0, rel=1e-15)


def test_gaussian_interval_with_zero_residuals_is_degenerate():
    model = GaussianResidualInterval(0.1).fit(np.zeros(10), np.zeros(10))
    assert model.sigma == 0.0
    assert model.half_width == 0.0


def test_extreme_severity_makes_the_likelihood_ratio_huge_but_finite():
    shift = CovariateShift(severity=20.0)
    ratio = shift.likelihood_ratio(np.array([9.0]), np.array([8.0]))
    assert np.all(np.isfinite(ratio))
    assert float(ratio[0]) > 1e3


def test_overflowing_likelihood_ratio_is_refused_not_silently_used():
    # log w = delta (x - mu) / sd^2 - delta^2 / (2 sd^2) is affine in mass, so a
    # far enough covariate value overflows the exponential. At severity 10 and
    # mass 400 kg (a numerical probe, not a physical airframe state),
    # log w = 0.5 * 10 * (400 - 6) - 0.045 * 100 = 1965, well past exp's limit
    # of about 709. The weighted calibrator must refuse the infinite weight
    # rather than return an interval built from it.
    shift = CovariateShift(severity=10.0)
    with np.errstate(over="ignore"):
        ratio = shift.likelihood_ratio(np.full(20, 400.0), np.zeros(20))
    assert math.isinf(float(ratio[0]))
    with pytest.raises(ValueError, match="finite"):
        WeightedConformal(0.1).calibrate(np.zeros(20), np.zeros(20), ratio)


def test_dataset_of_one_row():
    data = make_dataset(1, seed=58007)
    assert len(data) == 1
    assert data.features.shape == (1, 5)


def test_zero_noise_dataset_makes_conformal_width_zero():
    data = make_dataset(50, seed=58008, noise_fraction=0.0)
    model = SplitConformal(0.1).calibrate(data.energy, data.truth)
    assert model.quantile == pytest.approx(0.0, abs=1e-12)
