"""Unit tests for the three conformal variants and the interval container."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conformalband.conformal import (
    ConformalizedRegressor,
    Interval,
    MondrianConformal,
    SplitConformal,
    WeightedConformal,
    absolute_residual_score,
    assign_bins,
    conformal_rank,
    tercile_edges,
    weighted_quantile,
)


@pytest.fixture
def calibration():
    rng = np.random.default_rng(57201)
    prediction = rng.normal(3.0, 0.4, 600)
    observed = prediction + rng.normal(0.0, 0.2, 600)
    return observed, prediction


def test_absolute_residual_score_values():
    got = absolute_residual_score(np.array([1.0, 2.0]), np.array([1.5, 1.0]))
    assert np.allclose(got, [0.5, 1.0])


def test_absolute_residual_score_is_non_negative(calibration):
    assert np.all(absolute_residual_score(*calibration) >= 0.0)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2, 0.5])
def test_split_quantile_is_a_calibration_score(alpha, calibration):
    model = SplitConformal(alpha).calibrate(*calibration)
    scores = absolute_residual_score(*calibration)
    assert np.any(np.isclose(scores, model.quantile))


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2])
def test_split_quantile_decreases_with_alpha(alpha, calibration):
    tighter = SplitConformal(alpha).calibrate(*calibration).quantile
    looser = SplitConformal(min(alpha * 2.0, 0.9)).calibrate(*calibration).quantile
    assert looser <= tighter


def test_split_interval_is_symmetric(calibration):
    model = SplitConformal(0.1).calibrate(*calibration)
    interval = model.interval(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(interval.point - interval.lower, interval.upper - interval.point)


def test_split_width_is_twice_the_quantile(calibration):
    model = SplitConformal(0.1).calibrate(*calibration)
    interval = model.interval(np.array([2.0]))
    assert float(interval.width[0]) == pytest.approx(2.0 * model.quantile, rel=1e-15)


def test_split_empirical_coverage_near_nominal(calibration):
    observed, prediction = calibration
    model = SplitConformal(0.1).calibrate(observed[:500], prediction[:500])
    interval = model.interval(prediction[500:])
    assert 0.80 <= float(interval.covers(observed[500:]).mean()) <= 1.0


def test_split_n_calibration(calibration):
    assert SplitConformal(0.1).calibrate(*calibration).n_calibration == 600


def test_split_coverage_bound_is_attached(calibration):
    model = SplitConformal(0.1).calibrate(*calibration)
    bound = model.coverage_bound
    assert bound.n_calibration == 600
    assert bound.lower == pytest.approx(0.9, rel=1e-15)


def test_split_is_calibrated_flag():
    model = SplitConformal(0.1)
    assert not model.is_calibrated
    model.calibrate(np.zeros(50), np.arange(50.0))
    assert model.is_calibrated


def test_split_refuses_quantile_before_calibration():
    with pytest.raises(RuntimeError, match="calibrate"):
        _ = SplitConformal(0.1).quantile


def test_split_refuses_n_calibration_before_calibration():
    with pytest.raises(RuntimeError, match="calibrate"):
        _ = SplitConformal(0.1).n_calibration


def test_mondrian_quantiles_per_bin(calibration):
    observed, prediction = calibration
    edges = tercile_edges(prediction)
    bins = assign_bins(prediction, edges)
    model = MondrianConformal(0.1).calibrate(observed, prediction, bins)
    assert set(model.quantiles) == {0, 1, 2}
    assert sum(model.counts.values()) == 600


def test_mondrian_counts_are_roughly_balanced(calibration):
    observed, prediction = calibration
    bins = assign_bins(prediction, tercile_edges(prediction))
    counts = MondrianConformal(0.1).calibrate(observed, prediction, bins).counts
    assert max(counts.values()) - min(counts.values()) <= 2


def test_mondrian_interval_uses_the_right_bin(calibration):
    observed, prediction = calibration
    edges = tercile_edges(prediction)
    bins = assign_bins(prediction, edges)
    model = MondrianConformal(0.1).calibrate(observed, prediction, bins)
    test_prediction = np.array([edges[0] - 1.0, edges[0] + 1e-9, edges[1] + 1.0])
    interval = model.interval(test_prediction, assign_bins(test_prediction, edges))
    widths = interval.width / 2.0
    assert widths[0] == pytest.approx(model.quantiles[0], rel=0.0)
    assert widths[2] == pytest.approx(model.quantiles[2], rel=0.0)


def test_mondrian_rejects_unseen_bin(calibration):
    observed, prediction = calibration
    bins = np.zeros(600, dtype=int)
    model = MondrianConformal(0.1).calibrate(observed, prediction, bins)
    with pytest.raises(KeyError, match="no calibration quantile"):
        model.interval(np.array([1.0]), np.array([9]))


def test_mondrian_rejects_tiny_bin(calibration):
    observed, prediction = calibration
    bins = np.zeros(600, dtype=int)
    bins[:3] = 1
    with pytest.raises(ValueError, match="too few"):
        MondrianConformal(0.1).calibrate(observed, prediction, bins)


def test_mondrian_coverage_bounds_per_bin(calibration):
    observed, prediction = calibration
    bins = assign_bins(prediction, tercile_edges(prediction))
    bounds = MondrianConformal(0.1).calibrate(observed, prediction, bins).coverage_bounds()
    assert set(bounds) == {0, 1, 2}
    assert all(b.lower == pytest.approx(0.9, rel=1e-15) for b in bounds.values())


def test_mondrian_refuses_quantiles_before_calibration():
    with pytest.raises(RuntimeError, match="calibrate"):
        _ = MondrianConformal(0.1).quantiles


def test_weighted_quantiles_are_finite_for_moderate_weights(calibration):
    observed, prediction = calibration
    weights = np.full(600, 1.0)
    model = WeightedConformal(0.1).calibrate(observed, prediction, weights)
    assert np.all(np.isfinite(model.quantiles(np.full(10, 1.0))))


def test_weighted_quantile_is_infinite_for_a_dominant_test_weight(calibration):
    observed, prediction = calibration
    model = WeightedConformal(0.1).calibrate(observed, prediction, np.full(600, 1.0))
    # tail mass = w / (600 + w) > 0.1 requires w > 66.67.
    got = model.quantiles(np.array([10.0, 100.0]))
    assert np.isfinite(got[0])
    assert math.isinf(got[1])


def test_weighted_quantile_grows_with_test_weight(calibration):
    observed, prediction = calibration
    model = WeightedConformal(0.1).calibrate(observed, prediction, np.full(600, 1.0))
    got = model.quantiles(np.array([0.1, 1.0, 10.0, 50.0]))
    assert np.all(np.diff(got) >= 0.0)


def test_weighted_effective_sample_size(calibration):
    observed, prediction = calibration
    model = WeightedConformal(0.1).calibrate(observed, prediction, np.full(600, 3.0))
    assert model.effective_sample_size == pytest.approx(600.0, rel=1e-12)


def test_weighted_weight_sum(calibration):
    observed, prediction = calibration
    model = WeightedConformal(0.1).calibrate(observed, prediction, np.full(600, 0.5))
    assert model.weight_sum == pytest.approx(300.0, rel=1e-13)


def test_weighted_refuses_before_calibration():
    with pytest.raises(RuntimeError, match="calibrate"):
        WeightedConformal(0.1).quantiles(np.ones(3))


def test_interval_width_and_coverage():
    interval = Interval(
        lower=np.array([0.0, 1.0]), point=np.array([1.0, 2.0]), upper=np.array([2.0, 3.0])
    )
    assert np.allclose(interval.width, [2.0, 2.0])
    assert np.array_equal(interval.covers(np.array([0.5, 9.0])), [True, False])


def test_interval_covers_the_endpoints():
    interval = Interval(lower=np.array([1.0]), point=np.array([2.0]), upper=np.array([3.0]))
    assert bool(interval.covers(np.array([1.0]))[0])
    assert bool(interval.covers(np.array([3.0]))[0])


def test_interval_rejects_mismatched_truth():
    interval = Interval(lower=np.zeros(2), point=np.ones(2), upper=np.full(2, 2.0))
    with pytest.raises(ValueError, match="same shape"):
        interval.covers(np.zeros(3))


def test_infinite_interval_has_infinite_width():
    interval = Interval(
        lower=np.array([-np.inf]), point=np.array([1.0]), upper=np.array([np.inf])
    )
    assert math.isinf(float(interval.width[0]))
    assert bool(interval.covers(np.array([1e9]))[0])


def test_tercile_edges_are_increasing(calibration):
    edges = tercile_edges(calibration[1])
    assert edges[0] < edges[1]


def test_tercile_edges_split_into_thirds(calibration):
    bins = assign_bins(calibration[1], tercile_edges(calibration[1]))
    counts = np.bincount(bins, minlength=3)
    assert np.all(np.abs(counts - 200) <= 2)


def test_tercile_edges_rejects_short_input():
    with pytest.raises(ValueError, match="at least 3"):
        tercile_edges(np.array([1.0, 2.0]))


def test_assign_bins_boundary_goes_high():
    # searchsorted side="right" puts a value exactly on an edge in the UPPER bin,
    # so the bins are [-inf, e0), [e0, e1), [e1, inf).
    assert int(assign_bins(np.array([1.0]), np.array([1.0, 2.0]))[0]) == 1
    assert int(assign_bins(np.array([0.999]), np.array([1.0, 2.0]))[0]) == 0
    assert int(assign_bins(np.array([2.0]), np.array([1.0, 2.0]))[0]) == 2


def test_assign_bins_rejects_unsorted_edges():
    with pytest.raises(ValueError, match="strictly increasing"):
        assign_bins(np.array([1.0]), np.array([2.0, 1.0]))


def test_assign_bins_rejects_empty_edges():
    with pytest.raises(ValueError, match="non-empty"):
        assign_bins(np.array([1.0]), np.array([]))


def test_conformalized_regressor_round_trip(calibration):
    observed, prediction = calibration

    class Identity:
        def predict(self, features):
            return np.asarray(features, dtype=float).ravel()

    model = SplitConformal(0.1).calibrate(observed, prediction)
    wrapped = ConformalizedRegressor(Identity(), model)
    interval = wrapped.predict_interval(np.array([3.0, 3.5]))
    assert np.allclose(interval.point, [3.0, 3.5])
    assert np.allclose(interval.width, 2.0 * model.quantile)


def test_conformalized_regressor_refuses_uncalibrated():
    class Identity:
        def predict(self, features):
            return np.asarray(features, dtype=float).ravel()

    with pytest.raises(ValueError, match="calibrated"):
        ConformalizedRegressor(Identity(), SplitConformal(0.1))


@pytest.mark.parametrize("level", [0.1, 0.5, 0.9, 0.99])
def test_weighted_quantile_monotone_in_level(level):
    values = np.arange(1.0, 101.0)
    weights = np.ones(100)
    low = weighted_quantile(values, weights, level)
    high = weighted_quantile(values, weights, min(level + 0.009, 0.999))
    assert high >= low


@pytest.mark.parametrize("n", [9, 19, 50, 101, 500])
def test_conformal_rank_is_at_most_n(n):
    assert conformal_rank(n, 0.1) <= n
