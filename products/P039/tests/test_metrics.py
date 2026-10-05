"""Coverage, accuracy and the paired comparison test."""

from __future__ import annotations

import math

import numpy as np
import pytest

from latencynet.metrics import (
    interval_coverage,
    log_accuracy,
    paired_difference_test,
    wilson_interval,
)
from latencynet.predictors import IntervalPrediction


def _interval(point, half, level=0.9):
    point = np.asarray(point, dtype=float)
    return IntervalPrediction(point, point - half, point + half, level)


def test_coverage_known_answer():
    # Ten pipelines, truth offset by 0.5 for three of them and 0.05 for the
    # rest; a half-width of 0.1 therefore covers exactly 7 of 10.
    point = np.zeros(10)
    truth = np.array([0.5, 0.5, 0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    result = interval_coverage(_interval(point, 0.1), truth)
    assert result.n == 10
    assert result.n_covered == 7
    assert result.measured == pytest.approx(0.7, rel=1e-15)
    # Hand-calculated SE: sqrt(0.7 * 0.3 / 10) = sqrt(0.021) = 0.144913767...
    assert result.standard_error == pytest.approx(0.1449137674618944, rel=1e-12)
    assert result.mean_log_width == pytest.approx(0.2, rel=1e-14)
    assert result.median_log_width == pytest.approx(0.2, rel=1e-14)
    assert result.nominal == 0.9


def test_coverage_standard_error_at_nominal():
    # Hand-calculated: c = 0.9, n = 100 -> sqrt(0.9 * 0.1 / 100) = 0.03.
    point = np.zeros(100)
    truth = np.concatenate([np.full(90, 0.05), np.full(10, 0.5)])
    result = interval_coverage(_interval(point, 0.1), truth)
    assert result.measured == pytest.approx(0.9, rel=1e-15)
    assert result.standard_error == pytest.approx(0.03, rel=1e-12)
    assert result.deviation_sigma == pytest.approx(0.0, abs=1e-12)


def test_wilson_interval_known_answer():
    # Hand-calculated for 5/10 at z = 1:
    #   denom  = 1 + 1/10 = 1.1
    #   centre = (0.5 + 0.05) / 1.1 = 0.5
    #   half   = (1 / 1.1) * sqrt(0.25 / 10 + 1 / 400)
    #          = 0.909090909 * sqrt(0.0275) = 0.150755672
    lo, hi = wilson_interval(5, 10, z=1.0)
    assert lo == pytest.approx(0.3492443279, abs=1e-9)
    assert hi == pytest.approx(0.6507556721, abs=1e-9)


def test_wilson_interval_is_inside_the_unit_interval():
    for k in (0, 1, 50, 99, 100):
        lo, hi = wilson_interval(k, 100)
        assert 0.0 <= lo <= hi <= 1.0


def test_log_accuracy_known_answer():
    # Errors -0.1, 0.0, 0.3: mean |e| = 0.4/3 = 0.13333..., median 0.1,
    # rms = sqrt((0.01 + 0 + 0.09) / 3) = sqrt(0.0333...) = 0.18257419,
    # bias = 0.2/3 = 0.0666...
    pred = np.array([-0.1, 0.0, 0.3])
    result = log_accuracy(pred, np.zeros(3))
    assert result.mean_abs_log_error == pytest.approx(0.4 / 3.0, rel=1e-13)
    assert result.median_abs_log_error == pytest.approx(0.1, rel=1e-13)
    assert result.rms_log_error == pytest.approx(math.sqrt(0.1 / 3.0), rel=1e-13)
    assert result.bias_log == pytest.approx(0.2 / 3.0, rel=1e-13)
    assert result.n == 3
    assert result.mean_relative_error == pytest.approx(math.exp(0.4 / 3.0) - 1.0, rel=1e-13)


def test_paired_difference_test_detects_a_clear_winner():
    rng = np.random.default_rng(1)
    better = rng.normal(scale=0.01, size=80)
    worse = rng.normal(scale=0.10, size=80)
    mean_diff, t_stat, p_value = paired_difference_test(better, worse)
    assert mean_diff < 0.0
    assert t_stat < 0.0
    assert p_value < 1e-6


def test_paired_difference_test_on_identical_inputs():
    e = np.array([0.1, -0.2, 0.3, 0.05])
    mean_diff, _, p_value = paired_difference_test(e, e)
    assert mean_diff == 0.0
    assert math.isnan(p_value) or p_value > 0.99


def test_interval_prediction_validation():
    with pytest.raises(ValueError, match="share a shape"):
        IntervalPrediction(np.zeros(3), np.zeros(2), np.zeros(3), 0.9)
    with pytest.raises(ValueError, match="level"):
        IntervalPrediction(np.zeros(3), np.zeros(3), np.zeros(3), 1.0)


def test_metric_validation():
    with pytest.raises(ValueError, match="has shape"):
        interval_coverage(_interval(np.zeros(3), 0.1), np.zeros(4))
    with pytest.raises(ValueError, match="shapes differ"):
        log_accuracy(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="non-empty"):
        log_accuracy(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="at least 3"):
        paired_difference_test(np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="n must be"):
        wilson_interval(0, 0)
    with pytest.raises(ValueError, match="n_success"):
        wilson_interval(11, 10)


def test_interval_seconds_round_trip():
    pred = _interval(np.log(np.array([1e-3, 2e-3])), 0.1)
    lo, pt, hi = pred.seconds()
    assert np.allclose(pt, [1e-3, 2e-3], rtol=1e-14)
    assert np.all(lo < pt) and np.all(pt < hi)
    assert np.allclose(pred.log_width, 0.2, rtol=1e-14)
