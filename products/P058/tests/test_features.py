"""Feature tests, including the batch/online equality the benchmark depends on."""

from __future__ import annotations

import numpy as np
import pytest

from telemdrift.features import (
    FEATURE_NAMES,
    WINDOW,
    window_features,
    window_features_single,
)
from telemdrift.streams import stationary


def test_feature_names_are_the_documented_eight():
    assert len(FEATURE_NAMES) == 8
    assert FEATURE_NAMES[0] == "mean"
    assert FEATURE_NAMES[-1] == "range"


def test_window_features_shape_and_alignment():
    x = stationary(500, 61)
    f = window_features(x, 50)
    assert f.shape == (451, 8)
    # Row i comes from x[i : i+50], so row 0's mean is the mean of the first 50.
    assert f[0, 0] == pytest.approx(x[:50].mean())
    assert f[-1, 0] == pytest.approx(x[-50:].mean())


def test_batch_features_equal_single_window_features_exactly():
    """This equality is what licenses the batch score path in the benchmark.

    If it ever fails, every learned-detector ARL figure in this repository is
    measuring something other than the online detector.
    """
    x = stationary(400, 62)
    batch = window_features(x, WINDOW)
    worst = 0.0
    for i in range(0, batch.shape[0], 17):
        single = window_features_single(x[i : i + WINDOW])
        worst = max(worst, float(np.max(np.abs(batch[i] - single[0]))))
    assert worst == 0.0, f"worst absolute difference {worst:.3e}"


def test_mean_feature_tracks_a_shifted_window():
    f = window_features(np.full(60, 3.0), 50)
    assert np.allclose(f[:, 0], 3.0)
    assert np.allclose(f[:, 1], 0.0)


def test_half_mean_difference_is_positive_on_a_rising_window():
    """Window of 0 for the first 25 samples then 1 for the last 25:
    m1 = 0, m2 = 1, so half_mean_diff = m2 - m1 = 1."""
    w = np.concatenate([np.zeros(25), np.ones(25)])
    f = window_features_single(w)
    assert f[0, 2] == pytest.approx(1.0)


def test_slope_feature_recovers_an_exact_linear_trend():
    """x = 2 t over a window of 50 has least-squares slope exactly 2 per sample."""
    w = 2.0 * np.arange(50, dtype=float)
    f = window_features_single(w)
    assert f[0, 4] == pytest.approx(2.0)


def test_slope_feature_is_zero_on_a_constant_window():
    assert window_features_single(np.full(50, 5.0))[0, 4] == pytest.approx(0.0)


def test_log_half_std_ratio_is_zero_for_equal_halves_and_positive_for_a_louder_second_half():
    rng = np.random.default_rng(63)
    a = rng.standard_normal(50)
    f_sym = window_features_single(a)
    w = np.concatenate([a[:25], 4.0 * a[25:]])
    f_loud = window_features_single(w)
    assert abs(f_sym[0, 3]) < 1.0
    assert f_loud[0, 3] > 1.0


def test_lag1_autocorrelation_is_near_one_for_a_smooth_ramp():
    w = np.arange(50, dtype=float)
    assert window_features_single(w)[0, 5] > 0.99


def test_lag1_autocorrelation_is_near_minus_one_for_an_alternating_window():
    w = np.array([1.0, -1.0] * 25)
    assert window_features_single(w)[0, 5] < -0.9


def test_iqr_and_range_separate_a_single_outlier_from_a_scale_change():
    """The pair that the transient experiment turns on.

    A window with one extreme outlier has a large range and an unchanged IQR.
    A window whose scale has genuinely tripled has both.
    """
    rng = np.random.default_rng(64)
    base = rng.standard_normal(50)
    spiked = base.copy()
    spiked[25] = 40.0
    scaled = 3.0 * base
    f_base = window_features_single(base)[0]
    f_spike = window_features_single(spiked)[0]
    f_scale = window_features_single(scaled)[0]
    assert f_spike[7] > 5.0 * f_base[7]
    assert f_spike[6] == pytest.approx(f_base[6], rel=0.35)
    assert f_scale[6] > 2.0 * f_base[6]
    assert f_scale[7] > 2.0 * f_base[7]


def test_window_features_returns_empty_when_the_stream_is_shorter_than_the_window():
    assert window_features(np.zeros(10), 50).shape == (0, 8)


def test_window_features_rejects_a_two_dimensional_stream():
    with pytest.raises(ValueError, match="stream must be 1-D"):
        window_features(np.zeros((10, 2)), 4)


@pytest.mark.parametrize("w", [0, 1, 3])
def test_window_features_rejects_a_tiny_window(w):
    with pytest.raises(ValueError, match="window must be >= 4"):
        window_features(np.zeros(100), w)


def test_window_features_single_rejects_a_tiny_window():
    with pytest.raises(ValueError, match="window must be >= 4"):
        window_features_single(np.zeros(3))


def test_no_feature_is_nan_or_infinite_on_a_degenerate_constant_window():
    """A constant window has zero variance, which is where naive feature code
    produces NaN through a 0/0 autocorrelation or a log of zero."""
    f = window_features_single(np.zeros(50))
    assert np.isfinite(f).all()
