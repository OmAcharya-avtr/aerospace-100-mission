"""Known-answer tests for the nine windowed residual features."""

from __future__ import annotations

import numpy as np
import pytest

from twininvalidate import FEATURE_NAMES, N_FEATURES, window_features
from twininvalidate.features import flatten_features


def test_feature_names_and_count():
    assert N_FEATURES == 9
    assert FEATURE_NAMES[0] == "mean"
    assert FEATURE_NAMES[-1] == "mean_square"
    assert len(set(FEATURE_NAMES)) == 9


def test_known_answer_linear_ramp():
    # z = 0, 1, 2, ..., 7 over a single window of 8. Hand calculation:
    #   mean        = 28/8 = 3.5
    #   deviations  = -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5
    #   var (pop.)  = 2*(12.25 + 6.25 + 2.25 + 0.25)/8 = 42/8 = 5.25
    #   std         = sqrt(5.25) = 2.2912878475
    #   mean |z|    = 3.5 (all non-negative)
    #   max |z|     = 7
    #   lag-1 autocorr numerator = sum of d_i d_{i+1}
    #     = 8.75 + 3.75 + 0.75 - 0.25 + 0.75 + 3.75 + 8.75 = 26.25
    #     mean over 7 lags = 3.75; divided by 5.25 gives 5/7 = 0.7142857143
    #   m4 = 2*(150.0625 + 39.0625 + 5.0625 + 0.0625)/8 = 388.5/8 = 48.5625
    #     48.5625 / 5.25^2 = 48.5625 / 27.5625 = 37/21 = 1.7619047619
    #     excess kurtosis = 37/21 - 3 = -26/21 = -1.2380952381
    #   trend = OLS slope (1.0) times window (8) = 8.0
    #   fraction |z| > 2 = |{3,4,5,6,7}| / 8 = 0.625
    #   mean z^2 = 140/8 = 17.5
    f = window_features(np.arange(8, dtype=float).reshape(1, -1), window=8)[0, 0]
    assert f[0] == pytest.approx(3.5)
    assert f[1] == pytest.approx(2.2912878474779199, rel=1e-12)
    assert f[2] == pytest.approx(3.5)
    assert f[3] == pytest.approx(7.0)
    assert f[4] == pytest.approx(5.0 / 7.0, rel=1e-12)
    assert f[5] == pytest.approx(-26.0 / 21.0, rel=1e-12)
    assert f[6] == pytest.approx(8.0, rel=1e-12)
    assert f[7] == pytest.approx(0.625)
    assert f[8] == pytest.approx(17.5)


def test_known_answer_constant_stream():
    # A constant c = 3 over a window of 5: mean 3, std 0, mean |z| 3, max 3,
    # mean z^2 9, trend 0, fraction |z| > 2 is 1. The autocorrelation and the
    # kurtosis are undefined for zero variance and are defined here to be 0.
    f = window_features(np.full((1, 5), 3.0), window=5)[0, 0]
    assert f.tolist() == pytest.approx([3.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 1.0, 9.0])


def test_known_answer_alternating_stream():
    # z = 1, -1, 1, -1 over a window of 4:
    #   mean 0, var 1, std 1, mean |z| 1, max |z| 1, mean z^2 1
    #   lag-1 numerator = (-1 -1 -1)/3 = -1, divided by 1 gives -1
    #   m4 = 1 so kurtosis = 1/1 - 3 = -2
    #   trend: t = -1.5, -0.5, 0.5, 1.5; sum(d t) = -1.5 + 0.5 + 0.5 - 1.5 = -2
    #     sum(t^2) = 5, slope = -0.4, trend = -0.4 * 4 = -1.6
    #   fraction |z| > 2 = 0
    f = window_features(np.array([[1.0, -1.0, 1.0, -1.0]]), window=4)[0, 0]
    assert f.tolist() == pytest.approx([0.0, 1.0, 1.0, 1.0, -1.0, -2.0, -1.6, 0.0, 1.0])


def test_shape_and_alignment():
    z = np.arange(20, dtype=float).reshape(1, -1)
    f = window_features(z, window=5)
    assert f.shape == (1, 16, N_FEATURES)
    # Window j covers samples j .. j+4, so its mean is j + 2.
    assert f[0, :, 0] == pytest.approx(np.arange(16) + 2.0)


def test_features_of_in_control_data_are_near_their_expectations(in_control):
    f = window_features(in_control[:40, :600], window=50).reshape(-1, N_FEATURES)
    assert abs(f[:, 0].mean()) < 0.02  # mean
    assert f[:, 1].mean() == pytest.approx(0.99, abs=0.03)  # std, slight bias for ddof=0
    assert f[:, 8].mean() == pytest.approx(1.0, abs=0.03)  # mean z^2
    assert abs(f[:, 4].mean()) < 0.05  # lag-1 autocorrelation
    assert abs(f[:, 6].mean()) < 0.1  # trend


def test_flatten_features_round_trips():
    f = window_features(np.arange(20, dtype=float).reshape(2, 10), window=4)
    flat = flatten_features(f)
    assert flat.shape == (f.shape[0] * f.shape[1], N_FEATURES)
    assert np.allclose(flat.reshape(f.shape), f)


def test_one_dimensional_input_is_accepted():
    f = window_features(np.arange(10, dtype=float), window=4)
    assert f.shape == (1, 7, N_FEATURES)


@pytest.mark.parametrize(
    "z, window, message",
    [
        (np.zeros((1, 10)), 3, "window must be at least 4"),
        (np.zeros((1, 10)), 11, "exceeds stream length"),
        (np.zeros((2, 2, 2)), 4, "must be 1-D or 2-D"),
    ],
)
def test_window_features_validates_its_inputs(z, window, message):
    with pytest.raises(ValueError, match=message):
        window_features(z, window)


def test_flatten_features_validates_its_input():
    with pytest.raises(ValueError, match="expected shape"):
        flatten_features(np.zeros((2, 3)))
