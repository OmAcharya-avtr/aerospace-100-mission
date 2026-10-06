"""LLR tests: the AWGN known-answer limit, quadrature agreement, max-log bound."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from softdecode.channel import LognormalFading, amplitude_quadrature
from softdecode.llr import (
    clip_llr,
    gray_labels,
    llr_ook_known_csi,
    llr_ook_marginal,
    llr_ook_maxlog,
    llr_ppm_known_csi,
    llr_ppm_marginal,
    llr_ppm_maxlog,
    rescale_llr,
)

Y = np.linspace(-3.0, 9.0, 25)


def test_known_csi_llr_hand_calculation():
    # a = 2, h = 1, sigma = 1, y = 0:  (4 - 0) / 2 = 2.0
    # y = 1: (4 - 4) / 2 = 0.0, which is the OOK decision threshold a h / 2.
    assert llr_ook_known_csi(0.0, 2.0, 1.0, 1.0) == pytest.approx(2.0)
    assert llr_ook_known_csi(1.0, 2.0, 1.0, 1.0) == pytest.approx(0.0)
    assert llr_ook_known_csi(2.0, 2.0, 1.0, 1.0) == pytest.approx(-2.0)


def test_known_csi_llr_is_affine_in_y():
    a, h, s = 3.0, 0.7, 1.5
    llr = llr_ook_known_csi(Y, a, h, s)
    slope = np.diff(llr) / np.diff(Y)
    assert np.allclose(slope, -a * h / s**2, rtol=1e-12)


def test_known_csi_zero_h_gives_zero_llr():
    assert np.all(llr_ook_known_csi(Y, 2.0, 0.0, 1.0) == 0.0)


@pytest.mark.parametrize("sigma_i2", [1e-8, 1e-7, 1e-6])
def test_awgn_limit_known_answer(sigma_i2):
    # As the scintillation index goes to zero the marginalised LLR must reduce
    # to the textbook linear LLR at h = 1. The deviation is O(sigma_I**2);
    # tolerance below is 300 * sigma_I**2 over this y range, which is slack
    # against the measured coefficient of about 140 reported in
    # validation/validate_awgn_limit.py.
    a, s = 2.0, 1.0
    h, w = LognormalFading(sigma_i2).quadrature(60)
    exact = llr_ook_marginal(Y, a, h, w, s)
    linear = llr_ook_known_csi(Y, a, 1.0, s)
    assert np.max(np.abs(exact - linear)) < 300.0 * sigma_i2


def test_awgn_limit_error_scales_linearly_in_scintillation_index():
    # Stronger statement than a tolerance: the deviation is proportional to
    # sigma_I**2 over three decades, so the limit is first order, not accidental.
    a, s = 2.0, 1.0
    coeffs = []
    for sigma_i2 in (1e-8, 1e-7, 1e-6):
        h, w = LognormalFading(sigma_i2).quadrature(60)
        err = np.max(np.abs(llr_ook_marginal(Y, a, h, w, s) - llr_ook_known_csi(Y, a, 1.0, s)))
        coeffs.append(err / sigma_i2)
    assert max(coeffs) / min(coeffs) < 1.05


def test_maxlog_upper_bounds_exact(lognormal):
    h, w = amplitude_quadrature(lognormal)
    exact = llr_ook_marginal(Y, 4.0, h, w)
    maxlog = llr_ook_maxlog(Y, 4.0, h, w)
    assert np.all(maxlog >= exact - 1e-12)


def test_marginal_matches_direct_sum(lognormal):
    # Equation (3) against an explicit mixture evaluation in probability space.
    h, w = lognormal.quadrature(30)
    a, s = 3.0, 1.1
    direct = np.array(
        [
            np.log(np.exp(-(y**2) / (2 * s**2)))
            - np.log(np.sum(w * np.exp(-((y - a * h) ** 2) / (2 * s**2))))
            for y in Y
        ]
    )
    assert np.allclose(llr_ook_marginal(Y, a, h, w, s), direct, rtol=1e-9, atol=1e-9)


def test_single_node_quadrature_reduces_to_known_csi():
    nodes, weights = np.array([0.8]), np.array([1.0])
    assert np.allclose(
        llr_ook_marginal(Y, 2.5, nodes, weights, 1.3),
        llr_ook_known_csi(Y, 2.5, 0.8, 1.3),
        atol=1e-12,
    )


def test_gray_labels_properties():
    for order in (2, 4, 8, 16):
        labels = gray_labels(order)
        assert labels.shape == (order, int(np.log2(order)))
        # Adjacent Gray labels differ in exactly one position.
        assert np.all(np.sum(labels[1:] != labels[:-1], axis=1) == 1)
        assert len({tuple(row) for row in labels}) == order


def test_gray_labels_rejects_bad_order():
    for bad in (1, 3, 0, -4):
        with pytest.raises(ValueError, match="power of two"):
            gray_labels(bad)


def test_ppm_known_csi_binary_case_matches_difference():
    # M = 2: the single bit LLR is lambda_0 - lambda_1 = a h (y0 - y1) / sigma**2.
    a, h, s = 3.0, 0.9, 1.2
    y = np.array([[1.0, 4.0], [2.0, -1.0]])
    expected = a * h * (y[:, 0] - y[:, 1]) / s**2
    got = llr_ppm_known_csi(y, a, np.array([h, h]), s)[:, 0]
    assert np.allclose(got, expected, rtol=1e-12)


def test_ppm_maxlog_binary_equals_exact_when_only_inner_max_disabled(lognormal):
    # With M = 2 each bit sum has a single term, so max-log over the symbol
    # sums is exact; any difference must come from the inner fading average.
    h, w = amplitude_quadrature(lognormal)
    y = np.array([[1.0, 4.0], [2.0, -1.0], [0.0, 0.0]])
    exact = llr_ppm_marginal(y, 3.0, h, w)
    same = llr_ppm_maxlog(y, 3.0, h, w, inner_max=False)
    assert np.allclose(exact, same, atol=1e-12)


def test_ppm_marginal_single_node_matches_known_csi():
    y = np.array([[0.5, 3.0, -0.2, 0.1]])
    nodes, weights = np.array([1.4]), np.array([1.0])
    assert np.allclose(
        llr_ppm_marginal(y, 2.0, nodes, weights),
        llr_ppm_known_csi(y, 2.0, np.array([1.4])),
        atol=1e-12,
    )


def test_ppm_maxlog_bounds_exact_magnitudes(lognormal):
    h, w = amplitude_quadrature(lognormal)
    rng = np.random.default_rng(3)
    y = rng.standard_normal((200, 4)) + np.eye(4)[rng.integers(0, 4, 200)] * 5.0
    exact = llr_ppm_marginal(y, 5.0, h, w)
    maxlog = llr_ppm_maxlog(y, 5.0, h, w)
    # Max-log over both sums is not one-signed, but it must stay close.
    assert np.max(np.abs(maxlog - exact)) < 3.0


def test_clip_llr():
    x = np.array([-5.0, -1.0, 0.0, 2.0, 9.0])
    assert np.allclose(clip_llr(x, 2.0), [-2.0, -1.0, 0.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="limit"):
        clip_llr(x, 0.0)


def test_rescale_llr():
    x = np.array([-2.0, 3.0])
    assert np.allclose(rescale_llr(x, 0.5), [-1.0, 1.5])
    with pytest.raises(ValueError, match="scale"):
        rescale_llr(x, -1.0)


def test_input_validation(lognormal):
    h, w = lognormal.quadrature(10)
    with pytest.raises(ValueError, match="amplitude"):
        llr_ook_known_csi(Y, -1.0, 1.0)
    with pytest.raises(ValueError, match="sigma"):
        llr_ook_known_csi(Y, 1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="h must be non-negative"):
        llr_ook_known_csi(Y, 1.0, -0.5)
    with pytest.raises(ValueError, match="equal size"):
        llr_ook_marginal(Y, 1.0, h, w[:-1])
    with pytest.raises(ValueError, match="weights must be strictly positive"):
        llr_ook_marginal(Y, 1.0, h, np.zeros_like(w))
    with pytest.raises(ValueError, match="non-negative"):
        llr_ook_marginal(Y, 1.0, -h, w)
    with pytest.raises(ValueError, match="must be finite"):
        llr_ook_known_csi(np.array([np.nan]), 1.0, 1.0)


@given(st.floats(min_value=0.05, max_value=6.0), st.floats(min_value=0.2, max_value=3.0))
@settings(max_examples=40, deadline=None)
def test_known_csi_sign_matches_threshold(h, amplitude):
    # Algebraic identity: L(y) > 0 exactly when y < a h / 2.
    threshold = amplitude * h / 2.0
    for y in (threshold - 0.25, threshold + 0.25):
        llr = float(llr_ook_known_csi(y, amplitude, h, 1.0))
        assert (llr > 0.0) == (y < threshold)
