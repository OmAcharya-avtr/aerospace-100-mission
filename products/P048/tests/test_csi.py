"""Channel-state-error model tests, including the posterior known answers."""

from __future__ import annotations

import numpy as np
import pytest

from softdecode.channel import GammaGammaFading
from softdecode.csi import (
    MultiplicativeCsiError,
    StaleCsiError,
    csi_aware_llr_ook,
    db_to_log_gain,
    posterior_quadrature,
)
from softdecode.llr import llr_ook_marginal


def test_db_to_log_gain_hand_calculation():
    # 10 dB of irradiance is a factor of 10, i.e. a natural log gain of ln 10.
    assert db_to_log_gain(10.0) == pytest.approx(np.log(10.0), rel=1e-14)
    assert db_to_log_gain(0.0) == 0.0
    assert db_to_log_gain(-3.0) == pytest.approx(-0.3 * np.log(10.0), rel=1e-14)


def test_zero_jitter_estimate_is_deterministic(rng):
    error = MultiplicativeCsiError(3.0, 0.0)
    h = np.array([0.5, 1.0, 2.0])
    got = error.estimate(h, rng)
    assert np.allclose(got, h * 10.0 ** 0.3, rtol=1e-12)


def test_positive_bias_overestimates(rng):
    h = np.array([1.0])
    assert MultiplicativeCsiError(2.0, 0.0).estimate(h, rng)[0] > 1.0
    assert MultiplicativeCsiError(-2.0, 0.0).estimate(h, rng)[0] < 1.0


def test_estimate_preserves_positivity(lognormal, rng):
    h = lognormal.sample(5000, rng)
    assert np.all(MultiplicativeCsiError(0.0, 4.0).estimate(h, rng) > 0.0)


def test_multiplicative_posterior_moments_known_answer(lognormal):
    # Conjugate Gaussian update on log h. With sigma_x**2 = log(1.3) and
    # sigma_e = ln(10)/10 * 1.0, s**2 = 1 / (1/sx2 + 1/se2) exactly.
    error = MultiplicativeCsiError(2.0, 1.0)
    sx2 = lognormal.sigma_x**2
    se2 = error.jitter**2
    s2 = 1.0 / (1.0 / sx2 + 1.0 / se2)
    h_hat = np.array([1.7])
    m, s = error.log_posterior_moments(h_hat, lognormal)
    expected_m = s2 * (lognormal.mu_x / sx2 + (np.log(1.7) - error.bias) / se2)
    assert float(s[0]) == pytest.approx(np.sqrt(s2), rel=1e-14)
    assert float(m[0]) == pytest.approx(expected_m, rel=1e-14)


def test_posterior_shrinks_toward_prior_mean(lognormal):
    error = MultiplicativeCsiError(0.0, 2.0)
    h_hat = np.array([4.0])
    m, _ = error.log_posterior_moments(h_hat, lognormal)
    assert lognormal.mu_x < float(m[0]) < np.log(4.0)


def test_stale_posterior_known_answer(lognormal):
    error = StaleCsiError(0.8)
    h_hat = np.array([1.5])
    m, s = error.log_posterior_moments(h_hat, lognormal)
    assert float(m[0]) == pytest.approx(
        lognormal.mu_x + 0.8 * (np.log(1.5) - lognormal.mu_x), rel=1e-14
    )
    assert float(s[0]) == pytest.approx(lognormal.sigma_x * np.sqrt(1 - 0.64), rel=1e-14)


def test_stale_perfect_correlation_is_perfect_csi(lognormal, rng):
    h = lognormal.sample(2000, rng)
    got = StaleCsiError(1.0).estimate(h, rng, lognormal)
    assert np.allclose(got, h, rtol=1e-12)
    m, s = StaleCsiError(1.0).log_posterior_moments(h[:5], lognormal)
    assert np.allclose(np.exp(m), h[:5], rtol=1e-12)
    assert np.all(s == 0.0)


def test_stale_correlation_is_realised(lognormal, rng):
    h = lognormal.sample(200000, rng)
    h_hat = StaleCsiError(0.7).estimate(h, rng, lognormal)
    corr = np.corrcoef(np.log(h), np.log(h_hat))[0, 1]
    assert corr == pytest.approx(0.7, abs=0.01)


def test_posterior_quadrature_normalised(lognormal, gammagamma):
    error = MultiplicativeCsiError(1.0, 1.5)
    for fading in (lognormal, gammagamma):
        nodes, weights = posterior_quadrature(1.3, fading, error)
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)
        assert np.all(nodes > 0.0)


def test_posterior_quadrature_zero_jitter_is_single_node(lognormal):
    nodes, weights = posterior_quadrature(1.3, lognormal, MultiplicativeCsiError(2.0, 0.0))
    assert nodes.size == 1
    assert weights[0] == 1.0
    assert nodes[0] == pytest.approx(1.3 / 10.0**0.2, rel=1e-12)


def test_vectorised_csi_aware_matches_scalar_path(lognormal, gammagamma, rng):
    error = MultiplicativeCsiError(2.0, 1.5)
    for fading in (lognormal, gammagamma):
        h = fading.sample(8, rng)
        h_hat = error.estimate(h, rng)
        y = 5.0 * h * rng.integers(0, 2, 8) + rng.standard_normal(8)
        vector = csi_aware_llr_ook(y, 5.0, h_hat, fading, error)
        scalar = np.array(
            [
                llr_ook_marginal(
                    np.array([y[i]]), 5.0, *posterior_quadrature(float(h_hat[i]), fading, error)
                )[0]
                for i in range(8)
            ]
        )
        assert np.allclose(vector, scalar, rtol=1e-7, atol=1e-7)


def test_csi_aware_zero_jitter_zero_bias_is_known_csi(lognormal, rng):
    from softdecode.llr import llr_ook_known_csi

    error = MultiplicativeCsiError(0.0, 0.0)
    h = lognormal.sample(16, rng)
    y = rng.standard_normal(16) * 2.0
    got = csi_aware_llr_ook(y, 3.0, h, lognormal, error)
    assert np.allclose(got, llr_ook_known_csi(y, 3.0, h), rtol=1e-10)


def test_csi_aware_maxlog_upper_bounds_exact(lognormal, rng):
    error = MultiplicativeCsiError(1.0, 1.5)
    h = lognormal.sample(500, rng)
    h_hat = error.estimate(h, rng)
    y = rng.standard_normal(500) * 3.0
    exact = csi_aware_llr_ook(y, 4.0, h_hat, lognormal, error)
    maxlog = csi_aware_llr_ook(y, 4.0, h_hat, lognormal, error, max_log=True)
    assert np.all(maxlog >= exact - 1e-9)


def test_input_validation(lognormal, rng):
    with pytest.raises(ValueError, match="jitter_db"):
        MultiplicativeCsiError(0.0, -1.0)
    with pytest.raises(ValueError, match="bias_db"):
        MultiplicativeCsiError(np.nan, 1.0)
    with pytest.raises(ValueError, match="correlation"):
        StaleCsiError(0.0)
    with pytest.raises(ValueError, match="correlation"):
        StaleCsiError(1.5)
    with pytest.raises(TypeError, match="LognormalFading"):
        StaleCsiError(0.5).log_posterior_moments(np.array([1.0]), GammaGammaFading(4.0, 2.0))
    with pytest.raises(ValueError, match="h_hat"):
        posterior_quadrature(0.0, lognormal, MultiplicativeCsiError(0.0, 1.0))
    with pytest.raises(ValueError, match="undefined"):
        MultiplicativeCsiError(0.0, 0.0).log_likelihood(np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="h must be non-negative"):
        MultiplicativeCsiError(0.0, 1.0).estimate(np.array([-1.0]), rng)
    with pytest.raises(ValueError, match="equal size"):
        csi_aware_llr_ook(
            np.zeros(3), 1.0, np.ones(4), lognormal, MultiplicativeCsiError(0.0, 1.0)
        )
    with pytest.raises(ValueError, match="strictly positive"):
        csi_aware_llr_ook(
            np.zeros(3), 1.0, np.zeros(3), lognormal, MultiplicativeCsiError(0.0, 1.0)
        )
    with pytest.raises(TypeError, match="unsupported fading model"):
        csi_aware_llr_ook(np.zeros(2), 1.0, np.ones(2), object(),
                          MultiplicativeCsiError(0.0, 1.0))
    with pytest.raises(TypeError, match="MultiplicativeCsiError"):
        csi_aware_llr_ook(
            np.zeros(2), 1.0, np.ones(2), GammaGammaFading(4.0, 2.0), StaleCsiError(0.5)
        )
