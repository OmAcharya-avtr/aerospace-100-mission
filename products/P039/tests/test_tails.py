"""Tail-quantile estimators and the order-statistic theory behind them."""

from __future__ import annotations

import math

import numpy as np
import pytest

from latencynet.tails import (
    lognormal_quantile,
    lognormal_quantile_se,
    lognormal_tail_convergence,
    quantile,
    quantile_min_samples,
)


def test_nearest_rank_known_answers():
    data = np.arange(1.0, 11.0)  # 1..10
    # Hand-calculated: rank = ceil(p * n). p = 0.5, n = 10 -> rank 5 -> 5.0.
    assert quantile(data, 0.5, "nearest_rank") == 5.0
    # p = 0.95 -> ceil(9.5) = 10 -> 10.0 (the maximum).
    assert quantile(data, 0.95, "nearest_rank") == 10.0
    # p = 0.01 -> ceil(0.1) = 1 -> 1.0 (the minimum).
    assert quantile(data, 0.01, "nearest_rank") == 1.0


def test_linear_known_answer():
    data = np.arange(1.0, 11.0)
    # Type 7: h = (n - 1) p + 1 = 9 * 0.5 + 1 = 5.5 -> 5.5 by interpolation.
    assert quantile(data, 0.5, "linear") == pytest.approx(5.5, rel=1e-15)


def test_nearest_rank_is_always_an_observation():
    rng = np.random.default_rng(4)
    data = rng.normal(size=500)
    for p in (0.1, 0.5, 0.9, 0.99):
        assert quantile(data, p, "nearest_rank") in set(data.tolist())


def test_quantile_min_samples_known_answers():
    # ceil(1 / (1 - p)): p = 0.99 -> 100, p = 0.999 -> 1000, p = 0.5 -> 2.
    assert quantile_min_samples(0.99) == 100
    assert quantile_min_samples(0.999) == 1000
    assert quantile_min_samples(0.5) == 2


def test_lognormal_quantile_known_answer():
    # mu = 0, sigma = 1: the median is exp(0) = 1 exactly.
    assert lognormal_quantile(0.0, 1.0, 0.5) == pytest.approx(1.0, rel=1e-14)
    # p = 0.99: Phi^-1(0.99) = 2.3263478740408408, so q = exp(2.32634787...)
    #         = 10.240473656312131.
    assert lognormal_quantile(0.0, 1.0, 0.99) == pytest.approx(10.240473656312131, rel=1e-12)


def test_lognormal_quantile_se_matches_the_closed_form():
    # SE = sqrt(p(1-p)/n) * sigma * q_p / phi(z_p), recomputed independently.
    mu, sigma, p, n = -8.5, 0.4, 0.99, 10_000
    z = 2.3263478740408408
    phi = math.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    q = math.exp(mu + sigma * z)
    expected = math.sqrt(p * (1.0 - p) / n) * sigma * q / phi
    assert lognormal_quantile_se(mu, sigma, p, n) == pytest.approx(expected, rel=1e-10)


def test_se_scales_as_inverse_root_n():
    a = lognormal_quantile_se(-8.5, 0.4, 0.99, 1_000)
    b = lognormal_quantile_se(-8.5, 0.4, 0.99, 100_000)
    assert a / b == pytest.approx(10.0, rel=1e-12)


def test_convergence_slope_is_near_minus_one_half():
    # 120 repeats at each n: the RMSE estimate itself has a relative standard
    # error of about 1 / sqrt(2 * 120) = 6.5 %, so the fitted slope is not
    # expected to land on -0.5 exactly. A 0.08 band is the honest tolerance
    # at this repeat count; validation/validate_tail_convergence.py uses many
    # more repeats and reports the slope against a tighter band.
    result = lognormal_tail_convergence(-8.5, 0.4, 0.99, (200, 800, 3200, 12800), 120, 31)
    assert result.predicted_slope == -0.5
    assert result.fitted_slope == pytest.approx(-0.5, abs=0.08)
    assert len(result.rmse_s) == 4
    assert all(r > 0.0 for r in result.rmse_s)
    # Measured RMSE should approach the analytic SE from below or near 1.
    assert result.ratio_measured_over_analytic[-1] == pytest.approx(1.0, abs=0.25)


def test_rmse_decreases_monotonically_with_n():
    result = lognormal_tail_convergence(-8.0, 0.5, 0.99, (500, 4000, 32000), 100, 17)
    assert list(result.rmse_s) == sorted(result.rmse_s, reverse=True)


def test_quantile_input_validation():
    with pytest.raises(ValueError, match="non-empty"):
        quantile(np.array([]), 0.5)
    with pytest.raises(ValueError, match="finite"):
        quantile(np.array([1.0, np.nan]), 0.5)
    with pytest.raises(ValueError, match="strictly in"):
        quantile(np.array([1.0]), 1.0)
    with pytest.raises(ValueError, match="method must be"):
        quantile(np.array([1.0]), 0.5, "midpoint")
    with pytest.raises(ValueError, match="strictly in"):
        quantile_min_samples(0.0)


def test_lognormal_validation():
    with pytest.raises(ValueError, match="sigma"):
        lognormal_quantile(0.0, 0.0, 0.5)
    with pytest.raises(ValueError, match="strictly in"):
        lognormal_quantile(0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="n must be"):
        lognormal_quantile_se(0.0, 1.0, 0.5, 0)


def test_convergence_validation():
    with pytest.raises(ValueError, match="non-empty"):
        lognormal_tail_convergence(0.0, 1.0, 0.99, (), 10, 1)
    with pytest.raises(ValueError, match=">= 1"):
        lognormal_tail_convergence(0.0, 1.0, 0.99, (0,), 10, 1)
    with pytest.raises(ValueError, match="n_repeats"):
        lognormal_tail_convergence(0.0, 1.0, 0.99, (100,), 1, 1)
