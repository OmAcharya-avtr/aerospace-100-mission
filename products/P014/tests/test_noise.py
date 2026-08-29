"""Unit, KAT, edge-case and property tests for wavelab.noise."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wavelab.noise import add_slope_noise, apply_dropout, slope_sigma

# --------------------------------------------------------------------- slope_sigma


def test_slope_sigma_hand_calc_at_reference_flux():
    # By definition sigma(flux_ref) = sigma_ref exactly.
    assert slope_sigma(100.0, sigma_ref=0.5, flux_ref=100.0) == pytest.approx(0.5)


def test_slope_sigma_hand_calc_quadruple_flux_halves_sigma():
    # sigma propto 1/sqrt(N): N -> 4N halves sigma.
    s1 = slope_sigma(100.0, sigma_ref=1.0, flux_ref=100.0)
    s2 = slope_sigma(400.0, sigma_ref=1.0, flux_ref=100.0)
    assert s2 == pytest.approx(s1 / 2.0)


def test_slope_sigma_rejects_non_positive_flux():
    with pytest.raises(ValueError):
        slope_sigma(0.0)
    with pytest.raises(ValueError):
        slope_sigma(-5.0)


def test_slope_sigma_rejects_non_finite_flux():
    with pytest.raises(ValueError):
        slope_sigma(float("nan"))
    with pytest.raises(ValueError):
        slope_sigma(float("inf"))


def test_slope_sigma_rejects_non_positive_sigma_ref():
    with pytest.raises(ValueError):
        slope_sigma(100.0, sigma_ref=0.0)


def test_slope_sigma_rejects_non_positive_flux_ref():
    with pytest.raises(ValueError):
        slope_sigma(100.0, flux_ref=-1.0)


def test_slope_sigma_monotonically_decreasing_in_flux():
    fluxes = np.array([10, 100, 1000, 10000, 100000], dtype=float)
    sigmas = [slope_sigma(f) for f in fluxes]
    assert all(sigmas[i] > sigmas[i + 1] for i in range(len(sigmas) - 1))


@given(st.floats(min_value=1.0, max_value=1e8, allow_nan=False))
@settings(max_examples=30)
def test_slope_sigma_squared_times_flux_is_constant(flux):
    # Algebraic identity: sigma(N)^2 * N = sigma_ref^2 * N_ref, constant.
    sigma = slope_sigma(flux, sigma_ref=1.0, flux_ref=100.0)
    assert sigma**2 * flux == pytest.approx(1.0 * 100.0, rel=1e-9)


# --------------------------------------------------------------------- add_slope_noise


def test_add_slope_noise_rejects_non_finite_input():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        add_slope_noise(np.array([np.nan, 1.0]), 100.0, rng)


def test_add_slope_noise_preserves_shape():
    rng = np.random.default_rng(0)
    s = np.zeros((3, 4))
    out = add_slope_noise(s, 100.0, rng)
    assert out.shape == s.shape


def test_add_slope_noise_empirical_std_matches_prediction():
    rng = np.random.default_rng(0)
    n = 200000
    s = np.zeros(n)
    predicted = slope_sigma(500.0, sigma_ref=1.0, flux_ref=100.0)
    noisy = add_slope_noise(s, 500.0, rng, sigma_ref=1.0, flux_ref=100.0)
    empirical = noisy.std()
    assert empirical == pytest.approx(predicted, rel=0.02)


def test_add_slope_noise_deterministic_given_seeded_generator():
    s = np.zeros(10)
    out1 = add_slope_noise(s, 100.0, np.random.default_rng(42))
    out2 = add_slope_noise(s, 100.0, np.random.default_rng(42))
    np.testing.assert_array_equal(out1, out2)


# --------------------------------------------------------------------- apply_dropout


def test_apply_dropout_rejects_bad_n_sub():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        apply_dropout(0, 0.1, rng)
    with pytest.raises(TypeError):
        apply_dropout(3.5, 0.1, rng)


def test_apply_dropout_rejects_bad_rate():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        apply_dropout(10, -0.1, rng)
    with pytest.raises(ValueError):
        apply_dropout(10, 1.0, rng)


def test_apply_dropout_zero_rate_keeps_all():
    rng = np.random.default_rng(0)
    active = apply_dropout(50, 0.0, rng)
    assert np.all(active)


def test_apply_dropout_always_keeps_at_least_one():
    rng = np.random.default_rng(0)
    for _ in range(20):
        active = apply_dropout(5, 0.99, rng)
        assert np.any(active)


def test_apply_dropout_empirical_rate_matches_target():
    rng = np.random.default_rng(0)
    active = apply_dropout(20000, 0.3, rng)
    empirical_dropout = 1.0 - active.mean()
    assert empirical_dropout == pytest.approx(0.3, abs=0.02)


def test_apply_dropout_shape():
    rng = np.random.default_rng(0)
    active = apply_dropout(17, 0.2, rng)
    assert active.shape == (17,)
    assert active.dtype == bool
