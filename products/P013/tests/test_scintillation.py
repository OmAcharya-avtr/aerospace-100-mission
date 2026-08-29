"""Unit and known-answer tests for the scintillation forward model."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special

from turbscope import PathGeometry, gamma_gamma_parameters, saturation_peak, scintillation_index
from turbscope.scintillation import (
    RYTOV_COEFFICIENT,
    aperture_parameter_sq,
    log_irradiance_variances,
    rytov_variance,
    rytov_variance_from_average,
    uniform_cn2_from_beta0_sq,
)


def test_rytov_coefficient_matches_the_published_rounded_value():
    # C_R = 8 pi^2 (0.033)(0.5)[-Gamma(-5/6) cos(5 pi/12)] = 2.2522625...
    # Andrews & Phillips (2005) quote this as 2.25.
    assert RYTOV_COEFFICIENT == pytest.approx(2.2522625262, rel=1e-9)
    assert RYTOV_COEFFICIENT == pytest.approx(2.25, rel=1.1e-3)


def test_uniform_path_reduces_to_the_textbook_constants():
    # spherical: C_R * B(11/6,11/6) = 0.496704 -> quoted as 0.5 (Andrews & Phillips Eq. 8.13)
    spherical = RYTOV_COEFFICIENT * float(special.beta(11 / 6, 11 / 6))
    assert spherical == pytest.approx(0.4967041923, rel=1e-9)
    assert spherical == pytest.approx(0.5, rel=7e-3)
    # plane: C_R * 6/11 = 1.228507 -> quoted as 1.23 (Eq. 8.9)
    plane = RYTOV_COEFFICIENT * 6 / 11
    assert plane == pytest.approx(1.2285068324, rel=1e-9)
    assert plane == pytest.approx(1.23, rel=4e-3)


def test_rytov_variance_integral_matches_the_uniform_closed_form(path, uniform_profile):
    z, cn2 = uniform_profile
    from_integral = rytov_variance(z, cn2, path)
    from_average = rytov_variance_from_average(1e-15, path)
    assert from_integral == pytest.approx(from_average, rel=5e-6)


def test_rytov_variance_scales_as_L_to_the_11_over_6():
    # beta_0^2 ~ L^(11/6) at fixed uniform Cn2: doubling L multiplies it by 2^(11/6)=3.5637
    a = rytov_variance_from_average(1e-15, PathGeometry(1000.0, 1.55e-6))
    b = rytov_variance_from_average(1e-15, PathGeometry(2000.0, 1.55e-6))
    assert b / a == pytest.approx(2 ** (11 / 6), rel=1e-12)


def test_rytov_variance_scales_as_k_to_the_7_over_6():
    # halving lambda doubles k, multiplying beta_0^2 by 2^(7/6) = 2.2449
    a = rytov_variance_from_average(1e-15, PathGeometry(1000.0, 1.60e-6))
    b = rytov_variance_from_average(1e-15, PathGeometry(1000.0, 0.80e-6))
    assert b / a == pytest.approx(2 ** (7 / 6), rel=1e-12)


def test_weak_limit_of_the_saturation_model_is_the_rytov_variance():
    for b in (1e-6, 1e-4, 1e-3):
        assert float(scintillation_index(b, 0.0)) == pytest.approx(b, rel=2e-3)


def test_saturation_peak_and_asymptote():
    b_peak, s_peak = saturation_peak(0.0)
    # Located numerically; the model is Andrews & Phillips (2005) Eq. 9.60.
    assert b_peak == pytest.approx(7.2966, rel=1e-3)
    assert s_peak == pytest.approx(1.6921, rel=1e-3)
    # The classic saturation result: sigma_I^2 -> 1 as beta_0^2 -> infinity
    # (Gracheva & Gurvich 1965).  The Andrews-Phillips *fit* tends to
    # exp(0.51 * 0.69^(-5/6)) - 1 = 1.0033173, i.e. it overshoots the theoretical
    # limit by 0.33 %.  That is a property of the published fit and is asserted
    # here rather than smoothed over.
    asymptote = float(np.expm1(0.51 * 0.69 ** (-5 / 6)))
    assert asymptote == pytest.approx(1.0033173, rel=1e-6)
    assert float(scintillation_index(1e12, 0.0)) == pytest.approx(asymptote, rel=1e-4)
    assert float(scintillation_index(1e12, 0.0)) == pytest.approx(1.0, abs=4e-3)


def test_scintillation_index_is_non_monotonic():
    b = np.logspace(-2, 3, 2000)
    s = scintillation_index(b, 0.0)
    assert np.any(np.diff(s) < 0.0), "the model must decrease somewhere (saturation)"


def test_aperture_averaging_reduces_the_index_and_delays_the_peak():
    path = PathGeometry(1000.0, 1.55e-6)
    d_sq = aperture_parameter_sq(0.10, path)
    assert d_sq == pytest.approx(1000.0 * 0.0 + (2 * np.pi / 1.55e-6) * 0.01 / 4000.0, rel=1e-12)
    b = 1.0
    assert float(scintillation_index(b, d_sq)) < float(scintillation_index(b, 0.0))
    assert saturation_peak(d_sq)[0] > saturation_peak(0.0)[0]


def test_gamma_gamma_parameters_reproduce_the_scintillation_index():
    # sigma_I^2 = (1 + 1/alpha)(1 + 1/beta) - 1 for the gamma-gamma model.
    for b in (0.01, 0.5, 3.0, 30.0):
        alpha, beta = gamma_gamma_parameters(b, 0.0)
        assert (1 + 1 / alpha) * (1 + 1 / beta) - 1 == pytest.approx(
            float(scintillation_index(b, 0.0)), rel=1e-12
        )


def test_log_irradiance_variance_terms_sum_to_the_index():
    s_x, s_y = log_irradiance_variances(np.array([0.1, 2.0, 50.0]), 0.0)
    assert np.allclose(np.expm1(s_x + s_y), scintillation_index(np.array([0.1, 2.0, 50.0]), 0.0))


def test_uniform_cn2_round_trip(path):
    cn2 = 4.2e-15
    beta = rytov_variance_from_average(cn2, path)
    assert uniform_cn2_from_beta0_sq(beta, path) == pytest.approx(cn2, rel=1e-12)
