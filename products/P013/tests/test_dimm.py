"""Unit and hand-calculated known-answer tests for the DIMM forward model."""

from __future__ import annotations

import numpy as np
import pytest

from turbscope import (
    PathGeometry,
    cn2_average_from_fried,
    dimm_coefficient,
    dimm_variance,
    fried_from_average,
    fried_parameter,
    r0_from_dimm_variance,
    seeing_fwhm_rad,
)

ARCSEC = 180.0 * 3600.0 / np.pi


def test_dimm_coefficients_hand_calculated():
    # Sarazin & Roddier (1990) with D = 0.06 m, d = 0.20 m:
    #   D^(-1/3) = 1/0.3914868 = 2.5543653   -> 0.179 * 2.5543653 = 0.4572314
    #   d^(-1/3) = 1/0.5848035 = 1.7099759
    #   longitudinal: 0.0968 * 1.7099759 = 0.1655257 -> K_l = 0.2917056
    #   transverse:   0.145  * 1.7099759 = 0.2479465 -> K_t = 0.2092848
    assert dimm_coefficient(0.06, 0.20, "longitudinal") == pytest.approx(0.2917056, rel=1e-6)
    assert dimm_coefficient(0.06, 0.20, "transverse") == pytest.approx(0.2092848, rel=1e-6)


def test_dimm_variance_hand_calculated():
    # sigma_l^2 = 2 lambda^2 r0^(-5/3) K_l with lambda = 500 nm, r0 = 0.10 m:
    #   2 * (5e-7)^2 = 5e-13 ; r0^(-5/3) = 10^(5/3) = 46.415888
    #   5e-13 * 46.415888 = 2.3207944e-11
    #   * 0.2917056 = 6.769888e-12 rad^2  -> 2.601901e-6 rad = 0.536681 arcsec rms
    var = dimm_variance(0.10, 500e-9, 0.06, 0.20, "longitudinal")
    assert var == pytest.approx(6.769888e-12, rel=1e-6)
    assert np.sqrt(var) * ARCSEC == pytest.approx(0.536681, rel=1e-5)
    var_t = dimm_variance(0.10, 500e-9, 0.06, 0.20, "transverse")
    assert var_t == pytest.approx(4.857070e-12, rel=1e-6)
    # Kolmogorov predicts a fixed transverse/longitudinal ratio K_t/K_l = 0.717452
    assert var_t / var == pytest.approx(0.717452, rel=1e-6)


def test_dimm_variance_round_trip():
    for r0 in (0.02, 0.1, 0.5, 2.0):
        var = dimm_variance(r0, 850e-9, 0.06, 0.20)
        assert r0_from_dimm_variance(var, 850e-9, 0.06, 0.20) == pytest.approx(r0, rel=1e-12)


def test_fried_parameter_uniform_plane_wave_hand_calculated():
    # r0 = [0.423 k^2 L Cn2]^(-3/5); lambda = 500 nm -> k = 1.2566371e7, k^2 = 1.5791367e14
    #   0.423 * 1.5791367e14 = 6.6797483e13 ; * 1000 m = 6.6797483e16 ; * 1e-15 = 66.797483
    #   66.797483^(-0.6) = 0.0803792 m = 8.03792 cm
    p = PathGeometry(1000.0, 500e-9, "plane")
    assert fried_from_average(1e-15, p) == pytest.approx(0.0803792, rel=1e-5)


def test_spherical_r0_is_larger_than_plane_by_the_kernel_factor():
    # spherical uses N_co = 3/8, so r0_sph / r0_plane = (3/8)^(-3/5) = 1.7411
    plane = fried_from_average(1e-15, PathGeometry(1000.0, 500e-9, "plane"))
    sph = fried_from_average(1e-15, PathGeometry(1000.0, 500e-9, "spherical"))
    assert sph / plane == pytest.approx((3 / 8) ** (-3 / 5), rel=1e-12)


def test_fried_integral_matches_the_uniform_closed_form(path, uniform_profile):
    z, cn2 = uniform_profile
    assert fried_parameter(z, cn2, path) == pytest.approx(
        fried_from_average(1e-15, path), rel=1e-8
    )


def test_cn2_from_r0_round_trip(path):
    cn2 = 8.1e-15
    r0 = fried_from_average(cn2, path)
    assert cn2_average_from_fried(r0, path) == pytest.approx(cn2, rel=1e-12)


def test_cn2_recovered_from_a_dimm_is_wavelength_independent():
    # r0 ~ lambda^(6/5) and sigma^2 ~ lambda^2 r0^(-5/3) ~ lambda^0 * Cn2, so the
    # Cn2 recovered from a measured differential variance must not depend on lambda.
    cn2 = 5e-15
    recovered = []
    for lam in (500e-9, 850e-9, 1550e-9):
        p = PathGeometry(1500.0, lam)
        r0 = fried_from_average(cn2, p)
        var = dimm_variance(r0, lam, 0.06, 0.20)
        r0_back = r0_from_dimm_variance(var, lam, 0.06, 0.20)
        recovered.append(cn2_average_from_fried(r0_back, p))
    assert np.allclose(recovered, cn2, rtol=1e-12)


def test_seeing_fwhm_known_answer():
    # 0.98 * 500 nm / 0.10 m = 4.9e-6 rad = 1.0106976 arcsec
    assert seeing_fwhm_rad(0.10, 500e-9) * ARCSEC == pytest.approx(1.0106976, rel=1e-6)
