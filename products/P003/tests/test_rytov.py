"""Known-answer, property and input-validation tests for scintinet.rytov."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scintinet import aperture_averaging_factor, rytov_variance, scintillation_index_weak


class TestKnownAnswers:
    def test_rytov_plane_hand_checked(self):
        # Hand check (Andrews & Phillips 2005: sigma_R^2 = 1.23 Cn^2 k^(7/6) L^(11/6)):
        #   Cn^2 = 1e-15 m^(-2/3), lambda = 1.55e-6 m, L = 2000 m
        #   k = 2*pi/1.55e-6      = 4.053668e6 rad/m
        #   k^(7/6)               = 5.118659e7
        #   L^(11/6) = 2000^(11/6) = 1.126908e6
        #   sigma_R^2 = 1.23 * 1e-15 * 5.118659e7 * 1.126908e6 = 7.09495e-2
        assert rytov_variance(1e-15, 1.55e-6, 2000.0) == pytest.approx(0.0709495, rel=1e-4)

    def test_rytov_plane_second_point(self):
        # Cn^2 = 5e-16, lambda = 8.5e-7 m, L = 1000 m
        #   k = 7.391983e6, k^(7/6) = 1.031702e8, 1000^(11/6) = 3.162278e5
        #   sigma_R^2 = 1.23 * 5e-16 * 1.031702e8 * 3.162278e5 = 2.00646e-2
        assert rytov_variance(5e-16, 8.5e-7, 1000.0) == pytest.approx(0.0200646, rel=1e-4)

    def test_rytov_spherical(self):
        # Spherical-wave coefficient 0.50 instead of 1.23 (Andrews & Phillips 2005):
        #   0.50/1.23 * 7.09495e-2 = 2.88413e-2
        assert rytov_variance(1e-15, 1.55e-6, 2000.0, wave="spherical") == pytest.approx(
            0.0288413, rel=1e-4
        )

    def test_aperture_factor_hand_checked(self):
        # Andrews (1992): A = [1 + 1.062 kD^2/(4L)]^(-7/6)
        #   lambda = 1.55e-6, L = 2000, D = 0.1:
        #   kD^2/(4L) = 4.053668e6 * 0.01 / 8000 = 5.06708
        #   A = (1 + 1.062*5.06708)^(-7/6) = (6.38124)^(-7/6) = 0.115065
        assert aperture_averaging_factor(1.55e-6, 2000.0, 0.1) == pytest.approx(
            0.115065, rel=1e-4
        )

    def test_weak_index_with_aperture(self):
        # sigma_I^2(D) = A * sigma_R^2 = 0.115065 * 7.09495e-2 = 8.1638e-3
        got = scintillation_index_weak(1e-15, 1.55e-6, 2000.0, aperture_diameter=0.1)
        assert got == pytest.approx(8.1638e-3, rel=1e-3)

    def test_weak_index_point_equals_rytov(self):
        assert scintillation_index_weak(1e-15, 1.55e-6, 2000.0) == rytov_variance(
            1e-15, 1.55e-6, 2000.0
        )

    def test_zero_turbulence(self):
        assert rytov_variance(0.0, 1.55e-6, 2000.0) == 0.0


class TestProperties:
    @given(
        cn2=st.floats(1e-18, 1e-13),
        lam=st.floats(4e-7, 2e-6),
        ell=st.floats(100.0, 1e4),
        a=st.floats(0.1, 10.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_linear_in_cn2(self, cn2, lam, ell, a):
        # sigma_R^2 is exactly linear in Cn^2.
        base = rytov_variance(cn2, lam, ell)
        assert rytov_variance(a * cn2, lam, ell) == pytest.approx(a * base, rel=1e-9)

    @given(lam=st.floats(4e-7, 2e-6), ell=st.floats(100.0, 1e4))
    @settings(max_examples=50, deadline=None)
    def test_wavelength_scaling(self, lam, ell):
        # sigma_R^2 proportional to k^(7/6) = (2*pi/lambda)^(7/6).
        r1 = rytov_variance(1e-15, lam, ell)
        r2 = rytov_variance(1e-15, 2.0 * lam, ell)
        assert r1 / r2 == pytest.approx(2.0 ** (7.0 / 6.0), rel=1e-9)

    @given(
        lam=st.floats(4e-7, 2e-6),
        ell=st.floats(100.0, 1e4),
        d=st.floats(1e-3, 1.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_aperture_factor_bounds_and_monotone(self, lam, ell, d):
        a1 = aperture_averaging_factor(lam, ell, d)
        a2 = aperture_averaging_factor(lam, ell, 2.0 * d)
        assert 0.0 < a1 <= 1.0
        assert a2 < a1  # strictly decreasing in D

    def test_spherical_less_than_plane(self):
        assert rytov_variance(1e-15, 1.55e-6, 2000.0, wave="spherical") < rytov_variance(
            1e-15, 1.55e-6, 2000.0, wave="plane"
        )


class TestValidation:
    def test_negative_cn2_raises(self):
        with pytest.raises(ValueError, match="cn2"):
            rytov_variance(-1e-15, 1.55e-6, 2000.0)

    def test_zero_wavelength_raises(self):
        with pytest.raises(ValueError, match="wavelength"):
            rytov_variance(1e-15, 0.0, 2000.0)

    def test_negative_path_raises(self):
        with pytest.raises(ValueError, match="path_length"):
            rytov_variance(1e-15, 1.55e-6, -5.0)

    def test_bad_wave_raises(self):
        with pytest.raises(ValueError, match="wave"):
            rytov_variance(1e-15, 1.55e-6, 2000.0, wave="gaussian")

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            rytov_variance(np.nan, 1.55e-6, 2000.0)

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError):
            rytov_variance("strong", 1.55e-6, 2000.0)

    def test_zero_aperture_raises(self):
        with pytest.raises(ValueError, match="aperture_diameter"):
            aperture_averaging_factor(1.55e-6, 2000.0, 0.0)

    def test_spherical_with_aperture_raises(self):
        with pytest.raises(ValueError, match="plane"):
            scintillation_index_weak(
                1e-15, 1.55e-6, 2000.0, aperture_diameter=0.1, wave="spherical"
            )
