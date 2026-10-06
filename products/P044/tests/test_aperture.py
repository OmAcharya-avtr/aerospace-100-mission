"""Tests for aperture averaging, including the two derived closed-form limits."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import integrate

from aperturediv.aperture import (
    aperture_averaging_factor,
    aperture_averaging_large_d,
    aperture_averaging_small_d,
    circular_aperture_weight,
    effective_scintillation_index,
    equal_area_diameter,
    fresnel_scale,
    normalised_covariance,
)


class TestFresnelScale:
    def test_known_answer(self):
        # 1.55 um over 2000 m: sqrt(1.55e-6 * 2000) = sqrt(3.1e-3) = 0.05567764 m.
        assert fresnel_scale(1.55e-6, 2000.0) == pytest.approx(0.055677643628, rel=1e-10)

    def test_square_law(self):
        assert fresnel_scale(1e-6, 4000.0) == pytest.approx(
            2.0 * fresnel_scale(1e-6, 1000.0), rel=1e-12
        )

    @pytest.mark.parametrize(
        "lam,path", [(0.0, 100.0), (-1e-6, 100.0), (1e-6, 0.0), (1e-6, -5.0)]
    )
    def test_rejects_bad_input(self, lam, path):
        with pytest.raises(ValueError):
            fresnel_scale(lam, path)


class TestKernel:
    def test_normalisation_identity(self):
        # int_0^1 u (arccos u - u sqrt(1-u^2)) du = pi/16, derived in the module.
        val, _ = integrate.quad(
            lambda u: u * float(circular_aperture_weight(u)),
            0.0,
            1.0,
            epsabs=1e-14,
            epsrel=1e-13,
        )
        assert val == pytest.approx(np.pi / 16.0, rel=1e-11)

    def test_known_answers(self):
        # W(0) = arccos 0 = pi/2; W(1) = 0.
        assert float(circular_aperture_weight(0.0)) == pytest.approx(np.pi / 2, rel=1e-14)
        assert float(circular_aperture_weight(1.0)) == pytest.approx(0.0, abs=1e-15)
        # W(0.5) = arccos(0.5) - 0.5*sqrt(0.75) = 1.0471976 - 0.4330127 = 0.6141849.
        assert float(circular_aperture_weight(0.5)) == pytest.approx(0.61418485, rel=1e-7)

    def test_zero_outside_support(self):
        out = circular_aperture_weight(np.array([-0.5, 1.5, 2.0]))
        assert np.all(out == 0.0)

    def test_decreasing_on_support(self):
        u = np.linspace(0.0, 1.0, 200)
        assert np.all(np.diff(circular_aperture_weight(u)) < 0.0)


class TestCovariance:
    def test_unit_at_zero(self):
        for model in ("gaussian", "exponential"):
            assert float(normalised_covariance(0.0, 1.0, model)) == pytest.approx(1.0)

    def test_known_answers(self):
        assert float(normalised_covariance(1.0, 1.0, "gaussian")) == pytest.approx(
            np.exp(-1.0), rel=1e-14
        )
        assert float(normalised_covariance(2.0, 1.0, "exponential")) == pytest.approx(
            np.exp(-2.0), rel=1e-14
        )

    def test_rejects_unknown_model(self):
        with pytest.raises(ValueError, match="model must be one of"):
            normalised_covariance(1.0, 1.0, "lorentzian")

    def test_rejects_negative_separation(self):
        with pytest.raises(ValueError):
            normalised_covariance(-1.0, 1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_rejects_bad_scale(self, bad):
        with pytest.raises(ValueError):
            normalised_covariance(1.0, bad)


class TestAveragingFactor:
    def test_unity_at_zero_diameter(self):
        assert aperture_averaging_factor(0.0, 1.0) == 1.0

    def test_bounded_and_decreasing(self):
        grid = np.geomspace(1e-3, 200.0, 80)
        vals = np.array([aperture_averaging_factor(d, 1.0) for d in grid])
        assert np.all(vals > 0.0)
        assert np.all(vals <= 1.0)
        assert np.all(np.diff(vals) < 0.0)

    def test_scale_invariance(self):
        # A depends only on D/rho_c.
        assert aperture_averaging_factor(0.4, 0.2) == pytest.approx(
            aperture_averaging_factor(4.0, 2.0), rel=1e-10
        )

    @pytest.mark.parametrize("ratio", [0.0125, 0.025, 0.05, 0.1])
    def test_small_d_limit(self, ratio):
        got = aperture_averaging_factor(ratio, 1.0)
        want = aperture_averaging_small_d(ratio, 1.0)
        assert got == pytest.approx(want, rel=2e-4)

    @pytest.mark.parametrize("ratio", [20.0, 50.0, 100.0, 200.0])
    def test_large_d_two_term_limit(self, ratio):
        got = aperture_averaging_factor(ratio, 1.0)
        want = aperture_averaging_large_d(ratio, 1.0, order=2)
        assert got == pytest.approx(want, rel=1e-4)

    def test_large_d_leading_coefficient(self):
        # A D^2 / rho_c^2 -> 4; convergence is only O(rho_c/D).
        assert aperture_averaging_factor(1000.0, 1.0) * 1e6 == pytest.approx(4.0, abs=0.01)

    def test_exponential_and_gaussian_cross_once(self):
        grid = np.geomspace(1e-2, 100.0, 120)
        gauss = np.array([aperture_averaging_factor(d, 1.0, "gaussian") for d in grid])
        expo = np.array([aperture_averaging_factor(d, 1.0, "exponential") for d in grid])
        sign = np.sign(expo - gauss)
        assert int(np.count_nonzero(np.diff(sign) != 0)) == 1

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_rejects_bad_diameter(self, bad):
        with pytest.raises(ValueError):
            aperture_averaging_factor(bad, 1.0)

    def test_small_d_rejects_bad_scale(self):
        with pytest.raises(ValueError):
            aperture_averaging_small_d(1.0, 0.0)

    def test_large_d_rejects_bad_order(self):
        with pytest.raises(ValueError, match="order must be 1 or 2"):
            aperture_averaging_large_d(10.0, 1.0, order=3)

    def test_large_d_rejects_zero_diameter(self):
        with pytest.raises(ValueError):
            aperture_averaging_large_d(0.0, 1.0)


class TestEffectiveScintillationIndex:
    def test_scales_point_value(self):
        a = aperture_averaging_factor(0.1, 0.05)
        assert effective_scintillation_index(0.6, 0.1, 0.05) == pytest.approx(0.6 * a, rel=1e-12)

    def test_zero_diameter_leaves_si_unchanged(self):
        assert effective_scintillation_index(0.6, 0.0, 0.05) == pytest.approx(0.6, rel=1e-14)

    @pytest.mark.parametrize("bad", [0.0, -0.2, float("nan")])
    def test_rejects_bad_si(self, bad):
        with pytest.raises(ValueError):
            effective_scintillation_index(bad, 0.1, 0.05)


class TestEqualAreaDiameter:
    def test_known_answers(self):
        # n (d/2)^2 = (D/2)^2 -> d = D/sqrt(n). D = 0.2, n = 4 -> d = 0.1.
        assert equal_area_diameter(0.2, 4) == pytest.approx(0.1, rel=1e-14)
        assert equal_area_diameter(0.2, 1) == pytest.approx(0.2, rel=1e-14)
        assert equal_area_diameter(0.2, 2) == pytest.approx(0.2 / np.sqrt(2), rel=1e-14)

    @given(st.floats(0.01, 2.0), st.integers(1, 16))
    def test_total_area_preserved(self, d, n):
        each = equal_area_diameter(d, n)
        assert n * each**2 == pytest.approx(d**2, rel=1e-10)

    @pytest.mark.parametrize("d,n", [(0.0, 2), (-1.0, 2), (0.2, 0), (0.2, -1)])
    def test_rejects_bad_input(self, d, n):
        with pytest.raises(ValueError):
            equal_area_diameter(d, n)
