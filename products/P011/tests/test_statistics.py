"""Tests for waveforge.statistics against published Kolmogorov results."""

from __future__ import annotations

import numpy as np
import pytest

from waveforge.statistics import (
    KOLMOGOROV_PSD_CYCLIC,
    NOLL_RESIDUAL_TABLE,
    fried_parameter_from_cn2,
    greenwood_frequency,
    greenwood_time_constant,
    kolmogorov_psd_cyclic,
    noll_residual_asymptote,
    noll_residual_variance,
    phase_structure_function,
    total_phase_variance,
    zernike_variance,
)


class TestPSD:
    def test_constant_matches_roddier(self):
        # 0.49 (2 pi)^(-5/3) = 0.022903
        assert KOLMOGOROV_PSD_CYCLIC == pytest.approx(0.022903, abs=1e-6)

    def test_power_law_slope(self):
        f = np.array([1.0, 10.0])
        psd = kolmogorov_psd_cyclic(f, 0.1)
        assert psd[0] / psd[1] == pytest.approx(10.0 ** (11 / 3), rel=1e-12)

    def test_r0_scaling(self):
        a = kolmogorov_psd_cyclic(5.0, 0.1)
        b = kolmogorov_psd_cyclic(5.0, 0.2)
        assert a / b == pytest.approx(2.0 ** (5 / 3), rel=1e-12)

    def test_zero_frequency_is_infinite(self):
        assert np.isinf(kolmogorov_psd_cyclic(0.0, 0.1))

    @pytest.mark.parametrize("r0", [0.0, -0.1, float("nan")])
    def test_bad_r0(self, r0):
        with pytest.raises(ValueError, match="r0_m"):
            kolmogorov_psd_cyclic(1.0, r0)

    def test_negative_frequency_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            kolmogorov_psd_cyclic(-1.0, 0.1)


class TestStructureFunction:
    def test_value_at_r0(self):
        # D(r0) = 2 (24/5 Gamma(6/5))^(5/6) = 6.883877 rad^2 by definition of
        # the Fried parameter (the "6.88" of the textbooks, unrounded)
        assert phase_structure_function(0.1, 0.1) == pytest.approx(6.883877, abs=1e-6)

    def test_five_thirds_power(self):
        d1 = phase_structure_function(0.1, 0.1)
        d2 = phase_structure_function(0.2, 0.1)
        assert d2 / d1 == pytest.approx(2.0 ** (5 / 3), rel=1e-12)

    def test_zero_separation(self):
        assert phase_structure_function(0.0, 0.1) == pytest.approx(0.0)

    def test_negative_separation_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            phase_structure_function(-1.0, 0.1)

    def test_bad_r0(self):
        with pytest.raises(ValueError, match="r0_m"):
            phase_structure_function(1.0, -1.0)


class TestZernikeVariance:
    def test_tip_and_tilt_are_equal(self):
        assert zernike_variance(2) == pytest.approx(zernike_variance(3))

    def test_tilt_matches_published_value(self):
        # Noll Table IV: Delta_1 - Delta_2 = 1.0299 - 0.582 = 0.4479
        assert zernike_variance(2) == pytest.approx(0.4479, rel=0.01)

    def test_defocus_matches_published_value(self):
        # Delta_3 - Delta_4 = 0.134 - 0.111 = 0.023
        assert zernike_variance(4) == pytest.approx(0.023, rel=0.02)

    def test_same_order_modes_share_variance(self):
        for j in (5, 6):
            assert zernike_variance(j) == pytest.approx(zernike_variance(4))

    def test_decreases_with_order(self):
        values = [zernike_variance(j) for j in (2, 4, 7, 11, 16, 22)]
        assert all(a > b for a, b in zip(values, values[1:], strict=False))

    def test_d_over_r0_scaling(self):
        assert zernike_variance(4, 2.0) / zernike_variance(4, 1.0) == pytest.approx(
            2.0 ** (5 / 3), rel=1e-12
        )

    def test_piston_rejected(self):
        with pytest.raises(ValueError, match="piston"):
            zernike_variance(1)

    def test_bad_d_over_r0(self):
        with pytest.raises(ValueError, match="d_over_r0"):
            zernike_variance(4, 0.0)

    def test_no_overflow_at_large_order(self):
        assert np.isfinite(zernike_variance(5000))
        assert zernike_variance(5000) > 0.0


class TestNollResidual:
    def test_total_variance_matches_delta_one(self):
        assert total_phase_variance() == pytest.approx(NOLL_RESIDUAL_TABLE[1], rel=0.01)

    @pytest.mark.parametrize("j", sorted(NOLL_RESIDUAL_TABLE))
    def test_matches_published_table(self, j):
        computed = noll_residual_variance(j)
        assert computed == pytest.approx(NOLL_RESIDUAL_TABLE[j], rel=0.01)

    def test_monotonically_decreasing(self):
        values = [noll_residual_variance(j) for j in range(1, 30)]
        assert all(a > b for a, b in zip(values, values[1:], strict=False))

    def test_scaling(self):
        assert noll_residual_variance(10, 3.0) / noll_residual_variance(10, 1.0) == pytest.approx(
            3.0 ** (5 / 3), rel=1e-12
        )

    def test_asymptote_close_at_j21(self):
        # Noll Eq. 32 is an asymptote, so a few percent is the honest tolerance
        assert noll_residual_asymptote(21) == pytest.approx(NOLL_RESIDUAL_TABLE[21], rel=0.05)

    def test_asymptote_slope(self):
        ratio = noll_residual_asymptote(200) / noll_residual_asymptote(100)
        assert ratio == pytest.approx(2.0 ** (-np.sqrt(3) / 2), rel=1e-12)

    def test_residual_below_total(self):
        assert noll_residual_variance(5) < total_phase_variance()

    @pytest.mark.parametrize("j", [0, -3, 2.5])
    def test_bad_j(self, j):
        with pytest.raises(ValueError, match="j_max"):
            noll_residual_variance(j)

    def test_asymptote_bad_j(self):
        with pytest.raises(ValueError, match="j_max"):
            noll_residual_asymptote(0)

    def test_total_variance_bad_n_max(self):
        with pytest.raises(ValueError, match="n_max"):
            total_phase_variance(1.0, 0)


class TestIntegratedQuantities:
    def test_fried_parameter_scaling(self):
        # r0 propto (Cn2 integral)^(-3/5)
        a = fried_parameter_from_cn2(1e-13, 1.55e-6)
        b = fried_parameter_from_cn2(2e-13, 1.55e-6)
        assert a / b == pytest.approx(2.0 ** (3 / 5), rel=1e-12)

    def test_fried_parameter_wavelength_scaling(self):
        # r0 propto lambda^(6/5)
        a = fried_parameter_from_cn2(1e-13, 0.5e-6)
        b = fried_parameter_from_cn2(1e-13, 1.0e-6)
        assert b / a == pytest.approx(2.0 ** (6 / 5), rel=1e-12)

    def test_fried_parameter_known_answer(self):
        # r0 = [0.423 k^2 J]^(-3/5); k = 2 pi / 500 nm = 1.2566371e7 rad/m,
        # k^2 = 1.5791367e14, J = 1e-13 m^(1/3)
        # -> 0.423 * 1.5791367e14 * 1e-13 = 6.6797483
        # r0 = 6.6797483^(-0.6) = 0.31999 m
        assert fried_parameter_from_cn2(1e-13, 500e-9) == pytest.approx(0.31999, rel=1e-4)

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_fried_bad_integral(self, value):
        with pytest.raises(ValueError, match="cn2_path_integral"):
            fried_parameter_from_cn2(value, 1e-6)

    def test_fried_bad_wavelength(self):
        with pytest.raises(ValueError, match="wavelength_m"):
            fried_parameter_from_cn2(1e-13, 0.0)

    def test_greenwood_frequency_known_answer(self):
        # f_G = 0.427 * 10 / 0.1 = 42.7 Hz
        assert greenwood_frequency(0.1, 10.0) == pytest.approx(42.7)

    def test_greenwood_frequency_zero_wind(self):
        assert greenwood_frequency(0.1, 0.0) == 0.0

    def test_greenwood_time_constant_known_answer(self):
        # tau0 = 0.314 * 0.1 / 10 = 3.14 ms
        assert greenwood_time_constant(0.1, 10.0) == pytest.approx(3.14e-3)

    def test_time_constant_inverse_wind(self):
        assert greenwood_time_constant(0.1, 20.0) == pytest.approx(
            0.5 * greenwood_time_constant(0.1, 10.0)
        )

    def test_time_constant_requires_positive_wind(self):
        with pytest.raises(ValueError, match="wind_speed_m_s"):
            greenwood_time_constant(0.1, 0.0)

    def test_greenwood_bad_r0(self):
        with pytest.raises(ValueError, match="r0_m"):
            greenwood_frequency(-1.0, 10.0)

    def test_greenwood_negative_wind(self):
        with pytest.raises(ValueError, match="wind_speed_m_s"):
            greenwood_frequency(0.1, -1.0)

    def test_time_constant_bad_r0(self):
        with pytest.raises(ValueError, match="r0_m"):
            greenwood_time_constant(0.0, 10.0)
