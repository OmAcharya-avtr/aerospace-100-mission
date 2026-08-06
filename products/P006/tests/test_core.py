"""Known-answer and edge-case tests for the LinkBudget model.

Hand calculations are shown in comments; the same cases (with full working)
are in validation/VALIDATION.md.
"""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from linkbudgetx import LinkBudget


def nominal(**overrides):
    """Baseline 10 km terrestrial link used across tests."""
    params = dict(
        tx_power_dbm=20.0,
        wavelength_nm=1550.0,
        beam_divergence_rad=1.0e-3,  # FULL angle
        range_km=10.0,
        rx_aperture_diameter_m=0.1,
        rx_sensitivity_dbm=-40.0,
        tx_optics_efficiency=0.8,
        rx_optics_efficiency=0.8,
        pointing_error_rad=0.0,
        atmos_attenuation_db_per_km=0.5,
        beam_profile="gaussian",
    )
    params.update(overrides)
    return LinkBudget(**params)


class TestGeometricLoss:
    def test_flattop_known_answer(self):
        # Hand calc: theta_full = 1 mrad, R = 10 km -> spot diameter = 10 m.
        # Aperture D = 0.1 m -> fraction = (0.1/10)^2 = 1e-4 -> loss = 40.000 dB.
        b = nominal(beam_profile="flattop")
        assert b.capture_fraction() == pytest.approx(1e-4, rel=1e-12)
        assert b.geometric_loss_db() == pytest.approx(40.0, abs=1e-9)

    def test_gaussian_known_answer(self):
        # Hand calc: theta_half = 0.5 mrad, R = 10 km -> w = 5 m, a = 0.05 m.
        # f = 1 - exp(-2*(0.05/5)^2) = 1 - exp(-2e-4) = 1.999800e-4
        # loss = -10 log10(1.999800e-4) = 36.9901 dB.
        b = nominal()
        f = 1.0 - math.exp(-2.0e-4)
        assert b.capture_fraction() == pytest.approx(f, rel=1e-12)
        assert b.geometric_loss_db() == pytest.approx(-10.0 * math.log10(f), abs=1e-9)
        assert b.geometric_loss_db() == pytest.approx(36.9901, abs=5e-4)

    def test_flattop_capture_saturates_at_unity(self):
        # Aperture larger than the spot: fraction capped at 1 -> 0 dB loss.
        b = nominal(beam_profile="flattop", range_km=0.001, rx_aperture_diameter_m=1.0)
        assert b.capture_fraction() == 1.0
        assert b.geometric_loss_db() == pytest.approx(0.0, abs=1e-12)

    def test_gaussian_more_loss_than_flattop_for_small_aperture(self):
        # For a << w a Gaussian spreads power over a larger effective area
        # relative to its 1/e^2 edge: f_gauss ~= 2(a/w)^2 vs f_flat = (a/w)^2
        # with SAME w, so Gaussian captures MORE. Check the documented ratio.
        g = nominal().capture_fraction()
        f = nominal(beam_profile="flattop").capture_fraction()
        assert g / f == pytest.approx(2.0, rel=1e-3)  # small-aperture limit


class TestPointingLoss:
    def test_zero_error_is_zero_db(self):
        assert nominal(pointing_error_rad=0.0).pointing_loss_db() == 0.0

    def test_half_angle_offset_known_answer(self):
        # Hand calc: theta_err = theta_half -> L = -10 log10(exp(-2))
        # = 20 log10(e) = 8.6859 dB.
        b = nominal(pointing_error_rad=0.5e-3)  # = full/2 = half angle
        assert b.pointing_loss_db() == pytest.approx(20.0 * math.log10(math.e), abs=1e-9)
        assert b.pointing_loss_db() == pytest.approx(8.6859, abs=5e-4)

    def test_quarter_of_full_angle(self):
        # theta_err = theta_full/4 = theta_half/2 -> exp(-0.5) -> 2.1715 dB.
        b = nominal(pointing_error_rad=0.25e-3)
        assert b.pointing_loss_db() == pytest.approx(5.0 * math.log10(math.e), abs=1e-9)

    @given(st.floats(min_value=0.0, max_value=5e-3))
    def test_monotone_in_error(self, err):
        # Loss is quadratic (hence monotone non-decreasing) in the offset.
        b1 = nominal(pointing_error_rad=err)
        b2 = nominal(pointing_error_rad=err + 1e-4)
        assert b2.pointing_loss_db() >= b1.pointing_loss_db()


class TestFullBudget:
    def test_budget_is_sum_of_terms(self):
        r = nominal(pointing_error_rad=0.25e-3).compute()
        expected_rx = (
            r.tx_power_dbm
            - r.tx_optics_loss_db
            - r.geometric_loss_db
            - r.pointing_loss_db
            - r.atmospheric_loss_db
            - r.rx_optics_loss_db
        )
        assert r.rx_power_dbm == pytest.approx(expected_rx, abs=1e-12)
        assert r.margin_db == pytest.approx(r.rx_power_dbm - r.rx_sensitivity_dbm, abs=1e-12)

    def test_known_answer_full_budget(self):
        # Hand calc (validation/VALIDATION.md case 4):
        # optics loss each: -10log10(0.8) = 0.9691 dB; geometric 36.9901 dB;
        # pointing (0.25 mrad) 2.1715 dB; atmos 0.5*10 = 5 dB.
        # Rx = 20 - 0.9691 - 36.9901 - 2.1715 - 5 - 0.9691 = -26.0997 dBm
        # margin = -26.0997 - (-40) = 13.9003 dB.
        r = nominal(pointing_error_rad=0.25e-3).compute()
        assert r.atmospheric_loss_db == pytest.approx(5.0, abs=1e-12)
        assert r.tx_optics_loss_db == pytest.approx(0.9691, abs=5e-5)
        assert r.rx_power_dbm == pytest.approx(-26.0997, abs=5e-4)
        assert r.margin_db == pytest.approx(13.9003, abs=5e-4)

    def test_atmospheric_loss_scales_with_range(self):
        assert nominal(range_km=20.0).atmospheric_loss_db() == pytest.approx(10.0)

    def test_result_dict_and_table(self):
        r = nominal().compute()
        d = r.as_dict()
        assert set(d) >= {"margin_db", "rx_power_dbm", "geometric_loss_db"}
        table = r.format_table()
        assert "Link margin" in table and "dB" in table

    def test_replace_returns_validated_copy(self):
        b = nominal()
        b2 = b.replace(range_km=5.0)
        assert b2.range_km == 5.0 and b.range_km == 10.0
        with pytest.raises(ValueError):
            b.replace(range_km=-1.0)


class TestInputValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"range_km": -1.0},
            {"range_km": 0.0},
            {"beam_divergence_rad": 0.0},
            {"beam_divergence_rad": -1e-3},
            {"wavelength_nm": 0.0},
            {"wavelength_nm": -1550.0},
            {"rx_aperture_diameter_m": 0.0},
            {"tx_optics_efficiency": 0.0},
            {"tx_optics_efficiency": 1.2},
            {"rx_optics_efficiency": -0.1},
            {"pointing_error_rad": -1e-4},
            {"atmos_attenuation_db_per_km": -0.5},
            {"beam_profile": "bessel"},
            {"tx_power_dbm": float("nan")},
            {"rx_sensitivity_dbm": float("inf")},
        ],
    )
    def test_invalid_inputs_raise_value_error(self, overrides):
        with pytest.raises(ValueError):
            nominal(**overrides)
