"""Unit and known-answer tests for beamtwin.budget."""

from __future__ import annotations

import math

import pytest

from beamtwin.budget import (
    LinkParams,
    beam_radius,
    compute_budget,
    db_from_fraction,
    dbm_from_watts,
    gaussian_divergence_half_angle,
    geometric_capture_fraction,
    kim_attenuation_db_per_km,
    pointing_loss_fraction,
    rayleigh_range,
    watts_from_dbm,
)


class TestUnitConversions:
    def test_dbm_to_watts_known_answer(self):
        # 30 dBm = 1 W by definition (10^(30/10) mW = 1000 mW).
        assert watts_from_dbm(30.0) == pytest.approx(1.0)

    def test_watts_to_dbm_known_answer(self):
        # 0.1 W = 100 mW = 20 dBm.
        assert dbm_from_watts(0.1) == pytest.approx(20.0)

    def test_dbm_watts_roundtrip(self):
        for p in (-40.0, -3.0, 0.0, 17.5, 33.0):
            assert dbm_from_watts(watts_from_dbm(p)) == pytest.approx(p, abs=1e-12)

    def test_db_from_fraction_known_answers(self):
        # -10log10(0.5) = 3.0103 dB; -10log10(1) = 0 dB.
        assert db_from_fraction(0.5) == pytest.approx(3.010299956639812)
        assert db_from_fraction(1.0) == pytest.approx(0.0)

    def test_db_from_fraction_rejects_zero(self):
        with pytest.raises(ValueError, match="fraction"):
            db_from_fraction(0.0)

    def test_db_from_fraction_rejects_above_one(self):
        with pytest.raises(ValueError, match="fraction"):
            db_from_fraction(1.5)

    def test_dbm_from_watts_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            dbm_from_watts(0.0)


class TestGaussianBeam:
    def test_divergence_known_answer(self):
        # theta = lambda/(pi w0) = 1.55e-6/(pi*0.02) = 2.4670e-5 rad.
        theta = gaussian_divergence_half_angle(1550e-9, 0.02)
        assert theta == pytest.approx(1.55e-6 / (math.pi * 0.02))
        assert theta == pytest.approx(2.46700e-5, rel=1e-4)

    def test_rayleigh_range_known_answer(self):
        # z_R = pi w0^2/lambda = pi*4e-4/1.55e-6 = 810.7336 m.
        assert rayleigh_range(1550e-9, 0.02) == pytest.approx(810.7336, rel=1e-6)

    def test_beam_radius_at_waist_is_w0(self):
        # w(z) -> w0 as z -> 0.
        assert beam_radius(1550e-9, 0.02, 1e-9) == pytest.approx(0.02, rel=1e-12)

    def test_beam_radius_at_rayleigh_range_is_sqrt2_w0(self):
        z_r = rayleigh_range(1550e-9, 0.02)
        assert beam_radius(1550e-9, 0.02, z_r) == pytest.approx(0.02 * math.sqrt(2))

    def test_beam_radius_known_answer_10km(self):
        # Hand-check (validation V1): w(10 km) = 0.247500 m.
        assert beam_radius(1550e-9, 0.02, 10_000.0) == pytest.approx(0.2475, rel=1e-4)

    def test_beam_radius_far_field_matches_divergence(self):
        # In the far field w(z) -> theta * z.
        theta = gaussian_divergence_half_angle(1550e-9, 0.02)
        z = 1e6
        assert beam_radius(1550e-9, 0.02, z) == pytest.approx(theta * z, rel=1e-6)

    def test_beam_radius_monotone_in_range(self):
        radii = [beam_radius(1550e-9, 0.02, r) for r in (100.0, 1000.0, 5000.0, 20000.0)]
        assert all(b > a for a, b in zip(radii, radii[1:]))

    def test_divergence_rejects_out_of_band_wavelength(self):
        with pytest.raises(ValueError, match="optical band"):
            gaussian_divergence_half_angle(1550.0, 0.02)  # metres vs nm mistake

    def test_divergence_rejects_negative_waist(self):
        with pytest.raises(ValueError):
            gaussian_divergence_half_angle(1550e-9, -0.02)


class TestGeometricCapture:
    def test_capture_known_answer(self):
        # eta = 1-exp(-2*0.05^2/0.2475^2) = 0.078382 (validation V1 step 3).
        assert geometric_capture_fraction(0.2475, 0.05) == pytest.approx(0.078382, rel=1e-4)

    def test_capture_tends_to_one_for_huge_aperture(self):
        assert geometric_capture_fraction(0.1, 10.0) == pytest.approx(1.0)

    def test_capture_small_aperture_limit(self):
        # For a << w, eta ~= 2a^2/w^2.
        w, a = 1.0, 0.001
        assert geometric_capture_fraction(w, a) == pytest.approx(2 * a**2 / w**2, rel=1e-4)

    def test_capture_in_unit_interval(self):
        for a in (0.001, 0.01, 0.1, 1.0):
            eta = geometric_capture_fraction(0.5, a)
            assert 0.0 < eta < 1.0

    def test_capture_monotone_in_aperture(self):
        etas = [geometric_capture_fraction(0.25, a) for a in (0.01, 0.05, 0.1, 0.2)]
        assert all(b > a for a, b in zip(etas, etas[1:]))

    def test_capture_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            geometric_capture_fraction(0.0, 0.05)


class TestPointingLoss:
    def test_zero_offset_is_lossless(self):
        assert pointing_loss_fraction(0.0, 0.25) == pytest.approx(1.0)

    def test_pointing_loss_known_answer(self):
        # L_p = exp(-2*0.02^2/0.2475^2) = 0.987025 (validation V1 step 4).
        assert pointing_loss_fraction(0.02, 0.2475) == pytest.approx(0.987025, rel=1e-5)

    def test_pointing_loss_monotone_decreasing(self):
        vals = [pointing_loss_fraction(d, 0.25) for d in (0.0, 0.05, 0.1, 0.2)]
        assert all(b < a for a, b in zip(vals, vals[1:]))

    def test_pointing_loss_rejects_negative_offset(self):
        with pytest.raises(ValueError, match="offset_m"):
            pointing_loss_fraction(-0.01, 0.25)


class TestKimModel:
    def test_kim_known_answer_v7km_1550nm(self):
        # Validation V1 step 8: 0.630817 dB/km.
        assert kim_attenuation_db_per_km(7.0, 1550e-9) == pytest.approx(0.630817, rel=1e-5)

    def test_kim_dense_fog_is_wavelength_independent(self):
        # V <= 0.5 km -> q = 0, so attenuation must not depend on wavelength.
        a1 = kim_attenuation_db_per_km(0.3, 850e-9)
        a2 = kim_attenuation_db_per_km(0.3, 1550e-9)
        assert a1 == pytest.approx(a2)

    def test_kim_dense_fog_known_answer(self):
        # V=0.2 km, q=0: alpha = 4.343*3.91/0.2 = 84.90 dB/km.
        expected = (10.0 / math.log(10.0)) * (3.91 / 0.2)
        assert kim_attenuation_db_per_km(0.2, 1550e-9) == pytest.approx(expected)

    def test_kim_decreases_with_visibility(self):
        vals = [kim_attenuation_db_per_km(v, 1550e-9) for v in (0.5, 1.0, 5.0, 20.0, 60.0)]
        assert all(b < a for a, b in zip(vals, vals[1:]))

    def test_kim_1550_better_than_850_in_haze(self):
        # In haze (q > 0) the longer wavelength suffers less attenuation.
        assert kim_attenuation_db_per_km(10.0, 1550e-9) < kim_attenuation_db_per_km(10.0, 850e-9)

    def test_kim_rejects_nonpositive_visibility(self):
        with pytest.raises(ValueError):
            kim_attenuation_db_per_km(0.0, 1550e-9)


class TestLinkParamsValidation:
    def test_defaults_construct(self):
        assert LinkParams().wavelength_m == pytest.approx(1550e-9)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"wavelength_m": 0.0},
            {"wavelength_m": 1550.0},
            {"tx_efficiency": 0.0},
            {"tx_efficiency": 1.5},
            {"rx_efficiency": -0.1},
            {"beam_waist_radius_m": 0.0},
            {"rx_aperture_radius_m": -1.0},
            {"range_m": 0.0},
            {"pointing_bias_rad": -1e-6},
            {"attenuation_db_per_km": -0.1},
            {"tx_power_dbm": 80.0},
        ],
    )
    def test_invalid_params_raise(self, kwargs):
        with pytest.raises((ValueError, TypeError)):
            LinkParams(**kwargs)

    def test_nan_wavelength_raises(self):
        with pytest.raises((ValueError, TypeError)):
            LinkParams(wavelength_m=float("nan"))

    def test_string_input_raises_type_error(self):
        with pytest.raises((TypeError, ValueError)):
            LinkParams(range_m="10000")


class TestComputeBudget:
    def test_budget_known_answer_10km(self):
        # Full hand-check in validation/v1_budget_handcheck.py.
        p = LinkParams(range_m=10_000.0, attenuation_db_per_km=2.5, rx_sensitivity_dbm=-30.0,
                       pointing_bias_rad=2e-6)
        b = compute_budget(p)
        assert b.geometric_loss_db == pytest.approx(11.057829, rel=1e-6)
        assert b.pointing_loss_db == pytest.approx(0.056719, rel=1e-5)
        assert b.atmospheric_loss_db == pytest.approx(25.0)
        assert b.received_power_dbm == pytest.approx(-18.052748, rel=1e-6)
        assert b.margin_db == pytest.approx(11.947252, rel=1e-6)

    def test_terms_sum_to_received_power(self):
        b = compute_budget(LinkParams(range_m=5000.0))
        total = (
            b.params.tx_power_dbm
            - b.tx_optics_loss_db
            - b.geometric_loss_db
            - b.pointing_loss_db
            - b.atmospheric_loss_db
            - b.rx_optics_loss_db
        )
        assert total == pytest.approx(b.received_power_dbm)

    def test_margin_definition(self):
        b = compute_budget(LinkParams(range_m=5000.0))
        assert b.margin_db == pytest.approx(b.received_power_dbm - b.params.rx_sensitivity_dbm)

    def test_all_losses_non_negative(self):
        b = compute_budget(LinkParams(range_m=8000.0, pointing_bias_rad=3e-6))
        for loss in (
            b.tx_optics_loss_db,
            b.geometric_loss_db,
            b.pointing_loss_db,
            b.atmospheric_loss_db,
            b.rx_optics_loss_db,
        ):
            assert loss >= 0.0

    def test_negative_margin_flagged(self):
        # Failure mode: sensitivity above achievable power -> flagged, not silent.
        b = compute_budget(LinkParams(range_m=20_000.0, rx_sensitivity_dbm=10.0))
        assert b.margin_db < 0.0
        assert b.margin_negative is True

    def test_positive_margin_not_flagged(self):
        b = compute_budget(LinkParams(range_m=1000.0, rx_sensitivity_dbm=-40.0))
        assert b.margin_negative is False

    def test_margin_decreases_with_range(self):
        margins = [compute_budget(LinkParams(range_m=r)).margin_db
                   for r in (1000.0, 5000.0, 10_000.0, 20_000.0)]
        assert all(b < a for a, b in zip(margins, margins[1:]))

    def test_margin_increases_with_tx_power(self):
        m1 = compute_budget(LinkParams(range_m=5000.0, tx_power_dbm=10.0)).margin_db
        m2 = compute_budget(LinkParams(range_m=5000.0, tx_power_dbm=20.0)).margin_db
        assert m2 - m1 == pytest.approx(10.0, abs=1e-9)

    def test_as_dict_is_json_serialisable(self):
        import json

        d = compute_budget(LinkParams()).as_dict()
        json.loads(json.dumps(d))
        assert "margin_db" in d and "margin_negative" in d
