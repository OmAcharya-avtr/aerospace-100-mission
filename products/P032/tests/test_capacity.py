"""RF and optical capacity models."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from constellink.capacity import (
    BOLTZMANN_DB_W_K_HZ,
    SPEED_OF_LIGHT_M_S,
    OpticalTerminal,
    RfTerminal,
    free_space_path_loss_db,
    hybrid_capacity_bps,
    kim_specific_attenuation_db_km,
    optical_link,
    rf_link,
    slant_path_attenuation_db,
)


def rf_term(**kwargs):
    base = {"tx_power_dbw": 3.0, "tx_gain_dbi": 30.0,
            "rx_g_over_t_db_per_k": 15.0, "frequency_hz": 26.0e9,
            "bandwidth_hz": 500.0e6, "required_ebn0_db": 4.5}
    base.update(kwargs)
    return RfTerminal(**base)


def opt_term(**kwargs):
    base = {"tx_power_w": 1.0, "wavelength_m": 1550e-9,
            "beam_divergence_full_rad": 40e-6,
            "rx_aperture_diameter_m": 0.08, "photons_per_bit": 500.0}
    base.update(kwargs)
    return OpticalTerminal(**base)


def test_boltzmann_constant_in_db():
    # k = 1.380649e-23 J/K exactly (SI, 2019); 10 log10 k = -228.5991 dB(W/K/Hz).
    assert pytest.approx(-228.5991, abs=1e-4) == BOLTZMANN_DB_W_K_HZ


def test_fspl_hand_value_1km_1ghz():
    # L = 20 log10(4 pi * 1000 m / 0.299792458 m) = 92.4478 dB.
    assert free_space_path_loss_db(1.0, 1.0e9) == pytest.approx(92.4478, abs=1e-4)


def test_fspl_doubles_range_adds_6db():
    a = free_space_path_loss_db(1000.0, 26.0e9)
    b = free_space_path_loss_db(2000.0, 26.0e9)
    assert b - a == pytest.approx(20.0 * math.log10(2.0), abs=1e-12)


@pytest.mark.parametrize("args", [(0.0, 1e9), (-1.0, 1e9), (100.0, 0.0),
                                  (100.0, -1e9)])
def test_fspl_rejects_bad_input(args):
    with pytest.raises(ValueError):
        free_space_path_loss_db(*args)


def test_eirp_definition():
    term = rf_term(tx_power_dbw=3.0, tx_gain_dbi=30.0, tx_loss_db=1.0)
    assert term.eirp_dbw == pytest.approx(32.0)


@pytest.mark.parametrize("kwargs", [{"frequency_hz": 0.0}, {"bandwidth_hz": -1.0},
                                    {"tx_loss_db": -0.1},
                                    {"other_loss_db": -0.1}, {"margin_db": -0.1}])
def test_rf_terminal_validation(kwargs):
    with pytest.raises(ValueError):
        rf_term(**kwargs)


def test_rf_link_reduces_to_friis_with_losses_zeroed():
    # With T = 1 K, G/T is numerically G_rx, so P_rx = C/N0 + k.
    p_tx_w, g_t, g_r, r_km, f_hz = 2.0, 1000.0, 10000.0, 1500.0, 26.0e9
    term = rf_term(tx_power_dbw=10.0 * math.log10(p_tx_w),
                   tx_gain_dbi=10.0 * math.log10(g_t),
                   rx_g_over_t_db_per_k=10.0 * math.log10(g_r),
                   frequency_hz=f_hz, tx_loss_db=0.0, other_loss_db=0.0,
                   margin_db=0.0)
    res = rf_link(r_km, term)
    p_rx_chain = 10.0 ** ((res.c_over_n0_dbhz + BOLTZMANN_DB_W_K_HZ) / 10.0)
    lam = SPEED_OF_LIGHT_M_S / f_hz
    p_rx_friis = p_tx_w * g_t * g_r * (lam / (4.0 * math.pi * r_km * 1e3)) ** 2
    assert p_rx_chain == pytest.approx(p_rx_friis, rel=1e-12)


def test_rf_achievable_rate_follows_the_ebn0_relation():
    term = rf_term(required_ebn0_db=4.5, margin_db=3.0)
    res = rf_link(1500.0, term)
    expected = 10.0 ** ((res.c_over_n0_dbhz - 4.5 - 3.0) / 10.0)
    assert res.achievable_rate_bps == pytest.approx(expected, rel=1e-12)


def test_rf_shannon_bound_exceeds_the_achievable_rate():
    res = rf_link(1500.0, rf_term())
    assert res.shannon_capacity_bps > res.achievable_rate_bps


def test_rf_extra_loss_reduces_the_rate_by_the_right_factor():
    base = rf_link(1500.0, rf_term())
    worse = rf_link(1500.0, rf_term(), extra_loss_db=3.0)
    assert worse.achievable_rate_bps == pytest.approx(
        base.achievable_rate_bps / 10.0 ** 0.3, rel=1e-12)


def test_rf_link_rejects_negative_extra_loss():
    with pytest.raises(ValueError):
        rf_link(1500.0, rf_term(), extra_loss_db=-1.0)


def test_rf_closes_flag():
    assert rf_link(1500.0, rf_term()).closes
    assert not rf_link(1.0e9, rf_term()).closes


@given(r_km=st.floats(100.0, 40000.0))
@settings(max_examples=40, deadline=None)
def test_rf_rate_is_inverse_square_in_range(r_km):
    # Algebraic identity: the achievable rate scales as 1/R^2.
    a = rf_link(r_km, rf_term()).achievable_rate_bps
    b = rf_link(2.0 * r_km, rf_term()).achievable_rate_bps
    assert b == pytest.approx(a / 4.0, rel=1e-9)


def test_optical_geometric_capture_closed_form():
    term = opt_term(rx_aperture_diameter_m=0.08)
    res = optical_link(1000.0, term)
    w = (40e-6 / 2.0) * 1e6
    a = 0.04
    assert res.beam_radius_m == pytest.approx(w, rel=1e-12)
    assert res.geometric_capture_fraction == pytest.approx(
        1.0 - math.exp(-2.0 * a ** 2 / w ** 2), rel=1e-12)


def test_optical_pointing_loss_closed_form():
    half = 40e-6 / 2.0
    term = opt_term(pointing_error_rad=0.5 * half)
    res = optical_link(1000.0, term)
    assert res.pointing_loss_db == pytest.approx(
        (20.0 / math.log(10.0)) * 0.25, rel=1e-12)


def test_optical_zero_pointing_error_is_zero_loss():
    assert optical_link(1000.0, opt_term()).pointing_loss_db == 0.0


def test_optical_photon_energy():
    # E = h c / lambda; at 1550 nm this is 1.2820e-19 J.
    res = optical_link(1000.0, opt_term())
    assert res.photon_energy_j == pytest.approx(1.2820e-19, rel=1e-3)


def test_optical_rate_follows_the_photon_relation():
    term = opt_term(margin_db=0.0)
    res = optical_link(1000.0, term)
    expected = res.rx_power_w / (res.photon_energy_j * 500.0)
    assert res.achievable_rate_bps == pytest.approx(expected, rel=1e-12)


def test_optical_doubling_photons_per_bit_halves_the_rate():
    a = optical_link(1000.0, opt_term(photons_per_bit=500.0)).achievable_rate_bps
    b = optical_link(1000.0, opt_term(photons_per_bit=1000.0)).achievable_rate_bps
    assert b == pytest.approx(a / 2.0, rel=1e-12)


def test_optical_atmospheric_loss_applies():
    a = optical_link(1000.0, opt_term()).achievable_rate_bps
    b = optical_link(1000.0, opt_term(),
                     atmospheric_loss_db=3.0).achievable_rate_bps
    assert b == pytest.approx(a / 10.0 ** 0.3, rel=1e-12)


@pytest.mark.parametrize("kwargs", [{"tx_power_w": 0.0}, {"wavelength_m": -1.0},
                                    {"beam_divergence_full_rad": 0.0},
                                    {"rx_aperture_diameter_m": 0.0},
                                    {"photons_per_bit": 0.0},
                                    {"pointing_error_rad": -1e-6},
                                    {"margin_db": -1.0}])
def test_optical_terminal_validation(kwargs):
    with pytest.raises(ValueError):
        opt_term(**kwargs)


@pytest.mark.parametrize("args", [(0.0, None), (-1.0, None)])
def test_optical_link_rejects_bad_range(args):
    with pytest.raises(ValueError):
        optical_link(args[0], opt_term())


def test_optical_link_rejects_negative_atmospheric_loss():
    with pytest.raises(ValueError):
        optical_link(1000.0, opt_term(), atmospheric_loss_db=-1.0)


def test_kim_attenuation_monotone_in_visibility():
    vals = [kim_specific_attenuation_db_km(v, 1550.0)
            for v in (0.2, 0.5, 1.0, 5.0, 20.0, 60.0)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def test_kim_attenuation_hand_value_clear_air():
    # V = 60 km, so q = 1.6; lambda = 1550 nm, so (1550/550)^-1.6:
    #   ln(2.818182) = 1.036092, x -1.6 = -1.657747, exp = 0.190568
    #   beta = (3.912 / 60) * 0.190568 = 0.0652 * 0.190568 = 0.0124250 dB/km
    assert kim_specific_attenuation_db_km(60.0, 1550.0) == pytest.approx(
        0.0124250, abs=1e-6)


def test_kim_dense_fog_is_wavelength_independent():
    # For V <= 0.5 km the exponent q is 0, so attenuation is the same at any
    # wavelength -- the well-known "fog is grey" result of the Kim model.
    a = kim_specific_attenuation_db_km(0.3, 850.0)
    b = kim_specific_attenuation_db_km(0.3, 1550.0)
    assert a == pytest.approx(b, rel=1e-14)


@pytest.mark.parametrize("args", [(0.0, 1550.0), (-1.0, 1550.0), (10.0, 0.0)])
def test_kim_rejects_bad_input(args):
    with pytest.raises(ValueError):
        kim_specific_attenuation_db_km(*args)


def test_slant_path_cosecant_law():
    assert slant_path_attenuation_db(1.0, 30.0) == pytest.approx(2.0, rel=1e-12)
    assert slant_path_attenuation_db(2.0, 90.0) == pytest.approx(2.0, rel=1e-12)


def test_slant_path_refuses_below_validity_elevation():
    with pytest.raises(ValueError, match="plane-parallel"):
        slant_path_attenuation_db(1.0, 5.0)


@pytest.mark.parametrize("args", [(-1.0, 30.0), (1.0, 95.0)])
def test_slant_path_rejects_bad_input(args):
    with pytest.raises(ValueError):
        slant_path_attenuation_db(*args)


def test_hybrid_selects_the_better_leg():
    rate_rf = rf_link(1500.0, rf_term()).achievable_rate_bps
    rate_opt = optical_link(1500.0, opt_term()).achievable_rate_bps
    assert hybrid_capacity_bps(1500.0, rf=rf_term()) == pytest.approx(rate_rf)
    assert hybrid_capacity_bps(1500.0, optical=opt_term()) == pytest.approx(
        rate_opt)
    assert hybrid_capacity_bps(1500.0, rf=rf_term(),
                               optical=opt_term()) == pytest.approx(
        max(rate_rf, rate_opt))


def test_hybrid_requires_a_terminal():
    with pytest.raises(ValueError, match="at least one"):
        hybrid_capacity_bps(1500.0)
