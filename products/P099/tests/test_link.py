"""The link budget: known answers, monotonicity, validity range, units."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from passyield.geometry import slant_range_spherical_km
from passyield.link import (
    DEFAULT_BUDGET,
    MINIMUM_VALID_ELEVATION_DEG,
    MINUS_TEN_LOG_K_DB,
    SPEED_OF_LIGHT_M_S,
    LinkBudget,
    carrier_to_noise_density_dbhz,
    free_space_loss_db,
    rate_mbit_s,
    slant_attenuation_db,
    spectral_efficiency_bits_per_hz,
)


def test_boltzmann_term_known_answer():
    """``-10*log10(1.380649e-23) = 228.5991671...`` dB, hand-checked.

    ``log10(1.380649e-23) = log10(1.380649) - 23 = 0.1398328... - 23``, so the
    term is ``10 * (23 - 0.1398328...) = 228.59916...`` dB. The Boltzmann
    constant is an exact SI defining constant since 2019, so this number
    carries no uncertainty.
    """
    assert MINUS_TEN_LOG_K_DB == pytest.approx(228.5991671732, rel=0.0, abs=1.0e-9)


def test_free_space_loss_known_answer_at_one_kilometre_and_one_metre():
    """Hand-calculated: ``d = 1`` km, ``lambda = 1`` m.

    ``L = 20*log10(4*pi*1000/1) = 20*log10(12566.3706...)``. Since
    ``log10(12566.3706) = 4.0992098...``, ``L = 81.984196...`` dB.
    """
    value = float(free_space_loss_db(1.0, 1.0))
    expected = 20.0 * math.log10(4.0 * math.pi * 1000.0)
    assert value == pytest.approx(expected, rel=0.0, abs=1.0e-12)
    assert value == pytest.approx(81.98419, rel=0.0, abs=1.0e-5)


def test_free_space_loss_grows_by_six_db_per_doubling_of_range():
    """``20*log10(2) = 6.0206`` dB, the textbook doubling rule."""
    near = float(free_space_loss_db(500.0, 0.0365))
    far = float(free_space_loss_db(1000.0, 0.0365))
    assert far - near == pytest.approx(20.0 * math.log10(2.0), rel=0.0, abs=1.0e-12)
    assert far - near == pytest.approx(6.0206, rel=0.0, abs=1.0e-4)


def test_slant_attenuation_known_answer_at_thirty_degrees():
    """``sin(30) = 0.5`` exactly, so the slant attenuation is twice the zenith.

    With a 0.4 dB zenith attenuation the slant value is 0.8 dB.
    """
    assert float(slant_attenuation_db(30.0, 0.4)) == pytest.approx(0.8, rel=0.0, abs=1.0e-12)


def test_slant_attenuation_at_the_zenith_equals_the_zenith_value():
    """``sin(90) = 1``, so the slant attenuation is the zenith attenuation."""
    assert float(slant_attenuation_db(90.0, 0.37)) == pytest.approx(0.37, rel=0.0, abs=1.0e-12)


@pytest.mark.parametrize("elevation", [0.0, 5.0, 9.999, -1.0, 90.001, 180.0])
def test_slant_attenuation_refuses_outside_its_validity_range(elevation):
    """The ITU-R P.618 secant law is not extrapolated below 10 degrees.

    Refusing is the documented behaviour: a model used outside the range its
    source states it in is an unsourced model.
    """
    with pytest.raises(ValueError, match="validity range|P.618"):
        slant_attenuation_db(elevation, 0.4)


def test_minimum_valid_elevation_is_ten_degrees():
    """The declared validity floor, so a change to it is visible in a diff."""
    assert MINIMUM_VALID_ELEVATION_DEG == 10.0


def test_wavelength_known_answer_at_x_band():
    """``lambda = c/f`` with ``c = 299792458`` m/s and ``f = 8.2`` GHz.

    ``299792458 / 8.2e9 = 0.0365600558...`` m.
    """
    assert DEFAULT_BUDGET.wavelength_m == pytest.approx(
        SPEED_OF_LIGHT_M_S / 8.2e9, rel=0.0, abs=0.0
    )
    assert DEFAULT_BUDGET.wavelength_m == pytest.approx(0.03656006, rel=0.0, abs=1.0e-8)


def test_rate_ceiling_known_answer():
    """``R_max = B * eta_max = 50 MHz * 5 bit/s/Hz = 250 Mbit/s``."""
    assert DEFAULT_BUDGET.max_rate_mbit_s == pytest.approx(250.0, rel=0.0, abs=0.0)


def test_rate_is_bounded_by_the_ceiling_across_the_whole_elevation_range():
    """The Shannon envelope is capped, so the rate never exceeds ``R_max``."""
    elevation = np.linspace(MINIMUM_VALID_ELEVATION_DEG, 90.0, 161)
    slant = slant_range_spherical_km(elevation, 400.0)
    rates = rate_mbit_s(elevation, slant, 24.0, 0.3)
    assert float(rates.max()) <= DEFAULT_BUDGET.max_rate_mbit_s + 1.0e-12
    assert float(rates.min()) > 0.0


@given(
    low=st.floats(min_value=10.0, max_value=89.0),
    gain=st.floats(min_value=14.0, max_value=30.0),
    attenuation=st.floats(min_value=0.0, max_value=3.0),
)
@settings(max_examples=150, deadline=None)
def test_rate_is_non_decreasing_in_elevation(low, gain, attenuation):
    """Property: slant range falls and attenuation falls as elevation rises.

    Non-decreasing and not strictly increasing, because the envelope
    saturates at ``eta_max`` and the saturated region is flat.
    """
    high = min(low + 1.0, 90.0)
    ranges = slant_range_spherical_km(np.asarray([low, high]), 400.0)
    rates = rate_mbit_s(np.asarray([low, high]), ranges, gain, attenuation)
    assert float(rates[1]) >= float(rates[0]) - 1.0e-9


def test_carrier_to_noise_rises_six_db_per_six_db_of_figure_of_merit():
    """``C/N0`` is linear in G/T in dB, by construction of the link equation."""
    slant = float(slant_range_spherical_km(30.0, 400.0))
    low = float(carrier_to_noise_density_dbhz(30.0, slant, 18.0, 0.4))
    high = float(carrier_to_noise_density_dbhz(30.0, slant, 24.0, 0.4))
    assert high - low == pytest.approx(6.0, rel=0.0, abs=1.0e-9)


def test_spectral_efficiency_is_dimensionless_and_bounded():
    """``eta`` lies in ``(0, eta_max]`` over the whole valid elevation range."""
    elevation = np.linspace(10.0, 90.0, 81)
    slant = slant_range_spherical_km(elevation, 400.0)
    efficiency = spectral_efficiency_bits_per_hz(elevation, slant, 21.0, 0.45)
    assert float(efficiency.min()) > 0.0
    assert float(efficiency.max()) <= DEFAULT_BUDGET.max_spectral_efficiency_bits_per_hz + 1e-12


def test_rate_is_bandwidth_times_efficiency_exactly():
    """``R = B * eta``: the units of the product are MHz times bit/s/Hz."""
    slant = float(slant_range_spherical_km(25.0, 400.0))
    efficiency = float(spectral_efficiency_bits_per_hz(25.0, slant, 21.0, 0.45))
    rate = float(rate_mbit_s(25.0, slant, 21.0, 0.45))
    assert rate == pytest.approx(DEFAULT_BUDGET.bandwidth_mhz * efficiency, rel=1.0e-14, abs=0.0)


@pytest.mark.parametrize(
    ("eirp", "frequency", "bandwidth", "margin", "efficiency"),
    [
        (6.0, 0.0, 50.0, 3.0, 5.0),
        (6.0, 8.2e9, 0.0, 3.0, 5.0),
        (6.0, 8.2e9, 50.0, -1.0, 5.0),
        (6.0, 8.2e9, 50.0, 3.0, 0.0),
        (float("inf"), 8.2e9, 50.0, 3.0, 5.0),
    ],
)
def test_link_budget_rejects_out_of_range_inputs(eirp, frequency, bandwidth, margin, efficiency):
    """Input validation on every budget field."""
    with pytest.raises(ValueError):
        LinkBudget(eirp, frequency, bandwidth, margin, efficiency)


def test_free_space_loss_rejects_a_non_positive_range():
    """Input validation: the Friis formula has a logarithm of the range."""
    with pytest.raises(ValueError, match="slant_range_km"):
        free_space_loss_db(0.0, 0.0365)


def test_free_space_loss_rejects_a_non_positive_wavelength():
    """Input validation: a wavelength is positive."""
    with pytest.raises(ValueError, match="wavelength_m"):
        free_space_loss_db(500.0, 0.0)


def test_slant_attenuation_rejects_a_negative_zenith_value():
    """Input validation: an attenuation is non-negative."""
    with pytest.raises(ValueError, match="zenith_attenuation_db"):
        slant_attenuation_db(30.0, -0.1)
