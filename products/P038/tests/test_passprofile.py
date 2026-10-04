"""Pass profiles and event locators."""

from __future__ import annotations

import numpy as np
import pytest

from dopplerkit.analytic import CircularOverheadPass
from dopplerkit.constants import WGS84_A_M
from dopplerkit.doppler import LinkDirection
from dopplerkit.passprofile import (
    compute_profile,
    doppler_zero_crossing_s,
    time_of_closest_approach_s,
)

CARRIER_HZ = 2.2e9


@pytest.fixture
def orbit() -> CircularOverheadPass:
    return CircularOverheadPass(orbit_radius_m=WGS84_A_M + 500.0e3)


def test_profile_shapes_and_conventions(orbit):
    horizon = orbit.horizon_time_s()
    times = np.linspace(-horizon, horizon, 31)
    p = compute_profile(orbit.observer_state, orbit.satellite_state, times, CARRIER_HZ)
    assert len(p) == 31
    for arr in (p.range_m, p.range_rate_mps, p.doppler_hz, p.doppler_rate_hz_per_s):
        assert arr.shape == (31,)
    assert p.precomp_offset_hz is not None
    assert "receding" in p.range_rate_convention
    assert "approaching" in p.doppler_convention
    assert p.ways == 1
    assert p.direction is LinkDirection.DOWNLINK
    assert p.carrier_hz == pytest.approx(CARRIER_HZ)


def test_profile_matches_the_closed_form_elementwise(orbit):
    times = np.linspace(-300.0, 300.0, 25)
    p = compute_profile(orbit.observer_state, orbit.satellite_state, times, CARRIER_HZ)
    assert np.allclose(p.range_m, orbit.range_m(times), rtol=1e-12)
    assert np.allclose(p.range_rate_mps, orbit.range_rate_mps(times), rtol=1e-9, atol=1e-9)
    assert np.allclose(
        p.doppler_hz, orbit.one_way_doppler_hz(times, CARRIER_HZ), rtol=1e-9, atol=1e-6
    )


def test_two_way_profile_is_exactly_twice_the_one_way_profile(orbit):
    times = np.linspace(-300.0, 300.0, 11)
    one = compute_profile(orbit.observer_state, orbit.satellite_state, times, CARRIER_HZ)
    two = compute_profile(
        orbit.observer_state, orbit.satellite_state, times, CARRIER_HZ,
        direction=LinkDirection.TWO_WAY,
    )
    assert np.array_equal(two.doppler_hz, 2.0 * one.doppler_hz)
    assert np.array_equal(two.doppler_rate_hz_per_s, 2.0 * one.doppler_rate_hz_per_s)
    assert two.ways == 2
    assert two.precomp_offset_hz is None


def test_precomp_is_the_negative_of_the_doppler_array(orbit):
    times = np.linspace(-300.0, 300.0, 11)
    p = compute_profile(orbit.observer_state, orbit.satellite_state, times, CARRIER_HZ)
    assert np.allclose(p.precomp_offset_hz, -p.doppler_hz, rtol=1e-15, atol=1e-9)


def test_peak_and_span(orbit):
    horizon = orbit.horizon_time_s()
    times = np.linspace(-horizon, horizon, 101)
    p = compute_profile(orbit.observer_state, orbit.satellite_state, times, CARRIER_HZ)
    assert p.peak_doppler_hz() == pytest.approx(
        orbit.one_way_doppler_hz(-horizon, CARRIER_HZ), rel=1e-9
    )
    assert p.doppler_span_hz() == pytest.approx(2.0 * abs(p.peak_doppler_hz()), rel=1e-9)
    assert p.doppler_span_hz() > 0.0


def test_times_validation(orbit):
    with pytest.raises(ValueError, match="length >= 2"):
        compute_profile(orbit.observer_state, orbit.satellite_state, [0.0], CARRIER_HZ)
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_profile(
            orbit.observer_state, orbit.satellite_state, [0.0, 10.0, 5.0], CARRIER_HZ
        )


def test_closest_approach_is_the_zenith(orbit):
    horizon = orbit.horizon_time_s()
    tca = time_of_closest_approach_s(
        orbit.observer_state, orbit.satellite_state, -horizon, horizon
    )
    assert tca == pytest.approx(0.0, abs=1e-5)


def test_doppler_zero_crossing_is_the_zenith(orbit):
    horizon = orbit.horizon_time_s()
    zero = doppler_zero_crossing_s(
        orbit.observer_state, orbit.satellite_state, -horizon, horizon
    )
    assert zero == pytest.approx(0.0, abs=1e-9)


def test_zero_crossing_and_closest_approach_agree(orbit):
    """The L1 check: two independent methods must land on the same instant."""
    horizon = orbit.horizon_time_s()
    tca = time_of_closest_approach_s(
        orbit.observer_state, orbit.satellite_state, -horizon, horizon, xatol_s=1e-6
    )
    zero = doppler_zero_crossing_s(
        orbit.observer_state, orbit.satellite_state, -horizon, horizon
    )
    assert abs(tca - zero) < 1e-5


def test_zero_crossing_requires_a_sign_change(orbit):
    """A window entirely after the zenith has no crossing and must say so."""
    with pytest.raises(ValueError, match="does not change sign"):
        doppler_zero_crossing_s(orbit.observer_state, orbit.satellite_state, 50.0, 300.0)


def test_event_locators_reject_a_reversed_window(orbit):
    with pytest.raises(ValueError, match="require t_hi_s > t_lo_s"):
        time_of_closest_approach_s(orbit.observer_state, orbit.satellite_state, 10.0, 10.0)
    with pytest.raises(ValueError, match="require t_hi_s > t_lo_s"):
        doppler_zero_crossing_s(orbit.observer_state, orbit.satellite_state, 10.0, 1.0)


def test_zero_crossing_is_independent_of_the_carrier(orbit):
    """The crossing is a geometric instant; the carrier cancels."""
    horizon = orbit.horizon_time_s()
    zero = doppler_zero_crossing_s(
        orbit.observer_state, orbit.satellite_state, -horizon, horizon
    )
    for carrier in (401e6, 2.2e9, 26.0e9):
        times = np.linspace(-horizon, horizon, 201)
        p = compute_profile(
            orbit.observer_state, orbit.satellite_state, times, carrier
        )
        crossing_index = int(np.argmin(np.abs(p.doppler_hz)))
        assert abs(times[crossing_index] - zero) <= times[1] - times[0]
