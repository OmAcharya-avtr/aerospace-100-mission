"""Sign-convention tests.

Every assertion in this file fails if a sign is flipped anywhere in the
package.  They are collected here, separately from the numeric tests, because
the sign discipline is this product's reason to exist: one-way and two-way
kept apart, with the convention stated at every interface.

The convention under test, verbatim:

    range-rate POSITIVE when RECEDING;
    Doppler    POSITIVE (up-shift) when APPROACHING;
    Delta_f = -f_carrier * rho_dot / c;
    pre-compensation offset = -Delta_f.
"""

from __future__ import annotations

import numpy as np
import pytest

from dopplerkit.analytic import CircularOverheadPass
from dopplerkit.constants import WGS84_A_M
from dopplerkit.doppler import (
    LinkDirection,
    doppler_observable,
    doppler_rate_hz_per_s,
    gravitational_shift_fraction,
    one_way_doppler_hz,
    precompensation_offset_hz,
    two_way_doppler_hz,
    two_way_doppler_two_leg_hz,
)
from dopplerkit.geometry import State, range_rate_mps

CARRIER_HZ = 2.2e9


def _approaching_pair() -> tuple[State, State]:
    """Target closing on the observer along +x at 1000 m/s."""
    obs = State(position_m=[0.0, 0.0, 0.0], velocity_mps=np.zeros(3), label="obs")
    tgt = State(position_m=[1.0e6, 0.0, 0.0], velocity_mps=[-1000.0, 0.0, 0.0], label="tgt")
    return obs, tgt


def _receding_pair() -> tuple[State, State]:
    """Target moving away from the observer along +x at 1000 m/s."""
    obs = State(position_m=[0.0, 0.0, 0.0], velocity_mps=np.zeros(3), label="obs")
    tgt = State(position_m=[1.0e6, 0.0, 0.0], velocity_mps=[1000.0, 0.0, 0.0], label="tgt")
    return obs, tgt


# --- range-rate sign -------------------------------------------------------

def test_approaching_gives_negative_range_rate():
    obs, tgt = _approaching_pair()
    assert range_rate_mps(obs, tgt) == pytest.approx(-1000.0, abs=1e-9)


def test_receding_gives_positive_range_rate():
    obs, tgt = _receding_pair()
    assert range_rate_mps(obs, tgt) == pytest.approx(+1000.0, abs=1e-9)


def test_range_rate_is_invariant_under_reversing_the_link():
    """Reversing the link does NOT flip the range-rate sign. Bit-exact."""
    for pair in (_approaching_pair(), _receding_pair()):
        obs, tgt = pair
        assert range_rate_mps(obs, tgt) == range_rate_mps(tgt, obs)


# --- one-way Doppler sign --------------------------------------------------

def test_approaching_gives_positive_doppler_up_shift():
    obs, tgt = _approaching_pair()
    shift = one_way_doppler_hz(range_rate_mps(obs, tgt), CARRIER_HZ)
    assert shift > 0.0
    assert shift == pytest.approx(CARRIER_HZ * 1000.0 / 299792458.0, rel=1e-12)


def test_receding_gives_negative_doppler_down_shift():
    obs, tgt = _receding_pair()
    assert one_way_doppler_hz(range_rate_mps(obs, tgt), CARRIER_HZ) < 0.0


def test_uplink_and_downlink_doppler_have_the_same_sign_at_equal_carrier():
    """The direction label must not change the sign. Bit-exact equality."""
    obs, tgt = _approaching_pair()
    rate = range_rate_mps(obs, tgt)
    up = doppler_observable(rate, CARRIER_HZ, direction=LinkDirection.UPLINK)
    down = doppler_observable(rate, CARRIER_HZ, direction=LinkDirection.DOWNLINK)
    assert up.doppler_hz == down.doppler_hz
    assert up.direction is LinkDirection.UPLINK
    assert down.direction is LinkDirection.DOWNLINK
    assert up.ways == down.ways == 1


def test_uplink_and_downlink_scale_only_by_the_carrier_ratio():
    obs, tgt = _approaching_pair()
    rate = range_rate_mps(obs, tgt)
    f_up, f_down = 2.0255e9, 2.2e9
    ratio = one_way_doppler_hz(rate, f_down) / one_way_doppler_hz(rate, f_up)
    assert ratio == pytest.approx(f_down / f_up, rel=1e-15)


# --- two-way Doppler sign --------------------------------------------------

def test_two_way_has_the_same_sign_as_one_way():
    for pair in (_approaching_pair(), _receding_pair()):
        obs, tgt = pair
        rate = range_rate_mps(obs, tgt)
        one = one_way_doppler_hz(rate, CARRIER_HZ)
        two = two_way_doppler_hz(rate, CARRIER_HZ)
        assert np.sign(one) == np.sign(two)


def test_two_way_is_exactly_twice_one_way_at_unit_turnaround():
    """Bit-exact, not approximate: the non-relativistic-limit identity."""
    for rate in (-7000.0, -1.0, 0.0, 12.5, 7000.0):
        assert two_way_doppler_hz(rate, CARRIER_HZ, 1.0) == 2.0 * one_way_doppler_hz(
            rate, CARRIER_HZ
        )


def test_two_leg_form_reduces_to_the_single_epoch_form():
    first, cross = two_way_doppler_two_leg_hz(-6000.0, -6000.0, CARRIER_HZ, 1.0)
    assert first == two_way_doppler_hz(-6000.0, CARRIER_HZ, 1.0)
    # Cross term sign: product of two negative range-rates is positive, and the
    # term carries a leading +, so the dropped term is positive when closing.
    assert cross > 0.0


def test_two_leg_cross_term_is_second_order_small():
    first, cross = two_way_doppler_two_leg_hz(-7000.0, -7000.0, CARRIER_HZ, 1.0)
    assert abs(cross / first) < 1e-4


# --- Doppler rate sign -----------------------------------------------------

def test_positive_range_acceleration_gives_negative_doppler_rate():
    assert doppler_rate_hz_per_s(+107.6, CARRIER_HZ) < 0.0
    assert doppler_rate_hz_per_s(-107.6, CARRIER_HZ) > 0.0


def test_two_way_doppler_rate_is_exactly_twice_one_way():
    assert doppler_rate_hz_per_s(107.6, CARRIER_HZ, ways=2) == 2.0 * doppler_rate_hz_per_s(
        107.6, CARRIER_HZ, ways=1
    )


# --- pre-compensation sign -------------------------------------------------

def test_precompensation_is_the_negative_of_the_doppler_shift():
    for rate in (-7000.0, -3.0, 0.0, 42.0, 7000.0):
        assert precompensation_offset_hz(rate, CARRIER_HZ) == pytest.approx(
            -one_way_doppler_hz(rate, CARRIER_HZ), rel=1e-15, abs=1e-12
        )


def test_precompensation_sends_low_when_approaching():
    """Approaching up-shifts the link, so the transmitter must send LOW."""
    obs, tgt = _approaching_pair()
    assert precompensation_offset_hz(range_rate_mps(obs, tgt), CARRIER_HZ) < 0.0


def test_precompensation_sends_high_when_receding():
    obs, tgt = _receding_pair()
    assert precompensation_offset_hz(range_rate_mps(obs, tgt), CARRIER_HZ) > 0.0


def test_exact_precompensation_cancels_the_doppler_to_machine_precision():
    """The exact form must land the far end on nominal, not merely close."""
    c = 299792458.0
    for rate in (-7000.0, -100.0, 250.0, 7000.0):
        offset = precompensation_offset_hz(rate, CARRIER_HZ, exact=True)
        received = (CARRIER_HZ + offset) * (1.0 - rate / c)
        assert received == pytest.approx(CARRIER_HZ, rel=1e-15)


def test_first_order_precompensation_leaves_a_second_order_residual():
    """Honesty check: the default first-order offset does NOT cancel exactly."""
    c = 299792458.0
    rate = -7000.0
    offset = precompensation_offset_hz(rate, CARRIER_HZ, exact=False)
    received = (CARRIER_HZ + offset) * (1.0 - rate / c)
    residual = received - CARRIER_HZ
    assert residual != 0.0
    # The predicted residual is -f*beta^2 = -1.19944 Hz. The tolerance is 1e-6
    # relative, not tighter, because `received - CARRIER_HZ` subtracts two
    # numbers near 2.2e9 whose float64 spacing is 4.8e-7 Hz: the check itself
    # is cancellation-limited to about one ulp, which is 5e-7 of the residual.
    assert abs(residual) == pytest.approx(CARRIER_HZ * (rate / c) ** 2, rel=1e-6)


# --- gravitational term sign ----------------------------------------------

def test_downlink_gravitational_shift_is_a_blueshift():
    """Photon falling into the well: received frequency is up-shifted."""
    assert gravitational_shift_fraction(WGS84_A_M + 500e3, WGS84_A_M) > 0.0


def test_uplink_gravitational_shift_is_a_redshift():
    assert gravitational_shift_fraction(WGS84_A_M, WGS84_A_M + 500e3) < 0.0


def test_gravitational_shift_is_antisymmetric():
    a, b = WGS84_A_M, WGS84_A_M + 800e3
    assert gravitational_shift_fraction(a, b) == pytest.approx(
        -gravitational_shift_fraction(b, a), rel=1e-15
    )


# --- pass-level sign ordering ---------------------------------------------

def test_overhead_pass_doppler_goes_positive_then_negative():
    orbit = CircularOverheadPass(orbit_radius_m=WGS84_A_M + 500e3)
    horizon = orbit.horizon_time_s()
    before = orbit.one_way_doppler_hz(-0.5 * horizon, CARRIER_HZ)
    at = orbit.one_way_doppler_hz(0.0, CARRIER_HZ)
    after = orbit.one_way_doppler_hz(+0.5 * horizon, CARRIER_HZ)
    assert before > 0.0
    assert at == pytest.approx(0.0, abs=1e-9)
    assert after < 0.0
    assert before == pytest.approx(-after, rel=1e-12)


def test_overhead_pass_doppler_rate_is_negative_throughout():
    orbit = CircularOverheadPass(orbit_radius_m=WGS84_A_M + 500e3)
    horizon = orbit.horizon_time_s()
    times = np.linspace(-horizon, horizon, 51)
    rates = np.array(
        [doppler_rate_hz_per_s(orbit.range_acceleration_mps2(float(t)), CARRIER_HZ)
         for t in times]
    )
    # Strictly negative in the interior. At the two horizon endpoints the
    # range acceleration is exactly zero for this geometry (see
    # test_analytic.py::test_range_acceleration_is_exactly_zero_at_the_horizon),
    # so the Doppler rate there is zero to round-off, not negative.
    assert np.all(rates[1:-1] < 0.0)
    assert rates[0] == pytest.approx(0.0, abs=1e-9)
    assert rates[-1] == pytest.approx(0.0, abs=1e-9)
    assert rates[len(rates) // 2] == pytest.approx(
        orbit.max_doppler_rate_hz_per_s(CARRIER_HZ), rel=1e-12
    )
