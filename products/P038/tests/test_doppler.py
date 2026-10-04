"""Doppler quantities: known answers, validation, relativistic magnitudes."""

from __future__ import annotations

import pytest

from dopplerkit.constants import C_M_S, WGS84_A_M
from dopplerkit.doppler import (
    DOPPLER_CONVENTION,
    DopplerObservable,
    LinkDirection,
    doppler_observable,
    doppler_rate_hz_per_s,
    gravitational_shift_fraction,
    one_way_doppler_hz,
    precompensation_offset_hz,
    relativistic_correction_hz,
    relativistic_fraction_second_order,
    two_way_doppler_hz,
    two_way_doppler_two_leg_hz,
)

CARRIER_HZ = 2.2e9


def test_hand_calculated_one_way_doppler():
    # Hand calculation: f_c = 2.2e9 Hz, rho_dot = -7000 m/s (approaching).
    # Delta_f = -2.2e9 * (-7000) / 299792458 = +51368.870660... Hz
    assert one_way_doppler_hz(-7000.0, CARRIER_HZ) == pytest.approx(
        51368.87066051542, rel=1e-13
    )


def test_hand_calculated_two_way_doppler():
    # Twice the above: +102737.74132103... Hz
    assert two_way_doppler_hz(-7000.0, CARRIER_HZ) == pytest.approx(
        102737.74132103084, rel=1e-13
    )


def test_zero_range_rate_gives_exactly_zero_doppler():
    assert one_way_doppler_hz(0.0, CARRIER_HZ) == 0.0
    assert two_way_doppler_hz(0.0, CARRIER_HZ) == 0.0
    assert doppler_rate_hz_per_s(0.0, CARRIER_HZ) == 0.0
    assert precompensation_offset_hz(0.0, CARRIER_HZ) == 0.0


def test_doppler_is_exactly_linear_in_the_carrier():
    rate = -4321.0
    base = one_way_doppler_hz(rate, 1.0e9)
    assert one_way_doppler_hz(rate, 3.0e9) == pytest.approx(3.0 * base, rel=1e-15)
    assert one_way_doppler_hz(rate, 0.5e9) == pytest.approx(0.5 * base, rel=1e-15)


def test_turnaround_ratio_scales_two_way_linearly():
    rate = -5000.0
    base = two_way_doppler_hz(rate, CARRIER_HZ, 1.0)
    assert two_way_doppler_hz(rate, CARRIER_HZ, 240.0 / 221.0) == pytest.approx(
        base * 240.0 / 221.0, rel=1e-15
    )


def test_carrier_validation():
    with pytest.raises(ValueError, match="carrier_hz must be > 0"):
        one_way_doppler_hz(0.0, 0.0)
    with pytest.raises(ValueError, match="carrier_hz must be > 0"):
        one_way_doppler_hz(0.0, -1.0)
    with pytest.raises(TypeError, match="carrier_hz must be a real number"):
        one_way_doppler_hz(0.0, "2.2e9")
    with pytest.raises(TypeError, match="carrier_hz must be a real number"):
        one_way_doppler_hz(0.0, True)


def test_superluminal_range_rate_is_rejected():
    with pytest.raises(ValueError, match=r"\|rho_dot\| < c"):
        one_way_doppler_hz(C_M_S, CARRIER_HZ)
    with pytest.raises(ValueError, match=r"\|rho_dot\| < c"):
        two_way_doppler_hz(-2.0 * C_M_S, CARRIER_HZ)
    with pytest.raises(ValueError, match="range_rate_up_mps"):
        two_way_doppler_two_leg_hz(C_M_S, 0.0, CARRIER_HZ)
    with pytest.raises(ValueError, match="range_rate_down_mps"):
        two_way_doppler_two_leg_hz(0.0, C_M_S, CARRIER_HZ)


def test_turnaround_ratio_validation():
    with pytest.raises(ValueError, match="turnaround_ratio must be > 0"):
        two_way_doppler_hz(0.0, CARRIER_HZ, 0.0)
    with pytest.raises(ValueError, match="turnaround_ratio must be > 0"):
        two_way_doppler_two_leg_hz(0.0, 0.0, CARRIER_HZ, -1.0)


def test_ways_validation():
    with pytest.raises(ValueError, match="ways must be 1 or 2"):
        doppler_rate_hz_per_s(1.0, CARRIER_HZ, ways=3)
    with pytest.raises(ValueError, match="ways must be 1 or 2"):
        doppler_rate_hz_per_s(1.0, CARRIER_HZ, ways=0)


def test_two_leg_sum_reproduces_the_exact_classical_composition():
    """first_order + cross == the exact product of the two one-way legs."""
    up, down = -6500.0, -6400.0
    first, cross = two_way_doppler_two_leg_hz(up, down, CARRIER_HZ, 1.0)
    exact = CARRIER_HZ * ((1.0 - up / C_M_S) * (1.0 - down / C_M_S) - 1.0)
    assert first + cross == pytest.approx(exact, rel=1e-9)


def test_doppler_observable_carries_the_convention():
    obs = doppler_observable(-7000.0, CARRIER_HZ, direction=LinkDirection.DOWNLINK)
    assert isinstance(obs, DopplerObservable)
    assert obs.convention == DOPPLER_CONVENTION
    assert "approaching" in obs.convention
    assert obs.order_kept.startswith("O(beta^1)")
    assert obs.as_dict()["direction"] == "downlink"
    assert obs.as_dict()["ways"] == 1


def test_doppler_observable_two_way_reference_carrier_includes_turnaround():
    obs = doppler_observable(
        -7000.0, 2.0e9, direction=LinkDirection.TWO_WAY, turnaround_ratio=1.5
    )
    assert obs.ways == 2
    assert obs.carrier_hz == pytest.approx(3.0e9, rel=1e-15)
    assert obs.doppler_hz == pytest.approx(two_way_doppler_hz(-7000.0, 2.0e9, 1.5), rel=1e-15)


def test_relativistic_term_at_zero_range_rate_is_pure_time_dilation():
    # rho_dot = 0 leaves -v^2/(2c^2). For v = 7612.560... m/s this is
    # -(7612.56/299792458)^2 / 2. Hand value for v = 7600 m/s:
    #   beta = 2.53509...e-5, beta^2/2 = 3.21334...e-10
    v = 7600.0
    expected = -(v / C_M_S) ** 2 / 2.0
    assert relativistic_fraction_second_order(0.0, v) == pytest.approx(expected, rel=1e-14)
    assert relativistic_fraction_second_order(0.0, v) < 0.0


def test_relativistic_term_is_tiny_compared_with_the_classical_shift():
    v, rate = 7612.6, -7059.2
    classical = one_way_doppler_hz(rate, CARRIER_HZ)
    rel = relativistic_correction_hz(rate, v, CARRIER_HZ)
    assert abs(rel / classical) < 1e-4
    assert abs(rel) < 1.0  # sub-Hz at S-band


def test_relativistic_term_scales_with_carrier():
    v, rate = 7612.6, -3000.0
    a = relativistic_correction_hz(rate, v, 1.0e9)
    b = relativistic_correction_hz(rate, v, 2.0e9)
    assert b == pytest.approx(2.0 * a, rel=1e-15)


def test_relativistic_input_validation():
    with pytest.raises(ValueError, match="speed_mps must be >= 0"):
        relativistic_fraction_second_order(0.0, -1.0)
    with pytest.raises(ValueError, match="speed_mps must be < c"):
        relativistic_fraction_second_order(0.0, C_M_S)


def test_gravitational_shift_validation():
    with pytest.raises(ValueError, match="radii must be > 0"):
        gravitational_shift_fraction(0.0, WGS84_A_M)
    with pytest.raises(ValueError, match="mu_m3_s2 must be > 0"):
        gravitational_shift_fraction(WGS84_A_M, WGS84_A_M + 1.0, mu_m3_s2=0.0)


def test_gravitational_shift_vanishes_at_equal_radii():
    assert gravitational_shift_fraction(WGS84_A_M, WGS84_A_M) == 0.0


def test_exact_precompensation_differs_from_first_order_at_second_order():
    rate = -7000.0
    first = precompensation_offset_hz(rate, CARRIER_HZ, exact=False)
    exact = precompensation_offset_hz(rate, CARRIER_HZ, exact=True)
    beta = rate / C_M_S
    # beta/(1-beta) = beta + beta^2 + beta^3 + ..., so the difference is
    # f*beta^2 to within f*beta^3, i.e. a relative error of |beta| = 2.3e-5.
    # The tolerance is that term, not a fudge: at rel=1e-5 this assertion
    # fails by exactly f*beta^3 = -2.8e-5 Hz.
    assert exact - first == pytest.approx(CARRIER_HZ * beta**2, rel=1e-4)
    # The next term down is f*beta^3 = -2.80e-5 Hz. It is only resolvable to
    # about 2 %: the exact offset is computed as f*(1/(1-beta) - 1), whose
    # subtraction near 1.0 leaves an absolute error of f*eps = 4.9e-7 Hz, which
    # is 1.8 % of f*beta^3. Hence rel=0.05, a round-off floor, not a fudge.
    assert (exact - first) - CARRIER_HZ * beta**2 == pytest.approx(
        CARRIER_HZ * beta**3, rel=0.05
    )
