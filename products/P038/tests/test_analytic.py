"""Closed-form circular overhead pass: known answers and cross-checks."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from dopplerkit.analytic import CircularOverheadPass
from dopplerkit.constants import C_M_S, MU_EARTH_M3_S2, WGS84_A_M
from dopplerkit.geometry import (
    range_acceleration_mps2,
    range_rate_finite_difference_mps,
    range_rate_mps,
    slant_range_m,
)

CARRIER_HZ = 2.2e9
ALT_M = 500.0e3


@pytest.fixture
def orbit() -> CircularOverheadPass:
    return CircularOverheadPass(orbit_radius_m=WGS84_A_M + ALT_M)


def test_construction_validation():
    with pytest.raises(ValueError, match="orbit_radius_m must be > 0"):
        CircularOverheadPass(orbit_radius_m=-1.0)
    with pytest.raises(ValueError, match="observer_radius_m must be > 0"):
        CircularOverheadPass(orbit_radius_m=7.0e6, observer_radius_m=0.0)
    with pytest.raises(ValueError, match="must exceed observer_radius_m"):
        CircularOverheadPass(orbit_radius_m=6.0e6, observer_radius_m=6.378137e6)
    with pytest.raises(ValueError, match="mu_m3_s2 must be > 0"):
        CircularOverheadPass(orbit_radius_m=7.0e6, mu_m3_s2=-1.0)


def test_mean_motion_matches_hand_calculation(orbit):
    # n = sqrt(mu / r^3), mu = 3.986004418e14, r = 6878137 m.
    # r^3 = 3.2545...e20 -> n = 1.1070...e-3 rad/s. Period = 2 pi / n.
    expected = math.sqrt(MU_EARTH_M3_S2 / (WGS84_A_M + ALT_M) ** 3)
    assert orbit.mean_motion_rad_s == pytest.approx(expected, rel=1e-15)
    assert orbit.orbital_period_s == pytest.approx(2.0 * math.pi / expected, rel=1e-15)
    # Sanity against the textbook value: a 500 km circular LEO period is about
    # 94.6 min (Vallado, 4th ed., two-body circular orbit).
    assert 94.0 < orbit.orbital_period_s / 60.0 < 95.0


def test_orbital_speed_matches_vis_viva_circular(orbit):
    # v = sqrt(mu/r) for a circular orbit; also v = r*n. The two must agree.
    assert orbit.orbital_speed_mps == pytest.approx(
        math.sqrt(MU_EARTH_M3_S2 / orbit.orbit_radius_m), rel=1e-14
    )
    assert 7.5e3 < orbit.orbital_speed_mps < 7.7e3


def test_range_at_zenith_is_exactly_the_altitude(orbit):
    assert orbit.range_m(0.0) == pytest.approx(ALT_M, abs=1e-6)


def test_range_rate_at_zenith_is_exactly_zero(orbit):
    assert orbit.range_rate_mps(0.0) == 0.0


def test_range_acceleration_at_zenith_matches_the_closed_form(orbit):
    # rho_ddot(0) = r_s r_o n^2 / (r_s - r_o), the maximum over the pass.
    n = orbit.mean_motion_rad_s
    expected = (
        orbit.orbit_radius_m * orbit.observer_radius_m * n**2
        / (orbit.orbit_radius_m - orbit.observer_radius_m)
    )
    assert orbit.range_acceleration_mps2(0.0) == pytest.approx(expected, rel=1e-12)
    assert orbit.max_doppler_rate_hz_per_s(CARRIER_HZ) == pytest.approx(
        -CARRIER_HZ * expected / C_M_S, rel=1e-12
    )


def test_range_rate_at_the_horizon_is_exactly_observer_radius_times_mean_motion(orbit):
    """Known answer derived in validation/VALIDATION.md.

    At the geometric horizon cos(nt) = r_o/r_s, so sin(nt) = rho/r_s and
    rho_dot = r_s r_o n sin(nt)/rho collapses to exactly r_o * n.
    """
    t_h = orbit.horizon_time_s()
    expected = orbit.observer_radius_m * orbit.mean_motion_rad_s
    assert orbit.range_rate_mps(t_h) == pytest.approx(expected, rel=1e-12)
    assert orbit.range_rate_mps(-t_h) == pytest.approx(-expected, rel=1e-12)


def test_range_acceleration_is_exactly_zero_at_the_horizon(orbit):
    """Corollary of the previous known answer: rho_ddot = (r_o^2 n^2 - rho_dot^2)/rho."""
    t_h = orbit.horizon_time_s()
    scale = orbit.range_acceleration_mps2(0.0)
    assert abs(orbit.range_acceleration_mps2(t_h)) < 1e-12 * scale
    assert abs(orbit.range_acceleration_mps2(-t_h)) < 1e-12 * scale


def test_horizon_range_matches_pythagoras(orbit):
    # At zero elevation the line of sight is tangent to the observer sphere:
    # rho = sqrt(r_s^2 - r_o^2).
    t_h = orbit.horizon_time_s()
    expected = math.sqrt(orbit.orbit_radius_m**2 - orbit.observer_radius_m**2)
    assert orbit.range_m(t_h) == pytest.approx(expected, rel=1e-12)


def test_peak_one_way_doppler_matches_hand_calculation(orbit):
    # Delta_f at the horizon = -f_c * r_o * n / c.
    t_h = orbit.horizon_time_s()
    expected = -CARRIER_HZ * orbit.observer_radius_m * orbit.mean_motion_rad_s / C_M_S
    assert orbit.one_way_doppler_hz(t_h, CARRIER_HZ) == pytest.approx(expected, rel=1e-12)


def test_range_is_symmetric_about_the_zenith(orbit):
    for t in (10.0, 123.4, 300.0):
        assert orbit.range_m(t) == pytest.approx(orbit.range_m(-t), rel=1e-15)


def test_range_rate_is_antisymmetric_about_the_zenith(orbit):
    for t in (10.0, 123.4, 300.0):
        assert orbit.range_rate_mps(t) == pytest.approx(-orbit.range_rate_mps(-t), rel=1e-12)


def test_array_and_scalar_calls_agree(orbit):
    times = np.array([-200.0, -10.0, 0.0, 55.5, 300.0])
    assert np.allclose(orbit.range_m(times), [orbit.range_m(float(t)) for t in times])
    assert np.allclose(
        orbit.range_rate_mps(times), [orbit.range_rate_mps(float(t)) for t in times]
    )
    assert np.allclose(
        orbit.range_acceleration_mps2(times),
        [orbit.range_acceleration_mps2(float(t)) for t in times],
    )
    assert np.allclose(
        orbit.one_way_doppler_hz(times, CARRIER_HZ),
        [orbit.one_way_doppler_hz(float(t), CARRIER_HZ) for t in times],
    )


def test_vector_geometry_reproduces_the_closed_forms(orbit):
    """The generic dot-product code and the closed forms must agree."""
    for t in (-300.0, -50.0, 0.0, 123.4, 320.0):
        obs = orbit.observer_state(t)
        sat = orbit.satellite_state(t)
        assert slant_range_m(obs, sat) == pytest.approx(orbit.range_m(t), rel=1e-12)
        assert range_rate_mps(obs, sat) == pytest.approx(
            orbit.range_rate_mps(t), rel=1e-9, abs=1e-9
        )
        assert range_acceleration_mps2(obs, sat) == pytest.approx(
            orbit.range_acceleration_mps2(t), rel=1e-11
        )


def test_satellite_state_is_a_consistent_circular_orbit(orbit):
    for t in (-100.0, 0.0, 250.0):
        sat = orbit.satellite_state(t)
        assert np.linalg.norm(sat.position_m) == pytest.approx(orbit.orbit_radius_m, rel=1e-12)
        assert sat.speed_mps == pytest.approx(orbit.orbital_speed_mps, rel=1e-12)
        # Velocity perpendicular to position for a circular orbit.
        assert abs(sat.position_m @ sat.velocity_mps) < 1e-6 * (
            orbit.orbit_radius_m * orbit.orbital_speed_mps
        )
        # Two-body acceleration points inward with magnitude mu/r^2.
        assert np.linalg.norm(sat.acceleration_mps2) == pytest.approx(
            MU_EARTH_M3_S2 / orbit.orbit_radius_m**2, rel=1e-12
        )


def test_observer_state_is_inertially_fixed(orbit):
    for t in (-100.0, 0.0, 250.0):
        obs = orbit.observer_state(t)
        assert np.array_equal(obs.velocity_mps, np.zeros(3))
        assert np.array_equal(obs.acceleration_mps2, np.zeros(3))
        assert obs.position_m[0] == pytest.approx(orbit.observer_radius_m, rel=0.0)


def test_finite_difference_matches_the_analytic_range_rate(orbit):
    """The L1 check, as a unit test: central difference vs dot product.

    Step 0.0390625 s is inside the measured second-order regime, where the
    worst error over these four epochs is 1.6e-5 m/s and the round-off floor
    is about 3e-6 m/s (see validation/validate_range_rate.py for the full
    convergence table). The tolerance is 1e-4 m/s, a factor 6 of headroom.
    """
    for t in (-300.0, -120.0, 40.0, 250.0):
        fd = range_rate_finite_difference_mps(orbit.range_m, t, 0.0390625)
        assert fd == pytest.approx(orbit.range_rate_mps(t), abs=1e-4)


def test_finite_difference_converges_at_second_order(orbit):
    """Halving the step must quarter the error, to within 1 %."""
    t = -120.0
    exact = orbit.range_rate_mps(t)
    errs = [
        abs(range_rate_finite_difference_mps(orbit.range_m, t, h) - exact)
        for h in (5.0, 2.5, 1.25, 0.625)
    ]
    for a, b in itertools.pairwise(errs):
        assert math.log2(a / b) == pytest.approx(2.0, abs=0.01)
