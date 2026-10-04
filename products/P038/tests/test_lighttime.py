"""Light-time iteration: convergence, residual, legs, and ordering."""

from __future__ import annotations

import numpy as np
import pytest

from dopplerkit.analytic import CircularOverheadPass
from dopplerkit.constants import C_M_S, WGS84_A_M
from dopplerkit.lighttime import (
    down_leg_light_time,
    two_way_light_time,
    up_leg_light_time,
)


@pytest.fixture
def orbit() -> CircularOverheadPass:
    return CircularOverheadPass(orbit_radius_m=WGS84_A_M + 500.0e3)


@pytest.fixture
def sat_fn(orbit):
    return lambda t: orbit.satellite_state(t).position_m


@pytest.fixture
def sta_fn(orbit):
    return lambda t: orbit.observer_state(t).position_m


def test_static_geometry_gives_the_exact_light_time():
    """A stationary pair: tau = rho/c in one iteration, zero residual."""
    rho = 1.0e6
    sat = lambda t: np.array([rho, 0.0, 0.0])
    sta = lambda t: np.zeros(3)
    sol = down_leg_light_time(sat, sta, 0.0)
    assert sol.light_time_s == pytest.approx(rho / C_M_S, rel=1e-15)
    assert sol.residual_m == 0.0
    assert sol.converged
    assert sol.iterations == 1


def test_down_leg_converges_with_small_residual(orbit, sat_fn, sta_fn):
    for t in (-300.0, 0.0, 150.0, 330.0):
        sol = down_leg_light_time(sat_fn, sta_fn, t, tol_m=1e-6)
        assert sol.converged
        assert sol.residual_m <= 1e-6
        assert sol.iterations <= 4
        assert sol.light_time_s > 0.0
        assert sol.leg == "down"
        assert sol.emission_time_s < sol.reception_time_s


def test_up_leg_converges_with_small_residual(orbit, sat_fn, sta_fn):
    for t in (-300.0, 0.0, 150.0, 330.0):
        sol = up_leg_light_time(sat_fn, sta_fn, t, tol_m=1e-6)
        assert sol.converged
        assert sol.residual_m <= 1e-6
        assert sol.iterations <= 4
        assert sol.leg == "up"
        assert sol.emission_time_s == pytest.approx(t, rel=0.0, abs=0.0)
        assert sol.reception_time_s > sol.emission_time_s


def test_range_equals_c_times_light_time(orbit, sat_fn, sta_fn):
    sol = down_leg_light_time(sat_fn, sta_fn, 200.0)
    assert sol.range_m == pytest.approx(C_M_S * sol.light_time_s, rel=1e-15)


def test_light_time_at_zenith_matches_the_instantaneous_range(orbit, sat_fn, sta_fn):
    """At closest approach the range-rate is zero, so tau = rho(0)/c exactly."""
    sol = down_leg_light_time(sat_fn, sta_fn, 0.0)
    assert sol.light_time_s == pytest.approx(orbit.range_m(0.0) / C_M_S, rel=1e-12)


def test_up_and_down_legs_differ_by_about_the_range_rate_ratio(orbit, sat_fn, sta_fn):
    """|tau_up - tau_down| / tau ~ |rho_dot| / c. Checks the legs are not aliased."""
    t = 200.0
    up = up_leg_light_time(sat_fn, sta_fn, t)
    down = down_leg_light_time(sat_fn, sta_fn, t)
    # tau_down = rho/(c + rho_dot) and tau_up = rho/(c - rho_dot) to first
    # order, so the fractional difference is 2*rho_dot/(c - rho_dot) ~ 2*beta.
    rel_diff = abs(up.light_time_s - down.light_time_s) / down.light_time_s
    beta = abs(orbit.range_rate_mps(t)) / C_M_S
    assert rel_diff == pytest.approx(2.0 * beta, rel=0.05)


def test_light_time_is_not_the_instantaneous_range_away_from_zenith(orbit, sat_fn, sta_fn):
    """The correction is real: at 200 s it is tens of metres, not zero."""
    t = 200.0
    sol = down_leg_light_time(sat_fn, sta_fn, t)
    instantaneous = orbit.range_m(t)
    assert abs(sol.range_m - instantaneous) > 10.0


def test_two_way_round_trip_is_self_consistent(orbit, sat_fn, sta_fn):
    tw = two_way_light_time(sat_fn, sta_fn, 150.0)
    assert tw.round_trip_s == pytest.approx(
        tw.up.light_time_s + tw.down.light_time_s, rel=1e-15
    )
    assert tw.receive_time_s == pytest.approx(
        tw.transmit_time_s + tw.round_trip_s, rel=1e-15
    )
    assert tw.up.converged and tw.down.converged
    assert tw.up.leg == "up" and tw.down.leg == "down"


def test_two_way_round_trip_is_about_twice_a_single_leg(orbit, sat_fn, sta_fn):
    tw = two_way_light_time(sat_fn, sta_fn, 150.0)
    one = down_leg_light_time(sat_fn, sta_fn, 150.0)
    assert tw.round_trip_s == pytest.approx(2.0 * one.light_time_s, rel=1e-3)


def test_tolerance_and_max_iter_validation(sat_fn, sta_fn):
    with pytest.raises(ValueError, match="tol_m must be > 0"):
        down_leg_light_time(sat_fn, sta_fn, 0.0, tol_m=0.0)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        up_leg_light_time(sat_fn, sta_fn, 0.0, max_iter=0)
    with pytest.raises(ValueError, match="tol_m must be > 0"):
        two_way_light_time(sat_fn, sta_fn, 0.0, tol_m=-1.0)


def test_bad_position_shape_raises(sat_fn, sta_fn):
    with pytest.raises(ValueError, match=r"satellite position callable .* shape \(3,\)"):
        down_leg_light_time(lambda t: np.zeros(2), sta_fn, 0.0)
    with pytest.raises(ValueError, match=r"station position callable .* shape \(3,\)"):
        down_leg_light_time(sat_fn, lambda t: np.zeros(4), 0.0)


def test_non_convergence_is_reported_not_raised(orbit, sat_fn, sta_fn):
    """One iteration is not enough away from the zenith; report, do not raise."""
    sol = down_leg_light_time(sat_fn, sta_fn, 200.0, tol_m=1e-6, max_iter=1)
    assert not sol.converged
    assert sol.iterations == 1
    # The first iterate ignores satellite motion over the light time, so the
    # residual is of order |rho_dot| * tau = 4.9 km/s * 5.2 ms ~ 25 m.
    assert sol.residual_m > 1.0


def test_as_dict_round_trips(orbit, sat_fn, sta_fn):
    d = down_leg_light_time(sat_fn, sta_fn, 100.0).as_dict()
    assert set(d) == {
        "light_time_s", "range_m", "iterations", "residual_m", "converged",
        "emission_time_s", "reception_time_s", "leg",
    }
    assert d["leg"] == "down"
