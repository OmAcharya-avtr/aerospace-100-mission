"""Unit tests for the level-flight energy model."""

from __future__ import annotations

import numpy as np
import pytest

from conformalband.physics import (
    DEFAULT_AIRFRAME,
    GRAVITY,
    SECONDS_PER_HOUR,
    Airframe,
    leg_energy,
    level_flight_power,
    propulsive_efficiency,
)


def test_gravity_constant():
    assert GRAVITY == pytest.approx(9.80665, abs=0.0)


def test_seconds_per_hour():
    assert SECONDS_PER_HOUR == 3600.0


def test_default_aspect_ratio():
    # b^2 / S = 1.20^2 / 0.30 = 1.44 / 0.30 = 4.8 exactly
    assert DEFAULT_AIRFRAME.aspect_ratio == pytest.approx(4.8, rel=1e-15)


def test_efficiency_at_design_airspeed_is_eta_prop():
    value = propulsive_efficiency(DEFAULT_AIRFRAME.design_airspeed)
    assert float(value) == pytest.approx(DEFAULT_AIRFRAME.eta_prop, rel=1e-15)


@pytest.mark.parametrize("airspeed", [16.0, 18.0, 22.0, 24.0])
def test_efficiency_below_eta_prop_off_design(airspeed):
    assert float(propulsive_efficiency(airspeed)) < DEFAULT_AIRFRAME.eta_prop


def test_efficiency_is_symmetric_about_design():
    low = float(propulsive_efficiency(DEFAULT_AIRFRAME.design_airspeed - 3.0))
    high = float(propulsive_efficiency(DEFAULT_AIRFRAME.design_airspeed + 3.0))
    assert low == pytest.approx(high, rel=1e-14)


def test_efficiency_curvature_zero_is_constant():
    frame = Airframe(eta_curvature=0.0)
    values = propulsive_efficiency(np.array([14.0, 20.0, 26.0]), frame)
    assert np.allclose(values, frame.eta_prop, rtol=0.0, atol=0.0)


def test_power_is_monotone_in_mass():
    masses = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
    power = level_flight_power(20.0, masses, 1.18)
    assert np.all(np.diff(power) > 0.0)


def test_power_is_monotone_in_airspeed_above_design():
    speeds = np.array([21.0, 22.0, 23.0])
    power = level_flight_power(speeds, 6.0, 1.18)
    assert np.all(np.diff(power) > 0.0)


def test_induced_power_dominates_at_low_airspeed():
    low = level_flight_power(12.0, 6.0, 1.18, constant_efficiency=True)
    high = level_flight_power(20.0, 6.0, 1.18, constant_efficiency=True)
    assert low > high


def test_power_curve_has_an_interior_minimum():
    # Parasite power grows as V^3 and induced power falls as 1/V, so the sum has
    # a minimum at V = (2(mg)^2 / (1.5 rho^2 S C_D0 pi b^2 e))^(1/4). With the
    # shipped airframe that is inside [14, 19] m/s, which is why the fitted
    # baseline cannot be checked by monotonicity alone.
    speeds = np.linspace(10.0, 26.0, 161)
    power = level_flight_power(speeds, 6.0, 1.18, constant_efficiency=True)
    argmin = float(speeds[int(np.argmin(power))])
    assert 14.0 < argmin < 19.0


def test_constant_efficiency_power_is_lower_than_curved_off_design():
    constant = level_flight_power(16.0, 6.0, 1.18, constant_efficiency=True)
    curved = level_flight_power(16.0, 6.0, 1.18, constant_efficiency=False)
    assert curved > constant


def test_energy_scales_linearly_with_distance():
    one = leg_energy(20.0, 6.0, 1.18, 0.0, 1000.0)
    two = leg_energy(20.0, 6.0, 1.18, 0.0, 2000.0)
    assert float(two) == pytest.approx(2.0 * float(one), rel=1e-14)


def test_energy_increases_with_headwind():
    still = leg_energy(20.0, 6.0, 1.18, 0.0, 1000.0)
    windy = leg_energy(20.0, 6.0, 1.18, 5.0, 1000.0)
    assert float(windy) > float(still)


def test_energy_with_tailwind_is_lower():
    still = leg_energy(20.0, 6.0, 1.18, 0.0, 1000.0)
    tail = leg_energy(20.0, 6.0, 1.18, -5.0, 1000.0)
    assert float(tail) < float(still)


def test_energy_broadcasts():
    out = leg_energy(np.full(4, 20.0), np.linspace(5.0, 7.0, 4), 1.18, 0.0, 1000.0)
    assert out.shape == (4,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wing_area": 0.0},
        {"wingspan": -1.0},
        {"cd0": 0.0},
        {"oswald": 1.5},
        {"oswald": 0.0},
        {"eta_prop": 0.0},
        {"eta_prop": 1.2},
        {"avionics_power": -1.0},
        {"design_airspeed": 0.0},
        {"eta_curvature": -0.1},
    ],
)
def test_airframe_rejects_bad_coefficients(kwargs):
    with pytest.raises(ValueError):
        Airframe(**kwargs)


def test_efficiency_rejects_non_positive_airspeed():
    with pytest.raises(ValueError, match="airspeed"):
        propulsive_efficiency(np.array([10.0, 0.0]))


def test_efficiency_rejects_airspeed_outside_validity():
    # eta(V) = 0 when ((V - 20)/20)^2 = 1/2.5 = 0.4, i.e. V = 20(1 +/- 0.6325) = 32.65 m/s
    with pytest.raises(ValueError, match="validity range"):
        propulsive_efficiency(60.0)


def test_power_rejects_negative_mass():
    with pytest.raises(ValueError, match="mass"):
        level_flight_power(20.0, -1.0, 1.18)


def test_power_rejects_zero_density():
    with pytest.raises(ValueError, match="air_density"):
        level_flight_power(20.0, 6.0, 0.0)


def test_energy_rejects_zero_distance():
    with pytest.raises(ValueError, match="distance"):
        leg_energy(20.0, 6.0, 1.18, 0.0, 0.0)


def test_energy_rejects_headwind_at_airspeed():
    with pytest.raises(ValueError, match="ground speed"):
        leg_energy(20.0, 6.0, 1.18, 20.0, 1000.0)


def test_energy_rejects_headwind_above_airspeed():
    with pytest.raises(ValueError, match="ground speed"):
        leg_energy(20.0, 6.0, 1.18, 25.0, 1000.0)
