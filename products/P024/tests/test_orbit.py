"""Circular Keplerian orbit."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from detumblesim.constants import MU_EARTH, R_EARTH_M
from detumblesim.orbit import CircularOrbit


class TestConstruction:
    @pytest.mark.parametrize("alt", [0.0, -100.0, float("nan")])
    def test_rejects_bad_altitude(self, alt):
        with pytest.raises(ValueError, match="altitude_km"):
            CircularOrbit(altitude_km=alt)

    @pytest.mark.parametrize("inc", [-1.0, 181.0])
    def test_rejects_bad_inclination(self, inc):
        with pytest.raises(ValueError, match="inclination_deg"):
            CircularOrbit(inclination_deg=inc)


class TestKnownAnswers:
    def test_period_at_500_km(self):
        # a = 6378137 + 500000 = 6878137 m; T = 2 pi sqrt(a^3 / mu) with
        # mu = 3.986004418e14 m^3/s^2 gives 5676.978 s (hand check below).
        o = CircularOrbit(altitude_km=500.0)
        a = R_EARTH_M + 500e3
        expected = 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH)
        assert np.isclose(o.period_s, expected, rtol=1e-14)
        assert np.isclose(o.period_s, 5676.978, atol=1e-2)

    def test_mean_motion_and_period_are_consistent(self):
        o = CircularOrbit(altitude_km=700.0, inclination_deg=98.2)
        assert np.isclose(o.mean_motion_rad_s * o.period_s, 2.0 * np.pi)

    def test_equatorial_orbit_stays_in_the_xy_plane(self):
        o = CircularOrbit(inclination_deg=0.0)
        t = np.linspace(0.0, o.period_s, 37)
        assert np.allclose(o.position_eci(t)[:, 2], 0.0, atol=1e-6)

    def test_polar_orbit_reaches_full_latitude(self):
        o = CircularOrbit(inclination_deg=90.0)
        t = np.linspace(0.0, o.period_s, 721)
        z = o.position_eci(t)[:, 2]
        assert np.isclose(z.max() / o.radius_m, 1.0, atol=1e-4)
        assert np.isclose(z.min() / o.radius_m, -1.0, atol=1e-4)


class TestGeometry:
    def test_radius_is_constant(self):
        o = CircularOrbit(altitude_km=650.0, inclination_deg=52.0, raan_deg=31.0)
        t = np.linspace(0.0, 3.0 * o.period_s, 200)
        r = np.linalg.norm(o.position_eci(t), axis=1)
        assert np.allclose(r, o.radius_m, rtol=1e-12)

    def test_velocity_matches_numerical_derivative(self):
        o = CircularOrbit(altitude_km=550.0, inclination_deg=63.4, raan_deg=17.0)
        h = 1e-3
        num = (o.position_eci(1234.0 + h) - o.position_eci(1234.0 - h)) / (2 * h)
        assert np.allclose(num, o.velocity_eci(1234.0), rtol=1e-6, atol=1e-6)

    def test_normal_is_perpendicular_to_position_and_velocity(self):
        o = CircularOrbit(altitude_km=500.0, inclination_deg=97.4, raan_deg=210.0)
        n = o.orbit_normal_eci()
        assert np.isclose(float(np.linalg.norm(n)), 1.0)
        for t in (0.0, 900.0, 4321.0):
            assert abs(float(n @ o.position_eci(t))) < 1e-3
            assert abs(float(n @ o.velocity_eci(t))) < 1e-6

    def test_normal_matches_specific_angular_momentum_direction(self):
        o = CircularOrbit(altitude_km=500.0, inclination_deg=97.4, raan_deg=45.0)
        h = np.cross(o.position_eci(777.0), o.velocity_eci(777.0))
        assert np.allclose(h / np.linalg.norm(h), o.orbit_normal_eci(), atol=1e-9)

    def test_scalar_and_array_time_agree(self):
        o = CircularOrbit()
        assert np.allclose(o.position_eci(100.0), o.position_eci(np.array([100.0]))[0])

    @given(
        alt=st.floats(200.0, 2000.0),
        inc=st.floats(0.0, 180.0),
        raan=st.floats(0.0, 360.0),
    )
    @settings(max_examples=40, deadline=None)
    def test_speed_matches_circular_orbit_speed(self, alt, inc, raan):
        o = CircularOrbit(altitude_km=alt, inclination_deg=inc, raan_deg=raan)
        v = float(np.linalg.norm(o.velocity_eci(500.0)))
        assert np.isclose(v, np.sqrt(MU_EARTH / o.radius_m), rtol=1e-12)
