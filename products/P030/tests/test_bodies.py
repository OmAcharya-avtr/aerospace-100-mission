"""Unit and known-answer tests for keepout.bodies."""

import datetime as dt

import numpy as np
import pytest

from keepout import bodies as b


class TestAngularRadius:
    def test_half_distance_is_ninety_degrees(self):
        # Observer on the surface: d = R, sin(alpha) = 1, alpha = 90 deg.
        assert b.angular_radius(1.0, 1.0) == pytest.approx(np.pi / 2)

    def test_double_radius_is_thirty_degrees(self):
        # d = 2R -> sin(alpha) = 1/2 -> alpha = 30 deg exactly.
        assert b.angular_radius(1.0, 2.0) == pytest.approx(np.pi / 6)

    def test_inside_body_rejected(self):
        with pytest.raises(ValueError, match="inside the body"):
            b.angular_radius(2.0, 1.0)

    def test_nonpositive_radius_rejected(self):
        with pytest.raises(ValueError, match="must be > 0"):
            b.angular_radius(0.0, 1.0)

    def test_array_input(self):
        out = b.angular_radius(1.0, np.array([1.0, 2.0]))
        assert np.allclose(out, [np.pi / 2, np.pi / 6])


class TestEarthAngularRadius:
    def test_surface(self):
        assert b.earth_angular_radius(0.0) == pytest.approx(np.pi / 2)

    def test_matches_analytic_expression(self):
        # The whole point of the function: arcsin(R_E / (R_E + h)).
        h = np.array([0.0, 200e3, 400e3, 800e3, 35786e3])
        expected = np.arcsin(b.EARTH_RADIUS_M / (b.EARTH_RADIUS_M + h))
        assert np.allclose(b.earth_angular_radius(h), expected, atol=0.0, rtol=0.0)

    def test_geo_is_small(self):
        # Geostationary radius 42164 km -> arcsin(6378.137/42164.0) computed
        # from the WGS-84 radius plus the standard GEO altitude 35786 km.
        alpha = b.earth_angular_radius(35786e3)
        assert np.degrees(alpha) == pytest.approx(8.6997, abs=1e-3)

    def test_monotone_decreasing(self):
        h = np.linspace(0.0, 5e7, 200)
        alpha = b.earth_angular_radius(h)
        assert np.all(np.diff(alpha) < 0.0)

    def test_negative_altitude_rejected(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            b.earth_angular_radius(-1.0)


class TestJulianDate:
    def test_j2000_epoch(self):
        # 2000-01-01 12:00:00 is JD 2451545.0 by definition.
        assert b.julian_date(dt.datetime(2000, 1, 1, 12, 0, 0)) == pytest.approx(2451545.0)

    def test_known_date(self):
        # 2026-01-01 00:00:00 UTC. Days from 2000-01-01 12:00 UT:
        # 26 years, of which 2000, 2004, 2008, 2012, 2016, 2020, 2024 are leap
        # years -> 26*365 + 7 = 9497 days from 2000-01-01 to 2026-01-01, minus
        # the half day, so JD = 2451545.0 + 9497 - 0.5 = 2461041.5.
        assert b.julian_date(dt.datetime(2026, 1, 1)) == pytest.approx(2461041.5)

    def test_timezone_aware(self):
        naive = b.julian_date(dt.datetime(2026, 6, 1, 12))
        aware = b.julian_date(dt.datetime(2026, 6, 1, 12, tzinfo=dt.UTC))
        assert naive == pytest.approx(aware)

    def test_out_of_range_year(self):
        with pytest.raises(ValueError, match="validity range"):
            b.julian_date(dt.datetime(1850, 1, 1))


class TestSunDirection:
    def test_unit_vector(self):
        d, r = b.sun_direction_mod(2461041.5)
        assert np.linalg.norm(d) == pytest.approx(1.0)
        assert 0.98 * b.ASTRONOMICAL_UNIT_M < r < 1.02 * b.ASTRONOMICAL_UNIT_M

    def test_apparent_radius_about_a_quarter_degree(self):
        _, r = b.sun_direction_mod(2461041.5)
        alpha = b.angular_radius(b.SUN_RADIUS_M, r)
        assert 0.26 < np.degrees(alpha) < 0.28

    def test_array_input(self):
        jd = np.array([2461041.5, 2461141.5])
        d, r = b.sun_direction_mod(jd)
        assert d.shape == (2, 3)
        assert r.shape == (2,)

    def test_declination_bounded_by_obliquity(self):
        jd = 2461041.5 + np.arange(0.0, 366.0, 1.0)
        d, _ = b.sun_direction_mod(jd)
        dec = np.degrees(np.arcsin(d[:, 2]))
        assert dec.max() == pytest.approx(23.44, abs=0.05)
        assert dec.min() == pytest.approx(-23.44, abs=0.05)


class TestMoonDirection:
    def test_unit_vector_and_distance_range(self):
        d, r = b.moon_direction_mod(2461041.5)
        assert np.linalg.norm(d) == pytest.approx(1.0)
        assert 3.5e8 < r < 4.1e8

    def test_mean_distance(self):
        # Mean over one year against the published mean geocentric distance
        # 384 400 km (Archinal et al. 2018; Vallado 2013, Table D-3).
        jd = 2461041.5 + np.arange(0.0, 366.0, 0.25)
        _, r = b.moon_direction_mod(jd)
        assert r.mean() == pytest.approx(3.844e8, rel=0.01)

    def test_ecliptic_latitude_within_lunar_orbit_inclination(self):
        # The lunar orbit is inclined about 5.145 deg to the ecliptic, so the
        # ecliptic latitude never exceeds roughly 5.3 deg.
        jd = 2461041.5 + np.arange(0.0, 366.0, 0.1)
        d, _ = b.moon_direction_mod(jd)
        eps = np.radians(23.439291)
        z_ecl = -np.sin(eps) * d[:, 1] + np.cos(eps) * d[:, 2]
        assert np.degrees(np.arcsin(np.abs(z_ecl))).max() < 6.0

    def test_apparent_radius_about_half_a_degree_diameter(self):
        _, r = b.moon_direction_mod(2461041.5)
        alpha = b.angular_radius(b.MOON_RADIUS_M, r)
        assert 0.22 < np.degrees(alpha) < 0.30


class TestEarthDirectionFromPosition:
    def test_points_at_the_geocentre(self):
        d, alpha = b.earth_direction_from_position([b.EARTH_RADIUS_M + 400e3, 0.0, 0.0])
        assert np.allclose(d, [-1.0, 0.0, 0.0])
        assert alpha == pytest.approx(b.earth_angular_radius(400e3))

    def test_stack(self):
        r = np.array([[7e6, 0, 0], [0, 8e6, 0]])
        d, alpha = b.earth_direction_from_position(r)
        assert d.shape == (2, 3)
        assert alpha.shape == (2,)
        assert alpha[0] > alpha[1]


class TestConstants:
    def test_wgs84_radius(self):
        assert b.EARTH_RADIUS_M == 6378137.0

    def test_astronomical_unit_is_the_iau_exact_value(self):
        assert b.ASTRONOMICAL_UNIT_M == 149597870700.0

    def test_solar_angular_diameter_at_one_au(self):
        # arcsin(6.957e8 / 1.495978707e11) doubled: about 0.533 deg, the
        # familiar half-degree solar disc.
        alpha = b.angular_radius(b.SUN_RADIUS_M, b.ASTRONOMICAL_UNIT_M)
        assert np.degrees(2 * alpha) == pytest.approx(0.5329, abs=1e-3)
