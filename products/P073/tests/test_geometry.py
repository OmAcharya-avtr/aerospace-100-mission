"""Earth geometry, sidereal time and look angles.

Known answers in this file are either closed-form exact values of the WGS-84
ellipsoid or configurations placed so that the answer is 0 or 90 degrees by
construction. The sidereal-time check is against ``sgp4.propagation.gstime``,
which is installed in this environment; nothing here is compared against a
package that is not installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sgp4.propagation import gstime

from contactsched.geometry import (
    EARTH_ROTATION_RAD_S,
    JD_J2000,
    MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG,
    WGS84_A_KM,
    WGS84_B_KM,
    WGS84_E2,
    WGS84_F,
    elevation_geocentric_norms_deg,
    elevation_geocentric_vector_deg,
    geodetic_to_ecef_km,
    gmst_rad,
    gmst_rad_array,
    look_angles_deg,
    max_geodetic_geocentric_deviation_deg,
    max_pass_duration_s,
    sez_matrix,
    teme_to_ecef_km,
)


class TestConstants:
    def test_eccentricity_squared_follows_from_flattening(self):
        # e2 = 2f - f**2 identically; this guards against a hand-typed constant.
        assert WGS84_E2 == pytest.approx(2.0 * WGS84_F - WGS84_F**2, rel=1e-15, abs=0.0)

    def test_semi_minor_axis(self):
        # b = a(1 - f) = 6356.752314245179 km for WGS-84.
        assert WGS84_B_KM == pytest.approx(6356.752314245179, rel=1e-12, abs=0.0)

    def test_eccentricity_squared_value(self):
        assert WGS84_E2 == pytest.approx(6.69437999014e-3, rel=1e-11, abs=0.0)

    def test_geodetic_geocentric_deviation_matches_closed_form(self):
        assert max_geodetic_geocentric_deviation_deg() == pytest.approx(
            MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG, rel=1e-8, abs=0.0
        )

    def test_earth_rotation_rate_gives_a_sidereal_day(self):
        # 2 pi / omega = 86164.09 s, the sidereal day, to within 0.01 s.
        assert 2.0 * math.pi / EARTH_ROTATION_RAD_S == pytest.approx(86164.0905, abs=0.01)


class TestGeodeticToEcef:
    def test_equator_prime_meridian_is_the_semi_major_axis(self):
        # Exact: at phi = 0 the curvature radius is a and z vanishes.
        xyz = geodetic_to_ecef_km(0.0, 0.0, 0.0)
        assert xyz[0] == pytest.approx(WGS84_A_KM, rel=1e-15, abs=0.0)
        assert xyz[1] == pytest.approx(0.0, abs=1e-12)
        assert xyz[2] == pytest.approx(0.0, abs=1e-12)

    def test_equator_ninety_east(self):
        xyz = geodetic_to_ecef_km(0.0, 90.0, 0.0)
        assert xyz[0] == pytest.approx(0.0, abs=1e-9)
        assert xyz[1] == pytest.approx(WGS84_A_KM, rel=1e-15, abs=0.0)

    def test_north_pole_is_the_semi_minor_axis(self):
        # At phi = 90, N = a / sqrt(1 - e2) and z = N(1 - e2) = a sqrt(1 - e2) = b.
        xyz = geodetic_to_ecef_km(90.0, 0.0, 0.0)
        assert xyz[2] == pytest.approx(WGS84_B_KM, rel=1e-12, abs=0.0)
        assert np.linalg.norm(xyz[:2]) == pytest.approx(0.0, abs=1e-9)

    def test_altitude_adds_along_the_normal_at_the_pole(self):
        low = geodetic_to_ecef_km(90.0, 0.0, 0.0)
        high = geodetic_to_ecef_km(90.0, 0.0, 10.0)
        assert high[2] - low[2] == pytest.approx(10.0, rel=1e-12, abs=0.0)

    def test_longitude_is_reduced_modulo_360(self):
        a = geodetic_to_ecef_km(45.0, 10.0, 0.3)
        b = geodetic_to_ecef_km(45.0, 370.0, 0.3)
        assert np.allclose(a, b, rtol=0.0, atol=1e-9)

    @pytest.mark.parametrize("lat", [-90.1, 90.1, 1e3, -1e3])
    def test_latitude_out_of_range_raises(self, lat):
        with pytest.raises(ValueError, match="geodetic latitude"):
            geodetic_to_ecef_km(lat, 0.0, 0.0)

    def test_altitude_below_the_centre_raises(self):
        with pytest.raises(ValueError, match="Earth's centre"):
            geodetic_to_ecef_km(0.0, 0.0, -WGS84_B_KM)

    def test_non_finite_longitude_raises(self):
        with pytest.raises(ValueError, match="longitude must be finite"):
            geodetic_to_ecef_km(0.0, float("nan"), 0.0)

    def test_geocentric_radius_lies_between_the_two_axes(self):
        for lat in np.linspace(-90.0, 90.0, 37):
            radius = float(np.linalg.norm(geodetic_to_ecef_km(float(lat), 17.0, 0.0)))
            assert WGS84_B_KM - 1e-9 <= radius <= WGS84_A_KM + 1e-9


class TestSezMatrix:
    def test_rows_are_orthonormal(self):
        matrix = sez_matrix(37.5, -121.0)
        product = matrix @ matrix.T
        assert np.allclose(product, np.eye(3), rtol=0.0, atol=1e-14)

    def test_determinant_is_plus_one(self):
        # (south, east, zenith) in that order is right-handed: E x Z = S, so the
        # determinant is S . S = +1. An earlier version of this test asserted -1
        # on the assumption that SEZ is left-handed, and was wrong.
        assert np.linalg.det(sez_matrix(12.0, 34.0)) == pytest.approx(1.0, rel=1e-13, abs=0.0)

    def test_east_cross_zenith_is_south(self):
        matrix = sez_matrix(-33.0, 151.0)
        assert np.allclose(np.cross(matrix[1], matrix[2]), matrix[0], rtol=0.0, atol=1e-15)

    def test_zenith_row_is_the_ellipsoid_normal_at_the_equator(self):
        matrix = sez_matrix(0.0, 0.0)
        assert np.allclose(matrix[2], np.array([1.0, 0.0, 0.0]), rtol=0.0, atol=1e-15)

    def test_east_row_has_no_vertical_component(self):
        for lat in (-80.0, -10.0, 0.0, 10.0, 80.0):
            assert sez_matrix(lat, 44.0)[1, 2] == pytest.approx(0.0, abs=0.0)


class TestGmst:
    def test_agrees_with_sgp4_gstime_at_j2000(self):
        mine = gmst_rad(JD_J2000)
        theirs = gstime(JD_J2000)
        assert mine == pytest.approx(theirs, rel=0.0, abs=1e-9)

    @pytest.mark.parametrize("offset_days", [-20000.0, -365.25, 0.0, 1.0, 1234.5678, 20000.0])
    def test_agrees_with_sgp4_gstime_over_sixty_years(self, offset_days):
        jd = JD_J2000 + offset_days
        difference = abs(gmst_rad(jd) - gstime(jd))
        difference = min(difference, 2.0 * math.pi - difference)
        assert difference < 1e-8

    def test_is_within_the_principal_range(self):
        for jd in (JD_J2000 - 1e4, JD_J2000, JD_J2000 + 1e4, 2461000.5):
            assert 0.0 <= gmst_rad(jd) < 2.0 * math.pi

    def test_advances_by_one_sidereal_day_per_solar_day(self):
        # One solar day advances GMST by 2 pi * 86400 / 86164.0905 radians.
        first = gmst_rad(2461000.5)
        second = gmst_rad(2461001.5)
        advance = (second - first) % (2.0 * math.pi)
        expected = (2.0 * math.pi * 86400.0 / 86164.0905) % (2.0 * math.pi)
        assert advance == pytest.approx(expected, rel=0.0, abs=1e-5)

    def test_array_form_matches_the_scalar_form(self):
        jds = np.array([JD_J2000, 2461000.5, 2461000.75, 2470000.123])
        assert np.allclose(
            gmst_rad_array(jds), [gmst_rad(float(j)) for j in jds], rtol=0.0, atol=1e-14
        )


class TestTemeToEcef:
    def test_rotation_preserves_the_norm(self):
        teme = np.array([[7000.0, -1200.0, 300.0], [-500.0, 6800.0, -4000.0]])
        jd = np.array([2461000.5, 2461000.6])
        ecef = teme_to_ecef_km(teme, jd)
        assert np.allclose(
            np.linalg.norm(ecef, axis=1), np.linalg.norm(teme, axis=1), rtol=0.0, atol=1e-9
        )

    def test_rotation_leaves_z_alone(self):
        teme = np.array([[7000.0, -1200.0, 321.5]])
        ecef = teme_to_ecef_km(teme, np.array([2461000.5]))
        assert ecef[0, 2] == pytest.approx(321.5, rel=0.0, abs=0.0)

    def test_single_position_returns_a_vector(self):
        out = teme_to_ecef_km(np.array([7000.0, 0.0, 0.0]), 2461000.5)
        assert out.shape == (3,)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="positions and"):
            teme_to_ecef_km(np.zeros((3, 3)), np.array([1.0, 2.0]))

    def test_wrong_last_dimension_raises(self):
        with pytest.raises(ValueError, match="last dimension 3"):
            teme_to_ecef_km(np.zeros((2, 4)), np.array([1.0, 2.0]))


class TestLookAngles:
    def test_overhead_is_ninety_degrees(self):
        station = geodetic_to_ecef_km(0.0, 0.0, 0.0)
        satellite = (station * 1.1)[None, :]
        _az, el, rng = look_angles_deg(satellite, station, sez_matrix(0.0, 0.0))
        assert el[0] == pytest.approx(90.0, rel=0.0, abs=1e-9)
        assert rng[0] == pytest.approx(0.1 * WGS84_A_KM, rel=1e-12, abs=0.0)

    def test_due_east_on_the_horizon_at_the_equator(self):
        # Station at (a, 0, 0); a point displaced purely in +y is due east and,
        # measured against the equatorial ellipsoid normal (which is +x there),
        # exactly on the horizon.
        station = geodetic_to_ecef_km(0.0, 0.0, 0.0)
        satellite = (station + np.array([0.0, 500.0, 0.0]))[None, :]
        az, el, rng = look_angles_deg(satellite, station, sez_matrix(0.0, 0.0))
        assert el[0] == pytest.approx(0.0, rel=0.0, abs=1e-12)
        assert az[0] == pytest.approx(90.0, rel=0.0, abs=1e-12)
        assert rng[0] == pytest.approx(500.0, rel=1e-15, abs=0.0)

    def test_due_north_is_azimuth_zero(self):
        station = geodetic_to_ecef_km(0.0, 0.0, 0.0)
        satellite = (station + np.array([0.0, 0.0, 500.0]))[None, :]
        az, el, _rng = look_angles_deg(satellite, station, sez_matrix(0.0, 0.0))
        assert az[0] == pytest.approx(0.0, rel=0.0, abs=1e-12)
        assert el[0] == pytest.approx(0.0, rel=0.0, abs=1e-12)

    def test_azimuth_is_in_the_principal_range(self):
        rng_gen = np.random.default_rng(3)
        station = geodetic_to_ecef_km(45.0, 12.0, 0.2)
        satellite = station[None, :] + rng_gen.normal(scale=3000.0, size=(200, 3))
        az, _el, _r = look_angles_deg(satellite, station, sez_matrix(45.0, 12.0))
        assert np.all((az >= 0.0) & (az < 360.0))

    def test_elevation_is_bounded(self):
        rng_gen = np.random.default_rng(4)
        station = geodetic_to_ecef_km(-20.0, 100.0, 0.0)
        satellite = station[None, :] + rng_gen.normal(scale=5000.0, size=(300, 3))
        _az, el, _r = look_angles_deg(satellite, station, sez_matrix(-20.0, 100.0))
        assert np.all((el >= -90.0 - 1e-12) & (el <= 90.0 + 1e-12))


class TestGeocentricElevations:
    def test_radial_point_is_ninety_degrees_both_ways(self):
        station = geodetic_to_ecef_km(30.0, 70.0, 0.0)
        radius = float(np.linalg.norm(station))
        satellite = (station * 2.0)[None, :]
        vector = elevation_geocentric_vector_deg(satellite, station)
        norms = elevation_geocentric_norms_deg(
            np.array([2.0 * radius]), radius, np.array([radius])
        )
        assert vector[0] == pytest.approx(90.0, rel=0.0, abs=1e-9)
        assert norms[0] == pytest.approx(90.0, rel=0.0, abs=1e-9)

    def test_tangential_point_is_zero_both_ways(self):
        # Station at (R, 0, 0), satellite at (R, d, 0): the slant vector is
        # perpendicular to the station position vector, so the geocentric
        # elevation is exactly zero, and the law of cosines gives
        # (R**2 + d**2 - R**2 - d**2) / (2 R d) = 0.
        station = np.array([WGS84_A_KM, 0.0, 0.0])
        offset = 500.0
        satellite = np.array([[WGS84_A_KM, offset, 0.0]])
        radius_sat = float(np.linalg.norm(satellite[0]))
        assert elevation_geocentric_vector_deg(satellite, station)[0] == pytest.approx(
            0.0, rel=0.0, abs=1e-12
        )
        assert elevation_geocentric_norms_deg(
            np.array([radius_sat]), WGS84_A_KM, np.array([offset])
        )[0] == pytest.approx(0.0, rel=0.0, abs=1e-9)

    def test_the_two_derivations_agree_away_from_the_horizon(self):
        rng_gen = np.random.default_rng(11)
        station = geodetic_to_ecef_km(55.0, -3.0, 0.1)
        radius = float(np.linalg.norm(station))
        satellite = station[None, :] + rng_gen.normal(scale=1500.0, size=(400, 3))
        satellite = satellite[np.linalg.norm(satellite, axis=1) > radius + 200.0]
        vector = elevation_geocentric_vector_deg(satellite, station)
        norms = elevation_geocentric_norms_deg(
            np.linalg.norm(satellite, axis=1),
            radius,
            np.linalg.norm(satellite - station[None, :], axis=1),
        )
        high = np.abs(vector) > 10.0
        assert np.max(np.abs(vector[high] - norms[high])) < 1e-6

    def test_geodetic_and_geocentric_differ_by_less_than_the_derived_bound(self):
        # An independently derived BOUND, not a second implementation: the angle
        # between the ellipsoid normal and the position vector is at most
        # atan(e2 / (2 sqrt(1 - e2))), so the two elevations of the same slant
        # vector cannot differ by more than that.
        rng_gen = np.random.default_rng(12)
        for lat in (-75.0, -45.0, -10.0, 0.0, 10.0, 45.0, 75.0):
            station = geodetic_to_ecef_km(lat, 25.0, 0.0)
            satellite = station[None, :] + rng_gen.normal(scale=2000.0, size=(120, 3))
            _az, geodetic, _r = look_angles_deg(satellite, station, sez_matrix(lat, 25.0))
            geocentric = elevation_geocentric_vector_deg(satellite, station)
            worst = float(np.max(np.abs(geodetic - geocentric)))
            assert worst <= MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG + 1e-9


class TestMaxPassDuration:
    def test_decreases_with_minimum_elevation(self):
        durations = [
            max_pass_duration_s(7000.0, 0.0, WGS84_A_KM, el) for el in (0.0, 5.0, 10.0, 20.0)
        ]
        assert durations == sorted(durations, reverse=True)

    def test_increases_with_altitude(self):
        low = max_pass_duration_s(6900.0, 0.0, WGS84_A_KM, 5.0)
        high = max_pass_duration_s(7600.0, 0.0, WGS84_A_KM, 5.0)
        assert high > low

    def test_a_seven_thousand_kilometre_circular_orbit_bound(self):
        # gamma_max = acos(6378.137 / 7000) - 0 = 0.42452 rad; n = sqrt(mu / a**3)
        # = 1.07847e-3 rad/s; relative rate = n - omega_E = 1.00555e-3 rad/s;
        # bound = 2 gamma / rate. Recomputed here from the same closed form at a
        # loose tolerance, which checks the assembly and not the algebra.
        gamma = math.acos(WGS84_A_KM / 7000.0)
        rate = math.sqrt(398600.5 / 7000.0**3) - EARTH_ROTATION_RAD_S
        assert max_pass_duration_s(7000.0, 0.0, WGS84_A_KM, 0.0) == pytest.approx(
            2.0 * gamma / rate, rel=1e-12, abs=0.0
        )

    def test_above_geosynchronous_raises(self):
        with pytest.raises(ValueError, match="Earth rotation rate"):
            max_pass_duration_s(60000.0, 0.0, WGS84_A_KM, 5.0)

    def test_near_geosynchronous_bound_is_vacuous_rather_than_raising(self):
        # At a = 42164 km the mean motion exceeds the Earth rotation rate by
        # 4.5e-10 rad/s, so the guard does not fire and the bound is correct but
        # useless: of order 1e9 s. Asserted rather than left implicit, because a
        # reader could otherwise take a returned number as informative.
        bound = max_pass_duration_s(42164.0, 0.0, WGS84_A_KM, 0.0)
        assert bound > 1.0e9

    def test_orbit_below_the_station_raises(self):
        with pytest.raises(ValueError, match="never rises above"):
            max_pass_duration_s(6000.0, 0.0, WGS84_A_KM, 5.0)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(semi_major_axis_km=-1.0), "semi-major axis"),
            (dict(eccentricity=1.0), "eccentricity"),
            (dict(eccentricity=-0.1), "eccentricity"),
            (dict(station_radius_km=0.0), "station radius"),
            (dict(min_elevation_deg=90.0), "minimum elevation"),
        ],
    )
    def test_invalid_inputs_raise(self, kwargs, match):
        base = dict(
            semi_major_axis_km=7000.0,
            eccentricity=0.0,
            station_radius_km=WGS84_A_KM,
            min_elevation_deg=5.0,
        )
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            max_pass_duration_s(**base)
