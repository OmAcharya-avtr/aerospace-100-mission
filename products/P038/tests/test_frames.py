"""Frames: geodetic conversion, GMST, station state with rotation velocity."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sgp4.api import Satrec

from dopplerkit.constants import OMEGA_EARTH_RAD_S, WGS84_A_M
from dopplerkit.doppler import one_way_doppler_hz
from dopplerkit.frames import (
    OMEGA_EARTH_VEC_RAD_S,
    ecef_to_teme_m,
    elevation_deg,
    geodetic_to_ecef_m,
    gmst_rad,
    julian_date,
    satellite_state_teme,
    station_state_teme,
)
from dopplerkit.geometry import range_rate_mps, slant_range_m

# ISS TLE from the sgp4 package documentation example, epoch 2019-12-09.
ISS_LINE1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
ISS_LINE2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"
CARRIER_HZ = 2.2e9


def test_geodetic_to_ecef_equator_prime_meridian():
    # lat 0, lon 0, alt 0 -> (a, 0, 0) exactly, a = 6378137 m.
    r = geodetic_to_ecef_m(0.0, 0.0, 0.0)
    assert r[0] == pytest.approx(WGS84_A_M, rel=1e-15)
    assert r[1] == pytest.approx(0.0, abs=1e-9)
    assert r[2] == pytest.approx(0.0, abs=1e-9)


def test_geodetic_to_ecef_north_pole_is_the_polar_radius():
    # b = a (1 - f) = 6356752.3142... m (WGS-84 derived semi-minor axis).
    r = geodetic_to_ecef_m(90.0, 0.0, 0.0)
    b = WGS84_A_M * (1.0 - 1.0 / 298.257223563)
    assert r[2] == pytest.approx(b, rel=1e-12)
    assert math.hypot(r[0], r[1]) == pytest.approx(0.0, abs=1e-6)


def test_geodetic_to_ecef_altitude_adds_along_the_normal_at_the_equator():
    r0 = geodetic_to_ecef_m(0.0, 0.0, 0.0)
    r1 = geodetic_to_ecef_m(0.0, 0.0, 1000.0)
    assert np.linalg.norm(r1) - np.linalg.norm(r0) == pytest.approx(1000.0, rel=1e-9)


def test_geodetic_validation():
    with pytest.raises(ValueError, match=r"lat_deg must be in \[-90, 90\]"):
        geodetic_to_ecef_m(91.0, 0.0, 0.0)
    with pytest.raises(ValueError, match=r"lon_deg must be in \[-540, 540\]"):
        geodetic_to_ecef_m(0.0, 1000.0, 0.0)
    with pytest.raises(ValueError, match="alt_m must be >= -500"):
        geodetic_to_ecef_m(0.0, 0.0, -1000.0)


def test_ecef_to_teme_preserves_length_and_z():
    r = geodetic_to_ecef_m(45.0, 30.0, 200.0)
    jd, fr = julian_date(datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC))
    r_teme = ecef_to_teme_m(r, jd, fr)
    assert np.linalg.norm(r_teme) == pytest.approx(np.linalg.norm(r), rel=1e-14)
    assert r_teme[2] == pytest.approx(r[2], rel=1e-15)


def test_ecef_to_teme_shape_validation():
    with pytest.raises(ValueError, match=r"r_ecef_m must have shape \(3,\)"):
        ecef_to_teme_m(np.zeros(2), 2451545.0)


def test_gmst_at_j2000_matches_the_published_value():
    # The IAU 1982 polynomial at d = 0 gives GMST = 280.46061837 deg, which is
    # 18h 41m 50.548 s of sidereal time (Aoki et al. 1982; Meeus 2nd ed.
    # Eq. 12.4). Checking the constant term pins the polynomial.
    assert math.degrees(gmst_rad(2451545.0, 0.0)) == pytest.approx(280.46061837, abs=1e-8)


def test_gmst_advances_by_one_sidereal_rotation_per_day():
    g0 = math.degrees(gmst_rad(2451545.0, 0.0))
    g1 = math.degrees(gmst_rad(2451546.0, 0.0))
    advance = (g1 - g0) % 360.0
    assert advance == pytest.approx(360.98564736629 % 360.0, abs=1e-6)


def test_julian_date_requires_a_datetime():
    with pytest.raises(TypeError, match="expected datetime"):
        julian_date("2026-01-01")


def test_julian_date_treats_naive_as_utc():
    naive = datetime(2026, 1, 1, 0, 0, 0)  # noqa: DTZ001 -- naive input is the point
    aware = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert julian_date(naive) == julian_date(aware)


def test_station_velocity_is_omega_cross_r():
    t = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
    sta = station_state_teme(0.0, 0.0, 0.0, t)
    expected = np.cross(OMEGA_EARTH_VEC_RAD_S, sta.position_m)
    assert np.allclose(sta.velocity_mps, expected, rtol=1e-14)
    # Equatorial station: |v| = omega * a = 7.292115e-5 * 6378137 = 465.1 m/s.
    assert sta.speed_mps == pytest.approx(OMEGA_EARTH_RAD_S * WGS84_A_M, rel=1e-9)
    assert 464.0 < sta.speed_mps < 466.0


def test_station_velocity_scales_with_cosine_of_latitude():
    t = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
    eq = station_state_teme(0.0, 0.0, 0.0, t).speed_mps
    for lat in (30.0, 51.1450, 78.23):
        v = station_state_teme(lat, 0.0, 0.0, t).speed_mps
        # Geodetic vs geocentric latitude differ, so allow 0.5 % on the cosine.
        assert v == pytest.approx(eq * math.cos(math.radians(lat)), rel=5e-3)


def test_station_velocity_is_never_zero():
    """The whole point of this module: no silent zero-velocity station."""
    t = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
    for lat in (-60.0, 0.0, 45.0, 80.0):
        assert station_state_teme(lat, 10.0, 0.0, t).speed_mps > 80.0


def test_station_rotation_velocity_is_worth_kilohertz():
    """Quantifies the error that dropping station velocity would introduce."""
    t = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
    sta = station_state_teme(0.0, 0.0, 0.0, t)
    shift = abs(one_way_doppler_hz(sta.speed_mps, CARRIER_HZ))
    assert shift > 3.0e3  # > 3 kHz at 2.2 GHz


def test_station_acceleration_is_centripetal():
    t = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
    sta = station_state_teme(0.0, 0.0, 0.0, t)
    # |a| = omega^2 * r = 0.0339 m/s^2 at the equator, pointing inward.
    assert np.linalg.norm(sta.acceleration_mps2) == pytest.approx(
        OMEGA_EARTH_RAD_S**2 * np.linalg.norm(sta.position_m), rel=1e-12
    )
    assert sta.acceleration_mps2 @ sta.position_m < 0.0


def test_sgp4_state_has_plausible_leo_magnitudes():
    satrec = Satrec.twoline2rv(ISS_LINE1, ISS_LINE2)
    t = datetime(2019, 12, 9, 16, 39, 5, tzinfo=UTC)
    sat = satellite_state_teme(satrec, t)
    r = float(np.linalg.norm(sat.position_m))
    assert 6.6e6 < r < 6.9e6  # ISS altitude band
    assert 7.5e3 < sat.speed_mps < 7.8e3
    assert np.array_equal(sat.acceleration_mps2, np.zeros(3))


def test_known_iss_pass_culmination_geometry():
    """Regression values from this repository's own sgp4 run.

    Culmination of the documentation-example ISS TLE over Chilbolton
    (51.1450 N, 1.4365 W, 100 m) on 2019-12-09, located by a 5 s elevation
    scan in validation/validate_tle_pass.py. These are self-consistency
    regression values, not an external reference: they pin the frame
    reduction and the station velocity so a change in either is visible.
    """
    satrec = Satrec.twoline2rv(ISS_LINE1, ISS_LINE2)
    lat, lon, alt = 51.1450, -1.4365, 100.0
    t = datetime(2019, 12, 9, 16, 39, 5, tzinfo=UTC)
    sat = satellite_state_teme(satrec, t)
    sta = station_state_teme(lat, lon, alt, t)
    jd, fr = julian_date(t)
    el = elevation_deg(sta, sat, lat, lon, jd, fr)
    assert el == pytest.approx(59.739, abs=0.01)
    assert slant_range_m(sta, sat) / 1e3 == pytest.approx(483.066, abs=0.01)
    assert range_rate_mps(sta, sat) == pytest.approx(242.359, abs=0.01)


def test_sgp4_error_is_raised_not_swallowed():
    satrec = Satrec.twoline2rv(ISS_LINE1, ISS_LINE2)
    # Far from epoch the ISS TLE decays below the Earth and SGP4 errors.
    with pytest.raises(RuntimeError, match="sgp4 propagation failed with error code"):
        satellite_state_teme(satrec, datetime(2200, 1, 1, tzinfo=UTC))


def test_elevation_is_ninety_degrees_straight_up():
    t = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
    lat, lon = 35.0, 140.0
    sta = station_state_teme(lat, lon, 0.0, t)
    jd, fr = julian_date(t)
    # Place the satellite 500 km along the station's own radius-ish direction:
    # the local up at the geodetic latitude, built the same way as in the code.
    theta = gmst_rad(jd, fr)
    lat_r, lon_r = math.radians(lat), math.radians(lon) + theta
    up = np.array([
        math.cos(lat_r) * math.cos(lon_r),
        math.cos(lat_r) * math.sin(lon_r),
        math.sin(lat_r),
    ])
    from dopplerkit.geometry import State

    sat = State(position_m=sta.position_m + 500e3 * up, velocity_mps=np.zeros(3))
    assert elevation_deg(sta, sat, lat, lon, jd, fr) == pytest.approx(90.0, abs=1e-6)


def test_elevation_is_negative_below_the_horizon():
    t = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
    lat, lon = 35.0, 140.0
    sta = station_state_teme(lat, lon, 0.0, t)
    opposite = station_state_teme(-lat, lon + 180.0, 500e3, t)
    jd, fr = julian_date(t)
    assert elevation_deg(sta, opposite, lat, lon, jd, fr) < 0.0


def test_station_state_is_consistent_over_a_short_interval():
    """Finite-differencing the station position must reproduce omega x r."""
    t = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
    lat, lon, alt = 51.145, -1.4365, 100.0
    h = 0.5
    p_minus = station_state_teme(lat, lon, alt, t - timedelta(seconds=h / 2)).position_m
    p_plus = station_state_teme(lat, lon, alt, t + timedelta(seconds=h / 2)).position_m
    fd = (p_plus - p_minus) / h
    analytic = station_state_teme(lat, lon, alt, t).velocity_mps
    # GMST advances at the sidereal rate 7.29212e-5 rad/s while the analytic
    # velocity uses the WGS-84 nominal 7.292115e-5; agreement to 1e-4 relative
    # is the size of that difference plus the central-difference truncation.
    assert np.allclose(fd, analytic, rtol=1e-4, atol=1e-3)
