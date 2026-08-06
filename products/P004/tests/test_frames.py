"""Frame / geometry known-answer and property tests."""

from datetime import datetime, timezone

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from passplanner.frames import (
    WGS84_A_KM,
    datetime_to_jd,
    ecef_to_azel,
    geodetic_to_ecef,
    gmst_rad,
    teme_to_ecef,
    to_utc,
)


def test_geodetic_to_ecef_equator_prime_meridian():
    # Hand calculation: lat = 0 -> N = a, so r = (a + h, 0, 0) exactly.
    # a = 6378.137 km (WGS-84), h = 0 -> (6378.137, 0, 0).
    r = geodetic_to_ecef(0.0, 0.0, 0.0)
    assert r == pytest.approx([WGS84_A_KM, 0.0, 0.0], abs=1e-9)


def test_geodetic_to_ecef_north_pole():
    # Hand calculation: lat = 90 -> N = a / sqrt(1 - e^2), z = N(1 - e^2) = a(1 - f)
    # = 6378.137 * (1 - 1/298.257223563) = 6356.752314... km (WGS-84 polar radius b).
    r = geodetic_to_ecef(90.0, 0.0, 0.0)
    b_km = WGS84_A_KM * (1.0 - 1.0 / 298.257223563)
    assert r[0] == pytest.approx(0.0, abs=1e-9)
    assert r[1] == pytest.approx(0.0, abs=1e-9)
    assert r[2] == pytest.approx(b_km, abs=1e-6)
    assert b_km == pytest.approx(6356.752314, abs=1e-5)


def _local_vertical(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Geodetic local vertical (zenith) unit vector in ECEF.

    NOTE: this is NOT the geocentric radial direction r_site/|r_site| except
    at the equator and poles; the two differ by up to ~0.19 deg on WGS-84.
    """
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


@pytest.mark.parametrize("lat,lon", [(0.0, 0.0), (45.0, 10.0), (-33.0, 151.0), (89.0, -70.0)])
def test_overhead_elevation_is_90_deg(lat, lon):
    # Known answer: a satellite on the geodetic local vertical is at elevation
    # exactly 90 deg, by definition of the SEZ frame.
    r_site = geodetic_to_ecef(lat, lon, 0.0)
    r_sat = r_site + 500.0 * _local_vertical(lat, lon)
    az, el, rng = ecef_to_azel(r_sat, lat, lon, 0.0)
    assert el == pytest.approx(90.0, abs=1e-9)
    assert rng == pytest.approx(500.0, rel=1e-12)
    assert 0.0 <= az < 360.0


def test_horizon_elevation_is_zero_and_azimuth_north():
    # Station at (0, 0); a point due north along the local horizon.
    # Local north at the equator/prime meridian is +Z in ECEF, so a target at
    # r_site + d * z_hat is at elevation 0 (hand check) and azimuth 0 deg.
    r_site = geodetic_to_ecef(0.0, 0.0, 0.0)
    r_sat = r_site + np.array([0.0, 0.0, 1000.0])
    az, el, rng = ecef_to_azel(r_sat, 0.0, 0.0, 0.0)
    assert el == pytest.approx(0.0, abs=1e-9)
    assert az == pytest.approx(0.0, abs=1e-9)
    assert rng == pytest.approx(1000.0, rel=1e-12)


def test_east_horizon_azimuth_is_90_deg():
    r_site = geodetic_to_ecef(0.0, 0.0, 0.0)
    r_sat = r_site + np.array([0.0, 1000.0, 0.0])
    az, el, _ = ecef_to_azel(r_sat, 0.0, 0.0, 0.0)
    assert el == pytest.approx(0.0, abs=1e-9)
    assert az == pytest.approx(90.0, abs=1e-9)


def test_nadir_direction_is_minus_90_deg():
    r_site = geodetic_to_ecef(10.0, 20.0, 0.0)
    r_sat = r_site - 100.0 * _local_vertical(10.0, 20.0)
    _, el, rng = ecef_to_azel(r_sat, 10.0, 20.0, 0.0)
    assert el == pytest.approx(-90.0, abs=1e-9)
    assert rng == pytest.approx(100.0, rel=1e-12)


def test_gmst_j2000_reference_value():
    # IAU 1982 polynomial at JD 2451545.0 (2000-01-01 12:00 UT1):
    # GMST = 280.46061837 deg exactly (the constant term), i.e. 18h 41m 50.55s.
    theta = np.rad2deg(gmst_rad(2451545.0, 0.0))
    assert theta == pytest.approx(280.46061837, abs=1e-9)


def test_gmst_advances_by_one_sidereal_day():
    # A mean sidereal day is 360 / 360.98564736629 mean solar days.
    jd0 = 2451545.0
    dt = 360.0 / 360.98564736629
    t0 = gmst_rad(jd0, 0.0)
    t1 = gmst_rad(jd0, dt)
    assert np.rad2deg(abs(((t1 - t0) + np.pi) % (2 * np.pi) - np.pi)) < 1e-4


def test_teme_to_ecef_preserves_norm_and_z():
    r = np.array([4000.0, 5000.0, 3000.0])
    out = teme_to_ecef(r, 2454730.0, 0.5)
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(r), rel=1e-12)
    assert out[2] == pytest.approx(r[2], rel=1e-12)


def test_datetime_to_jd_j2000():
    # 2000-01-01 12:00:00 UTC == JD 2451545.0
    jd, fr = datetime_to_jd(datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    assert jd + fr == pytest.approx(2451545.0, abs=1e-9)


def test_to_utc_naive_is_treated_as_utc():
    t = to_utc(datetime(2020, 1, 1, 0, 0, 0))
    assert t.tzinfo is timezone.utc


@settings(max_examples=50, deadline=None)
@given(lat=st.floats(-89.9, 89.9), lon=st.floats(-179.9, 179.9),
       alt=st.floats(0.0, 5.0))
def test_property_site_at_own_location_has_zero_range_ratio(lat, lon, alt):
    # Property: |r_site| from geodetic_to_ecef must lie between the WGS-84
    # polar and equatorial radii (plus altitude).
    r = np.linalg.norm(geodetic_to_ecef(lat, lon, alt))
    b = WGS84_A_KM * (1.0 - 1.0 / 298.257223563)
    assert b - 1e-6 <= r <= WGS84_A_KM + alt + 1e-6


@settings(max_examples=50, deadline=None)
@given(lat=st.floats(-89.0, 89.0), lon=st.floats(-179.0, 179.0),
       h=st.floats(200.0, 2000.0))
def test_property_overhead_always_90(lat, lon, h):
    r_site = geodetic_to_ecef(lat, lon, 0.0)
    r_sat = r_site + h * _local_vertical(lat, lon)
    _, el, rng = ecef_to_azel(r_sat, lat, lon, 0.0)
    assert el == pytest.approx(90.0, abs=1e-6)
    assert rng == pytest.approx(h, rel=1e-9)


@pytest.mark.parametrize("bad", [(-91.0, 0.0, 0.0), (91.0, 0.0, 0.0), (0.0, 0.0, -1.0)])
def test_geodetic_to_ecef_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        geodetic_to_ecef(*bad)


def test_azel_rejects_bad_shape():
    with pytest.raises(ValueError):
        ecef_to_azel(np.array([1.0, 2.0]), 0.0, 0.0, 0.0)


def test_azel_rejects_coincident_position():
    r_site = geodetic_to_ecef(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="coincides"):
        ecef_to_azel(r_site, 0.0, 0.0, 0.0)


def test_to_utc_rejects_non_datetime():
    with pytest.raises(TypeError):
        to_utc("2020-01-01")
