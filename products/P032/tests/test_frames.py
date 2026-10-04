"""Frame and time utilities."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from constellink.frames import (
    WGS84_A_KM,
    WGS84_E2,
    datetime_to_jd,
    ecef_to_azel,
    geodetic_to_ecef,
    gmst_rad,
    teme_to_ecef,
    to_utc,
)


def test_to_utc_treats_naive_as_utc():
    naive = datetime(2026, 4, 1, 12, 0, 0)
    assert to_utc(naive).tzinfo is UTC
    assert to_utc(naive).hour == 12


def test_to_utc_rejects_non_datetime():
    with pytest.raises(TypeError):
        to_utc("2026-04-01")


def test_jd_of_j2000_epoch():
    # J2000.0 is 2000-01-01 12:00:00 TT; the JD of 2000-01-01 12:00 UTC is
    # 2451545.0 by definition of the Julian date (Vallado 2013, Ch. 3).
    jd, fr = datetime_to_jd(datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC))
    assert jd + fr == pytest.approx(2451545.0, abs=1e-9)


def test_gmst_rate_matches_the_iau1982_linear_term():
    # GMST advances 360.98564736629 deg per Julian day; measure the rate over
    # one day from the function itself.
    jd0 = 2451545.0
    g0 = gmst_rad(jd0)
    g1 = gmst_rad(jd0 + 1.0)
    rate_deg = np.rad2deg((g1 - g0) % (2.0 * np.pi))
    assert rate_deg == pytest.approx(360.98564736629 - 360.0, abs=1e-6)


def test_geodetic_to_ecef_equator_prime_meridian():
    # Latitude 0, longitude 0, altitude 0 is exactly (a, 0, 0) on WGS-84.
    r = geodetic_to_ecef(0.0, 0.0, 0.0)
    assert r[0] == pytest.approx(WGS84_A_KM, abs=1e-9)
    assert r[1] == pytest.approx(0.0, abs=1e-12)
    assert r[2] == pytest.approx(0.0, abs=1e-12)


def test_geodetic_to_ecef_pole():
    # At the pole the geocentric radius is the semi-minor axis b = a sqrt(1-e^2).
    b = WGS84_A_KM * np.sqrt(1.0 - WGS84_E2)
    r = geodetic_to_ecef(90.0, 0.0, 0.0)
    assert np.linalg.norm(r) == pytest.approx(b, abs=1e-9)
    assert r[2] == pytest.approx(b, abs=1e-9)


@pytest.mark.parametrize("lat,lon,alt", [(91.0, 0.0, 0.0), (0.0, 600.0, 0.0),
                                         (0.0, 0.0, -1.0)])
def test_geodetic_to_ecef_rejects_out_of_range(lat, lon, alt):
    with pytest.raises(ValueError):
        geodetic_to_ecef(lat, lon, alt)


def test_teme_to_ecef_preserves_norm_and_z():
    r = np.array([7000.0, -1200.0, 3400.0])
    out = teme_to_ecef(r, 2451545.0, 0.25)
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(r), rel=1e-14)
    assert out[2] == pytest.approx(r[2], rel=1e-14)


def test_teme_to_ecef_accepts_stacked_input():
    r = np.array([[7000.0, 0.0, 0.0], [0.0, 7000.0, 0.0]])
    out = teme_to_ecef(r, 2451545.0)
    assert out.shape == (2, 3)
    assert np.allclose(np.linalg.norm(out, axis=1), 7000.0)


def test_teme_to_ecef_rejects_bad_shape():
    with pytest.raises(ValueError):
        teme_to_ecef(np.zeros(4), 2451545.0)


def test_azel_zenith():
    # A satellite straight up from the equator/prime-meridian site must read
    # 90 deg elevation and a range equal to the altitude.
    site = geodetic_to_ecef(0.0, 0.0, 0.0)
    sat = site * (1.0 + 550.0 / WGS84_A_KM)
    _, el, rng = ecef_to_azel(sat, 0.0, 0.0, 0.0)
    assert el == pytest.approx(90.0, abs=1e-9)
    assert rng == pytest.approx(550.0, abs=1e-6)


def test_azel_horizon_due_north():
    # A point on the local horizon due north: start at the site, step along
    # the local north direction (which at the equator is +Z).
    site = geodetic_to_ecef(0.0, 0.0, 0.0)
    target = site + np.array([0.0, 0.0, 100.0])
    az, el, rng = ecef_to_azel(target, 0.0, 0.0, 0.0)
    assert az == pytest.approx(0.0, abs=1e-9)
    assert el == pytest.approx(0.0, abs=1e-9)
    assert rng == pytest.approx(100.0, abs=1e-12)


def test_azel_rejects_coincident_position():
    site = geodetic_to_ecef(10.0, 20.0, 0.0)
    with pytest.raises(ValueError):
        ecef_to_azel(site, 10.0, 20.0, 0.0)


@given(lat=st.floats(-89.9, 89.9), lon=st.floats(-179.9, 179.9),
       alt=st.floats(0.0, 10.0))
@settings(max_examples=40, deadline=None)
def test_geodetic_roundtrip_radius_is_monotone_in_altitude(lat, lon, alt):
    # Algebraic identity: raising the altitude strictly increases the
    # geocentric radius for any geodetic latitude.
    r0 = np.linalg.norm(geodetic_to_ecef(lat, lon, alt))
    r1 = np.linalg.norm(geodetic_to_ecef(lat, lon, alt + 1.0))
    assert r1 > r0


@given(jd_offset=st.floats(0.0, 3650.0))
@settings(max_examples=40, deadline=None)
def test_gmst_in_range(jd_offset):
    g = gmst_rad(2451545.0 + jd_offset)
    assert 0.0 <= g < 2.0 * np.pi


def test_datetime_to_jd_monotone():
    t0 = datetime(2026, 4, 1, tzinfo=UTC)
    a = sum(datetime_to_jd(t0))
    b = sum(datetime_to_jd(t0 + timedelta(hours=6)))
    assert b - a == pytest.approx(0.25, abs=1e-9)
