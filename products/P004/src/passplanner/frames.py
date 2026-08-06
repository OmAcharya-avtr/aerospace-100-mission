"""Coordinate frames and time utilities for pass planning.

Conventions
-----------
* Positions in kilometres, angles in degrees at the public API (radians internally).
* Times are Python ``datetime`` objects; naive datetimes are interpreted as UTC.

Earth-rotation model (documented simplification)
------------------------------------------------
SGP4 outputs positions in the TEME frame.  This module converts TEME to an
Earth-fixed frame (ECEF) using a single rotation about the Z axis by the
Greenwich Mean Sidereal Time (GMST, IAU 1982 model; Vallado 2013,
"Fundamentals of Astrodynamics and Applications", 4th ed., Eq. 3-45).
The simplification neglects:

* polar motion (< ~1 arcsec, sub-metre-level ground displacement),
* the difference UT1 - UTC (bounded by 0.9 s, i.e. <= 0.00375 deg of Earth
  rotation, ~0.45 km at the equator),
* the equation of the equinoxes / TEME-vs-PEF subtleties (arcsecond class).

Accuracy class: the induced along-track timing error on LEO pass rise/set
times is well below one second, which is negligible for contact scheduling
(coarse scan steps are tens of seconds).  This is the standard "GMST-only"
reduction described in Vallado 2013 Sec. 3.7 for scheduling-class work.
It is NOT suitable for precision pointing or orbit determination.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sgp4.functions import jday

# WGS-84 defining parameters (NIMA TR8350.2, 3rd ed., 2000)
WGS84_A_KM = 6378.137          # semi-major axis [km]
WGS84_F = 1.0 / 298.257223563  # flattening [-]
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared [-]

MU_EARTH_KM3_S2 = 398600.4418  # WGS-84 gravitational parameter [km^3/s^2]


def to_utc(t: datetime) -> datetime:
    """Return ``t`` as a timezone-aware UTC datetime (naive input = UTC)."""
    if not isinstance(t, datetime):
        raise TypeError(f"expected datetime, got {type(t).__name__}")
    if t.tzinfo is None:
        return t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def datetime_to_jd(t: datetime) -> tuple[float, float]:
    """Convert a UTC datetime to Julian date as (whole, fraction).

    Uses the algorithm shipped with the ``sgp4`` package (valid 1900-2100,
    Vallado 2013 Alg. 14).  UTC is used in place of UT1 (see module notes).
    """
    t = to_utc(t)
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                  t.second + t.microsecond * 1e-6)
    return jd, fr


def gmst_rad(jd: float, fr: float = 0.0) -> float:
    """Greenwich Mean Sidereal Time [rad] for Julian date jd + fr (UT1~UTC).

    IAU 1982 GMST polynomial (Vallado 2013 Eq. 3-45; equivalently Meeus 1998,
    "Astronomical Algorithms", 2nd ed., Eq. 12.4)::

        GMST[deg] = 280.46061837 + 360.98564736629 * d
                    + 0.000387933 * T^2 - T^3 / 38 710 000

    with d = JD_UT1 - 2451545.0 and T = d / 36525.  Valid for the era around
    J2000 (formal accuracy ~0.1 arcsec, dominated in practice here by the
    UT1-UTC neglect documented in the module docstring).
    """
    d = (jd - 2451545.0) + fr
    t_cent = d / 36525.0
    gmst_deg = (280.46061837
                + 360.98564736629 * d
                + 0.000387933 * t_cent**2
                - t_cent**3 / 38710000.0)
    return np.deg2rad(gmst_deg % 360.0)


def teme_to_ecef(r_teme_km: np.ndarray, jd: float, fr: float = 0.0) -> np.ndarray:
    """Rotate a TEME position vector [km] into ECEF via GMST (Vallado Sec. 3.7).

    r_ECEF = R3(theta_GMST) . r_TEME, where R3 is the standard rotation about
    +Z.  See module docstring for the neglected terms and accuracy class.
    """
    r = np.asarray(r_teme_km, dtype=float)
    if r.shape != (3,):
        raise ValueError(f"position vector must have shape (3,), got {r.shape}")
    theta = gmst_rad(jd, fr)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * r[0] + s * r[1],
                     -s * r[0] + c * r[1],
                     r[2]])


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    """WGS-84 geodetic coordinates to ECEF position [km].

    Standard ellipsoidal formulae (Vallado 2013 Alg. 51; WGS-84 constants from
    NIMA TR8350.2).  lat in [-90, 90] deg (geodetic), lon in deg east, alt in
    km above the ellipsoid.
    """
    if not -90.0 <= lat_deg <= 90.0:
        raise ValueError(f"latitude must be in [-90, 90] deg, got {lat_deg}")
    if not -540.0 <= lon_deg <= 540.0:
        raise ValueError(f"longitude must be in [-540, 540] deg, got {lon_deg}")
    if alt_km < -0.5:
        raise ValueError(f"altitude must be >= -0.5 km, got {alt_km}")
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    n = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (n + alt_km) * np.cos(lat) * np.cos(lon)
    y = (n + alt_km) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt_km) * sin_lat
    return np.array([x, y, z])


def ecef_to_azel(r_sat_ecef_km: np.ndarray,
                 lat_deg: float, lon_deg: float, alt_km: float,
                 ) -> tuple[float, float, float]:
    """Topocentric azimuth/elevation/range of a satellite from a ground site.

    Transforms the site->satellite ECEF vector into the SEZ (south-east-zenith)
    frame using the geodetic latitude (Vallado 2013 Alg. 27 'RAZEL')::

        el = asin(rho_Z / |rho|),   az = atan2(rho_E, -rho_S)  (from north, CW)

    Returns (az_deg in [0, 360), el_deg in [-90, 90], range_km > 0).
    Assumptions: no atmospheric refraction (adds <= ~0.5 deg near the horizon,
    Vallado 2013 Sec. 4.1); geodetic latitude used for the local vertical.
    """
    r_sat = np.asarray(r_sat_ecef_km, dtype=float)
    if r_sat.shape != (3,):
        raise ValueError(f"position vector must have shape (3,), got {r_sat.shape}")
    r_site = geodetic_to_ecef(lat_deg, lon_deg, alt_km)
    rho = r_sat - r_site
    rng = float(np.linalg.norm(rho))
    if rng == 0.0:
        raise ValueError("satellite position coincides with the ground station")
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    rho_s = sin_lat * cos_lon * rho[0] + sin_lat * sin_lon * rho[1] - cos_lat * rho[2]
    rho_e = -sin_lon * rho[0] + cos_lon * rho[1]
    rho_z = cos_lat * cos_lon * rho[0] + cos_lat * sin_lon * rho[1] + sin_lat * rho[2]
    el = np.rad2deg(np.arcsin(np.clip(rho_z / rng, -1.0, 1.0)))
    az = np.rad2deg(np.arctan2(rho_e, -rho_s)) % 360.0
    return float(az), float(el), rng
