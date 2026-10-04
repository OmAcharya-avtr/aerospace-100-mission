"""Time and coordinate-frame utilities for constellation link planning.

Conventions
-----------
* Lengths in kilometres, angles in degrees at the public API (radians internally).
* Times are Python ``datetime`` objects; naive datetimes are interpreted as UTC.
* Velocities in km/s.

Earth-rotation model (documented simplification)
------------------------------------------------
SGP4 produces positions in the TEME frame.  This module rotates TEME to an
Earth-fixed frame (ECEF) by a single rotation about +Z through the Greenwich
Mean Sidereal Time (GMST), using the IAU 1982 GMST polynomial (Aoki et al.
1982, Astron. Astrophys. 105, 359; the same expression is given by Meeus 1998,
"Astronomical Algorithms", 2nd ed., Eq. 12.4, and discussed in Vallado 2013,
"Fundamentals of Astrodynamics and Applications", 4th ed., Ch. 3).

Neglected terms and their magnitudes:

* polar motion (< ~1 arcsec, sub-metre ground displacement),
* UT1 - UTC (bounded by 0.9 s, i.e. <= 0.00375 deg of Earth rotation,
  ~0.42 km at the equator),
* equation of the equinoxes / TEME-versus-PEF subtleties (arcsecond class).

Accuracy class: the induced timing error on LEO contact rise/set times is well
below one second.  Contact scheduling uses coarse scans of tens of seconds, so
this is the standard "GMST-only" reduction used for scheduling-class work
(Vallado 2013, Ch. 3).  It is NOT suitable for precision pointing or orbit
determination.

Inter-satellite geometry is computed in TEME directly (both endpoints share the
frame, so the Earth-rotation reduction cancels out of the relative vector).
Only the ground leg needs ECEF.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from sgp4.functions import jday

__all__ = [
    "WGS84_A_KM",
    "WGS84_F",
    "WGS84_E2",
    "MU_EARTH_KM3_S2",
    "EARTH_ROTATION_RAD_S",
    "SPEED_OF_LIGHT_KM_S",
    "to_utc",
    "datetime_to_jd",
    "gmst_rad",
    "teme_to_ecef",
    "geodetic_to_ecef",
    "ecef_to_azel",
]

# WGS-84 defining parameters (NIMA TR8350.2, 3rd ed., 2000).
WGS84_A_KM = 6378.137                  # semi-major axis [km]
WGS84_F = 1.0 / 298.257223563          # flattening [-]
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)   # first eccentricity squared [-]

# WGS-84 Earth gravitational parameter [km^3/s^2] (NIMA TR8350.2, 3rd ed.).
MU_EARTH_KM3_S2 = 398600.4418

# WGS-84 Earth rotation rate [rad/s] (NIMA TR8350.2, 3rd ed.): the sidereal
# rate, not 2*pi/86400.
EARTH_ROTATION_RAD_S = 7.292115e-5

# Speed of light in vacuum [km/s] (SI defining constant, 299 792 458 m/s exactly).
SPEED_OF_LIGHT_KM_S = 299792.458


def to_utc(t: datetime) -> datetime:
    """Return ``t`` as a timezone-aware UTC datetime (naive input treated as UTC)."""
    if not isinstance(t, datetime):
        raise TypeError(f"expected datetime, got {type(t).__name__}")
    if t.tzinfo is None:
        return t.replace(tzinfo=UTC)
    return t.astimezone(UTC)


def datetime_to_jd(t: datetime) -> tuple[float, float]:
    """Convert a UTC datetime to a Julian date as ``(whole, fraction)``.

    Uses the ``jday`` routine shipped with ``sgp4`` (valid 1900-2100).  UTC is
    used in place of UT1; see the module docstring for the error budget.
    """
    t = to_utc(t)
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                  t.second + t.microsecond * 1e-6)
    return jd, fr


def gmst_rad(jd: float, fr: float = 0.0) -> float:
    """Greenwich Mean Sidereal Time [rad] for Julian date ``jd + fr`` (UT1 ~ UTC).

    IAU 1982 GMST polynomial (Aoki et al. 1982; stated in this form by
    Meeus 1998, Eq. 12.4)::

        GMST[deg] = 280.46061837 + 360.98564736629 * d
                    + 0.000387933 * T^2 - T^3 / 38 710 000

    with ``d = JD_UT1 - 2451545.0`` and ``T = d / 36525``.  Formal accuracy
    ~0.1 arcsec near J2000; in this package the error is dominated by the
    neglect of UT1 - UTC (module docstring).
    """
    d = (jd - 2451545.0) + fr
    t_cent = d / 36525.0
    gmst_deg = (280.46061837
                + 360.98564736629 * d
                + 0.000387933 * t_cent ** 2
                - t_cent ** 3 / 38710000.0)
    return float(np.deg2rad(gmst_deg % 360.0))


def teme_to_ecef(r_teme_km: np.ndarray, jd: float, fr: float = 0.0) -> np.ndarray:
    """Rotate a TEME position [km] into ECEF via GMST (Vallado 2013, Ch. 3).

    ``r_ECEF = R3(theta_GMST) . r_TEME`` with ``R3`` the standard rotation
    about +Z.  Accepts shape ``(3,)`` or ``(n, 3)`` and returns the same shape.
    """
    r = np.asarray(r_teme_km, dtype=float)
    single = r.ndim == 1
    if single:
        r = r[None, :]
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"position must have shape (3,) or (n, 3), got {np.shape(r_teme_km)}")
    theta = gmst_rad(jd, fr)
    c, s = np.cos(theta), np.sin(theta)
    out = np.empty_like(r)
    out[:, 0] = c * r[:, 0] + s * r[:, 1]
    out[:, 1] = -s * r[:, 0] + c * r[:, 1]
    out[:, 2] = r[:, 2]
    return out[0] if single else out


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    """WGS-84 geodetic coordinates to an ECEF position [km].

    Standard ellipsoidal site-position algorithm (Vallado 2013, Ch. 3) with
    WGS-84 constants from NIMA TR8350.2, 3rd ed.  ``lat_deg`` is geodetic
    latitude in [-90, 90], ``lon_deg`` is longitude east, ``alt_km`` is height
    above the ellipsoid.
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
    n = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    return np.array([
        (n + alt_km) * np.cos(lat) * np.cos(lon),
        (n + alt_km) * np.cos(lat) * np.sin(lon),
        (n * (1.0 - WGS84_E2) + alt_km) * sin_lat,
    ])


def ecef_to_azel(r_sat_ecef_km: np.ndarray,
                 lat_deg: float, lon_deg: float, alt_km: float,
                 ) -> tuple[float, float, float]:
    """Topocentric azimuth, elevation and range of a satellite from a ground site.

    Transforms the site-to-satellite ECEF vector into the SEZ
    (south-east-zenith) frame using geodetic latitude -- the RAZEL algorithm
    (Vallado 2013, Ch. 4)::

        el = atan2(rho_Z, hypot(rho_S, rho_E))
        az = atan2(rho_E, -rho_S)

    with azimuth from north, clockwise.  The two-argument elevation form is
    used rather than ``asin(rho_Z / |rho|)`` because ``asin`` is
    ill-conditioned at the zenith.

    No atmospheric refraction is applied (refraction raises apparent elevation
    by up to roughly half a degree at the horizon; Vallado 2013, Ch. 4).

    Returns ``(az_deg in [0, 360), el_deg in [-90, 90], range_km > 0)``.
    """
    r_sat = np.asarray(r_sat_ecef_km, dtype=float)
    if r_sat.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {r_sat.shape}")
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
    el = np.rad2deg(np.arctan2(rho_z, np.hypot(rho_s, rho_e)))
    az = np.rad2deg(np.arctan2(rho_e, -rho_s)) % 360.0
    return float(az), float(el), rng
