"""Ground-station state in TEME, so that station motion is never dropped.

Why this module exists
----------------------
``sgp4`` returns satellite position and velocity in the True Equator Mean
Equinox (TEME) frame, which is inertial for this purpose.  A ground station is
naturally described in Earth-fixed coordinates, where its velocity is zero --
and a station velocity of zero is wrong by up to 465 m/s, which at 2.2 GHz is
3.4 kHz of carrier Doppler.  That is roughly 7 % of the whole LEO Doppler
excursion and about four orders of magnitude larger than every relativistic
term this package reports.  Dropping it is the second most common way to get
a Doppler prediction wrong, after the sign, so the station state is built here
with its rotation velocity included and never offered without it.

Equations
---------
Geodetic to Earth-fixed position (site-position algorithm, Vallado,
*Fundamentals of Astrodynamics and Applications*, 4th ed., Ch. 3; WGS-84
constants from NIMA TR8350.2, 3rd ed., 2000)::

    N = a / sqrt(1 - e^2 sin^2(lat))
    x = (N + h) cos(lat) cos(lon)
    y = (N + h) cos(lat) sin(lon)
    z = (N (1 - e^2) + h) sin(lat)

Earth-fixed to TEME is a single rotation about +Z by the Greenwich Mean
Sidereal Time, with GMST from the IAU 1982 polynomial (Aoki et al. 1982,
*Astron. Astrophys.* 105, 359; quoted in this form by Meeus, *Astronomical
Algorithms*, 2nd ed., Eq. 12.4)::

    r_TEME = R3(-theta) r_ECEF
    v_TEME = R3(-theta) r_ECEF x-product term = omega_E_vec x r_TEME

The station velocity in TEME is ``omega_E_vec x r_TEME`` with
``omega_E_vec = (0, 0, OMEGA_EARTH_RAD_S)``, because the station is fixed in
the rotating frame.  Its acceleration is the centripetal term
``omega_E_vec x (omega_E_vec x r_TEME)``, which is included so that Doppler
*rates* from the TLE path are not silently missing it (it reaches 0.034 m/s^2
at the equator, about 0.25 Hz/s at 2.2 GHz -- small but not zero).

Accuracy class and what is dropped
----------------------------------
The GMST-only reduction neglects polar motion (under about 1 arcsec), the
UT1 - UTC difference (bounded by 0.9 s, i.e. at most 0.00375 deg of Earth
rotation, about 0.45 km of station displacement at the equator), and the
TEME-versus-pseudo-Earth-fixed subtleties at the arcsecond level.  This is the
standard scheduling-class reduction (Vallado, 4th ed., Ch. 3) and it is not
suitable for precision orbit determination.

For Doppler specifically: a 0.45 km station position error maps to a
range-rate error of order ``|v_rel| * (0.45 km / rho)`` only in the worst
alignment, but more usefully, a 0.00375 deg rotation error misplaces the
station velocity direction by the same angle, giving a range-rate error up to
about ``465 m/s * 6.5e-5 = 0.03 m/s``, i.e. about 0.22 Hz at 2.2 GHz.  That
bound is computed, not measured, and it is the dominant frame-related error
term in this package -- larger than the relativistic terms it reports.

Units: degrees at the public API for angles, metres for lengths, metres per
second for velocities, seconds for time.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np

from .constants import OMEGA_EARTH_RAD_S, WGS84_A_M, WGS84_E2
from .geometry import State

OMEGA_EARTH_VEC_RAD_S = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S])
"""Earth rotation vector in TEME [rad/s]; +Z by construction."""


def julian_date(t: datetime) -> tuple[float, float]:
    """UTC ``datetime`` to Julian date as ``(whole, fraction)``.

    Naive datetimes are interpreted as UTC.  UTC is used in place of UT1; see
    the module docstring for the size of that approximation.  Uses the
    ``jday`` routine shipped with ``sgp4`` (valid 1900-2100).
    """
    from sgp4.functions import jday

    if not isinstance(t, datetime):
        raise TypeError(f"expected datetime, got {type(t).__name__}")
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    t = t.astimezone(UTC)
    return jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)


def gmst_rad(jd: float, fr: float = 0.0) -> float:
    """Greenwich Mean Sidereal Time [rad] for Julian date ``jd + fr`` (UT1 ~ UTC).

    IAU 1982 GMST polynomial (Aoki et al. 1982; given in this form by Meeus,
    *Astronomical Algorithms*, 2nd ed., Eq. 12.4).  Formal accuracy about 0.1
    arcsec, dominated in practice by the UT1 - UTC neglect.
    """
    d = (jd - 2451545.0) + fr
    t_cent = d / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * d
        + 0.000387933 * t_cent**2
        - t_cent**3 / 38710000.0
    )
    return math.radians(gmst_deg % 360.0)


def geodetic_to_ecef_m(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """WGS-84 geodetic coordinates to Earth-fixed position [m], shape (3,).

    ``lat_deg`` geodetic in [-90, 90], ``lon_deg`` east, ``alt_m`` above the
    ellipsoid.  Vallado, 4th ed., Ch. 3 site-position algorithm; WGS-84
    constants from NIMA TR8350.2.  Raises ``ValueError`` on out-of-range input.
    """
    if not -90.0 <= lat_deg <= 90.0:
        raise ValueError(f"lat_deg must be in [-90, 90], got {lat_deg}")
    if not -540.0 <= lon_deg <= 540.0:
        raise ValueError(f"lon_deg must be in [-540, 540], got {lon_deg}")
    if alt_m < -500.0:
        raise ValueError(f"alt_m must be >= -500 m, got {alt_m}")
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    return np.array(
        [
            (n + alt_m) * math.cos(lat) * math.cos(lon),
            (n + alt_m) * math.cos(lat) * math.sin(lon),
            (n * (1.0 - WGS84_E2) + alt_m) * sin_lat,
        ]
    )


def ecef_to_teme_m(r_ecef_m: np.ndarray, jd: float, fr: float = 0.0) -> np.ndarray:
    """Rotate an Earth-fixed position [m] into TEME: ``R3(-theta_GMST) r_ECEF``.

    Vallado, 4th ed., Ch. 3.  See the module docstring for the neglected terms.
    """
    r = np.asarray(r_ecef_m, dtype=float)
    if r.shape != (3,):
        raise ValueError(f"r_ecef_m must have shape (3,), got {r.shape}")
    theta = gmst_rad(jd, fr)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c * r[0] - s * r[1], s * r[0] + c * r[1], r[2]])


def station_state_teme(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    t: datetime,
    label: str = "station",
) -> State:
    """Ground-station :class:`~dopplerkit.geometry.State` in TEME at epoch ``t``.

    Position from :func:`geodetic_to_ecef_m` rotated by GMST; velocity
    ``omega_E x r_TEME`` [m/s]; acceleration ``omega_E x (omega_E x r_TEME)``
    [m/s^2].  The velocity term is never optional -- see the module docstring
    for why, and for its 3.4 kHz Doppler magnitude at 2.2 GHz.

    Units: degrees, metres, UTC ``datetime``; returns SI metres and m/s in
    TEME, matching what ``sgp4`` returns once converted from km.
    """
    jd, fr = julian_date(t)
    r_teme = ecef_to_teme_m(geodetic_to_ecef_m(lat_deg, lon_deg, alt_m), jd, fr)
    v_teme = np.cross(OMEGA_EARTH_VEC_RAD_S, r_teme)
    a_teme = np.cross(OMEGA_EARTH_VEC_RAD_S, v_teme)
    return State(position_m=r_teme, velocity_mps=v_teme, acceleration_mps2=a_teme, label=label)


def satellite_state_teme(satrec: object, t: datetime, label: str = "satellite") -> State:
    """Satellite :class:`~dopplerkit.geometry.State` in TEME from an ``sgp4`` ``Satrec``.

    Calls ``satrec.sgp4(jd, fr)`` and converts km and km/s to m and m/s.
    Acceleration is left at zero, which means Doppler *rates* from this path
    carry only the kinematic term; that is a documented limitation, and the
    closed-form model in :mod:`dopplerkit.analytic` is the reference for
    Doppler rate.

    Raises ``RuntimeError`` with the SGP4 error code if propagation fails
    (code 1-6; see the ``sgp4`` documentation), rather than returning the
    silently invalid vector SGP4 emits in that case.
    """
    jd, fr = julian_date(t)
    err, r_km, v_km_s = satrec.sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(
            f"sgp4 propagation failed with error code {err} at {t.isoformat()}; "
            "see the sgp4 package documentation for the code meanings"
        )
    return State(
        position_m=np.asarray(r_km, dtype=float) * 1.0e3,
        velocity_mps=np.asarray(v_km_s, dtype=float) * 1.0e3,
        acceleration_mps2=np.zeros(3),
        label=label,
    )


def elevation_deg(station: State, satellite: State, lat_deg: float, lon_deg: float,
                  jd: float, fr: float = 0.0) -> float:
    """Topocentric elevation of ``satellite`` from ``station`` [deg].

    Both states are in TEME; the local vertical is rebuilt from the geodetic
    latitude and longitude rotated by GMST, so this is the same reduction as
    the rest of the module.  ``el = atan2(rho_up, |rho_horizontal|)``, the
    two-argument form, which stays well-conditioned at the zenith.  No
    atmospheric refraction (up to roughly half a degree near the horizon;
    Vallado, 4th ed., Ch. 4).
    """
    rho = satellite.position_m - station.position_m
    theta = gmst_rad(jd, fr)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg) + theta
    up = np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
    )
    rho_up = float(rho @ up)
    rho_horiz = float(np.linalg.norm(rho - rho_up * up))
    return math.degrees(math.atan2(rho_up, rho_horiz))


__all__ = [
    "OMEGA_EARTH_VEC_RAD_S",
    "ecef_to_teme_m",
    "elevation_deg",
    "geodetic_to_ecef_m",
    "gmst_rad",
    "julian_date",
    "satellite_state_teme",
    "station_state_teme",
]
