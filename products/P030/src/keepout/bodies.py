"""Celestial body constants, angular radii, and low-precision directions.

Units are SI throughout: metres, seconds, radians. Frames: the ephemeris
routines return vectors in the mean equator and equinox of date (MOD). For
keep-out geometry the difference between MOD, TOD and GCRF is at the
arcsecond-to-arcminute level and is far below the degrees-wide exclusion
angles this package deals with, but it is a real approximation and is listed
in the README limitations.

References
----------
D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
    Microcosm Press (2013): Algorithm 29 (low-precision Sun position),
    Algorithm 31 (low-precision Moon position), Sec. 5.1-5.3.
J. R. Wertz (ed.), *Spacecraft Attitude Determination and Control*, Reidel
    (1978), Sec. 5.2 -- solar and lunar direction models for attitude work.
National Imagery and Mapping Agency, TR8350.2, 3rd ed. (1997, amended 2000) --
    WGS-84 equatorial radius and gravitational parameter.
B. A. Archinal et al., "Report of the IAU Working Group on Cartographic
    Coordinates and Rotational Elements: 2015", *Celest. Mech. Dyn. Astron.*
    130, 22 (2018) -- lunar mean radius.
IAU 2015 Resolution B3 -- nominal solar radius.
IAU 2012 Resolution B2 -- astronomical unit, exact.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import unit

__all__ = [
    "EARTH_RADIUS_M",
    "EARTH_MU",
    "MOON_RADIUS_M",
    "SUN_RADIUS_M",
    "ASTRONOMICAL_UNIT_M",
    "J2000_JD",
    "angular_radius",
    "earth_angular_radius",
    "julian_date",
    "sun_direction_mod",
    "moon_direction_mod",
    "earth_direction_from_position",
]

#: WGS-84 semi-major (equatorial) axis [m], NIMA TR8350.2 3rd ed. -- exact by definition.
EARTH_RADIUS_M = 6378137.0
#: WGS-84 Earth gravitational parameter [m^3 s^-2], NIMA TR8350.2 3rd ed.
EARTH_MU = 3.986004418e14
#: Lunar mean radius [m], IAU/IAG WG report, Archinal et al. (2018), 1737.4 km.
MOON_RADIUS_M = 1737400.0
#: Nominal solar radius [m], IAU 2015 Resolution B3, 6.957e8 m.
SUN_RADIUS_M = 6.957e8
#: Astronomical unit [m], IAU 2012 Resolution B2 -- exact by definition.
ASTRONOMICAL_UNIT_M = 149597870700.0
#: Julian date of the J2000.0 epoch (2000-01-01 12:00:00 TT).
J2000_JD = 2451545.0


def angular_radius(body_radius_m: ArrayLike, distance_m: ArrayLike) -> NDArray[np.float64] | float:
    """Apparent angular radius of a sphere [rad].

    ``alpha = arcsin(R / d)``: the half-angle of the cone from the observer that
    is tangent to a sphere of radius ``R`` whose centre is at distance ``d``.
    The tangent point satisfies ``sin(alpha) = R / d`` because the radius to the
    tangent point is perpendicular to the line of sight (Wertz 1978, Sec. 5.2).

    Parameters
    ----------
    body_radius_m : array_like
        Body radius [m], ``> 0``.
    distance_m : array_like
        Observer-to-centre distance [m], must be ``>= body_radius_m``.

    Returns
    -------
    float or ndarray
        Angular radius [rad], in ``(0, pi/2]``.

    Raises
    ------
    ValueError
        If the radius is non-positive or the observer is inside the body.

    Notes
    -----
    Validity: rigid sphere, geometric (no refraction, no oblateness, no light
    bending). For the Earth the equatorial radius is used, so the value is an
    upper bound on the true oblate angular radius; see README limitations.
    """
    r = np.asarray(body_radius_m, dtype=float)
    d = np.asarray(distance_m, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("body_radius_m must be > 0")
    if np.any(d < r):
        raise ValueError("distance_m must be >= body_radius_m (observer is inside the body)")
    out = np.arcsin(np.clip(r / d, -1.0, 1.0))
    return float(out) if np.ndim(out) == 0 else out


def earth_angular_radius(
    altitude_m: ArrayLike, radius_m: float = EARTH_RADIUS_M
) -> NDArray[np.float64] | float:
    """Angular radius of the Earth seen from altitude ``h`` [rad].

    ``alpha = arcsin(R_E / (R_E + h))`` -- :func:`angular_radius` with
    ``d = R_E + h``. Limits: ``h -> 0`` gives ``pi/2`` (the horizon is a great
    circle), ``h -> infinity`` gives ``0``.

    Parameters
    ----------
    altitude_m : array_like
        Altitude above the reference sphere [m], ``>= 0``.
    radius_m : float, optional
        Reference body radius [m]. Defaults to the WGS-84 equatorial radius.

    Returns
    -------
    float or ndarray
        Angular radius [rad].
    """
    h = np.asarray(altitude_m, dtype=float)
    if np.any(h < 0.0):
        raise ValueError("altitude_m must be >= 0")
    return angular_radius(radius_m, radius_m + h)


def julian_date(when: _dt.datetime) -> float:
    """Julian date of a timezone-aware or naive UTC ``datetime``.

    Uses the standard Fliegel-Van Flandern civil-calendar conversion as given in
    Vallado (2013), Algorithm 14, valid for dates from 1900 to 2100.

    Parameters
    ----------
    when : datetime.datetime
        UTC instant. A naive datetime is interpreted as UTC.

    Returns
    -------
    float
        Julian date [days].
    """
    if when.tzinfo is not None:
        when = when.astimezone(_dt.UTC).replace(tzinfo=None)
    y, m = when.year, when.month
    if not 1900 <= y <= 2100:
        raise ValueError(f"year {y} outside the 1900-2100 validity range of Algorithm 14")
    if m <= 2:
        y -= 1
        m += 12
    day_frac = (
        when.hour + when.minute / 60.0 + (when.second + when.microsecond * 1e-6) / 3600.0
    ) / 24.0
    a = int(y / 100)
    b = 2 - a + int(a / 4)
    return (
        int(365.25 * (y + 4716))
        + int(30.6001 * (m + 1))
        + when.day
        + b
        - 1524.5
        + day_frac
    )


def _centuries(jd: ArrayLike) -> NDArray[np.float64]:
    return (np.asarray(jd, dtype=float) - J2000_JD) / 36525.0


def sun_direction_mod(jd: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Geocentric Sun direction and distance, low precision.

    Vallado (2013), Algorithm 29. With ``T`` Julian centuries from J2000.0
    (UT1 and TDB are not distinguished at this precision)::

        lambda_M = 280.460 + 36000.771 T                        [deg]
        M        = 357.5291092 + 35999.05034 T                  [deg]
        lambda   = lambda_M + 1.914666471 sin M
                            + 0.019994643 sin 2M                [deg]
        r        = 1.000140612 - 0.016708617 cos M
                              - 0.000139589 cos 2M              [AU]
        eps      = 23.439291 - 0.0130042 T                      [deg]
        r_vec    = r (cos lambda, cos eps sin lambda, sin eps sin lambda)

    The ecliptic latitude of the Sun is taken as zero, which is the defining
    approximation of the algorithm.

    Parameters
    ----------
    jd : array_like
        Julian date(s) [days].

    Returns
    -------
    (direction, distance) : tuple of ndarray
        ``direction`` shape ``(..., 3)``, unit vectors in the mean equator and
        equinox of date; ``distance`` in metres.

    Notes
    -----
    Validity: a truncated two-term series. It is a convenience so the package
    can be exercised end to end; for operational geometry supply directions from
    a real ephemeris (Skyfield, astropy, SPICE). The checks actually run on it
    are in ``validation/validate_ephemeris.py``.
    """
    t = _centuries(jd)
    lam_m = np.radians(280.460 + 36000.771 * t)
    m = np.radians(357.5291092 + 35999.05034 * t)
    lam = lam_m + np.radians(1.914666471) * np.sin(m) + np.radians(0.019994643) * np.sin(2 * m)
    r_au = 1.000140612 - 0.016708617 * np.cos(m) - 0.000139589 * np.cos(2 * m)
    eps = np.radians(23.439291 - 0.0130042 * t)
    vec = np.stack(
        [np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], axis=-1
    )
    return unit(vec), np.asarray(r_au * ASTRONOMICAL_UNIT_M, dtype=float)


def moon_direction_mod(jd: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Geocentric Moon direction and distance, low precision.

    Vallado (2013), Algorithm 31, with ``T`` Julian centuries from J2000.0. The
    ecliptic longitude and latitude series and the horizontal parallax series
    are reproduced as printed; the geocentric distance follows from the parallax
    as ``r = R_E / sin(P)``.

    Parameters
    ----------
    jd : array_like
        Julian date(s) [days].

    Returns
    -------
    (direction, distance) : tuple of ndarray
        ``direction`` shape ``(..., 3)``, unit vectors in the mean equator and
        equinox of date; ``distance`` in metres.

    Notes
    -----
    Validity: a short truncated series. Checked in
    ``validation/validate_ephemeris.py`` against published lunar orbital
    constants (mean distance, sidereal period, inclination of the lunar orbit
    to the ecliptic), not against a numerical ephemeris.
    """
    t = _centuries(jd)
    d2r = np.pi / 180.0
    lam = d2r * (
        218.32
        + 481267.8813 * t
        + 6.29 * np.sin(d2r * (134.9 + 477198.85 * t))
        - 1.27 * np.sin(d2r * (259.2 - 413335.38 * t))
        + 0.66 * np.sin(d2r * (235.7 + 890534.23 * t))
        + 0.21 * np.sin(d2r * (269.9 + 954397.70 * t))
        - 0.19 * np.sin(d2r * (357.5 + 35999.05 * t))
        - 0.11 * np.sin(d2r * (186.6 + 966404.05 * t))
    )
    phi = d2r * (
        5.13 * np.sin(d2r * (93.3 + 483202.03 * t))
        + 0.28 * np.sin(d2r * (228.2 + 960400.87 * t))
        - 0.28 * np.sin(d2r * (318.3 + 6003.18 * t))
        - 0.17 * np.sin(d2r * (217.6 - 407332.20 * t))
    )
    parallax = d2r * (
        0.9508
        + 0.0518 * np.cos(d2r * (134.9 + 477198.85 * t))
        + 0.0095 * np.cos(d2r * (259.2 - 413335.38 * t))
        + 0.0078 * np.cos(d2r * (235.7 + 890534.23 * t))
        + 0.0028 * np.cos(d2r * (269.9 + 954397.70 * t))
    )
    eps = d2r * (23.439291 - 0.0130042 * t - 1.64e-7 * t**2 + 5.04e-7 * t**3)
    cphi, sphi = np.cos(phi), np.sin(phi)
    vec = np.stack(
        [
            cphi * np.cos(lam),
            np.cos(eps) * cphi * np.sin(lam) - np.sin(eps) * sphi,
            np.sin(eps) * cphi * np.sin(lam) + np.cos(eps) * sphi,
        ],
        axis=-1,
    )
    distance = EARTH_RADIUS_M / np.sin(parallax)
    return unit(vec), np.asarray(distance, dtype=float)


def earth_direction_from_position(
    position_m: ArrayLike, radius_m: float = EARTH_RADIUS_M
) -> tuple[NDArray[np.float64], NDArray[np.float64] | float]:
    """Direction to the Earth's centre and its angular radius, from orbit.

    For a spacecraft at inertial position ``r`` the Earth's centre lies along
    ``-r_hat`` and subtends ``arcsin(R_E / |r|)``.

    Parameters
    ----------
    position_m : array_like
        Geocentric inertial position [m], shape ``(3,)`` or ``(..., 3)``.
    radius_m : float, optional
        Reference Earth radius [m].

    Returns
    -------
    (direction, angular_radius) : tuple
        Unit vector(s) towards the Earth's centre and the angular radius [rad].
    """
    r = np.asarray(position_m, dtype=float)
    d = np.linalg.norm(r, axis=-1)
    return -unit(r), angular_radius(radius_m, d)
