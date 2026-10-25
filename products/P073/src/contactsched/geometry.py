"""Earth geometry, sidereal time and topocentric look angles.

Everything in this module is in kilometres, seconds and radians unless a name
ends in ``_deg``. Angles returned by public functions that describe look angles
are in degrees, because that is what a station mask is written in.

Reference frames
----------------
``ECEF`` here means the Earth-fixed frame obtained by rotating TEME about the
z-axis by the Greenwich Mean Sidereal Angle. This is the pseudo-Earth-fixed
(PEF) frame of Vallado (2013) §3.7 with polar motion omitted. The omission is
deliberate and bounded: polar motion displaces the pole by under 0.5 arcsecond,
which moves a station by under 15 m and perturbs an elevation angle by under
0.0002 deg at the slant ranges this package deals with. The equation of the
equinoxes between TEME and PEF is likewise omitted; it is below 1.2 arcsecond
for the period SGP4 is used over. Neither omission is relevant at the one-second
window resolution this package reports, and both are stated in README.md under
Limitations. Do not use this module where sub-arcsecond pointing matters.

Ellipsoid
---------
WGS-84 (NIMA TR8350.2, third edition, 2000): semi-major axis
a = 6378.137 km, flattening f = 1/298.257223563, so the first eccentricity
squared is e2 = 2f - f**2 = 6.69437999014e-3.

Sidereal time
-------------
``gmst_rad`` implements the IAU-1982 GMST polynomial in the form given by
Vallado (2013) eq. 3-47, which is the same expression SGP4 itself uses. It is an
independent implementation here, written from the polynomial rather than by
calling SGP4, and ``validation/validate_geometry.py`` measures it against
``sgp4.propagation.gstime`` over a 60-year span so that the agreement is a
measured number rather than an assumption.

References
----------
Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*, 4th
edition, Microcosm Press. Sections 3.2 (geodetic/geocentric), 3.7 (GMST and
Earth-fixed frames) and 4.4 (topocentric SEZ look angles).

NIMA (2000). *Department of Defense World Geodetic System 1984*, Technical
Report TR8350.2, third edition, amendment 1.
"""

from __future__ import annotations

import math

import numpy as np

WGS84_A_KM = 6378.137
"""WGS-84 semi-major axis, km."""

WGS84_F = 1.0 / 298.257223563
"""WGS-84 flattening, dimensionless."""

WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F
"""WGS-84 first eccentricity squared, dimensionless."""

WGS84_B_KM = WGS84_A_KM * (1.0 - WGS84_F)
"""WGS-84 semi-minor axis, km."""

EARTH_ROTATION_RAD_S = 7.292115146706979e-5
"""Earth rotation rate, rad/s (WGS-84 nominal mean angular velocity)."""

MU_EARTH_KM3_S2 = 398600.5
"""Earth gravitational parameter, km**3/s**2 (WGS-72 value, as used by SGP4)."""

MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG = 0.19242430115956036
"""Largest angle between the WGS-84 ellipsoid normal and the position vector.

The deviation is ``phi_geodetic - phi_geocentric``, maximised at geodetic
latitude 45.0962 deg. Setting ``d(phi - psi)/d phi = 0`` with
``tan psi = (1 - e2) tan phi`` gives ``tan phi = 1 / sqrt(1 - e2)``, whence
``tan psi = sqrt(1 - e2)``, their product is exactly 1, and
``tan(phi - psi) = e2 / (2 sqrt(1 - e2))``. Evaluated for WGS-84 this is
0.19242430115956036 deg, which agrees with a brute-force scan of latitude at
1e-4 deg resolution to 2.0e-14 deg. A literal 0.19242218 stood here through the
first run of this module and was falsified by
``tests/test_geometry.py::TestConstants::test_geodetic_geocentric_deviation_matches_closed_form``;
the bound test that uses it was failing by 1.0e-6 deg at the same time. It is
used in
``validation/validate_geometry.py`` as an independently derived bound on the
difference between the geodetic and geocentric elevation of the same slant
vector, which is a different class of check from comparing two implementations
of the same expression.
"""

JD_J2000 = 2451545.0
"""Julian date of the J2000.0 epoch."""


def max_geodetic_geocentric_deviation_deg() -> float:
    """Closed form of :data:`MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG`, degrees.

    Derived rather than tabulated so the constant can be checked against it.
    The maximum of ``phi - atan((1 - e2) tan(phi))`` over latitude is
    ``atan(e2 / (2 sqrt(1 - e2)))``.
    """
    return math.degrees(math.atan(WGS84_E2 / (2.0 * math.sqrt(1.0 - WGS84_E2))))


def gmst_rad(jd_ut1: float) -> float:
    """Greenwich Mean Sidereal Angle at ``jd_ut1``, radians in ``[0, 2 pi)``.

    Parameters
    ----------
    jd_ut1 : float
        Julian date in UT1. UTC is accepted in practice; the difference is under
        0.9 s, which rotates the Earth by under 0.004 deg. That error is carried
        into every look angle computed by this package and is stated in
        README.md under Limitations.

    Notes
    -----
    IAU-1982 GMST, Vallado (2013) eq. 3-47. The polynomial is in seconds of
    sidereal time; it is reduced modulo 86400 before conversion to radians.
    """
    t = (float(jd_ut1) - JD_J2000) / 36525.0
    seconds = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t * t * t
    )
    seconds = math.fmod(seconds, 86400.0)
    if seconds < 0.0:
        seconds += 86400.0
    return math.radians(seconds / 240.0)


def gmst_rad_array(jd_ut1: np.ndarray) -> np.ndarray:
    """Vectorised :func:`gmst_rad`. Input and output are 1-D arrays."""
    t = (np.asarray(jd_ut1, dtype=float) - JD_J2000) / 36525.0
    seconds = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t * t * t
    )
    seconds = np.mod(seconds, 86400.0)
    return np.radians(seconds / 240.0)


def geodetic_to_ecef_km(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    """WGS-84 geodetic position to Earth-fixed Cartesian, km.

    Parameters
    ----------
    lat_deg : float
        Geodetic latitude, degrees, in ``[-90, 90]``.
    lon_deg : float
        Longitude east of Greenwich, degrees. Any real value is accepted and
        reduced modulo 360.
    alt_km : float
        Height above the WGS-84 ellipsoid, km. Must be greater than
        ``-WGS84_B_KM`` so the point is not inside the Earth's centre.

    Returns
    -------
    numpy.ndarray
        Shape ``(3,)``, km.

    Raises
    ------
    ValueError
        If ``lat_deg`` is outside ``[-90, 90]`` or ``alt_km`` is below the
        Earth's centre.
    """
    lat_deg = float(lat_deg)
    alt_km = float(alt_km)
    if not -90.0 <= lat_deg <= 90.0:
        raise ValueError(f"geodetic latitude must be within [-90, 90] deg, got {lat_deg}")
    if alt_km <= -WGS84_B_KM:
        raise ValueError(f"altitude {alt_km} km is at or below the Earth's centre")
    if not math.isfinite(float(lon_deg)):
        raise ValueError(f"longitude must be finite, got {lon_deg}")
    phi = math.radians(lat_deg)
    lam = math.radians(float(lon_deg))
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    curvature = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)
    return np.array(
        [
            (curvature + alt_km) * cos_phi * math.cos(lam),
            (curvature + alt_km) * cos_phi * math.sin(lam),
            (curvature * (1.0 - WGS84_E2) + alt_km) * sin_phi,
        ]
    )


def sez_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rotation from Earth-fixed to topocentric south-east-zenith, dimensionless.

    Rows are the south, east and zenith unit vectors expressed in the
    Earth-fixed frame. Zenith is the outward WGS-84 ellipsoid normal, which is
    what a levelled antenna mount is aligned to, not the position vector.
    Vallado (2013) eq. 4-13.
    """
    phi = math.radians(float(lat_deg))
    lam = math.radians(float(lon_deg))
    sp, cp = math.sin(phi), math.cos(phi)
    sl, cl = math.sin(lam), math.cos(lam)
    return np.array(
        [
            [sp * cl, sp * sl, -cp],
            [-sl, cl, 0.0],
            [cp * cl, cp * sl, sp],
        ]
    )


def teme_to_ecef_km(r_teme_km: np.ndarray, jd_ut1: np.ndarray) -> np.ndarray:
    """Rotate TEME positions into the Earth-fixed frame, km.

    Parameters
    ----------
    r_teme_km : numpy.ndarray
        Shape ``(n, 3)`` or ``(3,)``, km, in the true-equator-mean-equinox frame
        SGP4 produces.
    jd_ut1 : numpy.ndarray
        Shape ``(n,)`` or scalar, Julian date.

    Returns
    -------
    numpy.ndarray
        Same shape as ``r_teme_km``, km.
    """
    r = np.atleast_2d(np.asarray(r_teme_km, dtype=float))
    jd = np.atleast_1d(np.asarray(jd_ut1, dtype=float))
    if r.shape[-1] != 3:
        raise ValueError(f"positions must have last dimension 3, got shape {r.shape}")
    if r.shape[0] != jd.shape[0]:
        raise ValueError(f"got {r.shape[0]} positions and {jd.shape[0]} epochs")
    theta = gmst_rad_array(jd)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    out = np.empty_like(r)
    out[:, 0] = cos_t * r[:, 0] + sin_t * r[:, 1]
    out[:, 1] = -sin_t * r[:, 0] + cos_t * r[:, 1]
    out[:, 2] = r[:, 2]
    if np.ndim(r_teme_km) == 1:
        return out[0]
    return out


def look_angles_deg(
    r_sat_ecef_km: np.ndarray,
    r_station_ecef_km: np.ndarray,
    sez: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Topocentric azimuth, elevation and slant range.

    Parameters
    ----------
    r_sat_ecef_km : numpy.ndarray
        Shape ``(n, 3)``, km, Earth-fixed.
    r_station_ecef_km : numpy.ndarray
        Shape ``(3,)``, km, Earth-fixed.
    sez : numpy.ndarray
        Shape ``(3, 3)``, from :func:`sez_matrix` for the same station.

    Returns
    -------
    tuple of numpy.ndarray
        ``(azimuth_deg, elevation_deg, range_km)``, each shape ``(n,)``.
        Azimuth is measured clockwise from north in ``[0, 360)``. Elevation is
        with respect to the local ellipsoid-normal horizontal and is in
        ``[-90, 90]``.
    """
    r = np.atleast_2d(np.asarray(r_sat_ecef_km, dtype=float))
    rho = r - np.asarray(r_station_ecef_km, dtype=float)[None, :]
    local = rho @ np.asarray(sez, dtype=float).T
    rng = np.linalg.norm(rho, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        elevation = np.degrees(np.arcsin(np.clip(local[:, 2] / rng, -1.0, 1.0)))
    azimuth = np.degrees(np.arctan2(local[:, 1], -local[:, 0])) % 360.0
    # The modulo alone does not close the interval. For an azimuth a hair below
    # due south the arctangent returns a tiny negative angle, and in double
    # precision ``-1e-64 % 360.0`` evaluates to exactly 360.0, which is outside
    # the documented half-open range. Hypothesis found this at
    # dy = -2.6e-66 in tests/test_properties.py; the literal case is pinned in
    # tests/test_regressions.py.
    azimuth = np.where(azimuth >= 360.0, 0.0, azimuth)
    return azimuth, elevation, rng


def elevation_geocentric_vector_deg(
    r_sat_ecef_km: np.ndarray, r_station_ecef_km: np.ndarray
) -> np.ndarray:
    """Elevation above the plane normal to the station *position vector*, degrees.

    This is the geocentric elevation. It is not the quantity an antenna mount
    measures; it exists so that the geodetic elevation can be checked against a
    quantity derived from the vectors in a different way. Computed as
    ``arcsin((rho . r_station_hat) / |rho|)``.
    """
    r = np.atleast_2d(np.asarray(r_sat_ecef_km, dtype=float))
    station = np.asarray(r_station_ecef_km, dtype=float)
    rho = r - station[None, :]
    rng = np.linalg.norm(rho, axis=1)
    unit = station / np.linalg.norm(station)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.degrees(np.arcsin(np.clip((rho @ unit) / rng, -1.0, 1.0)))


def elevation_geocentric_norms_deg(
    r_sat_norm_km: np.ndarray, r_station_norm_km: float, range_km: np.ndarray
) -> np.ndarray:
    """Geocentric elevation from three scalar lengths only, degrees.

    The triangle formed by the Earth's centre, the station and the spacecraft
    has sides ``r_station``, ``range`` and ``r_sat``; the angle at the station
    between the two known sides is ``90 deg + elevation``. The law of cosines
    therefore gives

        ``sin(elevation) = (r_sat**2 - r_station**2 - range**2)
                            / (2 * r_station * range)``

    exactly, in exact arithmetic. This uses only the three lengths and no
    component of any vector, so it is an independent derivation of the same
    angle rather than a second implementation of the same expression. It is
    also badly conditioned near the horizon, where the numerator is a
    difference of nearly equal large numbers: that is the point. The
    disagreement between this and :func:`elevation_geocentric_vector_deg`
    measures the cancellation, and the measurement is in
    ``validation/validate_geometry.py``.
    """
    r_sat = np.asarray(r_sat_norm_km, dtype=float)
    rng = np.asarray(range_km, dtype=float)
    r_st = float(r_station_norm_km)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (r_sat * r_sat - r_st * r_st - rng * rng) / (2.0 * r_st * rng)
    return np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0)))


def max_pass_duration_s(
    semi_major_axis_km: float,
    eccentricity: float,
    station_radius_km: float,
    min_elevation_deg: float,
) -> float:
    """Independently derived upper bound on one pass duration, seconds.

    The bound is deliberately loose and is derived from closed-form spherical
    geometry rather than from any propagator, so it can falsify a window
    computation that has gone wrong without sharing any expression with it.

    Derivation. Treat the Earth as a sphere of radius ``station_radius_km`` and
    the orbit as a circle of radius ``r = a (1 + e)``, the largest radius the
    orbit reaches, which maximises the visible arc. The spacecraft is above
    elevation ``el`` as seen from the station when the geocentric central angle
    between them satisfies ``gamma <= arccos((R / r) cos el) - el``. That
    follows from the sine rule in the triangle with vertices at the Earth's
    centre, the station and the spacecraft: the interior angle at the station is
    ``90 deg + el``, so the angle at the spacecraft is ``90 deg - el - gamma``,
    and ``R / cos(el + gamma) = r / cos(el)``. The
    spacecraft sweeps central angle at mean rate ``n = sqrt(mu / a**3)``; the
    station sweeps with the Earth at ``omega_E``. In the worst case for the
    bound the two rotations are aligned, so the relative rate is at least
    ``n - omega_E``. Hence ``duration <= 2 gamma_max / (n - omega_E)``.

    Assumptions and validity. Spherical Earth at the station's own geocentric
    radius; circular orbit at apogee radius; mean rather than instantaneous
    angular rate. For a near-circular LEO this over-estimates a real pass by
    tens of per cent, which is acceptable for a bound. It is not valid for
    ``n <= omega_E`` -- above geosynchronous -- and raises there. Just *below*
    geosynchronous the guard does not fire and the bound is correct but vacuous:
    at ``a = 42164`` km the relative rate is 4.5e-10 rad/s and the bound is of
    order 1e9 s. Do not read a returned number as informative without checking
    that it is smaller than the horizon.

    Raises
    ------
    ValueError
        If the orbit is not bound, the eccentricity is outside ``[0, 1)``, the
        orbit does not reach above the station, or the mean motion does not
        exceed the Earth rotation rate.
    """
    a = float(semi_major_axis_km)
    e = float(eccentricity)
    radius = float(station_radius_km)
    el = math.radians(float(min_elevation_deg))
    if a <= 0.0:
        raise ValueError(f"semi-major axis must be positive, got {a} km")
    if not 0.0 <= e < 1.0:
        raise ValueError(f"eccentricity must be within [0, 1), got {e}")
    if radius <= 0.0:
        raise ValueError(f"station radius must be positive, got {radius} km")
    if not -90.0 < float(min_elevation_deg) < 90.0:
        raise ValueError(f"minimum elevation must be within (-90, 90) deg, got {min_elevation_deg}")
    r_apogee = a * (1.0 + e)
    ratio = (radius / r_apogee) * math.cos(el)
    if ratio >= 1.0:
        raise ValueError(
            f"orbit apogee radius {r_apogee:.3f} km never rises above "
            f"{min_elevation_deg} deg for a station at radius {radius:.3f} km"
        )
    gamma_max = math.acos(ratio) - el
    if gamma_max <= 0.0:
        raise ValueError("no visible arc above the declared minimum elevation")
    mean_motion = math.sqrt(MU_EARTH_KM3_S2 / (a * a * a))
    relative = mean_motion - EARTH_ROTATION_RAD_S
    if relative <= 0.0:
        raise ValueError(
            "mean motion does not exceed the Earth rotation rate, so this bound "
            "does not apply; it is valid for sub-geosynchronous orbits only"
        )
    return 2.0 * gamma_max / relative
