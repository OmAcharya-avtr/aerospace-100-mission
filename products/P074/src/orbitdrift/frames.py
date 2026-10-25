"""Coordinate transforms and the radial/in-track/cross-track decomposition.

Everything here is implemented in this module. ``astropy`` is not installed in
the environment these numbers were measured in, so no transform in this
package is cross-checked against it and no claim of such a cross-check is
made anywhere in this repository. The checks that are made are stated in
``validation/validate_frames.py``: rotation matrices are verified orthogonal
to a declared tolerance, round trips are verified to close, Greenwich mean
sidereal time is verified against an independently derived value at the J2000
epoch, and the in-track axis is verified to be the one the along-track growth
actually appears on.

Frames
------
``TEME``
    True equator, mean equinox of date. This is what SGP4 returns and what the
    numerical propagator in :mod:`orbitdrift.forces` integrates in. All
    distances km, velocities km/s.
``ECEF``
    Earth-centred, Earth-fixed, obtained from TEME by a rotation about the z
    axis through Greenwich mean sidereal time. This is the conventional
    SGP4-practice approximation: it omits polar motion and the equation of the
    equinoxes, which together displace a LEO ground point by of order a
    hundred metres. Used here only to get a geodetic altitude for the
    atmosphere model, where that is far below the model's own error.
``RIC``
    Radial, in-track, cross-track, built from a reference state. The radial
    axis is along the position vector, cross-track along the orbit normal, and
    in-track completes the right-handed set. In-track is not exactly the
    velocity direction for an eccentric orbit, and the docstring of
    :func:`ric_basis` says what it is instead.
"""

from __future__ import annotations

import numpy as np

from .constants import OMEGA_EARTH_RAD_S, WGS84

_JD_J2000 = 2451545.0
_JULIAN_CENTURY_DAYS = 36525.0


def gmst_rad(jd_ut1: np.ndarray | float) -> np.ndarray:
    """Greenwich mean sidereal time, radians in ``[0, 2*pi)``.

    IAU 1982 expression, Vallado 2013 eq. (3-47):

        theta_GMST = 67310.54841
                     + (876600 h + 8640184.812866) T
                     + 0.093104 T^2
                     - 6.2e-6 T^3      seconds of time,

    with ``T`` in Julian centuries of UT1 from J2000.0. One second of time is
    1/240 degree. UT1 and UTC are not distinguished here; the difference is
    under 0.9 s by definition of leap seconds, which is 0.004 degree of Earth
    rotation and is far below the atmosphere model error this feeds.

    Parameters
    ----------
    jd_ut1
        Julian date, UT1.
    """
    t = (np.asarray(jd_ut1, dtype=float) - _JD_J2000) / _JULIAN_CENTURY_DAYS
    seconds = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t * t * t
    )
    degrees = np.mod(seconds / 240.0, 360.0)
    return np.deg2rad(degrees)


def rotation_z(angle_rad: float) -> np.ndarray:
    """Right-handed rotation of the *frame* about z by ``angle_rad``.

    Returns the matrix ``R`` such that ``R @ v_inertial`` gives the components
    of ``v`` in a frame rotated by ``angle_rad`` about z.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def teme_to_ecef(r_teme_km: np.ndarray, jd_ut1: np.ndarray | float) -> np.ndarray:
    """Rotate TEME position(s) to ECEF through GMST. Shapes ``(..., 3)``."""
    r = np.asarray(r_teme_km, dtype=float)
    theta = np.atleast_1d(gmst_rad(jd_ut1))
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    out = np.empty_like(r)
    out[..., 0] = c * x + s * y
    out[..., 1] = -s * x + c * y
    out[..., 2] = z
    return out


def geodetic_altitude_km(r_ecef_km: np.ndarray) -> np.ndarray:
    """Height above the WGS84 ellipsoid, km, by Bowring's direct formula.

    Bowring 1985, "The accuracy of geodetic latitude and height equations",
    *Survey Review* 28(218), as given in Vallado 2013 section 3.2.2. The
    closed form needs no iteration, which matters because this is evaluated
    once per integrator stage per satellite.

    It is a single-pass formula built for points near the surface, and its
    departure from a converged fixed-point solution of the same geometry grows
    with height. Measured over a latitude sweep in
    ``validation/validate_frames.py``: 0.3 mm at 200 km, 1.2 mm at 400 km,
    8.8 mm at 1200 km, 15.5 mm at 1700 km. The only consumer is the
    atmosphere model, whose own density is uncertain by a factor of several at
    these altitudes, so the approximation is declared rather than iterated
    away.
    """
    r = np.asarray(r_ecef_km, dtype=float)
    a = WGS84.radius_km
    f = WGS84.flattening
    assert f is not None  # WGS84 declares a flattening; WGS72 here would be a defect
    e2 = f * (2.0 - f)
    b = a * (1.0 - f)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    p = np.hypot(x, y)
    # Bowring's auxiliary ("reduced") latitude, then the closed-form geodetic
    # latitude. arctan2 handles p == 0 without a branch.
    u = np.arctan2(z * a, p * b)
    sin_u, cos_u = np.sin(u), np.cos(u)
    num = z * (1.0 - f) + a * e2 * sin_u**3
    den = (1.0 - f) * (p - a * e2 * cos_u**3)
    lat = np.arctan2(num, den)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    n_curv = a / np.sqrt(1.0 - e2 * sin_lat**2)
    # h = p/cos(lat) - N is singular at the pole and h = z/sin(lat) - N(1-e2)
    # is singular at the equator. Pick whichever denominator is larger, so
    # neither singularity is ever approached; the two agree identically in
    # exact arithmetic.
    near_pole = np.abs(sin_lat) > np.abs(cos_lat)
    safe_cos = np.where(near_pole, 1.0, cos_lat)
    safe_sin = np.where(near_pole, sin_lat, 1.0)
    height = np.where(
        near_pole,
        z / safe_sin - n_curv * (1.0 - e2),
        p / safe_cos - n_curv,
    )
    return height


def ric_basis(r_ref_km: np.ndarray, v_ref_km_s: np.ndarray) -> np.ndarray:
    """Radial / in-track / cross-track unit vectors of a reference state.

    Returns an array of shape ``(..., 3, 3)`` whose rows are the radial,
    in-track and cross-track unit vectors expressed in the input frame, so
    that ``basis @ d`` gives the RIC components of a TEME difference vector
    ``d``.

    The axes are: radial ``= r/|r|``; cross-track ``= (r x v)/|r x v|``, the
    orbit normal; in-track ``= cross x radial``, which completes a
    right-handed set. In-track is **not** the velocity direction unless the
    orbit is circular -- for an eccentric orbit the velocity has a radial
    component and the two differ by the flight path angle. The along-track
    secular growth this package measures appears on the in-track axis as
    defined here, and ``validation/validate_frames.py`` checks that it does.
    """
    r = np.asarray(r_ref_km, dtype=float)
    v = np.asarray(v_ref_km_s, dtype=float)
    radial = r / np.linalg.norm(r, axis=-1, keepdims=True)
    normal = np.cross(r, v)
    cross = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    in_track = np.cross(cross, radial)
    return np.stack([radial, in_track, cross], axis=-2)


def to_ric(
    delta_km: np.ndarray, r_ref_km: np.ndarray, v_ref_km_s: np.ndarray
) -> np.ndarray:
    """Project TEME difference vector(s) onto the reference state's RIC axes."""
    basis = ric_basis(r_ref_km, v_ref_km_s)
    d = np.asarray(delta_km, dtype=float)
    return np.einsum("...ij,...j->...i", basis, d)


def sun_position_teme_km(jd_tt: np.ndarray | float) -> np.ndarray:
    """Geocentric Sun position, km, in the equator-and-equinox of date.

    Vallado 2013, Algorithm 29 ("Sun"), the low-precision series. Stated
    accuracy 0.01 degree in longitude over 1950-2050, which at one
    astronomical unit is about 26000 km of transverse position and therefore
    about 1 part in 6000 of the direction to the Sun. The resulting error in
    the solar third-body acceleration on a LEO satellite is of order 1e-13
    km/s^2 against a two-body term of 8e-3 km/s^2, which is below the
    integrator's own truncation floor measured in
    ``validation/validate_step_refinement.py``.

    The result is treated as TEME. The difference between the mean equinox of
    date and the true equinox of date is the equation of the equinoxes, at
    most about 1.1 arcsecond, which is 5e-6 radian of direction and is
    negligible for a perturbing term at this amplitude.
    """
    t = (np.asarray(jd_tt, dtype=float) - _JD_J2000) / _JULIAN_CENTURY_DAYS
    lambda_mean = np.deg2rad(np.mod(280.460 + 36000.771 * t, 360.0))
    m = np.deg2rad(np.mod(357.5291092 + 35999.05034 * t, 360.0))
    lambda_ecl = lambda_mean + np.deg2rad(
        1.914666471 * np.sin(m) + 0.019994643 * np.sin(2.0 * m)
    )
    r_au = 1.000140612 - 0.016708617 * np.cos(m) - 0.000139589 * np.cos(2.0 * m)
    eps = np.deg2rad(23.439291 - 0.0130042 * t)
    r_km = r_au * 149597870.7
    return np.stack(
        [
            r_km * np.cos(lambda_ecl),
            r_km * np.cos(eps) * np.sin(lambda_ecl),
            r_km * np.sin(eps) * np.sin(lambda_ecl),
        ],
        axis=-1,
    )


def moon_position_teme_km(jd_tt: np.ndarray | float) -> np.ndarray:
    """Geocentric Moon position, km, in the equator-and-equinox of date.

    Vallado 2013, Algorithm 31 ("Moon"), the low-precision series: six terms
    in ecliptic longitude, four in latitude and four in horizontal parallax.
    Stated accuracy is about 0.3 degree in longitude, which is about 2000 km
    of transverse position at the lunar distance, or 1 part in 190 of the
    direction. The lunar third-body acceleration on a LEO satellite is of
    order 1e-9 km/s^2, so a 0.5 per cent direction error contributes about
    5e-12 km/s^2. That is declared rather than hidden: it is the largest
    single modelling approximation in the third-body term and it is still two
    orders below the integrator truncation floor.
    """
    t = (np.asarray(jd_tt, dtype=float) - _JD_J2000) / _JULIAN_CENTURY_DAYS
    d = np.deg2rad

    lam = d(
        218.32
        + 481267.8813 * t
        + 6.29 * np.sin(d(134.9 + 477198.85 * t))
        - 1.27 * np.sin(d(259.2 - 413335.38 * t))
        + 0.66 * np.sin(d(235.7 + 890534.23 * t))
        + 0.21 * np.sin(d(269.9 + 954397.70 * t))
        - 0.19 * np.sin(d(357.5 + 35999.05 * t))
        - 0.11 * np.sin(d(186.6 + 966404.05 * t))
    )
    phi = d(
        5.13 * np.sin(d(93.3 + 483202.03 * t))
        + 0.28 * np.sin(d(228.2 + 960400.87 * t))
        - 0.28 * np.sin(d(318.3 + 6003.18 * t))
        - 0.17 * np.sin(d(217.6 - 407332.20 * t))
    )
    parallax = d(
        0.9508
        + 0.0518 * np.cos(d(134.9 + 477198.85 * t))
        + 0.0095 * np.cos(d(259.2 - 413335.38 * t))
        + 0.0078 * np.cos(d(235.7 + 890534.23 * t))
        + 0.0028 * np.cos(d(269.9 + 954397.70 * t))
    )
    eps = d(23.439291 - 0.0130042 * t - 1.64e-7 * t**2 + 5.04e-7 * t**3)
    r_km = WGS84.radius_km / np.sin(parallax)

    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    cos_eps, sin_eps = np.cos(eps), np.sin(eps)
    return np.stack(
        [
            r_km * cos_phi * np.cos(lam),
            r_km * (cos_eps * cos_phi * np.sin(lam) - sin_eps * sin_phi),
            r_km * (sin_eps * cos_phi * np.sin(lam) + cos_eps * sin_phi),
        ],
        axis=-1,
    )


def earth_rotation_vector_rad_s() -> np.ndarray:
    """Earth angular velocity in TEME, rad/s. Polar motion is not modelled."""
    return np.array([0.0, 0.0, OMEGA_EARTH_RAD_S])
