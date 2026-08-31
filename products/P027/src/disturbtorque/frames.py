"""Orbit geometry, reference frames, Sun direction and eclipse.

Frames used throughout the package
----------------------------------
ECI
    Earth-centred inertial, z along the Earth rotation axis, x toward the mean
    equinox. The Sun model below is referred to the mean equator and equinox *of
    date*; treating that as J2000 introduces a precession error of about
    0.014 deg per year from J2000, negligible for disturbance-torque sizing and
    unacceptable for pointing.
LVLH
    Local-vertical/local-horizontal, defined here as z along nadir (-r_hat),
    y along the negative orbit normal, x = y x z, which for a circular orbit is the
    velocity direction. This is the "orbit frame" of Wertz, *Spacecraft Attitude
    Determination and Control*.
BODY
    Spacecraft body frame, obtained from LVLH by a 3-2-1 (yaw-pitch-roll) sequence of
    fixed pointing offsets. A zero offset means perfect nadir pointing, body z down.

All angles are radians, all lengths metres, all times seconds unless a name says
otherwise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _validate as _v
from .constants import MU_EARTH, R_EARTH_EQUATORIAL

__all__ = [
    "rot_x",
    "rot_y",
    "rot_z",
    "circular_orbit_state",
    "orbit_normal",
    "node_axes",
    "orbital_period",
    "lvlh_from_eci",
    "body_from_lvlh",
    "julian_date",
    "sun_unit_vector_eci",
    "sun_distance_au",
    "sun_direction_for_beta",
    "beta_angle",
    "eclipse_fraction_cylindrical",
    "in_eclipse_cylindrical",
]


def rot_x(angle_rad: float) -> NDArray[np.float64]:
    """Frame rotation about x by ``angle_rad`` [rad]; maps a vector into the rotated frame."""
    a = _v.as_finite_float(angle_rad, "angle_rad")
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def rot_y(angle_rad: float) -> NDArray[np.float64]:
    """Frame rotation about y by ``angle_rad`` [rad]; maps a vector into the rotated frame."""
    a = _v.as_finite_float(angle_rad, "angle_rad")
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def rot_z(angle_rad: float) -> NDArray[np.float64]:
    """Frame rotation about z by ``angle_rad`` [rad]; maps a vector into the rotated frame."""
    a = _v.as_finite_float(angle_rad, "angle_rad")
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def orbital_period(radius_m: float, mu: float = MU_EARTH) -> float:
    """Keplerian period of a circular orbit [s].

    T = 2*pi*sqrt(r^3/mu). Source: two-body problem, any astrodynamics text
    (Vallado, *Fundamentals of Astrodynamics and Applications*).
    Units: ``radius_m`` in m, ``mu`` in m^3 s^-2, return in s.
    Assumptions: point-mass central field, circular orbit, no drag decay.
    Validity: r > Earth radius; J2 changes the nodal period by of order 0.1 % in LEO,
    which is not modelled.
    """
    r = _v.positive(radius_m, "radius_m")
    m = _v.positive(mu, "mu")
    return float(2.0 * np.pi * np.sqrt(r**3 / m))


def node_axes(
    inclination_rad: float, raan_rad: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the orthonormal orbit-plane triad ``(P_hat, Q_hat, h_hat)`` in ECI.

    ``P_hat`` points at the ascending node (argument of latitude u = 0), ``Q_hat`` is
    90 deg along-track from it (u = pi/2), and ``h_hat = P_hat x Q_hat`` is the orbit
    normal. Units: dimensionless unit vectors.
    """
    inc = _v.in_range(inclination_rad, "inclination_rad", -np.pi, np.pi)
    raan = _v.as_finite_float(raan_rad, "raan_rad")
    ci, si = np.cos(inc), np.sin(inc)
    co, so = np.cos(raan), np.sin(raan)
    p_hat = np.array([co, so, 0.0])
    q_hat = np.array([-so * ci, co * ci, si])
    h_hat = np.array([so * si, -co * si, ci])
    return p_hat, q_hat, h_hat


def orbit_normal(inclination_rad: float, raan_rad: float) -> NDArray[np.float64]:
    """Unit orbit-normal vector in ECI (dimensionless)."""
    return node_axes(inclination_rad, raan_rad)[2]


def circular_orbit_state(
    radius_m: float,
    inclination_rad: float,
    raan_rad: float,
    arg_lat_rad: ArrayLike,
    mu: float = MU_EARTH,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Position and inertial velocity on a circular orbit, in ECI.

    r_ECI = R3(-RAAN) R1(-i) [R cos u, R sin u, 0], with u the argument of latitude.
    Source: the standard perifocal-to-ECI rotation sequence (Vallado, *Fundamentals of
    Astrodynamics and Applications*).

    Parameters
    ----------
    radius_m : float
        Orbit radius from the Earth centre [m].
    inclination_rad, raan_rad : float
        Inclination and right ascension of the ascending node [rad].
    arg_lat_rad : array_like
        Argument of latitude u [rad]; scalar or shape (N,).
    mu : float
        Gravitational parameter [m^3 s^-2].

    Returns
    -------
    (r, v) : ndarray
        Position [m] and velocity [m s^-1], shape (3,) for a scalar u or (N, 3).

    Assumptions: circular two-body orbit, no J2 nodal regression during the arc.
    Validity: any circular Earth orbit; the package is exercised only in LEO.
    """
    r_mag = _v.positive(radius_m, "radius_m")
    mu_v = _v.positive(mu, "mu")
    u = np.asarray(arg_lat_rad, dtype=float)
    if not np.all(np.isfinite(u)):
        raise ValueError("arg_lat_rad must be finite")
    p_hat, q_hat, _ = node_axes(inclination_rad, raan_rad)
    cu = np.cos(u)[..., None]
    su = np.sin(u)[..., None]
    r = r_mag * (cu * p_hat + su * q_hat)
    speed = np.sqrt(mu_v / r_mag)
    v = speed * (-su * p_hat + cu * q_hat)
    return r, v


def lvlh_from_eci(r_eci: ArrayLike, v_eci: ArrayLike) -> NDArray[np.float64]:
    """Direction-cosine matrix mapping an ECI vector into the LVLH frame.

    Rows are the LVLH axes expressed in ECI: x along velocity (circular orbit),
    y along the negative orbit normal, z along nadir. Use as ``v_lvlh = C @ v_eci``.
    Units: dimensionless.
    """
    r = _v.as_vector3(r_eci, "r_eci")
    v = _v.as_vector3(v_eci, "v_eci")
    r_norm = float(np.linalg.norm(r))
    if r_norm == 0.0:
        raise ValueError("r_eci must be non-zero")
    z_hat = -r / r_norm
    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    if h_norm == 0.0:
        raise ValueError("r_eci and v_eci are parallel; the orbit normal is undefined")
    y_hat = -h / h_norm
    x_hat = np.cross(y_hat, z_hat)
    return np.vstack([x_hat, y_hat, z_hat])


def body_from_lvlh(yaw_rad: float = 0.0, pitch_rad: float = 0.0, roll_rad: float = 0.0):
    """DCM mapping an LVLH vector into the body frame for a fixed 3-2-1 pointing offset.

    C_bl = R1(roll) R2(pitch) R3(yaw). Zero angles give perfect nadir pointing.
    Units: angles in rad, return dimensionless. Use as ``v_body = C @ v_lvlh``.
    """
    return rot_x(roll_rad) @ rot_y(pitch_rad) @ rot_z(yaw_rad)


def julian_date(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: float = 0.0
) -> float:
    """Julian date from a UTC calendar date, Vallado's algorithm.

    JD = 367 Y - floor(7(Y + floor((M+9)/12))/4) + floor(275 M / 9) + D + 1721013.5
         + (h + min/60 + s/3600)/24

    Source: Vallado, *Fundamentals of Astrodynamics and Applications*.
    Units: return in days. Validity: 1900 March to 2100 February (the algorithm's
    stated range); outside that a ValueError is raised. UTC-UT1 (<0.9 s) is ignored.
    """
    if not (1900 <= int(year) <= 2100):
        raise ValueError(f"year must lie in [1900, 2100] for this algorithm, got {year!r}")
    if not (1 <= int(month) <= 12):
        raise ValueError(f"month must lie in [1, 12], got {month!r}")
    if not (1 <= int(day) <= 31):
        raise ValueError(f"day must lie in [1, 31], got {day!r}")
    y, m, d = int(year), int(month), int(day)
    sec = _v.non_negative(second, "second")
    jd0 = (
        367 * y
        - (7 * (y + (m + 9) // 12)) // 4
        + (275 * m) // 9
        + d
        + 1721013.5
    )
    frac = (int(hour) + int(minute) / 60.0 + sec / 3600.0) / 24.0
    return float(jd0 + frac)


def _sun_mean_anomaly_and_lambda(jd: float) -> tuple[float, float, float]:
    t = (jd - 2451545.0) / 36525.0
    lam_m = np.radians(280.460 + 36000.771 * t)
    m_anom = np.radians(357.5291092 + 35999.05034 * t)
    lam_ecl = lam_m + np.radians(1.914666471 * np.sin(m_anom) + 0.019994643 * np.sin(2 * m_anom))
    return t, m_anom, lam_ecl


def sun_unit_vector_eci(jd: float) -> NDArray[np.float64]:
    """Unit vector from the Earth to the Sun, in ECI (mean equator and equinox of date).

    Low-precision analytic series (Vallado, *Fundamentals of Astrodynamics and
    Applications*, following the *Astronomical Almanac* low-precision formulae):
    mean longitude 280.460 + 36000.771 T, mean anomaly 357.5291092 + 35999.05034 T,
    equation of centre 1.914666471 sin M + 0.019994643 sin 2M, obliquity
    23.439291 - 0.0130042 T, with T in Julian centuries from J2000.

    Units: dimensionless unit vector. Assumptions: geometric direction, no aberration,
    no nutation, no light-time. Validity: roughly 1950-2050 at the few-hundredths of a
    degree level; adequate for solar-radiation-pressure sizing, not for pointing.
    """
    jd_v = _v.as_finite_float(jd, "jd")
    t, _, lam = _sun_mean_anomaly_and_lambda(jd_v)
    eps = np.radians(23.439291 - 0.0130042 * t)
    return np.array(
        [float(np.cos(lam)), float(np.cos(eps) * np.sin(lam)), float(np.sin(eps) * np.sin(lam))]
    )


def sun_distance_au(jd: float) -> float:
    """Earth-Sun distance [AU] from the same low-precision series.

    r = 1.000140612 - 0.016708617 cos M - 0.000139589 cos 2M.
    Source: Vallado, *Fundamentals of Astrodynamics and Applications*.
    Units: AU. Validity: as :func:`sun_unit_vector_eci`; reproduces the perihelion and
    aphelion distances to about 1e-4 AU.
    """
    jd_v = _v.as_finite_float(jd, "jd")
    _, m_anom, _ = _sun_mean_anomaly_and_lambda(jd_v)
    return float(1.000140612 - 0.016708617 * np.cos(m_anom) - 0.000139589 * np.cos(2 * m_anom))


def sun_direction_for_beta(
    inclination_rad: float, raan_rad: float, beta_rad: float, phase_rad: float = 0.0
) -> NDArray[np.float64]:
    """Construct a Sun unit vector giving an exact beta angle for a given orbit plane.

    s_hat = sin(beta) h_hat + cos(beta) (cos(phase) P_hat + sin(phase) Q_hat).

    Beta is the angle between the Sun direction and the orbit plane, positive toward the
    orbit normal; ``phase_rad`` places the in-plane component at that argument of
    latitude. Units: rad in, dimensionless unit vector out. This is the deterministic
    way to size for a chosen illumination geometry without pinning a calendar date.
    """
    beta = _v.in_range(beta_rad, "beta_rad", -np.pi / 2, np.pi / 2)
    phase = _v.as_finite_float(phase_rad, "phase_rad")
    p_hat, q_hat, h_hat = node_axes(inclination_rad, raan_rad)
    in_plane = np.cos(phase) * p_hat + np.sin(phase) * q_hat
    s = np.sin(beta) * h_hat + np.cos(beta) * in_plane
    return s / float(np.linalg.norm(s))


def beta_angle(sun_hat: ArrayLike, inclination_rad: float, raan_rad: float) -> float:
    """Beta angle [rad]: the elevation of the Sun above the orbit plane.

    beta = arcsin(s_hat . h_hat). Source: standard mission-geometry definition (Larson &
    Wertz, *Space Mission Analysis and Design*). Units: rad, in [-pi/2, pi/2].
    """
    s = _v.as_unit_vector(sun_hat, "sun_hat")
    h_hat = orbit_normal(inclination_rad, raan_rad)
    return float(np.arcsin(np.clip(float(np.dot(s, h_hat)), -1.0, 1.0)))


def in_eclipse_cylindrical(
    r_eci: ArrayLike, sun_hat: ArrayLike, body_radius_m: float = R_EARTH_EQUATORIAL
) -> NDArray[np.bool_]:
    """Cylindrical-shadow eclipse test.

    A point is in shadow when it lies anti-sunward of the Earth centre
    (r . s_hat < 0) and its distance from the Earth-Sun line is below the Earth radius.

    Source: the cylindrical (umbra-only) shadow model described in Vallado,
    *Fundamentals of Astrodynamics and Applications*. Units: m in, boolean out.
    Assumptions: parallel sunlight, spherical Earth, no penumbra, no atmospheric
    refraction. Validity: LEO sizing. The neglected penumbra spans roughly 10-20 s of a
    LEO eclipse entry or exit and is a fraction of a percent of the orbit; the model is
    therefore slightly pessimistic on illuminated time near the terminator.
    """
    s = _v.as_unit_vector(sun_hat, "sun_hat")
    rad = _v.positive(body_radius_m, "body_radius_m")
    r = np.atleast_2d(np.asarray(r_eci, dtype=float))
    if r.shape[-1] != 3:
        raise ValueError(f"r_eci must have trailing dimension 3, got shape {r.shape}")
    along = r @ s
    perp = np.linalg.norm(r - along[..., None] * s, axis=-1)
    out = (along < 0.0) & (perp < rad)
    return out if np.ndim(r_eci) > 1 else out.reshape(())


def eclipse_fraction_cylindrical(
    radius_m: float,
    inclination_rad: float,
    raan_rad: float,
    sun_hat: ArrayLike,
    n_samples: int = 3600,
    body_radius_m: float = R_EARTH_EQUATORIAL,
) -> float:
    """Fraction of one orbit spent in the cylindrical umbra (dimensionless, 0 to 1).

    Evaluated by uniform sampling in argument of latitude, which for a circular orbit is
    uniform in time. Discretisation error is bounded by 1 / ``n_samples``.
    """
    n = int(n_samples)
    if n < 8:
        raise ValueError(f"n_samples must be >= 8, got {n_samples!r}")
    u = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r, _ = circular_orbit_state(radius_m, inclination_rad, raan_rad, u)
    return float(np.mean(in_eclipse_cylindrical(r, sun_hat, body_radius_m)))
