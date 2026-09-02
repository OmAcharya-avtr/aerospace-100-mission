"""Orbit geometry, frames, atmosphere, geomagnetic field and eclipse.

Frames
------
ECI
    Earth-centred inertial; z along the rotation axis, x toward the mean equinox.
LVLH
    Orbit frame of Wertz, *Spacecraft Attitude Determination and Control*: z along nadir
    (``-r_hat``), y along the negative orbit normal, x = y x z, which on a circular orbit
    is the velocity direction.
BODY
    Spacecraft frame, reached from LVLH by a fixed 3-2-1 (yaw, pitch, roll) sequence.
    All-zero offsets are exact nadir pointing with body z down.

Angles are radians, lengths metres, times seconds, unless a name says otherwise.

This module is an independent implementation of the same standard environment models
that P027 ``disturbtorque`` implements, written from the cited sources for the explicit
purpose of cross-checking that package's published momentum accumulation. Nothing is
imported from it. Where a convention had to match for the comparison to mean anything
(the LVLH axis definition, the reduced dipole moment, the exponential density table) the
convention is stated here and its source given.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _validate as _v
from .constants import (
    DEFAULT_DRAG_COEFFICIENT,
    EARTH_REDUCED_DIPOLE,
    MU_EARTH,
    OMEGA_EARTH,
    R_EARTH_EQUATORIAL,
)

__all__ = [
    "frame_rot_x",
    "frame_rot_y",
    "frame_rot_z",
    "node_axes",
    "orbital_period",
    "circular_state",
    "lvlh_dcm",
    "body_dcm_from_lvlh",
    "sun_direction_for_beta",
    "beta_angle",
    "eclipse_boundaries",
    "is_illuminated",
    "eclipse_fraction",
    "EXPONENTIAL_DENSITY_TABLE",
    "density",
    "dipole_field_eci",
    "CircularOrbit",
    "SpacecraftProperties",
    "reference_smallsat",
    "reference_orbit",
]


# --------------------------------------------------------------------------------
# Elementary frame rotations
# --------------------------------------------------------------------------------
def frame_rot_x(angle_rad: float) -> NDArray[np.float64]:
    """Frame rotation about x [rad]: maps a vector's components into the rotated frame."""
    a = _v.as_finite_float(angle_rad, "angle_rad")
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def frame_rot_y(angle_rad: float) -> NDArray[np.float64]:
    """Frame rotation about y [rad]."""
    a = _v.as_finite_float(angle_rad, "angle_rad")
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def frame_rot_z(angle_rad: float) -> NDArray[np.float64]:
    """Frame rotation about z [rad]."""
    a = _v.as_finite_float(angle_rad, "angle_rad")
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def node_axes(
    inclination_rad: float, raan_rad: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Orbit-plane triad ``(P_hat, Q_hat, h_hat)`` in ECI, dimensionless unit vectors.

    Built as the columns of the active rotation ``Rz(RAAN) Rx(inclination)`` applied to
    the ECI basis, which is the standard perifocal-to-inertial sequence for zero argument
    of perigee (Vallado, *Fundamentals of Astrodynamics and Applications*). ``P_hat``
    points at the ascending node, ``Q_hat`` is 90 deg along-track from it, and
    ``h_hat = P_hat x Q_hat`` is the orbit normal.
    """
    inc = _v.in_range(inclination_rad, "inclination_rad", -np.pi, np.pi)
    raan = _v.as_finite_float(raan_rad, "raan_rad")
    rz = np.array(
        [[np.cos(raan), -np.sin(raan), 0.0], [np.sin(raan), np.cos(raan), 0.0], [0.0, 0.0, 1.0]]
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(inc), -np.sin(inc)], [0.0, np.sin(inc), np.cos(inc)]]
    )
    m = rz @ rx
    return m[:, 0].copy(), m[:, 1].copy(), m[:, 2].copy()


def orbital_period(radius_m: float, mu: float = MU_EARTH) -> float:
    """Keplerian period of a circular orbit [s], ``T = 2 pi sqrt(r^3 / mu)``.

    Source: the two-body problem (Vallado). Units: m, m^3 s^-2 in; s out. Assumes a
    point-mass central field; J2 shifts the nodal period by of order 0.1 % in LEO and is
    not modelled.
    """
    r = _v.positive(radius_m, "radius_m")
    m = _v.positive(mu, "mu")
    return float(2.0 * np.pi * np.sqrt(r**3 / m))


def circular_state(
    radius_m: float,
    inclination_rad: float,
    raan_rad: float,
    arg_lat_rad: ArrayLike,
    mu: float = MU_EARTH,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Position [m] and inertial velocity [m s^-1] on a circular orbit, in ECI.

    ``r = R (cos u P_hat + sin u Q_hat)``, ``v = sqrt(mu/R) (-sin u P_hat + cos u
    Q_hat)`` with ``u`` the argument of latitude [rad]. Scalar or shape-``(N,)`` ``u``;
    returns shape ``(3,)`` or ``(N, 3)``.
    """
    r_mag = _v.positive(radius_m, "radius_m")
    mu_v = _v.positive(mu, "mu")
    u = np.asarray(arg_lat_rad, dtype=float)
    if not np.all(np.isfinite(u)):
        raise ValueError("arg_lat_rad must be finite")
    p_hat, q_hat, _ = node_axes(inclination_rad, raan_rad)
    cu, su = np.cos(u)[..., None], np.sin(u)[..., None]
    r = r_mag * (cu * p_hat + su * q_hat)
    v = np.sqrt(mu_v / r_mag) * (-su * p_hat + cu * q_hat)
    return r, v


def lvlh_dcm(r_eci: ArrayLike, v_eci: ArrayLike) -> NDArray[np.float64]:
    """DCM mapping ECI components into LVLH: ``v_lvlh = lvlh_dcm(r, v) @ v_eci``.

    Rows are the LVLH axes in ECI, in the order (x along velocity on a circular orbit,
    y along the negative orbit normal, z along nadir). Dimensionless.
    """
    r = _v.as_vector3(r_eci, "r_eci")
    v = _v.as_vector3(v_eci, "v_eci")
    r_norm = float(np.linalg.norm(r))
    if r_norm == 0.0:
        raise ValueError("r_eci must be non-zero; the nadir direction is undefined at the origin")
    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    if h_norm == 0.0:
        raise ValueError("r_eci and v_eci are parallel; the orbit normal is undefined")
    z_hat = -r / r_norm
    y_hat = -h / h_norm
    x_hat = np.cross(y_hat, z_hat)
    return np.vstack([x_hat, y_hat, z_hat])


def body_dcm_from_lvlh(
    yaw_rad: float = 0.0, pitch_rad: float = 0.0, roll_rad: float = 0.0
) -> NDArray[np.float64]:
    """DCM mapping LVLH components into the body frame for a fixed 3-2-1 offset.

    ``C_bl = R1(roll) R2(pitch) R3(yaw)`` with ``R1, R2, R3`` the frame rotations above.
    Zero angles give exact nadir pointing. Angles in rad; result dimensionless.
    """
    return frame_rot_x(roll_rad) @ frame_rot_y(pitch_rad) @ frame_rot_z(yaw_rad)


# --------------------------------------------------------------------------------
# Sun geometry and eclipse
# --------------------------------------------------------------------------------
def sun_direction_for_beta(
    inclination_rad: float, raan_rad: float, beta_rad: float, phase_rad: float = 0.0
) -> NDArray[np.float64]:
    """Unit Sun direction in ECI giving an exact beta angle for a given orbit plane.

    ``s_hat = sin(beta) h_hat + cos(beta) (cos(phase) P_hat + sin(phase) Q_hat)``.

    Beta is the elevation of the Sun above the orbit plane, positive toward the orbit
    normal; ``phase_rad`` places the in-plane component at that argument of latitude.
    This pins an illumination geometry without pinning a calendar date, which is what a
    reproducible sizing case needs. Units: rad in, dimensionless unit vector out.
    """
    beta = _v.in_range(beta_rad, "beta_rad", -np.pi / 2, np.pi / 2)
    phase = _v.as_finite_float(phase_rad, "phase_rad")
    p_hat, q_hat, h_hat = node_axes(inclination_rad, raan_rad)
    s = np.sin(beta) * h_hat + np.cos(beta) * (np.cos(phase) * p_hat + np.sin(phase) * q_hat)
    return s / float(np.linalg.norm(s))


def beta_angle(sun_hat: ArrayLike, inclination_rad: float, raan_rad: float) -> float:
    """Solar beta angle [rad]: ``arcsin(s_hat . h_hat)``, in ``[-pi/2, pi/2]``.

    Standard mission-geometry definition (Larson & Wertz, *Space Mission Analysis and
    Design*).
    """
    s = _v.as_unit_vector(sun_hat, "sun_hat")
    h_hat = node_axes(inclination_rad, raan_rad)[2]
    return float(np.arcsin(np.clip(float(s @ h_hat), -1.0, 1.0)))


def eclipse_boundaries(
    radius_m: float,
    inclination_rad: float,
    raan_rad: float,
    sun_hat: ArrayLike,
    body_radius_m: float = R_EARTH_EQUATORIAL,
) -> tuple[float, float] | None:
    r"""Closed-form entry and exit argument of latitude of the cylindrical umbra [rad].

    On a circular orbit ``r_hat(u) = cos u P_hat + sin u Q_hat``. With
    ``a = s.P_hat``, ``b = s.Q_hat``, ``A = sqrt(a^2 + b^2) = cos(beta)`` and
    ``phi = atan2(b, a)``, the projection of the position on the Sun line is
    ``R A cos(u - phi)`` and the perpendicular distance is ``R sqrt(1 - A^2 cos^2(u -
    phi))``. The vehicle is in shadow when it is anti-sunward *and* within one Earth
    radius of the Sun line, i.e. exactly when

    .. math:: A \cos(u - \varphi) < -\sqrt{1 - (R_e/R)^2}.

    Returns ``(u_entry, u_exit)`` in ``[0, 2 pi)``, or ``None`` when the orbit never
    enters the umbra (a full-sun orbit, ``A <= sqrt(1 - (Re/R)^2)``).

    Source: the cylindrical (umbra-only) shadow model of Vallado, *Fundamentals of
    Astrodynamics and Applications*. Assumes parallel sunlight, a spherical Earth and no
    penumbra; the neglected penumbra spans roughly 10-20 s of a LEO eclipse edge.
    """
    r = _v.positive(radius_m, "radius_m")
    re = _v.positive(body_radius_m, "body_radius_m")
    if re >= r:
        raise ValueError(f"body_radius_m ({re}) must be smaller than radius_m ({r})")
    s = _v.as_unit_vector(sun_hat, "sun_hat")
    p_hat, q_hat, _ = node_axes(inclination_rad, raan_rad)
    a, b = float(s @ p_hat), float(s @ q_hat)
    amp = float(np.hypot(a, b))
    c0 = float(np.sqrt(1.0 - (re / r) ** 2))
    if amp <= c0:
        return None
    phi = float(np.arctan2(b, a))
    psi = float(np.arccos(-c0 / amp))
    return float((phi + psi) % (2.0 * np.pi)), float((phi - psi) % (2.0 * np.pi))


def is_illuminated(
    r_eci: ArrayLike, sun_hat: ArrayLike, body_radius_m: float = R_EARTH_EQUATORIAL
) -> NDArray[np.bool_]:
    """Cylindrical-shadow illumination flag; True in sunlight. See :func:`eclipse_boundaries`."""
    s = _v.as_unit_vector(sun_hat, "sun_hat")
    re = _v.positive(body_radius_m, "body_radius_m")
    r = np.atleast_2d(np.asarray(r_eci, dtype=float))
    if r.shape[-1] != 3:
        raise ValueError(f"r_eci must have trailing dimension 3, got shape {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ValueError("r_eci must be finite")
    along = r @ s
    perp = np.linalg.norm(r - along[:, None] * s, axis=-1)
    lit = ~((along < 0.0) & (perp < re))
    return lit if np.ndim(r_eci) > 1 else lit.reshape(())


def eclipse_fraction(
    radius_m: float,
    inclination_rad: float,
    raan_rad: float,
    sun_hat: ArrayLike,
    body_radius_m: float = R_EARTH_EQUATORIAL,
) -> float:
    """Closed-form fraction of a circular orbit spent in the cylindrical umbra.

    ``f = (pi - psi) / pi`` with ``psi = arccos(-sqrt(1 - (Re/R)^2) / cos(beta))``; zero
    when the orbit is in full sun. Dimensionless, in ``[0, 1)``. No sampling is involved,
    which is why this is used as the reference for a sampled estimate.
    """
    bounds = eclipse_boundaries(radius_m, inclination_rad, raan_rad, sun_hat, body_radius_m)
    if bounds is None:
        return 0.0
    u_in, u_out = bounds
    return float(((u_out - u_in) % (2.0 * np.pi)) / (2.0 * np.pi))


# --------------------------------------------------------------------------------
# Atmosphere
# --------------------------------------------------------------------------------
EXPONENTIAL_DENSITY_TABLE: tuple[tuple[float, float, float], ...] = (
    (0.0, 1.225, 7.249),
    (25.0, 3.899e-2, 6.349),
    (30.0, 1.774e-2, 6.682),
    (40.0, 3.972e-3, 7.554),
    (50.0, 1.057e-3, 8.382),
    (60.0, 3.206e-4, 7.714),
    (70.0, 8.770e-5, 6.549),
    (80.0, 1.905e-5, 5.799),
    (90.0, 3.396e-6, 5.382),
    (100.0, 5.297e-7, 5.877),
    (110.0, 9.661e-8, 7.263),
    (120.0, 2.438e-8, 9.473),
    (130.0, 8.484e-9, 12.636),
    (140.0, 3.845e-9, 16.149),
    (150.0, 2.070e-9, 22.523),
    (180.0, 5.464e-10, 29.740),
    (200.0, 2.789e-10, 37.105),
    (250.0, 7.248e-11, 45.546),
    (300.0, 2.418e-11, 53.628),
    (350.0, 9.518e-12, 53.298),
    (400.0, 3.725e-12, 58.515),
    (450.0, 1.585e-12, 60.828),
    (500.0, 6.967e-13, 63.822),
    (600.0, 1.454e-13, 71.835),
    (700.0, 3.614e-14, 88.667),
    (800.0, 1.170e-14, 124.64),
    (900.0, 5.245e-15, 181.05),
    (1000.0, 3.019e-15, 268.00),
)
"""Piecewise-exponential atmosphere: ``(base altitude [km], base density [kg m^-3],
scale height [km])``.

The standard exponential table reproduced in Vallado, *Fundamentals of Astrodynamics and
Applications*, whose values derive from the US Standard Atmosphere 1976 below 86 km and
from CIRA-72 above it. Transcribed here from that published table, not from any other
software.

Assumptions and validity: spherically symmetric, static, non-rotating, and — the
dominant limitation — **no solar-activity dependence**. Thermospheric density above
400 km varies by more than an order of magnitude over a solar cycle; aerodynamic torque
is linear in density, so any aerodynamic number produced from this table above 400 km
carries at least a factor-of-several uncertainty. Valid 0 to 1000 km."""

_BASE_ALT_M = np.array([row[0] for row in EXPONENTIAL_DENSITY_TABLE]) * 1000.0
_BASE_RHO = np.array([row[1] for row in EXPONENTIAL_DENSITY_TABLE])
_SCALE_H_M = np.array([row[2] for row in EXPONENTIAL_DENSITY_TABLE]) * 1000.0
DENSITY_MODEL_MAX_ALTITUDE_M: float = 1_000_000.0


def density(altitude_m: ArrayLike, allow_extrapolation: bool = False) -> NDArray[np.float64]:
    """Neutral mass density [kg m^-3] at geometric altitude ``altitude_m`` [m].

    ``rho(h) = rho0_k exp(-(h - h0_k) / H_k)`` for the band ``k`` containing ``h``; see
    :data:`EXPONENTIAL_DENSITY_TABLE` for the model, its source and its validity.

    Raises ``ValueError`` below 0 km, and above 1000 km unless ``allow_extrapolation``.
    """
    h = np.asarray(altitude_m, dtype=float)
    if not np.all(np.isfinite(h)):
        raise ValueError("altitude_m must be finite")
    if np.any(h < 0.0):
        raise ValueError(f"altitude_m must be >= 0 m, got a minimum of {float(h.min())} m")
    if not allow_extrapolation and np.any(h > DENSITY_MODEL_MAX_ALTITUDE_M):
        raise ValueError(
            f"altitude_m must be <= {DENSITY_MODEL_MAX_ALTITUDE_M} m for this table "
            f"(got a maximum of {float(h.max())} m); pass allow_extrapolation=True to "
            "extrapolate the top band, accepting that it is unvalidated"
        )
    idx = np.clip(np.searchsorted(_BASE_ALT_M, h, side="right") - 1, 0, len(_BASE_ALT_M) - 1)
    return _BASE_RHO[idx] * np.exp(-(h - _BASE_ALT_M[idx]) / _SCALE_H_M[idx])


# --------------------------------------------------------------------------------
# Geomagnetic field
# --------------------------------------------------------------------------------
def dipole_field_eci(
    r_eci: ArrayLike,
    reduced_moment: float = EARTH_REDUCED_DIPOLE,
    tilt_rad: float = 0.0,
    rotation_angle_rad: ArrayLike = 0.0,
) -> NDArray[np.float64]:
    r"""Centred, non-tilted geomagnetic dipole field in ECI [T].

    .. math:: \mathbf{B}(\mathbf{r}) = \frac{k}{r^3}\left[3(\hat{\mathbf{m}}\cdot
              \hat{\mathbf{r}})\hat{\mathbf{r}} - \hat{\mathbf{m}}\right],
              \qquad \hat{\mathbf{m}} = -\hat{\mathbf{z}}

    with ``k`` the reduced moment ``B0 Re^3`` [T m^3]. The moment points geographic
    south, so the equatorial field magnitude is ``k/r^3`` and the polar magnitude is
    ``2k/r^3``.

    Source: the centred-dipole reduction of the geomagnetic field in Wertz, *Spacecraft
    Attitude Determination and Control*; the same form appears in Markley and Crassidis,
    *Fundamentals of Spacecraft Attitude Determination and Control*.

    Units: ``r_eci`` m, return T; shape ``(3,)`` or ``(N, 3)`` following the input.

    Assumptions and validity: no dipole tilt (11 deg in reality), no offset (about
    500 km), no secular variation, no South Atlantic Anomaly, no external field.
    Pointwise magnitude errors against IGRF are of order 20-30 % and direction errors
    reach tens of degrees. **That matters for this package**, because magnetic
    desaturation is a function of the field direction; see README Limitations.

    Dipole tilt
    -----------
    ``tilt_rad`` tilts the moment away from the ``-z`` axis and ``rotation_angle_rad``
    (scalar, or one value per position) turns the tilted moment about ``z``, so passing
    ``omega_earth * t`` gives a dipole that rotates with the Earth:

    ``m_hat = -Rz(rotation) [sin(tilt), 0, cos(tilt)]``.

    Both default to zero, which reproduces the centred non-tilted model exactly, and the
    cross-check against P027 uses that default. A representative tilt for recent IGRF
    epochs is about 9.4 deg (geomagnetic pole near 80.6 deg N); it is a parameter here,
    not a constant, because this package does not carry IGRF coefficients and cannot
    claim an epoch. Adding tilt and rotation makes the field geometry evolve from orbit
    to orbit instead of repeating exactly, which is why the scheduling episodes use it.
    """
    k = _v.positive(reduced_moment, "reduced_moment")
    tilt = _v.in_range(tilt_rad, "tilt_rad", -np.pi / 2, np.pi / 2)
    r = np.asarray(r_eci, dtype=float)
    single = r.ndim == 1
    r2 = np.atleast_2d(r)
    if r2.shape[-1] != 3:
        raise ValueError(f"r_eci must have trailing dimension 3, got shape {r.shape}")
    if not np.all(np.isfinite(r2)):
        raise ValueError("r_eci must be finite")
    norm = np.linalg.norm(r2, axis=-1)
    if np.any(norm == 0.0):
        raise ValueError("r_eci must be non-zero")
    r_hat = r2 / norm[:, None]
    rot = np.atleast_1d(np.asarray(rotation_angle_rad, dtype=float))
    if not np.all(np.isfinite(rot)):
        raise ValueError("rotation_angle_rad must be finite")
    if rot.size not in (1, r2.shape[0]):
        raise ValueError(
            f"rotation_angle_rad must be a scalar or have length {r2.shape[0]}, "
            f"got length {rot.size}"
        )
    st, ct = np.sin(tilt), np.cos(tilt)
    cr, sr = np.cos(rot), np.sin(rot)
    m_hat = -np.stack([st * cr, st * sr, np.full(rot.size, ct)], axis=1)
    dot = np.sum(m_hat * r_hat, axis=1)[:, None]
    b = (k / norm[:, None] ** 3) * (3.0 * dot * r_hat - m_hat)
    return b[0] if single else b


# --------------------------------------------------------------------------------
# Vehicle and orbit description
# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpacecraftProperties:
    """Rigid-body and surface properties needed for the four disturbance torques.

    Offsets are the vector from the centre of mass to the relevant centre of pressure,
    **body frame, metres**; torque is ``offset x force``.

    Attributes
    ----------
    inertia : (3, 3) or (3,)
        Inertia tensor about the centre of mass [kg m^2]; a length-3 sequence is read as
        principal moments.
    drag_area_m2, drag_coefficient, cp_aero_offset_m
        Projected area [m^2], dimensionless Cd, centre-of-pressure offset [m].
    srp_area_m2, srp_reflectance, cp_srp_offset_m
        Projected sunlit area [m^2], reflectance factor q in [0, 1], optical offset [m].
    residual_dipole_am2 : (3,)
        Residual magnetic dipole of the vehicle, body frame [A m^2].
    mass_kg : float or None
        Mass [kg]; carried for reporting only. No torque here depends on it.
    """

    inertia: NDArray[np.float64]
    drag_area_m2: float = 0.0
    drag_coefficient: float = DEFAULT_DRAG_COEFFICIENT
    cp_aero_offset_m: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    srp_area_m2: float = 0.0
    srp_reflectance: float = 0.6
    cp_srp_offset_m: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    residual_dipole_am2: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    mass_kg: float | None = None

    def __post_init__(self) -> None:
        s = object.__setattr__
        s(self, "inertia", _v.as_inertia_matrix(self.inertia))
        s(self, "drag_area_m2", _v.non_negative(self.drag_area_m2, "drag_area_m2"))
        s(self, "drag_coefficient", _v.non_negative(self.drag_coefficient, "drag_coefficient"))
        s(self, "cp_aero_offset_m", _v.as_vector3(self.cp_aero_offset_m, "cp_aero_offset_m"))
        s(self, "srp_area_m2", _v.non_negative(self.srp_area_m2, "srp_area_m2"))
        s(self, "srp_reflectance", _v.in_range(self.srp_reflectance, "srp_reflectance", 0.0, 1.0))
        s(self, "cp_srp_offset_m", _v.as_vector3(self.cp_srp_offset_m, "cp_srp_offset_m"))
        s(
            self,
            "residual_dipole_am2",
            _v.as_vector3(self.residual_dipole_am2, "residual_dipole_am2"),
        )
        if self.mass_kg is not None:
            s(self, "mass_kg", _v.positive(self.mass_kg, "mass_kg"))


@dataclass(frozen=True)
class CircularOrbit:
    """Circular Earth orbit with a fixed pointing offset from nadir.

    ``altitude_m`` is above the WGS-84 equatorial radius; the pointing offsets are the
    fixed 3-2-1 angles from LVLH to body [rad]; ``mu`` is the gravitational parameter
    [m^3 s^-2].
    """

    altitude_m: float
    inclination_rad: float = 0.0
    raan_rad: float = 0.0
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    mu: float = MU_EARTH

    def __post_init__(self) -> None:
        s = object.__setattr__
        s(self, "altitude_m", _v.positive(self.altitude_m, "altitude_m"))
        s(
            self,
            "inclination_rad",
            _v.in_range(self.inclination_rad, "inclination_rad", -np.pi, np.pi),
        )
        for name in ("raan_rad", "yaw_rad", "pitch_rad", "roll_rad"):
            s(self, name, _v.as_finite_float(getattr(self, name), name))
        s(self, "mu", _v.positive(self.mu, "mu"))

    @property
    def radius_m(self) -> float:
        """Orbit radius from the Earth centre [m]."""
        return R_EARTH_EQUATORIAL + self.altitude_m

    @property
    def period_s(self) -> float:
        """Keplerian period [s]."""
        return orbital_period(self.radius_m, self.mu)

    @property
    def mean_motion_rad_s(self) -> float:
        """Mean motion ``sqrt(mu / r^3)`` [rad s^-1]."""
        return float(np.sqrt(self.mu / self.radius_m**3))

    @property
    def speed_ms(self) -> float:
        """Inertial circular speed ``sqrt(mu / r)`` [m s^-1]."""
        return float(np.sqrt(self.mu / self.radius_m))

    @property
    def body_rate_body_rad_s(self) -> NDArray[np.float64]:
        """Body angular velocity w.r.t. inertial for LVLH-locked pointing [rad s^-1].

        The LVLH frame turns once per orbit about the orbit normal, which is LVLH ``-y``,
        so ``omega_lvlh = (0, -n, 0)`` and ``omega_body = C_bl omega_lvlh``. This is the
        rate that produces the gyroscopic ``omega x h`` coupling in the wheel equation.
        """
        c_bl = body_dcm_from_lvlh(self.yaw_rad, self.pitch_rad, self.roll_rad)
        return c_bl @ np.array([0.0, -self.mean_motion_rad_s, 0.0])


def reference_smallsat() -> SpacecraftProperties:
    """The cross-check reference vehicle: a 100 kg ESPA-class LEO microsatellite.

    Inertia ``diag(4, 8, 10)`` kg m^2; drag area 0.6 m^2 with Cd 2.2; sunlit area 1.2 m^2
    with q = 0.6; both centres of pressure offset ``(0.02, 0.02, 0.05)`` m from the centre
    of mass; residual dipole ``(0.05, 0.05, 0.10)`` A m^2.

    These numbers are **not measured from any spacecraft**. They are the published
    reference-vehicle definition of P027 ``disturbtorque``, reproduced here as the agreed
    environment for the independent cross-check documented in
    ``validation/p027_cross_check.py``. Reproducing an input definition is the point; no
    code is shared with that package.
    """
    return SpacecraftProperties(
        inertia=np.diag([4.0, 8.0, 10.0]),
        drag_area_m2=0.6,
        drag_coefficient=2.2,
        cp_aero_offset_m=np.array([0.02, 0.02, 0.05]),
        srp_area_m2=1.2,
        srp_reflectance=0.6,
        cp_srp_offset_m=np.array([0.02, 0.02, 0.05]),
        residual_dipole_am2=np.array([0.05, 0.05, 0.10]),
        mass_kg=100.0,
    )


def reference_orbit(altitude_km: float = 500.0) -> CircularOrbit:
    """The cross-check reference orbit: circular, ``altitude_km``, i = 51.6 deg, RAAN 0,
    nadir pointing with 5 deg pitch and 5 deg roll so the gravity-gradient torque is
    non-zero. Same provenance note as :func:`reference_smallsat`."""
    return CircularOrbit(
        altitude_m=_v.positive(altitude_km, "altitude_km") * 1000.0,
        inclination_rad=np.radians(51.6),
        raan_rad=0.0,
        pitch_rad=np.radians(5.0),
        roll_rad=np.radians(5.0),
    )


def relative_wind_eci(
    r_eci: NDArray[np.float64], v_eci: NDArray[np.float64], co_rotating: bool = True
) -> NDArray[np.float64]:
    """Velocity of the vehicle relative to the atmosphere, ECI [m s^-1].

    ``v_rel = v - omega_E x r`` when ``co_rotating`` (a rigidly co-rotating atmosphere,
    the standard sizing assumption), otherwise the inertial velocity. The correction
    lowers the relative speed by up to about 6 % on a prograde equatorial LEO orbit and
    the aerodynamic torque, quadratic in speed, by up to about 12 %.
    """
    if not co_rotating:
        return np.asarray(v_eci, dtype=float)
    omega = np.array([0.0, 0.0, OMEGA_EARTH])
    return np.asarray(v_eci, dtype=float) - np.cross(omega, np.asarray(r_eci, dtype=float))
