"""Tilted centred-dipole geomagnetic field model.

This module implements the **degree-1 (dipole) truncation of the IGRF** and
nothing more.  It is *not* a full IGRF or WMM evaluation: no degree >= 2 terms,
no secular variation beyond the single 2025.0 epoch, no external field, no
crustal field.  Section "Validation" of the repository README quantifies the
error this truncation costs against IGRF-14 values at reference points.

Model
-----
The IGRF magnetic scalar potential (Schmidt semi-normalised, ``B = -grad V``)
truncated at degree n = 1 is

    V(r) = a^3 (g . r) / r^3,     g = (g(1,1), h(1,1), g(1,0))     [nT]

with ``a`` the IGRF reference radius (taken here as the WGS 84 equatorial
radius) and the components of ``g`` expressed in an Earth-centred Earth-fixed
(ECEF) frame whose x-axis points at (0 deg N, 0 deg E), y-axis at (0 deg N,
90 deg E) and z-axis at the geographic north pole.  Taking the gradient,

    B(r) = (a / r)^3 [ 3 (g . r_hat) r_hat - g ]                   [nT]

which is the standard centred-dipole field.  ``|g| = B0`` is the dipole
equatorial surface field strength.

Validity
--------
Valid for ``r > a`` (outside the source region), i.e. altitudes above the
surface.  Accuracy degrades where the non-dipole field is large, most severely
over the South Atlantic Anomaly.  Errors are measured, not asserted:
see ``validation/field_model_check.py``.

References
----------
Alken, P. et al. / IAGA Working Group V-MOD, "International Geomagnetic
    Reference Field: the fourteenth generation", Earth, Planets and Space,
    2025; coefficient file ``igrf14coeffs.txt``.  Main-field epoch 2025.0
    degree-1 terms only: g(1,0) = -29350.0 nT, g(1,1) = -1410.3 nT,
    h(1,1) = 4545.5 nT.
Wertz, J. R. (ed.), "Spacecraft Attitude Determination and Control", D. Reidel,
    1978, Appendix H (geomagnetic field models, dipole approximation).
Markley, F. L. and Crassidis, J. L., "Fundamentals of Spacecraft Attitude
    Determination and Control", Springer, 2014, sec. 5.1 (magnetometer models
    and the dipole field).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import (
    B0_NT,
    IGRF14_2025_G10_NT,
    IGRF14_2025_G11_NT,
    IGRF14_2025_H11_NT,
    NT_TO_T,
    OMEGA_EARTH_RAD_S,
    R_EARTH_M,
)

#: Degree-1 Gauss coefficient vector in ECEF axes, ``(g11, h11, g10)`` [nT].
DIPOLE_G_NT: NDArray[np.float64] = np.array(
    [IGRF14_2025_G11_NT, IGRF14_2025_H11_NT, IGRF14_2025_G10_NT], dtype=float
)

#: Same vector in tesla.
DIPOLE_G_T: NDArray[np.float64] = DIPOLE_G_NT * NT_TO_T

#: Dipole equatorial surface field strength B0 [T].
B0_T: float = B0_NT * NT_TO_T


def dipole_field_ecef(r_ecef_m: ArrayLike) -> NDArray[np.float64]:
    """Centred-dipole field in ECEF axes [T] at ECEF position [m].

    Parameters
    ----------
    r_ecef_m : array_like, shape (3,) or (N, 3)
        Position(s) in the Earth-fixed frame [m].  Must be outside the Earth
        (``|r| > R_EARTH_M``); the dipole solution is only valid there.

    Returns
    -------
    ndarray, same shape as input
        Magnetic flux density [T].

    Raises
    ------
    ValueError
        If any position lies at or below the WGS 84 equatorial radius.
    """
    r = np.asarray(r_ecef_m, dtype=float)
    single = r.ndim == 1
    if single:
        r = r[None, :]
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"r_ecef_m must have shape (3,) or (N, 3), got {r.shape}")
    rn = np.linalg.norm(r, axis=1)
    if np.any(rn <= R_EARTH_M):
        raise ValueError(
            "dipole_field_ecef is only valid outside the Earth: "
            f"min |r| = {rn.min():.1f} m <= R_EARTH_M = {R_EARTH_M:.1f} m"
        )
    rhat = r / rn[:, None]
    scale = (R_EARTH_M / rn) ** 3
    gdotr = rhat @ DIPOLE_G_T
    b = scale[:, None] * (3.0 * gdotr[:, None] * rhat - DIPOLE_G_T[None, :])
    return b[0] if single else b


def eci_to_ecef_angle(t_s: float, gmst0_rad: float = 0.0) -> float:
    """Earth-rotation angle [rad] at time ``t_s`` [s] past epoch.

    Uses a uniform rotation rate ``OMEGA_EARTH_RAD_S``; precession, nutation,
    polar motion and length-of-day variation are all neglected.  Over the
    hours-long detumble horizons modelled here that omission is far smaller
    than the dipole-truncation error.
    """
    return gmst0_rad + OMEGA_EARTH_RAD_S * float(t_s)


def rot_z(angle_rad: float) -> NDArray[np.float64]:
    """Rotation matrix about +z by ``angle_rad`` [rad] (active on coordinates).

    ``rot_z(theta) @ v_eci`` gives the components of ``v`` in a frame rotated
    by ``+theta`` about z, i.e. ECEF components when ``theta`` is the Earth
    rotation angle.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def dipole_field_eci(
    r_eci_m: ArrayLike, t_s: float, gmst0_rad: float = 0.0
) -> NDArray[np.float64]:
    """Centred-dipole field in ECI axes [T].

    The dipole is fixed in ECEF and therefore rotates with the Earth; this
    function rotates the position into ECEF, evaluates the dipole, and rotates
    the field back.

    Parameters
    ----------
    r_eci_m : array_like, shape (3,)
        Inertial position [m].
    t_s : float
        Seconds past the epoch at which the Earth rotation angle is
        ``gmst0_rad``.
    gmst0_rad : float
        Earth rotation angle at ``t = 0`` [rad].
    """
    rot = rot_z(eci_to_ecef_angle(t_s, gmst0_rad))
    r_ecef = rot @ np.asarray(r_eci_m, dtype=float)
    b_ecef = dipole_field_ecef(r_ecef)
    return rot.T @ b_ecef


def spherical_position_ecef(
    lat_deg: float, lon_deg: float, alt_km: float, radius_m: float = R_EARTH_M
) -> NDArray[np.float64]:
    """ECEF position [m] from spherical geocentric latitude/longitude/altitude.

    A *spherical* Earth of radius ``radius_m`` is used, consistent with the
    dipole model's own spherical-harmonic reference sphere.  This differs from
    geodetic (WGS 84 ellipsoidal) latitude by up to ~0.19 deg; the resulting
    field-magnitude difference is small compared with the dipole truncation
    error and is reported in ``validation/field_model_check.py``.
    """
    if not -90.0 <= lat_deg <= 90.0:
        raise ValueError(f"lat_deg must lie in [-90, 90], got {lat_deg}")
    if alt_km < 0.0:
        raise ValueError(f"alt_km must be non-negative, got {alt_km}")
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = radius_m + alt_km * 1000.0
    return np.array(
        [r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)],
        dtype=float,
    )


def field_magnitude_nt(lat_deg: float, lon_deg: float, alt_km: float) -> float:
    """Total dipole field magnitude [nT] at a geocentric lat/lon/altitude."""
    pos = spherical_position_ecef(lat_deg, lon_deg, alt_km)
    if alt_km == 0.0:
        pos = pos * (1.0 + 1e-12)  # keep strictly outside the reference sphere
    return float(np.linalg.norm(dipole_field_ecef(pos))) / NT_TO_T


def geomagnetic_north_pole_deg() -> tuple[float, float]:
    """Geomagnetic (dipole) north pole, ``(latitude_deg, longitude_deg_east)``.

    The geomagnetic north pole is the surface point where the dipole field is
    vertical and downward, i.e. the direction ``-g / |g|``.  Longitude is
    returned in ``(-180, 180]`` degrees east.
    """
    axis = -DIPOLE_G_NT / np.linalg.norm(DIPOLE_G_NT)
    lat = float(np.degrees(np.arcsin(axis[2])))
    lon = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return lat, lon


def dipole_tilt_deg() -> float:
    """Tilt of the dipole axis from the Earth rotation axis [deg]."""
    lat, _ = geomagnetic_north_pole_deg()
    return 90.0 - lat
