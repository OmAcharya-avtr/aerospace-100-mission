"""Centred, non-tilted geomagnetic dipole field.

Model
-----
B(r) = (k / r^3) * [ 3 (m_hat . r_hat) r_hat - m_hat ],  m_hat = -z_hat (ECI),

so that

B(r) = (k / r^3) * [ z_hat - 3 sin(dec) r_hat ],   |B| = (k / r^3) sqrt(1 + 3 sin^2 dec),

with ``k`` the reduced dipole moment B0 * Re^3 [T m^3] and ``dec`` the geocentric
declination. The moment points geographic *south*, which is why the field at the north
pole points into the Earth and has magnitude 2k/r^3 while the equatorial field is k/r^3.

Source: the centred-dipole reduction of the geomagnetic field as given in Wertz,
*Spacecraft Attitude Determination and Control*, and Larson & Wertz, *Space Mission
Analysis and Design*.

Units: ``r`` in m, ``B`` in tesla, ``k`` in T m^3.

Assumptions and validity
------------------------
* Centred dipole: the real field's offset (about 500 km) and 11 deg tilt of the dipole
  axis are **not** modelled, nor is the Earth's rotation of that tilt through the orbit,
  nor the South Atlantic Anomaly.
* Consequence: pointwise errors against IGRF are of order 20-30 % in magnitude and can
  be tens of degrees in direction. This is a worst-case sizing model. Any product that
  needs the field direction (a magnetometer or magnetorquer model) must use IGRF; on
  PyPI, ``ppigrf`` or ``pyIGRF`` provide it.
* No secular variation, no external (magnetospheric or ionospheric) contributions.
* Validity: 1 to about 6 Earth radii. Beyond that the magnetospheric field departs from
  a dipole entirely.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _validate as _v
from .constants import EARTH_DIPOLE_MOMENT

__all__ = ["dipole_field_eci", "dipole_field_magnitude", "mean_dipole_field_over_orbit"]


def dipole_field_eci(
    r_eci: ArrayLike, dipole_moment: float = EARTH_DIPOLE_MOMENT
) -> NDArray[np.float64]:
    """Geomagnetic field vector in ECI [T] at position ``r_eci`` [m].

    Parameters
    ----------
    r_eci : array_like
        Position from the Earth centre, shape (3,) or (N, 3) [m].
    dipole_moment : float
        Reduced dipole moment B0 * Re^3 [T m^3]; see
        :data:`disturbtorque.constants.EARTH_DIPOLE_MOMENT`.

    Returns
    -------
    ndarray
        Field vector [T], same leading shape as the input.
    """
    k = _v.positive(dipole_moment, "dipole_moment")
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
    z_hat = np.array([0.0, 0.0, 1.0])
    sin_dec = r_hat[:, 2]
    b = (k / norm[:, None] ** 3) * (z_hat - 3.0 * sin_dec[:, None] * r_hat)
    return b[0] if single else b


def dipole_field_magnitude(
    radius_m: ArrayLike, declination_rad: ArrayLike, dipole_moment: float = EARTH_DIPOLE_MOMENT
) -> NDArray[np.float64]:
    """Closed-form field magnitude [T]: (k / r^3) sqrt(1 + 3 sin^2(declination)).

    Units: ``radius_m`` in m, ``declination_rad`` in rad, return in T. Provided as an
    independent check on :func:`dipole_field_eci`.
    """
    k = _v.positive(dipole_moment, "dipole_moment")
    r = np.asarray(radius_m, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("radius_m must be > 0")
    dec = np.asarray(declination_rad, dtype=float)
    return (k / r**3) * np.sqrt(1.0 + 3.0 * np.sin(dec) ** 2)


def mean_dipole_field_over_orbit(
    radius_m: float,
    inclination_rad: float,
    raan_rad: float,
    dipole_moment: float = EARTH_DIPOLE_MOMENT,
) -> NDArray[np.float64]:
    """Exact orbit-averaged dipole field in ECI [T] for a circular orbit.

    Derivation (this package; the integral is elementary). With
    ``r_hat(u) = cos(u) P_hat + sin(u) Q_hat`` and ``sin(dec) = sin(i) sin(u)``,

        <B> = (k/r^3) [ z_hat - 3 <sin(i) sin(u) r_hat(u)> ]
            = (k/r^3) [ z_hat - (3/2) sin(i) Q_hat ],

    because <sin(u) cos(u)> = 0 and <sin^2(u)> = 1/2 over a full revolution.

    Units: m and rad in, T out. This closed form is used in
    ``validation/magnetic_orbit_average.py`` as an independent check on the numerically
    sampled profile.
    """
    from .frames import node_axes  # local import to avoid a module cycle

    k = _v.positive(dipole_moment, "dipole_moment")
    r = _v.positive(radius_m, "radius_m")
    _, q_hat, _ = node_axes(inclination_rad, raan_rad)
    z_hat = np.array([0.0, 0.0, 1.0])
    return (k / r**3) * (z_hat - 1.5 * np.sin(float(inclination_rad)) * q_hat)
