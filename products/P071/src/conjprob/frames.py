"""Radial / along-track / cross-track frame helpers.

Operators declare position offsets and covariances in the local orbital frame,
not in an inertial frame, so a package that accepts an inertial covariance and
nothing else is asking the user to do a rotation by hand. These two functions
do it.

Units: metres, metres per second, metres squared.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def rtn_basis(position_m: ArrayLike, velocity_mps: ArrayLike) -> NDArray[np.float64]:
    """Radial / along-track / cross-track orthonormal basis, shape ``(3, 3)``.

    Columns are the radial unit vector ``r / |r|``, the along-track unit vector
    (the component of velocity perpendicular to radial, normalised) and the
    cross-track unit vector completing a right-handed set. For a circular orbit
    the along-track column is the velocity direction exactly; for an eccentric
    orbit it is not, and the difference is the radial velocity component.

    Raises
    ------
    ValueError
        If the position is zero or the velocity is parallel to it, in which
        case the frame is undefined.
    """
    radial = np.asarray(position_m, dtype=float).reshape(-1)
    velocity = np.asarray(velocity_mps, dtype=float).reshape(-1)
    if radial.shape != (3,) or velocity.shape != (3,):
        raise ValueError("position_m and velocity_mps must each have 3 components")
    norm = float(np.linalg.norm(radial))
    if norm <= 0.0:
        raise ValueError("position_m must be non-zero")
    radial = radial / norm
    along = velocity - (velocity @ radial) * radial
    along_norm = float(np.linalg.norm(along))
    if along_norm <= 0.0:
        raise ValueError("velocity_mps must have a component perpendicular to position_m")
    along = along / along_norm
    return np.column_stack((radial, along, np.cross(radial, along)))


def covariance_from_rtn_sigmas(
    position_m: ArrayLike,
    velocity_mps: ArrayLike,
    sigma_rtn_m: ArrayLike,
    *,
    correlation_ra: float = 0.0,
) -> NDArray[np.float64]:
    """Inertial 3x3 position covariance from declared RTN standard deviations.

    Parameters
    ----------
    position_m, velocity_mps
        State defining the frame, metres and metres per second.
    sigma_rtn_m
        Radial, along-track and cross-track standard deviations, metres, all
        strictly positive.
    correlation_ra
        Correlation coefficient between the radial and along-track components,
        dimensionless, in ``(-1, 1)``. Radial and along-track errors are
        correlated in practice; the default of zero is a declaration, not a
        physical claim.

    Returns
    -------
    ndarray
        Symmetric positive-definite ``(3, 3)`` covariance, metres squared.
    """
    sigmas = np.asarray(sigma_rtn_m, dtype=float).reshape(-1)
    if sigmas.shape != (3,):
        raise ValueError("sigma_rtn_m must have 3 components")
    if np.any(sigmas <= 0.0):
        raise ValueError("sigma_rtn_m components must be strictly positive metres")
    if not -1.0 < correlation_ra < 1.0:
        raise ValueError(f"correlation_ra must lie in (-1, 1), got {correlation_ra!r}")
    local = np.diag(sigmas**2)
    local[0, 1] = local[1, 0] = correlation_ra * sigmas[0] * sigmas[1]
    basis = rtn_basis(position_m, velocity_mps)
    covariance = basis @ local @ basis.T
    return 0.5 * (covariance + covariance.T)
