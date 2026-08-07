"""Analytic Cartesian gradients (wavefront slopes) of Zernike modes.

A Shack-Hartmann sensor measures the average wavefront **slope** over each
subaperture, not the wavefront itself, so a Zernike interaction matrix needs
``dZ/dx`` and ``dZ/dy``. Finite differences of the polynomials are noisy and
slow; the closed-form derivative below is exact to machine precision.

Derivation
----------
With ``Z = N_n^m R_n^|m|(rho) Theta_m(theta)``, ``rho = sqrt(x^2 + y^2)``,
``theta = atan2(y, x)``, the chain rule with
``d(rho)/dx = cos(theta)``, ``d(theta)/dx = -sin(theta)/rho``,
``d(rho)/dy = sin(theta)``, ``d(theta)/dy =  cos(theta)/rho`` gives::

    dZ/dx = N [ R'(rho) Theta(theta) cos(theta)
                - (R(rho)/rho) Theta'(theta) sin(theta) ]
    dZ/dy = N [ R'(rho) Theta(theta) sin(theta)
                + (R(rho)/rho) Theta'(theta) cos(theta) ]

with ``Theta'`` the derivative of the angular factor:
``-m sin(m theta)`` for ``m > 0``, ``|m| cos(|m| theta)`` for ``m < 0``, and
``0`` for ``m = 0``.

Handling of the origin
----------------------
The ``1/rho`` looks singular but is not: for ``|m| >= 1`` the radial polynomial
has ``R_n^m(rho) = O(rho^|m|)``, so ``R/rho`` is itself a polynomial. This
module divides the *coefficient array* rather than the values, so ``rho = 0``
is evaluated exactly with no special-casing, no ``nan``, and no epsilon fudge.
For ``m = 0`` the ``Theta'`` factor is identically zero and the term is dropped
before it can be formed.

Units
-----
``x``, ``y`` are dimensionless pupil coordinates normalised so the pupil edge
is at ``x^2 + y^2 = 1``. The returned gradients are therefore *per unit
normalised pupil radius*. To convert to physical slope in radians for a pupil
of radius ``R_pupil`` metres and a wavefront expressed in metres, divide by
``R_pupil`` [m]. This library performs no such conversion for you.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .indexing import noll_to_nm, osa_to_nm, validate_nm
from .polynomials import azimuthal_factor, normalization, radial_coefficients

__all__ = [
    "zernike_gradient",
    "zernike_gradient_noll",
    "zernike_gradient_osa",
    "zernike_slope_matrix",
]


def _azimuthal_derivative(m: int, theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """d/dtheta of the angular factor (dimensionless, per radian)."""
    if m == 0:
        return np.zeros_like(theta)
    if m > 0:
        return -m * np.sin(m * theta)
    return (-m) * np.cos(-m * theta)


def zernike_gradient(
    n: int,
    m: int,
    x: ArrayLike,
    y: ArrayLike,
    normalized: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Analytic ``(dZ/dx, dZ/dy)`` of the Zernike mode ``Z_n^m``.

    Parameters
    ----------
    n, m : int
        Zernike indices; ``n - |m|`` must be even.
    x, y : array_like
        Cartesian pupil coordinates normalised to the pupil radius
        (dimensionless; pupil edge at ``x^2 + y^2 = 1``).
    normalized : bool, optional
        Use the Noll/ANSI orthonormal scaling (default True).

    Returns
    -------
    tuple of numpy.ndarray
        ``(dZ/dx, dZ/dy)``, per unit normalised pupil radius, broadcast to the
        common shape of ``x`` and ``y``.

    Notes
    -----
    Known values used as tests: for Noll ``j = 2`` (``n = 1, m = +1``,
    normalised) ``Z = 2x`` so ``dZ/dx = 2`` and ``dZ/dy = 0`` everywhere; for
    ``j = 4`` (defocus, ``Z = sqrt(3)(2 rho^2 - 1)``)
    ``dZ/dx = 4 sqrt(3) x``.
    """
    validate_nm(n, m)
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr, y_arr = np.broadcast_arrays(x_arr, y_arr)

    rho = np.hypot(x_arr, y_arr)
    theta = np.arctan2(y_arr, x_arr)

    coeffs = radial_coefficients(n, m)  # ascending powers of rho
    powers = np.arange(coeffs.size)
    d_coeffs = (coeffs * powers)[1:]  # dR/drho, ascending powers
    dR = np.polyval(d_coeffs[::-1], rho) if d_coeffs.size else np.zeros_like(rho)

    ang = azimuthal_factor(m, theta)
    norm = normalization(n, m, normalized)

    radial_term_x = dR * ang * np.cos(theta)
    radial_term_y = dR * ang * np.sin(theta)

    if m == 0:
        # Theta' == 0: the 1/rho term is identically absent.
        return norm * radial_term_x, norm * radial_term_y

    # coeffs[0] is exactly 0 for |m| >= 1, so R/rho is the shifted polynomial.
    r_over_rho = np.polyval(coeffs[1:][::-1], rho)
    d_ang = _azimuthal_derivative(m, theta)

    dzdx = norm * (radial_term_x - r_over_rho * d_ang * np.sin(theta))
    dzdy = norm * (radial_term_y + r_over_rho * d_ang * np.cos(theta))
    return dzdx, dzdy


def zernike_gradient_noll(
    j: int, x: ArrayLike, y: ArrayLike, normalized: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Analytic gradient of the Noll-indexed mode ``Z_j`` (``j >= 1``)."""
    n, m = noll_to_nm(j)
    return zernike_gradient(n, m, x, y, normalized=normalized)


def zernike_gradient_osa(
    j: int, x: ArrayLike, y: ArrayLike, normalized: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Analytic gradient of the OSA/ANSI-indexed mode ``Z_j`` (``j >= 0``)."""
    n, m = osa_to_nm(j)
    return zernike_gradient(n, m, x, y, normalized=normalized)


def zernike_slope_matrix(
    indices: list[tuple[int, int]],
    x: ArrayLike,
    y: ArrayLike,
    normalized: bool = True,
) -> NDArray[np.float64]:
    """Shack-Hartmann style slope interaction matrix.

    Builds the ``(2 P, M)`` matrix whose columns are the analytic slopes of
    each requested mode sampled at ``P`` subaperture centres, stacked as
    ``[dZ/dx at all points; dZ/dy at all points]``. Multiplying by a
    coefficient vector gives the noise-free slope vector in the same stacking.

    Parameters
    ----------
    indices : list of (int, int)
        Modes as ``(n, m)`` pairs, in the order the coefficient vector uses.
    x, y : array_like
        Subaperture centre coordinates, normalised to pupil radius. Flattened.
    normalized : bool, optional
        Noll/ANSI orthonormal scaling (default True).

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(2 * n_points, len(indices))``.

    Notes
    -----
    This is a *point-sampled* slope model. A real Shack-Hartmann measures the
    subaperture-averaged slope; the two agree only when the mode varies slowly
    across a subaperture. Sensor modelling, spot formation and centroiding are
    deliberately out of scope here.
    """
    if len(indices) == 0:
        raise ValueError("indices must contain at least one (n, m) pair")
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"x and y must have the same size, got {x_arr.size} and {y_arr.size}")
    n_pts = x_arr.size
    mat = np.empty((2 * n_pts, len(indices)), dtype=np.float64)
    for col, (n, m) in enumerate(indices):
        gx, gy = zernike_gradient(n, m, x_arr, y_arr, normalized=normalized)
        mat[:n_pts, col] = gx
        mat[n_pts:, col] = gy
    return mat
