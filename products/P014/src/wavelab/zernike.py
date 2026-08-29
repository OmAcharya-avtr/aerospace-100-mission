"""Noll-indexed Zernike polynomials, evaluation, analytic gradients, and fitting.

WaveLab implements its own Zernike machinery so the package is self-contained
(no cross-product imports; see the mission build rules). The definitions and
normalisation below reproduce the standard results used throughout adaptive
optics and are the same conventions documented independently by product P016
(ZernKit) and used by product P018 (ShackSim); WaveLab cites that overlap as
related prior work rather than importing it.

Conventions (stated explicitly, per the mission engineering-honesty rule)
--------------------------------------------------------------------------
Radial polynomial, source M. Born & E. Wolf, *Principles of Optics*, 7th
(expanded) ed., Cambridge University Press, 1999, Sec. 9.2::

    R_n^m(rho) = sum_{k=0}^{(n-|m|)/2} (-1)^k (n-k)!
                 ----------------------------------------------------- rho^(n-2k)
                 k! ((n+|m|)/2 - k)! ((n-|m|)/2 - k)!

defined only for integer ``n >= 0``, ``|m| <= n``, ``n - |m|`` even.

Noll (1976) single index and orthonormal scaling: R. J. Noll, "Zernike
polynomials and atmospheric turbulence", *J. Opt. Soc. Am.* **66** (3),
207-211 (1976). Index ``j`` is 1-based, ``j = 1`` is piston; within radial
order ``n`` (occupying ``j = n(n+1)/2 + 1 .. (n+1)(n+2)/2``), ``|m|``
increases, and the even ``j`` of a pair carries the cosine (``m > 0``)
member. Orthonormal mode::

    Z_n^m(rho, theta) = N_n^m R_n^|m|(rho) * { cos(m theta)   m > 0
                                                sin(|m| theta) m < 0
                                                1              m = 0 }
    N_n^m = sqrt(2(n+1))  (m != 0),   N_n^0 = sqrt(n+1)

orthonormal under the area-normalised weight ``1/pi`` over the unit disc::

    (1/pi) int_0^{2pi} int_0^1 Z_i Z_j rho drho dtheta = delta_ij

Pupil coordinates ``(x, y)`` are dimensionless, normalised so the pupil edge
is at ``x^2 + y^2 = 1``; the wavefront/phase ``W`` carried by a coefficient
vector is in whatever unit the caller assigns (WaveLab uses radians of phase
throughout — see README "Engineering theory"). Gradients ``dZ/dx``, ``dZ/dy``
are then *per unit normalised pupil radius*; converting to a physical slope
(rad of ray angle per metre) requires dividing by the physical pupil radius,
a step WaveLab does not perform (it is a sensor-geometry detail out of scope
here; see ShackSim P018 for a full physical Shack-Hartmann model).

Validity range: the unit disc ``0 <= rho <= 1``. Values are computed for
``rho > 1`` without error but are neither orthogonal nor physically
meaningful there.
"""

from __future__ import annotations

from math import factorial, isqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "validate_nm",
    "noll_to_nm",
    "nm_to_noll",
    "radial_order_from_noll",
    "normalization",
    "radial_coefficients",
    "radial_polynomial",
    "azimuthal_factor",
    "zernike",
    "zernike_noll",
    "zernike_gradient",
    "zernike_gradient_noll",
    "unit_disc_grid",
    "zernike_basis_matrix",
    "zernike_slope_matrix",
    "fit_zernike",
    "MODE_NAMES",
]

#: Traditional names for the low-order modes, keyed by (n, m).
#: Source: Born & Wolf 1999, Sec. 9.2; ANSI Z80.28.
MODE_NAMES: dict[tuple[int, int], str] = {
    (0, 0): "piston",
    (1, -1): "tilt (y)",
    (1, 1): "tilt (x)",
    (2, -2): "oblique astigmatism",
    (2, 0): "defocus",
    (2, 2): "vertical astigmatism",
    (3, -3): "oblique trefoil",
    (3, -1): "vertical coma",
    (3, 1): "horizontal coma",
    (3, 3): "vertical trefoil",
    (4, -4): "oblique quadrafoil",
    (4, -2): "oblique secondary astigmatism",
    (4, 0): "primary spherical",
    (4, 2): "vertical secondary astigmatism",
    (4, 4): "vertical quadrafoil",
}


def validate_nm(n: int, m: int) -> None:
    """Raise if ``(n, m)`` is not a legal Zernike index pair.

    Legal iff ``n >= 0``, ``|m| <= n``, ``n - |m|`` even (Born & Wolf 1999,
    Sec. 9.2 -- ``R_n^m`` vanishes identically otherwise).
    """
    bad = isinstance(n, bool) or isinstance(m, bool)
    if bad or not isinstance(n, (int, np.integer)) or not isinstance(m, (int, np.integer)):
        raise TypeError(f"n and m must be integers, got n={n!r}, m={m!r}")
    n, m = int(n), int(m)
    if n < 0:
        raise ValueError(f"radial degree n must be >= 0, got n={n}")
    if abs(m) > n:
        raise ValueError(f"|m| must be <= n, got n={n}, m={m}")
    if (n - abs(m)) % 2 != 0:
        raise ValueError(f"n - |m| must be even, got n={n}, m={m}")


def radial_order_from_noll(j: int) -> int:
    """Radial degree ``n`` of Noll index ``j >= 1``, exact integer arithmetic.

    ``n`` is the unique integer with ``n(n+1)/2 < j <= (n+1)(n+2)/2``.
    """
    if isinstance(j, bool) or not isinstance(j, (int, np.integer)):
        raise TypeError(f"Noll index j must be an integer, got {j!r}")
    j = int(j)
    if j < 1:
        raise ValueError(f"Noll indexing starts at j = 1 (piston); got j = {j}")
    return (isqrt(8 * (j - 1) + 1) - 1) // 2


def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index ``j >= 1`` to ``(n, m)`` (Noll 1976 ordering).

    Examples
    --------
    >>> [noll_to_nm(j) for j in (1, 2, 3, 4, 5, 6)]
    [(0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2)]
    """
    n = radial_order_from_noll(j)
    p = int(j) - n * (n + 1) // 2  # position within the order, 1..n+1
    k = n % 2
    m = ((p + k) // 2) * 2 - k
    if m != 0 and int(j) % 2 == 1:
        m = -m
    return n, m


def nm_to_noll(n: int, m: int) -> int:
    """Convert ``(n, m)`` to the Noll index ``j >= 1``. Exact inverse of `noll_to_nm`."""
    validate_nm(n, m)
    n, m = int(n), int(m)
    base = n * (n + 1) // 2
    if m == 0:
        return base + 1
    j = base + abs(m)
    wants_even = m > 0
    if (j % 2 == 0) != wants_even:
        j += 1
    return j


def normalization(n: int, m: int, normalized: bool = True) -> float:
    """Noll orthonormalisation factor ``N_n^m`` (Noll 1976, Eq. 2). Dimensionless."""
    validate_nm(n, m)
    if not normalized:
        return 1.0
    return float(np.sqrt(2 * (n + 1))) if m != 0 else float(np.sqrt(n + 1))


def radial_coefficients(n: int, m: int) -> NDArray[np.float64]:
    """Ascending-power coefficients of ``R_n^m``: ``R(rho) = sum_p c[p] rho**p``."""
    validate_nm(n, m)
    n, mm = int(n), abs(int(m))
    coeffs = np.zeros(n + 1, dtype=np.float64)
    for k in range((n - mm) // 2 + 1):
        num = factorial(n - k)
        den = factorial(k) * factorial((n + mm) // 2 - k) * factorial((n - mm) // 2 - k)
        coeffs[n - 2 * k] = (-1) ** k * (num / den)
    return coeffs


def radial_polynomial(n: int, m: int, rho: ArrayLike) -> NDArray[np.float64]:
    """Evaluate ``R_n^m(rho)`` (dimensionless) at normalised radius ``rho >= 0``."""
    rho_arr = np.asarray(rho, dtype=np.float64)
    if np.any(rho_arr < 0):
        raise ValueError("rho must be non-negative")
    coeffs = radial_coefficients(n, m)
    return np.polyval(coeffs[::-1], rho_arr)


def azimuthal_factor(m: int, theta: ArrayLike) -> NDArray[np.float64]:
    """Angular factor: ``cos(m theta)`` (m>0), ``sin(|m| theta)`` (m<0), ``1`` (m=0)."""
    theta_arr = np.asarray(theta, dtype=np.float64)
    m = int(m)
    if m == 0:
        return np.ones_like(theta_arr)
    if m > 0:
        return np.cos(m * theta_arr)
    return np.sin(-m * theta_arr)


def zernike(n: int, m: int, rho: ArrayLike, theta: ArrayLike, normalized: bool = True):
    """Evaluate the Zernike mode ``Z_n^m(rho, theta)`` (dimensionless)."""
    validate_nm(n, m)
    return (
        normalization(n, m, normalized)
        * radial_polynomial(n, m, rho)
        * azimuthal_factor(m, theta)
    )


def zernike_noll(j: int, rho: ArrayLike, theta: ArrayLike, normalized: bool = True):
    """Evaluate the Noll-indexed mode ``Z_j`` (``j >= 1``, ``j = 1`` piston)."""
    n, m = noll_to_nm(j)
    return zernike(n, m, rho, theta, normalized=normalized)


def _azimuthal_derivative(m: int, theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """d/dtheta of the angular factor."""
    if m == 0:
        return np.zeros_like(theta)
    if m > 0:
        return -m * np.sin(m * theta)
    return (-m) * np.cos(-m * theta)


def zernike_gradient(
    n: int, m: int, x: ArrayLike, y: ArrayLike, normalized: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Analytic ``(dZ/dx, dZ/dy)`` of mode ``Z_n^m``, per unit normalised pupil radius.

    Derivation: with ``rho = sqrt(x^2+y^2)``, ``theta = atan2(y,x)`` and the
    chain rule (``d rho/dx = cos theta``, ``d theta/dx = -sin theta / rho``,
    etc.), ``dZ/dx = N[R'(rho) Theta(theta) cos theta
    - (R(rho)/rho) Theta'(theta) sin theta]`` and the ``y`` analogue. The
    apparent ``1/rho`` singularity at the origin is not real: for ``|m| >= 1``
    every term of ``R_n^m`` has a factor ``rho^|m|``, so dividing the
    *coefficient array* by one power of ``rho`` (rather than the values)
    evaluates ``rho = 0`` exactly, with no NaN and no epsilon fudge. For
    ``m = 0`` the ``Theta'`` term is identically zero and is dropped before
    it can be formed.
    """
    validate_nm(n, m)
    x_arr, y_arr = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    rho = np.hypot(x_arr, y_arr)
    theta = np.arctan2(y_arr, x_arr)

    coeffs = radial_coefficients(n, m)
    powers = np.arange(coeffs.size)
    d_coeffs = (coeffs * powers)[1:]
    dR = np.polyval(d_coeffs[::-1], rho) if d_coeffs.size else np.zeros_like(rho)

    ang = azimuthal_factor(m, theta)
    norm = normalization(n, m, normalized)
    rx = dR * ang * np.cos(theta)
    ry = dR * ang * np.sin(theta)

    if m == 0:
        return norm * rx, norm * ry

    r_over_rho = np.polyval(coeffs[1:][::-1], rho)  # coeffs[0] == 0 for |m| >= 1
    d_ang = _azimuthal_derivative(m, theta)
    dzdx = norm * (rx - r_over_rho * d_ang * np.sin(theta))
    dzdy = norm * (ry + r_over_rho * d_ang * np.cos(theta))
    return dzdx, dzdy


def zernike_gradient_noll(j: int, x: ArrayLike, y: ArrayLike, normalized: bool = True):
    """Analytic gradient of the Noll-indexed mode ``Z_j`` (``j >= 1``)."""
    n, m = noll_to_nm(j)
    return zernike_gradient(n, m, x, y, normalized=normalized)


def unit_disc_grid(
    n_pix: int, include_edge: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Square Cartesian grid over ``[-1, 1]^2`` with a unit-disc mask.

    Parameters
    ----------
    n_pix: samples across the full diameter, ``>= 2``.
    include_edge: keep ``rho <= 1`` (True, default) or strictly ``rho < 1``.

    Returns
    -------
    x, y: ``(n_pix, n_pix)`` coordinate arrays. mask: bool array, True inside.
    """
    if isinstance(n_pix, bool) or not isinstance(n_pix, (int, np.integer)):
        raise TypeError(f"n_pix must be an integer, got {n_pix!r}")
    if n_pix < 2:
        raise ValueError(f"n_pix must be >= 2, got {n_pix}")
    axis = np.linspace(-1.0, 1.0, int(n_pix))
    x, y = np.meshgrid(axis, axis)
    rho = np.hypot(x, y)
    mask = rho <= 1.0 if include_edge else rho < 1.0
    return x, y, mask


def zernike_basis_matrix(
    noll_indices: list[int], x: ArrayLike, y: ArrayLike, normalized: bool = True
) -> NDArray[np.float64]:
    """Design matrix of mode values at sample points.

    Parameters
    ----------
    noll_indices: Noll ``j`` values (1-based), in the order of output columns.
    x, y: sample coordinates (flattened), dimensionless pupil units.

    Returns
    -------
    ``(n_points, n_modes)`` matrix ``B`` with ``B[:, k] = Z_{j_k}(x, y)``.
    """
    if len(noll_indices) == 0:
        raise ValueError("noll_indices must be non-empty")
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"x and y must have the same size, got {x_arr.size} and {y_arr.size}")
    cols = [zernike_noll(j, np.hypot(x_arr, y_arr), np.arctan2(y_arr, x_arr), normalized) for j in noll_indices]
    return np.stack(cols, axis=1)


def zernike_slope_matrix(
    noll_indices: list[int], x: ArrayLike, y: ArrayLike, normalized: bool = True
) -> NDArray[np.float64]:
    """Slope interaction matrix: columns are analytic gradients of each mode.

    Builds the ``(2P, M)`` matrix whose columns are ``[dZ_j/dx at P points;
    dZ_j/dy at P points]`` (x-block stacked above y-block). Multiplying by a
    Noll coefficient vector (``M,``) gives the noise-free slope vector
    (``2P,``) in the same stacking. This is a *point-sampled* gradient model:
    a real subaperture measures the area-averaged slope, which coincides with
    the point value only when the wavefront varies slowly across one
    subaperture (documented limitation; see README).

    Parameters
    ----------
    noll_indices: Noll ``j`` values, in coefficient-vector order.
    x, y: subaperture centre coordinates (flattened), dimensionless.

    Returns
    -------
    ``(2 * n_points, len(noll_indices))`` matrix.
    """
    if len(noll_indices) == 0:
        raise ValueError("noll_indices must be non-empty")
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"x and y must have the same size, got {x_arr.size} and {y_arr.size}")
    n_pts = x_arr.size
    mat = np.empty((2 * n_pts, len(noll_indices)), dtype=np.float64)
    for col, j in enumerate(noll_indices):
        n, m = noll_to_nm(j)
        gx, gy = zernike_gradient(n, m, x_arr, y_arr, normalized=normalized)
        mat[:n_pts, col] = gx
        mat[n_pts:, col] = gy
    return mat


def fit_zernike(
    noll_indices: list[int],
    x: ArrayLike,
    y: ArrayLike,
    values: ArrayLike,
    normalized: bool = True,
) -> NDArray[np.float64]:
    """Least-squares fit of sampled wavefront values to Zernike coefficients.

    Solves ``min_a || B a - values ||_2`` with ``B`` the mode design matrix
    from `zernike_basis_matrix`. Requires ``n_points >= len(noll_indices)``
    for a determined fit (an under-determined fit is rejected explicitly
    rather than silently returning the minimum-norm solution of a rank
    deficient problem).

    Returns
    -------
    ``(len(noll_indices),)`` fitted coefficients, in the units of `values`.
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    v_arr = np.asarray(values, dtype=np.float64).ravel()
    if not (x_arr.shape == y_arr.shape == v_arr.shape):
        raise ValueError(
            f"x, y, values must share one shape, got {x_arr.shape}, {y_arr.shape}, {v_arr.shape}"
        )
    if x_arr.size < len(noll_indices):
        raise ValueError(
            f"fit is under-determined: {x_arr.size} points < {len(noll_indices)} modes"
        )
    basis = zernike_basis_matrix(noll_indices, x_arr, y_arr, normalized=normalized)
    coeffs, *_ = np.linalg.lstsq(basis, v_arr, rcond=None)
    return coeffs
