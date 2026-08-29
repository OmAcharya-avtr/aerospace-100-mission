"""Zernike basis and analytic gradients on the unit disc.

This module is a **self-contained** implementation. It deliberately reproduces
the indexing and normalisation conventions used by the mission's Zernike
reference product (P016 ZernKit) so that coefficient vectors are numerically
interchangeable across the portfolio, but it shares no code with it: WaveLab
imports nothing from any other product.

Conventions (stated explicitly, because mixing them is the classic silent bug)
-----------------------------------------------------------------------------
**Indexing — Noll (1976).** R. J. Noll, "Zernike polynomials and atmospheric
turbulence", *Journal of the Optical Society of America* **66** (3), 207-211
(1976).

* ``j`` is 1-based; ``j = 1`` is piston.
* Radial order ``n`` occupies ``n(n+1)/2 + 1 <= j <= (n+1)(n+2)/2``.
* Within an order ``|m|`` increases; even ``j`` carries the ``cos(m*theta)``
  member (``m > 0``), odd ``j`` the ``sin(|m|*theta)`` member (``m < 0``).
* Noll's own Table I: ``j = 1..6`` -> ``(0,0) (1,1) (1,-1) (2,0) (2,-2) (2,2)``.

**Indexing — OSA/ANSI.** ANSI Z80.28; equivalently L. N. Thibos, R. A.
Applegate, J. T. Schwiegerling, R. Webb, "Standards for reporting the optical
aberrations of eyes", *Journal of Refractive Surgery* **18**, S652-S660 (2002).
``j`` is 0-based with the closed form ``j = (n(n + 2) + m) / 2``.

**Radial polynomial.** M. Born and E. Wolf, *Principles of Optics*, 7th
(expanded) ed., Cambridge University Press 1999, Sec. 9.2::

    R_n^|m|(rho) = sum_{k=0}^{(n-|m|)/2}
                   (-1)^k (n-k)! rho^(n-2k)
                   / [ k! ((n+|m|)/2 - k)! ((n-|m|)/2 - k)! ]

**Normalisation.** Noll (1976), Eq. 2: ``N_n^m = sqrt(2(n+1))`` for ``m != 0``
and ``sqrt(n+1)`` for ``m = 0``, giving orthonormality under the area-normalised
weight ``1/pi`` on the unit disc::

    (1/pi) * int_disc Z_i Z_j dA = delta_ij

so that for a coefficient vector ``a`` excluding piston, the wavefront RMS over
the pupil is ``sqrt(sum_j a_j^2)`` in whatever unit ``a`` carries.

Units and validity range
------------------------
``rho`` is a dimensionless normalised pupil radius, valid on ``0 <= rho <= 1``.
Outside the unit disc the polynomials are finite but neither orthogonal nor
physically meaningful. Gradients returned by :func:`zernike_gradient_noll` are
*per unit normalised pupil radius*; to obtain a physical wavefront slope for a
pupil of radius ``R`` metres, divide by ``R`` (this module never does that
conversion for you). Annular (obscured) pupils are **not** supported -- the
circle polynomials are not orthogonal on an annulus and require the annular set
of V. N. Mahajan, *JOSA* **71**, 75-85 (1981).
"""

from __future__ import annotations

from math import factorial, isqrt

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "noll_to_nm",
    "nm_to_noll",
    "osa_to_nm",
    "nm_to_osa",
    "noll_to_osa",
    "zernike_norm",
    "radial_coefficients",
    "zernike_noll",
    "zernike_gradient_noll",
    "zernike_basis",
    "zernike_gradient_basis",
]


def _check_noll(j: int) -> int:
    if isinstance(j, bool) or not isinstance(j, (int, np.integer)):
        raise TypeError(f"Noll index j must be an integer, got {j!r} ({type(j).__name__})")
    j = int(j)
    if j < 1:
        raise ValueError(f"Noll indexing is 1-based (j = 1 is piston); got j = {j}")
    return j


def noll_to_nm(j: int) -> tuple[int, int]:
    """Map a 1-based Noll index to ``(n, m)``.

    Parameters
    ----------
    j : int
        Noll index, ``j >= 1``. Dimensionless.

    Returns
    -------
    tuple[int, int]
        ``(n, m)``; ``m > 0`` selects ``cos(m*theta)``, ``m < 0`` selects
        ``sin(|m|*theta)``.

    Notes
    -----
    Source: Noll (1976) JOSA 66(3) 207, Table I. Pure integer arithmetic, so
    the mapping is exact for arbitrarily large ``j``.
    """
    j = _check_noll(j)
    n = (isqrt(8 * (j - 1) + 1) - 1) // 2
    p = j - n * (n + 1) // 2  # 1-based position within radial order n
    k = n % 2
    m = ((p + k) // 2) * 2 - k
    if m != 0 and j % 2 == 1:
        m = -m
    return n, m


def _validate_nm(n: int, m: int) -> None:
    if isinstance(n, bool) or isinstance(m, bool):
        raise TypeError("n and m must be integers, not bool")
    if not isinstance(n, (int, np.integer)) or not isinstance(m, (int, np.integer)):
        raise TypeError(f"n and m must be integers, got n={n!r}, m={m!r}")
    if n < 0:
        raise ValueError(f"radial degree n must be >= 0, got {n}")
    if abs(m) > n:
        raise ValueError(f"|m| <= n required, got n={n}, m={m}")
    if (n - abs(m)) % 2 != 0:
        raise ValueError(f"n - |m| must be even, got n={n}, m={m}")


def nm_to_noll(n: int, m: int) -> int:
    """Inverse of :func:`noll_to_nm`. Returns the 1-based Noll index."""
    _validate_nm(n, m)
    base = n * (n + 1) // 2
    if m == 0:
        return base + 1
    j = base + abs(m)
    if (j % 2 == 0) != (m > 0):
        j += 1
    return j


def osa_to_nm(j: int) -> tuple[int, int]:
    """Map a 0-based OSA/ANSI index to ``(n, m)`` (ANSI Z80.28)."""
    if isinstance(j, bool) or not isinstance(j, (int, np.integer)):
        raise TypeError(f"OSA index j must be an integer, got {j!r}")
    j = int(j)
    if j < 0:
        raise ValueError(f"OSA/ANSI indexing is 0-based (j = 0 is piston); got j = {j}")
    n = (isqrt(8 * j + 1) - 1) // 2
    return n, 2 * j - n * (n + 2)


def nm_to_osa(n: int, m: int) -> int:
    """Map ``(n, m)`` to the 0-based OSA/ANSI index ``j = (n(n+2) + m)/2``."""
    _validate_nm(n, m)
    return (n * (n + 2) + m) // 2


def noll_to_osa(j: int) -> int:
    """Convert a 1-based Noll index to the 0-based OSA/ANSI index."""
    return nm_to_osa(*noll_to_nm(j))


def zernike_norm(n: int, m: int) -> float:
    """Noll orthonormalisation factor ``N_n^m`` (dimensionless).

    ``sqrt(2(n+1))`` for ``m != 0``, ``sqrt(n+1)`` for ``m = 0``
    (Noll 1976, Eq. 2).
    """
    _validate_nm(n, m)
    return float(np.sqrt(2.0 * (n + 1)) if m != 0 else np.sqrt(n + 1.0))


def radial_coefficients(n: int, m: int) -> NDArray[np.float64]:
    """Coefficients of ``R_n^|m|(rho)`` in ascending powers of ``rho``.

    Returns an array ``c`` of length ``n + 1`` with
    ``R_n^|m|(rho) = sum_i c[i] * rho**i`` (Born & Wolf 1999, Sec. 9.2).
    Dimensionless.
    """
    _validate_nm(n, m)
    am = abs(m)
    coeffs = np.zeros(n + 1, dtype=float)
    for k in range((n - am) // 2 + 1):
        num = ((-1) ** k) * factorial(n - k)
        den = factorial(k) * factorial((n + am) // 2 - k) * factorial((n - am) // 2 - k)
        coeffs[n - 2 * k] = num / den
    return coeffs


def _polyval_asc(coeffs: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate ``sum_i coeffs[i] * x**i`` by Horner from the top."""
    out = np.zeros_like(x)
    for c in coeffs[::-1]:
        out = out * x + c
    return out


def _angular(m: int, theta: NDArray[np.float64]) -> NDArray[np.float64]:
    if m > 0:
        return np.cos(m * theta)
    if m < 0:
        return np.sin(-m * theta)
    return np.ones_like(theta)


def _d_angular(m: int, theta: NDArray[np.float64]) -> NDArray[np.float64]:
    if m > 0:
        return -m * np.sin(m * theta)
    if m < 0:
        return (-m) * np.cos(-m * theta)
    return np.zeros_like(theta)


def zernike_noll(j: int, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the Noll-indexed, Noll-normalised Zernike mode ``Z_j`` at ``(x, y)``.

    Parameters
    ----------
    j : int
        Noll index, 1-based.
    x, y : ndarray
        Cartesian pupil coordinates normalised so the pupil rim is at
        ``x**2 + y**2 = 1``. Dimensionless.

    Returns
    -------
    ndarray
        Mode values, dimensionless, orthonormal under the ``1/pi`` weight on
        the unit disc. Values outside ``rho <= 1`` are computed but are not
        meaningful (see module docstring).
    """
    j = _check_noll(j)
    n, m = noll_to_nm(j)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    rad = _polyval_asc(radial_coefficients(n, m), rho)
    return zernike_norm(n, m) * rad * _angular(m, theta)


def zernike_gradient_noll(
    j: int, x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Analytic Cartesian gradient ``(dZ_j/dx, dZ_j/dy)`` of a Noll mode.

    With ``Z = N R(rho) T(theta)``, ``rho = hypot(x, y)``,
    ``theta = atan2(y, x)``::

        dZ/dx = N [ R'(rho) T(theta) cos(theta) - (R(rho)/rho) T'(theta) sin(theta) ]
        dZ/dy = N [ R'(rho) T(theta) sin(theta) + (R(rho)/rho) T'(theta) cos(theta) ]

    The apparent ``1/rho`` singularity is removable: for ``|m| >= 1`` the radial
    polynomial is ``O(rho^|m|)``, so ``R/rho`` is itself a polynomial. This
    implementation divides the *coefficient array*, so ``rho = 0`` evaluates
    exactly with no epsilon fudge and no NaN; for ``m = 0`` the ``T'`` factor
    vanishes identically and the term is dropped before it is formed.

    Units: per unit normalised pupil radius (dimensionless input, dimensionless
    output). Divide by the physical pupil radius in metres for slope in
    wavefront-units per metre.
    """
    j = _check_noll(j)
    n, m = noll_to_nm(j)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    norm = zernike_norm(n, m)

    c = radial_coefficients(n, m)
    dc = c[1:] * np.arange(1, len(c))  # ascending coefficients of R'
    rad_d = _polyval_asc(dc, rho) if len(dc) else np.zeros_like(rho)
    ang = _angular(m, theta)

    dzdx = norm * rad_d * ang * np.cos(theta)
    dzdy = norm * rad_d * ang * np.sin(theta)

    if m != 0:
        # R(rho)/rho as a polynomial: shift the ascending coefficients down one.
        c_over = c[1:]
        rad_over = _polyval_asc(c_over, rho)
        ang_d = _d_angular(m, theta)
        dzdx -= norm * rad_over * ang_d * np.sin(theta)
        dzdy += norm * rad_over * ang_d * np.cos(theta)
    return dzdx, dzdy


def zernike_basis(
    j_list: list[int] | NDArray[np.int_],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Stack Zernike modes into a ``(n_points, n_modes)`` design matrix.

    ``x`` and ``y`` are flattened; the returned matrix has one column per Noll
    index in ``j_list``, in the given order. Dimensionless.
    """
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    if xs.shape != ys.shape:
        raise ValueError("x and y must have the same number of elements")
    js = [int(j) for j in np.asarray(j_list).ravel()]
    if len(js) == 0:
        raise ValueError("j_list must contain at least one Noll index")
    return np.column_stack([zernike_noll(j, xs, ys) for j in js])


def zernike_gradient_basis(
    j_list: list[int] | NDArray[np.int_],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(Gx, Gy)`` gradient design matrices, each ``(n_points, n_modes)``.

    Units: per unit normalised pupil radius. See :func:`zernike_gradient_noll`.
    """
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    if xs.shape != ys.shape:
        raise ValueError("x and y must have the same number of elements")
    js = [int(j) for j in np.asarray(j_list).ravel()]
    if len(js) == 0:
        raise ValueError("j_list must contain at least one Noll index")
    cols = [zernike_gradient_noll(j, xs, ys) for j in js]
    return (
        np.column_stack([c[0] for c in cols]),
        np.column_stack([c[1] for c in cols]),
    )
