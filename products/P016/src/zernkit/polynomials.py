"""Zernike radial polynomials and full Zernike modes on the unit disc.

Definitions and conventions (stated explicitly; see README "Engineering theory")
-------------------------------------------------------------------------------
Source for the polynomial definition: M. Born and E. Wolf, *Principles of
Optics*, 7th (expanded) edition, Cambridge University Press 1999, Sec. 9.2 and
Appendix VII. Normalisation follows R. J. Noll, JOSA **66**(3), 207-211 (1976),
which is the same normalisation adopted by ANSI Z80.28 / Thibos et al. (2002).

Radial polynomial, for ``n - |m|`` even (it is identically zero otherwise)::

    R_n^m(rho) = sum_{k=0}^{(n-|m|)/2} (-1)^k (n-k)!
                 -----------------------------------------------------  rho^(n-2k)
                 k! ((n+|m|)/2 - k)! ((n-|m|)/2 - k)!

Properties used as tests: ``R_n^m(1) = 1`` for every legal ``(n, m)``;
``R_n^m`` contains only powers of ``rho`` with the parity of ``n``.

Full mode, **unnormalised** form (peak value 1 on the rim for the cosine part)::

    Z_n^m(rho, theta) = R_n^|m|(rho) * cos(m*theta)      m > 0
                      = R_n^|m|(rho) * sin(|m|*theta)    m < 0
                      = R_n^0(rho)                       m = 0

Full mode, **normalised** (orthonormal) form used by Noll and by ANSI::

    Z_n^m = N_n^m * (unnormalised form),
    N_n^m = sqrt(2(n+1))  for m != 0,      N_n^0 = sqrt(n+1)

with the orthonormality relation taken over the unit disc with the
**area-normalised** weight ``W = 1/pi``::

    (1/pi) * int_0^{2pi} int_0^1 Z_i(rho,theta) Z_j(rho,theta) rho drho dtheta
        = delta_ij

so a coefficient vector ``a`` in this basis has RMS wavefront
``sqrt(sum_{j != piston} a_j^2)`` directly, in whatever unit the wavefront is
given (waves, radians, or metres -- the library is unit-agnostic and never
converts).

Validity range: the expansion is defined on the unit disc ``0 <= rho <= 1``.
The polynomials are finite outside it, but they are neither orthogonal nor
physically meaningful there and they diverge rapidly; see
:mod:`zernkit.fitting` for the sampling policy.

Obscured (annular) pupils are **not** supported: the Zernike circle
polynomials are not orthogonal on an annulus, which requires the annular
polynomials of V. N. Mahajan, JOSA **71**, 75-85 (1981).
"""

from __future__ import annotations

from math import factorial

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .indexing import noll_to_nm, osa_to_nm, validate_nm

__all__ = [
    "normalization",
    "radial_coefficients",
    "radial_polynomial",
    "azimuthal_factor",
    "zernike",
    "zernike_cartesian",
    "zernike_noll",
    "zernike_osa",
    "unit_disc_grid",
]


def normalization(n: int, m: int, normalized: bool = True) -> float:
    """Noll/ANSI orthonormalisation factor ``N_n^m`` (dimensionless).

    ``N_n^m = sqrt(2(n+1))`` for ``m != 0`` and ``sqrt(n+1)`` for ``m = 0``
    (Noll 1976, Eq. 2). Returns ``1.0`` when ``normalized`` is False, i.e. the
    Born & Wolf unnormalised convention in which the mode peaks at 1.

    Parameters
    ----------
    n, m : int
        Zernike indices.
    normalized : bool, optional
        If False return 1.0 (unnormalised convention).

    Returns
    -------
    float
        The multiplicative normalisation constant.
    """
    validate_nm(n, m)
    if not normalized:
        return 1.0
    return float(np.sqrt(2 * (n + 1))) if m != 0 else float(np.sqrt(n + 1))


def radial_coefficients(n: int, m: int) -> NDArray[np.float64]:
    """Coefficients of ``R_n^m`` in ascending powers of ``rho``.

    Returns an array ``c`` of length ``n + 1`` such that
    ``R_n^m(rho) = sum_p c[p] * rho**p``. Entries are exact rational values of
    the Born & Wolf sum (computed with Python integer factorials, then cast to
    float), so ``c[p]`` is nonzero only for ``p >= |m|`` with ``p`` of the same
    parity as ``n``.

    Parameters
    ----------
    n, m : int
        Zernike indices; ``n - |m|`` must be even.

    Returns
    -------
    numpy.ndarray
        Float64 coefficient array, index = power of ``rho``.
    """
    validate_nm(n, m)
    mm = abs(m)
    coeffs = np.zeros(n + 1, dtype=np.float64)
    for k in range((n - mm) // 2 + 1):
        num = factorial(n - k)
        den = factorial(k) * factorial((n + mm) // 2 - k) * factorial((n - mm) // 2 - k)
        coeffs[n - 2 * k] = (-1) ** k * (num / den)
    return coeffs


def radial_polynomial(n: int, m: int, rho: ArrayLike) -> NDArray[np.float64]:
    """Evaluate ``R_n^m(rho)`` (dimensionless) at radial coordinate ``rho``.

    ``rho`` is the pupil radius normalised to the pupil edge; the physical
    aperture radius never enters. Values with ``rho > 1`` are evaluated as
    polynomials without complaint -- see the module docstring for why that is
    not a wavefront.

    Parameters
    ----------
    n, m : int
        Zernike indices.
    rho : array_like
        Normalised radial coordinate (dimensionless).

    Returns
    -------
    numpy.ndarray
        ``R_n^m(rho)``, same shape as ``rho``.
    """
    rho_arr = np.asarray(rho, dtype=np.float64)
    if np.any(rho_arr < 0):
        raise ValueError("rho must be non-negative; got at least one negative radius")
    coeffs = radial_coefficients(n, m)
    # np.polyval expects descending powers.
    return np.polyval(coeffs[::-1], rho_arr)


def azimuthal_factor(m: int, theta: ArrayLike) -> NDArray[np.float64]:
    """Angular part of a Zernike mode (dimensionless).

    ``cos(m*theta)`` for ``m > 0``, ``sin(|m|*theta)`` for ``m < 0``, ``1`` for
    ``m = 0``. ``theta`` is in **radians**, measured counter-clockwise from the
    ``+x`` axis.
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    if m == 0:
        return np.ones_like(theta_arr)
    if m > 0:
        return np.cos(m * theta_arr)
    return np.sin(-m * theta_arr)


def zernike(
    n: int,
    m: int,
    rho: ArrayLike,
    theta: ArrayLike,
    normalized: bool = True,
) -> NDArray[np.float64]:
    """Evaluate the Zernike mode ``Z_n^m(rho, theta)``.

    Parameters
    ----------
    n, m : int
        Zernike indices; ``n - |m|`` must be even.
    rho : array_like
        Normalised radial coordinate, dimensionless (unit disc: ``0..1``).
    theta : array_like
        Azimuth in radians, counter-clockwise from ``+x``.
    normalized : bool, optional
        True (default) applies the Noll/ANSI factor ``N_n^m`` so the mode is
        orthonormal under the ``1/pi`` area weight; False gives the Born & Wolf
        unnormalised mode with unit peak.

    Returns
    -------
    numpy.ndarray
        Mode values, broadcast over ``rho`` and ``theta``. Dimensionless; the
        wavefront unit is whatever unit its coefficient carries.
    """
    validate_nm(n, m)
    return (
        normalization(n, m, normalized)
        * radial_polynomial(n, m, rho)
        * azimuthal_factor(m, theta)
    )


def zernike_cartesian(
    n: int,
    m: int,
    x: ArrayLike,
    y: ArrayLike,
    normalized: bool = True,
) -> NDArray[np.float64]:
    """Evaluate ``Z_n^m`` on Cartesian pupil coordinates normalised to the pupil radius.

    ``x`` and ``y`` are dimensionless: the pupil edge is at ``x^2 + y^2 = 1``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    rho = np.hypot(x_arr, y_arr)
    theta = np.arctan2(y_arr, x_arr)
    return zernike(n, m, rho, theta, normalized=normalized)


def zernike_noll(
    j: int, rho: ArrayLike, theta: ArrayLike, normalized: bool = True
) -> NDArray[np.float64]:
    """Evaluate the Noll-indexed mode ``Z_j`` (``j >= 1``, ``j = 1`` is piston)."""
    n, m = noll_to_nm(j)
    return zernike(n, m, rho, theta, normalized=normalized)


def zernike_osa(
    j: int, rho: ArrayLike, theta: ArrayLike, normalized: bool = True
) -> NDArray[np.float64]:
    """Evaluate the OSA/ANSI-indexed mode ``Z_j`` (``j >= 0``, ``j = 0`` is piston)."""
    n, m = osa_to_nm(j)
    return zernike(n, m, rho, theta, normalized=normalized)


def unit_disc_grid(
    n_pix: int, include_edge: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Square Cartesian sampling grid covering the unit disc.

    Parameters
    ----------
    n_pix : int
        Number of samples across the full pupil diameter (``n_pix >= 2``).
    include_edge : bool, optional
        If True (default) the mask keeps points with ``rho <= 1``; if False the
        strict interior ``rho < 1`` is kept.

    Returns
    -------
    x, y : numpy.ndarray
        ``(n_pix, n_pix)`` coordinate arrays spanning ``[-1, 1]``,
        dimensionless (normalised to pupil radius).
    mask : numpy.ndarray of bool
        True inside the unit disc.
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
