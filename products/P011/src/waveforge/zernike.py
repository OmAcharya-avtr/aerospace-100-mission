"""Zernike polynomials in the Noll (1976) convention.

Conventions used throughout :mod:`waveforge` (state these before quoting any
coefficient — index and normalisation confusion is the classic silent bug):

* **Indexing:** Noll single index ``j``, starting at ``j = 1`` (piston).
  Within radial order ``n`` the modes occupy ``n(n+1)/2 + 1 <= j <=
  (n+1)(n+2)/2``; ``|m|`` increases within an order and, for ``m != 0``,
  **even ``j`` carries ``cos(m theta)`` (m > 0) and odd ``j`` carries
  ``sin(|m| theta)`` (m < 0)**.
* **Normalisation:** orthonormal on the unit disc with the area-normalised
  weight ``1/pi``, i.e. ``(1/pi) int Z_i Z_j rho drho dtheta = delta_ij``
  (Noll 1976, Eq. 3).  Consequently the RMS of a piston-free expansion is
  ``sqrt(sum_j a_j^2)`` directly.
* **Angle:** ``theta`` measured counter-clockwise from the ``+x`` axis.

Sources
-------
R. J. Noll, "Zernike polynomials and atmospheric turbulence", *J. Opt. Soc.
Am.* **66**(3), 207-211 (1976) — indexing, normalisation, Eqs. 2-3.
M. Born and E. Wolf, *Principles of Optics*, 7th ed., Cambridge University
Press (1999), Sec. 9.2 and Appendix VII — radial polynomials.

Units: the polynomials are dimensionless; coefficients carry whatever unit the
sampled wavefront carried (this package uses radians of phase).
Validity: the unit disc.  On an annulus the circle polynomials are **not**
orthogonal; that case needs Mahajan's annular polynomials (*JOSA* **71**,
75-85, 1981), which are not implemented here.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb, factorial

import numpy as np

__all__ = [
    "noll_indices",
    "noll_to_nm",
    "nm_to_noll",
    "radial_polynomial",
    "zernike_norm",
    "zernike_polar",
    "zernike_cartesian",
    "zernike_basis",
    "zernike_gradient_basis",
    "fit_zernike",
    "n_modes_for_order",
]


def n_modes_for_order(n_max: int) -> int:
    """Number of Noll modes up to and including radial order ``n_max``."""
    if int(n_max) != n_max or n_max < 0:
        raise ValueError(f"n_max must be a non-negative integer, got {n_max!r}")
    return (n_max + 1) * (n_max + 2) // 2


@lru_cache(maxsize=4096)
def noll_to_nm(j: int) -> tuple[int, int]:
    """Map a Noll index ``j >= 1`` to ``(n, m)``.

    Pure integer arithmetic, so there is no rounding failure at large ``j``.

    Examples (Noll 1976 Table 1): ``1 -> (0, 0)``, ``2 -> (1, 1)``,
    ``3 -> (1, -1)``, ``4 -> (2, 0)``, ``5 -> (2, -2)``, ``6 -> (2, 2)``.
    """
    if int(j) != j or j < 1:
        raise ValueError(f"Noll index j must be an integer >= 1, got {j!r}")
    j = int(j)
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    # Position within the radial order, 1-based.
    k = j - n * (n + 1) // 2
    # |m| values in ascending order, each non-zero value appearing twice.
    abs_m_sequence: list[int] = []
    mm = n % 2
    while mm <= n:
        if mm == 0:
            abs_m_sequence.append(0)
        else:
            abs_m_sequence.extend([mm, mm])
        mm += 2
    abs_m = abs_m_sequence[k - 1]
    if abs_m == 0:
        return n, 0
    # Noll parity rule: even j -> cosine (m > 0), odd j -> sine (m < 0).
    return n, abs_m if j % 2 == 0 else -abs_m


def nm_to_noll(n: int, m: int) -> int:
    """Inverse of :func:`noll_to_nm`.  Raises for invalid ``(n, m)`` pairs."""
    if int(n) != n or n < 0:
        raise ValueError(f"radial order n must be a non-negative integer, got {n!r}")
    if int(m) != m:
        raise ValueError(f"azimuthal order m must be an integer, got {m!r}")
    n, m = int(n), int(m)
    if abs(m) > n or (n - abs(m)) % 2 != 0:
        raise ValueError(f"invalid Zernike pair (n={n}, m={m}): need |m| <= n and n-|m| even")
    base = n * (n + 1) // 2
    for j in range(base + 1, base + n + 2):
        if noll_to_nm(j) == (n, m):
            return j
    raise ValueError(f"could not locate Noll index for (n={n}, m={m})")  # pragma: no cover


def noll_indices(j_max: int) -> list[tuple[int, int]]:
    """``[(n, m), ...]`` for Noll indices ``1 .. j_max``."""
    if int(j_max) != j_max or j_max < 1:
        raise ValueError(f"j_max must be an integer >= 1, got {j_max!r}")
    return [noll_to_nm(j) for j in range(1, int(j_max) + 1)]


@lru_cache(maxsize=1024)
def _radial_coefficients(n: int, m: int) -> tuple[tuple[int, float], ...]:
    """Return ``((power, coefficient), ...)`` of ``R_n^m(rho)``.

    Born & Wolf Sec. 9.2::

        R_n^m(rho) = sum_k (-1)^k (n-k)! rho^(n-2k)
                     / [ k! ((n+|m|)/2 - k)! ((n-|m|)/2 - k)! ]
    """
    m = abs(m)
    if (n - m) % 2 != 0:
        return ()
    out: list[tuple[int, float]] = []
    for k in range((n - m) // 2 + 1):
        c = (
            (-1) ** k
            * factorial(n - k)
            / (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))
        )
        out.append((n - 2 * k, float(c)))
    return tuple(out)


def radial_polynomial(n: int, m: int, rho: np.ndarray | float) -> np.ndarray:
    """Evaluate ``R_n^{|m|}(rho)``; zero if ``n - |m|`` is odd.

    ``rho`` is the normalised pupil radius (dimensionless).  ``R_n^m(1) = 1``
    and ``|R_n^m| <= 1`` on ``0 <= rho <= 1``; values outside the disc are
    evaluated by the same polynomial and grow without bound, so callers must
    mask.
    """
    rho_arr = np.asarray(rho, dtype=float)
    coeffs = _radial_coefficients(int(n), int(m))
    out = np.zeros_like(rho_arr, dtype=float)
    for power, c in coeffs:
        out = out + c * rho_arr**power
    return out


def zernike_norm(n: int, m: int) -> float:
    """Noll normalisation factor ``sqrt(2(n+1))`` (``m != 0``) or ``sqrt(n+1)``."""
    return float(np.sqrt(2 * (n + 1))) if m != 0 else float(np.sqrt(n + 1))


def zernike_polar(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Orthonormal Noll mode ``Z_j(rho, theta)`` — dimensionless."""
    n, m = noll_to_nm(j)
    rad = radial_polynomial(n, m, rho)
    nrm = zernike_norm(n, m)
    if m > 0:
        return nrm * rad * np.cos(m * np.asarray(theta, dtype=float))
    if m < 0:
        return nrm * rad * np.sin(abs(m) * np.asarray(theta, dtype=float))
    return nrm * rad


def zernike_cartesian(j: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Orthonormal Noll mode evaluated at normalised Cartesian ``(x, y)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return zernike_polar(j, np.hypot(x, y), np.arctan2(y, x))


def zernike_basis(
    j_max: int,
    rho: np.ndarray,
    theta: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    include_piston: bool = True,
) -> np.ndarray:
    """Stack of Noll modes with shape ``(n_modes, n_samples)``.

    ``rho``/``theta`` may be any shape; they are flattened.  When ``mask`` is
    given only masked samples are returned.  ``include_piston=False`` drops
    ``j = 1`` (the usual choice, since piston is unobservable).
    """
    rho = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if rho.shape != theta.shape:
        raise ValueError(f"rho shape {rho.shape} != theta shape {theta.shape}")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != rho.shape:
            raise ValueError(f"mask shape {mask.shape} != rho shape {rho.shape}")
        rho, theta = rho[mask], theta[mask]
    else:
        rho, theta = rho.ravel(), theta.ravel()
    first = 1 if include_piston else 2
    if int(j_max) != j_max or j_max < first:
        raise ValueError(f"j_max must be an integer >= {first}, got {j_max!r}")
    return np.stack([zernike_polar(j, rho, theta) for j in range(first, int(j_max) + 1)])


def _radial_derivative(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """``dR_n^{|m|}/drho`` evaluated term by term (exact, no finite differences)."""
    out = np.zeros_like(rho, dtype=float)
    for power, c in _radial_coefficients(int(n), int(m)):
        if power == 0:
            continue
        out = out + c * power * rho ** (power - 1)
    return out


def zernike_gradient_basis(
    j_max: int,
    x: np.ndarray,
    y: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    include_piston: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form Cartesian gradients ``(dZ/dx, dZ/dy)`` in normalised coords.

    Each returned array has shape ``(n_modes, n_samples)`` and units of "mode
    amplitude per unit normalised pupil radius".  To convert to a physical
    gradient in rad/m, divide by the pupil radius ``R``.

    The ``1/rho`` singularity of the polar chain rule is removed analytically:
    the azimuthal term is ``(m / rho) * R_n^m(rho)`` and ``R_n^m`` has a factor
    ``rho^{|m|}`` with ``|m| >= 1`` whenever the azimuthal term is non-zero, so
    the quotient is a polynomial.  It is evaluated as such, giving exact values
    at ``rho = 0`` and no ``nan``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x shape {x.shape} != y shape {y.shape}")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != x.shape:
            raise ValueError(f"mask shape {mask.shape} != x shape {x.shape}")
        x, y = x[mask], y[mask]
    else:
        x, y = x.ravel(), y.ravel()
    first = 1 if include_piston else 2
    if int(j_max) != j_max or j_max < first:
        raise ValueError(f"j_max must be an integer >= {first}, got {j_max!r}")

    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    safe = np.where(rho > 0.0, rho, 1.0)
    cos_t = np.where(rho > 0.0, x / safe, 1.0)
    sin_t = np.where(rho > 0.0, y / safe, 0.0)

    dzdx_rows, dzdy_rows = [], []
    for j in range(first, int(j_max) + 1):
        n, m = noll_to_nm(j)
        nrm = zernike_norm(n, m)
        drad = _radial_derivative(n, m, rho)
        am = abs(m)
        if m > 0:
            ang, dang = np.cos(m * theta), -m * np.sin(m * theta)
        elif m < 0:
            ang, dang = np.sin(am * theta), am * np.cos(am * theta)
        else:
            ang, dang = np.ones_like(rho), np.zeros_like(rho)
        # radial-over-rho, evaluated as a polynomial (no division by rho)
        if m == 0:
            rad_over_rho = np.zeros_like(rho)
        else:
            rad_over_rho = np.zeros_like(rho)
            for power, c in _radial_coefficients(n, m):
                rad_over_rho = rad_over_rho + c * rho ** (power - 1)
        d_rho = nrm * drad * ang
        d_theta_over_rho = nrm * rad_over_rho * dang
        dzdx_rows.append(d_rho * cos_t - d_theta_over_rho * sin_t)
        dzdy_rows.append(d_rho * sin_t + d_theta_over_rho * cos_t)
    return np.stack(dzdx_rows), np.stack(dzdy_rows)


def fit_zernike(
    phase: np.ndarray,
    rho: np.ndarray,
    theta: np.ndarray,
    j_max: int,
    *,
    mask: np.ndarray | None = None,
    include_piston: bool = True,
    rcond: float | None = None,
) -> np.ndarray:
    """Least-squares fit of a sampled wavefront to Noll coefficients.

    Returns coefficients for ``j = 1 .. j_max`` (or ``2 .. j_max`` when
    ``include_piston`` is ``False``) in the same unit as ``phase``.

    The fit is a discrete least-squares projection, not the exact continuous
    inner product; on a finite grid the two differ by the sampling error of the
    orthonormality integral (quantified in ``validation/``).
    """
    basis = zernike_basis(j_max, rho, theta, mask=mask, include_piston=include_piston)
    phase = np.asarray(phase, dtype=float)
    values = phase[np.asarray(mask, dtype=bool)] if mask is not None else phase.ravel()
    if values.size != basis.shape[1]:
        raise ValueError(f"phase has {values.size} samples but basis has {basis.shape[1]}")
    coeffs, *_ = np.linalg.lstsq(basis.T, values, rcond=rcond)
    return coeffs


def zernike_mode_count_check(j_max: int) -> int:
    """Number of complete radial orders contained in ``j_max`` modes.

    Helper used by the CLI and tests; ``comb`` import keeps the intent explicit.
    """
    n = 0
    while comb(n + 2, 2) <= j_max:
        n += 1
    return n - 1
