"""Zernike polynomials in Noll ordering, and Kolmogorov modal statistics.

Definitions follow Noll, R. J. (1976), "Zernike polynomials and atmospheric
turbulence", *Journal of the Optical Society of America* **66**, 207-211.

Normalisation (Noll 1976, eqs. 1-4). With ``rho`` in ``[0, 1]`` and ``theta``
in radians,

```
Z_even j (rho, theta) = sqrt(n+1) R_n^m(rho) * sqrt(2) cos(m theta)     m != 0
Z_odd  j (rho, theta) = sqrt(n+1) R_n^m(rho) * sqrt(2) sin(m theta)     m != 0
Z_j    (rho, theta)   = sqrt(n+1) R_n^0(rho)                            m == 0
```

with the radial polynomial

```
R_n^m(rho) = sum_{s=0}^{(n-m)/2}
             (-1)^s (n-s)! rho^(n-2s)
             / [ s! ((n+m)/2 - s)! ((n-m)/2 - s)! ]
```

This normalisation makes the polynomials orthonormal on the **unit circle**
with unit weight:

```
(1/pi) * integral_{rho<=1} Z_i Z_j dA = delta_ij           [-]
```

so a wavefront expanded as ``phi = sum_j a_j Z_j`` has piston-removed variance
``sum_{j>=2} a_j^2`` exactly. Units: ``Z_j`` is dimensionless, so ``a_j`` has
the units of ``phi`` (radians of optical phase here).

*Validity:* the closed unit disc only. On an annular pupil the polynomials are
**not** orthogonal; :func:`orthonormality_matrix` measures the actual Gram
matrix of the sampled basis so the user can see the error, and
:class:`ZernikeBasis` orthonormalises numerically when asked.

Kolmogorov statistics
---------------------
For Kolmogorov turbulence over a circular aperture of diameter ``D`` with Fried
parameter ``r0`` (both [m], at the same wavelength), Noll's Table IV gives the
mean-square residual phase [rad^2] left after the first ``J`` Zernike terms
(``J = 1`` = piston) are removed:

```
Delta_J = NOLL_RESIDUALS[J] * (D/r0)^(5/3)
```

and for large ``J`` the asymptotic form (Noll 1976, eq. 34)

```
Delta_J ~= 0.2944 J^(-sqrt(3)/2) (D/r0)^(5/3)
```

*Assumptions:* Kolmogorov spectrum (infinite outer scale), near-field, weak
scintillation (phase-only), unobscured circular aperture.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from math import factorial, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import IntegrationWarning, quad
from scipy.special import gammaln, jv

__all__ = [
    "NOLL_RESIDUALS",
    "NOLL_MODE_VARIANCE_COEFF",
    "ZernikeBasis",
    "noll_to_nm",
    "nm_to_noll",
    "radial_polynomial",
    "zernike",
    "noll_residual",
    "noll_mode_variance",
    "orthonormality_matrix",
    "zernike_filter",
    "kolmogorov_mode_variance",
    "kolmogorov_residual_variance",
]

# Noll (1976) Table IV: residual mean-square phase after removing the first J
# Zernike terms, in units of (D/r0)^(5/3) [rad^2]. Index J = 1 (piston only)
# through J = 21. Reproduced verbatim from the published table.
NOLL_RESIDUALS: dict[int, float] = {
    1: 1.0299,
    2: 0.582,
    3: 0.134,
    4: 0.111,
    5: 0.0880,
    6: 0.0648,
    7: 0.0587,
    8: 0.0525,
    9: 0.0463,
    10: 0.0401,
    11: 0.0377,
    12: 0.0352,
    13: 0.0328,
    14: 0.0304,
    15: 0.0279,
    16: 0.0267,
    17: 0.0255,
    18: 0.0243,
    19: 0.0232,
    20: 0.0220,
    21: 0.0208,
}

# Prefactor of the single-mode variance law
#     <a_j^2> = C (n+1) Gamma(n-5/6)/Gamma(n+23/6) (D/r0)^(5/3).
# The n-dependence is Noll's (1976, sec. III); the prefactor C here is *not*
# quoted from the paper -- it is fixed by requiring the sum over all modes with
# n >= 1 to equal Noll's tabulated Delta_1 = 1.0299 (D/r0)^(5/3). Its agreement
# with the differences of Noll's Table IV is measured in
# validation/validate_zernike.py (worst case < 1.5%, dominated by the 3-figure
# rounding of the published table).
NOLL_MODE_VARIANCE_COEFF: float = 0.753383


@lru_cache(maxsize=4096)
def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert a Noll index ``j >= 1`` to ``(n, m)`` with the Noll sign convention.

    Returns ``(n, m)`` where ``n`` is the radial order and ``m >= 0`` the
    azimuthal order; the cosine/sine choice is recovered by
    :func:`noll_is_cosine`. Noll orders modes by increasing ``n``, and within
    a radial order by increasing ``m``, with even ``j`` taking the cosine term.
    """
    j = int(j)
    if j < 1:
        raise ValueError(f"Noll index j must be >= 1, got {j}")
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    # Index of j within radial order n, 0-based.
    k = j - n * (n + 1) // 2 - 1
    # Allowed |m| for this n, in Noll's order: n%2, n%2+2, ... each appearing
    # twice (cos, sin) except m = 0.
    ms: list[int] = []
    m = n % 2
    while m <= n:
        if m == 0:
            ms.append(0)
        else:
            ms.extend([m, m])
        m += 2
    return n, ms[k]


def noll_is_cosine(j: int) -> bool:
    """True if Noll mode ``j`` carries ``cos(m theta)`` (or is an ``m = 0`` mode)."""
    n, m = noll_to_nm(j)
    if m == 0:
        return True
    return j % 2 == 0


def nm_to_noll(n: int, m: int, cosine: bool = True) -> int:
    """Inverse of :func:`noll_to_nm`. ``m >= 0``; ``cosine`` selects cos vs sin."""
    n = int(n)
    m = int(m)
    if n < 0:
        raise ValueError(f"radial order n must be >= 0, got {n}")
    if m < 0 or m > n or (n - m) % 2 != 0:
        raise ValueError(f"invalid (n, m) = ({n}, {m}): need 0 <= m <= n and n-m even")
    j0 = n * (n + 1) // 2 + 1
    for j in range(j0, j0 + n + 1):
        nn, mm = noll_to_nm(j)
        if nn == n and mm == m and noll_is_cosine(j) == (cosine or m == 0):
            return j
    raise ValueError(f"no Noll index for (n, m) = ({n}, {m})")  # pragma: no cover


def radial_polynomial(n: int, m: int, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Zernike radial polynomial ``R_n^m(rho)`` [-] (Noll 1976, eq. 2).

    Parameters
    ----------
    n, m:
        Radial and azimuthal order, ``0 <= m <= n`` with ``n - m`` even.
    rho:
        Normalised radius [-], any shape. Values outside ``[0, 1]`` are
        evaluated by the same polynomial (they are not clipped); callers should
        mask them.
    """
    n = int(n)
    m = int(m)
    if m < 0 or m > n or (n - m) % 2 != 0:
        raise ValueError(f"invalid (n, m) = ({n}, {m}): need 0 <= m <= n and n-m even")
    r = np.asarray(rho, dtype=np.float64)
    out = np.zeros_like(r)
    for s in range((n - m) // 2 + 1):
        coeff = (
            (-1) ** s
            * factorial(n - s)
            / (factorial(s) * factorial((n + m) // 2 - s) * factorial((n - m) // 2 - s))
        )
        out += coeff * r ** (n - 2 * s)
    return out


def zernike(j: int, rho: NDArray[np.float64], theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Noll-normalised Zernike polynomial ``Z_j(rho, theta)`` [-].

    Parameters
    ----------
    j:
        Noll index, ``j >= 1`` (1 = piston, 2/3 = tip/tilt, 4 = defocus).
    rho:
        Normalised radius [-].
    theta:
        Azimuth [rad].

    Notes
    -----
    Normalised so that ``(1/pi) * int_{rho<=1} Z_i Z_j dA = delta_ij``
    (Noll 1976, eq. 4). Piston is ``Z_1 = 1``.
    """
    n, m = noll_to_nm(j)
    r = np.asarray(rho, dtype=np.float64)
    t = np.asarray(theta, dtype=np.float64)
    rad = sqrt(n + 1) * radial_polynomial(n, m, r)
    if m == 0:
        return rad
    ang = np.cos(m * t) if noll_is_cosine(j) else np.sin(m * t)
    return sqrt(2.0) * rad * ang


def noll_residual(n_modes: int) -> float:
    """Residual phase variance after removing ``n_modes`` Zernike terms.

    Parameters
    ----------
    n_modes:
        Number of Noll terms removed, ``J >= 1`` (``J = 1`` is piston only).

    Returns
    -------
    float
        ``Delta_J / (D/r0)^(5/3)`` [rad^2 per (D/r0)^(5/3)].

    Notes
    -----
    ``J <= 21`` returns Noll (1976) Table IV verbatim. Above 21 the published
    asymptotic form ``Delta_J = 0.2944 J^(-sqrt(3)/2)`` (Noll 1976, eq. 34) is
    used; Noll states it is accurate to a few per cent for large ``J``, and the
    error at ``J = 21`` measured against the table is reported in
    validation/validate_zernike.py.
    """
    j = int(n_modes)
    if j < 1:
        raise ValueError(f"n_modes must be >= 1, got {j}")
    if j in NOLL_RESIDUALS:
        return NOLL_RESIDUALS[j]
    return 0.2944 * j ** (-sqrt(3.0) / 2.0)


def noll_mode_variance(j: int) -> float:
    """Variance of a single Zernike coefficient under Kolmogorov statistics.

    Parameters
    ----------
    j:
        Noll index, ``j >= 2`` (piston has no defined variance -- it diverges
        for an infinite outer scale).

    Returns
    -------
    float
        ``<a_j^2> / (D/r0)^(5/3)`` [rad^2 per (D/r0)^(5/3)].

    Notes
    -----
    ``<a_j^2> = C (n+1) Gamma(n-5/6)/Gamma(n+23/6) (D/r0)^(5/3)``: modes of the
    same radial order ``n`` share a variance. See
    :data:`NOLL_MODE_VARIANCE_COEFF` for the provenance of ``C``.
    """
    j = int(j)
    if j < 2:
        raise ValueError(f"j must be >= 2 (piston variance is undefined), got {j}")
    n, _ = noll_to_nm(j)
    log_ratio = gammaln(n - 5.0 / 6.0) - gammaln(n + 23.0 / 6.0)
    return float(NOLL_MODE_VARIANCE_COEFF * (n + 1) * np.exp(log_ratio))


class ZernikeBasis:
    """Sampled Zernike basis on a :class:`~waveforge.pupil.Pupil`.

    Parameters
    ----------
    pupil:
        The pupil grid to sample on.
    n_modes:
        Number of Noll terms, starting at ``j = 1`` (piston). Must be >= 1.
    orthonormalize:
        If True, replace the sampled basis by its QR-orthonormalisation over the
        masked samples. This is what makes the basis usable on an obscured
        pupil, where the analytic polynomials are not orthogonal. Default False
        (the analytic polynomials are kept).

    Attributes
    ----------
    matrix:
        ``(n_valid, n_modes)`` array of basis values at the masked samples [-].
    """

    def __init__(self, pupil, n_modes: int, orthonormalize: bool = False) -> None:
        n_modes = int(n_modes)
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}")
        self.pupil = pupil
        self.n_modes = n_modes
        self.orthonormalized = bool(orthonormalize)

        rho, theta = pupil.polar()
        mask = pupil.mask
        cols = np.empty((int(np.count_nonzero(mask)), n_modes), dtype=np.float64)
        rr = rho[mask]
        tt = theta[mask]
        for k in range(n_modes):
            cols[:, k] = zernike(k + 1, rr, tt)
        if orthonormalize:
            q, r = np.linalg.qr(cols)
            # Fix sign so each orthonormal mode keeps the polarity of its parent.
            signs = np.sign(np.diag(r))
            signs[signs == 0] = 1.0
            cols = q * signs * sqrt(cols.shape[0])
        self.matrix: NDArray[np.float64] = cols
        # Pseudo-inverse for projection; cached.
        self._pinv: NDArray[np.float64] | None = None

    # --------------------------------------------------------------- projection
    @property
    def pinv(self) -> NDArray[np.float64]:
        """Least-squares projection matrix, shape ``(n_modes, n_valid)``."""
        if self._pinv is None:
            self._pinv = np.linalg.pinv(self.matrix)
        return self._pinv

    def to_phase(self, coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
        """Expand modal coefficients into a phase map [same units as coefficients]."""
        a = np.asarray(coefficients, dtype=np.float64)
        if a.shape != (self.n_modes,):
            raise ValueError(f"coefficients must have shape {(self.n_modes,)}, got {a.shape}")
        out = np.zeros((self.pupil.n_grid, self.pupil.n_grid), dtype=np.float64)
        out[self.pupil.mask] = self.matrix @ a
        return out

    def project(self, phase: NDArray[np.float64]) -> NDArray[np.float64]:
        """Least-squares modal coefficients of ``phase`` [same units as phase]."""
        arr = np.asarray(phase, dtype=np.float64)
        if arr.shape != (self.pupil.n_grid, self.pupil.n_grid):
            raise ValueError(
                f"phase must have shape {(self.pupil.n_grid,) * 2}, got {arr.shape}"
            )
        return self.pinv @ arr[self.pupil.mask]

    def residual(self, phase: NDArray[np.float64]) -> NDArray[np.float64]:
        """``phase`` minus its projection onto the basis [same units as phase]."""
        arr = np.asarray(phase, dtype=np.float64)
        out = np.zeros_like(arr)
        vals = arr[self.pupil.mask]
        out[self.pupil.mask] = vals - self.matrix @ (self.pinv @ vals)
        return out


def orthonormality_matrix(pupil, n_modes: int) -> NDArray[np.float64]:
    """Discrete Gram matrix ``G_ij = <Z_i Z_j>`` over the masked pupil samples.

    For an unobscured pupil sampled finely enough this approaches the identity
    (Noll 1976, eq. 4). The departure from the identity is the discretisation
    plus edge-pixel error and is quantified in
    validation/validate_zernike.py.
    """
    basis = ZernikeBasis(pupil, n_modes)
    m = basis.matrix
    return (m.T @ m) / m.shape[0]


# ---------------------------------------------------------------------------
# Kolmogorov statistics by direct integration of the Zernike filter functions
# ---------------------------------------------------------------------------

def zernike_filter(j: int, f: NDArray[np.float64], diameter: float) -> NDArray[np.float64]:
    """Azimuthally averaged squared Zernike spatial filter ``|F_j(f)|^2`` [-].

    The Fourier transform of the Noll-normalised Zernike ``Z_j`` over a
    circular pupil of diameter ``D``, normalised by the pupil area, has modulus

    ```
    |F_j(f)| = sqrt(n+1) * 2 J_(n+1)(pi D f) / (pi D f)
    ```

    times an azimuthal factor (``sqrt(2) cos(m theta)`` or ``sqrt(2) sin(m theta)``
    for ``m != 0``, unity for ``m = 0``) whose mean square over azimuth is 1.
    This is Noll (1976) eq. 8 written for spatial frequency ``f`` in cycles per
    metre. Only the radial order ``n`` enters, so all modes of one radial order
    share a filter -- and hence, for an isotropic spectrum, a variance.

    Parameters
    ----------
    j:
        Noll index, >= 1.
    f:
        Spatial frequency [cycles/m]. ``f = 0`` returns the limit
        ``n+1`` for ``n = 0`` (piston) and 0 otherwise.
    diameter:
        Pupil diameter ``D`` [m], > 0.

    Notes
    -----
    Verified two independent ways in ``validation/validate_zernike.py``:
    against the FFT of the sampled polynomial, and by reproducing Noll's
    Table IV residuals through :func:`kolmogorov_residual_variance`.
    """
    n, _ = noll_to_nm(j)
    d = float(diameter)
    if not np.isfinite(d) or d <= 0:
        raise ValueError(f"diameter must be > 0, got {diameter!r}")
    freq = np.atleast_1d(np.asarray(f, dtype=np.float64))
    if np.any(freq < 0):
        raise ValueError("spatial frequency must be >= 0")
    u = np.pi * d * freq
    out = np.zeros_like(u)
    small = u < 1.0e-8
    out[~small] = (n + 1) * (2.0 * jv(n + 1, u[~small]) / u[~small]) ** 2
    out[small] = float(n + 1) if n == 0 else 0.0
    return out.reshape(np.shape(f)) if np.ndim(f) else float(out[0])


def _kolmogorov_integral(filter_fn, diameter: float, r0: float, psd_coeff: float) -> float:
    """``int_0^inf 0.023 r0^-5/3 f^-11/3 * filter(f) * 2 pi f df`` [rad^2]."""
    d = float(diameter)
    r = float(r0)
    if not np.isfinite(d) or d <= 0:
        raise ValueError(f"diameter must be > 0, got {diameter!r}")
    if not np.isfinite(r) or r <= 0:
        raise ValueError(f"r0 must be > 0, got {r0!r}")

    def integrand(f: float) -> float:
        return psd_coeff * r ** (-5.0 / 3.0) * f ** (-11.0 / 3.0) * filter_fn(f) * 2.0 * np.pi * f

    # The integrand oscillates like a Bessel function above f ~ 1/D, so the
    # range is split into many panels; below 1/D it is a smooth power law.
    lo = np.logspace(-6, -0.5, 40) / d
    mid = np.linspace(10 ** -0.5 / d, 60.0 / d, 4000)
    hi = np.logspace(np.log10(60.0 / d), 4.0 / d + 4.0, 200)
    edges = np.concatenate([lo, mid, hi])
    total = 0.0
    with warnings.catch_warnings():
        # Panel-wise adaptive quadrature on an oscillatory Bessel integrand:
        # individual panels can report slow convergence while the panel sum is
        # accurate to <1% (measured against Noll's Table IV in validation).
        warnings.simplefilter("ignore", IntegrationWarning)
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            total += quad(integrand, a, b, limit=200)[0]
    return float(total)


def kolmogorov_mode_variance(
    j: int, diameter: float, r0: float, psd_coeff: float = 0.023
) -> float:
    """``<a_j^2>`` [rad^2] by integrating the Kolmogorov PSD against ``|F_j|^2``.

    Parameters
    ----------
    j:
        Noll index, >= 2 (the piston variance diverges for an infinite outer
        scale).
    diameter, r0:
        Pupil diameter and Fried parameter [m], at the same wavelength.
    psd_coeff:
        Kolmogorov PSD coefficient [-]. Default 0.023 (Roddier 1981 eq. 3.42);
        the value exactly consistent with ``D_phi = 6.88 (r/r0)^(5/3)`` is
        0.022919, a 0.35 % difference that is reported in
        ``validation/validate_atmosphere.py``.
    """
    if int(j) < 2:
        raise ValueError(f"j must be >= 2 (piston variance diverges), got {j}")
    return _kolmogorov_integral(
        lambda f: float(zernike_filter(j, f, diameter)), diameter, r0, psd_coeff
    )


def kolmogorov_residual_variance(
    n_modes: int, diameter: float, r0: float, psd_coeff: float = 0.023
) -> float:
    """Residual variance ``Delta_J`` [rad^2] after removing ``J`` Zernike terms.

    Computed as a single integral of the *residual* filter
    ``1 - sum_{j<=J} |F_j(f)|^2`` against the Kolmogorov PSD, which avoids the
    cancellation error of summing and subtracting individual mode variances.
    Compare with :func:`noll_residual` (Noll's published table): the two agree
    to better than 1 % for ``J = 1..21``, measured in
    ``validation/validate_zernike.py``.
    """
    jmax = int(n_modes)
    if jmax < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}")
    orders = [noll_to_nm(j)[0] for j in range(1, jmax + 1)]

    def residual(f: float) -> float:
        u = np.pi * diameter * f
        if u < 1.0e-8:
            return 0.0
        s = sum((n + 1) * (2.0 * jv(n + 1, u) / u) ** 2 for n in orders)
        return max(1.0 - s, 0.0)

    return _kolmogorov_integral(residual, diameter, r0, psd_coeff)
