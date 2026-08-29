"""Covariance housekeeping shared by every estimator in NavBench.

The dominant practical failure of a Kalman filter is not a wrong equation but a
covariance matrix that quietly stops being one: it loses symmetry, then positive
definiteness, and the estimate follows. Every routine here exists to make that
failure *visible* rather than silent.

References
----------
* Bar-Shalom, Y., Rong Li, X. and Kirubarajan, T. (2001), *Estimation with
  Applications to Tracking and Navigation*, Wiley, §5.2 (Joseph form).
* Simon, D. (2006), *Optimal State Estimation*, Wiley, §6.3.
* Bierman, G. J. (1977), *Factorization Methods for Discrete Sequential
  Estimation*, Academic Press — the reference for when Joseph form is not
  enough and square-root filtering is required.

Units: unit-agnostic. Covariance entries carry squared state units.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "symmetrize",
    "joseph_update",
    "is_positive_definite",
    "nearest_spd",
    "mahalanobis_sq",
    "safe_cholesky",
    "condition_number",
]


def symmetrize(p: ArrayLike) -> NDArray[np.float64]:
    """Return ``(P + Pᵀ)/2``.

    Enforcing symmetry after every update costs one add and one scale and
    removes the accumulation of asymmetry that otherwise destroys the Cholesky
    factorisations used by the UKF.
    """
    a = np.asarray(p, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"symmetrize() needs a square matrix, got shape {a.shape}")
    return 0.5 * (a + a.T)


def joseph_update(
    p_prior: ArrayLike, gain: ArrayLike, h: ArrayLike, r: ArrayLike
) -> NDArray[np.float64]:
    r"""Joseph-form posterior covariance.

    .. math:: P^+ = (I - K H) P^- (I - K H)^\mathsf{T} + K R K^\mathsf{T}

    Algebraically equal to ``(I − KH)P⁻`` **only** for the optimal gain, but it
    stays symmetric and positive semi-definite for *any* gain and for any
    round-off, which is exactly why it is used here (Bar-Shalom et al. 2001,
    §5.2, where the Joseph form is derived).
    """
    pp = np.asarray(p_prior, dtype=float)
    k = np.asarray(gain, dtype=float)
    hh = np.asarray(h, dtype=float)
    rr = np.asarray(r, dtype=float)
    ikh = np.eye(pp.shape[0]) - k @ hh
    return symmetrize(ikh @ pp @ ikh.T + k @ rr @ k.T)


def is_positive_definite(p: ArrayLike) -> bool:
    """True if ``P`` admits a Cholesky factorisation (i.e. is SPD)."""
    try:
        np.linalg.cholesky(np.asarray(p, dtype=float))
    except np.linalg.LinAlgError:
        return False
    return True


def safe_cholesky(p: ArrayLike, jitter: float = 1e-12, max_tries: int = 6) -> NDArray[np.float64]:
    """Lower-triangular Cholesky factor, adding scaled jitter if needed.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the matrix is still not factorisable after ``max_tries`` inflations
        of ``jitter`` by 10× each. Failing loudly is deliberate: a silent
        fallback would hide a broken filter.
    """
    a = symmetrize(p)
    scale = float(np.trace(a)) / a.shape[0] if a.shape[0] else 1.0
    scale = scale if np.isfinite(scale) and scale > 0.0 else 1.0
    eps = jitter
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(a)
        except np.linalg.LinAlgError:
            a = a + eps * scale * np.eye(a.shape[0])
            eps *= 10.0
    raise np.linalg.LinAlgError(
        f"matrix is not positive definite even after {max_tries} jitter inflations; "
        f"min eigenvalue {float(np.linalg.eigvalsh(symmetrize(p)).min()):.3e}"
    )


def mahalanobis_sq(residual: ArrayLike, cov: ArrayLike) -> float:
    r"""Squared Mahalanobis distance ``rᵀ S⁻¹ r``.

    Solved with a Cholesky factorisation rather than an explicit inverse. This
    is the quantity behind both NEES and NIS.

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``cov`` is not positive definite — reported, never patched.
    """
    r = np.asarray(residual, dtype=float).reshape(-1)
    s = symmetrize(cov)
    if s.shape[0] != r.size:
        raise ValueError(f"residual of size {r.size} does not match covariance {s.shape}")
    lo = np.linalg.cholesky(s)
    y = np.linalg.solve(lo, r)
    return float(np.dot(y, y))


def nearest_spd(p: ArrayLike, floor: float = 0.0) -> NDArray[np.float64]:
    """Symmetric matrix with eigenvalues clipped from below at ``floor``.

    Provided as an explicit, opt-in repair. NavBench never calls it inside a
    filter: the diagnostics are supposed to reveal covariance collapse, not
    paper over it.
    """
    a = symmetrize(p)
    w, v = np.linalg.eigh(a)
    return symmetrize(v @ np.diag(np.maximum(w, floor)) @ v.T)


def condition_number(p: ArrayLike) -> float:
    """2-norm condition number of a symmetric matrix (``inf`` if singular)."""
    w = np.linalg.eigvalsh(symmetrize(p))
    lo, hi = float(np.min(np.abs(w))), float(np.max(np.abs(w)))
    return np.inf if lo == 0.0 else hi / lo
