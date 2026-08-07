"""Covariance housekeeping: Joseph-form update and numerical-health checks.

Why a separate module
---------------------
The dominant failure mode of a hand-rolled Kalman filter is not a wrong
equation but a covariance matrix that quietly stops being a covariance
matrix: it loses symmetry, then positive definiteness, then the filter
either ignores its measurements (covariance collapse) or blows up
(divergence). Everything in this module exists to make that failure
visible and to make it less likely.

References
----------
- Bar-Shalom, Y., Rong Li, X. and Kirubarajan, T., *Estimation with
  Applications to Tracking and Navigation*, Wiley 2001 — Chapter 5
  (linear estimation in dynamic systems) gives the Joseph form and the
  covariance-update alternatives.
- Simon, D., *Optimal State Estimation: Kalman, H-infinity, and Nonlinear
  Approaches*, Wiley 2006 — Chapter 6 discusses the Joseph stabilised
  update and its arithmetic cost.
- Bierman, G. J., *Factorization Methods for Discrete Sequential
  Estimation*, Academic Press 1977 — the standard reference for UD and
  square-root filtering, i.e. what to use when Joseph form is not enough.
- Maybeck, P. S., *Stochastic Models, Estimation, and Control*, Vol. 1,
  Academic Press 1979 — Chapter 7 covers square-root implementations and
  the conditioning argument.

Units
-----
This module is unit-agnostic: covariance entries carry the squared units
of the corresponding state components (m^2, (m/s)^2, m^2/s, ...). The
caller is responsible for a consistent unit system.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "symmetrize",
    "is_symmetric",
    "min_eigenvalue",
    "is_positive_semidefinite",
    "joseph_update",
    "simple_update",
    "covariance_health",
]


def _as_matrix(a: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return ``a`` as a 2-D float array, or raise ``ValueError``."""
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {arr.shape}")
    return arr


def _as_square(a: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return ``a`` as a square 2-D float array, or raise ``ValueError``."""
    arr = _as_matrix(a, name)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square, got shape {arr.shape}")
    return arr


def symmetrize(p: ArrayLike) -> NDArray[np.float64]:
    """Return the symmetric part ``(P + P.T) / 2`` of a covariance matrix.

    Parameters
    ----------
    p : array_like, shape (n, n)
        Covariance matrix [units: squared state units].

    Returns
    -------
    ndarray, shape (n, n)
        Exactly symmetric matrix: ``M[i, j] == M[j, i]`` bit-for-bit.

    Notes
    -----
    Bit-exactness holds because IEEE-754 addition is commutative for
    non-NaN operands, so ``0.5*(P[i, j] + P[j, i])`` and
    ``0.5*(P[j, i] + P[i, j])`` produce the identical bit pattern. No
    explicit triangle copy is needed. (NaN inputs propagate NaN, which is
    then unequal to itself; that is a caller error, not a symmetry
    failure.)

    Even the Joseph form accumulates asymmetry of order machine epsilon
    through its matrix products. Re-symmetrising after each update is the
    cheapest available insurance and is standard practice (Bar-Shalom,
    Rong Li & Kirubarajan 2001, Ch. 5).
    """
    arr = _as_square(p, "P")
    return 0.5 * (arr + arr.T)


def is_symmetric(p: ArrayLike, atol: float = 1e-12, rtol: float = 1e-9) -> bool:
    """Return ``True`` if ``P`` equals its transpose within tolerance.

    Parameters
    ----------
    p : array_like, shape (n, n)
        Covariance matrix.
    atol, rtol : float
        Absolute and relative tolerances passed to :func:`numpy.allclose`.
    """
    arr = _as_square(p, "P")
    return bool(np.allclose(arr, arr.T, atol=atol, rtol=rtol))


def min_eigenvalue(p: ArrayLike) -> float:
    """Return the smallest eigenvalue of the symmetric part of ``P``.

    Uses :func:`numpy.linalg.eigvalsh` on ``(P + P.T)/2``, so the result is
    real by construction. A negative return value means the matrix is not
    a valid covariance.
    """
    return float(np.linalg.eigvalsh(symmetrize(p))[0])


def is_positive_semidefinite(p: ArrayLike, tol: float = -1e-12) -> bool:
    """Return ``True`` if the symmetric part of ``P`` has no eigenvalue below ``tol``.

    Parameters
    ----------
    p : array_like, shape (n, n)
        Covariance matrix.
    tol : float
        Threshold on the minimum eigenvalue. The default of ``-1e-12``
        admits eigenvalues that are negative only at round-off level.
        Pass ``0.0`` for a strict test.
    """
    return min_eigenvalue(p) >= tol


def joseph_update(
    p_prior: ArrayLike,
    gain: ArrayLike,
    h: ArrayLike,
    r: ArrayLike,
) -> NDArray[np.float64]:
    r"""Joseph-form (stabilised) covariance update.

    .. math::
        P^{+} = (I - K H) P^{-} (I - K H)^{\mathsf{T}} + K R K^{\mathsf{T}}

    Parameters
    ----------
    p_prior : array_like, shape (n, n)
        Prior (predicted) covariance :math:`P^{-}` [squared state units].
    gain : array_like, shape (n, m)
        Gain matrix :math:`K` [state units / measurement units].
    h : array_like, shape (m, n)
        Measurement matrix :math:`H` (or the EKF/UKF equivalent
        linearisation) [measurement units / state units].
    r : array_like, shape (m, m)
        Measurement-noise covariance :math:`R` [squared measurement units].

    Returns
    -------
    ndarray, shape (n, n)
        Posterior covariance, re-symmetrised.

    Notes
    -----
    **Source.** Bar-Shalom, Rong Li & Kirubarajan 2001, Ch. 5; Simon 2006,
    Ch. 6. The form is usually attributed to P. D. Joseph (in Bucy &
    Joseph, *Filtering for Stochastic Processes with Applications to
    Guidance*, 1968).

    **Why this form.** The algebraically equivalent short form
    :math:`P^{+} = (I - KH)P^{-}` is a *difference* of two positive
    semi-definite matrices, so cancellation can push :math:`P^{+}`
    indefinite, and it is only valid for the optimal Kalman gain. The
    Joseph form is a *sum* of two positive semi-definite congruences,
    :math:`A P^{-} A^{\mathsf{T}}` and :math:`K R K^{\mathsf{T}}`, so in
    exact arithmetic it is positive semi-definite and symmetric for *any*
    gain, optimal or not, and its first-order sensitivity to gain error
    vanishes. That makes it the safe choice for sub-optimal, quantised, or
    hand-tuned gains, and for the EKF/UKF where ``H`` is a linearisation
    rather than the true measurement map.

    **Cost.** Roughly 2x the flops of the short form (two extra n x n
    products). At the state dimensions this package targets (n <~ 20) the
    cost is irrelevant next to the robustness.

    **Validity / when this is not enough.** Joseph form still stores
    :math:`P` itself, whose condition number is the *square* of the
    condition number of any square-root factor. When the eigenvalue spread
    of :math:`P` approaches the reciprocal of the machine epsilon --
    nearly-exact measurements, nearly-unobservable states, single-precision
    embedded arithmetic, or large integrated navigation states -- no
    symmetric-form update is safe and a factorised filter is required
    (Potter/Carlson square root, Bierman UD; Bierman 1977, Maybeck 1979
    Ch. 7). Those propagate a factor whose condition number is the square
    root of that of :math:`P` and cannot represent a negative-definite
    covariance at all. This package does *not* implement them; see the
    README Limitations section.
    """
    p_arr = _as_square(p_prior, "P")
    k_arr = _as_matrix(gain, "K")
    h_arr = _as_matrix(h, "H")
    r_arr = _as_square(r, "R")

    n = p_arr.shape[0]
    if h_arr.shape[1] != n:
        raise ValueError(f"H has {h_arr.shape[1]} columns but P is {n}x{n}")
    m = h_arr.shape[0]
    if k_arr.shape != (n, m):
        raise ValueError(f"K must have shape ({n}, {m}), got {k_arr.shape}")
    if r_arr.shape != (m, m):
        raise ValueError(f"R must have shape ({m}, {m}), got {r_arr.shape}")

    a = np.eye(n) - k_arr @ h_arr
    return symmetrize(a @ p_arr @ a.T + k_arr @ r_arr @ k_arr.T)


def simple_update(
    p_prior: ArrayLike,
    gain: ArrayLike,
    h: ArrayLike,
) -> NDArray[np.float64]:
    r"""Short-form covariance update :math:`P^{+} = (I - KH)P^{-}`.

    Provided for comparison and teaching only. It is valid **only** for
    the optimal Kalman gain and is numerically fragile (see
    :func:`joseph_update`). No re-symmetrisation is applied, deliberately,
    so that its asymmetry growth can be measured.

    Parameters
    ----------
    p_prior : array_like, shape (n, n)
        Prior covariance.
    gain : array_like, shape (n, m)
        Kalman gain.
    h : array_like, shape (m, n)
        Measurement matrix.
    """
    p_arr = _as_square(p_prior, "P")
    k_arr = _as_matrix(gain, "K")
    h_arr = _as_matrix(h, "H")
    n = p_arr.shape[0]
    return (np.eye(n) - k_arr @ h_arr) @ p_arr


def covariance_health(p: ArrayLike) -> dict[str, float]:
    """Return diagnostics used to spot covariance collapse or divergence.

    Parameters
    ----------
    p : array_like, shape (n, n)
        Covariance matrix.

    Returns
    -------
    dict
        ``asymmetry``   -- ``max |P - P.T|`` (absolute).
        ``min_eig``     -- smallest eigenvalue of the symmetric part.
        ``max_eig``     -- largest eigenvalue of the symmetric part.
        ``trace``       -- trace of ``P``.
        ``condition``   -- ``max_eig / min_eig`` (``inf`` if ``min_eig <= 0``).

    Notes
    -----
    Observable symptoms, and what they mean:

    - **Covariance collapse.** ``trace`` decays monotonically towards zero
      while the innovation sequence stays large or grows. The gain goes to
      zero, so the filter stops listening to its measurements and coasts on
      the model. Usual cause: process noise ``Q`` set too small (or set to
      zero) for the real dynamics, or repeated updates with an
      over-optimistic ``R``.
    - **Loss of positive definiteness.** ``min_eig`` crosses zero. With the
      short-form update this can happen from cancellation alone. Downstream
      symptom: Cholesky failure in the UKF sigma-point generation, or a
      negative "variance" reported for a state.
    - **Asymmetry growth.** ``asymmetry`` climbing above round-off level
      indicates the update is not being re-symmetrised; it usually precedes
      the loss of definiteness.
    - **Divergence.** ``trace`` and the normalised innovation squared (NIS,
      returned by the filters in this package) both grow without bound, or
      the NIS sits far above its chi-squared bound with ``m`` degrees of
      freedom while ``trace`` stays small -- the filter is confident and
      wrong. Usual causes: unmodelled dynamics, wrong ``Q``/``R`` ratio, or
      an EKF linearisation error that is large compared with the state
      uncertainty.
    """
    arr = _as_square(p, "P")
    eig = np.linalg.eigvalsh(0.5 * (arr + arr.T))
    min_eig = float(eig[0])
    max_eig = float(eig[-1])
    condition = float(max_eig / min_eig) if min_eig > 0.0 else float("inf")
    return {
        "asymmetry": float(np.max(np.abs(arr - arr.T))),
        "min_eig": min_eig,
        "max_eig": max_eig,
        "trace": float(np.trace(arr)),
        "condition": condition,
    }
