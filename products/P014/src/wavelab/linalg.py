"""Regularized least-squares solvers shared by the modal and zonal reconstructors.

Both baselines invert a rank-deficient or ill-conditioned linear map from
slopes to an unknown (Zernike coefficients or grid phase). Two regularization
strategies are implemented, both standard in the inverse-problems and
adaptive-optics literature:

* **Tikhonov regularization** -- solve ``min_u ||G u - s||^2 + lambda^2 ||u||^2``,
  closed form ``u = (G^T G + lambda^2 I)^-1 G^T s``. Source: A. N. Tikhonov &
  V. Y. Arsenin, *Solutions of Ill-Posed Problems*, Winston & Sons, 1977.
  ``lambda`` trades reconstruction fidelity against noise amplification; as
  ``lambda -> 0`` this reduces to the ordinary least-squares (Moore-Penrose)
  solution.
* **Truncated SVD (TSVD)** -- discard singular vectors below a relative
  threshold before pseudo-inverting. Source: standard regularization result,
  e.g. P. C. Hansen, "The truncated SVD as a method for regularization",
  *BIT* **27**, 534-553 (1987). Unlike Tikhonov, TSVD gives an explicit,
  inspectable null space: any singular value ``sigma_i <= rel_tol * sigma_max``
  is dropped, and the corresponding right singular vector is *exactly*
  unobservable from the data -- the natural place to confirm that piston (and,
  for the Fried zonal geometry, waffle) is being handled explicitly rather
  than left to chance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["tikhonov_solve", "tsvd_solve", "null_space", "noise_propagation_coefficients"]


def _check_system(matrix: NDArray[np.float64], rhs: NDArray[np.float64]) -> None:
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
    if rhs.ndim not in (1, 2):
        raise ValueError(f"rhs must be 1-D or 2-D, got shape {rhs.shape}")
    if rhs.shape[0] != matrix.shape[0]:
        raise ValueError(f"matrix has {matrix.shape[0]} rows but rhs has {rhs.shape[0]}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix contains non-finite values")
    if not np.all(np.isfinite(rhs)):
        raise ValueError("rhs contains non-finite values")


def tikhonov_solve(
    matrix: NDArray[np.float64], rhs: NDArray[np.float64], lam: float
) -> NDArray[np.float64]:
    """Tikhonov-regularized least-squares solution.

    Parameters
    ----------
    matrix: ``(P, M)`` forward operator ``G``.
    rhs: ``(P,)`` or ``(P, K)`` measured data (``K`` independent solves).
    lam: regularization strength ``lambda >= 0`` [same units as one singular
        value of ``G``]. ``lam = 0`` gives the unregularized normal-equations
        solution (fails if ``G`` is rank deficient; use `tsvd_solve` then).

    Returns
    -------
    ``(M,)`` or ``(M, K)`` solution ``u``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    _check_system(matrix, rhs)
    lam = float(lam)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError(f"lam must be finite and >= 0, got {lam!r}")
    m = matrix.shape[1]
    normal = matrix.T @ matrix + (lam**2) * np.eye(m)
    return np.linalg.solve(normal, matrix.T @ rhs)


def tsvd_solve(
    matrix: NDArray[np.float64], rhs: NDArray[np.float64], rel_tol: float = 1e-6
) -> NDArray[np.float64]:
    """Truncated-SVD regularized least-squares solution.

    Parameters
    ----------
    matrix: ``(P, M)`` forward operator ``G``.
    rhs: ``(P,)`` or ``(P, K)`` data.
    rel_tol: singular values ``sigma_i <= rel_tol * sigma_max`` are dropped
        [-], in ``[0, 1)``. Larger values regularize more aggressively.

    Returns
    -------
    ``(M,)`` or ``(M, K)`` minimum-norm solution over the retained subspace.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    _check_system(matrix, rhs)
    rel_tol = float(rel_tol)
    if not (0.0 <= rel_tol < 1.0):
        raise ValueError(f"rel_tol must be in [0, 1), got {rel_tol!r}")
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    if s[0] <= 0.0:
        raise ValueError("matrix is identically zero; no solution exists")
    keep = s > rel_tol * s[0]
    s_inv = np.zeros_like(s)
    s_inv[keep] = 1.0 / s[keep]
    # (M, P) pseudo-inverse restricted to the retained singular subspace.
    pinv = (vt[keep].T * s_inv[keep]) @ u[:, keep].T
    return pinv @ rhs


def null_space(matrix: NDArray[np.float64], rel_tol: float = 1e-6) -> NDArray[np.float64]:
    """Right singular vectors with singular value ``<= rel_tol * sigma_max``.

    These are the directions in the unknown (Zernike coefficients or grid
    phase) that produce (numerically) zero data -- the modes the geometry
    cannot see, e.g. piston for every slope-sensing geometry, plus waffle for
    the Fried zonal geometry (README "Engineering theory").

    Returns
    -------
    ``(M, K)`` array whose ``K`` columns are an orthonormal basis of the
    (numerical) null space; ``K = 0`` if none is found within `rel_tol`.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
    rel_tol = float(rel_tol)
    if not (0.0 <= rel_tol < 1.0):
        raise ValueError(f"rel_tol must be in [0, 1), got {rel_tol!r}")
    _u, s, vt = np.linalg.svd(matrix, full_matrices=True)
    threshold = rel_tol * s[0] if s.size else 0.0
    n_sig = s.size
    n_cols = vt.shape[0]
    drop = np.ones(n_cols, dtype=bool)
    drop[:n_sig] = s <= threshold
    return vt[drop].T


def noise_propagation_coefficients(
    matrix: NDArray[np.float64], rel_tol: float = 1e-6
) -> NDArray[np.float64]:
    """Per-unknown noise propagation coefficient of the TSVD pseudo-inverse.

    For i.i.d. measurement noise of variance ``sigma_s^2`` on every entry of
    the data, the reconstructed unknown ``u_hat = R s`` (``R`` the pseudo-
    inverse) has ``Var(u_hat_k) = sigma_s^2 * sum_p R[k, p]^2`` -- the
    diagonal of ``R R^T`` scaled by the (uniform) data noise variance. The
    per-mode coefficient returned here is exactly that row-norm-squared term,
    i.e. ``Var(u_hat_k) = coeff_k * sigma_s^2``. This is the standard noise
    propagation formalism used to size a reconstructor's photon-noise
    sensitivity (Wallner 1983, "Optimal wave-front correction using slope
    measurements", *J. Opt. Soc. Am.* **73**, 1771; Hardy 1998,
    *Adaptive Optics for Astronomical Telescopes*, ch. 9).

    Returns
    -------
    ``(M,)`` array, coefficient per unknown [1 / (unit of one matrix entry)^2].
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
    rel_tol = float(rel_tol)
    if not (0.0 <= rel_tol < 1.0):
        raise ValueError(f"rel_tol must be in [0, 1), got {rel_tol!r}")
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    if s.size == 0 or s[0] <= 0.0:
        raise ValueError("matrix is identically zero; no solution exists")
    keep = s > rel_tol * s[0]
    s_inv = np.zeros_like(s)
    s_inv[keep] = 1.0 / s[keep]
    pinv = (vt[keep].T * s_inv[keep]) @ u[:, keep].T  # (M, P)
    return np.sum(pinv**2, axis=1)
