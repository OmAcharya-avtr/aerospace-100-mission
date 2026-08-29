"""Modal (Zernike-coefficient) least-squares slope-to-phase reconstruction.

This is the "regularized least-squares baseline" the mission requires the
learned reconstructor (`wavelab.ml.ZernikeSlopeEnsemble`) to be benchmarked
against: both map a slope vector directly to a Noll-Zernike coefficient
vector, on identical data, so their errors are directly comparable
(README "Benchmark results"; `validation/VALIDATION.md`).

The forward model is the analytic point-sampled Zernike gradient interaction
matrix from `wavelab.zernike.zernike_slope_matrix`; the inverse is solved with
either Tikhonov or truncated-SVD regularization (`wavelab.linalg`), matching
the mission rule that the classical reconstructor is implemented and
regularized before any ML model is written.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .linalg import noise_propagation_coefficients, tikhonov_solve, tsvd_solve
from .zernike import zernike_slope_matrix

__all__ = ["ModalReconstructor"]


@dataclass
class ModalReconstructor:
    """Regularized least-squares reconstructor: slopes -> Noll-Zernike coefficients.

    Parameters
    ----------
    noll_indices: Noll ``j`` values (piston, ``j = 1``, excluded -- it has no
        gradient and is unobservable from slopes by construction) to
        reconstruct, in coefficient-vector order.
    sub_x, sub_y: ``(n_sub,)`` subaperture centre coordinates, dimensionless
        pupil units (unit disc).
    method: ``"tikhonov"`` or ``"tsvd"``.
    reg: regularization parameter -- ``lambda`` for Tikhonov (``>= 0``) or
        ``rel_tol`` for TSVD (``[0, 1)``).

    Notes
    -----
    Subaperture dropout is handled by row selection: `reconstruct` accepts an
    optional boolean ``active`` mask and solves the regularized problem using
    only the rows (both x and y) of the interaction matrix that belong to
    active subapertures. This is the "known dropout pattern used exactly"
    baseline the learned model (which only sees dropout as a zeroed input
    feature) is compared against -- documented explicitly because it gives
    the classical baseline structural information the ML model does not get
    for free (README "Limitations").
    """

    noll_indices: list[int]
    sub_x: NDArray[np.float64]
    sub_y: NDArray[np.float64]
    method: str = "tikhonov"
    reg: float = 1e-3
    _matrix: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if 1 in self.noll_indices:
            raise ValueError("piston (Noll j = 1) has zero gradient and cannot be reconstructed")
        if len(self.noll_indices) == 0:
            raise ValueError("noll_indices must be non-empty")
        if self.method not in ("tikhonov", "tsvd"):
            raise ValueError(f"method must be 'tikhonov' or 'tsvd', got {self.method!r}")
        sx = np.asarray(self.sub_x, dtype=np.float64).ravel()
        sy = np.asarray(self.sub_y, dtype=np.float64).ravel()
        if sx.shape != sy.shape:
            raise ValueError(f"sub_x and sub_y must match, got {sx.shape} and {sy.shape}")
        if sx.size < len(self.noll_indices):
            raise ValueError(
                f"reconstruction is under-determined: {sx.size} subapertures "
                f"({2 * sx.size} slopes) < {len(self.noll_indices)} modes"
            )
        self.sub_x, self.sub_y = sx, sy
        self._matrix = zernike_slope_matrix(list(self.noll_indices), sx, sy)

    @property
    def n_sub(self) -> int:
        """Number of subapertures the matrix was built for."""
        return self.sub_x.size

    @property
    def matrix(self) -> NDArray[np.float64]:
        """``(2 * n_sub, n_modes)`` full (no-dropout) interaction matrix."""
        return self._matrix

    def _active_matrix_and_rows(
        self, active: NDArray[np.bool_] | None
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        n = self.n_sub
        if active is None:
            row_mask = np.ones(2 * n, dtype=bool)
        else:
            a = np.asarray(active, dtype=bool).ravel()
            if a.shape != (n,):
                raise ValueError(f"active must have shape ({n},), got {a.shape}")
            if not np.any(a):
                raise ValueError("active mask has no active subapertures")
            row_mask = np.concatenate([a, a])
        return self._matrix[row_mask], row_mask

    def reconstruct(
        self, slopes: NDArray[np.float64], active: NDArray[np.bool_] | None = None
    ) -> NDArray[np.float64]:
        """Reconstruct Zernike coefficients from a slope vector.

        Parameters
        ----------
        slopes: ``(2 * n_sub,)`` measured slopes, x-block then y-block, or
            ``(2 * n_sub,)`` with entries at inactive rows ignored when
            `active` is given (their values do not matter).
        active: optional ``(n_sub,)`` bool mask; False marks a dropped-out
            subaperture whose two slope rows are excluded from the solve.

        Returns
        -------
        ``(n_modes,)`` reconstructed Noll coefficients.
        """
        s = np.asarray(slopes, dtype=np.float64).ravel()
        if s.shape != (2 * self.n_sub,):
            raise ValueError(f"slopes must have shape ({2 * self.n_sub},), got {s.shape}")
        if not np.all(np.isfinite(s)):
            raise ValueError("slopes contain non-finite values")
        matrix, row_mask = self._active_matrix_and_rows(active)
        if matrix.shape[0] < matrix.shape[1] and self.method == "tikhonov" and self.reg == 0.0:
            raise ValueError(
                "system is under-determined for unregularized Tikhonov (reg=0); "
                "increase reg or use method='tsvd'"
            )
        rhs = s[row_mask]
        if self.method == "tikhonov":
            return tikhonov_solve(matrix, rhs, self.reg)
        return tsvd_solve(matrix, rhs, self.reg)

    def noise_propagation(self, active: NDArray[np.bool_] | None = None) -> NDArray[np.float64]:
        """Per-mode noise propagation coefficient (`wavelab.linalg.noise_propagation_coefficients`).

        Only meaningful for the TSVD solver (the coefficient formula assumes
        an exact pseudo-inverse); for Tikhonov reconstruction the equivalent
        quantity is the diagonal of ``R R^T`` with
        ``R = (G^T G + lambda^2 I)^-1 G^T``, which is what is actually used
        here regardless of `self.method`, so the returned coefficients apply
        to whichever pseudo-inverse `self.method`/`self.reg` selects.

        Returns
        -------
        ``(n_modes,)`` array such that, for i.i.d. slope noise of variance
        ``sigma_s^2``, ``Var(a_hat_k) = coeff_k * sigma_s^2``.
        """
        matrix, _ = self._active_matrix_and_rows(active)
        if self.method == "tsvd":
            return noise_propagation_coefficients(matrix, self.reg)
        m = matrix.shape[1]
        normal_inv = np.linalg.inv(matrix.T @ matrix + (self.reg**2) * np.eye(m))
        r = normal_inv @ matrix.T  # (M, P)
        return np.sum(r**2, axis=1)
