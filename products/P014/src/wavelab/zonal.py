"""Zonal (grid-phase) least-squares reconstruction on the Hudgin/Fried geometries.

Recovers phase at the active points of a `wavelab.geometry.PupilGrid` from a
slope vector, using either the Hudgin or Fried finite-difference geometry
matrix (`wavelab.geometry`) and Tikhonov or truncated-SVD regularization
(`wavelab.linalg`). This is the "Southwell/Fried geometry matrix baseline"
required by the mission scope, implemented and validated independently of the
Zernike-modal baseline in `wavelab.modal` (see README "Engineering theory").

Piston is unobservable for both geometries; waffle is additionally
unobservable for the Fried geometry (module docstring of `wavelab.geometry`).
Both are handled explicitly:

* TSVD reconstruction drops exactly the null-space directions below
  `reg` (relative to the largest singular value) -- piston and, for Fried,
  waffle are excluded from the returned solution by construction, and
  `null_space_dimension` reports how many were dropped.
* Tikhonov reconstruction cannot drop a mode exactly (it shrinks every
  direction, including the null space, toward zero by an amount that depends
  on `reg`), so a returned phase vector is always mean-subtracted (piston
  removed by convention) before being handed back; the residual, un-damped
  waffle component of a Tikhonov/Fried reconstruction is reported by
  `waffle_component` rather than silently left in the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .geometry import PupilGrid, fried_matrix, hudgin_matrix, prune_unconstrained
from .linalg import null_space, tikhonov_solve, tsvd_solve

__all__ = ["ZonalReconstructor"]

_BUILDERS = {"hudgin": hudgin_matrix, "fried": fried_matrix}


@dataclass
class ZonalReconstructor:
    """Regularized zonal reconstructor for one `PupilGrid` and one geometry.

    Parameters
    ----------
    grid: the phase-point grid.
    geometry: ``"hudgin"`` or ``"fried"``.
    method: ``"tikhonov"`` or ``"tsvd"``.
    reg: ``lambda`` (Tikhonov, ``>= 0``) or ``rel_tol`` (TSVD, ``[0, 1)``).
    """

    grid: PupilGrid
    geometry: str = "fried"
    method: str = "tsvd"
    reg: float = 1e-6
    _matrix: NDArray[np.float64] = field(init=False, repr=False)
    _keep_idx: NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.geometry not in _BUILDERS:
            raise ValueError(f"geometry must be one of {sorted(_BUILDERS)}, got {self.geometry!r}")
        if self.method not in ("tikhonov", "tsvd"):
            raise ValueError(f"method must be 'tikhonov' or 'tsvd', got {self.method!r}")
        full_matrix = _BUILDERS[self.geometry](self.grid)
        pruned, keep_idx = prune_unconstrained(full_matrix)
        self._matrix = pruned
        self._keep_idx = keep_idx

    @property
    def matrix(self) -> NDArray[np.float64]:
        """``(n_slopes, n_used)`` geometry matrix, unconstrained points already dropped."""
        return self._matrix

    @property
    def n_slopes(self) -> int:
        return self._matrix.shape[0]

    @property
    def n_used(self) -> int:
        """Number of grid points actually reconstructed (active points minus unconstrained ones)."""
        return self._matrix.shape[1]

    @property
    def keep_idx(self) -> NDArray[np.int64]:
        """Indices into `PupilGrid.active_coords` order of the `n_used` reconstructed points."""
        return self._keep_idx

    def null_space_dimension(self) -> int:
        """Dimension of the numerically detected null space at `reg` (TSVD sense)."""
        tol = self.reg if self.method == "tsvd" else 1e-6
        return null_space(self._matrix, tol).shape[1]

    def reconstruct(self, slopes: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruct phase at the grid's active points from a slope vector.

        Parameters
        ----------
        slopes: ``(n_slopes,)``, in the row order of `matrix`.

        Returns
        -------
        ``(n_used,)`` phase at `keep_idx`, piston removed (zero mean over
        used points).
        """
        s = np.asarray(slopes, dtype=np.float64).ravel()
        if s.shape != (self.n_slopes,):
            raise ValueError(f"slopes must have shape ({self.n_slopes},), got {s.shape}")
        if not np.all(np.isfinite(s)):
            raise ValueError("slopes contain non-finite values")
        if self.method == "tikhonov":
            phi = tikhonov_solve(self._matrix, s, self.reg)
        else:
            phi = tsvd_solve(self._matrix, s, self.reg)
        return phi - phi.mean()

    def waffle_component(self, phi: NDArray[np.float64]) -> float:
        """Projection of a reconstructed phase vector onto the waffle pattern.

        Only meaningful for ``geometry="fried"``; returns 0 for Hudgin runs
        by convention (waffle is not a distinguished direction there). The
        checkerboard pattern is restricted to the `n_used` reconstructed
        points and re-normalised there (dropping the unconstrained points
        does not break the zero-cell-average property for any complete
        Fried cell, since an unconstrained point belongs to no complete
        cell), then re-scaled to unit norm so the result is comparable
        across grids.
        """
        if self.geometry != "fried":
            return 0.0
        n = self.grid.n_grid
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
        checker = np.where((ii + jj) % 2 == 0, 1.0, -1.0)
        w = checker[self.grid.mask][self._keep_idx]
        norm = np.linalg.norm(w)
        if norm == 0.0:
            return 0.0
        w = w / norm
        return float(np.dot(np.asarray(phi, dtype=np.float64), w))
