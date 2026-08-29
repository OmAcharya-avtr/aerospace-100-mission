"""Circular-pupil subaperture layout and zonal (grid) slope-to-phase geometry matrices.

WaveLab implements the two classical finite-difference geometries reviewed and
compared by Southwell (1980), *J. Opt. Soc. Am.* **70** (8), 998-1006, "Wave-
front estimator for wave-front sensing", the paper the mission cites for
slope-to-phase geometries in general:

* **Hudgin geometry** -- R. H. Hudgin, "Wave-front reconstruction for
  compensated imaging", *J. Opt. Soc. Am.* **67** (3), 375-378 (1977). Phase
  points sit on an ``n x n`` grid; each slope is the finite difference of two
  *adjacent* phase points, ``s_x(i,j) = phi(i+1,j) - phi(i,j)`` /
  ``s_y(i,j) = phi(i,j+1) - phi(i,j)``. Every phase point is linked to its
  neighbours by at least one measurement, so the measurement graph is
  connected and the *only* unobservable mode is a global additive constant
  (piston).
* **Fried geometry** -- D. L. Fried, "Least-square fitting a wave-front
  distortion estimate to an array of phase-difference measurements",
  *J. Opt. Soc. Am.* **67** (3), 370-375 (1977). Each slope is the *average*
  of the two finite differences along one edge of a unit grid cell,
  ``s_x(i,j) = [(phi(i+1,j)-phi(i,j)) + (phi(i+1,j+1)-phi(i,j+1))] / 2``, and
  the ``y`` analogue. Averaging over the cell decouples the grid into two
  independent interleaved (checkerboard) sub-lattices that are each internally
  connected but linked to each other only through a spatial pattern that
  cancels exactly in every averaged slope -- the classic "waffle mode" null
  space (documented e.g. in Hardy 1998, *Adaptive Optics for Astronomical
  Telescopes*, Oxford University Press, ch. 5; and Herrmann 1980, "Least-
  squares wave front errors of minimum norm", *J. Opt. Soc. Am.* **70** (1),
  28-35). The Fried geometry therefore has a **two-dimensional** null space
  (piston + waffle), handled explicitly here via `waffle_pattern` and via
  `wavelab.linalg.null_space` / `wavelab.linalg.tsvd_solve`.

Grid and pupil convention
--------------------------
Phase points live on an ``n_grid x n_grid`` square grid covering the
normalised pupil ``[-1, 1]^2`` with unit spacing in grid-index units (the
absolute spacing does not matter for the algebra; only the ratios among
matrix entries do). A point is *active* when it falls inside the unit disc
(the illuminated pupil, mirroring `wavelab.zernike.unit_disc_grid`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["PupilGrid", "hudgin_matrix", "fried_matrix", "waffle_pattern", "prune_unconstrained"]


@dataclass(frozen=True)
class PupilGrid:
    """A regular grid of phase points over the normalised pupil, with a circular mask.

    Parameters
    ----------
    n_grid: phase points per side, ``>= 3``.
    obscuration: central obscuration as a fraction of the pupil radius,
        ``[0, 1)``. Points inside the obscuration are inactive.

    Attributes
    ----------
    x, y: ``(n_grid, n_grid)`` coordinate arrays over ``[-1, 1]``.
    mask: ``(n_grid, n_grid)`` bool, True where the point is active (inside
        the illuminated annulus/disc).
    """

    n_grid: int
    obscuration: float = 0.0
    x: NDArray[np.float64] = field(init=False, repr=False)
    y: NDArray[np.float64] = field(init=False, repr=False)
    mask: NDArray[np.bool_] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.n_grid, bool) or not isinstance(self.n_grid, (int, np.integer)):
            raise TypeError(f"n_grid must be an integer, got {self.n_grid!r}")
        if self.n_grid < 3:
            raise ValueError(f"n_grid must be >= 3, got {self.n_grid}")
        obs = float(self.obscuration)
        if not (0.0 <= obs < 1.0):
            raise ValueError(f"obscuration must be in [0, 1), got {obs!r}")
        axis = np.linspace(-1.0, 1.0, int(self.n_grid))
        x, y = np.meshgrid(axis, axis, indexing="xy")
        rho = np.hypot(x, y)
        mask = (rho <= 1.0) & (rho >= obs)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "mask", mask)

    @property
    def n_active(self) -> int:
        """Number of active (illuminated) phase points."""
        return int(self.mask.sum())

    def active_coords(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(n_active,)`` x and y of the active points, row-major grid order."""
        return self.x[self.mask], self.y[self.mask]

    def to_full(self, values: NDArray[np.float64], fill: float = np.nan) -> NDArray[np.float64]:
        """Scatter an ``(n_active,)`` vector back onto the full ``(n_grid, n_grid)`` grid."""
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.n_active,):
            raise ValueError(f"values must have shape ({self.n_active},), got {values.shape}")
        full = np.full((self.n_grid, self.n_grid), fill, dtype=np.float64)
        full[self.mask] = values
        return full


def _index_map(grid: PupilGrid) -> NDArray[np.int64]:
    """(n_grid, n_grid) array mapping active grid cells to a 0-based row-major index, -1 elsewhere."""
    idx = np.full((grid.n_grid, grid.n_grid), -1, dtype=np.int64)
    idx[grid.mask] = np.arange(grid.n_active)
    return idx


def hudgin_matrix(grid: PupilGrid) -> NDArray[np.float64]:
    """Hudgin-geometry slope matrix ``G`` such that ``s = G @ phi`` (noise-free).

    ``phi`` is the ``(n_active,)`` vector of phase values at active points, in
    the order of `PupilGrid.active_coords`. Each row is one finite-difference
    slope between two *adjacent, both-active* grid points, unit spacing.
    Rows are ordered x-differences first (row-major over ``i``, i.e. over the
    grid's first axis / columns), then y-differences.

    Returns
    -------
    ``(n_slopes, n_active)`` sparse-structured but dense-typed matrix (the
    grids used in this package are small enough that density is not a
    performance concern; each row has exactly two nonzero entries, +1/-1).

    Notes
    -----
    Null space: piston only (`wavelab.linalg.null_space` on this matrix
    returns one column for any grid with >= 1 active point and at least one
    measurement, i.e. the constant vector), because every pair of
    edge-adjacent active points is directly linked by one row.
    """
    idx = _index_map(grid)
    n = grid.n_grid
    rows_x: list[NDArray[np.float64]] = []
    rows_y: list[NDArray[np.float64]] = []
    n_active = grid.n_active
    for j in range(n):
        for i in range(n - 1):
            a, b = idx[j, i], idx[j, i + 1]
            if a >= 0 and b >= 0:
                row = np.zeros(n_active)
                row[a] = -1.0
                row[b] = 1.0
                rows_x.append(row)
    for j in range(n - 1):
        for i in range(n):
            a, b = idx[j, i], idx[j + 1, i]
            if a >= 0 and b >= 0:
                row = np.zeros(n_active)
                row[a] = -1.0
                row[b] = 1.0
                rows_y.append(row)
    if not rows_x and not rows_y:
        raise ValueError("grid produces zero slope measurements (too few active points)")
    return np.array(rows_x + rows_y, dtype=np.float64)


def fried_matrix(grid: PupilGrid) -> NDArray[np.float64]:
    """Fried-geometry slope matrix ``G`` such that ``s = G @ phi`` (noise-free).

    Each row is the average of the two finite differences along one edge of a
    unit grid cell (four active corners required), unit spacing. Rows are
    x-slopes (row-major over cells) followed by y-slopes.

    Returns
    -------
    ``(n_slopes, n_active)`` matrix.

    Notes
    -----
    Null space: piston *and* waffle (see module docstring and
    `waffle_pattern`) -- verified numerically in
    ``tests/test_geometry.py::test_fried_null_space_is_two_dimensional``.
    """
    idx = _index_map(grid)
    n = grid.n_grid
    n_active = grid.n_active
    rows_x: list[NDArray[np.float64]] = []
    rows_y: list[NDArray[np.float64]] = []
    for j in range(n - 1):
        for i in range(n - 1):
            p00, p10 = idx[j, i], idx[j, i + 1]
            p01, p11 = idx[j + 1, i], idx[j + 1, i + 1]
            if min(p00, p10, p01, p11) < 0:
                continue
            rx = np.zeros(n_active)
            rx[p00] -= 0.5
            rx[p10] += 0.5
            rx[p01] -= 0.5
            rx[p11] += 0.5
            rows_x.append(rx)
            ry = np.zeros(n_active)
            ry[p00] -= 0.5
            ry[p01] += 0.5
            ry[p10] -= 0.5
            ry[p11] += 0.5
            rows_y.append(ry)
    if not rows_x and not rows_y:
        raise ValueError("grid produces zero slope measurements (too few active/connected cells)")
    return np.array(rows_x + rows_y, dtype=np.float64)


def waffle_pattern(grid: PupilGrid) -> NDArray[np.float64]:
    """The checkerboard "waffle" vector ``(-1)^(i+j)`` at active points, unit-normalised.

    Every Fried-geometry cell average of this pattern is exactly zero (two
    ``+1`` and two ``-1`` corners on every cell), so it lies in the Fried
    matrix's null space along with piston; it is *not* generally in the
    Hudgin matrix's null space (adjacent-point differences of a checkerboard
    are not all zero).

    Returns
    -------
    ``(n_active,)`` unit-norm vector.
    """
    n = grid.n_grid
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
    checker = np.where((ii + jj) % 2 == 0, 1.0, -1.0)
    vec = checker[grid.mask]
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError("waffle pattern is identically zero for this mask")
    return vec / norm


def prune_unconstrained(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Drop unknowns (columns) that no row of `matrix` touches.

    A point on the edge of a non-rectangular pupil mask can end up with no
    complete Fried cell (all four corners active) touching it, or -- for a
    degenerately thin mask -- no Hudgin neighbour either. Such a column is
    identically zero and every value of that unknown produces the same
    (zero) contribution to every measurement: it is not "in the null space"
    in the physically interesting sense (an unobservable *combination* of
    otherwise-linked points, e.g. piston or waffle), it is simply absent from
    the data and must be dropped before null-space analysis or reconstruction
    is meaningful, otherwise it inflates the apparent null space with trivial
    one-point directions.

    Parameters
    ----------
    matrix: ``(n_rows, n_cols)`` geometry matrix.

    Returns
    -------
    pruned: ``(n_rows, n_kept)`` matrix with the zero columns removed.
    keep_idx: ``(n_kept,)`` original column indices that were kept, ascending.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
    used = np.any(matrix != 0.0, axis=0)
    if not np.any(used):
        raise ValueError("matrix has no nonzero columns; nothing is reconstructable")
    keep_idx = np.flatnonzero(used)
    return matrix[:, used], keep_idx
