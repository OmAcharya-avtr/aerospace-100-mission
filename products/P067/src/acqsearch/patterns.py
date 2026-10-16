"""Scan-pattern path generators and the cell visit orders derived from them.

A pattern is produced in two layers.

1. A *continuous path*: an ordered array of dwell boresight positions in the
   pointing-error plane, shape ``(n, 2)``, radians. This is what the coverage
   geometry in :mod:`acqsearch.coverage` integrates.
2. A *cell visit order*: a permutation of the cells of a
   :class:`~acqsearch.geometry.CellGrid`, obtained by walking the path and
   taking each cell the first time the path enters it, then appending any cell
   the path never entered in nearest-neighbour order from the last visited
   cell. This is what the analytic acquisition model in
   :mod:`acqsearch.acquisition` schedules.

Layer 2 makes the four patterns comparable on exactly one abstraction: the
order in which the same 89-odd cells are visited and the slew distance that
order costs. No pattern is allowed a different cell set from another.

References
----------
Koopman, B. O. (1980). *Search and Screening: General Principles with
    Historical Applications*. Pergamon Press. Ch. 4-5 give the classical
    treatment of parallel-sweep (raster) and spiral search and the exponential
    detection law that motivates comparing patterns by covered probability per
    unit time.
Lissajous, J. A. (1857). "Note sur une methode nouvelle applicable a l'etude
    des mouvements vibratoires", *Comptes Rendus* 44, 727-731. The curve
    x = A sin(a t + delta), y = B sin(b t).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import CellGrid

PATTERN_NAMES = ("spiral", "raster", "lissajous", "random_restart")


@dataclass(frozen=True)
class ScanPath:
    """An ordered sequence of dwell boresight positions.

    ``points_rad`` has shape ``(n, 2)`` in radians. ``name`` is one of
    :data:`PATTERN_NAMES`. ``metadata`` records the generating parameters so a
    path can be reproduced from the object alone.
    """

    name: str
    points_rad: np.ndarray
    metadata: dict[str, float]

    @property
    def n_points(self) -> int:
        """Number of dwell positions on the path."""
        return int(self.points_rad.shape[0])

    def path_length_rad(self) -> float:
        """Total along-track angular length of the path, radians."""
        if self.n_points < 2:
            return 0.0
        steps = np.diff(self.points_rad, axis=0)
        return float(np.hypot(steps[:, 0], steps[:, 1]).sum())


def spiral_path(
    pitch_rad: float,
    turns: float,
    along_track_spacing_rad: float,
) -> ScanPath:
    """Archimedean spiral ``r = pitch * theta / (2 pi)`` sampled at fixed arc spacing.

    Parameters
    ----------
    pitch_rad:
        Radial advance per full turn [rad]. For the coverage closed form to
        hold this must satisfy the inter-turn condition in
        :func:`acqsearch.coverage.spiral_closed_form_valid`.
    turns:
        Number of full turns, > 0. Need not be an integer, but the closed form
        in :mod:`acqsearch.coverage` is derived for an integer number.
    along_track_spacing_rad:
        Target arc-length separation between consecutive dwells [rad].

    The first point is the origin. Arc length of the Archimedean spiral is
    integrated numerically on a dense parameter grid and resampled uniformly,
    so the realised spacing is uniform to within the dense-grid resolution.
    """
    if not math.isfinite(pitch_rad) or pitch_rad <= 0.0:
        raise ValueError(f"pitch_rad must be finite and > 0, got {pitch_rad!r}")
    if not math.isfinite(turns) or turns <= 0.0:
        raise ValueError(f"turns must be finite and > 0, got {turns!r}")
    if not math.isfinite(along_track_spacing_rad) or along_track_spacing_rad <= 0.0:
        raise ValueError("along_track_spacing_rad must be finite and > 0")
    theta_max = 2.0 * math.pi * turns
    dense = np.linspace(0.0, theta_max, 20001)
    b = pitch_rad / (2.0 * math.pi)
    radius = b * dense
    dx = np.diff(radius * np.cos(dense))
    dy = np.diff(radius * np.sin(dense))
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(dx, dy))])
    n = max(2, int(math.floor(arc[-1] / along_track_spacing_rad)) + 1)
    wanted = np.arange(n) * along_track_spacing_rad
    theta = np.interp(wanted, arc, dense)
    r = b * theta
    points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return ScanPath(
        name="spiral",
        points_rad=points,
        metadata={
            "pitch_rad": float(pitch_rad),
            "turns": float(turns),
            "along_track_spacing_rad": float(along_track_spacing_rad),
            "end_radius_rad": float(b * theta_max),
        },
    )


def raster_path(extent_rad: float, spacing_rad: float) -> ScanPath:
    """Boustrophedon raster over the square ``[-extent, extent]^2``.

    Rows are spaced ``spacing_rad`` apart in ``y`` and sampled every
    ``spacing_rad`` in ``x``, with alternate rows reversed so no row-return
    slew is needed. Koopman 1980 Ch. 5 calls this the parallel-sweep search.
    """
    if not math.isfinite(extent_rad) or extent_rad <= 0.0:
        raise ValueError(f"extent_rad must be finite and > 0, got {extent_rad!r}")
    if not math.isfinite(spacing_rad) or spacing_rad <= 0.0:
        raise ValueError(f"spacing_rad must be finite and > 0, got {spacing_rad!r}")
    half_n = int(math.ceil(extent_rad / spacing_rad))
    axis = np.arange(-half_n, half_n + 1, dtype=float) * spacing_rad
    rows = []
    for j, y in enumerate(axis):
        xs = axis if j % 2 == 0 else axis[::-1]
        rows.append(np.column_stack([xs, np.full_like(xs, y)]))
    points = np.vstack(rows)
    return ScanPath(
        name="raster",
        points_rad=points,
        metadata={"extent_rad": float(extent_rad), "spacing_rad": float(spacing_rad)},
    )


def lissajous_path(
    extent_rad: float,
    n_points: int,
    freq_x: float = 5.0,
    freq_y: float = 6.0,
    phase_rad: float = 0.5 * math.pi,
) -> ScanPath:
    """Lissajous figure ``x = A sin(a t + delta)``, ``y = A sin(b t)`` over one period.

    ``freq_x`` and ``freq_y`` should be coprime integers for the curve to close
    after a single ``2 pi`` period in ``t``; non-integer values are accepted
    and simply produce an open curve. Amplitude ``A = extent_rad`` [rad].
    """
    if not math.isfinite(extent_rad) or extent_rad <= 0.0:
        raise ValueError(f"extent_rad must be finite and > 0, got {extent_rad!r}")
    if int(n_points) < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points!r}")
    t = np.linspace(0.0, 2.0 * math.pi, int(n_points), endpoint=False)
    points = np.column_stack(
        [extent_rad * np.sin(freq_x * t + phase_rad), extent_rad * np.sin(freq_y * t)]
    )
    return ScanPath(
        name="lissajous",
        points_rad=points,
        metadata={
            "extent_rad": float(extent_rad),
            "n_points": float(int(n_points)),
            "freq_x": float(freq_x),
            "freq_y": float(freq_y),
            "phase_rad": float(phase_rad),
        },
    )


def random_restart_path(extent_rad: float, n_points: int, seed: int) -> ScanPath:
    """``n_points`` positions drawn uniformly over the disc of radius ``extent_rad``.

    Uniform in area, not in radius: ``r = extent * sqrt(u)``. Seeded with
    ``numpy.random.default_rng(seed)`` so the path is reproducible.
    """
    if not math.isfinite(extent_rad) or extent_rad <= 0.0:
        raise ValueError(f"extent_rad must be finite and > 0, got {extent_rad!r}")
    if int(n_points) < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points!r}")
    rng = np.random.default_rng(int(seed))
    u = rng.random(int(n_points))
    angle = rng.random(int(n_points)) * 2.0 * math.pi
    r = extent_rad * np.sqrt(u)
    points = np.column_stack([r * np.cos(angle), r * np.sin(angle)])
    return ScanPath(
        name="random_restart",
        points_rad=points,
        metadata={
            "extent_rad": float(extent_rad),
            "n_points": float(int(n_points)),
            "seed": float(int(seed)),
        },
    )


def cell_visit_order(path: ScanPath, grid: CellGrid) -> np.ndarray:
    """Permutation of ``grid`` cell indices induced by walking ``path``.

    Each cell appears exactly once. Cells the path never enters are appended
    in nearest-neighbour order starting from the last visited cell, so every
    returned order is a complete scan of the grid and the four patterns differ
    only in sequence and slew cost.
    """
    assigned = grid.nearest_cell(path.points_rad)
    seen = np.zeros(grid.n_cells, dtype=bool)
    order: list[int] = []
    for index in assigned:
        if not seen[index]:
            seen[index] = True
            order.append(int(index))
    remaining = [int(i) for i in np.nonzero(~seen)[0]]
    current = grid.centres_rad[order[-1]] if order else np.zeros(2)
    while remaining:
        centres = grid.centres_rad[remaining]
        d2 = ((centres - current) ** 2).sum(axis=1)
        pick = int(np.argmin(d2))
        chosen = remaining.pop(pick)
        order.append(chosen)
        current = grid.centres_rad[chosen]
    return np.asarray(order, dtype=int)


def standard_paths(grid: CellGrid, seed: int = 20261016) -> dict[str, ScanPath]:
    """The four comparison paths sized to cover ``grid`` with its own footprint.

    The spiral pitch and the raster spacing are both set to the cell side, the
    Lissajous curve is sampled at eight points per cell and the random-restart
    path is drawn with ten draws per cell, which is enough for the induced
    orders to reach every cell before the nearest-neighbour tail is used in
    the configurations measured here. The tail length is reported by
    :func:`tail_fraction`.
    """
    side = grid.side_rad
    extent = grid.extent_rad
    turns = max(1.0, math.ceil(extent / side))
    return {
        "spiral": spiral_path(side, turns, 0.5 * side),
        "raster": raster_path(extent, side),
        "lissajous": lissajous_path(extent, 8 * grid.n_cells),
        "random_restart": random_restart_path(extent, 10 * grid.n_cells, seed),
    }


def tail_fraction(path: ScanPath, grid: CellGrid) -> float:
    """Fraction of cells reached only by the nearest-neighbour tail, dimensionless.

    Zero means the path itself entered every cell. A large value means the
    reported order is mostly not the named pattern, which is a property the
    caller should know about.
    """
    assigned = np.unique(grid.nearest_cell(path.points_rad))
    return float(1.0 - assigned.size / grid.n_cells)
