"""Grid-cell occupancy over the discretised space, and the largest empty block.

What is computed
----------------
Take a subset of the declared parameters, take the declared discretisation of
each, and form the product grid. With ``C`` cells in total and a campaign of
``n`` scenarios,

    occupancy = (number of distinct cells containing at least one scenario) / C

Both the numerator and ``C`` are exact integers; the cell index of a scenario
comes from the integer level indices produced by
:meth:`ScenarioSpace.encode_many`, so no floating-point comparison decides
whether a cell is occupied. The float caveat lives one step earlier, in the
binning itself, and is reported by :meth:`ScenarioSpace.edge_ambiguities`.

**The discretisation is an assumption.** Occupancy is a statement about the
grid you declared, not about the continuous space. Doubling the number of bins
per parameter in a d-dimensional grid multiplies ``C`` by ``2^d`` and can only
lower the occupancy of a fixed campaign. This module therefore reports ``C``,
the per-parameter bin counts and the scenario count next to every occupancy
figure, and the reports print the discretisation alongside the number.

Reference point for an unplanned campaign
-----------------------------------------
If ``n`` scenarios are thrown independently and uniformly into ``C`` equally
likely cells, the expected number of occupied cells is

    E[occupied] = C * (1 - (1 - 1/C)^n)                                  (3)

the classical occupancy-problem result: Feller, *An Introduction to Probability
Theory and Its Applications*, Volume I, 3rd edition, Wiley, 1968, Chapter II
(occupancy and matching problems) -- the cell is empty with probability
``(1 - 1/C)^n`` and expectation is additive over cells, which needs no
independence between cells. Equation (3) is the yardstick a *uniform random*
campaign should meet; it is **not** a target, because a deliberately designed
campaign should beat it and a campaign concentrated on nominal cases will fall
below it. Units: cells (dimensionless counts). Validity: exact for independent
uniform sampling with replacement over equiprobable cells; the fixture campaign
in this repository is not uniform, which is the point of measuring.

Largest empty block
-------------------
``largest_empty_block`` is the size of the largest connected component of empty
cells under von Neumann connectivity (two cells adjacent when they differ by
one in exactly one coordinate). It is computed by an integer flood fill and is
exact. A large block in a campaign that already has high occupancy is the
signature of a contiguous untested region, which is what a planted gap looks
like; scattered single empty cells are not.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .space import ScenarioSpace

__all__ = [
    "CellOccupancy",
    "EmptyBlock",
    "expected_occupied_cells",
    "grid_occupancy",
]

DEFAULT_MAX_CELLS = 5_000_000


@dataclass(frozen=True)
class EmptyBlock:
    """A connected component of empty cells.

    Attributes:
        size: number of cells in the component (exact integer).
        cells: the cell coordinates, as level-index tuples over the grid
            parameters, sorted.
    """

    size: int
    cells: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class CellOccupancy:
    """Result of :func:`grid_occupancy`.

    Attributes:
        parameters: the parameters forming the grid, in order.
        shape: bins per grid parameter.
        n_cells: ``C``, the product of ``shape``.
        n_scenarios: number of scenarios placed in the grid.
        occupied: number of distinct non-empty cells.
        counts: integer count per cell, shaped ``shape``.
    """

    parameters: tuple[str, ...]
    shape: tuple[int, ...]
    n_cells: int
    n_scenarios: int
    occupied: int
    counts: np.ndarray

    @property
    def fraction(self) -> float:
        """``occupied / n_cells``, dimensionless in [0, 1]."""
        return self.occupied / self.n_cells

    @property
    def empty(self) -> int:
        """``n_cells - occupied``, exact integer."""
        return self.n_cells - self.occupied

    @property
    def expected_fraction_uniform(self) -> float:
        """Occupancy equation (3) would predict for a uniform random campaign."""
        return expected_occupied_cells(self.n_scenarios, self.n_cells) / self.n_cells

    def empty_cells(self) -> tuple[tuple[int, ...], ...]:
        """Every empty cell, as a level-index tuple. Enumerated, not sampled."""
        empty = np.nonzero(self.counts == 0)
        return tuple(tuple(int(i) for i in idx) for idx in zip(*empty, strict=True))

    def empty_blocks(self) -> tuple[EmptyBlock, ...]:
        """Connected components of empty cells, largest first (von Neumann)."""
        empty = self.counts == 0
        seen = np.zeros_like(empty, dtype=bool)
        blocks: list[EmptyBlock] = []
        for start in zip(*np.nonzero(empty), strict=True):
            if seen[start]:
                continue
            component: list[tuple[int, ...]] = []
            queue: deque[tuple[int, ...]] = deque([tuple(int(i) for i in start)])
            seen[start] = True
            while queue:
                cell = queue.popleft()
                component.append(cell)
                for axis, extent in enumerate(self.shape):
                    for step in (-1, 1):
                        nxt = list(cell)
                        nxt[axis] += step
                        if 0 <= nxt[axis] < extent:
                            key = tuple(nxt)
                            if empty[key] and not seen[key]:
                                seen[key] = True
                                queue.append(key)
            blocks.append(EmptyBlock(size=len(component), cells=tuple(sorted(component))))
        return tuple(sorted(blocks, key=lambda b: (-b.size, b.cells)))

    def largest_empty_block(self) -> int:
        """Size of the largest empty connected component; 0 when nothing is empty."""
        blocks = self.empty_blocks()
        return blocks[0].size if blocks else 0


def expected_occupied_cells(n_scenarios: int, n_cells: int) -> float:
    """Equation (3): expected occupied cells for uniform random throws.

    Args:
        n_scenarios: number of scenarios, ``n >= 0``.
        n_cells: number of equiprobable cells, ``C >= 1``.

    Returns:
        ``C * (1 - (1 - 1/C)^n)``, in cells (dimensionless). Exact for the
        stated model; carries ordinary floating-point rounding.
    """
    if n_scenarios < 0:
        raise ValueError("n_scenarios must be non-negative")
    if n_cells < 1:
        raise ValueError("n_cells must be at least 1")
    return float(n_cells) * (1.0 - math.pow(1.0 - 1.0 / n_cells, n_scenarios))


def grid_occupancy(
    space: ScenarioSpace,
    encoded: np.ndarray,
    parameters: Sequence[str] | None = None,
    max_cells: int = DEFAULT_MAX_CELLS,
) -> CellOccupancy:
    """Occupancy of the product grid over a subset of the declared parameters.

    Args:
        space: the declared scenario space.
        encoded: ``(n_scenarios, n_parameters)`` integer level indices.
        parameters: the grid parameters. Defaults to every continuous
            parameter in the space, since grid occupancy is the measure that
            exists because continuous parameters had to be discretised.
        max_cells: refuse to allocate a larger grid than this.

    Returns:
        A :class:`CellOccupancy`.
    """
    if parameters is None:
        chosen = tuple(p.name for p in space.parameters if p.kind == "continuous")
        if not chosen:
            raise ValueError(
                "the space declares no continuous parameters; pass parameters= explicitly"
            )
    else:
        chosen = tuple(parameters)
        if not chosen:
            raise ValueError("parameters must name at least one parameter")
        if len(set(chosen)) != len(chosen):
            raise ValueError("parameters must not repeat a name")

    columns = [space.index_of(name) for name in chosen]
    shape = tuple(space.parameters[c].n_levels for c in columns)
    n_cells = math.prod(shape)
    if n_cells > max_cells:
        raise ValueError(
            f"grid of {n_cells} cells exceeds max_cells={max_cells}; choose fewer "
            "parameters, coarser bins, or raise max_cells deliberately"
        )

    array = np.asarray(encoded)
    if array.ndim != 2 or array.shape[1] != len(space):
        raise ValueError(f"encoded must have shape (n_scenarios, {len(space)}), got {array.shape}")
    counts = np.zeros(shape, dtype=np.int64)
    if array.size:
        sub = array[:, columns].astype(np.int64, copy=False)
        if sub.min() < 0:
            raise ValueError("encoded contains a negative level index")
        for axis, extent in enumerate(shape):
            if sub[:, axis].max() >= extent:
                raise ValueError(f"level index out of range for parameter {chosen[axis]!r}")
        flat = np.ravel_multi_index(tuple(sub[:, i] for i in range(sub.shape[1])), shape)
        counts = np.bincount(flat, minlength=n_cells).astype(np.int64).reshape(shape)

    return CellOccupancy(
        parameters=chosen,
        shape=shape,
        n_cells=n_cells,
        n_scenarios=int(array.shape[0]),
        occupied=int(np.count_nonzero(counts)),
        counts=counts,
    )
