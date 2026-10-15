"""The UCA enumeration grid: every control action against all four classes.

STPA step 3 asks, for each control action, whether each of the four classes of
unsafe control action applies (STPA Handbook, Leveson and Thomas, 2018). The
grid is therefore ``n_control_actions x 4`` cells and its size is known in
advance from the control structure alone, which is what makes completeness
checkable rather than assertable.

A cell is **covered** when at least one UCA occupies it, or when an explicit
justified exclusion declares that it has been considered and found not to
apply. A cell that is neither is an **enumeration gap**: the analysis does not
record whether anybody looked.

Units: counts and dimensionless ratios only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .model import UCA_CLASSES, Analysis, UcaClass


class CellState(StrEnum):
    """How one grid cell is covered."""

    UCA = "uca"
    EXCLUDED = "excluded"
    CONFLICT = "conflict"
    GAP = "gap"

    @property
    def is_covered(self) -> bool:
        """True when the cell records a decision of any kind.

        A conflicting cell counts as covered for the completeness ratio -
        somebody did look at it - and is reported separately as a
        contradiction, because the two records disagree.
        """
        return self is not CellState.GAP


@dataclass(frozen=True)
class Cell:
    """One (control action, UCA class) cell of the enumeration grid."""

    control_action: str
    uca_class: UcaClass
    uca_ids: tuple[str, ...]
    excluded: bool
    justification: str = ""

    @property
    def state(self) -> CellState:
        """Coverage state of this cell."""
        if self.uca_ids and self.excluded:
            return CellState.CONFLICT
        if self.uca_ids:
            return CellState.UCA
        if self.excluded:
            return CellState.EXCLUDED
        return CellState.GAP


@dataclass(frozen=True)
class EnumerationGrid:
    """The whole grid, in declaration order of control actions.

    Attributes
    ----------
    cells : tuple of Cell
        Length ``n_control_actions * 4``, ordered control action major, UCA
        class minor, with the class order of :data:`hazardtrace.model.UCA_CLASSES`.
    """

    cells: tuple[Cell, ...]

    @property
    def total(self) -> int:
        """Number of cells, that is ``n_control_actions * 4``."""
        return len(self.cells)

    @property
    def covered(self) -> int:
        """Number of cells holding a UCA, an exclusion, or both."""
        return sum(1 for c in self.cells if c.state.is_covered)

    @property
    def gaps(self) -> tuple[Cell, ...]:
        """Cells with neither a UCA nor an exclusion, in grid order."""
        return tuple(c for c in self.cells if c.state is CellState.GAP)

    @property
    def conflicts(self) -> tuple[Cell, ...]:
        """Cells holding both a UCA and an exclusion, in grid order."""
        return tuple(c for c in self.cells if c.state is CellState.CONFLICT)

    def count(self, state: CellState) -> int:
        """Number of cells in one state."""
        return sum(1 for c in self.cells if c.state is state)

    def completeness(self) -> tuple[int, int]:
        """Return ``(covered, total)``. The caller prints the denominator."""
        return (self.covered, self.total)

    def cell(self, control_action: str, uca_class: UcaClass) -> Cell | None:
        """The cell for one control action and class, or None if absent."""
        for item in self.cells:
            if item.control_action == control_action and item.uca_class is uca_class:
                return item
        return None


def build_grid(analysis: Analysis) -> EnumerationGrid:
    """Build the enumeration grid for a parsed analysis.

    The grid has exactly ``4 * len(analysis.structure.control_actions)`` cells
    by construction, independently of how many UCAs were declared.
    """
    by_cell: dict[tuple[str, UcaClass], list[str]] = {}
    for uca in analysis.ucas:
        by_cell.setdefault(uca.cell, []).append(uca.id)
    excluded: dict[tuple[str, UcaClass], str] = {
        item.cell: item.justification for item in analysis.exclusions
    }

    cells: list[Cell] = []
    for action in analysis.structure.control_actions:
        for uca_class in UCA_CLASSES:
            key = (action.id, uca_class)
            cells.append(
                Cell(
                    control_action=action.id,
                    uca_class=uca_class,
                    uca_ids=tuple(by_cell.get(key, ())),
                    excluded=key in excluded,
                    justification=excluded.get(key, ""),
                )
            )
    return EnumerationGrid(cells=tuple(cells))
