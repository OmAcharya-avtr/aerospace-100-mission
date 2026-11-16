"""Gate branch reachability, counted from the campaign and asserted, not assumed.

SPEC §9 requires a test asserting that **every one of the five gate branches
fires at least once across the campaign**. P093 shipped a gate two of whose
five reject reasons never fired in 268425 inferences, and that is recorded in
this mission as a weak result: an acceptance figure from a gate whose rejection
paths were never exercised is a figure about a narrower gate than the one in the
source.

The five branches are ``accept`` plus the four
``pat_pipeline.REJECT_REASONS`` — ``no_candidate``, ``low_score``, ``area_out``,
``border`` — and ``runs.csv`` carries all five per cell.

The counting rule, stated because it is not the obvious one
-----------------------------------------------------------
``staged_exec.run_timed`` counts **per candidate** for the three
candidate-level rejections and **per frame** for ``no_candidate`` and
``accept``. One frame with eight candidates can therefore contribute up to
eight ``low_score`` counts and one ``accept``. The totals are not a partition
of the frames and are not reported as one; the per-frame acceptance rate is
``accept / n_frames`` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .sources import reject_reasons
from .traces import GATE_BRANCHES, Campaign


@dataclass(frozen=True)
class GateCensus:
    """Per-cell and campaign-total gate branch counts."""

    branches: tuple[str, ...]
    per_cell: pd.DataFrame
    totals: dict[str, int]
    n_frames_total: int
    cells: tuple[str, ...]

    @property
    def silent_branches(self) -> tuple[str, ...]:
        """Branches that never fired across the whole campaign."""
        return tuple(name for name in self.branches if self.totals[name] == 0)

    @property
    def all_branches_fired(self) -> bool:
        """Whether every branch fired at least once across the campaign."""
        return not self.silent_branches

    def cells_where_silent(self, branch: str) -> tuple[str, ...]:
        """Cells in which ``branch`` never fired."""
        if branch not in self.branches:
            raise KeyError(f"{branch!r} is not one of {', '.join(self.branches)}")
        column = self.per_cell.set_index("cell")[branch]
        return tuple(str(cell) for cell, value in column.items() if int(value) == 0)

    @property
    def branches_firing_in_every_cell(self) -> tuple[str, ...]:
        """Branches that fired at least once in every one of the nine cells."""
        return tuple(
            name for name in self.branches if not self.cells_where_silent(name)
        )

    @property
    def acceptance_rate(self) -> float:
        """Frames whose gate accepted a candidate, as a fraction of all frames."""
        return float(self.totals["accept"]) / float(self.n_frames_total)

    @property
    def rejection_counts(self) -> dict[str, int]:
        """The four candidate-or-frame level rejection counts, without accept."""
        return {name: self.totals[name] for name in self.branches if name != "accept"}

    def frame(self) -> pd.DataFrame:
        """Per-cell counts with a totals row appended."""
        table = self.per_cell.copy()
        total_row = {"cell": "TOTAL", "n_frames": self.n_frames_total}
        total_row.update({name: self.totals[name] for name in self.branches})
        return pd.concat([table, pd.DataFrame([total_row])], ignore_index=True)


def gate_census(campaign: Campaign) -> GateCensus:
    """Count every gate branch per cell and over the whole campaign.

    Raises
    ------
    ValueError
        If ``runs.csv`` lacks a branch column, or if the committed board source
        declares a reject reason the campaign did not record. The branch set is
        taken from the board source and checked against the trace, so a product
        counting four branches against a five-branch gate is a refusal.
    """
    runs = campaign.runs
    missing = [name for name in GATE_BRANCHES if name not in runs.columns]
    if missing:
        raise ValueError(
            f"{campaign.label}/runs.csv has no column for gate branch(es) "
            f"{', '.join(missing)}. Columns present: {', '.join(runs.columns)}."
        )
    source_reasons = reject_reasons()
    unrecorded = [name for name in source_reasons if name not in GATE_BRANCHES]
    if unrecorded:
        raise ValueError(
            "the committed board source declares reject reason(s) the campaign "
            f"did not record: {', '.join(unrecorded)}"
        )
    columns = ["cell", "n_frames", *GATE_BRANCHES]
    per_cell = runs[columns].copy()
    totals = {
        name: int(per_cell[name].to_numpy(dtype=np.int64).sum()) for name in GATE_BRANCHES
    }
    return GateCensus(
        branches=GATE_BRANCHES,
        per_cell=per_cell,
        totals=totals,
        n_frames_total=int(per_cell["n_frames"].to_numpy(dtype=np.int64).sum()),
        cells=tuple(str(value) for value in per_cell["cell"]),
    )
