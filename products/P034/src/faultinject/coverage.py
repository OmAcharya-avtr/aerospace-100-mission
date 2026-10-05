"""Coverage accounting over the taxonomy cross product.

A *cell* is one combination of

    (kind, channel, start_frac bin, duration_frac bin, parameter bins)

Every executed case maps to exactly one cell, so coverage is the fraction of
cells hit at least once.  With the binning declared in
:mod:`faultinject.taxonomy` the full cross product is 248 cells, small enough
that ``validation/validate_coverage.py`` enumerates a subset by hand and
compares it against this module cell by cell.

What this metric does not claim: cell coverage says a region of the fault space
was *visited*, not that the target was *exercised* in any deeper sense, and
certainly not that no fault outside the taxonomy exists.  It is a campaign
bookkeeping measure.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from .faults import Injection
from .taxonomy import DURATION_FRAC, START_FRAC, FaultKind, iter_param_bins, kinds, spec


@dataclass(frozen=True, order=True)
class CoverageCell:
    """One cell of the taxonomy cross product."""

    kind: FaultKind
    channel: str
    start_bin: int
    duration_bin: int
    param_bins: tuple[int, ...]

    def label(self) -> str:
        """Compact human-readable identifier."""
        pb = ",".join(str(b) for b in self.param_bins)
        return f"{self.kind.value}/{self.channel}/s{self.start_bin}/d{self.duration_bin}/p[{pb}]"


def cell_of(injection: Injection, n_steps: int) -> CoverageCell:
    """Coverage cell an injection falls in, given the run length.

    ``start_frac = start_step / n_steps`` and
    ``duration_frac = duration_steps / n_steps``; both are clamped into their
    declared ranges by the binning, so a duration longer than the run lands in
    the last duration bin.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    sp = spec(injection.kind)
    start_frac = injection.start_step / n_steps
    duration_frac = injection.duration_steps / n_steps
    params = injection.params
    bins = tuple(p.bin_of(params[p.name]) for p in sp.params)
    return CoverageCell(
        kind=sp.kind,
        channel=injection.channel,
        start_bin=START_FRAC.bin_of(start_frac),
        duration_bin=DURATION_FRAC.bin_of(duration_frac),
        param_bins=bins,
    )


def iter_cells(subset: Sequence[FaultKind] | None = None) -> Iterator[CoverageCell]:
    """Enumerate every cell of ``subset`` (default: the whole taxonomy)."""
    ks = tuple(subset) if subset is not None else kinds()
    for kind in ks:
        sp = spec(kind)
        for channel in sp.channels:
            for sb in range(START_FRAC.n_bins):
                for db in range(DURATION_FRAC.n_bins):
                    for pb in iter_param_bins(kind):
                        yield CoverageCell(sp.kind, channel, sb, db, pb)


def all_cells(subset: Sequence[FaultKind] | None = None) -> tuple[CoverageCell, ...]:
    """Sorted tuple of every cell of ``subset``."""
    return tuple(sorted(iter_cells(subset)))


@dataclass
class CoverageTracker:
    """Accumulates which cells a campaign has visited."""

    subset: tuple[FaultKind, ...]

    def __post_init__(self) -> None:
        self._cells = all_cells(self.subset)
        self._index = {c: i for i, c in enumerate(self._cells)}
        self.hits: dict[CoverageCell, int] = {}

    @classmethod
    def full(cls) -> CoverageTracker:
        """Tracker over the whole taxonomy."""
        return cls(kinds())

    @property
    def total(self) -> int:
        """Number of cells in the tracked cross product."""
        return len(self._cells)

    @property
    def covered(self) -> int:
        """Number of distinct cells hit at least once."""
        return len(self.hits)

    @property
    def fraction(self) -> float:
        """``covered / total``, in ``[0, 1]``."""
        return self.covered / self.total if self.total else 0.0

    def cells(self) -> tuple[CoverageCell, ...]:
        """The tracked cells, in canonical sorted order."""
        return self._cells

    def add(self, injection: Injection, n_steps: int) -> CoverageCell:
        """Record one executed injection; returns the cell it landed in.

        Raises ``ValueError`` if the cell is outside the tracked subset, which
        would otherwise show up as a coverage fraction above one.
        """
        cell = cell_of(injection, n_steps)
        if cell not in self._index:
            raise ValueError(
                f"cell {cell.label()} is outside the tracked subset "
                f"{[k.value for k in self.subset]}"
            )
        self.hits[cell] = self.hits.get(cell, 0) + 1
        return cell

    def add_all(self, injections: Iterable[Injection], n_steps: int) -> None:
        """Record many injections."""
        for inj in injections:
            self.add(inj, n_steps)

    def missing(self) -> tuple[CoverageCell, ...]:
        """Cells not yet hit, in canonical order."""
        return tuple(c for c in self._cells if c not in self.hits)

    def per_kind(self) -> dict[FaultKind, tuple[int, int]]:
        """``kind -> (covered, total)``."""
        out: dict[FaultKind, tuple[int, int]] = {}
        for kind in self.subset:
            cells = [c for c in self._cells if c.kind is kind]
            cov = sum(1 for c in cells if c in self.hits)
            out[kind] = (cov, len(cells))
        return out
