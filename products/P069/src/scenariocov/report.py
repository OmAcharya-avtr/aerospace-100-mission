"""The coverage report for a campaign that has already run.

:class:`CampaignReport` is the deterministic core assembled in one place:
1-way and t-way combinatorial coverage with every uncovered tuple enumerated,
equivalence-class and boundary coverage against declared classes, and grid
occupancy over the continuous parameters. No model is involved; nothing here
proposes anything.

Every rendered report repeats the discretisation next to the numbers, because
every number in it is a number about the binned space (see
:mod:`scenariocov.occupancy`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .classes import BoundaryHit, BoundarySpec, ClassCoverage, EquivalenceClass
from .classes import boundary_coverage as _boundary_coverage
from .classes import class_coverage as _class_coverage
from .combinatorial import TwayCoverage, tway_coverage
from .occupancy import CellOccupancy, grid_occupancy
from .space import EdgeAmbiguity, ScenarioSpace

__all__ = ["CampaignReport", "build_report"]


@dataclass(frozen=True)
class CampaignReport:
    """Everything the deterministic core measures about one finished campaign."""

    space: ScenarioSpace
    n_scenarios: int
    strength: int
    one_way: TwayCoverage
    tway: TwayCoverage
    occupancy: CellOccupancy | None
    classes: ClassCoverage | None
    boundaries: tuple[BoundaryHit, ...]
    edge_ambiguities: tuple[EdgeAmbiguity, ...]

    def discretisation_text(self) -> str:
        """One line per parameter describing the declared discretisation."""
        lines = []
        for p in self.space.parameters:
            if p.kind == "continuous":
                edges = ", ".join(f"{e:g}" for e in p.edges)
                unit = f" {p.units}" if p.units else ""
                lines.append(f"  {p.name:<22} continuous, {p.n_levels} bins, edges [{edges}]{unit}")
            else:
                lines.append(
                    f"  {p.name:<22} categorical, {p.n_levels} levels: {', '.join(p.levels)}"
                )
        return "\n".join(lines)

    def to_text(self, max_uncovered: int = 20) -> str:
        """Render the report. ``max_uncovered`` caps only the printed listing.

        The uncovered tuples are all held in ``self.tway.uncovered``; the cap
        exists so a terminal is not flooded, and the rendered text says how
        many were withheld.
        """
        if max_uncovered < 0:
            raise ValueError("max_uncovered must be non-negative")
        out: list[str] = []
        out.append(f"scenariocov report: {self.n_scenarios} executed scenarios")
        out.append("=" * 78)
        out.append("Discretisation (an assumption imposed by this report, not a measurement):")
        out.append(self.discretisation_text())
        out.append(
            f"  fully crossed discretised space: {self.space.n_combinations} combinations"
        )
        out.append("")
        out.append(
            f"1-way coverage  {self.one_way.covered}/{self.one_way.total} "
            f"= {self.one_way.fraction:.6f}"
        )
        out.append(
            f"{self.strength}-way coverage  {self.tway.covered}/{self.tway.total} "
            f"= {self.tway.fraction:.6f}   ({self.tway.n_uncovered} uncovered)"
        )
        if self.tway.uncovered:
            out.append("")
            out.append(f"Uncovered {self.strength}-way tuples, enumerated:")
            for u in self.tway.uncovered[:max_uncovered]:
                out.append(f"  {u.as_text()}")
            held = len(self.tway.uncovered) - max_uncovered
            if held > 0:
                out.append(f"  ... {held} more, available in report.tway.uncovered")
        if self.occupancy is not None:
            occ = self.occupancy
            out.append("")
            out.append(
                f"Grid occupancy over ({', '.join(occ.parameters)}): "
                f"{occ.occupied}/{occ.n_cells} = {occ.fraction:.6f}"
            )
            out.append(
                f"  uniform-random reference, Feller eq. (3): "
                f"{occ.expected_fraction_uniform:.6f}"
            )
            out.append(f"  largest empty connected block: {occ.largest_empty_block()} cells")
        if self.classes is not None:
            out.append("")
            out.append(
                f"Equivalence-class coverage {self.classes.covered}/{self.classes.total} "
                f"= {self.classes.fraction:.6f}"
            )
            for cls in self.classes.uncovered:
                out.append(f"  UNCOVERED CLASS  {cls.parameter}/{cls.name}")
        if self.boundaries:
            out.append("")
            out.append("Boundary coverage (nearest distance shown; the hit test is a")
            out.append("floating-point comparison, so read the margin, not just the flag):")
            for hit in self.boundaries:
                flag = "hit " if hit.covered else "MISS"
                out.append(
                    f"  {flag} {hit.spec.parameter}/{hit.spec.name:<18} "
                    f"n={hit.count:<4} nearest={hit.nearest_distance:.6g} "
                    f"tol={hit.spec.tolerance:g} margin={hit.margin:+.6g}"
                )
        if self.edge_ambiguities:
            out.append("")
            out.append(
                f"{len(self.edge_ambiguities)} record(s) sit within tolerance of a bin edge; "
                "their bin assignment is not guaranteed."
            )
        return "\n".join(out)


def build_report(
    space: ScenarioSpace,
    records: Sequence[Mapping[str, Any]],
    strength: int = 2,
    classes: Sequence[EquivalenceClass] | None = None,
    boundaries: Sequence[BoundarySpec] | None = None,
    occupancy_parameters: Sequence[str] | None = None,
    edge_rtol: float = 1e-9,
) -> CampaignReport:
    """Measure everything the deterministic core measures.

    Args:
        space: the declared scenario space.
        records: the executed campaign.
        strength: interaction strength for the t-way measure, ``>= 1``.
        classes: declared equivalence classes, or None to skip that section.
        boundaries: declared boundaries, or None to skip that section.
        occupancy_parameters: grid parameters; defaults to the continuous ones,
            and the occupancy section is skipped when the space has none.
        edge_rtol: tolerance for the bin-edge ambiguity report.

    Returns:
        A :class:`CampaignReport`.
    """
    if not records:
        raise ValueError("the campaign is empty; there is nothing to report")
    encoded = space.encode_many(records)
    has_continuous = any(p.kind == "continuous" for p in space.parameters)
    occupancy = None
    if occupancy_parameters is not None or has_continuous:
        occupancy = grid_occupancy(space, encoded, occupancy_parameters)
    return CampaignReport(
        space=space,
        n_scenarios=len(records),
        strength=strength,
        one_way=tway_coverage(space, encoded, t=1),
        tway=tway_coverage(space, encoded, t=strength),
        occupancy=occupancy,
        classes=_class_coverage(classes, records) if classes else None,
        boundaries=_boundary_coverage(boundaries, records) if boundaries else (),
        edge_ambiguities=space.edge_ambiguities(records, rtol=edge_rtol),
    )
