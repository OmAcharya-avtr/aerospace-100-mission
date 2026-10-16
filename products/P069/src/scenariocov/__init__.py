"""scenariocov: which regions of the scenario space a finished campaign did not touch.

Research-grade software. Not flight-qualified, not certified, not approved for
operational aerospace use.

The package is in two layers, and the order matters:

1. a deterministic core -- :mod:`scenariocov.combinatorial` (t-way coverage
   with every uncovered tuple enumerated), :mod:`scenariocov.classes`
   (equivalence-class and boundary coverage against declared classes) and
   :mod:`scenariocov.occupancy` (grid occupancy over the discretised
   continuous parameters), assembled by :mod:`scenariocov.report`;
2. a learned gap finder, :mod:`scenariocov.gapfinder`, which proposes the next
   batch of scenarios and is benchmarked in :mod:`scenariocov.benchmark`
   against uniform random and greedy max-min-distance selection.

Read README.md before reading the benchmark numbers: on the bundled fixtures
the greedy max-min baseline wins.
"""

from __future__ import annotations

from .benchmark import (
    BenchmarkResult,
    StrategyCurves,
    coverage_benchmark,
    incremental_coverage_curve,
)
from .classes import (
    BoundaryHit,
    BoundarySpec,
    ClassCoverage,
    ClassHit,
    EquivalenceClass,
    boundary_coverage,
    class_coverage,
)
from .combinatorial import (
    CombinationCoverage,
    TwayCoverage,
    UncoveredTuple,
    count_tway_tuples,
    tway_coverage,
)
from .gapfinder import GapFinder, GapScores, Proposal
from .io import (
    load_campaign,
    load_class_document,
    load_space,
    save_campaign,
    save_space,
)
from .occupancy import CellOccupancy, EmptyBlock, expected_occupied_cells, grid_occupancy
from .report import CampaignReport, build_report
from .space import EdgeAmbiguity, Parameter, ScenarioSpace
from .strategies import (
    CoverageGreedyReference,
    GreedyMaxMinStrategy,
    LearnedGapFinderStrategy,
    SelectionContext,
    Strategy,
    UniformRandomStrategy,
    default_strategies,
)

__version__ = "0.1.0"

__all__ = [
    "BenchmarkResult",
    "BoundaryHit",
    "BoundarySpec",
    "CampaignReport",
    "CellOccupancy",
    "ClassCoverage",
    "ClassHit",
    "CombinationCoverage",
    "CoverageGreedyReference",
    "EdgeAmbiguity",
    "EmptyBlock",
    "EquivalenceClass",
    "GapFinder",
    "GapScores",
    "GreedyMaxMinStrategy",
    "LearnedGapFinderStrategy",
    "Parameter",
    "Proposal",
    "ScenarioSpace",
    "SelectionContext",
    "Strategy",
    "StrategyCurves",
    "TwayCoverage",
    "UncoveredTuple",
    "UniformRandomStrategy",
    "__version__",
    "boundary_coverage",
    "build_report",
    "class_coverage",
    "count_tway_tuples",
    "coverage_benchmark",
    "default_strategies",
    "expected_occupied_cells",
    "grid_occupancy",
    "incremental_coverage_curve",
    "load_campaign",
    "load_class_document",
    "load_space",
    "save_campaign",
    "save_space",
    "tway_coverage",
]
