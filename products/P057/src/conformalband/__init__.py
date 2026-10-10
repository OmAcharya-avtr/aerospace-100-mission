"""conformalband: distribution-free prediction intervals audited under declared shift.

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use.

The package is a measurement harness, not a new conformal primitive. Split,
Mondrian and weighted conformal are implemented here from their published
definitions so that one audit can run all three plus the parametric baseline on
identical data; for production conformal prediction use ``MAPIE`` or
``crepes``, which are named and recommended in README.md.
"""

from __future__ import annotations

from .audit import (
    METHODS,
    MODELS,
    AuditResult,
    BreakingPointResult,
    BreakingPointRow,
    CoverageRow,
    breaking_point_sweep,
    coverage_audit,
    stratified_coverage,
)
from .baseline import GaussianResidualInterval, PhysicsRegressor
from .bounds import (
    CoverageBound,
    clopper_pearson,
    effective_sample_size,
    split_conformal_coverage_bound,
)
from .conformal import (
    ConformalizedRegressor,
    Interval,
    MondrianConformal,
    SplitConformal,
    WeightedConformal,
    absolute_residual_score,
    assign_bins,
    conformal_rank,
    tercile_edges,
    weighted_quantile,
)
from .data import FEATURE_NAMES, FEATURE_UNITS, AuditSplit, Dataset, make_audit_split, make_dataset
from .learned import LearnedRegressor, LearnedWeightEstimator
from .physics import (
    DEFAULT_AIRFRAME,
    Airframe,
    leg_energy,
    level_flight_power,
    propulsive_efficiency,
)
from .shift import SHIFT_LEVELS, CovariateShift

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_AIRFRAME",
    "FEATURE_NAMES",
    "FEATURE_UNITS",
    "METHODS",
    "MODELS",
    "SHIFT_LEVELS",
    "Airframe",
    "AuditResult",
    "AuditSplit",
    "BreakingPointResult",
    "BreakingPointRow",
    "ConformalizedRegressor",
    "CoverageBound",
    "CoverageRow",
    "CovariateShift",
    "Dataset",
    "GaussianResidualInterval",
    "Interval",
    "LearnedRegressor",
    "LearnedWeightEstimator",
    "MondrianConformal",
    "PhysicsRegressor",
    "SplitConformal",
    "WeightedConformal",
    "__version__",
    "absolute_residual_score",
    "assign_bins",
    "breaking_point_sweep",
    "clopper_pearson",
    "conformal_rank",
    "coverage_audit",
    "effective_sample_size",
    "leg_energy",
    "level_flight_power",
    "make_audit_split",
    "make_dataset",
    "propulsive_efficiency",
    "split_conformal_coverage_bound",
    "stratified_coverage",
    "tercile_edges",
    "weighted_quantile",
]
