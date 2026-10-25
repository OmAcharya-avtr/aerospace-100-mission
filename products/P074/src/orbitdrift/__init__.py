"""orbitdrift: how wrong an ephemeris has become, as a measured curve against epoch age.

Read this first
---------------
This package does **not** measure the error of a two-line element set. It
cannot: it has no precision ephemerides and no element-set archive, so it has
nothing to call truth. What it measures is **model divergence** -- how fast two
propagations of the same object separate -- and the only rigorous statement
that supports is a triangle inequality: if two propagators disagree by ``D``,
then the larger of their two unknown errors is at least ``D/2``.
:mod:`orbitdrift.divergence` derives that and says what it is not.

Nothing here is a covariance, and nothing here may be used as the input
covariance of a conjunction assessment. That is enforced in code and tested;
see :mod:`orbitdrift.guard`, which names P071 ConjProb of this mission as the
product that computes collision probability and states that its covariance has
to come from orbit determination.

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use.

The two baselines and the learned component
-------------------------------------------
``baseline 1``
    :func:`orbitdrift.divergence.self_consistency_divergence` -- the same
    object's earlier element set against its later one, both supplied by the
    user. The standard practical proxy.
``baseline 2``
    :func:`orbitdrift.divergence.model_divergence` -- SGP4 against the
    independent numerical force model of :mod:`orbitdrift.forces`, with J2 to
    J4, a declared static atmosphere and luni-solar point masses.
``analytic growth``
    :mod:`orbitdrift.growth` and :class:`orbitdrift.predictors.AnalyticGrowthPredictor`
    -- the linear, quadratic, free-power and drag-exponential forms, each
    derived from a stated dominant error term. Implemented and validated
    first.
``learned``
    :class:`orbitdrift.predictors.LearnedQuantilePredictor` -- gradient-boosted
    quantile regression over the same features, benchmarked against the above
    on measured interval coverage and width by
    :func:`orbitdrift.benchmark.run_benchmark`.

The decision the product exists for
-----------------------------------
:func:`orbitdrift.staleness.staleness_verdict` answers "is this element set
still good enough for what I am about to do", as an epoch age at which the
upper end of a prediction interval reaches a declared position tolerance, with
the measured coverage of that interval attached.
"""

from __future__ import annotations

from .benchmark import BenchmarkResult, PredictorRow, run_benchmark
from .calibration import (
    CoverageCell,
    IntervalMetrics,
    age_stratum,
    altitude_stratum,
    clopper_pearson,
    coverage_by_group,
    coverage_table,
    interval_metrics,
)
from .constants import WGS72, WGS84, GravityModel
from .dataset import (
    POPULATION_SPEC,
    DivergenceTable,
    build_divergence_table,
    load_population,
    split_by_object,
    synthetic_population_tle,
)
from .divergence import (
    DivergenceCurve,
    larger_error_lower_bound_km,
    model_divergence,
    self_consistency_divergence,
    sgp4_states,
    window_rms,
)
from .errors import (
    CovarianceRefused,
    InsufficientDataError,
    InvalidTleError,
    OrbitDriftError,
    PropagationError,
)
from .features import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES,
    ObjectFeatures,
    design_matrix,
    features_from_satrec,
)
from .forces import ForceModel, atmospheric_density_kg_m3, cd_area_over_mass_m2_kg
from .growth import FORMS, GrowthFit, best_form, fit_all_forms, fit_power_law
from .guard import check_purpose, refuse_covariance
from .integrate import Trajectory, invariant_residuals, propagate, specific_energy_km2_s2
from .kepler import ClassicalElements, propagate_two_body, rv_to_elements
from .predictors import (
    ANALYTIC_FORMS,
    AnalyticGrowthPredictor,
    ConformalPredictor,
    LearnedQuantilePredictor,
    RuleOfThumbPredictor,
)
from .report import DriftReport, build_report
from .staleness import StalenessVerdict, staleness_verdict
from .vectors import PublishedCase, compare_to_published, published_cases

__version__ = "0.1.0"

SAFETY_STATEMENT = (
    "orbitdrift is research-grade software. It is not flight-qualified, not "
    "certified, and not approved for operational aerospace use. It measures "
    "model divergence between two propagators; it does not measure ephemeris "
    "error against truth, and no output of it is a covariance or an input to a "
    "conjunction assessment."
)

__all__ = [
    "ALL_FEATURE_NAMES",
    "ANALYTIC_FORMS",
    "FEATURE_NAMES",
    "FORMS",
    "POPULATION_SPEC",
    "SAFETY_STATEMENT",
    "WGS72",
    "WGS84",
    "AnalyticGrowthPredictor",
    "BenchmarkResult",
    "ClassicalElements",
    "ConformalPredictor",
    "CovarianceRefused",
    "CoverageCell",
    "DivergenceCurve",
    "DivergenceTable",
    "DriftReport",
    "ForceModel",
    "GravityModel",
    "GrowthFit",
    "InsufficientDataError",
    "IntervalMetrics",
    "InvalidTleError",
    "LearnedQuantilePredictor",
    "ObjectFeatures",
    "OrbitDriftError",
    "PredictorRow",
    "PropagationError",
    "PublishedCase",
    "RuleOfThumbPredictor",
    "StalenessVerdict",
    "Trajectory",
    "__version__",
    "age_stratum",
    "altitude_stratum",
    "atmospheric_density_kg_m3",
    "best_form",
    "build_divergence_table",
    "build_report",
    "cd_area_over_mass_m2_kg",
    "check_purpose",
    "clopper_pearson",
    "compare_to_published",
    "coverage_by_group",
    "coverage_table",
    "design_matrix",
    "features_from_satrec",
    "fit_all_forms",
    "fit_power_law",
    "interval_metrics",
    "invariant_residuals",
    "larger_error_lower_bound_km",
    "load_population",
    "model_divergence",
    "propagate",
    "propagate_two_body",
    "published_cases",
    "refuse_covariance",
    "run_benchmark",
    "rv_to_elements",
    "self_consistency_divergence",
    "sgp4_states",
    "specific_energy_km2_s2",
    "split_by_object",
    "staleness_verdict",
    "synthetic_population_tle",
    "window_rms",
]
