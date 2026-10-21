"""conjprob: collision probability with the validity regime stated next to the number.

The package computes the probability of collision for a close approach by
several independent routes and reports **their disagreement as the primary
output**. A single Pc is a choice of assumptions, and the assumptions are where
the operational mistakes live, so the public surface makes the choice explicit:
:class:`~conjprob.report.ConjunctionReport` has no scalar ``pc`` attribute.

What is here
------------
=============================== ================================================
:mod:`conjprob.geometry`        time of closest approach, encounter plane, projection
:mod:`conjprob.analytic`        Foster-Estes, Chan series, Alfano, small-body limit
:mod:`conjprob.quadrature`      independent 2-D quadrature reference
:mod:`conjprob.montecarlo`      3-D Monte Carlo reference with exact intervals
:mod:`conjprob.dilution`        the dilution region, its closed-form maximum, the warning
:mod:`conjprob.nonlinear`       measured breakdown of the rectilinear assumption
:mod:`conjprob.regimes`         measured validity boundaries of each analytic method
:mod:`conjprob.report`          the assembled report
:mod:`conjprob.features`        cheap screening features
:mod:`conjprob.screens`         analytic screens, implemented first
:mod:`conjprob.screening`       learned classifier and the comparison harness
:mod:`conjprob.dataset`         declared synthetic conjunction population
:mod:`conjprob.tle`             optional TLE state entry point via ``sgp4``
=============================== ================================================

Status: research grade. Not flight-qualified, not certified, not approved for
operational aerospace use. No output of this package is suitable as an input to
a manoeuvre decision.
"""

from __future__ import annotations

from .analytic import (
    chan_recommended_order,
    pc_alfano,
    pc_chan,
    pc_circular_closed_form,
    pc_foster,
    pc_small_body_approximation,
)
from .dataset import ConjunctionPopulation, generate_population
from .dilution import (
    DILUTION_MAHALANOBIS_THRESHOLD,
    DilutionAssessment,
    DilutionWarning,
    assess_dilution,
    dilution_curve,
    max_probability,
)
from .features import FEATURE_DESCRIPTIONS, FEATURE_NAMES, features_from_geometry
from .geometry import (
    EncounterGeometry,
    ObjectState,
    PrincipalFrame,
    encounter_geometry,
    encounter_plane_basis,
    mahalanobis_distance,
    miss_vector,
    principal_frame,
    project_covariance,
    relative_state,
    time_of_closest_approach,
)
from .montecarlo import (
    MonteCarloResult,
    clopper_pearson,
    pc_monte_carlo,
    pc_monte_carlo_curved,
    pc_monte_carlo_window,
)
from .nonlinear import (
    curvature_parameter,
    curvature_sweep,
    first_crossing,
    gravity_gradient_acceleration_scale,
    window_sweep,
)
from .quadrature import pc_quadrature
from .regimes import RegimeSweep, evaluate_methods, geometry_from_dimensionless, regime_sweep
from .report import (
    ANALYTIC_METHODS,
    ConjunctionReport,
    Disagreement,
    RegimeFlags,
    assess_conjunction,
)
from .screening import (
    OperatingPoint,
    ScreeningClassifier,
    brier_score,
    compare_screens,
    recall_at_fpr,
    reliability_table,
)
from .screens import (
    SCREEN_NAMES,
    all_screen_scores,
    mahalanobis_score,
    miss_distance_score,
    small_body_score,
)

__version__ = "0.1.0"

__all__ = [
    "ANALYTIC_METHODS",
    "DILUTION_MAHALANOBIS_THRESHOLD",
    "FEATURE_DESCRIPTIONS",
    "FEATURE_NAMES",
    "SCREEN_NAMES",
    "ConjunctionPopulation",
    "ConjunctionReport",
    "Disagreement",
    "DilutionAssessment",
    "DilutionWarning",
    "EncounterGeometry",
    "MonteCarloResult",
    "ObjectState",
    "OperatingPoint",
    "PrincipalFrame",
    "RegimeFlags",
    "RegimeSweep",
    "ScreeningClassifier",
    "__version__",
    "all_screen_scores",
    "assess_conjunction",
    "assess_dilution",
    "brier_score",
    "chan_recommended_order",
    "clopper_pearson",
    "compare_screens",
    "curvature_parameter",
    "curvature_sweep",
    "dilution_curve",
    "encounter_geometry",
    "encounter_plane_basis",
    "evaluate_methods",
    "features_from_geometry",
    "first_crossing",
    "geometry_from_dimensionless",
    "generate_population",
    "gravity_gradient_acceleration_scale",
    "mahalanobis_distance",
    "mahalanobis_score",
    "max_probability",
    "miss_distance_score",
    "miss_vector",
    "pc_alfano",
    "pc_chan",
    "pc_circular_closed_form",
    "pc_foster",
    "pc_monte_carlo",
    "pc_monte_carlo_curved",
    "pc_monte_carlo_window",
    "pc_quadrature",
    "pc_small_body_approximation",
    "principal_frame",
    "project_covariance",
    "recall_at_fpr",
    "regime_sweep",
    "relative_state",
    "reliability_table",
    "small_body_score",
    "time_of_closest_approach",
    "window_sweep",
]
