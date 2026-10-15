"""activityplan -- onboard activity planning with an auditable feasibility report.

Takes a declared set of activities (power profile, data produced, priority, time
window, precedence, exclusions) and a declared resource model (battery capacity
and depth-of-discharge limit, solar input profile, solid-state recorder
capacity, contact windows with rates), and produces a timeline plus the thing
that is actually defensible: **a feasibility report naming the binding
constraint and the slack on every other one.**

Three planners, in the order they were built:

1. :func:`activityplan.greedy.plan_greedy` -- priority-ordered
   earliest-feasible-start insertion with explicit resource simulation. Fast,
   documented, suboptimal.
2. :func:`activityplan.exact.solve_exact` -- the exact optimum of the
   discretised problem by MILP over HiGHS, the reference the other two are
   scored against.
3. :mod:`activityplan.predictor` -- a learned duration and energy predictor with
   a split-conformal prediction interval, used to warm-start the solver
   (:mod:`activityplan.warmstart`) and to produce a robust plan under
   predicted-duration uncertainty (:mod:`activityplan.robust`).

This software is research-grade. It is **not flight-qualified, not certified,
and not approved for operational aerospace use.** Nothing in it was identified
from any spacecraft; every constant is a declared input of a synthetic
benchmark.
"""

from __future__ import annotations

from .exact import ExactResult, MilpModel, build_model, enumerate_optimum, solve_exact
from .feasibility import RESOURCE_CONSTRAINTS, ConstraintSlack, FeasibilityReport, evaluate
from .features import (
    ACTIVITY_TYPES,
    FEATURE_NAMES,
    TYPE_CONSTANTS,
    ActivityRequest,
    RealisedActivity,
    analytic_duration_s,
    analytic_energy_wh,
    analytic_mean_power_w,
    feature_matrix,
    feature_vector,
    realise,
    sample_requests,
    to_activity,
)
from .greedy import GreedyResult, RejectionRecord, plan_greedy, topological_order
from .instances import (
    INSTANCE_ORDER,
    RECOMMENDED_STEP_S,
    InstanceSpec,
    instance_family,
    named_instance,
    sample_instance_spec,
)
from .model import (
    Activity,
    Contact,
    PowerSegment,
    Problem,
    ResourceModel,
    SolarProfile,
)
from .predictor import (
    ConformalRegressor,
    DurationEnergyPredictor,
    Interval,
    coverage,
    mean_interval_width,
)
from .report import (
    render_declaration,
    render_feasibility,
    render_plan,
    render_rejections,
    render_table,
)
from .robust import (
    PLANNING_MODES,
    Estimate,
    RealisationOutcome,
    compare_modes,
    estimate_for_mode,
    realise_spec,
    score_realisation,
)
from .simulate import Trace, simulate
from .timeline import Grid, Plan, RasterisedProblem
from .warmstart import (
    WarmStart,
    WarmStartComparison,
    compare_warm_start,
    warm_start_from_plan,
)

__version__ = "0.1.0"

__all__ = [
    "ACTIVITY_TYPES",
    "FEATURE_NAMES",
    "INSTANCE_ORDER",
    "PLANNING_MODES",
    "RECOMMENDED_STEP_S",
    "RESOURCE_CONSTRAINTS",
    "TYPE_CONSTANTS",
    "Activity",
    "ActivityRequest",
    "ConformalRegressor",
    "ConstraintSlack",
    "Contact",
    "DurationEnergyPredictor",
    "Estimate",
    "ExactResult",
    "FeasibilityReport",
    "GreedyResult",
    "Grid",
    "InstanceSpec",
    "Interval",
    "MilpModel",
    "Plan",
    "PowerSegment",
    "Problem",
    "RasterisedProblem",
    "RealisationOutcome",
    "RealisedActivity",
    "RejectionRecord",
    "ResourceModel",
    "SolarProfile",
    "Trace",
    "WarmStart",
    "WarmStartComparison",
    "__version__",
    "analytic_duration_s",
    "analytic_energy_wh",
    "analytic_mean_power_w",
    "build_model",
    "compare_modes",
    "compare_warm_start",
    "coverage",
    "enumerate_optimum",
    "estimate_for_mode",
    "evaluate",
    "feature_matrix",
    "feature_vector",
    "instance_family",
    "mean_interval_width",
    "named_instance",
    "plan_greedy",
    "realise",
    "realise_spec",
    "render_declaration",
    "render_feasibility",
    "render_plan",
    "render_rejections",
    "render_table",
    "sample_instance_spec",
    "sample_requests",
    "score_realisation",
    "simulate",
    "solve_exact",
    "to_activity",
    "topological_order",
    "warm_start_from_plan",
]
