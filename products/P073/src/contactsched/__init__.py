"""contactsched: ground-station contact scheduling with the fairness cost measured.

Several missions want the same antenna in the same ten minutes. Something has to
give, and the usual tools will tell you what the schedule is without telling you
what it cost: which request was dropped, by how many seconds, and whether the
priority scheme you declared actually delivered the ordering it promised. This
package computes the schedule two ways -- a documented priority-ordered first-fit
and an exact mixed-integer optimum on HiGHS -- and then measures the contention
and the fairness of whichever one you take.

Research grade. Not flight-qualified, not certified, not approved for operational
aerospace use. Real ground-station networks schedule with vendor-specific systems
against object catalogues and service level agreements that this package does not
model.

Quick start
-----------
>>> from contactsched import Scenario, default_network, default_fleet
>>> from contactsched import all_windows, build_problem, ContactRequest
>>> from contactsched import schedule_greedy, solve_exact, analyse
>>> scenario = Scenario.from_calendar(2026, 4, 30, duration_s=21600.0)
>>> stations, fleet = default_network()[:2], default_fleet()[:2]
>>> windows = tuple(all_windows(scenario, stations, fleet, step_s=45.0))
>>> requests = (
...     ContactRequest("R01", "POLARIS", "POLARIS-A", 1, 300.0, 2, 3),
...     ContactRequest("R02", "POLARIS", "POLARIS-B", 2, 300.0, 2, 3),
... )
>>> problem = build_problem(stations, requests, windows, slot_step_s=120.0)
>>> exact = solve_exact(problem)
>>> report = analyse(problem, exact)
>>> report.max_min >= 0.0
True
"""

from __future__ import annotations

from .assignment import (
    STATUS_HEURISTIC,
    STATUS_INFEASIBLE,
    STATUS_LIMIT,
    STATUS_OPTIMAL,
    Assignment,
    objective_value,
)
from .benchmark import (
    MethodResult,
    compare_methods,
    fairness_cost,
    gap_distribution,
    optimality_gap,
    run_method,
    score_predictor,
)
from .bruteforce import solve_bruteforce
from .contention import ContentionCell, contention_map, effective_seconds, request_contention
from .dataset import Instance, build_dataset, generate_instance, label_instance, split_by_instance
from .ephemeris import Scenario, Spacecraft, default_fleet
from .exact import FairnessPolicy, available_backends, solve_exact, standalone_capacity
from .fairness import (
    FairnessReport,
    Inversion,
    RequestOutcome,
    analyse,
    jain_index,
    max_min_share,
)
from .features import FEATURE_NAMES, design_matrix, request_features
from .geometry import (
    WGS84_A_KM,
    WGS84_E2,
    WGS84_F,
    elevation_geocentric_norms_deg,
    elevation_geocentric_vector_deg,
    geodetic_to_ecef_km,
    gmst_rad,
    look_angles_deg,
    max_pass_duration_s,
    sez_matrix,
    teme_to_ecef_km,
)
from .greedy import schedule_greedy
from .predictors import (
    LearnedConflictPredictor,
    RatioLogisticBaseline,
    RatioRankingBaseline,
    average_precision,
    best_f1_threshold,
    brier_score,
    expected_calibration_error,
    reliability_table,
    roc_auc,
    threshold_metrics,
)
from .problem import (
    Candidate,
    ContactRequest,
    SchedulingProblem,
    build_candidates,
    build_problem,
    check_feasible,
    priority_weight,
    shortfalls,
)
from .report import (
    format_backends,
    format_contention,
    format_fairness,
    format_schedule,
    format_windows,
)
from .station import GroundStation, HorizonMask, default_network
from .tle import OrbitElements, parse_lines, tle_checksum
from .visibility import (
    ContactWindow,
    all_windows,
    compute_windows,
    mask_margin_deg,
    supply_seconds_by_station,
)

__version__ = "0.1.0"

__all__ = [
    "FEATURE_NAMES",
    "STATUS_HEURISTIC",
    "STATUS_INFEASIBLE",
    "STATUS_LIMIT",
    "STATUS_OPTIMAL",
    "WGS84_A_KM",
    "WGS84_E2",
    "WGS84_F",
    "Assignment",
    "Candidate",
    "ContactRequest",
    "ContactWindow",
    "ContentionCell",
    "FairnessPolicy",
    "FairnessReport",
    "GroundStation",
    "HorizonMask",
    "Instance",
    "Inversion",
    "LearnedConflictPredictor",
    "MethodResult",
    "OrbitElements",
    "RatioLogisticBaseline",
    "RatioRankingBaseline",
    "RequestOutcome",
    "Scenario",
    "SchedulingProblem",
    "Spacecraft",
    "__version__",
    "all_windows",
    "analyse",
    "available_backends",
    "average_precision",
    "best_f1_threshold",
    "brier_score",
    "build_candidates",
    "build_dataset",
    "build_problem",
    "check_feasible",
    "compare_methods",
    "compute_windows",
    "contention_map",
    "default_fleet",
    "default_network",
    "design_matrix",
    "effective_seconds",
    "elevation_geocentric_norms_deg",
    "elevation_geocentric_vector_deg",
    "expected_calibration_error",
    "fairness_cost",
    "format_backends",
    "format_contention",
    "format_fairness",
    "format_schedule",
    "format_windows",
    "gap_distribution",
    "generate_instance",
    "geodetic_to_ecef_km",
    "gmst_rad",
    "jain_index",
    "label_instance",
    "look_angles_deg",
    "mask_margin_deg",
    "max_min_share",
    "max_pass_duration_s",
    "objective_value",
    "optimality_gap",
    "parse_lines",
    "priority_weight",
    "reliability_table",
    "request_contention",
    "request_features",
    "roc_auc",
    "run_method",
    "schedule_greedy",
    "score_predictor",
    "sez_matrix",
    "shortfalls",
    "solve_bruteforce",
    "solve_exact",
    "split_by_instance",
    "standalone_capacity",
    "supply_seconds_by_station",
    "teme_to_ecef_km",
    "threshold_metrics",
    "tle_checksum",
]
