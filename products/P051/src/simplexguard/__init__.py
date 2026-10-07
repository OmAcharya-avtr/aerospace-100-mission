"""simplexguard: a runtime-assurance (Simplex) architecture benchmark.

A learned or otherwise unverified performance controller wrapped by a
conservative controller with a computed robust invariant set, with the exact
switching condition, the assurance accounting that the switching costs, a
deliberate bound-violation experiment, and a learned switch predictor
benchmarked against the exact analytic condition.

This is **not** a verification tool. It computes a robust invariant set for the
discrete-time linear plant it is given, under the disturbance bound it is told
to assume, and it proves nothing whatsoever about any plant it was not given.
Research-grade: not flight-qualified, not certified, not approved for
operational aerospace use.

The usual order of use is

    plant      = reference_plant()
    baseline, performance = reference_controllers(plant)
    result     = robust_invariant_set(plant, baseline)
    guard      = SimplexGuard(plant, baseline, result.polytope)
    report     = account(guarded, unguarded, baseline_only, result.polytope)

and every step of it is exercised by ``validation/worked_example.py``.
"""

from __future__ import annotations

from .accounting import (
    AssuranceReport,
    DwellStatistics,
    account,
    dwell_statistics,
    mode_runs,
)
from .boundviolation import BoundSweep, BoundViolationPoint, bound_violation_sweep
from .controllers import (
    BaselineController,
    PerformanceController,
    closed_loop_matrix,
    dlqr_gain,
    reference_controllers,
    saturate,
)
from .guard import GuardDecision, Mode, SimplexGuard
from .invariant import (
    EmptyInvariantSet,
    InvariantSetResult,
    RecursionDidNotConverge,
    robust_invariant_set,
    verify_robust_invariance,
)
from .plant import Plant, reference_plant
from .polytope import Box, Polytope, box
from .predictor import (
    ClassificationScores,
    SwitchDataset,
    build_dataset,
    exact_predictor_scores,
    expected_calibration_error,
    fit_switch_predictor,
    guard_condition_scores,
    lead_times,
    measure_decision_cost,
    score_binary,
)
from .reachability import ExactLeadPredictor
from .simulate import (
    CostWeights,
    Episode,
    disturbance_sequence,
    simulate_baseline,
    simulate_guarded,
    simulate_unguarded,
    square_wave_reference,
)

__version__ = "0.1.0"

__all__ = [
    "AssuranceReport",
    "BaselineController",
    "BoundSweep",
    "BoundViolationPoint",
    "Box",
    "ClassificationScores",
    "CostWeights",
    "DwellStatistics",
    "EmptyInvariantSet",
    "Episode",
    "ExactLeadPredictor",
    "GuardDecision",
    "InvariantSetResult",
    "Mode",
    "PerformanceController",
    "Plant",
    "Polytope",
    "RecursionDidNotConverge",
    "SimplexGuard",
    "SwitchDataset",
    "__version__",
    "account",
    "bound_violation_sweep",
    "box",
    "build_dataset",
    "closed_loop_matrix",
    "disturbance_sequence",
    "dlqr_gain",
    "dwell_statistics",
    "exact_predictor_scores",
    "expected_calibration_error",
    "fit_switch_predictor",
    "guard_condition_scores",
    "lead_times",
    "measure_decision_cost",
    "mode_runs",
    "reference_controllers",
    "reference_plant",
    "robust_invariant_set",
    "saturate",
    "score_binary",
    "simulate_baseline",
    "simulate_guarded",
    "simulate_unguarded",
    "square_wave_reference",
    "verify_robust_invariance",
]
