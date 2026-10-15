"""safemodetree: from an FMEA table to a checked fault-response policy.

Reads an FMEA-shaped declaration (faults with effect, detectability and
declared detection latency; modes including declared safe modes; responses with
a transition, a cost and a precondition; an observability model saying which
faults are distinguishable from which observations), builds the response
automaton, and checks that from every fault state a declared safe mode is
reachable under the declared detection assumptions.

What it reports: unreachable safe modes, fault states with no response,
responses whose preconditions can never hold, and ambiguity classes where two
faults are indistinguishable but demand different responses.

The learned component is a calibrated fault classifier over the observation
vector, benchmarked against the declared threshold-logic baseline at an equal
false-alarm rate, and the policy consumes its confidence: a low-confidence
identification is routed to the conservative response for the whole ambiguity
class. The cost of that conservatism is measured in minutes of mission outage.

This software is research-grade. It is not flight-qualified, not certified and
not approved for operational aerospace use.
"""

from __future__ import annotations

from .automaton import (
    Automaton,
    AutomatonState,
    DetectionAssumption,
    Transition,
    build_automaton,
)
from .baseline import ThresholdLogicClassifier, symptom_word
from .calibration import (
    CalibrationResult,
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from .data import NOMINAL_LABEL, ObservationSet, make_observations
from .diagnostics import (
    AmbiguityClass,
    DiagnosticsReport,
    VacuousResponse,
    ambiguity_classes,
    analyse,
    conservative_response,
    vacuous_responses,
)
from .evaluate import (
    DetectorScore,
    PolicyScore,
    Split,
    compare_detectors,
    compare_policies,
    conservatism_sweep,
    false_alarm_rate,
    identification_accuracy,
    learned_decisions,
    noise_sweep,
    split_observations,
)
from .learned import LearnedFaultClassifier, accuracy_ceiling
from .policy import Decision, FaultResponsePolicy
from .reach import (
    ReachabilityReport,
    ReachabilityRow,
    exhaustive_min_cost_to_safe,
    exhaustive_safe_reachable,
    min_cost_to_safe,
    outcome_state,
    preferred_specific_response,
    reachability_report,
    resolution_cost,
    safe_reachable_set,
)
from .spec import (
    SEVERITY_CATEGORIES,
    Condition,
    Fault,
    Literal,
    Mode,
    ObservabilityModel,
    ObservationChannel,
    Precondition,
    Response,
    SafeModeSpec,
    load_spec,
    load_spec_text,
    packaged_spec_names,
    packaged_spec_path,
)
from .synth import random_declaration

__version__ = "0.1.0"

__all__ = [
    "NOMINAL_LABEL",
    "SEVERITY_CATEGORIES",
    "AmbiguityClass",
    "Automaton",
    "AutomatonState",
    "CalibrationResult",
    "Condition",
    "Decision",
    "DetectionAssumption",
    "DetectorScore",
    "DiagnosticsReport",
    "Fault",
    "FaultResponsePolicy",
    "LearnedFaultClassifier",
    "Literal",
    "Mode",
    "ObservabilityModel",
    "ObservationChannel",
    "ObservationSet",
    "PolicyScore",
    "Precondition",
    "ReachabilityReport",
    "ReachabilityRow",
    "Response",
    "SafeModeSpec",
    "Split",
    "ThresholdLogicClassifier",
    "Transition",
    "VacuousResponse",
    "__version__",
    "accuracy_ceiling",
    "ambiguity_classes",
    "analyse",
    "brier_score",
    "build_automaton",
    "compare_detectors",
    "compare_policies",
    "conservatism_sweep",
    "conservative_response",
    "exhaustive_min_cost_to_safe",
    "exhaustive_safe_reachable",
    "expected_calibration_error",
    "false_alarm_rate",
    "identification_accuracy",
    "learned_decisions",
    "load_spec",
    "load_spec_text",
    "make_observations",
    "min_cost_to_safe",
    "noise_sweep",
    "outcome_state",
    "packaged_spec_names",
    "packaged_spec_path",
    "preferred_specific_response",
    "random_declaration",
    "reachability_report",
    "reliability_curve",
    "resolution_cost",
    "safe_reachable_set",
    "split_observations",
    "symptom_word",
    "vacuous_responses",
]
