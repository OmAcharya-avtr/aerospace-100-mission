"""faultinject: fault-injection campaigns for GNC and communications software.

Research-grade software. Not flight-qualified, not certified, not approved for
operational aerospace use.

Layout
------
``taxonomy``     sixteen fault kinds with units, ranges and coverage binning
``faults``       the executable handler for each kind
``wrapper``      injection around an unmodified target, plus the numerical monitor
``target``       the reference GNC loop used as the system under test
``harness``      closed-loop execution and the trace type
``severity``     severity scoring from the target's response
``coverage``     coverage accounting over the taxonomy cross product
``campaign``     cases, case pools, execution and seeded replay
``search``       uniform-random and coverage-greedy baselines, plus the learned search
``prioritizer``  the learned severity predictor with an uncertainty output
``benchmark``    same-budget comparison with bootstrap confidence intervals
"""

from __future__ import annotations

from .campaign import (
    CampaignResult,
    CaseResult,
    FaultCase,
    build_pool,
    case_in_cell,
    evaluate_pool,
    execute_case,
    replay_case,
    run_campaign,
)
from .coverage import CoverageCell, CoverageTracker, all_cells, cell_of
from .faults import Injection, Stage, make_handler, stage_of
from .harness import Trace, nominal_trace, run_case
from .prioritizer import CampaignPrioritizer, SeverityPrediction, encode, feature_names
from .severity import SEVERE_THRESHOLD, SeverityReport, label_for, score
from .target import GncController, SanitisingGncController, kalman_gain
from .taxonomy import (
    TAXONOMY,
    FaultClass,
    FaultKind,
    FaultSpec,
    ParamSpec,
    kinds,
    kinds_of_class,
    spec,
    total_cells,
)
from .wrapper import InjectionWrapper, NumericalMonitor, classify_value

__version__ = "0.1.0"

__all__ = [
    "SEVERE_THRESHOLD",
    "TAXONOMY",
    "CampaignPrioritizer",
    "CampaignResult",
    "CaseResult",
    "CoverageCell",
    "CoverageTracker",
    "FaultCase",
    "FaultClass",
    "FaultKind",
    "FaultSpec",
    "GncController",
    "Injection",
    "InjectionWrapper",
    "NumericalMonitor",
    "ParamSpec",
    "SanitisingGncController",
    "SeverityPrediction",
    "SeverityReport",
    "Stage",
    "Trace",
    "__version__",
    "all_cells",
    "build_pool",
    "case_in_cell",
    "cell_of",
    "classify_value",
    "encode",
    "evaluate_pool",
    "execute_case",
    "feature_names",
    "kalman_gain",
    "kinds",
    "kinds_of_class",
    "label_for",
    "make_handler",
    "nominal_trace",
    "replay_case",
    "run_campaign",
    "run_case",
    "score",
    "spec",
    "stage_of",
    "total_cells",
]
