"""cmdguard: check a command sequence against the spacecraft state it will meet.

Research-grade software. Not flight-qualified, not certified, not approved for
operational aerospace use. A sequence that this package reports clean has been
shown consistent with the declarations it was given, and that is not a
statement that the sequence is safe to uplink.

Public surface
--------------
Declarations
    :class:`~cmdguard.model.CommandSpec`,
    :class:`~cmdguard.model.CommandDictionary`,
    :class:`~cmdguard.model.ModeMachine`,
    :class:`~cmdguard.model.InitialState`,
    :class:`~cmdguard.model.CommandSequence`,
    :class:`~cmdguard.model.Case`,
    :class:`~cmdguard.model.Verity`,
    :class:`~cmdguard.model.EffectPolicy`,
    :func:`~cmdguard.loader.load_case`.
Baseline 1, ground truth
    :func:`~cmdguard.trace.forward_execute` returning a
    :class:`~cmdguard.trace.CheckReport` of
    :class:`~cmdguard.trace.Violation` objects, each carrying the
    :class:`~cmdguard.trace.Provenance` chain that reaches it.
Baseline 2
    :func:`~cmdguard.lint.static_lint`.
The gate
    :func:`~cmdguard.gate.gate_case`, :func:`~cmdguard.gate.irreversible_gate`.
Learned component
    :class:`~cmdguard.ranker.RiskRanker` and the orderings it is scored
    against, :func:`~cmdguard.ranker.irreversibility_ordering` and
    :func:`~cmdguard.ranker.resource_draw_ordering`.
Measurement
    :func:`~cmdguard.benchmark.lint_miss_rate`,
    :func:`~cmdguard.benchmark.ranker_comparison`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .benchmark import lint_miss_rate, ranker_comparison
from .dataset import build_corpus, generate_clean_case, generate_defective_case
from .defects import FAMILIES, inject
from .features import FEATURE_NAMES, feature_matrix
from .gate import GateResult, Verdict, gate_case, irreversible_gate
from .lint import LintFinding, LintReport, LintStrength, static_lint
from .loader import load_case
from .model import (
    Case,
    CommandDictionary,
    CommandSequence,
    CommandSpec,
    EffectPolicy,
    InitialState,
    ModeMachine,
    ResourceLimits,
    SequenceStep,
    Verity,
)
from .ranker import (
    RiskRanker,
    RiskScores,
    irreversibility_ordering,
    precision_at_k,
    recall_at_k,
    resource_draw_ordering,
)
from .report import render_check, render_comparison, render_state_at
from .trace import CheckReport, Provenance, Violation, ViolationKind, forward_execute

__all__ = [
    "__version__",
    "Case",
    "CheckReport",
    "CommandDictionary",
    "CommandSequence",
    "CommandSpec",
    "EffectPolicy",
    "FAMILIES",
    "FEATURE_NAMES",
    "GateResult",
    "InitialState",
    "LintFinding",
    "LintReport",
    "LintStrength",
    "ModeMachine",
    "Provenance",
    "ResourceLimits",
    "RiskRanker",
    "RiskScores",
    "SequenceStep",
    "Verdict",
    "Verity",
    "Violation",
    "ViolationKind",
    "build_corpus",
    "feature_matrix",
    "forward_execute",
    "gate_case",
    "generate_clean_case",
    "generate_defective_case",
    "inject",
    "irreversibility_ordering",
    "irreversible_gate",
    "lint_miss_rate",
    "load_case",
    "precision_at_k",
    "ranker_comparison",
    "recall_at_k",
    "render_check",
    "render_comparison",
    "render_state_at",
    "resource_draw_ordering",
    "static_lint",
]
