"""The findings: what is wrong with a declared fault-response policy.

Six checks, all deterministic and all exact over the declared fragment:

1. Fault states from which no declared safe mode is reachable, under the
   detection assumption in force.
2. Declared safe modes that no fault state can reach by any path.
3. Fault states with no enabled response at all, excluding those already in a
   declared safe mode, which need none.
4. Faults with no declared response addressing them.
5. Responses whose precondition can never hold in any mode they could fire
   from, so they are present in the declaration and dead in the automaton.
6. Ambiguity classes: sets of faults the declared observability model cannot
   tell apart, that demand different responses. This is the check the product
   exists for, and for each conflicting class it also computes the
   class-conservative response, or reports that none exists.

The ambiguity relation is the static special case of the diagnosability
relation of Sampath, Sengupta, Lafortune, Sinnamohideen and Teneketzis,
"Diagnosability of Discrete-Event Systems", IEEE Transactions on Automatic
Control 40(9), pp. 1555-1575, September 1995: two faults are indistinguishable
when their declared observation signatures are equal. Costs are minutes of
mission outage (min); hop counts are dimensionless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .automaton import Automaton, AutomatonState, DetectionAssumption, build_automaton
from .reach import (
    ReachabilityReport,
    min_cost_to_safe,
    preferred_specific_response,
    reachability_report,
)
from .spec import Response, SafeModeSpec

__all__ = [
    "AmbiguityClass",
    "DiagnosticsReport",
    "VacuousResponse",
    "analyse",
    "ambiguity_classes",
    "conservative_response",
    "vacuous_responses",
]


@dataclass(frozen=True)
class VacuousResponse:
    """A declared response whose precondition can never hold.

    Attributes:
        response_id: the declared response.
        reasons: one line per candidate source mode saying why the conjunction
            is empty there, sorted by mode id.
    """

    response_id: str
    reasons: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.response_id}: " + "; ".join(self.reasons)


def _candidate_source_modes(spec: SafeModeSpec, response: Response) -> tuple[str, ...]:
    """Modes a response could fire from, ignoring its precondition."""
    if response.from_mode is not None:
        return (response.from_mode,)
    return tuple(m.id for m in spec.modes if not m.absorbing)


def vacuous_responses(spec: SafeModeSpec) -> tuple[VacuousResponse, ...]:
    """Responses whose precondition can never hold in any candidate source mode.

    A response with a trivial precondition is never vacuous. A response whose
    only candidate source mode is absorbing is reported, because no transition
    can leave an absorbing mode.
    """
    out: list[VacuousResponse] = []
    for response in spec.responses:
        modes = _candidate_source_modes(spec, response)
        if not modes:
            out.append(
                VacuousResponse(
                    response.id, ("every declared mode is absorbing, so nothing can fire",)
                )
            )
            continue
        if response.from_mode is not None and spec.mode(response.from_mode).absorbing:
            out.append(
                VacuousResponse(
                    response.id,
                    (f"from_mode {response.from_mode} is declared absorbing",),
                )
            )
            continue
        reasons: list[str] = []
        for mode_id in modes:
            conflict = spec.precondition_conflict(response.precondition, mode_id)
            if conflict is None:
                reasons = []
                break
            reasons.append(f"in {mode_id}: {conflict}")
        if reasons:
            out.append(VacuousResponse(response.id, tuple(sorted(reasons))))
    return tuple(out)


@dataclass(frozen=True)
class AmbiguityClass:
    """A set of faults the declared observability model cannot tell apart.

    Attributes:
        faults: the member fault ids, sorted.
        signature: the shared declared observation signature.
        mode: the mode the class was analysed in.
        preferred: fault id -> the cheapest response that NAMES that fault and
            is enabled in ``mode``, or None when the fault has none there.
        conflicting: the members do not all prefer the SAME response, so
            acting on an identification the observations cannot support can
            command the wrong response. This is the finding.
        conservative_response: the first action of the cheapest fault-agnostic
            route, enabled in ``mode``, from whose target every member of the
            class still reaches a declared safe mode. None when the declaration
            offers no such route.
        conservative_cost_min: cost of ``conservative_response`` plus the
            cheapest onward path to a safe mode, worst case over the class
            members [min]. ``math.inf`` when there is no conservative route.
        best_specific_cost_min: fault id -> cost of that fault's own cheapest
            safe-reaching response path [min].
    """

    faults: tuple[str, ...]
    signature: tuple[int, ...]
    mode: str
    preferred: dict[str, str | None]
    conflicting: bool
    conservative_response: str | None
    conservative_cost_min: float
    best_specific_cost_min: dict[str, float]

    @property
    def conservatism_cost_min(self) -> float:
        """Conservative worst case minus the mean fault-specific cost [min].

        ``math.inf`` when no conservative response exists. Negative values are
        possible and are reported as measured: a declaration can make the
        conservative route cheaper than the fault-specific one.
        """
        finite = [c for c in self.best_specific_cost_min.values() if math.isfinite(c)]
        if not finite or math.isinf(self.conservative_cost_min):
            return math.inf
        return self.conservative_cost_min - sum(finite) / len(finite)

    def __str__(self) -> str:
        members = ", ".join(self.faults)
        verdict = "CONFLICTING" if self.conflicting else "consistent"
        conservative = self.conservative_response or "NONE"
        return (
            f"[{members}] in {self.mode}: {verdict}; conservative={conservative} "
            f"({self.conservative_cost_min:g} min)"
        )


def ambiguity_classes(
    spec: SafeModeSpec,
    assumption: DetectionAssumption | None = None,
    automaton: Automaton | None = None,
) -> tuple[AmbiguityClass, ...]:
    """Ambiguity classes of ``spec``, one entry per class and analysed mode.

    A class is analysed in every non-absorbing mode in which every member of
    the class can occur, because a class whose members cannot co-occur in a
    mode is not ambiguous there.
    """
    if automaton is None:
        automaton = build_automaton(spec, assumption)
    costs = min_cost_to_safe(automaton)
    out: list[AmbiguityClass] = []
    for group in spec.observability.indistinguishable_groups():
        for mode in spec.modes:
            if mode.absorbing:
                continue
            if not all(spec.fault_occurs_in(fault_id, mode.id) for fault_id in group):
                continue
            preferred: dict[str, str | None] = {}
            specific_cost: dict[str, float] = {}
            for fault_id in group:
                response_id, cost = preferred_specific_response(
                    automaton, costs, mode.id, fault_id
                )
                preferred[fault_id] = response_id
                specific_cost[fault_id] = cost
            conflicting = len({preferred[fault_id] for fault_id in group}) > 1
            response_id, conservative_cost = conservative_response(
                spec, automaton, costs, group, mode.id
            )
            out.append(
                AmbiguityClass(
                    faults=group,
                    signature=spec.observability.signature(group[0]),
                    mode=mode.id,
                    preferred=preferred,
                    conflicting=conflicting,
                    conservative_response=response_id,
                    conservative_cost_min=conservative_cost,
                    best_specific_cost_min=specific_cost,
                )
            )
    return tuple(out)


def conservative_response(
    spec: SafeModeSpec,
    automaton: Automaton,
    costs: dict[AutomatonState, tuple[float, tuple[str, ...], int]],
    group: tuple[str, ...],
    mode_id: str,
) -> tuple[str | None, float]:
    """The class-conservative response in ``mode_id``, and its worst-case cost.

    A candidate must be fault-agnostic, because by definition the fault is not
    identified, and enabled in ``mode_id``. It is valid when every member of
    the class still reaches a declared safe mode from the resulting state. The
    cost is the response cost plus the worst, over the class members, of the
    cheapest onward path to a safe mode. Candidates are ordered by (target mode
    not safe, cost, response id), so a response that lands directly in a safe
    mode wins over a cheaper one that does not.
    """
    best: tuple[tuple[int, float, str], str, float] | None = None
    for response in spec.responses:
        if not response.is_fault_agnostic:
            continue
        if response.from_mode is not None and response.from_mode != mode_id:
            continue
        if spec.mode(mode_id).absorbing:
            continue
        if not spec.precondition_can_hold(response.precondition, mode_id):
            continue
        worst = 0.0
        valid = True
        for fault_id in group:
            after = AutomatonState(response.to_mode, None if response.clears_fault else fault_id)
            if automaton.is_safe(after):
                onward = 0.0
            else:
                entry = costs.get(after)
                if entry is None:
                    valid = False
                    break
                onward = entry[0]
            worst = max(worst, onward)
        if not valid:
            continue
        total = response.cost_min + worst
        key = (total, response.id)
        if best is None or key < best[0]:
            best = (key, response.id, total)
    if best is None:
        return None, math.inf
    return best[1], best[2]


@dataclass(frozen=True)
class DiagnosticsReport:
    """Every finding for one declaration under one detection assumption.

    Attributes:
        spec_name: the declaration's name.
        assumption: the detection assumption in force.
        reachability: the safe-mode reachability report.
        fault_states_without_response: fault states with no enabled transition
            whose mode is not already a declared safe mode. A fault state that
            is already safe needs no response and is not a finding.
        faults_without_response: declared faults no declared response names,
            and which no fault-agnostic response can help either.
        vacuous: responses whose precondition can never hold.
        ambiguity: the ambiguity classes, conflicting or not.
        undetectable_faults: faults the declaration marks undetectable.
        faults_over_latency_budget: faults whose declared detection latency
            exceeds the assumption's budget.
    """

    spec_name: str
    assumption: DetectionAssumption
    reachability: ReachabilityReport
    fault_states_without_response: tuple[AutomatonState, ...]
    faults_without_response: tuple[str, ...]
    vacuous: tuple[VacuousResponse, ...]
    ambiguity: tuple[AmbiguityClass, ...]
    undetectable_faults: tuple[str, ...]
    faults_over_latency_budget: tuple[str, ...]

    @property
    def conflicting_ambiguity(self) -> tuple[AmbiguityClass, ...]:
        """Ambiguity classes whose members demand different responses."""
        return tuple(c for c in self.ambiguity if c.conflicting)

    @property
    def classes_without_conservative_response(self) -> tuple[AmbiguityClass, ...]:
        """Ambiguity classes with no class-conservative response."""
        return tuple(c for c in self.ambiguity if c.conservative_response is None)

    @property
    def finding_counts(self) -> dict[str, int]:
        """Count per finding category, for a one-line verdict."""
        return {
            "fault_states_without_safe_mode": len(self.reachability.unreachable_rows),
            "unreachable_safe_modes": len(self.reachability.unreachable_safe_modes),
            "fault_states_without_response": len(self.fault_states_without_response),
            "faults_without_response": len(self.faults_without_response),
            "vacuous_responses": len(self.vacuous),
            "conflicting_ambiguity_classes": len(self.conflicting_ambiguity),
            "classes_without_conservative_response": len(
                self.classes_without_conservative_response
            ),
        }

    @property
    def total_findings(self) -> int:
        """Sum over the finding categories. Zero means every check passed."""
        return sum(self.finding_counts.values())

    @property
    def ok(self) -> bool:
        """True when no check produced a finding."""
        return self.total_findings == 0

    def report(self) -> str:
        """Multi-line human-readable report of every finding."""
        lines = [
            f"declaration: {self.spec_name}",
            f"assumption:  {self.assumption.describe()}",
            "",
            "findings",
            "--------",
        ]
        for key, value in self.finding_counts.items():
            lines.append(f"  {key:<42} {value}")
        lines.append("")
        if self.reachability.unreachable_rows:
            lines.append("fault states with no reachable safe mode:")
            for row in self.reachability.unreachable_rows:
                lines.append(f"  {row.state}")
        if self.reachability.unreachable_safe_modes:
            lines.append(
                "declared safe modes no fault state can reach: "
                + ", ".join(self.reachability.unreachable_safe_modes)
            )
        if self.fault_states_without_response:
            lines.append("fault states with no enabled response:")
            for state in self.fault_states_without_response:
                lines.append(f"  {state}")
        if self.faults_without_response:
            lines.append(
                "faults with no declared response: " + ", ".join(self.faults_without_response)
            )
        if self.vacuous:
            lines.append("responses whose precondition can never hold:")
            for item in self.vacuous:
                lines.append(f"  {item}")
        if self.undetectable_faults:
            lines.append("faults declared undetectable: " + ", ".join(self.undetectable_faults))
        if self.faults_over_latency_budget:
            lines.append(
                "faults whose declared detection latency exceeds the budget: "
                + ", ".join(self.faults_over_latency_budget)
            )
        if self.ambiguity:
            lines.append("ambiguity classes:")
            for item in self.ambiguity:
                lines.append(f"  {item}")
        lines.append("")
        lines.append(f"total findings: {self.total_findings}")
        return "\n".join(lines)


def analyse(
    spec: SafeModeSpec, assumption: DetectionAssumption | None = None
) -> DiagnosticsReport:
    """Run every check on ``spec`` under ``assumption``.

    Args:
        spec: the declaration.
        assumption: detection assumption; default is declared detectability
            with an unbounded latency budget.

    Returns:
        A :class:`DiagnosticsReport`. ``report.ok`` is True only when every
        check passed.
    """
    automaton = build_automaton(spec, assumption)
    reach = reachability_report(spec, automaton=automaton)
    no_response = tuple(
        state
        for state in automaton.fault_states
        if not automaton.outgoing(state) and not automaton.is_safe(state)
    )
    named = {r.fault for r in spec.responses if r.fault is not None}
    agnostic_exists = any(r.is_fault_agnostic for r in spec.responses)
    faults_without_response = tuple(
        fault_id
        for fault_id in spec.fault_ids
        if fault_id not in named and not agnostic_exists
    )
    budget = automaton.assumption
    undetectable = tuple(f.id for f in spec.faults if not f.detectable)
    over_budget = tuple(
        f.id
        for f in spec.faults
        if f.detectable
        and budget.require_detection
        and f.detection_latency_s > budget.latency_budget_s
    )
    return DiagnosticsReport(
        spec_name=spec.name,
        assumption=automaton.assumption,
        reachability=reach,
        fault_states_without_response=no_response,
        faults_without_response=faults_without_response,
        vacuous=vacuous_responses(spec),
        ambiguity=ambiguity_classes(spec, automaton=automaton),
        undetectable_faults=undetectable,
        faults_over_latency_budget=over_budget,
    )
