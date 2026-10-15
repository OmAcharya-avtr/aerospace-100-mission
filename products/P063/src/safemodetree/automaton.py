"""The response automaton built from a declaration.

A state is a pair ``(mode, fault)`` where ``fault`` is None for "no active
fault". A transition is one declared response, enabled when every declared
guard holds. This is an ordinary labelled transition system; nothing here is
probabilistic and nothing is learned.

Transition enabling rules, all of them from the declaration:

1. No transition leaves a mode declared ``absorbing``.
2. ``from_mode`` must be None (any mode) or equal to the state's mode.
3. A fault-specific response (``fault`` set) requires the state's active fault
   to be that fault AND the fault to be *actionable* under the detection
   assumption in force. A fault-agnostic response (``fault`` None) has no such
   requirement, which is what makes it the only thing available for an
   undetectable fault.
4. The response's precondition must be satisfiable in the state's mode, decided
   exactly by :meth:`safemodetree.spec.SafeModeSpec.precondition_conflict`.
5. A transition that changes neither the mode nor the fault is dropped, because
   a self-loop cannot contribute to reachability.

Cost is the declared ``cost_min`` in minutes of mission outage; path cost is
the sum along the path, which assumes the declared costs are additive and
independent of order. That assumption is the analyst's, is stated in the README
Limitations, and is not checked by this package.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

from .spec import Response, SafeModeSpec

__all__ = [
    "Automaton",
    "AutomatonState",
    "DetectionAssumption",
    "Transition",
    "build_automaton",
]


@dataclass(frozen=True)
class AutomatonState:
    """One automaton state: a mode and at most one active fault.

    Totally ordered by ``(mode, fault or "")`` so that states can appear in
    heap keys and sorted output deterministically.

    Attributes:
        mode: declared mode id.
        fault: declared fault id, or None for no active fault.
    """

    mode: str
    fault: str | None = None

    @property
    def sort_key(self) -> tuple[str, str]:
        """Total-order key: the mode, then the fault id or the empty string."""
        return (self.mode, self.fault or "")

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, AutomatonState):
            return NotImplemented
        return self.sort_key < other.sort_key

    def __le__(self, other: object) -> bool:
        if not isinstance(other, AutomatonState):
            return NotImplemented
        return self.sort_key <= other.sort_key

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, AutomatonState):
            return NotImplemented
        return self.sort_key > other.sort_key

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, AutomatonState):
            return NotImplemented
        return self.sort_key >= other.sort_key

    def __str__(self) -> str:
        return f"({self.mode}, {self.fault or 'no_fault'})"


@dataclass(frozen=True)
class DetectionAssumption:
    """The declared detection assumption reachability is checked under.

    A fault is *actionable* when the declaration says it is detectable and its
    declared detection latency fits the budget. Fault-specific responses
    require an actionable fault; fault-agnostic responses do not.

    Attributes:
        latency_budget_s: time available between fault occurrence and the first
            fault-specific response, in seconds. ``math.inf`` means no budget
            is imposed and only ``detectable`` matters.
        require_detection: when False, every fault is treated as actionable
            regardless of its declared detectability and latency. This is the
            optimistic assumption, and it exists so that the difference between
            the two can be measured rather than argued about.
    """

    latency_budget_s: float = math.inf
    require_detection: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.latency_budget_s, bool) or not isinstance(
            self.latency_budget_s, int | float
        ):
            raise TypeError("latency_budget_s must be a real number of seconds")
        if self.latency_budget_s < 0.0:
            raise ValueError(
                f"latency_budget_s must be >= 0 s, got {self.latency_budget_s} s"
            )
        object.__setattr__(self, "latency_budget_s", float(self.latency_budget_s))
        if not isinstance(self.require_detection, bool):
            raise TypeError("require_detection must be a bool")

    def describe(self) -> str:
        """One-line description of the assumption."""
        if not self.require_detection:
            return "detection assumed perfect (require_detection=False)"
        budget = (
            "unbounded" if math.isinf(self.latency_budget_s) else f"{self.latency_budget_s:g} s"
        )
        return f"declared detectability, latency budget {budget}"

    def is_actionable(self, spec: SafeModeSpec, fault_id: str) -> bool:
        """True if a fault-specific response for ``fault_id`` may fire."""
        if not self.require_detection:
            return True
        fault = spec.fault(fault_id)
        if not fault.detectable:
            return False
        return fault.detection_latency_s <= self.latency_budget_s


@dataclass(frozen=True)
class Transition:
    """One enabled transition of the automaton.

    Attributes:
        source: source state.
        target: target state.
        response: the declared response that labels it.
        cost_min: declared cost in minutes of mission outage (min).
    """

    source: AutomatonState
    target: AutomatonState
    response: Response
    cost_min: float

    def __str__(self) -> str:
        return (
            f"{self.source} -{self.response.id} ({self.cost_min:g} min)-> {self.target}"
        )


class Automaton:
    """The enabled transition system of a declaration under one assumption."""

    def __init__(
        self,
        spec: SafeModeSpec,
        assumption: DetectionAssumption,
        states: tuple[AutomatonState, ...],
        transitions: tuple[Transition, ...],
    ) -> None:
        self._spec = spec
        self._assumption = assumption
        self._states = states
        self._transitions = transitions
        self._out: dict[AutomatonState, tuple[Transition, ...]] = {}
        self._in: dict[AutomatonState, tuple[Transition, ...]] = {}
        out_lists: dict[AutomatonState, list[Transition]] = {s: [] for s in states}
        in_lists: dict[AutomatonState, list[Transition]] = {s: [] for s in states}
        for transition in transitions:
            out_lists[transition.source].append(transition)
            in_lists[transition.target].append(transition)
        self._out = {s: tuple(v) for s, v in out_lists.items()}
        self._in = {s: tuple(v) for s, v in in_lists.items()}

    @property
    def spec(self) -> SafeModeSpec:
        """The declaration this automaton was built from."""
        return self._spec

    @property
    def assumption(self) -> DetectionAssumption:
        """The detection assumption in force."""
        return self._assumption

    @property
    def states(self) -> tuple[AutomatonState, ...]:
        """All states, sorted by (mode, fault)."""
        return self._states

    @property
    def transitions(self) -> tuple[Transition, ...]:
        """All enabled transitions, in declaration order of the responses."""
        return self._transitions

    def outgoing(self, state: AutomatonState) -> tuple[Transition, ...]:
        """Transitions leaving ``state``. Raises KeyError for an unknown state."""
        return self._out[state]

    def incoming(self, state: AutomatonState) -> tuple[Transition, ...]:
        """Transitions entering ``state``. Raises KeyError for an unknown state."""
        return self._in[state]

    def is_safe(self, state: AutomatonState) -> bool:
        """True when the state's mode is declared safe."""
        return self._spec.mode(state.mode).safe

    @property
    def safe_states(self) -> tuple[AutomatonState, ...]:
        """States whose mode is declared safe."""
        return tuple(s for s in self._states if self.is_safe(s))

    @property
    def fault_states(self) -> tuple[AutomatonState, ...]:
        """States with an active fault the declaration allows in that mode.

        These are the states safe-mode reachability is required from. A fault
        state whose mode is already safe is included; it is trivially safe and
        is reported with distance 0.
        """
        return tuple(
            s
            for s in self._states
            if s.fault is not None and self._spec.fault_occurs_in(s.fault, s.mode)
        )

    def __iter__(self) -> Iterator[AutomatonState]:
        return iter(self._states)

    def __len__(self) -> int:
        return len(self._states)

    def summary(self) -> str:
        """One-line size description."""
        return (
            f"{len(self._states)} states, {len(self._transitions)} transitions, "
            f"{len(self.safe_states)} safe, {len(self.fault_states)} fault states; "
            f"{self._assumption.describe()}"
        )


def _enabled_target(
    spec: SafeModeSpec,
    assumption: DetectionAssumption,
    state: AutomatonState,
    response: Response,
) -> AutomatonState | None:
    """Target state if ``response`` is enabled in ``state``, else None."""
    if spec.mode(state.mode).absorbing:
        return None
    if response.from_mode is not None and response.from_mode != state.mode:
        return None
    if response.fault is not None:
        if state.fault != response.fault:
            return None
        if not assumption.is_actionable(spec, response.fault):
            return None
    if not spec.precondition_can_hold(response.precondition, state.mode):
        return None
    target = AutomatonState(response.to_mode, None if response.clears_fault else state.fault)
    if target == state:
        return None
    return target


def build_automaton(
    spec: SafeModeSpec, assumption: DetectionAssumption | None = None
) -> Automaton:
    """Build the enabled transition system of ``spec`` under ``assumption``.

    Args:
        spec: the declaration.
        assumption: detection assumption; defaults to declared detectability
            with an unbounded latency budget.

    Returns:
        An :class:`Automaton` over ``len(modes) * (len(faults) + 1)`` states.
    """
    if assumption is None:
        assumption = DetectionAssumption()
    faults: list[str | None] = [None, *spec.fault_ids]
    states = tuple(
        sorted(
            AutomatonState(mode_id, fault_id)
            for mode_id in spec.mode_ids
            for fault_id in faults
        )
    )
    transitions: list[Transition] = []
    for state in states:
        for response in spec.responses:
            target = _enabled_target(spec, assumption, state, response)
            if target is not None:
                transitions.append(
                    Transition(
                        source=state,
                        target=target,
                        response=response,
                        cost_min=response.cost_min,
                    )
                )
    return Automaton(spec, assumption, states, tuple(transitions))
