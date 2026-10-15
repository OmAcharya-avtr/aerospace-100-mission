"""The fault-response policy, and the confidence it consumes.

The policy is the thing the declaration is for: given the mode, a fault
identification and a confidence in that identification, it names the response
to command. The confidence is not decoration. When it falls below the declared
threshold the policy abandons the fault-specific response and commands the
class-conservative route instead, which is valid for every member of the
ambiguity class the identification belongs to.

That substitution has a price, and this module computes it from the
declaration rather than asserting it. Cost accounting, all in minutes of
mission outage (min):

- ``resolution_cost`` of a commanded response is the declared cost of the
  response plus, when the true fault is still active afterwards and the
  resulting mode is not safe, the cheapest declared onward path to a safe mode.
- A MISSED DETECTION, where the policy believes there is no fault and one is
  active, commands nothing. No cost is assigned to it, because any number put
  there would be invented. Missed detections are counted separately and the
  count is reported beside every cost.
- A FALSE ALARM, where the policy commands a response and no fault is active,
  costs the declared response cost. The fault is already absent so there is no
  onward path.

Assumptions: declared costs are additive along a path and independent of order;
one fault is active at a time; the mode is known exactly. None of the three is
checked by this package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .automaton import Automaton, AutomatonState, DetectionAssumption, build_automaton
from .data import NOMINAL_LABEL
from .diagnostics import conservative_response
from .reach import min_cost_to_safe, preferred_specific_response, resolution_cost
from .spec import SafeModeSpec

__all__ = ["Decision", "FaultResponsePolicy"]


@dataclass(frozen=True)
class Decision:
    """One policy decision.

    Attributes:
        mode: the mode the decision was taken in.
        believed_fault: the identification handed to the policy, or
            ``"nominal"``.
        confidence: the confidence handed to the policy, dimensionless in
            [0, 1].
        response_id: the response to command, or None when the policy commands
            nothing.
        routed_conservative: the fault-specific response was abandoned in
            favour of the class-conservative route.
        ambiguity_class: the members of the ambiguity class considered, sorted.
            A singleton when the identification is in no declared class.
        reason: one line naming why this response was chosen.
    """

    mode: str
    believed_fault: str
    confidence: float
    response_id: str | None
    routed_conservative: bool
    ambiguity_class: tuple[str, ...]
    reason: str

    def __str__(self) -> str:
        return (
            f"{self.mode}/{self.believed_fault} (conf {self.confidence:.3f}) -> "
            f"{self.response_id or 'no action'} [{self.reason}]"
        )


class FaultResponsePolicy:
    """A fault-response policy over one declaration and detection assumption.

    Args:
        spec: the declaration.
        assumption: detection assumption; default is declared detectability
            with an unbounded latency budget.
        confidence_threshold: dimensionless threshold, >= 0. An
            identification with confidence strictly below it is routed to the
            class-conservative response. A threshold of 0 never routes
            conservatively; a threshold strictly greater than 1 always does,
            because a probability cannot exceed 1. A threshold of exactly 1
            does NOT route an identification whose confidence is exactly 1,
            which is why the always-conservative arm of the benchmark uses 2.
    """

    def __init__(
        self,
        spec: SafeModeSpec,
        assumption: DetectionAssumption | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        if isinstance(confidence_threshold, bool) or not isinstance(
            confidence_threshold, int | float
        ):
            raise TypeError("confidence_threshold must be a real number")
        if float(confidence_threshold) < 0.0:
            raise ValueError(
                f"confidence_threshold must be >= 0, got {confidence_threshold}"
            )
        self._spec = spec
        self._automaton: Automaton = build_automaton(spec, assumption)
        self._costs = min_cost_to_safe(self._automaton)
        self._threshold = float(confidence_threshold)
        self._groups: dict[str, tuple[str, ...]] = {}
        for group in spec.observability.indistinguishable_groups():
            for fault_id in group:
                self._groups[fault_id] = group

    @property
    def spec(self) -> SafeModeSpec:
        """The declaration."""
        return self._spec

    @property
    def automaton(self) -> Automaton:
        """The automaton the policy reasons over."""
        return self._automaton

    @property
    def confidence_threshold(self) -> float:
        """The declared confidence threshold, dimensionless."""
        return self._threshold

    def ambiguity_class_of(self, fault_id: str, mode_id: str) -> tuple[str, ...]:
        """Members of the ambiguity class of ``fault_id`` that occur in ``mode_id``.

        A singleton when the fault is in no declared class, or when the other
        members of its declared class cannot occur in that mode.
        """
        group = self._groups.get(fault_id, (fault_id,))
        members = tuple(f for f in group if self._spec.fault_occurs_in(f, mode_id))
        return members if members else (fault_id,)

    def conservative_route(self, fault_id: str, mode_id: str) -> tuple[str | None, float]:
        """The class-conservative route for ``fault_id`` in ``mode_id``.

        Returns ``(first_response_id, total_cost_min)``, or
        ``(None, math.inf)`` when the declaration offers no fault-agnostic
        route that keeps every class member safe-reachable.
        """
        group = self.ambiguity_class_of(fault_id, mode_id)
        return conservative_response(
            self._spec, self._automaton, self._costs, group, mode_id
        )

    def specific_route(self, fault_id: str, mode_id: str) -> tuple[str | None, float]:
        """The cheapest response naming ``fault_id`` that is enabled in ``mode_id``."""
        return preferred_specific_response(self._automaton, self._costs, mode_id, fault_id)

    def decide(self, mode_id: str, believed_fault: str, confidence: float) -> Decision:
        """Choose a response from a mode, an identification and a confidence.

        Args:
            mode_id: declared mode id.
            believed_fault: a declared fault id, or ``"nominal"``.
            confidence: dimensionless in [0, 1].

        Returns:
            A :class:`Decision`. ``response_id`` is None when the policy
            commands nothing, which happens for a nominal identification and
            when neither a specific nor a conservative route exists.
        """
        self._spec.mode(mode_id)
        if believed_fault == NOMINAL_LABEL:
            return Decision(
                mode=mode_id,
                believed_fault=believed_fault,
                confidence=float(confidence),
                response_id=None,
                routed_conservative=False,
                ambiguity_class=(),
                reason="no fault identified",
            )
        self._spec.fault(believed_fault)
        members = self.ambiguity_class_of(believed_fault, mode_id)
        if float(confidence) < self._threshold:
            response_id, _cost = self.conservative_route(believed_fault, mode_id)
            if response_id is not None:
                return Decision(
                    mode=mode_id,
                    believed_fault=believed_fault,
                    confidence=float(confidence),
                    response_id=response_id,
                    routed_conservative=True,
                    ambiguity_class=members,
                    reason=(
                        f"confidence {confidence:.3f} < {self._threshold:.3f}; "
                        f"conservative route for class of {len(members)}"
                    ),
                )
            specific_id, _ = self.specific_route(believed_fault, mode_id)
            return Decision(
                mode=mode_id,
                believed_fault=believed_fault,
                confidence=float(confidence),
                response_id=specific_id,
                routed_conservative=False,
                ambiguity_class=members,
                reason="no conservative route declared; fell back to the specific response",
            )
        specific_id, _ = self.specific_route(believed_fault, mode_id)
        if specific_id is not None:
            return Decision(
                mode=mode_id,
                believed_fault=believed_fault,
                confidence=float(confidence),
                response_id=specific_id,
                routed_conservative=False,
                ambiguity_class=members,
                reason=f"confidence {confidence:.3f} >= {self._threshold:.3f}",
            )
        response_id, _cost = self.conservative_route(believed_fault, mode_id)
        return Decision(
            mode=mode_id,
            believed_fault=believed_fault,
            confidence=float(confidence),
            response_id=response_id,
            routed_conservative=response_id is not None,
            ambiguity_class=members,
            reason="no response names this fault here; conservative route",
        )

    def cost_of(self, decision: Decision, true_fault: str) -> float:
        """Resolution cost of a decision when ``true_fault`` is really active [min].

        ``0.0`` when nothing is commanded, which is the missed-detection case
        and is counted, not costed; see the module docstring. ``math.inf`` when
        the commanded response leaves the true fault active in a state with no
        reachable safe mode.
        """
        if decision.response_id is None:
            return 0.0
        response = self._spec.response(decision.response_id)
        if true_fault == NOMINAL_LABEL:
            return response.cost_min
        return resolution_cost(
            self._automaton, self._costs, decision.mode, true_fault, response
        )

    def oracle_cost(self, mode_id: str, true_fault: str) -> float:
        """Cost of the best route available with perfect identification [min].

        The cheaper of the fault's own specific route and the conservative
        route. ``math.inf`` when neither exists.
        """
        if true_fault == NOMINAL_LABEL:
            return 0.0
        _, specific = self.specific_route(true_fault, mode_id)
        _, conservative = self.conservative_route(true_fault, mode_id)
        return min(specific, conservative)

    def is_misrouted(self, decision: Decision, true_fault: str) -> bool:
        """True when a fault-specific response for the WRONG fault was commanded.

        A conservative route is never mis-routed, because it is valid for every
        member of the class by construction. Commanding nothing is not a
        mis-route either; it is a missed detection.
        """
        if decision.response_id is None or true_fault == NOMINAL_LABEL:
            return False
        response = self._spec.response(decision.response_id)
        if response.fault is None:
            return False
        return response.fault != true_fault

    def outcome_mode(self, decision: Decision, true_fault: str) -> str | None:
        """Mode the system is in after the decision, or None when nothing is commanded."""
        if decision.response_id is None:
            return None
        return self._spec.response(decision.response_id).to_mode

    def reaches_safe(self, decision: Decision, true_fault: str) -> bool:
        """True when a declared safe mode is still reachable after the decision."""
        if decision.response_id is None:
            return math.isfinite(self._costs.get(
                AutomatonState(decision.mode, None if true_fault == NOMINAL_LABEL else true_fault),
                (math.inf, (), 0),
            )[0])
        return math.isfinite(self.cost_of(decision, true_fault))
