"""Reachability of a declared safe mode, and the cheapest way to get there.

Two independent implementations are kept deliberately:

- :func:`safe_reachable_set` and :func:`min_cost_to_safe` are the ones the
  product uses. The first is a backward breadth-first fixpoint over the
  reversed transition relation, linear in states plus transitions. The second
  is multi-source Dijkstra on the reversed graph (Dijkstra, E. W., "A Note on
  Two Problems in Connexion with Graphs", Numerische Mathematik 1, pp.
  269-271, 1959), valid because every declared cost is non-negative, which the
  loader enforces.
- :func:`exhaustive_safe_reachable` and :func:`exhaustive_min_cost_to_safe`
  enumerate every simple path. They are exponential and exist only so the
  fixpoint and the Dijkstra result can be cross-checked against brute force on
  small automata, which is what ``validation/validate_reachability.py`` does.

Units: path cost is minutes of mission outage (min), the sum of declared
``cost_min`` along the path. Hop count is dimensionless.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass

from .automaton import Automaton, AutomatonState, DetectionAssumption, build_automaton
from .spec import Response, SafeModeSpec

__all__ = [
    "ReachabilityReport",
    "ReachabilityRow",
    "exhaustive_min_cost_to_safe",
    "exhaustive_safe_reachable",
    "min_cost_to_safe",
    "outcome_state",
    "preferred_specific_response",
    "reachability_report",
    "resolution_cost",
    "safe_reachable_set",
]


def safe_reachable_set(automaton: Automaton) -> frozenset[AutomatonState]:
    """States from which some declared safe state is reachable.

    Backward breadth-first fixpoint from the safe states over the reversed
    transition relation. A safe state is in the set by definition, with a path
    of length zero.
    """
    frontier: deque[AutomatonState] = deque(automaton.safe_states)
    seen: set[AutomatonState] = set(automaton.safe_states)
    while frontier:
        state = frontier.popleft()
        for transition in automaton.incoming(state):
            if transition.source not in seen:
                seen.add(transition.source)
                frontier.append(transition.source)
    return frozenset(seen)


def min_cost_to_safe(
    automaton: Automaton,
) -> dict[AutomatonState, tuple[float, tuple[str, ...], int]]:
    """Cheapest path to a declared safe state for every state that has one.

    Returns:
        Mapping state -> (cost_min, response_ids_along_the_path, hop_count).
        A safe state maps to ``(0.0, (), 0)``. States with no path are absent
        from the mapping.

    Multi-source Dijkstra on the reversed graph. Ties on cost are broken by
    fewer hops and then by the lexicographic order of the response id sequence,
    so the returned witness is deterministic.
    """
    best: dict[AutomatonState, tuple[float, int, tuple[str, ...]]] = {}
    heap: list[tuple[float, int, tuple[str, ...], AutomatonState]] = []
    for state in automaton.safe_states:
        best[state] = (0.0, 0, ())
        heapq.heappush(heap, (0.0, 0, (), state))
    while heap:
        cost, hops, path, state = heapq.heappop(heap)
        if best.get(state, (math.inf, 0, ()))[:3] < (cost, hops, path):
            continue
        for transition in automaton.incoming(state):
            source = transition.source
            candidate = (cost + transition.cost_min, hops + 1, (transition.response.id, *path))
            if source not in best or candidate < best[source]:
                best[source] = candidate
                heapq.heappush(heap, (candidate[0], candidate[1], candidate[2], source))
    return {state: (cost, path, hops) for state, (cost, hops, path) in best.items()}


def exhaustive_safe_reachable(
    automaton: Automaton, max_depth: int | None = None
) -> frozenset[AutomatonState]:
    """Brute-force reference: states from which a safe state is reachable.

    Enumerates simple paths by depth-first search from every state. Exponential
    in the automaton size; intended for cross-checking on automata of at most a
    few dozen states.

    Args:
        automaton: the automaton.
        max_depth: maximum path length in hops. None means ``len(automaton)``,
            which is the longest possible simple path.
    """
    limit = len(automaton) if max_depth is None else int(max_depth)
    if limit < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")

    def search(state: AutomatonState, visited: frozenset[AutomatonState], depth: int) -> bool:
        if automaton.is_safe(state):
            return True
        if depth >= limit:
            return False
        for transition in automaton.outgoing(state):
            nxt = transition.target
            if nxt in visited:
                continue
            if search(nxt, visited | {nxt}, depth + 1):
                return True
        return False

    return frozenset(s for s in automaton.states if search(s, frozenset({s}), 0))


def exhaustive_min_cost_to_safe(
    automaton: Automaton, max_depth: int | None = None
) -> dict[AutomatonState, float]:
    """Brute-force reference: cheapest simple-path cost to a safe state [min].

    States with no path are absent from the mapping. Exponential; for
    cross-checking only.
    """
    limit = len(automaton) if max_depth is None else int(max_depth)
    if limit < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")

    def search(state: AutomatonState, visited: frozenset[AutomatonState], depth: int) -> float:
        if automaton.is_safe(state):
            return 0.0
        if depth >= limit:
            return math.inf
        best = math.inf
        for transition in automaton.outgoing(state):
            nxt = transition.target
            if nxt in visited:
                continue
            rest = search(nxt, visited | {nxt}, depth + 1)
            if rest < math.inf:
                best = min(best, transition.cost_min + rest)
        return best

    out: dict[AutomatonState, float] = {}
    for state in automaton.states:
        cost = search(state, frozenset({state}), 0)
        if cost < math.inf:
            out[state] = cost
    return out


def outcome_state(
    response: Response, mode_id: str, true_fault: str | None
) -> AutomatonState:
    """The state after ``response`` fires while ``true_fault`` is the real fault.

    A response clears the active fault only when it is fault-agnostic or names
    the fault that is actually active. Commanding the response declared for
    fault ``g`` while fault ``f`` is active therefore changes the mode and
    leaves ``f`` active, which is what makes a mis-identification expensive.
    """
    clears = response.clears_fault and (response.fault is None or response.fault == true_fault)
    del mode_id
    return AutomatonState(response.to_mode, None if clears else true_fault)


def resolution_cost(
    automaton: Automaton,
    costs: dict[AutomatonState, tuple[float, tuple[str, ...], int]],
    mode_id: str,
    true_fault: str | None,
    response: Response,
) -> float:
    """Total cost of commanding ``response`` when ``true_fault`` is active [min].

    The declared response cost, plus the cheapest onward path to a declared
    safe mode when the fault is still active afterwards and the resulting mode
    is not itself safe. Zero onward cost when the fault is cleared or the
    resulting mode is safe.

    Returns ``math.inf`` when the fault is still active afterwards and no safe
    mode is reachable from the resulting state. Assumes declared costs are
    additive along a path and independent of order, which is the analyst's
    assumption and is not checked.
    """
    target = outcome_state(response, mode_id, true_fault)
    if target.fault is None or automaton.is_safe(target):
        return response.cost_min
    entry = costs.get(target)
    if entry is None:
        return math.inf
    return response.cost_min + entry[0]


def preferred_specific_response(
    automaton: Automaton,
    costs: dict[AutomatonState, tuple[float, tuple[str, ...], int]],
    mode_id: str,
    fault_id: str,
) -> tuple[str | None, float]:
    """The cheapest enabled response that names ``fault_id`` in ``mode_id``.

    Returns ``(response_id, resolution_cost_min)``, or ``(None, math.inf)``
    when no response naming the fault is enabled there. Ties on cost are broken
    by response id so the answer is deterministic.
    """
    state = AutomatonState(mode_id, fault_id)
    best: tuple[float, str] | None = None
    for transition in automaton.outgoing(state):
        response = transition.response
        if response.fault != fault_id:
            continue
        cost = resolution_cost(automaton, costs, mode_id, fault_id, response)
        candidate = (cost, response.id)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None, math.inf
    return best[1], best[0]


@dataclass(frozen=True)
class ReachabilityRow:
    """Safe-mode reachability for one fault state.

    Attributes:
        state: the fault state.
        reachable: a declared safe mode is reachable from it.
        cost_min: cheapest path cost in minutes of mission outage, or
            ``math.inf`` when unreachable.
        hops: number of responses on the cheapest path, 0 when already safe.
        path: response ids along the cheapest path.
        safe_mode: the safe mode the cheapest path ends in, or None.
    """

    state: AutomatonState
    reachable: bool
    cost_min: float
    hops: int
    path: tuple[str, ...]
    safe_mode: str | None

    def __str__(self) -> str:
        if not self.reachable:
            return f"{self.state}: NO SAFE MODE REACHABLE"
        route = " -> ".join(self.path) if self.path else "(already safe)"
        return (
            f"{self.state}: {self.safe_mode} in {self.hops} hop(s), "
            f"{self.cost_min:g} min [{route}]"
        )


@dataclass(frozen=True)
class ReachabilityReport:
    """Safe-mode reachability over every fault state of one declaration.

    Attributes:
        assumption: the detection assumption the result holds under.
        rows: one row per fault state, sorted by (mode, fault).
        reached_safe_modes: safe modes that are the endpoint of at least one
            cheapest path.
        unreachable_safe_modes: declared safe modes that no fault state can
            reach by any path.
    """

    assumption: DetectionAssumption
    rows: tuple[ReachabilityRow, ...]
    reached_safe_modes: tuple[str, ...]
    unreachable_safe_modes: tuple[str, ...]

    @property
    def all_reachable(self) -> bool:
        """True when every fault state can reach a declared safe mode."""
        return all(row.reachable for row in self.rows)

    @property
    def unreachable_rows(self) -> tuple[ReachabilityRow, ...]:
        """Fault states with no path to any declared safe mode."""
        return tuple(row for row in self.rows if not row.reachable)

    @property
    def worst_cost_min(self) -> float:
        """Largest cheapest-path cost over reachable fault states [min].

        ``math.inf`` if any fault state is unreachable, 0.0 if there are no
        fault states at all.
        """
        if not self.all_reachable:
            return math.inf
        return max((row.cost_min for row in self.rows), default=0.0)

    def table(self) -> str:
        """Fixed-width table of the rows."""
        lines = [
            f"{'mode':<16} {'fault':<18} {'safe?':<6} {'hops':>4} {'cost_min':>9}  path",
            "-" * 92,
        ]
        for row in self.rows:
            cost = "inf" if math.isinf(row.cost_min) else f"{row.cost_min:.3f}"
            lines.append(
                f"{row.state.mode:<16} {str(row.state.fault):<18} "
                f"{'yes' if row.reachable else 'NO':<6} {row.hops:>4} {cost:>9}  "
                f"{' -> '.join(row.path) if row.path else '(already safe)'}"
            )
        return "\n".join(lines)


def _endpoint_safe_mode(
    automaton: Automaton, state: AutomatonState, path: tuple[str, ...]
) -> str | None:
    """Replay a witness path and return the safe mode it ends in."""
    if automaton.is_safe(state):
        return state.mode
    current = state
    for response_id in path:
        for transition in automaton.outgoing(current):
            if transition.response.id == response_id:
                current = transition.target
                break
        else:  # pragma: no cover - a witness path is replayable by construction
            return None
    return current.mode if automaton.is_safe(current) else None


def _safe_modes_attained(automaton: Automaton) -> frozenset[str]:
    """Safe modes attainable by any path from any fault state.

    Forward breadth-first search from every fault state. A safe mode is
    attained when some state with that mode is forward-reachable from a fault
    state, which includes a fault state that is already in a safe mode.
    """
    frontier: deque[AutomatonState] = deque(automaton.fault_states)
    seen: set[AutomatonState] = set(automaton.fault_states)
    attained: set[str] = {s.mode for s in seen if automaton.is_safe(s)}
    while frontier:
        state = frontier.popleft()
        for transition in automaton.outgoing(state):
            target = transition.target
            if target in seen:
                continue
            seen.add(target)
            if automaton.is_safe(target):
                attained.add(target.mode)
            frontier.append(target)
    return frozenset(attained)


def reachability_report(
    spec: SafeModeSpec,
    assumption: DetectionAssumption | None = None,
    automaton: Automaton | None = None,
) -> ReachabilityReport:
    """Check safe-mode reachability from every fault state.

    Args:
        spec: the declaration.
        assumption: detection assumption; default is declared detectability
            with an unbounded latency budget.
        automaton: a pre-built automaton for ``spec`` and ``assumption``. Built
            if omitted.

    Returns:
        A :class:`ReachabilityReport`. Costs are in minutes of mission outage.
    """
    if automaton is None:
        automaton = build_automaton(spec, assumption)
    costs = min_cost_to_safe(automaton)
    rows: list[ReachabilityRow] = []
    reached: set[str] = set()
    for state in automaton.fault_states:
        entry = costs.get(state)
        if entry is None:
            rows.append(ReachabilityRow(state, False, math.inf, 0, (), None))
            continue
        cost, path, hops = entry
        safe_mode = _endpoint_safe_mode(automaton, state, path)
        if safe_mode is not None:
            reached.add(safe_mode)
        rows.append(ReachabilityRow(state, True, cost, hops, path, safe_mode))

    attained = _safe_modes_attained(automaton)
    unreachable = tuple(mode_id for mode_id in spec.safe_mode_ids if mode_id not in attained)
    return ReachabilityReport(
        assumption=automaton.assumption,
        rows=tuple(rows),
        reached_safe_modes=tuple(sorted(reached)),
        unreachable_safe_modes=unreachable,
    )
