"""Baseline 1: priority-ordered earliest-feasible-start insertion.

This is the analytic baseline, implemented before the exact solver and before
anything learned, and it is the thing the other two are measured against.

Algorithm
---------
1. Order the activities in a **topological order of the precedence graph**,
   tie-broken by descending priority then by name. The topological pass is not
   decoration: with a plain priority order, a successor considered before its
   predecessor can never be placed, because the feasibility check rejects a
   placed successor whose predecessor is absent. Kahn's algorithm is used
   (Kahn 1962), with the priority/name key choosing among the ready set.
2. For each activity in that order, try every admissible start step in
   increasing time order. For each candidate, insert the activity into the
   plan built so far and **simulate the whole horizon**
   (:func:`activityplan.simulate.simulate`), then evaluate every declared
   constraint (:func:`activityplan.feasibility.evaluate`). Accept the first
   candidate for which the whole plan is feasible.
3. If no candidate start is feasible, leave the activity off the timeline and
   record, per candidate, which constraint was binding. That record is the
   explanation of why the activity was dropped.

Properties, stated honestly
---------------------------
- **It is not optimal.** Earliest-feasible-start insertion in priority order is
  a one-pass constructive heuristic with no backtracking: a high-priority
  activity placed early can consume energy a pair of lower-priority activities
  would have used to better total effect. The measured gap against the exact
  optimum over a seeded instance family is in
  ``validation/validate_gap_output.txt`` and is quoted in the README; it is not
  bounded by any argument in this package.
- **It never emits an infeasible plan, provided the declaration admits the
  empty plan**, because every accepted candidate was simulated over the full
  horizon with all previously placed activities present. This is the useful
  property: it is cheap, and what it returns can be defended. The condition is
  not decoration. If the fixed loads alone break a constraint -- a bus load that
  outruns the array over an eclipse, say -- then no plan is feasible, the greedy
  planner places nothing, and the feasibility report says the *declaration* is
  infeasible. Hypothesis found that case against an unconditional statement of
  this property; it is pinned in ``tests/test_regressions.py``.
- **It is deterministic.** No random numbers, and the activity order and
  candidate order are both total.
- **Cost.** ``O(A * T)`` full-horizon simulations of ``O(T)`` steps each, so
  ``O(A * T^2)`` step evaluations for ``A`` activities and ``T`` steps. Measured
  wall clocks are in the README's compute-budget table.

Reference for the insertion heuristic family: Rabideau, G., Knight, R.,
Chien, S., Fukunaga, A. and Govindjee, A., "Iterative Repair Planning for
Spacecraft Operations using the ASPEN System", i-SAIRAS 1999 -- where
priority-ordered insertion followed by repair is the baseline the repair
iterations improve on. No repair phase is implemented here; the exact solver in
:mod:`activityplan.exact` plays the role of the reference optimum instead.

Reference for the topological ordering: Kahn, A. B., "Topological sorting of
large networks", Communications of the ACM 5(11), 1962.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .feasibility import evaluate
from .timeline import Plan, RasterisedProblem

__all__ = ["GreedyResult", "RejectionRecord", "plan_greedy", "topological_order"]


@dataclass(frozen=True)
class RejectionRecord:
    """Why one activity could not be placed.

    Attributes
    ----------
    name
        The activity that was dropped.
    candidates_tried
        Number of admissible start steps simulated.
    binding_counts
        How many of those candidates had each constraint as the binding one,
        most frequent first. Empty when ``candidates_tried`` is zero, which
        means the declaration admits no start step at all.
    """

    name: str
    candidates_tried: int
    binding_counts: tuple[tuple[str, int], ...]

    @property
    def dominant_reason(self) -> str:
        """The most frequent binding constraint, or a structural explanation."""
        if not self.binding_counts:
            return "no admissible start step in the declared window"
        return self.binding_counts[0][0]


@dataclass(frozen=True)
class GreedyResult:
    """Outcome of :func:`plan_greedy`.

    Attributes
    ----------
    plan
        The timeline produced.
    order
        The activity order actually used.
    simulations
        Number of full-horizon simulations run, the cost metric.
    rejections
        One record per activity that was not placed.
    value
        Objective value of ``plan``: the sum of the placed priorities.
    """

    plan: Plan
    order: tuple[str, ...]
    simulations: int
    rejections: tuple[RejectionRecord, ...]
    value: float


def topological_order(rasterised: RasterisedProblem) -> tuple[str, ...]:
    """Activity names in precedence-respecting, priority-descending order.

    Kahn's algorithm; among the ready set the key is
    ``(-priority, name)``. The precedence graph is known to be acyclic because
    :class:`activityplan.model.Problem` refuses a cycle at construction.
    """
    problem = rasterised.problem
    names = [activity.name for activity in problem.activities]
    successors: dict[str, list[str]] = {name: [] for name in names}
    indegree = dict.fromkeys(names, 0)
    for before, after in problem.precedences:
        successors[before].append(after)
        indegree[after] += 1

    def key(name: str) -> tuple[float, str]:
        return (-problem.activity(name).priority, name)

    ready = sorted((name for name in names if indegree[name] == 0), key=key)
    order: list[str] = []
    while ready:
        chosen = ready.pop(0)
        order.append(chosen)
        for child in successors[chosen]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort(key=key)
    if len(order) != len(names):
        raise ValueError("precedence graph is cyclic; Problem should have refused it")
    return tuple(order)


def plan_greedy(rasterised: RasterisedProblem) -> GreedyResult:
    """Build a timeline by priority-ordered earliest-feasible-start insertion.

    Parameters
    ----------
    rasterised
        The problem projected onto its grid.

    Returns
    -------
    GreedyResult
        The plan, the order used, the number of full-horizon simulations spent,
        and a rejection record for every activity left off the timeline.

    Notes
    -----
    The returned plan is feasible under
    :func:`activityplan.feasibility.evaluate` whenever the empty plan is, which
    is the condition for any plan to be feasible at all. When the declaration's
    fixed loads already violate a constraint the result is the empty plan and
    the report is infeasible. It is not optimal; see the module docstring.
    """
    order = topological_order(rasterised)
    plan = Plan({}, rasterised.grid)
    simulations = 0
    rejections: list[RejectionRecord] = []

    for name in order:
        index = rasterised.problem.index(name)
        candidates = rasterised.allowed_starts[index]
        placed = False
        binding: Counter[str] = Counter()
        for start_step in candidates.tolist():
            trial = plan.with_activity(name, int(start_step))
            report = evaluate(rasterised, trial)
            simulations += 1
            if report.feasible:
                plan = trial
                placed = True
                break
            binding[report.binding.name] += 1
        if not placed:
            rejections.append(
                RejectionRecord(
                    name=name,
                    candidates_tried=int(candidates.size),
                    binding_counts=tuple(binding.most_common()),
                )
            )

    return GreedyResult(
        plan=plan,
        order=order,
        simulations=simulations,
        rejections=tuple(rejections),
        value=plan.value(rasterised.problem),
    )
