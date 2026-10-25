"""Baseline 1: priority-ordered first-fit. Documented, cheap and suboptimal.

What it does
------------
Sort the requests by priority, highest first, breaking ties by request id so the
result is deterministic. Walk the sorted list; for each request, walk its
candidates in order of start time and take the first one that conflicts with
nothing already taken, until the request has ``max_contacts`` contacts or its
candidates are exhausted. Never revisit a decision.

Why it is suboptimal, specifically
----------------------------------
Three named failure modes, each of which is exhibited by a test in
``tests/test_greedy.py`` with hand-constructed inputs:

1. **Earliest-start blocking.** Taking the earliest feasible slot can occupy an
   antenna across the only window a later request has, when a later slot in the
   same window would have served both. First-fit by earliest *finish* time is
   optimal for a single machine with unit demand and no priorities; it is not
   optimal here, because candidates have different durations and the antenna is
   shared with other requests' spacecraft.

2. **Duration-blind blocking.** Greedy knows nothing about the objective. A
   short high-priority contact placed in the only slot a long lower-priority
   contact could have used destroys more value than it creates whenever the
   objective counts seconds: at the default weight base a priority-2 contact of
   300 s is worth 150 weighted seconds against a priority-1 contact's 100.
   Under the per-contact objective the trade does not arise, because each slot
   is worth more to the higher priority by construction; it arises as soon as
   durations differ and the objective counts time. ``tests/test_greedy.py``
   exhibits an instance where greedy scores 100.0 weighted seconds and the
   exact optimum scores 250.0 on the same candidate grid.

3. **Minimum contact counts are invisible to it.** Greedy optimises nothing and
   enforces nothing. A request with ``min_contacts=3`` that is reached late in
   the priority order can end with zero. The shortfall is measured by
   :func:`contactsched.problem.shortfalls`, not prevented.

This baseline is published because it is what an operations team does by hand,
and because the gap between it and the exact optimum is one of the two numbers
this package exists to report. Its win or loss against the exact method is
measured in ``validation/validate_baselines.py`` and is never tuned.
"""

from __future__ import annotations

from .assignment import STATUS_HEURISTIC, Assignment, objective_value
from .problem import SchedulingProblem


def schedule_greedy(
    problem: SchedulingProblem,
    objective_mode: str = "weighted_contacts",
    order: str = "priority",
) -> Assignment:
    """Priority-ordered first-fit assignment.

    Parameters
    ----------
    problem : SchedulingProblem
        The instance.
    objective_mode : str
        Only used to score the result; it does not change the decisions.
    order : str
        ``"priority"`` sorts by ascending priority then request id.
        ``"demand"`` sorts by descending ``min_contacts * duration_s`` then
        priority, which is a second declared heuristic kept because it wins on
        some instances and losing honestly to it is informative.

    Returns
    -------
    Assignment
        Status is always :data:`~contactsched.assignment.STATUS_HEURISTIC`.

    Raises
    ------
    ValueError
        On an unknown ``order``.
    """
    if order == "priority":
        requests = sorted(problem.requests, key=lambda r: (r.priority, r.request_id))
    elif order == "demand":
        requests = sorted(
            problem.requests,
            key=lambda r: (-r.min_contacts * r.duration_s, r.priority, r.request_id),
        )
    else:
        raise ValueError(f"unknown order {order!r}; expected 'priority' or 'demand'")

    selected: list[int] = []
    for request in requests:
        taken = 0
        candidates = sorted(
            problem.candidates_of(request.request_id), key=lambda c: (c.start_s, c.station)
        )
        for candidate in candidates:
            if taken >= request.max_contacts:
                break
            if any(problem.conflicts(candidate, problem.candidates[i]) for i in selected):
                continue
            selected.append(candidate.index)
            taken += 1

    chosen = tuple(sorted(selected))
    return Assignment(
        method=f"greedy-first-fit[{order}]",
        status=STATUS_HEURISTIC,
        selected=chosen,
        objective=objective_value(problem, chosen, objective_mode),
        objective_mode=objective_mode,
        notes=(
            "First-fit by earliest start within each request, requests taken in "
            f"{order} order. No backtracking, no lookahead, no optimality claim.",
        ),
    )
