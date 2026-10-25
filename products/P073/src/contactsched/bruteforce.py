"""Exhaustive enumeration, used only to check the exact method.

This module exists to falsify :mod:`contactsched.exact`, so it is written to be
as independent of it as the problem allows. It shares no formulation: there is
no constraint matrix, no clique argument, no solver. It enumerates every one of
the ``2 ** n`` subsets of the candidate set by bitmask, rejects the infeasible
ones with the same ``conflicts`` predicate the greedy baseline uses, and keeps
the best. If the clique formulation in :mod:`contactsched.exact` were wrong --
if testing left endpoints were *not* sufficient, or a turnaround were applied on
one side only -- the two would disagree, and that disagreement is what
``validation/validate_exact.py`` looks for.

Enumeration is literal on purpose. A branch-and-bound or dynamic-programming
enumerator would be faster and could share a bug with the thing it is checking.
The cost is the exponential: the default limit is 18 candidates, which is 262144
subsets and a few seconds of Python. That is why this is a check and not a
method.
"""

from __future__ import annotations

from .assignment import STATUS_OPTIMAL, Assignment, objective_value
from .problem import SchedulingProblem

DEFAULT_MAX_CANDIDATES = 18
"""Largest candidate set this module will enumerate: ``2 ** 18`` subsets."""


def conflict_masks(problem: SchedulingProblem) -> list[int]:
    """Bitmask of candidates that conflict with each candidate.

    Built from :meth:`SchedulingProblem.conflicts` applied to every ordered
    pair, with no shortcut. The masks are symmetric by construction, and
    ``tests/test_bruteforce.py`` asserts that symmetry rather than assuming it.
    """
    n = len(problem.candidates)
    masks = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and problem.conflicts(problem.candidates[i], problem.candidates[j]):
                masks[i] |= 1 << j
    return masks


def request_masks(problem: SchedulingProblem) -> dict[str, tuple[int, int]]:
    """``{request_id: (candidate_bitmask, max_contacts)}``."""
    out: dict[str, tuple[int, int]] = {}
    for request in problem.requests:
        mask = 0
        for candidate in problem.candidates_of(request.request_id):
            mask |= 1 << candidate.index
        out[request.request_id] = (mask, int(request.max_contacts))
    return out


def is_feasible_mask(
    subset: int, masks: list[int], request_limits: dict[str, tuple[int, int]]
) -> bool:
    """True when the bitmask ``subset`` violates no exclusion and no ceiling."""
    remaining = subset
    while remaining:
        low = remaining & -remaining
        index = low.bit_length() - 1
        if masks[index] & subset:
            return False
        remaining ^= low
    for mask, limit in request_limits.values():
        if (subset & mask).bit_count() > limit:
            return False
    return True


def solve_bruteforce(
    problem: SchedulingProblem,
    objective_mode: str = "weighted_contacts",
    min_contacts_required: bool = False,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> Assignment:
    """Best feasible assignment, by enumerating every subset.

    Parameters
    ----------
    problem : SchedulingProblem
        The instance.
    objective_mode : str
        As in :func:`contactsched.assignment.objective_value`.
    min_contacts_required : bool
        When true, subsets that leave any request below its ``min_contacts`` are
        rejected, matching
        :attr:`contactsched.exact.FairnessPolicy.enforce_min_contacts`.
    max_candidates : int
        Refuse to enumerate above this many candidates.

    Returns
    -------
    Assignment
        Status :data:`~contactsched.assignment.STATUS_OPTIMAL`, or
        :data:`~contactsched.assignment.STATUS_INFEASIBLE` when
        ``min_contacts_required`` admits nothing.

    Raises
    ------
    ValueError
        If the candidate set is larger than ``max_candidates``.
    """
    n = len(problem.candidates)
    if n > int(max_candidates):
        raise ValueError(
            f"refusing to enumerate {n} candidates ({2 ** n} subsets); the limit is "
            f"{max_candidates}. This is a checking tool, not a scheduling method."
        )
    masks = conflict_masks(problem)
    limits = request_masks(problem)
    floors = {
        r.request_id: (limits[r.request_id][0], int(r.min_contacts)) for r in problem.requests
    }
    values = [
        problem.candidates[i].weight
        if objective_mode == "weighted_contacts"
        else problem.candidates[i].weight * problem.candidates[i].duration_s
        for i in range(n)
    ]
    if objective_mode not in {"weighted_contacts", "weighted_seconds"}:
        raise ValueError(f"unknown objective mode {objective_mode!r}")

    best_subset = -1
    best_value = float("-inf")
    for subset in range(1 << n):
        if not is_feasible_mask(subset, masks, limits):
            continue
        if min_contacts_required and any(
            (subset & mask).bit_count() < floor for mask, floor in floors.values()
        ):
            continue
        total = 0.0
        remaining = subset
        while remaining:
            low = remaining & -remaining
            total += values[low.bit_length() - 1]
            remaining ^= low
        if total > best_value:
            best_value = total
            best_subset = subset
    if best_subset < 0:
        return Assignment(
            method="brute-force-enumeration",
            status="infeasible",
            selected=(),
            objective=0.0,
            objective_mode=objective_mode,
            notes=(f"No subset of {n} candidates meets every minimum contact count.",),
        )
    selected = tuple(i for i in range(n) if best_subset >> i & 1)
    return Assignment(
        method="brute-force-enumeration",
        status=STATUS_OPTIMAL,
        selected=selected,
        objective=objective_value(problem, selected, objective_mode),
        objective_mode=objective_mode,
        notes=(f"Enumerated all {1 << n} subsets of {n} candidates.",),
    )
