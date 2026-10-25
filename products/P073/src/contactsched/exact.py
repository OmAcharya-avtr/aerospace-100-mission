"""Baseline 2: exact assignment as a mixed-integer program on HiGHS.

Formulation
-----------
One binary variable per candidate. Maximise the priority-weighted objective
subject to

* **antenna exclusion.** For each antenna and each candidate start time on it,
  the candidates whose inflated interval ``[start, end + turnaround)`` covers
  that instant sum to at most one. For intervals this point formulation is
  *exact*, not a relaxation: a family of intervals is pairwise disjoint if and
  only if no instant is covered twice, and every maximal clique of an interval
  graph contains the left endpoint of one of its members, so testing the left
  endpoints is sufficient. It is also much smaller than the pairwise encoding.
* **spacecraft exclusion.** The same, per spacecraft, because a spacecraft has
  one radio.
* **contact ceilings.** Each request takes at most ``max_contacts`` candidates.
* **declared fairness constraints**, optional, described in
  :class:`FairnessPolicy`.

Why HiGHS through SciPy, and not PuLP
-------------------------------------
The specification for this product named a PuLP MILP. That was measured and
rejected, and the measurement is the reason:

* ``pulp`` 4.1.0, the version in the environment this product was built and
  measured in, returns ``[]`` from ``pulp.listSolvers(onlyAvailable=True)``. It
  ships no solver. ``PULP_CBC_CMD`` is not even present in
  ``pulp.listSolvers()`` any more, CBC having been unbundled in the 4.x series.
* A PuLP-backed exact baseline would therefore raise at solve time rather than
  return a schedule, both here and on any machine without a separately
  installed solver.

``scipy.optimize.milp`` wraps HiGHS, is part of the ``scipy`` dependency this
package already needs, and solves the instances in this repository to proven
optimality. ``pulp`` is **not** a dependency of this package. It is named in
README.md among the alternatives, as the specification requires, together with
this measurement.

:func:`available_backends` probes for optional backends at import-free runtime
and reports them absent rather than raising. HiGHS through SciPy is the default
and is the only backend any test or validation script exercises; nothing in this
repository has ever been measured on another one.

References
----------
Huangfu, Q. and Hall, J. A. J. (2018). Parallelizing the dual revised simplex
method. *Mathematical Programming Computation* 10(1), 119-142. The HiGHS solver
SciPy's ``milp`` dispatches to.

Wolsey, L. A. (1998). *Integer Programming*, Wiley. Chapter 1 for the clique
formulation of interval conflict and why it is tight for interval graphs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from .assignment import (
    STATUS_INFEASIBLE,
    STATUS_LIMIT,
    STATUS_OPTIMAL,
    Assignment,
    objective_value,
)
from .problem import SchedulingProblem

DEFAULT_BACKEND = "highs"
"""The only backend exercised by this repository's tests and validation."""


def available_backends() -> dict[str, str]:
    """Which solver backends this interpreter can actually run.

    Returns
    -------
    dict
        ``{name: detail}``. ``detail`` is a version string for an available
        backend and a sentence beginning ``"absent"`` or ``"present but
        unusable"`` otherwise. Nothing here raises, and nothing here imports a
        package this product depends on at module scope.

    Notes
    -----
    This function exists so that a reader can run it and see for themselves the
    measurement that justified the deviation from the specification, rather than
    taking a sentence in a README on trust. ``pulp`` is reported as present but
    unusable when it is installed with no solver, which is its default state as
    of 4.1.0.
    """
    import scipy

    detail: dict[str, str] = {"highs": f"available via scipy {scipy.__version__} (milp)"}
    try:
        import pulp
    except ImportError:
        detail["pulp"] = "absent: not installed, and not a dependency of this package"
    else:
        try:
            solvers = list(pulp.listSolvers(onlyAvailable=True))
        except Exception as error:  # noqa: BLE001 - a probe must never raise
            detail["pulp"] = f"present but unusable: listSolvers raised {error!r}"
        else:
            if solvers:
                detail["pulp"] = f"present with solvers {solvers}, not used by this package"
            else:
                detail["pulp"] = (
                    f"present ({pulp.__version__}) but unusable: "
                    "listSolvers(onlyAvailable=True) returned an empty list, so it "
                    "ships no solver"
                )
    try:
        from ortools.sat.python import cp_model  # noqa: F401
    except ImportError:
        detail["ortools-cpsat"] = "absent: not installed, and not a dependency of this package"
    else:
        detail["ortools-cpsat"] = "present, not used by this package"
    return detail


@dataclass(frozen=True)
class FairnessPolicy:
    """Fairness constraints the exact method can be asked to enforce.

    Attributes
    ----------
    enforce_min_contacts : bool
        When true, each request's ``min_contacts`` becomes a hard lower bound.
        This is what turns starvation from a measured outcome into an
        infeasibility, and it is why :func:`solve_exact` returns a diagnostic
        naming the binding floors: an operator cannot act on "infeasible" but
        can act on "POLARIS-A asked for three contacts and has at most two
        non-overlapping candidates".
    max_total_share : float or None
        Upper bound on each mission's share of all allocated contact seconds, in
        ``(0, 1]``. ``None`` disables the constraint. The constraint is linear:
        ``sum_{j in m} d_j x_j <= alpha * sum_j d_j x_j``.
    min_contacts_floor_override : dict or None
        Per-request override of the floor used when ``enforce_min_contacts`` is
        true, keyed by request id. Used by the renegotiation workflow: having
        been told which floor binds, a caller lowers that one floor and re-solves.
    """

    enforce_min_contacts: bool = False
    max_total_share: float | None = None
    min_contacts_floor_override: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.max_total_share is not None and not 0.0 < float(self.max_total_share) <= 1.0:
            raise ValueError(
                f"max_total_share must be within (0, 1], got {self.max_total_share}"
            )

    def floor_for(self, request_id: str, declared: int) -> int:
        """Floor actually applied to ``request_id``, after any override."""
        if self.min_contacts_floor_override is None:
            return int(declared)
        return int(self.min_contacts_floor_override.get(request_id, declared))


def standalone_capacity(problem: SchedulingProblem, request_id: str) -> int:
    """Most contacts one request could get with the whole network to itself.

    Computed by earliest-finish-time selection over the request's own
    candidates, which is optimal for maximum-cardinality interval scheduling on
    a single resource and is an upper bound here because it ignores the other
    requests entirely. A floor above this number cannot be met by any schedule,
    whatever the contention, so this is the test that separates "asked for too
    much" from "lost the contention".
    """
    candidates = sorted(problem.candidates_of(request_id), key=lambda c: (c.end_s, c.start_s))
    taken: list = []
    for candidate in candidates:
        if any(problem.conflicts(candidate, other) for other in taken):
            continue
        taken.append(candidate)
    return len(taken)


def _resource_cliques(problem: SchedulingProblem) -> list[list[int]]:
    """Rows of the exclusion constraints, as lists of candidate indices.

    One row per (resource, candidate start time) pair, deduplicated. See the
    module docstring for why left endpoints suffice.
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for candidate in problem.candidates:
        groups.setdefault(("station", candidate.station), []).append(candidate.index)
        groups.setdefault(("spacecraft", candidate.spacecraft), []).append(candidate.index)
    rows: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for (kind, _name), indices in groups.items():
        gap = (
            problem.turnaround_s(problem.candidates[indices[0]].station)
            if kind == "station"
            else float(problem.spacecraft_turnaround_s)
        )
        starts = sorted({problem.candidates[i].start_s for i in indices})
        for point in starts:
            row = [
                i
                for i in indices
                if problem.candidates[i].start_s <= point < problem.candidates[i].end_s + gap
            ]
            if len(row) < 2:
                continue
            key = tuple(sorted(row))
            if key in seen:
                continue
            seen.add(key)
            rows.append(sorted(row))
    return rows


def solve_exact(
    problem: SchedulingProblem,
    objective_mode: str = "weighted_contacts",
    policy: FairnessPolicy | None = None,
    backend: str = DEFAULT_BACKEND,
    time_limit_s: float = 60.0,
) -> Assignment:
    """Solve the assignment to proven optimality over the candidate grid.

    Parameters
    ----------
    problem : SchedulingProblem
        The instance.
    objective_mode : str
        ``"weighted_contacts"`` or ``"weighted_seconds"``.
    policy : FairnessPolicy or None
        Fairness constraints. ``None`` means none, which is the honest default:
        an unconstrained optimum is the thing a fairness constraint has to be
        measured against.
    backend : str
        ``"highs"`` only. Any other name raises with the output of
        :func:`available_backends`, so the error message states what the
        interpreter can actually run.
    time_limit_s : float
        Wall-clock limit handed to HiGHS, seconds. On expiry the status is
        :data:`~contactsched.assignment.STATUS_LIMIT` and the incumbent is
        returned.

    Returns
    -------
    Assignment
        Status :data:`~contactsched.assignment.STATUS_OPTIMAL`,
        :data:`~contactsched.assignment.STATUS_LIMIT` or
        :data:`~contactsched.assignment.STATUS_INFEASIBLE`. An infeasible result
        carries notes naming the floors that cannot be met.

    Raises
    ------
    ValueError
        On an unknown backend or objective mode, or an empty candidate set.
    """
    if backend != DEFAULT_BACKEND:
        raise ValueError(
            f"backend {backend!r} is not exercised by this package. Backends "
            f"detected in this interpreter: {available_backends()}"
        )
    policy = policy or FairnessPolicy()
    n = len(problem.candidates)
    if n == 0:
        raise ValueError("the problem has no candidates; nothing can be scheduled")

    durations = np.array([c.duration_s for c in problem.candidates], dtype=float)
    weights = np.array([c.weight for c in problem.candidates], dtype=float)
    if objective_mode == "weighted_contacts":
        objective = weights
    elif objective_mode == "weighted_seconds":
        objective = weights * durations
    else:
        raise ValueError(
            f"unknown objective mode {objective_mode!r}; expected "
            "'weighted_contacts' or 'weighted_seconds'"
        )

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row_index = 0

    for clique in _resource_cliques(problem):
        for column in clique:
            rows.append(row_index)
            cols.append(column)
            data.append(1.0)
        lower.append(-np.inf)
        upper.append(1.0)
        row_index += 1

    by_request: dict[str, list[int]] = {}
    for candidate in problem.candidates:
        by_request.setdefault(candidate.request_id, []).append(candidate.index)

    for request in problem.requests:
        indices = by_request.get(request.request_id, [])
        if not indices:
            continue
        for column in indices:
            rows.append(row_index)
            cols.append(column)
            data.append(1.0)
        lower.append(-np.inf)
        upper.append(float(request.max_contacts))
        row_index += 1

    binding: list[str] = []
    if policy.enforce_min_contacts:
        for request in problem.requests:
            floor = policy.floor_for(request.request_id, request.min_contacts)
            if floor <= 0:
                continue
            indices = by_request.get(request.request_id, [])
            capacity = standalone_capacity(problem, request.request_id)
            if capacity < floor:
                binding.append(
                    f"{request.request_id} ({request.mission}, priority "
                    f"{request.priority}) requires {floor} contacts but has at most "
                    f"{capacity} mutually compatible candidates even with the whole "
                    "network to itself"
                )
            for column in indices:
                rows.append(row_index)
                cols.append(column)
                data.append(1.0)
            lower.append(float(floor))
            upper.append(np.inf)
            row_index += 1

    if policy.max_total_share is not None:
        alpha = float(policy.max_total_share)
        for mission in problem.missions:
            for candidate in problem.candidates:
                coefficient = candidate.duration_s * (
                    (1.0 - alpha) if candidate.mission == mission else -alpha
                )
                if coefficient == 0.0:
                    continue
                rows.append(row_index)
                cols.append(candidate.index)
                data.append(coefficient)
            lower.append(-np.inf)
            upper.append(0.0)
            row_index += 1

    if row_index == 0:
        constraints: list[LinearConstraint] = []
    else:
        matrix = coo_array((data, (rows, cols)), shape=(row_index, n)).tocsr()
        constraints = [LinearConstraint(matrix, np.array(lower), np.array(upper))]

    result = milp(
        c=-objective,
        constraints=constraints,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        options={"time_limit": float(time_limit_s), "presolve": True},
    )

    if result.status == 2 or result.x is None:
        notes = tuple(binding) or (
            "No individual request floor exceeds its standalone capacity, so the "
            "infeasibility is joint: the floors are individually reachable but not "
            "simultaneously. Lower a floor, widen an allowed-station set, or accept "
            "a measured shortfall.",
        )
        return Assignment(
            method="exact-milp[highs]",
            status=STATUS_INFEASIBLE,
            selected=(),
            objective=0.0,
            objective_mode=objective_mode,
            notes=("HiGHS reported the program infeasible.",) + notes,
        )

    selected = tuple(sorted(int(i) for i in np.flatnonzero(np.asarray(result.x) > 0.5)))
    status = STATUS_OPTIMAL if result.status == 0 else STATUS_LIMIT
    notes_out = [
        "Exact over the candidate grid only; a finer slot_step_s can admit a "
        "better schedule. See validation/validate_exact_output.txt.",
        f"HiGHS status {int(result.status)}: {result.message}",
    ]
    if binding:
        notes_out.extend(binding)
    return Assignment(
        method="exact-milp[highs]",
        status=status,
        selected=selected,
        objective=objective_value(problem, selected, objective_mode),
        objective_mode=objective_mode,
        notes=tuple(notes_out),
    )
