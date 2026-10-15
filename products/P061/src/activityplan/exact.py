"""Baseline 2: the exact optimum of the discretised problem, by MILP.

This is the reference the greedy baseline and the learned warm start are both
scored against. It is the optimum **of the discretised problem** -- start times
restricted to grid points, occupancy rounded up -- and never a claim about the
continuous-time optimum, which is a lower bound relationship stated in
:mod:`activityplan.timeline`.

Formulation
-----------
Time-indexed binary start variables, the classical formulation of
Pritsker, A. A. B., Watters, L. J. and Wolfe, P. M., "Multiproject scheduling
with limited resources: a zero-one programming approach", Management Science
16(1), 1969. Let ``A`` be the activities, ``T`` the steps, ``n_i`` the occupancy
of activity ``i`` in steps, and ``St_i`` its admissible start steps.

Variables
    ``x[i,t]`` in {0,1} for ``t in St_i``  -- activity ``i`` starts at step ``t``
    ``E[k]``   in ``[floor, capacity]``    -- battery energy at the end of step ``k`` [W*h]
    ``S[k]``   in ``[0, recorder]``        -- recorder content at the end of step ``k`` [Mbit]
    ``d[k]``   in ``[0, cap_k]``           -- data downlinked in step ``k`` [Mbit]
    ``g[k]``   >= 0                        -- surplus energy shunted in step ``k`` [W*h]

Objective
    maximise ``sum_i p_i * sum_t x[i,t]``, the total priority placed
    (the oversubscribed-activity objective of Chien et al. 2000). Implemented as
    a minimisation of the negative, which is what ``scipy.optimize.milp`` takes.

Constraints
    selection    ``sum_t x[i,t] <= 1``                                    for each ``i``
    battery      ``E[k] - E[k-1] + sum_{i,t} e_i[k-t] x[i,t] + g[k]
                   = solar[k] - fixed[k]``                                for each ``k``
    recorder     ``S[k] - S[k-1] - sum_{i,t} v_i[k-t] x[i,t] + d[k] = 0`` for each ``k``
    drained      ``S[T-1] = 0``                 only if ``require_empty_recorder``
    exclusion    ``sum_{t: k in occ} x[i,t] + sum_{t: k in occ} x[j,t] <= 1``
                                                            for each pair and each ``k``
    precedence   ``sum_t x[j,t] - sum_t x[i,t] <= 0``
                 ``sum_t (t + n_i) x[i,t] - sum_t t x[j,t] + T sum_t x[j,t] <= T``

with ``E[-1]`` and ``S[-1]`` the declared initial values, ``e_i[.]`` the exact
per-step activity energies and ``v_i[.]`` the per-step data volumes, both from
:class:`activityplan.timeline.RasterisedProblem` so that this model and the
simulator cannot disagree.

The second precedence row is a big-M disjunction with ``M = T``, valid because
``t + n_i <= T`` holds for every admissible start, so the row is slack whenever
``x[j,.] = 0``. Together with the first row it enforces: if ``j`` is placed then
``i`` is placed and finishes at or before ``j`` starts.

Why the battery upper limit needs the shunt variable
----------------------------------------------------
Surplus array power above a full battery is dissipated, so the physical
recursion has a ``min(., capacity)`` in it, which is not linear. The shunt
variable ``g[k] >= 0`` represents exactly that dissipation and makes the
recursion linear and *exact*, not relaxed: any physically realisable trace is
representable, and any ``g`` the solver chooses is realisable by a shunt
regulator. The simulator's clamp policy is the pointwise-maximal-``E`` choice
among admissible shunt schedules, so a plan feasible here is feasible there --
see the lemma in :mod:`activityplan.simulate`, checked for every solved instance
in ``validation/validate_gap.py``.

The same argument covers the recorder: ``d[k]`` is a free continuous decision
here and the simulator drains as fast as the contact allows, which is the
pointwise-minimal-``S`` choice.

Solver
------
``scipy.optimize.milp``, which is a wrapper over HiGHS (Huangfu, Q. and
Hall, J. A. J., "Parallelizing the dual revised simplex method", Mathematical
Programming Computation 10(1), 2018). HiGHS is a branch-and-cut solver; the
returned ``mip_gap`` and ``mip_dual_bound`` are reported rather than discarded,
because an optimum returned at a nonzero gap is not an optimum.

Complexity: the oversubscribed scheduling problem with resource constraints is
NP-hard; the time-indexed model has ``O(A*T)`` binaries and
``O(T*(1 + |exclusions|))`` rows, so instance size is bounded in practice by the
step count. See Pinedo, M., *Scheduling: Theory, Algorithms, and Systems*, 5th
edition, Springer, 2016, chapter 3, for the complexity of the underlying
problem and the time-indexed formulation.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from .feasibility import evaluate
from .timeline import Plan, RasterisedProblem

__all__ = ["ExactResult", "MilpModel", "build_model", "enumerate_optimum", "solve_exact"]

_INTEGER_TOLERANCE = 1e-6
"""A relaxed binary above this is read as 1. HiGHS returns 0.9999999998, not 1."""


@dataclass(frozen=True)
class MilpModel:
    """The assembled MILP, kept separate from the solve so it can be inspected.

    Attributes
    ----------
    objective
        Cost vector of the minimisation, length ``n_variables``.
    pairs
        ``(activity_index, start_step)`` for each binary, in column order.
    n_variables, n_constraints
        Sizes of the assembled model.
    build_s
        Wall clock spent assembling it [s].
    """

    objective: np.ndarray
    integrality: np.ndarray
    bounds: Bounds
    constraints: tuple[LinearConstraint, ...]
    pairs: tuple[tuple[int, int], ...]
    n_steps: int
    build_s: float

    @property
    def n_variables(self) -> int:
        """Total column count."""
        return int(self.objective.size)

    @property
    def n_binaries(self) -> int:
        """Number of binary start variables."""
        return len(self.pairs)

    @property
    def n_constraints(self) -> int:
        """Total row count across every constraint block."""
        return int(sum(constraint.A.shape[0] for constraint in self.constraints))


@dataclass(frozen=True)
class ExactResult:
    """Outcome of :func:`solve_exact`.

    Attributes
    ----------
    plan
        The optimal timeline found, or the best feasible one within the limits.
    value
        Objective value of ``plan`` (total priority placed).
    dual_bound
        HiGHS dual bound on the objective, in the same maximisation sense. An
        upper bound on what any plan can achieve.
    gap
        Relative MIP gap HiGHS reported at termination. ``0.0`` means proven
        optimal.
    status, message
        ``scipy.optimize.milp`` status code and message, passed through.
    node_count
        Branch-and-bound nodes explored, the solver-effort metric used in the
        warm-start comparison.
    solve_s, build_s
        Wall clock spent in the solver and in model assembly [s].
    n_variables, n_constraints, n_binaries
        Model size.
    primal_bound_applied
        The warm-start primal bound added as a cut, or ``None``.
    """

    plan: Plan
    value: float
    dual_bound: float
    gap: float
    status: int
    message: str
    node_count: int
    solve_s: float
    build_s: float
    n_variables: int
    n_constraints: int
    n_binaries: int
    primal_bound_applied: float | None

    @property
    def proven_optimal(self) -> bool:
        """Whether HiGHS reported optimal status with a zero gap.

        ``status == 0`` and ``gap <= 1e-9``. A result that is not
        ``proven_optimal`` must not be used as the reference optimum, and
        ``validation/validate_gap.py`` reports how many instances fail this.
        """
        return self.status == 0 and self.gap <= 1e-9


def build_model(
    rasterised: RasterisedProblem, *, primal_bound: float | None = None
) -> MilpModel:
    """Assemble the time-indexed MILP for ``rasterised``.

    Parameters
    ----------
    rasterised
        The problem projected onto its grid.
    primal_bound
        Optional lower bound on the objective, added as the single row
        ``-sum p_i x[i,t] <= -primal_bound``. This is the warm-start mechanism
        described in :mod:`activityplan.warmstart`. **The caller is responsible
        for having verified that a plan achieving this value is feasible**; an
        unachievable bound makes the model infeasible and the optimum is lost.

    Returns
    -------
    MilpModel

    Raises
    ------
    ValueError
        If ``charge_efficiency != 1``. A charge efficiency below one makes the
        battery recursion piecewise in the sign of the surplus, which is not
        representable in a linear model without an extra binary per step. The
        greedy planner and the simulator handle any efficiency; the exact solver
        refuses rather than silently solving a different problem.
    """
    start = time.perf_counter()
    problem = rasterised.problem
    resources = problem.resources
    if abs(resources.charge_efficiency - 1.0) > 1e-12:
        raise ValueError(
            "the MILP formulation requires charge_efficiency == 1.0, got "
            f"{resources.charge_efficiency!r}; the efficiency kink at zero surplus is "
            "piecewise linear and is not represented. Use plan_greedy for that case."
        )

    n_steps = rasterised.n_steps
    pairs: list[tuple[int, int]] = []
    for index in range(rasterised.n_activities):
        for start_step in rasterised.allowed_starts[index].tolist():
            pairs.append((index, int(start_step)))
    n_x = len(pairs)
    if n_x == 0:
        raise ValueError(
            "no activity has an admissible start step; the declaration admits only "
            "the empty plan"
        )

    off_e = n_x
    off_s = off_e + n_steps
    off_d = off_s + n_steps
    off_g = off_d + n_steps
    n_vars = off_g + n_steps

    objective = np.zeros(n_vars, dtype=float)
    for column, (index, _) in enumerate(pairs):
        objective[column] = -problem.activities[index].priority

    integrality = np.zeros(n_vars, dtype=float)
    integrality[:n_x] = 1

    lower = np.empty(n_vars, dtype=float)
    upper = np.empty(n_vars, dtype=float)
    lower[:n_x] = 0.0
    upper[:n_x] = 1.0
    lower[off_e : off_e + n_steps] = resources.floor_wh
    upper[off_e : off_e + n_steps] = resources.battery_capacity_wh
    lower[off_s : off_s + n_steps] = 0.0
    upper[off_s : off_s + n_steps] = resources.recorder_capacity_mbit
    if problem.require_empty_recorder:
        upper[off_s + n_steps - 1] = 0.0
    lower[off_d : off_d + n_steps] = 0.0
    upper[off_d : off_d + n_steps] = rasterised.downlink_cap_mbit
    lower[off_g : off_g + n_steps] = 0.0
    upper[off_g : off_g + n_steps] = np.inf

    columns_for: list[list[int]] = [[] for _ in range(rasterised.n_activities)]
    for column, (index, _) in enumerate(pairs):
        columns_for[index].append(column)

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_vals: list[float] = []
    eq_rhs: list[float] = []

    for k in range(n_steps):
        row = len(eq_rhs)
        eq_rows.append(row)
        eq_cols.append(off_e + k)
        eq_vals.append(1.0)
        rhs = float(rasterised.solar_wh[k] - rasterised.fixed_wh[k])
        if k == 0:
            rhs += resources.initial_energy_wh
        else:
            eq_rows.append(row)
            eq_cols.append(off_e + k - 1)
            eq_vals.append(-1.0)
        eq_rows.append(row)
        eq_cols.append(off_g + k)
        eq_vals.append(1.0)
        for column, (index, start_step) in enumerate(pairs):
            offset = k - start_step
            profile = rasterised.activity_energy_wh[index]
            if 0 <= offset < profile.size:
                eq_rows.append(row)
                eq_cols.append(column)
                eq_vals.append(float(profile[offset]))
        eq_rhs.append(rhs)

    for k in range(n_steps):
        row = len(eq_rhs)
        eq_rows.append(row)
        eq_cols.append(off_s + k)
        eq_vals.append(1.0)
        rhs = 0.0
        if k == 0:
            rhs += resources.initial_recorder_mbit
        else:
            eq_rows.append(row)
            eq_cols.append(off_s + k - 1)
            eq_vals.append(-1.0)
        eq_rows.append(row)
        eq_cols.append(off_d + k)
        eq_vals.append(1.0)
        for column, (index, start_step) in enumerate(pairs):
            offset = k - start_step
            profile = rasterised.activity_data_mbit[index]
            if 0 <= offset < profile.size:
                eq_rows.append(row)
                eq_cols.append(column)
                eq_vals.append(-float(profile[offset]))
        eq_rhs.append(rhs)

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_vals: list[float] = []
    ub_rhs: list[float] = []

    for index in range(rasterised.n_activities):
        row = len(ub_rhs)
        for column in columns_for[index]:
            ub_rows.append(row)
            ub_cols.append(column)
            ub_vals.append(1.0)
        ub_rhs.append(1.0)

    for first, second in problem.exclusions:
        i = problem.index(first)
        j = problem.index(second)
        occupancy_i = int(rasterised.n_occupied[i])
        occupancy_j = int(rasterised.n_occupied[j])
        for k in range(n_steps):
            row = len(ub_rhs)
            touched = False
            for column in columns_for[i]:
                start_step = pairs[column][1]
                if start_step <= k < start_step + occupancy_i:
                    ub_rows.append(row)
                    ub_cols.append(column)
                    ub_vals.append(1.0)
                    touched = True
            for column in columns_for[j]:
                start_step = pairs[column][1]
                if start_step <= k < start_step + occupancy_j:
                    ub_rows.append(row)
                    ub_cols.append(column)
                    ub_vals.append(1.0)
                    touched = True
            if touched:
                ub_rhs.append(1.0)

    for before, after in problem.precedences:
        i = problem.index(before)
        j = problem.index(after)
        occupancy_i = int(rasterised.n_occupied[i])
        row = len(ub_rhs)
        for column in columns_for[j]:
            ub_rows.append(row)
            ub_cols.append(column)
            ub_vals.append(1.0)
        for column in columns_for[i]:
            ub_rows.append(row)
            ub_cols.append(column)
            ub_vals.append(-1.0)
        ub_rhs.append(0.0)

        row = len(ub_rhs)
        for column in columns_for[i]:
            ub_rows.append(row)
            ub_cols.append(column)
            ub_vals.append(float(pairs[column][1] + occupancy_i))
        for column in columns_for[j]:
            ub_rows.append(row)
            ub_cols.append(column)
            ub_vals.append(float(n_steps - pairs[column][1]))
        ub_rhs.append(float(n_steps))

    if primal_bound is not None:
        row = len(ub_rhs)
        for column, (index, _) in enumerate(pairs):
            ub_rows.append(row)
            ub_cols.append(column)
            ub_vals.append(-problem.activities[index].priority)
        ub_rhs.append(-float(primal_bound))

    constraints: list[LinearConstraint] = []
    if eq_rhs:
        matrix = coo_array(
            (np.array(eq_vals), (np.array(eq_rows), np.array(eq_cols))),
            shape=(len(eq_rhs), n_vars),
        ).tocsr()
        rhs_array = np.array(eq_rhs, dtype=float)
        constraints.append(LinearConstraint(matrix, rhs_array, rhs_array))
    if ub_rhs:
        matrix = coo_array(
            (np.array(ub_vals), (np.array(ub_rows), np.array(ub_cols))),
            shape=(len(ub_rhs), n_vars),
        ).tocsr()
        constraints.append(
            LinearConstraint(matrix, np.full(len(ub_rhs), -np.inf), np.array(ub_rhs, dtype=float))
        )

    return MilpModel(
        objective=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=tuple(constraints),
        pairs=tuple(pairs),
        n_steps=n_steps,
        build_s=time.perf_counter() - start,
    )


def solve_exact(
    rasterised: RasterisedProblem,
    *,
    primal_bound: float | None = None,
    time_limit_s: float = 60.0,
    mip_rel_gap: float = 0.0,
    presolve: bool = True,
) -> ExactResult:
    """Solve the discretised problem to proven optimality where the limits allow.

    Parameters
    ----------
    rasterised
        The problem projected onto its grid.
    primal_bound
        Optional warm-start objective cut; see :func:`build_model`.
    time_limit_s
        HiGHS wall-clock limit [s]. On expiry the best feasible plan found is
        returned with a nonzero ``gap``, and
        :attr:`ExactResult.proven_optimal` is ``False``.
    mip_rel_gap
        Relative gap at which HiGHS may stop. ``0.0`` demands proven optimality.
    presolve
        Passed through to HiGHS. Exposed because the warm-start comparison needs
        to be able to turn it off: with presolve on, the shipped instances are
        often solved before branching starts, and a node count of zero measures
        presolve rather than search.

    Returns
    -------
    ExactResult

    Raises
    ------
    ValueError
        If the model cannot be assembled (see :func:`build_model`) or if HiGHS
        reports the problem infeasible, which for a model without a primal bound
        means the declaration admits no plan at all -- note that the empty plan
        is always feasible, so this can only happen with a primal bound that no
        plan achieves.
    """
    model = build_model(rasterised, primal_bound=primal_bound)
    options = {
        "time_limit": float(time_limit_s),
        "mip_rel_gap": float(mip_rel_gap),
        "presolve": bool(presolve),
    }
    start = time.perf_counter()
    result = milp(
        c=model.objective,
        integrality=model.integrality,
        bounds=model.bounds,
        constraints=list(model.constraints),
        options=options,
    )
    solve_s = time.perf_counter() - start

    if result.x is None:
        detail = ""
        if primal_bound is None:
            empty = evaluate(rasterised, Plan({}, rasterised.grid))
            if not empty.feasible:
                worst = empty.violations[0]
                detail = (
                    " The declared resource envelope does not admit even the empty "
                    f"plan: {worst.name} has slack {worst.slack:.6g} {worst.unit} at "
                    f"{worst.where}. Fix the declaration, not the planner."
                )
            else:
                detail = (
                    " The empty plan is feasible under the simulator, so this is a "
                    "model assembly fault and should be reported as a defect."
                )
        else:
            detail = (
                f" A primal bound of {primal_bound!r} was applied; no plan achieves "
                "it, so the warm start was not verified before being used as a cut."
            )
        raise ValueError(
            f"milp returned no solution (status {result.status}): {result.message}.{detail}"
        )

    starts: dict[str, int] = {}
    for column, (index, start_step) in enumerate(model.pairs):
        if result.x[column] > 1.0 - _INTEGER_TOLERANCE:
            name = rasterised.problem.activities[index].name
            if name in starts:
                raise ValueError(
                    f"activity {name!r} was assigned two start steps "
                    f"({starts[name]} and {start_step}); the selection row failed"
                )
            starts[name] = start_step
    plan = Plan(starts, rasterised.grid)

    node_count = int(getattr(result, "mip_node_count", -1) or 0)
    dual = getattr(result, "mip_dual_bound", None)
    dual_bound = -float(dual) if dual is not None else math.inf
    gap = float(getattr(result, "mip_gap", math.inf) or 0.0)

    return ExactResult(
        plan=plan,
        value=plan.value(rasterised.problem),
        dual_bound=dual_bound,
        gap=gap,
        status=int(result.status),
        message=str(result.message),
        node_count=node_count,
        solve_s=solve_s,
        build_s=model.build_s,
        n_variables=model.n_variables,
        n_constraints=model.n_constraints,
        n_binaries=model.n_binaries,
        primal_bound_applied=None if primal_bound is None else float(primal_bound),
    )


def enumerate_optimum(
    rasterised: RasterisedProblem, *, max_candidates: int = 2_000_000
) -> tuple[Plan, float, int]:
    """Exhaustively enumerate every assignment and return the best feasible one.

    Independent of the MILP: every candidate is checked with
    :func:`activityplan.feasibility.evaluate`, which uses the resource
    *simulator*, not the linear model. This is the reference the MILP optimum is
    cross-checked against in ``validation/validate_exact.py``.

    Each activity is assigned either one of its admissible start steps or
    "not placed", so the candidate count is ``prod_i (|St_i| + 1)``.

    Parameters
    ----------
    rasterised
        The problem projected onto its grid. Must be small.
    max_candidates
        Refuse to start if the product above exceeds this, rather than hanging.

    Returns
    -------
    tuple
        ``(best_plan, best_value, candidates_examined)``. The empty plan is
        always a candidate, so a plan is always returned.

    Raises
    ------
    ValueError
        If the candidate count exceeds ``max_candidates``.
    """
    options: list[list[int | None]] = []
    total = 1
    for index in range(rasterised.n_activities):
        choices: list[int | None] = [None]
        choices.extend(int(step) for step in rasterised.allowed_starts[index].tolist())
        options.append(choices)
        total *= len(choices)
    if total > max_candidates:
        raise ValueError(
            f"exhaustive enumeration would need {total} candidates, above the "
            f"max_candidates limit of {max_candidates}; this reference is for small "
            "instances only"
        )

    names = [activity.name for activity in rasterised.problem.activities]
    best_plan = Plan({}, rasterised.grid)
    best_value = 0.0
    examined = 0
    for combination in itertools.product(*options):
        starts = {
            names[index]: step for index, step in enumerate(combination) if step is not None
        }
        candidate = Plan(starts, rasterised.grid)
        value = candidate.value(rasterised.problem)
        examined += 1
        if value <= best_value:
            continue
        if evaluate(rasterised, candidate).feasible:
            best_plan = candidate
            best_value = value
    return best_plan, best_value, examined
