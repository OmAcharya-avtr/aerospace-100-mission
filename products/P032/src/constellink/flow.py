"""Maximum-flow scheduling over the time-expanded graph (integer program).

Problem
-------
How many bits can be moved from a source node (released in a given slot) to a
destination node before the end of the horizon, given per-slot link
capacities?  This is a maximum dynamic flow problem, and the standard
reduction is to a static maximum flow on the time-expanded network (Ford &
Fulkerson 1958, "Constructing maximal dynamic flows from static flows",
Operations Research 6(3), 419-433).

Formulation
-----------
Flow is measured in integer *units*, each worth ``flow_unit_bits`` bits, so
the program has integral capacities.  Let ``E`` be the time-expanded edge set
augmented with sink edges ``(destination, k) -> T`` of unbounded capacity for
every slot ``k``.  With ``x_e`` the integer flow on edge ``e``:

    maximise    sum over e into T of x_e
    subject to  0 <= x_e <= floor(capacity_bits(e) / flow_unit_bits)
                sum_{e into v} x_e - sum_{e out of v} x_e = 0
                    for every time-expanded node v except the source and T

The source node carries no conservation constraint, so it is an unbounded
injector; the achievable flow is limited by downstream capacity.  A node in
the final slot has no outgoing edge, so flow stranded at the end of the
horizon is infeasible -- bits must be delivered inside the horizon.

Because the time-expanded graph is a DAG with integral capacities the
constraint matrix is a network matrix and therefore totally unimodular, so
the LP relaxation is already integral (Schrijver 1986, "Theory of Linear and
Integer Programming", Wiley, Ch. 19).  The program is nevertheless declared
integer, because integrality is what makes the brute-force cross-check in
``validation/validate_ilp.py`` a like-for-like comparison.

Solver backends, and an honest note about ``pulp``
--------------------------------------------------
Two backends build the SAME program:

* ``"scipy"`` -- ``scipy.optimize.milp``, which is a HiGHS MILP solve bundled
  with SciPy.  This is the backend that runs in this repository's validation.
* ``"pulp"`` -- a ``pulp.LpProblem``, which is the portable modelling form and
  can be written out as LP/MPS for any CBC/HiGHS/Gurobi install.

``pulp`` 4.0.0 does **not** ship a bundled CBC binary, and the build container
for this product has no external MILP solver installed, so
``pulp.listSolvers(onlyAvailable=True)`` is empty and the ``"pulp"`` backend
raises :class:`NoSolverError` there.  The ``pulp`` model is therefore built and
checked **structurally** against the SciPy program --
:func:`compare_formulations` asserts identical variable bounds, identical
objective and identical constraint coefficients -- but its objective value was
not obtained from CBC in this session.  ``validation/VALIDATION.md`` records
that explicitly rather than implying a CBC run that did not happen.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pulp
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from .graph import TeEdge, TimeExpandedGraph

__all__ = [
    "FlowResult",
    "FlowProgram",
    "NoSolverError",
    "build_flow_network",
    "build_program",
    "build_pulp_model",
    "pulp_solver_available",
    "compare_formulations",
    "ilp_max_flow",
    "brute_force_max_flow",
]

_SINK = ("__sink__", -1)


class NoSolverError(RuntimeError):
    """Raised when a requested solver backend has no usable solver installed."""


@dataclass(frozen=True)
class FlowResult:
    """Outcome of a maximum-flow schedule.

    Attributes
    ----------
    delivered_bits : total bits delivered to the destination within the horizon.
    flow_units : integer flow units delivered.
    flow_unit_bits : bits per unit.
    edge_flow_units : realised integer flow per edge index.
    status : solver status string.
    backend : which backend produced the answer.
    """

    delivered_bits: float
    flow_units: int
    flow_unit_bits: float
    edge_flow_units: dict[int, int]
    status: str
    backend: str


@dataclass(frozen=True)
class FlowProgram:
    """The integer program in matrix form, shared by every backend.

    Attributes
    ----------
    edges : edge list, including the sink edges.
    capacities : integer upper bound per edge [flow units].
    objective : objective coefficients (1 on sink edges, 0 elsewhere); the
        program MAXIMISES ``objective . x``.
    rows : one dict per conservation constraint, mapping edge index to its
        coefficient (+1 for inflow, -1 for outflow, 0 omitted).  The
        constraint matrix is stored row-sparse because it has at most a few
        nonzeros per row: a dense ``(n_constraints, n_edges)`` array for a
        24-satellite, 3 hour instance is about 0.5 GiB, which does not fit
        the compute budget this product is built to.
    constraint_nodes : the time-expanded node each row constrains.
    source_te_node : the injector node, which has no conservation row.
    """

    edges: list[TeEdge]
    capacities: np.ndarray
    objective: np.ndarray
    rows: list[dict[int, float]]
    constraint_nodes: list[tuple[str, int]]
    source_te_node: tuple[str, int]

    @property
    def n_edges(self) -> int:
        """Number of decision variables."""
        return len(self.edges)

    @property
    def n_constraints(self) -> int:
        """Number of conservation rows."""
        return len(self.rows)

    def a_eq_sparse(self) -> coo_array:
        """The conservation matrix as a SciPy sparse array."""
        data, ri, ci = [], [], []
        for r, row in enumerate(self.rows):
            for col, coef in row.items():
                ri.append(r)
                ci.append(col)
                data.append(coef)
        return coo_array((np.asarray(data, dtype=float),
                          (np.asarray(ri, dtype=int), np.asarray(ci, dtype=int))),
                         shape=(self.n_constraints, self.n_edges))


def build_flow_network(teg: TimeExpandedGraph, source: str, destination: str,
                       release_slot: int, flow_unit_bits: float,
                       ) -> tuple[list[TeEdge], list[int], tuple[str, int]]:
    """Augment the time-expanded edges with sink edges and integral capacities.

    Returns ``(edges, capacities_units, source_te_node)``.  Sink edges have
    ``dst == ("__sink__", -1)`` and a capacity large enough never to bind
    (the sum of all finite capacities, which bounds any feasible flow).
    """
    if flow_unit_bits <= 0.0:
        raise ValueError(f"flow_unit_bits must be > 0, got {flow_unit_bits}")
    if source not in teg.nodes:
        raise KeyError(f"unknown source node '{source}'")
    if destination not in teg.nodes:
        raise KeyError(f"unknown destination node '{destination}'")
    if source == destination:
        raise ValueError("source and destination must differ")
    if not 0 <= release_slot <= teg.n_slots:
        raise ValueError(f"release_slot must be in [0, {teg.n_slots}], got {release_slot}")

    edges = list(teg.edges)
    caps: list[int] = []
    for e in edges:
        if math.isinf(e.capacity_bits):
            caps.append(-1)
        else:
            caps.append(int(math.floor(e.capacity_bits / flow_unit_bits + 1e-9)))
    big = max(sum(c for c in caps if c >= 0), 1)
    caps = [big if c < 0 else c for c in caps]
    for k in range(teg.n_slots + 1):
        edges.append(TeEdge(src=(destination, k), dst=_SINK, cost_s=0.0,
                            capacity_bits=float("inf"), kind="sink"))
        caps.append(big)
    return edges, caps, (source, release_slot)


def build_program(teg: TimeExpandedGraph, source: str, destination: str,
                  release_slot: int = 0, flow_unit_bits: float = 1.0e6,
                  ) -> FlowProgram:
    """Assemble the shared integer program (see the module docstring)."""
    edges, caps, src_node = build_flow_network(
        teg, source, destination, release_slot, flow_unit_bits)
    inflow: dict[tuple[str, int], list[int]] = {}
    outflow: dict[tuple[str, int], list[int]] = {}
    for i, e in enumerate(edges):
        outflow.setdefault(e.src, []).append(i)
        inflow.setdefault(e.dst, []).append(i)
    nodes = sorted((set(inflow) | set(outflow)) - {src_node, _SINK})
    rows: list[dict[int, float]] = []
    for v in nodes:
        row: dict[int, float] = {}
        for i in inflow.get(v, []):
            row[i] = row.get(i, 0.0) + 1.0
        for i in outflow.get(v, []):
            row[i] = row.get(i, 0.0) - 1.0
        rows.append({k: c for k, c in row.items() if c != 0.0})
    obj = np.array([1.0 if e.kind == "sink" else 0.0 for e in edges])
    return FlowProgram(edges=edges, capacities=np.asarray(caps, dtype=float),
                       objective=obj, rows=rows, constraint_nodes=nodes,
                       source_te_node=src_node)


def _edge_name(i: int) -> str:
    return f"x_{i}"


def build_pulp_model(prog: FlowProgram) -> pulp.LpProblem:
    """Build the ``pulp`` model of ``prog``.  Does not solve, needs no solver."""
    problem = pulp.LpProblem("constellink_max_flow", pulp.LpMaximize)
    variables = [problem.add_variable(_edge_name(i), 0, float(prog.capacities[i]), "Integer")
                 for i in range(prog.n_edges)]
    problem += pulp.lpSum(prog.objective[i] * variables[i]
                          for i in range(prog.n_edges) if prog.objective[i] != 0.0)
    for row, v in zip(prog.rows, prog.constraint_nodes, strict=True):
        terms = [coef * variables[i] for i, coef in sorted(row.items())]
        problem += (pulp.lpSum(terms) == 0, f"cons_{v[0]}_{v[1]}")
    return problem


def pulp_solver_available() -> bool:
    """True when ``pulp`` can find an installed MILP solver on this machine."""
    try:
        return len(pulp.listSolvers(onlyAvailable=True)) > 0
    except Exception:  # pragma: no cover - defensive: pulp internals vary
        return False


def compare_formulations(prog: FlowProgram, problem: pulp.LpProblem,
                         ) -> dict[str, bool]:
    """Structural equality of the ``pulp`` model and the matrix program.

    Returns a dict of named checks, all of which must be ``True``.  This is
    what stands in for a CBC objective comparison when no MILP solver is
    installed; see the module docstring.
    """
    checks: dict[str, bool] = {}
    by_name = {v.name: v for v in problem.variables()}
    checks["variable_count"] = problem.numVariables() == prog.n_edges
    checks["constraint_count"] = problem.numConstraints() == prog.n_constraints
    bounds_ok = True
    for i in range(prog.n_edges):
        v = by_name.get(_edge_name(i))
        if v is None:
            bounds_ok = False
            break
        if v.lowBound != 0.0 or v.upBound != float(prog.capacities[i]) or v.cat != "Integer":
            bounds_ok = False
            break
    checks["variable_bounds_and_category"] = bounds_ok
    obj_terms = {v.name: c for v, c in problem.objective.items()}
    expected_obj = {_edge_name(i): prog.objective[i]
                    for i in range(prog.n_edges) if prog.objective[i] != 0.0}
    checks["objective_coefficients"] = obj_terms == expected_obj
    cons_ok = True
    by_cons = {c.name: {v.name: coef for v, coef in c.items()} for c in problem.constraints()}
    for row, v in zip(prog.rows, prog.constraint_nodes, strict=True):
        expected = {_edge_name(i): coef for i, coef in row.items()}
        if by_cons.get(f"cons_{v[0]}_{v[1]}") != expected:
            cons_ok = False
            break
    checks["constraint_coefficients"] = cons_ok
    return checks


def ilp_max_flow(teg: TimeExpandedGraph, source: str, destination: str,
                 release_slot: int = 0, flow_unit_bits: float = 1.0e6,
                 backend: str = "auto", time_limit_s: float = 60.0,
                 ) -> FlowResult:
    """Solve the integer maximum-flow schedule.

    Parameters
    ----------
    teg : the time-expanded graph.
    source, destination : node names, distinct, both present.
    release_slot : slot at which injection may begin.
    flow_unit_bits : flow quantum [bits], > 0.  A smaller quantum gives a
        finer answer and a larger program.
    backend : ``"scipy"`` (HiGHS via ``scipy.optimize.milp``), ``"pulp"``
        (needs an installed MILP solver), or ``"auto"`` -- ``"pulp"`` when a
        solver is available, otherwise ``"scipy"``.
    time_limit_s : solver wall-clock limit [s], > 0.

    Raises :class:`NoSolverError` for ``backend="pulp"`` with no solver, and
    ``RuntimeError`` if the solve does not reach optimality.
    """
    if time_limit_s <= 0.0:
        raise ValueError(f"time_limit_s must be > 0, got {time_limit_s}")
    if backend not in ("auto", "scipy", "pulp"):
        raise ValueError(f"backend must be 'auto', 'scipy' or 'pulp', got {backend!r}")
    prog = build_program(teg, source, destination, release_slot, flow_unit_bits)
    chosen = backend
    if backend == "auto":
        chosen = "pulp" if pulp_solver_available() else "scipy"
    if chosen == "pulp":
        return _solve_pulp(prog, flow_unit_bits, time_limit_s)
    return _solve_scipy(prog, flow_unit_bits, time_limit_s)


def _solve_scipy(prog: FlowProgram, flow_unit_bits: float,
                 time_limit_s: float) -> FlowResult:
    n = prog.n_edges
    constraints = ([LinearConstraint(prog.a_eq_sparse(), 0.0, 0.0)]
                   if prog.n_constraints else [])
    res = milp(c=-prog.objective,
               constraints=constraints,
               bounds=Bounds(np.zeros(n), prog.capacities),
               integrality=np.ones(n),
               options={"time_limit": time_limit_s})
    if res.status != 0:
        raise RuntimeError(
            f"scipy.optimize.milp did not reach optimality: status={res.status} "
            f"({res.message}). Reduce the horizon, increase flow_unit_bits, or raise "
            f"time_limit_s.")
    x = np.rint(np.asarray(res.x, dtype=float)).astype(int)
    units = {i: int(x[i]) for i in range(n)}
    delivered = int(sum(x[i] for i in range(n) if prog.edges[i].kind == "sink"))
    return FlowResult(delivered_bits=delivered * flow_unit_bits, flow_units=delivered,
                      flow_unit_bits=flow_unit_bits, edge_flow_units=units,
                      status="Optimal", backend="scipy-highs")


def _solve_pulp(prog: FlowProgram, flow_unit_bits: float,
                time_limit_s: float) -> FlowResult:
    # time_limit_s is accepted for signature parity with the SciPy path; pulp
    # passes a limit through its own solver object, which differs per solver
    # and is left at the solver default here.
    del time_limit_s
    if not pulp_solver_available():
        raise NoSolverError(
            "backend='pulp' requires an installed MILP solver, and "
            "pulp.listSolvers(onlyAvailable=True) is empty. pulp 4.0.0 ships no bundled "
            "CBC binary. Install CBC, HiGHS or GLPK and retry, or use backend='scipy' "
            "(HiGHS via scipy.optimize.milp), which needs nothing extra.")
    problem = build_pulp_model(prog)
    problem.solve()
    status = pulp.LpStatus[problem.status]
    if status != "Optimal":
        raise RuntimeError(f"pulp did not solve to optimality: status={status}")
    by_name = {v.name: v for v in problem.variables()}
    units = {i: int(round(by_name[_edge_name(i)].value() or 0.0))
             for i in range(prog.n_edges)}
    delivered = sum(units[i] for i in range(prog.n_edges)
                    if prog.edges[i].kind == "sink")
    return FlowResult(delivered_bits=delivered * flow_unit_bits, flow_units=delivered,
                      flow_unit_bits=flow_unit_bits, edge_flow_units=units,
                      status=status, backend="pulp")


def brute_force_max_flow(teg: TimeExpandedGraph, source: str, destination: str,
                         release_slot: int = 0, flow_unit_bits: float = 1.0e6,
                         max_combinations: int = 2_000_000) -> FlowResult:
    """Exhaustive integer maximum flow.  Independent reference only.

    Enumerates every point of the capacity box over the non-sink edges, keeps
    the feasible assignments, and returns the largest delivery.  Raises
    ``ValueError`` if the box has more than ``max_combinations`` points.
    """
    prog = build_program(teg, source, destination, release_slot, flow_unit_bits)
    edges, caps = prog.edges, prog.capacities.astype(int)
    real = [i for i, e in enumerate(edges) if e.kind != "sink"]
    total = 1
    for i in real:
        total *= int(caps[i]) + 1
        if total > max_combinations:
            raise ValueError(
                f"brute_force_max_flow: capacity box has more than {max_combinations} "
                f"points ({total} and counting); this reference implementation is for "
                f"small instances only")
    inflow: dict[tuple[str, int], list[int]] = {}
    outflow: dict[tuple[str, int], list[int]] = {}
    for i, e in enumerate(edges):
        outflow.setdefault(e.src, []).append(i)
        inflow.setdefault(e.dst, []).append(i)
    nodes = prog.constraint_nodes
    best_units = 0
    best_vec: dict[int, int] = {}
    for combo in itertools.product(*[range(int(caps[i]) + 1) for i in real]):
        assign = dict(zip(real, combo, strict=True))
        ok = True
        delivered = 0
        for v in nodes:
            net = (sum(assign.get(i, 0) for i in inflow.get(v, []))
                   - sum(assign.get(i, 0) for i in outflow.get(v, [])))
            if v[0] == destination:
                if net < 0:
                    ok = False
                    break
                delivered += net
            elif net != 0:
                ok = False
                break
        if ok and delivered > best_units:
            best_units = delivered
            best_vec = dict(assign)
    return FlowResult(delivered_bits=best_units * flow_unit_bits, flow_units=best_units,
                      flow_unit_bits=flow_unit_bits, edge_flow_units=best_vec,
                      status="BruteForce", backend="brute-force")
