"""Contact scheduling: greedy baseline and exact ILP (PuLP/CBC).

Problem
-------
Given candidate passes with values v_i (expected delivered data, Gbit),
select a subset S maximising sum(v_i) subject to:

* no two selected passes at the same station overlap in time (a station has
  one telescope), and
* no two selected passes of the same satellite overlap in time (a satellite
  has one downlink terminal).

Optional ``setup_time_s`` pads intervals to model slew/acquisition time.

This is a maximum-weight independent-set problem on the pass conflict graph.
Because conflicts come from interval overlaps it is an interval-graph-like
structure per resource; the ILP formulation with pairwise constraints

    max sum v_i x_i   s.t.  x_i + x_j <= 1 for conflicting (i, j), x in {0,1}

is exact (solved by CBC via PuLP).  The greedy baseline (descending value,
keep if compatible) is the classical 1/2-ish heuristic for interval
scheduling by weight; it carries no optimality guarantee and its measured
gap vs the ILP is reported in validation/VALIDATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import pulp

from .availability import Availability, expected_data
from .passes import Pass


@dataclass(frozen=True)
class ScheduleResult:
    """Scheduler output: chosen passes, their values [Gbit], and totals."""

    selected: tuple[Pass, ...]
    values: tuple[float, ...]          # value of each selected pass [Gbit]
    total_value: float                 # sum of selected values [Gbit]
    method: str                        # "greedy" or "ilp"
    n_candidates: int


def passes_conflict(p1: Pass, p2: Pass, setup_time_s: float = 0.0) -> bool:
    """True if p1 and p2 cannot both be scheduled (shared station or satellite
    and overlapping time intervals, padded by setup_time_s)."""
    if setup_time_s < 0:
        raise ValueError(f"setup_time_s must be >= 0, got {setup_time_s}")
    if p1.station.name != p2.station.name and p1.satellite != p2.satellite:
        return False
    return p1.overlaps(p2, setup_time_s=setup_time_s)


def _pass_values(passes: list[Pass], availability: Availability | None) -> list[float]:
    return [expected_data(p, availability) for p in passes]


def schedule_greedy(passes: list[Pass],
                    availability: Availability | None = None,
                    setup_time_s: float = 0.0) -> ScheduleResult:
    """Greedy baseline: consider passes by descending value, keep if compatible.

    Ties broken by earlier rise time for determinism.  O(n^2) conflict checks.
    """
    values = _pass_values(passes, availability)
    order = sorted(range(len(passes)),
                   key=lambda i: (-values[i], passes[i].t_rise, passes[i].station.name))
    chosen: list[int] = []
    for i in order:
        if all(not passes_conflict(passes[i], passes[j], setup_time_s) for j in chosen):
            chosen.append(i)
    chosen.sort(key=lambda i: passes[i].t_rise)
    return ScheduleResult(
        selected=tuple(passes[i] for i in chosen),
        values=tuple(values[i] for i in chosen),
        total_value=float(sum(values[i] for i in chosen)),
        method="greedy",
        n_candidates=len(passes),
    )


def schedule_ilp(passes: list[Pass],
                 availability: Availability | None = None,
                 setup_time_s: float = 0.0,
                 time_limit_s: float | None = 60.0) -> ScheduleResult:
    """Exact schedule via 0/1 ILP solved with CBC (bundled with PuLP).

    Raises RuntimeError if the solver does not report an optimal solution
    within ``time_limit_s``.
    """
    if not passes:
        return ScheduleResult((), (), 0.0, "ilp", 0)
    values = _pass_values(passes, availability)
    prob = pulp.LpProblem("pass_schedule", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(passes))]
    prob += pulp.lpSum(v * xi for v, xi in zip(values, x))
    for i in range(len(passes)):
        for j in range(i + 1, len(passes)):
            if passes_conflict(passes[i], passes[j], setup_time_s):
                prob += x[i] + x[j] <= 1
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"ILP solve did not reach optimality (status: {status})")
    chosen = [i for i in range(len(passes)) if (x[i].value() or 0.0) > 0.5]
    chosen.sort(key=lambda i: passes[i].t_rise)
    return ScheduleResult(
        selected=tuple(passes[i] for i in chosen),
        values=tuple(values[i] for i in chosen),
        total_value=float(sum(values[i] for i in chosen)),
        method="ilp",
        n_candidates=len(passes),
    )
