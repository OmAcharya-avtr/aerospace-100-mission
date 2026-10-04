"""Integer maximum-flow program vs exhaustive brute force, and the pulp model.

Part 1 -- objective agreement
-----------------------------
:func:`constellink.flow.ilp_max_flow` solves the integer program described in
:mod:`constellink.flow`.  :func:`constellink.flow.brute_force_max_flow`
enumerates every point of the capacity box, keeps the assignments that satisfy
conservation, and returns the largest delivery.  They share the network
construction and nothing else -- no solver, no algorithm, no data structure --
so agreement on the objective is a genuine cross-check.

Brute force costs ``prod_e (cap_e + 1)``, so the instances are tiny by
construction: 3-4 nodes, 2-3 slots, and capacities of a few flow units.  A
larger instance is not enumerable, which is why the ILP exists.

Part 2 -- hand-checkable instances
----------------------------------
Two instances whose optimum is written out in the script.

Part 3 -- the pulp model
------------------------
The spec for this product names ``pulp`` as the ILP front end.  ``pulp``
4.0.0 ships NO bundled CBC binary, and this build container has no external
MILP solver, so ``pulp.listSolvers(onlyAvailable=True)`` is empty and no
``pulp`` objective value can be produced here.  This part therefore reports:

* whether a pulp solver is available (and solves if one is);
* the structural equality of the pulp model and the matrix program that the
  SciPy/HiGHS backend solves -- variable count, bounds, integrality,
  objective coefficients and every conservation row.

Structural equality is weaker than an objective comparison and is labelled as
such.  It is not presented as a CBC run.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from constellink.contacts import ContactWindow  # noqa: E402
from constellink.flow import (  # noqa: E402
    brute_force_max_flow,
    build_program,
    build_pulp_model,
    compare_formulations,
    ilp_max_flow,
    pulp_solver_available,
)
from constellink.graph import ContactGraph, TimeExpandedGraph  # noqa: E402

T0 = datetime(2026, 4, 1, tzinfo=UTC)
SLOT_S = 60.0
UNIT_BITS = 60.0e6  # one flow unit = 60 Mbit = one slot at 1 Mbit/s


def window(a: str, b: str, open_s: float, close_s: float) -> ContactWindow:
    """Synthetic window at a fixed 1000 km range."""
    return ContactWindow(node_a=a, node_b=b, kind="isl",
                         t_open=T0 + timedelta(seconds=open_s),
                         t_close=T0 + timedelta(seconds=close_s),
                         min_range_km=1000.0, max_range_km=1000.0,
                         max_elevation_deg=None, grid_step_s=10.0)


def build(windows, horizon_s, rate_bps, store_units, nodes=None):
    """Assemble a time-expanded graph with integral capacities."""
    cg = ContactGraph(windows=windows, t0=T0,
                      t1=T0 + timedelta(seconds=horizon_s), nodes=nodes or [])
    return TimeExpandedGraph.from_contact_graph(
        cg, SLOT_S, default_rate_bps=rate_bps,
        store_capacity_bits=store_units * UNIT_BITS)


def hand_instances():
    """``(label, teg, src, dst, expected_units)``.

    Instance 1 -- a single chain A-B (slot 0) then B-C (slot 1), link rate
    1 Mbit/s, so each transmit edge carries one flow unit per slot.  Only one
    unit can reach C: the A-B contact exists in slot 0 only.  Buffer capacity
    is set to one flow unit per hold edge throughout, which keeps the capacity
    box small enough for brute force.  Expected 1.

    Instance 2 -- two parallel two-hop paths A-B-D and A-C-D, each hop at
    1 Mbit/s and open in the slot it is needed.  Two units reach D, one per
    path.  Expected 2.
    """
    out = []
    t1 = build([window("A", "B", 0, 60), window("B", "C", 60, 120)],
               180.0, 1.0e6, 1)
    out.append(("single chain, 1 unit", t1, "A", "C", 1))
    t2 = build([window("A", "B", 0, 60), window("B", "D", 60, 120),
                window("A", "C", 0, 60), window("C", "D", 60, 120)],
               180.0, 1.0e6, 1)
    out.append(("two parallel paths, 2 units", t2, "A", "D", 2))
    return out


def random_instance(rng: np.random.Generator, family: str):
    """A tiny random instance that brute force can still enumerate.

    Two families, because the brute-force cost is
    ``prod_e (cap_e + 1)`` and the two trade size against capacity:

    * ``"unit"`` -- 3-4 nodes, 2-3 slots, every edge capacity one flow unit
      (link rate exactly one quantum per slot).  This is the integral
      disjoint-paths case.
    * ``"mixed"`` -- 3 nodes, 2 slots, link rates of 1 or 2 quanta per slot,
      so the optimum is not forced to be a set of edge-disjoint paths.
    """
    if family == "unit":
        n_nodes = int(rng.integers(3, 5))
        n_slots = int(rng.integers(2, 4))
        rate = UNIT_BITS / SLOT_S
        store_units = 1
        occupancy = 0.55
    elif family == "mixed":
        n_nodes = 3
        n_slots = 2
        rate = float(rng.choice([1.0, 2.0])) * UNIT_BITS / SLOT_S
        store_units = 2
        occupancy = 0.7
    else:
        raise ValueError(f"unknown family {family!r}")
    nodes = ["A", "B", "C", "D"][:n_nodes]
    windows = []
    for k in range(n_slots):
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < occupancy:
                    windows.append(window(nodes[i], nodes[j], k * SLOT_S,
                                          (k + 1) * SLOT_S))
    teg = build(windows, SLOT_S * n_slots, rate, store_units, nodes=nodes)
    return teg, nodes[0], nodes[-1]


def main() -> int:
    print("Integer maximum flow: ILP vs brute force, and the pulp model")
    print("=" * 78)
    print(f"flow quantum: {UNIT_BITS / 1e6:g} Mbit; slot {SLOT_S:g} s")
    print("")

    print("Part 1 -- hand-checkable instances")
    head = (f"  {'instance':<30}{'ILP [units]':>13}{'brute [units]':>15}"
            f"{'expected':>10}{'pass':>6}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    ok1 = True
    for label, teg, src, dst, expected in hand_instances():
        a = ilp_max_flow(teg, src, dst, flow_unit_bits=UNIT_BITS)
        b = brute_force_max_flow(teg, src, dst, flow_unit_bits=UNIT_BITS)
        good = a.flow_units == b.flow_units == expected
        ok1 &= good
        print(f"  {label:<30}{a.flow_units:>13}{b.flow_units:>15}{expected:>10}"
              f"{'PASS' if good else 'FAIL':>6}")
    print(f"  backend used: {a.backend}")
    print(f"  result: {'PASS' if ok1 else 'FAIL'}")
    print("")

    print("Part 2 -- random instances small enough to enumerate")
    rng = np.random.default_rng(31415)
    ok2 = True
    n_done = 0
    n_skipped = 0
    n_nonzero = 0
    mismatches = 0
    max_units = 0
    for case in range(60):
        family = "unit" if case % 2 == 0 else "mixed"
        teg, src, dst = random_instance(rng, family)
        try:
            b = brute_force_max_flow(teg, src, dst, flow_unit_bits=UNIT_BITS,
                                     max_combinations=1_000_000)
        except ValueError:
            n_skipped += 1
            continue
        a = ilp_max_flow(teg, src, dst, flow_unit_bits=UNIT_BITS)
        n_done += 1
        max_units = max(max_units, a.flow_units)
        if a.flow_units > 0:
            n_nonzero += 1
        if a.flow_units != b.flow_units:
            mismatches += 1
            ok2 = False
            print(f"  MISMATCH: ILP {a.flow_units} vs brute {b.flow_units}")
    print(f"  instances compared        : {n_done}")
    print(f"  instances skipped (too big for brute force): {n_skipped}")
    print(f"  instances with nonzero flow: {n_nonzero}")
    print(f"  largest optimum seen      : {max_units} units "
          f"= {max_units * UNIT_BITS / 1e6:g} Mbit")
    print(f"  mismatches                : {mismatches}")
    print(f"  result: {'PASS' if ok2 else 'FAIL'}")
    print("")

    print("Part 3 -- the pulp model")
    teg, src, dst = hand_instances()[1][1], "A", "D"
    prog = build_program(teg, src, dst, 0, UNIT_BITS)
    model = build_pulp_model(prog)
    available = pulp_solver_available()
    print(f"  pulp solver available             : {available}")
    print(f"  program size                      : {prog.n_edges} variables, "
          f"{prog.n_constraints} conservation rows")
    checks = compare_formulations(prog, model)
    for name, value in checks.items():
        print(f"  structural check {name:<33}: {'PASS' if value else 'FAIL'}")
    ok3 = all(checks.values())
    if available:
        res = ilp_max_flow(teg, src, dst, flow_unit_bits=UNIT_BITS, backend="pulp")
        ref = ilp_max_flow(teg, src, dst, flow_unit_bits=UNIT_BITS, backend="scipy")
        print(f"  pulp objective                    : {res.flow_units} units")
        print(f"  scipy objective                   : {ref.flow_units} units")
        ok3 &= res.flow_units == ref.flow_units
    else:
        print("  NOT RUN: no pulp objective value was produced in this session.")
        print("           pulp 4.0.0 ships no bundled CBC binary and no external")
        print("           MILP solver is installed in this container. The pulp")
        print("           model is verified structurally only; the solved")
        print("           objective comes from scipy.optimize.milp (HiGHS).")
    print(f"  result: {'PASS' if ok3 else 'FAIL'}")
    print("")

    overall = ok1 and ok2 and ok3
    print("=" * 78)
    print(f"part 1 (hand instances)   : {'PASS' if ok1 else 'FAIL'}")
    print(f"part 2 (random instances) : {'PASS' if ok2 else 'FAIL'}")
    print(f"part 3 (pulp model)       : {'PASS' if ok3 else 'FAIL'} "
          f"(structural only -- see above)")
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
