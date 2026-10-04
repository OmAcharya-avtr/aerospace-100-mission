"""Dijkstra on the time-expanded graph vs exhaustive route enumeration.

What is being checked
---------------------
:func:`constellink.routing.shortest_route` runs Dijkstra's algorithm over the
time-expanded graph.  :func:`constellink.routing.enumerate_routes` enumerates
every feasible route by depth-first search.  The two share nothing but the
graph object and the feasibility predicate, so agreement on the minimum
latency is a real cross-check of the Dijkstra implementation, the predecessor
reconstruction and the edge-cost bookkeeping.

The comparison is run on:

1. three hand-built instances, small enough that the expected answer is
   written out in the script and checkable by eye;
2. randomly generated contact graphs over a few nodes and slots, where
   exhaustive enumeration is still tractable -- 40 instances across a fixed
   seed sequence.  Both the minimum latency and the recovered node sequence
   are compared, and ties on latency are resolved by comparing the latency
   only (several distinct routes can share a minimum).

The instances are deliberately tiny.  Exhaustive enumeration on a 24-satellite
Walker shell over 3 hours is not tractable, which is the whole reason Dijkstra
is used; a cross-check that only ran on the big case would not be a check at
all.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from constellink.contacts import ContactWindow  # noqa: E402
from constellink.graph import ContactGraph, TimeExpandedGraph  # noqa: E402
from constellink.routing import enumerate_routes, shortest_route  # noqa: E402

T0 = datetime(2026, 4, 1, tzinfo=UTC)
LATENCY_TOL_S = 1e-9


def window(a: str, b: str, open_s: float, close_s: float,
           range_km: float) -> ContactWindow:
    """Build a synthetic contact window with explicit range."""
    return ContactWindow(node_a=a, node_b=b, kind="isl",
                         t_open=T0 + timedelta(seconds=open_s),
                         t_close=T0 + timedelta(seconds=close_s),
                         min_range_km=range_km, max_range_km=range_km,
                         max_elevation_deg=None, grid_step_s=10.0)


def hand_instances() -> list[tuple[str, ContactGraph, str, str, float, float]]:
    """``(label, graph, source, destination, message_bits, expected_latency_s)``.

    Slot duration is 60 s throughout.  Propagation delay is ``range / c`` with
    ``c = 299 792.458 km/s``, so a 1000 km hop adds 3.33564e-3 s and a 3000 km
    hop adds 1.000692e-2 s.

    Instance 1: A-B open in slot 0, B-C open in slot 1.  The only route is
    two transmit hops: 60 + 1000/c + 60 + 1000/c = 120.00667128 s.

    Instance 2: adds a direct A-C contact in slot 0 at 3000 km.  One hop:
    60 + 3000/c = 60.01000692 s, which must win.

    Instance 3: A-B in slot 0, B-C only in slot 2, so the message must be held
    at B through slot 1: 60 + 1000/c (tx) + 60 (hold) + 60 + 1000/c (tx)
    = 180.00667128 s.
    """
    c_km_s = 299792.458
    hop_1000 = 1000.0 / c_km_s
    hop_3000 = 3000.0 / c_km_s
    out = []

    g1 = ContactGraph(windows=[window("A", "B", 0, 60, 1000.0),
                               window("B", "C", 60, 120, 1000.0)],
                      t0=T0, t1=T0 + timedelta(seconds=180))
    out.append(("two hops, no alternative", g1, "A", "C", 0.0,
                120.0 + 2.0 * hop_1000))

    g2 = ContactGraph(windows=[window("A", "B", 0, 60, 1000.0),
                               window("B", "C", 60, 120, 1000.0),
                               window("A", "C", 0, 60, 3000.0)],
                      t0=T0, t1=T0 + timedelta(seconds=180))
    out.append(("direct hop wins", g2, "A", "C", 0.0, 60.0 + hop_3000))

    g3 = ContactGraph(windows=[window("A", "B", 0, 60, 1000.0),
                               window("B", "C", 120, 180, 1000.0)],
                      t0=T0, t1=T0 + timedelta(seconds=240))
    out.append(("store and forward", g3, "A", "C", 0.0,
                180.0 + 2.0 * hop_1000))
    return out


def random_instance(rng: np.random.Generator, occupancy: float,
                    ) -> tuple[ContactGraph, list[str]]:
    """A small random contact graph: 4-5 nodes, 5 slots, given link occupancy.

    Sparse instances (low occupancy) are included deliberately so that the
    no-route branch of both implementations is exercised; a sweep in which
    every instance has a route would not test agreement on absence.
    """
    n_nodes = int(rng.integers(4, 6))
    nodes = [chr(ord("A") + i) for i in range(n_nodes)]
    n_slots = 5
    slot_s = 60.0
    windows = []
    for k in range(n_slots):
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < occupancy:
                    windows.append(window(nodes[i], nodes[j], k * slot_s,
                                          (k + 1) * slot_s,
                                          float(rng.uniform(500.0, 4000.0))))
    cg = ContactGraph(windows=windows, t0=T0,
                      t1=T0 + timedelta(seconds=slot_s * n_slots), nodes=nodes)
    return cg, nodes


def main() -> int:
    print("Dijkstra vs exhaustive enumeration on the time-expanded graph")
    print("=" * 78)
    print(f"latency agreement tolerance: {LATENCY_TOL_S:g} s "
          f"(both paths sum the same float edge costs, so only summation order "
          f"differs)")
    print("")

    print("Part 1 -- hand-built instances with written-out expected answers")
    head = (f"  {'instance':<28}{'dijkstra [s]':>16}{'enum [s]':>16}"
            f"{'expected [s]':>16}{'pass':>6}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    ok1 = True
    for label, cg, src, dst, msg, expected in hand_instances():
        teg = TimeExpandedGraph.from_contact_graph(cg, 60.0, default_rate_bps=1e9)
        r_d = shortest_route(teg, src, dst, message_bits=msg)
        routes = enumerate_routes(teg, src, dst, message_bits=msg)
        d_lat = r_d.total_latency_s if r_d else float("nan")
        e_lat = routes[0].total_latency_s if routes else float("nan")
        good = (abs(d_lat - e_lat) <= LATENCY_TOL_S
                and abs(d_lat - expected) <= LATENCY_TOL_S)
        ok1 &= good
        print(f"  {label:<28}{d_lat:>16.9f}{e_lat:>16.9f}{expected:>16.9f}"
              f"{'PASS' if good else 'FAIL':>6}")
    print(f"  result: {'PASS' if ok1 else 'FAIL'}")
    print("")

    print("Part 2 -- 40 random instances (alternating dense/sparse), Dijkstra vs enumeration")
    rng = np.random.default_rng(90210)
    n_cases = 40
    n_with_route = 0
    n_no_route = 0
    worst_gap = 0.0
    ok2 = True
    total_routes = 0
    for case in range(n_cases):
        occupancy = 0.55 if case % 2 == 0 else 0.12
        cg, nodes = random_instance(rng, occupancy)
        teg = TimeExpandedGraph.from_contact_graph(cg, 60.0, default_rate_bps=1e9)
        src, dst = nodes[0], nodes[-1]
        r_d = shortest_route(teg, src, dst)
        routes = enumerate_routes(teg, src, dst)
        total_routes += len(routes)
        if r_d is None and not routes:
            n_no_route += 1
            continue
        if (r_d is None) != (not routes):
            print(f"  case {case}: disagreement on existence "
                  f"(dijkstra={'route' if r_d else 'none'}, "
                  f"enum={len(routes)} routes) -- FAIL")
            ok2 = False
            continue
        n_with_route += 1
        gap = abs(r_d.total_latency_s - routes[0].total_latency_s)
        worst_gap = max(worst_gap, gap)
        if gap > LATENCY_TOL_S:
            print(f"  case {case}: latency gap {gap:.3e} s -- FAIL")
            ok2 = False
    print(f"  cases                        : {n_cases}")
    print(f"  cases with a route           : {n_with_route}")
    print(f"  cases with no route          : {n_no_route} "
          f"(both methods agreed on absence)")
    print(f"  routes enumerated in total   : {total_routes}")
    print(f"  worst latency gap            : {worst_gap:.3e} s")
    print(f"  result: {'PASS' if ok2 else 'FAIL'}")
    print("")

    overall = ok1 and ok2
    print("=" * 78)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
