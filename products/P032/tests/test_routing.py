"""Routing: Dijkstra against exhaustive enumeration and hand values."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from constellink.contacts import ContactWindow
from constellink.frames import SPEED_OF_LIGHT_KM_S
from constellink.graph import ContactGraph, TimeExpandedGraph
from constellink.routing import enumerate_routes, reachable_nodes, shortest_route

T0 = datetime(2026, 4, 1, tzinfo=UTC)
HOP_1000_S = 1000.0 / SPEED_OF_LIGHT_KM_S


def window(a, b, open_s, close_s, range_km=1000.0):
    return ContactWindow(node_a=a, node_b=b, kind="isl",
                         t_open=T0 + timedelta(seconds=open_s),
                         t_close=T0 + timedelta(seconds=close_s),
                         min_range_km=range_km, max_range_km=range_km,
                         max_elevation_deg=None, grid_step_s=10.0)


def build(windows, horizon_s, rate_bps=1e9, nodes=None, store_bits=float("inf")):
    cg = ContactGraph(windows=windows, t0=T0,
                      t1=T0 + timedelta(seconds=horizon_s), nodes=nodes or [])
    return TimeExpandedGraph.from_contact_graph(
        cg, 60.0, default_rate_bps=rate_bps, store_capacity_bits=store_bits)


def test_two_hop_latency_hand_value():
    # Hand calculation: two transmit hops, each 60 s of slot plus 1000 km of
    # propagation: 2 * (60 + 1000/299792.458) = 120.00667128 s.
    teg = build([window("A", "B", 0, 60), window("B", "C", 60, 120)], 180.0)
    route = shortest_route(teg, "A", "C")
    assert route is not None
    assert route.total_latency_s == pytest.approx(120.0 + 2.0 * HOP_1000_S,
                                                  rel=1e-12)
    assert route.node_sequence == ("A", "B", "C")
    assert route.n_transmissions == 2
    assert route.arrival_slot == 2


def test_direct_hop_is_chosen_when_shorter():
    teg = build([window("A", "B", 0, 60), window("B", "C", 60, 120),
                 window("A", "C", 0, 60, 3000.0)], 180.0)
    route = shortest_route(teg, "A", "C")
    assert route.node_sequence == ("A", "C")
    assert route.total_latency_s == pytest.approx(60.0 + 3000.0 / SPEED_OF_LIGHT_KM_S,
                                                   rel=1e-12)


def test_store_and_forward_adds_a_hold_hop():
    teg = build([window("A", "B", 0, 60), window("B", "C", 120, 180)], 240.0)
    route = shortest_route(teg, "A", "C")
    assert route.total_latency_s == pytest.approx(180.0 + 2.0 * HOP_1000_S,
                                                   rel=1e-12)
    assert route.n_transmissions == 2
    assert len(route.hops) == 3        # tx, hold, tx
    assert route.node_sequence == ("A", "B", "C")


def test_no_route_returns_none():
    teg = build([window("A", "B", 0, 60), window("C", "D", 0, 60)], 120.0)
    assert shortest_route(teg, "A", "C") is None
    assert enumerate_routes(teg, "A", "C") == []


def test_capacity_gate_blocks_an_oversized_message():
    # Each transmit edge carries rate * slot = 1e6 * 60 = 6e7 bits.
    teg = build([window("A", "B", 0, 60)], 120.0, rate_bps=1e6)
    assert shortest_route(teg, "A", "B", message_bits=6.0e7) is not None
    assert shortest_route(teg, "A", "B", message_bits=6.1e7) is None


def test_endpoint_validation():
    teg = build([window("A", "B", 0, 60)], 120.0)
    with pytest.raises(KeyError):
        shortest_route(teg, "Z", "B")
    with pytest.raises(KeyError):
        shortest_route(teg, "A", "Z")
    with pytest.raises(ValueError, match="must differ"):
        shortest_route(teg, "A", "A")
    with pytest.raises(ValueError, match="release_slot"):
        shortest_route(teg, "A", "B", release_slot=99)
    with pytest.raises(ValueError, match="message_bits"):
        shortest_route(teg, "A", "B", message_bits=-1.0)


def test_release_slot_shifts_the_route():
    teg = build([window("A", "B", 0, 60), window("A", "B", 120, 180)], 240.0)
    early = shortest_route(teg, "A", "B", release_slot=0)
    late = shortest_route(teg, "A", "B", release_slot=2)
    assert early.arrival_slot == 1
    assert late.arrival_slot == 3


def test_reachable_nodes_within_a_component():
    teg = build([window("A", "B", 0, 60), window("B", "C", 60, 120),
                 window("D", "E", 0, 60)], 180.0)
    assert reachable_nodes(teg, "A") == {"A", "B", "C"}
    assert reachable_nodes(teg, "D") == {"D", "E"}


def test_reachable_nodes_validation():
    teg = build([window("A", "B", 0, 60)], 120.0)
    with pytest.raises(KeyError):
        reachable_nodes(teg, "Z")
    with pytest.raises(ValueError):
        reachable_nodes(teg, "A", release_slot=42)


def test_enumerate_routes_guard():
    # All three links open for the whole 180 s horizon, so there are several
    # distinct routes and a max_routes of 1 must trip the guard.
    teg = build([window("A", "B", 0, 180), window("B", "C", 0, 180),
                 window("A", "C", 0, 180)], 180.0)
    assert len(enumerate_routes(teg, "A", "C")) > 1
    with pytest.raises(RuntimeError, match="max_routes"):
        enumerate_routes(teg, "A", "C", max_routes=1)


def test_route_node_sequence_collapses_holds():
    teg = build([window("A", "B", 0, 60), window("B", "C", 120, 180)], 240.0)
    route = shortest_route(teg, "A", "C")
    assert route.node_sequence == ("A", "B", "C")


@given(seed=st.integers(0, 2 ** 16))
@settings(max_examples=25, deadline=None)
def test_dijkstra_equals_enumeration_on_random_small_graphs(seed):
    rng = np.random.default_rng(seed)
    nodes = ["A", "B", "C", "D"]
    windows = []
    for k in range(3):
        for i in range(4):
            for j in range(i + 1, 4):
                if rng.random() < 0.5:
                    windows.append(window(nodes[i], nodes[j], k * 60.0,
                                          (k + 1) * 60.0,
                                          float(rng.uniform(500.0, 4000.0))))
    teg = build(windows, 180.0, nodes=nodes)
    d = shortest_route(teg, "A", "D")
    all_routes = enumerate_routes(teg, "A", "D")
    if d is None:
        assert all_routes == []
        return
    assert all_routes
    assert d.total_latency_s == pytest.approx(all_routes[0].total_latency_s,
                                              rel=1e-12)
    assert d.total_latency_s <= min(r.total_latency_s for r in all_routes) + 1e-12
