"""End-to-end integration: propagate, scan, unroll, route, schedule, size.

These tests run the whole pipeline the README advertises, on a small enough
scenario to fit the one-core compute budget, and check the pieces agree with
each other rather than only with themselves.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from constellink.capacity import rf_link
from constellink.contacts import all_contact_windows
from constellink.flow import ilp_max_flow
from constellink.graph import ContactGraph, TimeExpandedGraph
from constellink.queueing import route_delay
from constellink.routing import reachable_nodes, shortest_route
from constellink.synthdata import default_rf_terminal, default_stations


def test_full_pipeline_produces_a_consistent_route(small_constellation, epoch):
    stations = default_stations()
    t1 = epoch + timedelta(hours=2)
    eph = small_constellation.ephemeris(epoch, t1, 60.0)
    windows = all_contact_windows(eph, small_constellation.satellites, stations)
    nodes = ([s.name for s in small_constellation.satellites]
             + [g.name for g in stations])
    cg = ContactGraph(windows=windows, t0=epoch, t1=t1, nodes=nodes)
    teg = TimeExpandedGraph.from_contact_graph(cg, 60.0, default_rate_bps=1.0e8)

    assert windows
    assert len(cg.nodes) == len(nodes)
    route = shortest_route(teg, "W00-00", "W01-00")
    assert route is not None
    # Every transmit hop in the route must correspond to a contact window that
    # was open in that slot.
    for hop in route.hops:
        if hop.kind != "tx":
            continue
        k = hop.src[1]
        t_a = teg.slot_start(k)
        t_b = teg.slot_start(k + 1)
        open_links = {tuple(sorted((w.node_a, w.node_b)))
                      for w, _ in cg.windows_in_slot(t_a, t_b)}
        assert tuple(sorted((hop.src[0], hop.dst[0]))) in open_links


def test_route_latency_exceeds_the_pure_propagation_bound(small_graph):
    teg = TimeExpandedGraph.from_contact_graph(small_graph, 60.0,
                                                default_rate_bps=1.0e8)
    route = shortest_route(teg, "W00-00", "W01-01")
    assert route is not None
    # Each hop costs at least one slot, so total latency >= n_hops * slot_s.
    assert route.total_latency_s >= len(route.hops) * teg.slot_s


def test_reachability_and_components_agree(small_graph):
    teg = TimeExpandedGraph.from_contact_graph(small_graph, 60.0,
                                                default_rate_bps=1.0e8)
    comps = small_graph.components()
    for comp in comps:
        reach = reachable_nodes(teg, comp[0])
        # Reachability over the horizon cannot leave the union component.
        assert reach <= set(comp)


def test_max_flow_is_monotone_in_the_link_rate(small_graph):
    slow = TimeExpandedGraph.from_contact_graph(small_graph, 60.0,
                                                 default_rate_bps=1.0e7)
    fast = TimeExpandedGraph.from_contact_graph(small_graph, 60.0,
                                                default_rate_bps=1.0e8)
    unit = 6.0e8
    a = ilp_max_flow(slow, "W00-00", "W01-01", flow_unit_bits=unit)
    b = ilp_max_flow(fast, "W00-00", "W01-01", flow_unit_bits=unit)
    assert b.flow_units >= a.flow_units


def test_capacity_driven_graph_uses_the_real_link_model(small_graph):
    # Drive the unroll from the RF capacity model rather than a flat rate, and
    # check the realised edge capacities match a direct evaluation.
    term = default_rf_terminal()

    def rate_fn(window, _t_mid):
        return rf_link(window.min_range_km, term).achievable_rate_bps

    teg = TimeExpandedGraph.from_contact_graph(small_graph, 60.0,
                                                rate_fn=rate_fn)
    txs = [e for e in teg.edges if e.kind == "tx"]
    assert txs
    # A link can have several windows over the horizon, so the matching window
    # is the one open in this edge's own slot.
    for e in txs[:20]:
        k = e.src[1]
        t_a = teg.slot_start(k)
        t_b = teg.slot_start(k + 1)
        matches = [w for w, _ in small_graph.windows_in_slot(t_a, t_b)
                   if (w.node_a, w.node_b) == e.link]
        assert len(matches) == 1
        expected = rf_link(matches[0].min_range_km,
                           term).achievable_rate_bps * 60.0
        assert e.capacity_bits == pytest.approx(expected, rel=1e-9)


def test_rate_fn_graph_has_shorter_or_equal_flow_than_an_optimistic_flat_rate(
        small_graph):
    term = default_rf_terminal()
    real = TimeExpandedGraph.from_contact_graph(
        small_graph, 60.0,
        rate_fn=lambda w, _t: rf_link(w.min_range_km, term).achievable_rate_bps)
    best_rate = max(
        rf_link(w.min_range_km, term).achievable_rate_bps
        for w in small_graph.windows)
    optimistic = TimeExpandedGraph.from_contact_graph(
        small_graph, 60.0, default_rate_bps=best_rate)
    unit = 1.0e10
    a = ilp_max_flow(real, "W00-00", "W01-01", flow_unit_bits=unit)
    b = ilp_max_flow(optimistic, "W00-00", "W01-01", flow_unit_bits=unit)
    assert a.flow_units <= b.flow_units


def test_route_delay_decomposition_sums_to_the_total(small_graph):
    teg = TimeExpandedGraph.from_contact_graph(small_graph, 60.0,
                                                default_rate_bps=1.0e8)
    route = shortest_route(teg, "W00-00", "W01-01")
    assert route is not None
    rd = route_delay(route, message_bits=1.0e7, link_rate_bps=1.0e8,
                     link_range_km=2000.0, offered_load_fraction=0.4)
    assert rd.total_s == pytest.approx(sum(h.total_s for h in rd.hops),
                                       rel=1e-12)
    assert rd.total_s > rd.propagation_s


def test_dropping_a_relay_cannot_improve_the_flow(small_graph):
    teg_full = TimeExpandedGraph.from_contact_graph(small_graph, 60.0,
                                                     default_rate_bps=1.0e8)
    unit = 6.0e9
    full = ilp_max_flow(teg_full, "W00-00", "W01-01", flow_unit_bits=unit)
    mid = small_graph.t0 + (small_graph.t1 - small_graph.t0) / 2
    lost = small_graph.drop_node_after("W00-01", mid)
    teg_lost = TimeExpandedGraph.from_contact_graph(lost, 60.0,
                                                     default_rate_bps=1.0e8)
    reduced = ilp_max_flow(teg_lost, "W00-00", "W01-01", flow_unit_bits=unit)
    assert reduced.flow_units <= full.flow_units


def test_window_ranges_are_consistent_with_the_ephemeris(small_constellation,
                                                          epoch):
    eph = small_constellation.ephemeris(epoch, epoch + timedelta(hours=1), 60.0)
    windows = all_contact_windows(eph, small_constellation.satellites)
    assert windows
    for w in windows[:10]:
        ia = eph.index(w.node_a)
        ib = eph.index(w.node_b)
        ranges = np.linalg.norm(eph.r_teme_km[ib] - eph.r_teme_km[ia], axis=1)
        # The reported extremes come from the sampled interior of the window,
        # so they must lie inside the full sampled range of the link.
        assert ranges.min() - 1e-6 <= w.min_range_km
        assert w.max_range_km <= ranges.max() + 1e-6
