"""Contact graph and time-expanded unrolling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from constellink.contacts import ContactWindow
from constellink.frames import SPEED_OF_LIGHT_KM_S
from constellink.graph import ContactGraph, TimeExpandedGraph

T0 = datetime(2026, 4, 1, tzinfo=UTC)


def window(a, b, open_s, close_s, range_km=1000.0, kind="isl"):
    return ContactWindow(node_a=a, node_b=b, kind=kind,
                         t_open=T0 + timedelta(seconds=open_s),
                         t_close=T0 + timedelta(seconds=close_s),
                         min_range_km=range_km, max_range_km=range_km,
                         max_elevation_deg=None if kind == "isl" else 45.0,
                         grid_step_s=10.0)


def test_graph_collects_nodes_and_sorts_them():
    cg = ContactGraph(windows=[window("C", "A", 0, 60), window("B", "D", 0, 60)],
                      t0=T0, t1=T0 + timedelta(seconds=120))
    assert cg.nodes == ["A", "B", "C", "D"]


def test_graph_rejects_reversed_horizon():
    with pytest.raises(ValueError):
        ContactGraph(windows=[], t0=T0, t1=T0 - timedelta(seconds=1))


def test_graph_is_empty_and_horizon():
    cg = ContactGraph(windows=[], t0=T0, t1=T0 + timedelta(seconds=300))
    assert cg.is_empty
    assert cg.horizon_s == pytest.approx(300.0)


def test_windows_in_slot_full_coverage_default():
    cg = ContactGraph(windows=[window("A", "B", 0, 30)], t0=T0,
                      t1=T0 + timedelta(seconds=120))
    # Default min_overlap_fraction of 1.0 rejects a half-covered slot.
    assert cg.windows_in_slot(T0, T0 + timedelta(seconds=60)) == []
    got = cg.windows_in_slot(T0, T0 + timedelta(seconds=60),
                             min_overlap_fraction=0.5)
    assert len(got) == 1
    assert got[0][1] == pytest.approx(0.5)


def test_windows_in_slot_rejects_bad_fraction_and_slot():
    cg = ContactGraph(windows=[window("A", "B", 0, 60)], t0=T0,
                      t1=T0 + timedelta(seconds=120))
    with pytest.raises(ValueError):
        cg.windows_in_slot(T0, T0 + timedelta(seconds=60),
                           min_overlap_fraction=0.0)
    with pytest.raises(ValueError):
        cg.windows_in_slot(T0 + timedelta(seconds=60), T0)


def test_components_and_partition():
    cg = ContactGraph(windows=[window("A", "B", 0, 60), window("C", "D", 0, 60)],
                      t0=T0, t1=T0 + timedelta(seconds=120))
    comps = cg.components()
    assert [sorted(c) for c in comps] == [["A", "B"], ["C", "D"]]
    assert cg.is_partitioned()


def test_components_single_when_connected():
    cg = ContactGraph(windows=[window("A", "B", 0, 60), window("B", "C", 0, 60)],
                      t0=T0, t1=T0 + timedelta(seconds=120))
    assert len(cg.components()) == 1
    assert not cg.is_partitioned()


def test_adjacency_at_is_symmetric():
    cg = ContactGraph(windows=[window("A", "B", 0, 60)], t0=T0,
                      t1=T0 + timedelta(seconds=120))
    adj = cg.adjacency_at(T0, T0 + timedelta(seconds=60))
    assert adj["A"] == {"B"}
    assert adj["B"] == {"A"}


def test_drop_node_after_truncates_and_removes():
    cg = ContactGraph(windows=[window("A", "B", 0, 180), window("A", "C", 120, 180)],
                      t0=T0, t1=T0 + timedelta(seconds=240))
    cut = cg.drop_node_after("A", T0 + timedelta(seconds=90))
    kinds = {(w.node_a, w.node_b): w.duration_s for w in cut.windows}
    assert ("A", "C") not in kinds        # started after the loss
    assert kinds[("A", "B")] == pytest.approx(90.0)
    assert "A" in cut.nodes               # the node still existed


def test_drop_node_after_validates_arguments():
    cg = ContactGraph(windows=[window("A", "B", 0, 60)], t0=T0,
                      t1=T0 + timedelta(seconds=120))
    with pytest.raises(KeyError):
        cg.drop_node_after("Z", T0 + timedelta(seconds=30))
    with pytest.raises(ValueError):
        cg.drop_node_after("A", T0 + timedelta(seconds=999))


def test_time_expanded_edge_counts_and_costs():
    cg = ContactGraph(windows=[window("A", "B", 0, 60)], t0=T0,
                      t1=T0 + timedelta(seconds=180))
    teg = TimeExpandedGraph.from_contact_graph(cg, 60.0, default_rate_bps=1e6)
    assert teg.n_slots == 3
    holds = [e for e in teg.edges if e.kind == "hold"]
    txs = [e for e in teg.edges if e.kind == "tx"]
    assert len(holds) == 2 * 3           # two nodes, three slots
    assert len(txs) == 2                 # one window, both directions, one slot
    assert txs[0].cost_s == pytest.approx(60.0 + 1000.0 / SPEED_OF_LIGHT_KM_S)
    assert txs[0].capacity_bits == pytest.approx(1e6 * 60.0)
    assert teg.n_te_nodes == 2 * 4


def test_time_expanded_slot_start_and_bounds():
    cg = ContactGraph(windows=[window("A", "B", 0, 60)], t0=T0,
                      t1=T0 + timedelta(seconds=180))
    teg = TimeExpandedGraph.from_contact_graph(cg, 60.0)
    assert teg.slot_start(2) == T0 + timedelta(seconds=120)
    with pytest.raises(IndexError):
        teg.slot_start(99)


def test_time_expanded_rate_fn_is_used():
    cg = ContactGraph(windows=[window("A", "B", 0, 60)], t0=T0,
                      t1=T0 + timedelta(seconds=120))
    teg = TimeExpandedGraph.from_contact_graph(
        cg, 60.0, rate_fn=lambda _w, _t: 5.0e6)
    tx = next(e for e in teg.edges if e.kind == "tx")
    assert tx.capacity_bits == pytest.approx(5.0e6 * 60.0)


def test_time_expanded_rejects_negative_rate():
    cg = ContactGraph(windows=[window("A", "B", 0, 60)], t0=T0,
                      t1=T0 + timedelta(seconds=120))
    with pytest.raises(ValueError, match="negative rate"):
        TimeExpandedGraph.from_contact_graph(cg, 60.0,
                                            rate_fn=lambda _w, _t: -1.0)


def test_time_expanded_partial_overlap_scales_capacity():
    cg = ContactGraph(windows=[window("A", "B", 0, 30)], t0=T0,
                      t1=T0 + timedelta(seconds=120))
    teg = TimeExpandedGraph.from_contact_graph(
        cg, 60.0, default_rate_bps=1e6, min_overlap_fraction=0.4)
    tx = next(e for e in teg.edges if e.kind == "tx")
    assert tx.capacity_bits == pytest.approx(1e6 * 60.0 * 0.5)


@pytest.mark.parametrize("kwargs", [{"slot_s": 0.0}, {"slot_s": 1e6},
                                    {"default_rate_bps": 0.0}])
def test_time_expanded_rejects_bad_parameters(kwargs):
    cg = ContactGraph(windows=[window("A", "B", 0, 60)], t0=T0,
                      t1=T0 + timedelta(seconds=120))
    base = {"slot_s": 60.0}
    base.update(kwargs)
    with pytest.raises(ValueError):
        TimeExpandedGraph.from_contact_graph(cg, **base)


def test_empty_graph_still_builds_hold_skeleton():
    cg = ContactGraph(windows=[], t0=T0, t1=T0 + timedelta(seconds=180),
                      nodes=["A", "B"])
    teg = TimeExpandedGraph.from_contact_graph(cg, 60.0)
    assert all(e.kind == "hold" for e in teg.edges)
    assert len(teg.edges) == 2 * 3


def test_real_graph_has_both_link_kinds(small_graph):
    teg = TimeExpandedGraph.from_contact_graph(small_graph, 60.0,
                                                default_rate_bps=1e8)
    assert teg.n_slots == 120
    assert any(e.kind == "tx" for e in teg.edges)
    assert len(teg.nodes) == len(small_graph.nodes)
