"""Integer maximum-flow program: ILP vs brute force, and the pulp model."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from constellink.contacts import ContactWindow
from constellink.flow import (
    brute_force_max_flow,
    build_flow_network,
    build_program,
    build_pulp_model,
    compare_formulations,
    ilp_max_flow,
    pulp_solver_available,
)
from constellink.graph import ContactGraph, TimeExpandedGraph

T0 = datetime(2026, 4, 1, tzinfo=UTC)
UNIT = 60.0e6          # one flow unit = 60 Mbit
RATE = UNIT / 60.0     # one unit per 60 s slot


def window(a, b, open_s, close_s):
    return ContactWindow(node_a=a, node_b=b, kind="isl",
                         t_open=T0 + timedelta(seconds=open_s),
                         t_close=T0 + timedelta(seconds=close_s),
                         min_range_km=1000.0, max_range_km=1000.0,
                         max_elevation_deg=None, grid_step_s=10.0)


def build(windows, horizon_s, rate_bps=RATE, store_units=1, nodes=None):
    cg = ContactGraph(windows=windows, t0=T0,
                      t1=T0 + timedelta(seconds=horizon_s), nodes=nodes or [])
    return TimeExpandedGraph.from_contact_graph(
        cg, 60.0, default_rate_bps=rate_bps,
        store_capacity_bits=store_units * UNIT)


def test_single_chain_delivers_one_unit():
    teg = build([window("A", "B", 0, 60), window("B", "C", 60, 120)], 180.0)
    res = ilp_max_flow(teg, "A", "C", flow_unit_bits=UNIT)
    assert res.flow_units == 1
    assert res.delivered_bits == pytest.approx(UNIT)
    assert res.status == "Optimal"


def test_two_parallel_paths_deliver_two_units():
    teg = build([window("A", "B", 0, 60), window("B", "D", 60, 120),
                 window("A", "C", 0, 60), window("C", "D", 60, 120)], 180.0)
    res = ilp_max_flow(teg, "A", "D", flow_unit_bits=UNIT)
    assert res.flow_units == 2


def test_bottleneck_limits_the_flow():
    # A-B at two units per slot, B-C at one: the bottleneck is B-C.
    cg = ContactGraph(windows=[window("A", "B", 0, 60), window("B", "C", 60, 120)],
                      t0=T0, t1=T0 + timedelta(seconds=180))
    teg = TimeExpandedGraph.from_contact_graph(
        cg, 60.0,
        rate_fn=lambda w, _t: 2.0 * RATE if w.node_a == "A" else RATE,
        store_capacity_bits=UNIT)
    res = ilp_max_flow(teg, "A", "C", flow_unit_bits=UNIT)
    assert res.flow_units == 1


def test_disconnected_graph_delivers_nothing():
    teg = build([window("A", "B", 0, 60), window("C", "D", 0, 60)], 120.0)
    assert ilp_max_flow(teg, "A", "C", flow_unit_bits=UNIT).flow_units == 0


def test_empty_graph_delivers_nothing():
    cg = ContactGraph(windows=[], t0=T0, t1=T0 + timedelta(seconds=180),
                      nodes=["A", "B"])
    teg = TimeExpandedGraph.from_contact_graph(cg, 60.0)
    assert ilp_max_flow(teg, "A", "B", flow_unit_bits=UNIT).flow_units == 0


def test_ilp_matches_brute_force_on_hand_instances():
    for windows, horizon, src, dst in (
            ([window("A", "B", 0, 60), window("B", "C", 60, 120)], 180.0, "A", "C"),
            ([window("A", "B", 0, 60), window("B", "D", 60, 120),
              window("A", "C", 0, 60), window("C", "D", 60, 120)], 180.0, "A", "D"),
    ):
        teg = build(windows, horizon)
        a = ilp_max_flow(teg, src, dst, flow_unit_bits=UNIT)
        b = brute_force_max_flow(teg, src, dst, flow_unit_bits=UNIT)
        assert a.flow_units == b.flow_units


@pytest.mark.parametrize("seed", range(8))
def test_ilp_matches_brute_force_on_random_small_instances(seed):
    rng = np.random.default_rng(1000 + seed)
    nodes = ["A", "B", "C"]
    windows = []
    for k in range(2):
        for i in range(3):
            for j in range(i + 1, 3):
                if rng.random() < 0.7:
                    windows.append(window(nodes[i], nodes[j], k * 60.0,
                                          (k + 1) * 60.0))
    teg = build(windows, 120.0, nodes=nodes)
    a = ilp_max_flow(teg, "A", "C", flow_unit_bits=UNIT)
    b = brute_force_max_flow(teg, "A", "C", flow_unit_bits=UNIT)
    assert a.flow_units == b.flow_units


def test_brute_force_refuses_a_large_capacity_box():
    teg = build([window("A", "B", 0, 60), window("B", "C", 60, 120)], 600.0,
                store_units=20)
    with pytest.raises(ValueError, match="capacity box"):
        brute_force_max_flow(teg, "A", "C", flow_unit_bits=UNIT,
                             max_combinations=100)


def test_flow_network_adds_one_sink_edge_per_slot():
    teg = build([window("A", "B", 0, 60)], 180.0)
    edges, caps, src = build_flow_network(teg, "A", "B", 0, UNIT)
    sinks = [e for e in edges if e.kind == "sink"]
    assert len(sinks) == teg.n_slots + 1
    assert src == ("A", 0)
    assert len(caps) == len(edges)


def test_flow_network_validation():
    teg = build([window("A", "B", 0, 60)], 120.0)
    with pytest.raises(ValueError, match="flow_unit_bits"):
        build_flow_network(teg, "A", "B", 0, 0.0)
    with pytest.raises(KeyError):
        build_flow_network(teg, "Z", "B", 0, UNIT)
    with pytest.raises(ValueError, match="must differ"):
        build_flow_network(teg, "A", "A", 0, UNIT)
    with pytest.raises(ValueError, match="release_slot"):
        build_flow_network(teg, "A", "B", 99, UNIT)


def test_program_rows_are_sparse_and_balanced():
    teg = build([window("A", "B", 0, 60)], 180.0)
    prog = build_program(teg, "A", "B", 0, UNIT)
    assert prog.n_constraints == len(prog.rows)
    # Every coefficient is +1 or -1; nothing else can appear in a conservation
    # row of a flow network.
    for row in prog.rows:
        assert all(abs(c) == 1.0 for c in row.values())
    sparse = prog.a_eq_sparse()
    assert sparse.shape == (prog.n_constraints, prog.n_edges)
    assert sparse.nnz == sum(len(r) for r in prog.rows)


def test_pulp_model_matches_the_matrix_program():
    teg = build([window("A", "B", 0, 60), window("B", "C", 60, 120)], 180.0)
    prog = build_program(teg, "A", "C", 0, UNIT)
    model = build_pulp_model(prog)
    checks = compare_formulations(prog, model)
    assert all(checks.values()), checks
    assert model.numVariables() == prog.n_edges
    assert model.numConstraints() == prog.n_constraints


def test_pulp_backend_raises_without_a_solver():
    teg = build([window("A", "B", 0, 60)], 120.0)
    if pulp_solver_available():
        res = ilp_max_flow(teg, "A", "B", flow_unit_bits=UNIT, backend="pulp")
        assert res.backend == "pulp"
    else:
        from constellink.flow import NoSolverError
        with pytest.raises(NoSolverError, match="MILP solver"):
            ilp_max_flow(teg, "A", "B", flow_unit_bits=UNIT, backend="pulp")


def test_ilp_backend_validation():
    teg = build([window("A", "B", 0, 60)], 120.0)
    with pytest.raises(ValueError, match="backend"):
        ilp_max_flow(teg, "A", "B", backend="gurobi")
    with pytest.raises(ValueError, match="time_limit_s"):
        ilp_max_flow(teg, "A", "B", time_limit_s=0.0)


def test_flow_is_at_least_as_large_as_a_single_route(small_graph):
    # A route delivers one message; the maximum flow cannot be smaller than the
    # single-route capacity, which is a monotonicity the two solvers must share.
    teg = TimeExpandedGraph.from_contact_graph(small_graph, 60.0,
                                                default_rate_bps=1.0e8)
    from constellink.routing import shortest_route
    unit = 6.0e9
    route = shortest_route(teg, "W00-00", "AWARUA", message_bits=unit)
    res = ilp_max_flow(teg, "W00-00", "AWARUA", flow_unit_bits=unit)
    if route is not None:
        assert res.flow_units >= 1
