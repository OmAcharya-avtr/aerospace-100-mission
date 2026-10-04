"""Queueing and end-to-end delay accounting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from constellink.contacts import ContactWindow
from constellink.frames import SPEED_OF_LIGHT_KM_S
from constellink.graph import ContactGraph, TimeExpandedGraph
from constellink.queueing import md1_mean_delay_s, mm1_mean_delay_s, route_delay
from constellink.routing import shortest_route

T0 = datetime(2026, 4, 1, tzinfo=UTC)


def window(a, b, open_s, close_s, range_km=1000.0):
    return ContactWindow(node_a=a, node_b=b, kind="isl",
                         t_open=T0 + timedelta(seconds=open_s),
                         t_close=T0 + timedelta(seconds=close_s),
                         min_range_km=range_km, max_range_km=range_km,
                         max_elevation_deg=None, grid_step_s=10.0)


def test_mm1_hand_value():
    # mu = 1e6/1000 = 1000 msg/s, lambda = 500 msg/s, rho = 0.5.
    # W = (1/mu) / (1 - rho) = 0.001 / 0.5 = 0.002 s.
    assert mm1_mean_delay_s(0.5e6, 1.0e6, 1000.0) == pytest.approx(0.002,
                                                                   rel=1e-12)


def test_md1_hand_value():
    # W = (1/mu) (1 + rho / (2(1-rho))) = 0.001 * (1 + 0.5/1.0) = 0.0015 s.
    assert md1_mean_delay_s(0.5e6, 1.0e6, 1000.0) == pytest.approx(0.0015,
                                                                   rel=1e-12)


def test_md1_waiting_time_is_half_of_mm1():
    # Both have the same service time, so the WAITING components differ by 2.
    service = 1000.0 / 1.0e6
    for rho in (0.1, 0.3, 0.5, 0.8, 0.95):
        w_mm1 = mm1_mean_delay_s(rho * 1.0e6, 1.0e6, 1000.0) - service
        w_md1 = md1_mean_delay_s(rho * 1.0e6, 1.0e6, 1000.0) - service
        assert w_md1 == pytest.approx(w_mm1 / 2.0, rel=1e-12)


def test_zero_load_reduces_to_one_service_time():
    service = 1000.0 / 1.0e6
    assert mm1_mean_delay_s(0.0, 1.0e6, 1000.0) == pytest.approx(service,
                                                                 rel=1e-12)
    assert md1_mean_delay_s(0.0, 1.0e6, 1000.0) == pytest.approx(service,
                                                                 rel=1e-12)


@pytest.mark.parametrize("fn", [mm1_mean_delay_s, md1_mean_delay_s])
def test_unstable_queue_raises(fn):
    with pytest.raises(ValueError, match="unstable"):
        fn(1.0e6, 1.0e6, 1000.0)
    with pytest.raises(ValueError, match="unstable"):
        fn(2.0e6, 1.0e6, 1000.0)


@pytest.mark.parametrize("fn", [mm1_mean_delay_s, md1_mean_delay_s])
@pytest.mark.parametrize("args", [(-1.0, 1.0e6, 1000.0), (0.0, 0.0, 1000.0),
                                  (0.0, 1.0e6, 0.0)])
def test_queue_input_validation(fn, args):
    with pytest.raises(ValueError):
        fn(*args)


@given(rho=st.floats(0.0, 0.98))
@settings(max_examples=40, deadline=None)
def test_delay_increases_with_load(rho):
    base = md1_mean_delay_s(0.0, 1.0e6, 1000.0)
    assert md1_mean_delay_s(rho * 1.0e6, 1.0e6, 1000.0) >= base


def test_route_delay_hand_value():
    teg = TimeExpandedGraph.from_contact_graph(
        ContactGraph(windows=[window("A", "B", 0, 60), window("B", "C", 60, 120)],
                     t0=T0, t1=T0 + timedelta(seconds=180)),
        60.0, default_rate_bps=1.0e8)
    route = shortest_route(teg, "A", "C")
    rd = route_delay(route, message_bits=1.0e6, link_rate_bps=1.0e8,
                     link_range_km=1000.0, offered_load_fraction=0.0,
                     model="md1")
    # Two transmit hops, each: propagation 1000/c, serialisation 1e6/1e8 = 10 ms,
    # queueing zero at zero load.
    assert len(rd.hops) == 2
    assert rd.propagation_s == pytest.approx(2.0 * 1000.0 / SPEED_OF_LIGHT_KM_S,
                                             rel=1e-12)
    assert rd.queueing_s == pytest.approx(0.0, abs=1e-15)
    assert rd.total_s == pytest.approx(
        2.0 * (1000.0 / SPEED_OF_LIGHT_KM_S + 0.01), rel=1e-12)


def test_route_delay_counts_hold_edges_as_waiting():
    teg = TimeExpandedGraph.from_contact_graph(
        ContactGraph(windows=[window("A", "B", 0, 60), window("B", "C", 120, 180)],
                     t0=T0, t1=T0 + timedelta(seconds=240)),
        60.0, default_rate_bps=1.0e8)
    route = shortest_route(teg, "A", "C")
    rd = route_delay(route, message_bits=1.0e6, link_rate_bps=1.0e8,
                     link_range_km=1000.0)
    holds = [h for h in rd.hops if h.kind == "hold"]
    assert len(holds) == 1
    assert holds[0].queueing_s == pytest.approx(60.0)
    assert rd.total_s > 60.0


def test_route_delay_per_link_dicts():
    teg = TimeExpandedGraph.from_contact_graph(
        ContactGraph(windows=[window("A", "B", 0, 60, 500.0),
                              window("B", "C", 60, 120, 2500.0)],
                     t0=T0, t1=T0 + timedelta(seconds=180)),
        60.0, default_rate_bps=1.0e8)
    route = shortest_route(teg, "A", "C")
    rd = route_delay(route, message_bits=1.0e6,
                     link_rate_bps={("A", "B"): 1.0e8, ("B", "C"): 5.0e7},
                     link_range_km={("A", "B"): 500.0, ("B", "C"): 2500.0})
    assert rd.hops[0].serialisation_s == pytest.approx(0.01)
    assert rd.hops[1].serialisation_s == pytest.approx(0.02)
    assert rd.hops[1].propagation_s == pytest.approx(2500.0 / SPEED_OF_LIGHT_KM_S)


def test_route_delay_missing_link_entry_raises():
    teg = TimeExpandedGraph.from_contact_graph(
        ContactGraph(windows=[window("A", "B", 0, 60)], t0=T0,
                     t1=T0 + timedelta(seconds=120)),
        60.0, default_rate_bps=1.0e8)
    route = shortest_route(teg, "A", "B")
    with pytest.raises(KeyError):
        route_delay(route, 1.0e6, {("X", "Y"): 1.0e8}, 1000.0)


def test_route_delay_validation():
    teg = TimeExpandedGraph.from_contact_graph(
        ContactGraph(windows=[window("A", "B", 0, 60)], t0=T0,
                     t1=T0 + timedelta(seconds=120)),
        60.0, default_rate_bps=1.0e8)
    route = shortest_route(teg, "A", "B")
    with pytest.raises(ValueError, match="message_bits"):
        route_delay(route, 0.0, 1.0e8, 1000.0)
    with pytest.raises(ValueError, match="offered_load_fraction"):
        route_delay(route, 1.0e6, 1.0e8, 1000.0, offered_load_fraction=1.0)
    with pytest.raises(ValueError, match="processing_s"):
        route_delay(route, 1.0e6, 1.0e8, 1000.0, processing_s=-1.0)
    with pytest.raises(ValueError, match="model"):
        route_delay(route, 1.0e6, 1.0e8, 1000.0, model="mg1")


def test_route_delay_table_renders():
    teg = TimeExpandedGraph.from_contact_graph(
        ContactGraph(windows=[window("A", "B", 0, 60), window("B", "C", 60, 120)],
                     t0=T0, t1=T0 + timedelta(seconds=180)),
        60.0, default_rate_bps=1.0e8)
    route = shortest_route(teg, "A", "C")
    text = route_delay(route, 1.0e6, 1.0e8, 1000.0,
                       offered_load_fraction=0.5).format_table()
    assert "TOTAL" in text
    assert "md1" in text
    assert text.count("\n") >= 5
