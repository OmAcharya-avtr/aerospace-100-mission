"""Level 3 failure-mode tests.

One class per required failure mode:

* an empty contact graph,
* a partitioned constellation,
* a satellite lost mid-horizon,
* a TLE epoch far from the requested window, which must RAISE rather than
  silently extrapolate.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from constellink.constellation import (
    TLE,
    Satellite,
    TleEpochError,
    walker_delta,
)
from constellink.contacts import ContactWindow
from constellink.flow import ilp_max_flow
from constellink.graph import ContactGraph, TimeExpandedGraph
from constellink.routing import enumerate_routes, reachable_nodes, shortest_route

T0 = datetime(2026, 4, 1, tzinfo=UTC)
LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
LINE2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"


def window(a, b, open_s, close_s):
    return ContactWindow(node_a=a, node_b=b, kind="isl",
                         t_open=T0 + timedelta(seconds=open_s),
                         t_close=T0 + timedelta(seconds=close_s),
                         min_range_km=1000.0, max_range_km=1000.0,
                         max_elevation_deg=None, grid_step_s=10.0)


class TestEmptyContactGraph:
    """An empty contact graph must degrade cleanly, not raise."""

    @staticmethod
    def graph():
        return ContactGraph(windows=[], t0=T0, t1=T0 + timedelta(seconds=300),
                            nodes=["A", "B", "C"])

    def test_reports_itself_empty(self):
        cg = self.graph()
        assert cg.is_empty
        assert cg.nodes == ["A", "B", "C"]

    def test_unroll_builds_only_hold_edges(self):
        teg = TimeExpandedGraph.from_contact_graph(self.graph(), 60.0)
        assert teg.edges
        assert all(e.kind == "hold" for e in teg.edges)

    def test_routing_returns_none_not_an_exception(self):
        teg = TimeExpandedGraph.from_contact_graph(self.graph(), 60.0)
        assert shortest_route(teg, "A", "C") is None
        assert enumerate_routes(teg, "A", "C") == []

    def test_reachability_is_the_source_alone(self):
        teg = TimeExpandedGraph.from_contact_graph(self.graph(), 60.0)
        assert reachable_nodes(teg, "A") == {"A"}

    def test_max_flow_is_zero(self):
        teg = TimeExpandedGraph.from_contact_graph(self.graph(), 60.0)
        res = ilp_max_flow(teg, "A", "C", flow_unit_bits=1.0e6)
        assert res.flow_units == 0
        assert res.delivered_bits == 0.0

    def test_every_node_is_its_own_component(self):
        cg = self.graph()
        assert len(cg.components()) == 3
        assert cg.is_partitioned()


class TestPartitionedConstellation:
    """Two groups with no link between them."""

    @staticmethod
    def graph():
        return ContactGraph(windows=[window("A", "B", 0, 120),
                                     window("C", "D", 0, 120)],
                            t0=T0, t1=T0 + timedelta(seconds=240))

    def test_components_found(self):
        comps = self.graph().components()
        assert [sorted(c) for c in comps] == [["A", "B"], ["C", "D"]]

    def test_is_partitioned(self):
        assert self.graph().is_partitioned()

    def test_route_across_is_none_but_within_exists(self):
        teg = TimeExpandedGraph.from_contact_graph(self.graph(), 60.0,
                                                    default_rate_bps=1e6)
        assert shortest_route(teg, "A", "C") is None
        assert shortest_route(teg, "A", "B") is not None

    def test_unreachable_set_is_the_other_component(self):
        cg = self.graph()
        teg = TimeExpandedGraph.from_contact_graph(cg, 60.0, default_rate_bps=1e6)
        reach = reachable_nodes(teg, "A")
        assert set(cg.nodes) - reach == {"C", "D"}

    def test_max_flow_across_the_partition_is_zero(self):
        teg = TimeExpandedGraph.from_contact_graph(self.graph(), 60.0,
                                                    default_rate_bps=1e6)
        assert ilp_max_flow(teg, "A", "C",
                            flow_unit_bits=1.0e6).flow_units == 0

    def test_slotwise_partition_can_differ_from_the_union(self):
        # A link that only exists in the second slot leaves the first slot
        # partitioned even though the union graph is connected.
        cg = ContactGraph(windows=[window("A", "B", 0, 60),
                                   window("B", "C", 60, 120)],
                          t0=T0, t1=T0 + timedelta(seconds=120))
        assert not cg.is_partitioned()
        slot0 = cg.adjacency_at(T0, T0 + timedelta(seconds=60))
        assert cg.is_partitioned(slot0)


class TestSatelliteLostMidHorizon:
    """A node stops participating part-way through the horizon."""

    def test_route_disappears_when_the_relay_is_lost(self):
        cg = ContactGraph(windows=[window("A", "B", 0, 60),
                                   window("B", "C", 120, 180)],
                          t0=T0, t1=T0 + timedelta(seconds=240))
        before = TimeExpandedGraph.from_contact_graph(cg, 60.0,
                                                       default_rate_bps=1e6)
        assert shortest_route(before, "A", "C") is not None
        lost = cg.drop_node_after("B", T0 + timedelta(seconds=90))
        after = TimeExpandedGraph.from_contact_graph(lost, 60.0,
                                                     default_rate_bps=1e6)
        assert shortest_route(after, "A", "C") is None

    def test_window_straddling_the_loss_is_truncated(self):
        cg = ContactGraph(windows=[window("A", "B", 0, 180)], t0=T0,
                          t1=T0 + timedelta(seconds=240))
        cut = cg.drop_node_after("B", T0 + timedelta(seconds=90))
        assert len(cut.windows) == 1
        assert cut.windows[0].duration_s == pytest.approx(90.0)
        assert not cut.windows[0].clipped_end

    def test_window_starting_after_the_loss_is_removed(self):
        cg = ContactGraph(windows=[window("A", "B", 120, 180)], t0=T0,
                          t1=T0 + timedelta(seconds=240))
        cut = cg.drop_node_after("B", T0 + timedelta(seconds=90))
        assert cut.windows == []

    def test_lost_node_stays_in_the_node_set(self):
        cg = ContactGraph(windows=[window("A", "B", 0, 180)], t0=T0,
                          t1=T0 + timedelta(seconds=240))
        cut = cg.drop_node_after("B", T0 + timedelta(seconds=90))
        assert "B" in cut.nodes

    def test_other_links_are_untouched(self):
        cg = ContactGraph(windows=[window("A", "B", 0, 180),
                                   window("C", "D", 0, 180)],
                          t0=T0, t1=T0 + timedelta(seconds=240))
        cut = cg.drop_node_after("B", T0 + timedelta(seconds=90))
        cd = [w for w in cut.windows if (w.node_a, w.node_b) == ("C", "D")]
        assert len(cd) == 1
        assert cd[0].duration_s == pytest.approx(180.0)

    def test_ephemeris_drop_removes_the_satellite_entirely(self, epoch):
        const = walker_delta(4, 2, 1, 53.0, 550.0, epoch,
                             max_epoch_age_days=2.0)
        eph = const.ephemeris(epoch, epoch + timedelta(minutes=20), 60.0)
        kept = eph.drop(["W00-00"])
        assert "W00-00" not in kept.sat_names
        with pytest.raises(KeyError):
            kept.index("W00-00")

    def test_loss_time_must_be_inside_the_horizon(self):
        cg = ContactGraph(windows=[window("A", "B", 0, 180)], t0=T0,
                          t1=T0 + timedelta(seconds=240))
        with pytest.raises(ValueError, match="outside the horizon"):
            cg.drop_node_after("B", T0 + timedelta(seconds=1000))


class TestTleEpochFarFromWindow:
    """The requirement: raise, never silently extrapolate."""

    @staticmethod
    def satellite(max_age=7.0):
        return Satellite.from_tle(TLE("TEST", LINE1, LINE2),
                                  max_epoch_age_days=max_age)

    def test_inside_the_window_propagates(self):
        sat = self.satellite()
        r, _ = sat.propagate([sat.epoch + timedelta(days=3)])
        assert np.all(np.isfinite(r))

    @pytest.mark.parametrize("days", [8.0, 30.0, 365.0, 9000.0])
    def test_forward_extrapolation_raises(self, days):
        sat = self.satellite()
        with pytest.raises(TleEpochError, match="element epoch"):
            sat.propagate([sat.epoch + timedelta(days=days)])

    @pytest.mark.parametrize("days", [-8.0, -30.0, -365.0])
    def test_backward_extrapolation_raises(self, days):
        sat = self.satellite()
        with pytest.raises(TleEpochError):
            sat.propagate([sat.epoch + timedelta(days=days)])

    def test_the_whole_requested_window_is_checked_not_just_its_start(self):
        # The window starts inside the limit and ends outside it.
        sat = self.satellite(max_age=7.0)
        times = [sat.epoch + timedelta(days=d) for d in (1.0, 3.0, 20.0)]
        with pytest.raises(TleEpochError):
            sat.propagate(times)

    def test_the_message_names_the_epoch_and_the_limit(self):
        sat = self.satellite()
        with pytest.raises(TleEpochError) as exc:
            sat.propagate([sat.epoch + timedelta(days=30)])
        text = str(exc.value)
        assert "max_epoch_age_days" in text
        assert sat.epoch.isoformat() in text
        assert "degrades silently" in text

    def test_constellation_ephemeris_raises_too(self, epoch):
        const = walker_delta(4, 2, 1, 53.0, 550.0, epoch,
                             max_epoch_age_days=1.0)
        with pytest.raises(TleEpochError):
            const.ephemeris(epoch + timedelta(days=20),
                            epoch + timedelta(days=20, hours=1), 60.0)

    def test_the_escape_hatch_is_explicit_and_works(self):
        sat = self.satellite()
        r, _ = sat.propagate([sat.epoch + timedelta(days=30)],
                             check_epoch=False)
        assert np.all(np.isfinite(r))

    def test_epoch_age_days_is_signed(self):
        sat = self.satellite()
        assert sat.epoch_age_days(sat.epoch + timedelta(days=2)) == pytest.approx(
            2.0, abs=1e-6)
        assert sat.epoch_age_days(sat.epoch - timedelta(days=2)) == pytest.approx(
            -2.0, abs=1e-6)

    def test_a_wider_window_can_be_requested_deliberately(self):
        sat = self.satellite(max_age=400.0)
        r, _ = sat.propagate([sat.epoch + timedelta(days=100)])
        assert np.all(np.isfinite(r))
