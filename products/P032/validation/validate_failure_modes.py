"""Failure-mode behaviour (Level 3): what happens when the inputs are broken.

Each case states the required behaviour, exercises it, and reports whether the
library did that.  A failure mode that returns a plausible-looking number
instead of refusing is the defect this script exists to catch.

Cases
-----
1. Empty contact graph -- no windows at all.  Routing must return ``None``
   (no route) rather than raise, the graph must report itself empty, and the
   time-expanded unroll must still produce the hold-edge skeleton.
2. Partitioned constellation -- two groups with no link between them.  The
   component decomposition must find both, ``is_partitioned`` must be true,
   and routing across the partition must return ``None`` while routing within
   a component still succeeds.
3. Satellite lost mid-horizon -- a node stops participating at a stated time.
   Windows touching it must be truncated at the loss time, routes that
   depended on it after that time must disappear, and the node must remain in
   the node set so it is reported unreachable rather than unknown.
4. TLE epoch far from the requested window -- propagation MUST raise
   ``TleEpochError``, not silently extrapolate.  This is the case where a
   quiet wrong answer is most dangerous, because SGP4 returns a position with
   no error code however far from epoch it is asked.
5. Supporting input validation -- a sample of the argument checks, so that
   the failure modes above are not the only guarded paths.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from constellink.capacity import RfTerminal, rf_link, slant_path_attenuation_db  # noqa: E402
from constellink.constellation import (  # noqa: E402
    PropagationError,
    TleEpochError,
    walker_delta,
)
from constellink.contacts import ContactWindow  # noqa: E402
from constellink.graph import ContactGraph, TimeExpandedGraph  # noqa: E402
from constellink.queueing import mm1_mean_delay_s  # noqa: E402
from constellink.routing import reachable_nodes, shortest_route  # noqa: E402

T0 = datetime(2026, 4, 1, tzinfo=UTC)


def window(a: str, b: str, open_s: float, close_s: float) -> ContactWindow:
    """Synthetic window at 1000 km."""
    return ContactWindow(node_a=a, node_b=b, kind="isl",
                         t_open=T0 + timedelta(seconds=open_s),
                         t_close=T0 + timedelta(seconds=close_s),
                         min_range_km=1000.0, max_range_km=1000.0,
                         max_elevation_deg=None, grid_step_s=10.0)


def case_empty() -> bool:
    """Case 1."""
    print("Case 1 -- empty contact graph")
    cg = ContactGraph(windows=[], t0=T0, t1=T0 + timedelta(seconds=300),
                      nodes=["A", "B", "C"])
    teg = TimeExpandedGraph.from_contact_graph(cg, 60.0)
    route = shortest_route(teg, "A", "C")
    reach = reachable_nodes(teg, "A")
    comps = cg.components()
    checks = {
        "graph reports is_empty": cg.is_empty,
        "node set preserved": cg.nodes == ["A", "B", "C"],
        "hold-edge skeleton built": len(teg.edges) == 3 * 5,
        "routing returns None (not an exception)": route is None,
        "only the source is reachable": reach == {"A"},
        "every node is its own component": len(comps) == 3,
        "is_partitioned is True": cg.is_partitioned(),
    }
    for name, value in checks.items():
        print(f"  {name:<45}: {'PASS' if value else 'FAIL'}")
    ok = all(checks.values())
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    print("")
    return ok


def case_partition() -> bool:
    """Case 2."""
    print("Case 2 -- partitioned constellation")
    cg = ContactGraph(windows=[window("A", "B", 0, 120),
                               window("C", "D", 0, 120)],
                      t0=T0, t1=T0 + timedelta(seconds=240))
    teg = TimeExpandedGraph.from_contact_graph(cg, 60.0, default_rate_bps=1e6)
    comps = cg.components()
    across = shortest_route(teg, "A", "C")
    within = shortest_route(teg, "A", "B")
    reach = reachable_nodes(teg, "A")
    checks = {
        "two components found": len(comps) == 2,
        "components are {A,B} and {C,D}": sorted(map(tuple, comps))
                                          == [("A", "B"), ("C", "D")],
        "is_partitioned is True": cg.is_partitioned(),
        "route across the partition is None": across is None,
        "route within a component exists": within is not None,
        "reachable set is the component": reach == {"A", "B"},
        "unreachable set is the other component":
            set(cg.nodes) - reach == {"C", "D"},
    }
    for name, value in checks.items():
        print(f"  {name:<45}: {'PASS' if value else 'FAIL'}")
    ok = all(checks.values())
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    print("")
    return ok


def case_satellite_loss() -> bool:
    """Case 3."""
    print("Case 3 -- satellite lost mid-horizon")
    # A reaches C only through B; B is lost at t = 90 s, after the A-B contact
    # but before the B-C contact that the route needs.
    cg = ContactGraph(windows=[window("A", "B", 0, 60),
                               window("B", "C", 120, 180)],
                      t0=T0, t1=T0 + timedelta(seconds=240))
    teg0 = TimeExpandedGraph.from_contact_graph(cg, 60.0, default_rate_bps=1e6)
    before = shortest_route(teg0, "A", "C")
    lost = cg.drop_node_after("B", T0 + timedelta(seconds=90))
    teg1 = TimeExpandedGraph.from_contact_graph(lost, 60.0, default_rate_bps=1e6)
    after = shortest_route(teg1, "A", "C")
    truncated = [w for w in lost.windows if "B" in (w.node_a, w.node_b)]
    checks = {
        "route exists before the loss": before is not None,
        "route is gone after the loss": after is None,
        "node B stays in the node set": "B" in lost.nodes,
        "B-C window removed entirely": all(
            not (w.node_a == "B" and w.node_b == "C") for w in lost.windows),
        "A-B window survives (it closed before the loss)": any(
            w.node_a == "A" and w.node_b == "B" for w in lost.windows),
        "no surviving B window extends past the loss time": all(
            w.t_close <= T0 + timedelta(seconds=90) for w in truncated),
        "B unreachable after the loss":
            "B" in set(lost.nodes) - reachable_nodes(teg1, "C"),
    }
    for name, value in checks.items():
        print(f"  {name:<45}: {'PASS' if value else 'FAIL'}")
    # Truncation case: a window that straddles the loss time must be cut.
    cg2 = ContactGraph(windows=[window("A", "B", 0, 180)], t0=T0,
                       t1=T0 + timedelta(seconds=240))
    cut = cg2.drop_node_after("B", T0 + timedelta(seconds=90))
    straddle_ok = (len(cut.windows) == 1
                   and abs(cut.windows[0].duration_s - 90.0) < 1e-9)
    print(f"  {'straddling window truncated to 90 s':<45}: "
          f"{'PASS' if straddle_ok else 'FAIL'} "
          f"(measured {cut.windows[0].duration_s:.6f} s)")
    ok = all(checks.values()) and straddle_ok
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    print("")
    return ok


def case_tle_epoch() -> bool:
    """Case 4."""
    print("Case 4 -- TLE epoch far from the requested window (must RAISE)")
    const = walker_delta(4, 2, 1, 53.0, 550.0, T0, max_epoch_age_days=7.0)
    sat = const.satellites[0]
    results = {}
    # Inside the window: must succeed.
    try:
        sat.propagate([T0 + timedelta(days=3)])
        results["3 days from epoch (inside 7-day window) propagates"] = True
    except TleEpochError:
        results["3 days from epoch (inside 7-day window) propagates"] = False
    # Outside the window, both directions: must raise.
    for label, dt in (("+30 days raises TleEpochError", timedelta(days=30)),
                      ("-30 days raises TleEpochError", timedelta(days=-30)),
                      ("+365 days raises TleEpochError", timedelta(days=365))):
        try:
            sat.propagate([T0 + dt])
            results[label] = False
        except TleEpochError:
            results[label] = True
    # The constellation-level ephemeris must raise too, not just the satellite.
    try:
        const.ephemeris(T0 + timedelta(days=20), T0 + timedelta(days=20, hours=2),
                        60.0)
        results["Constellation.ephemeris raises for a far window"] = False
    except TleEpochError:
        results["Constellation.ephemeris raises for a far window"] = True
    # And the escape hatch must still be explicit and work.
    try:
        r, _ = const.satellites[0].propagate([T0 + timedelta(days=30)],
                                              check_epoch=False)
        results["check_epoch=False is an explicit, working escape hatch"] = (
            np.all(np.isfinite(r)))
    except (TleEpochError, PropagationError):
        results["check_epoch=False is an explicit, working escape hatch"] = False
    for name, value in results.items():
        print(f"  {name:<58}: {'PASS' if value else 'FAIL'}")
    # Show the message, because an actionable message is part of the requirement.
    try:
        sat.propagate([T0 + timedelta(days=30)])
    except TleEpochError as exc:
        print("  raised message:")
        for line in str(exc).split(". "):
            print(f"    {line.strip()}")
    ok = all(results.values())
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    print("")
    return ok


def case_input_validation() -> bool:
    """Case 5."""
    print("Case 5 -- a sample of input validation")
    cases = [
        ("negative RF frequency", ValueError,
         lambda: RfTerminal(tx_power_dbw=0.0, tx_gain_dbi=0.0,
                            rx_g_over_t_db_per_k=0.0, frequency_hz=-1.0,
                            bandwidth_hz=1.0, required_ebn0_db=0.0)),
        ("negative loss (losses are positive dB)", ValueError,
         lambda: RfTerminal(tx_power_dbw=0.0, tx_gain_dbi=0.0,
                            rx_g_over_t_db_per_k=0.0, frequency_hz=1e9,
                            bandwidth_hz=1.0, required_ebn0_db=0.0,
                            other_loss_db=-1.0)),
        ("zero range in rf_link", ValueError,
         lambda: rf_link(0.0, RfTerminal(
             tx_power_dbw=0.0, tx_gain_dbi=0.0, rx_g_over_t_db_per_k=0.0,
             frequency_hz=1e9, bandwidth_hz=1.0, required_ebn0_db=0.0))),
        ("cosecant law below its validity elevation", ValueError,
         lambda: slant_path_attenuation_db(1.0, 2.0)),
        ("unstable queue (rho >= 1)", ValueError,
         lambda: mm1_mean_delay_s(2.0e6, 1.0e6, 1000.0)),
        ("Walker n_total not a multiple of n_planes", ValueError,
         lambda: walker_delta(25, 4, 1, 53.0, 550.0, T0)),
        ("Walker phasing out of range", ValueError,
         lambda: walker_delta(24, 4, 9, 53.0, 550.0, T0)),
        ("unknown routing endpoint", KeyError,
         lambda: shortest_route(
             TimeExpandedGraph.from_contact_graph(
                 ContactGraph(windows=[window("A", "B", 0, 60)], t0=T0,
                              t1=T0 + timedelta(seconds=120)), 60.0),
             "A", "ZZZ")),
    ]
    ok = True
    for label, exc_type, fn in cases:
        try:
            fn()
            raised = "nothing"
            good = False
        except exc_type:
            raised = exc_type.__name__
            good = True
        except Exception as exc:  # noqa: BLE001 - we want the actual type here
            raised = type(exc).__name__
            good = False
        ok &= good
        print(f"  {label:<45}{raised:<14}{'PASS' if good else 'FAIL'}")
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    print("")
    return ok


def main() -> int:
    print("Failure-mode validation")
    print("=" * 80)
    print("")
    results = {
        "empty contact graph": case_empty(),
        "partitioned constellation": case_partition(),
        "satellite lost mid-horizon": case_satellite_loss(),
        "TLE epoch far from window": case_tle_epoch(),
        "input validation": case_input_validation(),
    }
    print("=" * 80)
    for name, value in results.items():
        print(f"{name:<32}: {'PASS' if value else 'FAIL'}")
    overall = all(results.values())
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
