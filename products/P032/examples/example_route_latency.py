"""Minimum-latency routes over a time-expanded graph, as the release slot moves.

Produces ``screenshots/route_latency.png``:

* top panel -- end-to-end latency from one satellite to each of two ground
  stations as a function of the slot the message is released in, with the
  pure-propagation lower bound drawn for comparison;
* bottom panel -- hop count of the chosen route, showing where the router
  trades hops for waiting.

A gap in a curve is a release slot from which the destination cannot be
reached inside the remaining horizon.  That is a real answer, not missing
data, and it is left as a gap deliberately.

The two stations are GOLDSTONE (35.4 deg N) and AWARUA (46.5 deg S).  The
SVALBARD station in the built-in set is deliberately NOT used here: a 53 deg
inclination shell at 550 km has a coverage half-angle of about 15 deg, so it
never reaches 78 deg N and SVALBARD is unreachable for the whole horizon.
Running this example with ``SVALBARD`` in ``TARGETS`` prints zero reachable
release slots, which is the correct answer and a useful thing to see.

Run: ``python examples/example_route_latency.py``
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from constellink.capacity import rf_link  # noqa: E402
from constellink.constellation import walker_delta  # noqa: E402
from constellink.contacts import all_contact_windows  # noqa: E402
from constellink.graph import ContactGraph, TimeExpandedGraph  # noqa: E402
from constellink.routing import shortest_route  # noqa: E402
from constellink.synthdata import default_rf_terminal, default_stations  # noqa: E402

EPOCH = datetime(2026, 4, 1, tzinfo=UTC)
HOURS = 3.0
STEP_S = 60.0
SLOT_S = 60.0
SOURCE = "W00-00"
TARGETS = ("GOLDSTONE", "AWARUA")
MESSAGE_MBIT = 50.0
OUT = os.path.join(os.path.dirname(__file__), "..", "screenshots",
                   "route_latency.png")


def main() -> int:
    const = walker_delta(24, 4, 1, 53.0, 550.0, EPOCH, max_epoch_age_days=2.0)
    stations = default_stations()
    t1 = EPOCH + timedelta(hours=HOURS)
    eph = const.ephemeris(EPOCH, t1, STEP_S)
    windows = all_contact_windows(eph, const.satellites, stations)
    nodes = [s.name for s in const.satellites] + [g.name for g in stations]
    cg = ContactGraph(windows=windows, t0=EPOCH, t1=t1, nodes=nodes)

    term = default_rf_terminal()
    teg = TimeExpandedGraph.from_contact_graph(
        cg, SLOT_S,
        rate_fn=lambda w, _t: rf_link(w.min_range_km, term).achievable_rate_bps)

    msg_bits = MESSAGE_MBIT * 1e6
    slots = np.arange(0, teg.n_slots)
    results = {}
    for target in TARGETS:
        lat = np.full(slots.size, np.nan)
        hops = np.full(slots.size, np.nan)
        prop = np.full(slots.size, np.nan)
        for i, k in enumerate(slots):
            route = shortest_route(teg, SOURCE, target, release_slot=int(k),
                                   message_bits=msg_bits)
            if route is None:
                continue
            lat[i] = route.total_latency_s
            hops[i] = len(route.hops)
            prop[i] = route.total_latency_s - len(route.hops) * teg.slot_s
        results[target] = (lat, hops, prop)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.0, 7.6), sharex=True,
                                  constrained_layout=True)
    colours = {"GOLDSTONE": "tab:blue", "AWARUA": "tab:orange"}
    for target, (lat, hops, prop) in results.items():
        ax.plot(slots, lat, lw=1.5, color=colours[target],
                label=f"{SOURCE} to {target}")
        ax.plot(slots, prop, lw=1.0, ls=":", color=colours[target],
                label=f"{target}: propagation only")
        ax2.step(slots, hops, where="mid", lw=1.4, color=colours[target],
                 label=target)
    ax.set_ylabel("end-to-end latency [s]")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title(f"Minimum-latency route vs release slot, {const.name}\n"
                 f"{MESSAGE_MBIT:g} Mbit message, {SLOT_S:g} s slots, "
                 f"RF capacity from the Ka-band terminal assumptions\n"
                 f"gaps are release slots with no route inside the "
                 f"{HOURS:g} h horizon")
    ax2.set_xlabel("release slot (60 s each, from epoch 2026-04-01T00:00:00Z)")
    ax2.set_ylabel("hops in the chosen route")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)

    print(f"time-expanded graph : {teg.n_te_nodes} nodes, "
          f"{len(teg.edges)} edges, {teg.n_slots} slots")
    for target, (lat, hops, _) in results.items():
        ok = ~np.isnan(lat)
        print(f"{SOURCE} -> {target}:")
        print(f"  release slots with a route : {int(ok.sum())} of {slots.size}")
        if ok.any():
            print(f"  latency  min / median / max: {np.nanmin(lat):.2f} / "
                  f"{np.nanmedian(lat):.2f} / {np.nanmax(lat):.2f} s")
            print(f"  hops     min / max         : {int(np.nanmin(hops))} / "
                  f"{int(np.nanmax(hops))}")
    print(f"written             : {os.path.normpath(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
