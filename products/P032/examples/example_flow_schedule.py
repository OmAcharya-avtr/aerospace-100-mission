"""Integer maximum-flow schedule: how much can be delivered, and where it goes.

Produces ``screenshots/flow_schedule.png``:

* top panel -- delivered volume against horizon length, for the integer
  maximum-flow schedule and for the single best route, so the gain from
  fragmenting a transfer across contacts is visible as a number;
* bottom panel -- per-slot delivered volume of the chosen schedule at the
  longest horizon, showing which contacts the optimiser actually used.

The optimiser is ``scipy.optimize.milp`` (HiGHS).  The same program is also
built as a ``pulp`` model for export, but ``pulp`` 4.0.0 ships no bundled
solver and none is installed in the environment this example was written in,
so the number here comes from HiGHS.  See ``validation/validate_ilp.py``.

Run: ``python examples/example_flow_schedule.py``
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
from constellink.flow import build_flow_network, ilp_max_flow  # noqa: E402
from constellink.graph import ContactGraph, TimeExpandedGraph  # noqa: E402
from constellink.routing import shortest_route  # noqa: E402
from constellink.synthdata import default_rf_terminal, default_stations  # noqa: E402

EPOCH = datetime(2026, 4, 1, tzinfo=UTC)
STEP_S = 60.0
SLOT_S = 60.0
SOURCE = "W00-00"
DEST = "AWARUA"
UNIT_BITS = 2.0e9        # flow quantum: 2 Gbit
HORIZONS_H = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
OUT = os.path.join(os.path.dirname(__file__), "..", "screenshots",
                   "flow_schedule.png")


def build(hours: float):
    const = walker_delta(24, 4, 1, 53.0, 550.0, EPOCH, max_epoch_age_days=2.0)
    stations = default_stations()
    t1 = EPOCH + timedelta(hours=hours)
    eph = const.ephemeris(EPOCH, t1, STEP_S)
    windows = all_contact_windows(eph, const.satellites, stations)
    nodes = [s.name for s in const.satellites] + [g.name for g in stations]
    cg = ContactGraph(windows=windows, t0=EPOCH, t1=t1, nodes=nodes)
    term = default_rf_terminal()
    teg = TimeExpandedGraph.from_contact_graph(
        cg, SLOT_S,
        rate_fn=lambda w, _t: rf_link(w.min_range_km, term).achievable_rate_bps,
        store_capacity_bits=50.0 * UNIT_BITS)
    return cg, teg


def main() -> int:
    flow_gbit = []
    route_gbit = []
    for hours in HORIZONS_H:
        _, teg = build(hours)
        res = ilp_max_flow(teg, SOURCE, DEST, flow_unit_bits=UNIT_BITS)
        flow_gbit.append(res.delivered_bits / 1e9)
        # The best single route carries at most one quantum per transmission
        # along its bottleneck edge; take the largest quantum the route
        # supports as the single-route comparison.
        best = 0.0
        route = shortest_route(teg, SOURCE, DEST)
        if route is not None:
            tx = [e.capacity_bits for e in route.hops if e.kind == "tx"]
            best = min(tx) / 1e9 if tx else 0.0
        route_gbit.append(best)
        print(f"horizon {hours:4.1f} h: max flow "
              f"{res.delivered_bits / 1e9:8.2f} Gbit "
              f"({res.flow_units} units of {UNIT_BITS / 1e9:g} Gbit), "
              f"best single route bottleneck {best:7.2f} Gbit, "
              f"backend {res.backend}")

    _, teg = build(HORIZONS_H[-1])
    res = ilp_max_flow(teg, SOURCE, DEST, flow_unit_bits=UNIT_BITS)
    # The edge list the solver indexed into is the time-expanded edge list plus
    # one sink edge per slot; rebuild it once to label the realised flow.
    edges, _, _ = build_flow_network(teg, SOURCE, DEST, 0, UNIT_BITS)
    per_slot = np.zeros(teg.n_slots + 1)
    for idx, units in res.edge_flow_units.items():
        edge = edges[idx]
        if edge.kind == "sink" and units:
            per_slot[edge.src[1]] += units * UNIT_BITS / 1e9

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.0, 7.4),
                                  constrained_layout=True)
    ax.plot(HORIZONS_H, flow_gbit, "o-", lw=1.6, color="tab:blue",
            label="integer maximum-flow schedule")
    ax.plot(HORIZONS_H, route_gbit, "s--", lw=1.4, color="tab:orange",
            label="best single route, bottleneck capacity")
    ax.set_xlabel("horizon [h]")
    ax.set_ylabel("delivered volume [Gbit]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"{SOURCE} to {DEST}, 24/4/1 Walker at 550 km, "
                 f"{UNIT_BITS / 1e9:g} Gbit quantum\n"
                 f"RF capacity per link from the Ka-band terminal assumptions; "
                 f"solver: {res.backend}")

    slots = np.arange(per_slot.size)
    ax2.bar(slots, per_slot, width=1.0, color="tab:blue")
    ax2.set_xlabel(f"arrival slot ({SLOT_S:g} s each, from epoch)")
    ax2.set_ylabel("delivered [Gbit]")
    ax2.grid(alpha=0.3, axis="y")
    used = int((per_slot > 0).sum())
    ax2.set_title(f"Where the {res.delivered_bits / 1e9:.0f} Gbit arrives at "
                  f"{HORIZONS_H[-1]:g} h: {used} of {per_slot.size} slots "
                  f"carry traffic")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)

    print("")
    print(f"slots carrying traffic at {HORIZONS_H[-1]:g} h: {used} of "
          f"{per_slot.size}")
    print(f"largest single-slot delivery : {per_slot.max():.2f} Gbit")
    print(f"written: {os.path.normpath(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
