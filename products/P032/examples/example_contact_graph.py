"""Contact-graph connectivity of a 24-satellite Walker shell over three hours.

Produces ``screenshots/contact_graph.png``:

* top panel -- number of open inter-satellite and satellite-ground links per
  60 s slot, with the number of connected components on a second axis;
* bottom panel -- per-link occupancy raster, one row per satellite pair that
  is ever open, so the intermittency of inter-plane links is visible directly.

Run: ``python examples/example_contact_graph.py``
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

from constellink.constellation import walker_delta  # noqa: E402
from constellink.contacts import all_contact_windows  # noqa: E402
from constellink.graph import ContactGraph  # noqa: E402
from constellink.synthdata import default_stations  # noqa: E402

EPOCH = datetime(2026, 4, 1, tzinfo=UTC)
HOURS = 3.0
STEP_S = 60.0
SLOT_S = 60.0
OUT = os.path.join(os.path.dirname(__file__), "..", "screenshots",
                   "contact_graph.png")


def main() -> int:
    const = walker_delta(24, 4, 1, 53.0, 550.0, EPOCH, max_epoch_age_days=2.0)
    stations = default_stations()
    t1 = EPOCH + timedelta(hours=HOURS)
    eph = const.ephemeris(EPOCH, t1, STEP_S)
    windows = all_contact_windows(eph, const.satellites, stations)
    nodes = ([s.name for s in const.satellites] + [g.name for g in stations])
    cg = ContactGraph(windows=windows, t0=EPOCH, t1=t1, nodes=nodes)

    n_slots = int(cg.horizon_s // SLOT_S)
    t_min = np.arange(n_slots) * SLOT_S / 60.0
    n_isl = np.zeros(n_slots, dtype=int)
    n_gnd = np.zeros(n_slots, dtype=int)
    n_comp = np.zeros(n_slots, dtype=int)
    for k in range(n_slots):
        t_a = cg.t0 + timedelta(seconds=SLOT_S * k)
        t_b = t_a + timedelta(seconds=SLOT_S)
        got = cg.windows_in_slot(t_a, t_b)
        n_isl[k] = sum(1 for w, _ in got if w.kind == "isl")
        n_gnd[k] = sum(1 for w, _ in got if w.kind == "ground")
        n_comp[k] = len(cg.components(cg.adjacency_at(t_a, t_b)))

    links = sorted({(w.node_a, w.node_b) for w in windows})
    raster = np.zeros((len(links), n_slots))
    index = {link: i for i, link in enumerate(links)}
    for k in range(n_slots):
        t_a = cg.t0 + timedelta(seconds=SLOT_S * k)
        t_b = t_a + timedelta(seconds=SLOT_S)
        for w, _ in cg.windows_in_slot(t_a, t_b):
            raster[index[(w.node_a, w.node_b)], k] = (
                2.0 if w.kind == "ground" else 1.0)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10.5, 9.0), gridspec_kw={"height_ratios": [1, 2]},
        constrained_layout=True)
    ax.plot(t_min, n_isl, lw=1.4, color="tab:blue", label="open ISLs")
    ax.plot(t_min, n_gnd, lw=1.4, color="tab:orange", label="open ground links")
    ax.set_ylabel("open links per 60 s slot")
    ax.set_xlabel("minutes from epoch 2026-04-01T00:00:00Z")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    axr = ax.twinx()
    axr.plot(t_min, n_comp, lw=1.0, ls="--", color="tab:red",
             label="connected components")
    axr.set_ylabel("connected components (dashed)", color="tab:red")
    axr.tick_params(axis="y", colors="tab:red")
    axr.set_ylim(0, max(4, int(n_comp.max()) + 1))
    ax.set_title(f"{const.name}, 550 km, {len(stations)} ground stations, "
                 f"{HOURS:g} h horizon\n{len(windows)} contact windows over "
                 f"{len(links)} distinct links")

    ax2.imshow(raster, aspect="auto", interpolation="nearest",
               extent=(0.0, float(t_min[-1]), float(len(links)), 0.0),
               cmap="viridis", vmin=0.0, vmax=2.0)
    ax2.set_xlabel("minutes from epoch")
    ax2.set_ylabel("link index (sorted by endpoint names)")
    ax2.set_title("Per-link occupancy: dark = closed, mid = ISL open, "
                  "bright = ground link open")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)

    print(f"constellation      : {const.name}")
    print(f"horizon            : {HOURS:g} h, {STEP_S:g} s scan step")
    print(f"nodes              : {len(cg.nodes)}")
    print(f"contact windows    : {len(windows)} "
          f"({sum(1 for w in windows if w.kind == 'isl')} ISL, "
          f"{sum(1 for w in windows if w.kind == 'ground')} ground)")
    print(f"distinct links     : {len(links)}")
    print(f"open ISLs per slot : min {n_isl.min()}, median "
          f"{int(np.median(n_isl))}, max {n_isl.max()}")
    print(f"components per slot: min {n_comp.min()}, max {n_comp.max()}")
    print(f"union graph        : {len(cg.components())} component(s)")
    print(f"written            : {os.path.normpath(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
