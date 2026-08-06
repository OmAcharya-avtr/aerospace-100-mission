"""Example 2: availability-weighted vs unweighted scheduling.

Two plans are built from the SAME candidate passes over an 8-station example
network (fictional placeholder sites):

* unweighted plan -- the optimizer maximises raw contact volume
  (rate x duration, i.e. p_clear assumed 1 for every pass);
* weighted plan   -- the optimizer maximises expected delivered data
  (rate x duration x p_clear from the climatological model).

Both plans are then SCORED with the availability model, which is the quantity
that actually matters operationally.  The setup (slew/acquisition) time is
swept to vary contention: with no contention the two plans coincide, and the
weighted plan pulls ahead as passes start to compete.

Writes ../screenshots/weighted_vs_unweighted.png
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from passplanner import (  # noqa: E402
    ClimatologyAvailability,
    expected_data,
    find_passes,
    load_stations,
    schedule_greedy,
    schedule_ilp,
)
from passplanner.fixtures import ISS_2008  # noqa: E402

T0 = datetime(2008, 9, 20, 12, 0, tzinfo=timezone.utc)
T1 = T0 + timedelta(days=1)
SETUP_TIMES_S = (0.0, 900.0, 1800.0, 2700.0, 3600.0, 5400.0)
DETAIL_SETUP_S = 3600.0


def main() -> int:
    stations = load_stations(HERE / "stations_network_example.yaml")
    availability = ClimatologyAvailability.from_stations(stations)
    passes = []
    for st in stations:
        passes.extend(find_passes(ISS_2008, st, T0, T1))
    passes.sort(key=lambda p: p.t_rise)

    weighted_scores, unweighted_scores, greedy_scores = [], [], []
    for setup in SETUP_TIMES_S:
        w_plan = schedule_ilp(passes, availability, setup_time_s=setup)
        u_plan = schedule_ilp(passes, None, setup_time_s=setup)
        g_plan = schedule_greedy(passes, availability, setup_time_s=setup)
        weighted_scores.append(w_plan.total_value)
        # Re-score the unweighted plan with the availability model.
        unweighted_scores.append(sum(expected_data(p, availability) for p in u_plan.selected))
        greedy_scores.append(g_plan.total_value)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    x = np.arange(len(SETUP_TIMES_S))
    width = 0.28
    ax1.bar(x - width, np.array(unweighted_scores) / 1000.0, width,
            label="unweighted plan (scored with availability)", color="#999999")
    ax1.bar(x, np.array(greedy_scores) / 1000.0, width,
            label="weighted greedy plan", color="#7fb3d5")
    ax1.bar(x + width, np.array(weighted_scores) / 1000.0, width,
            label="weighted ILP plan", color="#1f77b4")
    for i in range(len(x)):
        gain = 100.0 * (weighted_scores[i] - unweighted_scores[i]) / max(unweighted_scores[i], 1e-9)
        ax1.text(x[i], weighted_scores[i] / 1000.0 + 0.6, f"+{gain:.1f}%",
                 ha="center", fontsize=8, color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s / 60:.0f}" for s in SETUP_TIMES_S])
    ax1.set_xlabel("setup / slew time per contact [min]")
    ax1.set_ylabel("expected delivered data [Tbit]")
    ax1.set_title("Expected delivered data over 24 h\n"
                  f"ISS fixture TLE, {len(stations)} example stations, "
                  f"{len(passes)} candidate passes", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=8, loc="lower left")

    # Detail: which passes each plan picks at one setup time.
    w_plan = schedule_ilp(passes, availability, setup_time_s=DETAIL_SETUP_S)
    u_plan = schedule_ilp(passes, None, setup_time_s=DETAIL_SETUP_S)
    w_ids, u_ids = {id(p) for p in w_plan.selected}, {id(p) for p in u_plan.selected}
    p_clear = [availability.p_clear(p.station.name, p.t_culminate) for p in passes]
    dur_min = [p.duration_s / 60.0 for p in passes]
    both = [i for i, p in enumerate(passes) if id(p) in w_ids and id(p) in u_ids]
    only_w = [i for i, p in enumerate(passes) if id(p) in w_ids and id(p) not in u_ids]
    only_u = [i for i, p in enumerate(passes) if id(p) in u_ids and id(p) not in w_ids]
    neither = [i for i, p in enumerate(passes) if id(p) not in w_ids and id(p) not in u_ids]

    for idx, colour, marker, label in (
            (neither, "#cccccc", "o", "selected by neither"),
            (both, "#1f77b4", "o", "selected by both"),
            (only_u, "#d62728", "v", "unweighted plan only"),
            (only_w, "#2ca02c", "^", "weighted plan only")):
        if idx:
            ax2.scatter([dur_min[i] for i in idx], [p_clear[i] for i in idx],
                        c=colour, marker=marker, s=70, edgecolor="k", linewidth=0.4,
                        label=label, zorder=3 if colour != "#cccccc" else 2)
    ax2.set_xlabel("pass duration above mask [min]")
    ax2.set_ylabel("climatological clear-sky probability")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    ax2.set_title(f"Which passes each plan keeps (setup {DETAIL_SETUP_S / 60:.0f} min)\n"
                  "weighting trades long cloudy passes for shorter clear ones", fontsize=10)
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle("PassPlanner example 2 -- research-grade, not flight-qualified; "
                 "station statistics are fictional placeholders", fontsize=8)
    fig.tight_layout()
    out = HERE.parent / "screenshots" / "weighted_vs_unweighted.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)

    sys.stdout.write(f"candidate passes: {len(passes)}\n")
    sys.stdout.write(f"{'setup [min]':>12} {'unweighted [Gbit]':>19} {'weighted [Gbit]':>17} "
                     f"{'gain [%]':>9}\n")
    for s, u, wv in zip(SETUP_TIMES_S, unweighted_scores, weighted_scores):
        gain = 100.0 * (wv - u) / max(u, 1e-9)
        sys.stdout.write(f"{s / 60:>12.0f} {u:>19,.1f} {wv:>17,.1f} {gain:>9.2f}\n")
    sys.stdout.write(f"detail plan @ {DETAIL_SETUP_S / 60:.0f} min setup: "
                     f"weighted keeps {len(w_ids)} passes, unweighted keeps {len(u_ids)}, "
                     f"differing on {len(only_w) + len(only_u)}\n")
    sys.stdout.write(f"written: {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
