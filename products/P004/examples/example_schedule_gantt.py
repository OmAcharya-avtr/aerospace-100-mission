"""Example 1: pass timeline and schedule Gantt chart.

Finds 24 h of ISS (historic 2008 fixture TLE) passes over the three example
stations, builds the ILP schedule with a 30-minute setup time, and plots

* top:    every candidate pass as a marker on a per-station timeline, filled
          by the climatological clear-sky probability, with the pass duration
          drawn as a whisker and selected passes outlined in red;
* bottom: the elevation profile of each selected pass vs time from rise.

Writes ../screenshots/pass_schedule_gantt.png
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
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
from passplanner.frames import datetime_to_jd, ecef_to_azel, teme_to_ecef  # noqa: E402

SETUP_TIME_S = 1800.0
T0 = datetime(2008, 9, 20, 12, 0, tzinfo=timezone.utc)
T1 = T0 + timedelta(days=1)


def elevation_profile(tle, station, p, n=120):
    """Sample the elevation [deg] across a pass; returns (times, elevations)."""
    sat = tle.to_satrec()
    times = [p.t_rise + timedelta(seconds=s)
             for s in np.linspace(0.0, p.duration_s, n)]
    els = []
    for t in times:
        jd, fr = datetime_to_jd(t)
        _err, r_teme, _v = sat.sgp4(jd, fr)
        r_ecef = teme_to_ecef(np.array(r_teme), jd, fr)
        els.append(ecef_to_azel(r_ecef, station.lat_deg, station.lon_deg, station.alt_km)[1])
    return times, els


def main() -> int:
    stations = load_stations(HERE / "stations_example.yaml")
    availability = ClimatologyAvailability.from_stations(stations)

    passes = []
    for st in stations:
        passes.extend(find_passes(ISS_2008, st, T0, T1))
    passes.sort(key=lambda p: p.t_rise)

    ilp = schedule_ilp(passes, availability, setup_time_s=SETUP_TIME_S)
    greedy = schedule_greedy(passes, availability, setup_time_s=SETUP_TIME_S)
    selected = set(id(p) for p in ilp.selected)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5),
                                   gridspec_kw={"height_ratios": [1.1, 1.0]})
    names = [s.name for s in stations]
    cmap = plt.get_cmap("YlGnBu")

    # Passes are only a few minutes long on a 24 h axis, so each is drawn as a
    # marker at culmination (filled by p_clear) with a duration whisker.
    for p in passes:
        row = names.index(p.station.name)
        pc = availability.p_clear(p.station.name, p.t_culminate)
        chosen = id(p) in selected
        ax1.plot([p.t_rise, p.t_set], [row, row], color="#555555", lw=1.2, zorder=2)
        ax1.scatter([p.t_culminate], [row], s=190, marker="o",
                    color=cmap(0.20 + 0.75 * pc),
                    edgecolor="#d62728" if chosen else "#999999",
                    linewidth=2.4 if chosen else 1.0, zorder=4 if chosen else 3)
        ax1.annotate(f"{p.max_elevation_deg:.0f}°\n{p.duration_s / 60:.1f} min",
                     (p.t_culminate, row), textcoords="offset points", xytext=(0, 16),
                     ha="center", fontsize=7, color="#333333")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, 1.0))
    cbar = fig.colorbar(sm, ax=ax1, pad=0.01, fraction=0.03)
    cbar.set_label("climatological clear-sky probability", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels([f"{n}\n(p_clear Sep = "
                         f"{availability.p_clear(n, T0):.2f})" for n in names], fontsize=8)
    ax1.set_xlim(T0, T1)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax1.set_xlabel("UTC (2008-09-20 12:00 .. 2008-09-21 12:00)")
    ax1.set_title(f"ISS candidate passes and ILP schedule "
                  f"(setup time {SETUP_TIME_S / 60:.0f} min)\n"
                  f"marker fill = clear-sky probability, red outline = selected; "
                  f"labels are peak elevation and duration",
                  fontsize=10)
    ax1.grid(axis="x", alpha=0.3)
    ax1.set_ylim(-0.8, len(names) - 0.2)
    ax1.invert_yaxis()

    # Elevation profiles are plotted against time-from-rise so the short arcs
    # are legible; the 20 deg station mask is the common baseline.
    for p in ilp.selected:
        times, els = elevation_profile(ISS_2008, p.station, p)
        mins = [(t - p.t_rise).total_seconds() / 60.0 for t in times]
        ax2.plot(mins, els, lw=1.8,
                 label=f"{p.station.name} {p.t_rise.strftime('%H:%M')} "
                       f"({expected_data(p, availability):.0f} Gbit)")
    ax2.axhline(20.0, color="k", ls="--", lw=1, label="20$^\\circ$ elevation mask")
    ax2.set_ylabel("elevation [deg]")
    ax2.set_xlabel("minutes from rise")
    ax2.set_ylim(0, 95)
    ax2.set_xlim(0, None)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7, ncol=2, loc="upper right")
    ax2.set_title(f"Elevation profiles of the selected contacts -- "
                  f"ILP total {ilp.total_value:,.0f} Gbit vs greedy "
                  f"{greedy.total_value:,.0f} Gbit", fontsize=10)

    fig.suptitle("PassPlanner example 1 -- research-grade, not flight-qualified; "
                 "stations are fictional placeholders", fontsize=8, y=0.995)
    fig.tight_layout()
    out = HERE.parent / "screenshots" / "pass_schedule_gantt.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)

    gap = 100.0 * (ilp.total_value - greedy.total_value) / ilp.total_value
    sys.stdout.write(
        f"candidate passes: {len(passes)}\n"
        f"ILP    : {len(ilp.selected)} contacts, {ilp.total_value:,.1f} Gbit\n"
        f"greedy : {len(greedy.selected)} contacts, {greedy.total_value:,.1f} Gbit "
        f"(gap {gap:.2f} %)\n"
        f"written: {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
