"""Visibility windows of the declared network over twelve hours.

What to notice: the two antennas at the same high-latitude site see almost the
same passes, so a second dish there buys far less than its window count suggests
once the overlap is counted; and the equatorial antenna sees the near-polar
spacecraft rarely and briefly.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _common import STATION_COLOURS, bootstrap, hours, relative, save, style

bootstrap()

from contactsched.ephemeris import Scenario, default_fleet  # noqa: E402
from contactsched.station import default_network  # noqa: E402
from contactsched.visibility import all_windows, supply_seconds_by_station  # noqa: E402


def main() -> None:
    style()
    scenario = Scenario.from_calendar(2026, 4, 30, duration_s=43200.0)
    stations = default_network()
    fleet = default_fleet()
    windows = all_windows(scenario, stations, fleet, step_s=30.0, tolerance_s=0.25)
    supply = supply_seconds_by_station(windows)

    figure, (timeline, bars) = plt.subplots(
        2, 1, figsize=(10.5, 7.0), gridspec_kw={"height_ratios": [3, 1]}
    )

    labels = []
    for row, spacecraft in enumerate(fleet):
        labels.append(spacecraft.name)
        for index, station in enumerate(stations):
            for window in windows:
                if window.spacecraft != spacecraft.name or window.station != station.name:
                    continue
                offset = (index - (len(stations) - 1) / 2.0) * 0.17
                timeline.barh(
                    row + offset,
                    hours(window.duration_s),
                    left=hours(window.rise_s),
                    height=0.15,
                    color=STATION_COLOURS[index % len(STATION_COLOURS)],
                )
    timeline.set_yticks(range(len(fleet)))
    timeline.set_yticklabels(labels)
    timeline.set_xlabel("hours after 2026-04-30T00:00:00")
    timeline.set_xlim(0.0, hours(scenario.duration_s))
    timeline.set_title(
        f"Visibility windows, {len(windows)} in total, "
        f"{sum(w.duration_s for w in windows):.0f} s of visibility"
    )
    handles = [
        plt.Line2D([0], [0], color=STATION_COLOURS[i % len(STATION_COLOURS)], lw=5)
        for i in range(len(stations))
    ]
    timeline.legend(handles, [s.name for s in stations], ncol=len(stations), loc="upper center",
                    bbox_to_anchor=(0.5, -0.16))

    names = [s.name for s in stations]
    raw = [sum(w.duration_s for w in windows if w.station == name) for name in names]
    union = [supply[name] for name in names]
    positions = range(len(names))
    bars.bar([p - 0.18 for p in positions], raw, width=0.35, color="#b8cfe3",
             label="sum of window durations")
    bars.bar([p + 0.18 for p in positions], union, width=0.35, color=STATION_COLOURS[0],
             label="union coverage (usable supply)")
    bars.set_xticks(list(positions))
    bars.set_xticklabels(names, fontsize=8)
    bars.set_ylabel("seconds")
    bars.set_title("Overlapping visibility supplies an antenna only once")
    bars.legend(loc="upper right", fontsize=8)

    figure.tight_layout()
    path = save(figure, "contact_windows")
    print(f"windows        : {len(windows)}")
    for name, raw_seconds, union_seconds in zip(names, raw, union, strict=True):
        print(
            f"  {name:<14} sum {raw_seconds:9.1f} s  union {union_seconds:9.1f} s  "
            f"overlap {raw_seconds - union_seconds:8.1f} s"
        )
    print(f"figure written : {path}")
    print(f"repository root: {relative(path)}")


if __name__ == "__main__":
    main()
