"""Plot the geometry and rate profile of one contact, and the terms it loses.

The left panel is the pass: elevation and the achievable rate against time
from mask rise. The right panel is the cumulative delivered volume, with the
acquisition, re-synchronisation and idle intervals shaded, so the areas the
decomposition charges to each term are visible as areas.

Writes ``screenshots/pass_geometry.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()
_bootstrap.style()

import matplotlib.pyplot as plt  # noqa: E402

from passyield.baselines import cumulative_volume  # noqa: E402
from passyield.dataset import (  # noqa: E402
    events_of,
    load_passes,
    station_by_name,
    window_of,
)
from passyield.decompose import (  # noqa: E402
    decompose_pass,
    normalise_intervals,
    rate_at_offsets,
)
from passyield.geometry import satellite, topocentric_profile  # noqa: E402
from passyield.report import format_decomposition  # noqa: E402

#: The pass drawn. Chosen as a long, high-elevation contact with every
#: decomposition term non-zero, so the figure shows all four.
PASS_ID = 5


def main() -> int:
    """Draw the pass and print its decomposition."""
    print(f"fast path (PASSYIELD_FAST=1): {_bootstrap.FAST}")
    record = load_passes()[: _bootstrap.record_limit()]
    index = {row.pass_id: row for row in record}
    row = index.get(PASS_ID, record[0])
    satrec = satellite()
    station = station_by_name(row.station)
    window = window_of(row)
    events = events_of(row)
    item = decompose_pass(satrec, station, window, events)

    print(f"pass_id {row.pass_id}, station {row.station}")
    print(format_decomposition(item))

    offsets = np.linspace(0.0, window.duration_s, 601)
    elevation, _azimuth, _slant = topocentric_profile(satrec, station, window, offsets)
    rates, _clamp = rate_at_offsets(satrec, station, window, offsets)
    times, cumulative = cumulative_volume(satrec, station, window)
    acquisition, resync, idle, _truncated = normalise_intervals(
        window.duration_s, events.acquisition_s,
        events.resync_intervals_s, events.idle_intervals_s,
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))

    left = axes[0]
    left.plot(offsets, elevation, lw=1.6, color="#1f4e79", label="elevation")
    left.axhline(station.mask_deg, ls="--", lw=1.0, color="0.4")
    left.annotate(
        f"mask {station.mask_deg:g} deg", xy=(0.0, station.mask_deg),
        xytext=(6, 6), textcoords="offset points", fontsize=8, color="0.3",
    )
    left.set_xlabel("time from mask rise, s")
    left.set_ylabel("elevation, deg")
    twin = left.twinx()
    twin.plot(offsets, rates, lw=1.6, color="#b35806", label="achievable rate")
    twin.set_ylabel("achievable rate, Mbit/s")
    twin.grid(False)
    left.set_title(f"{row.station}: pass geometry and rate")
    handles = [
        plt.Line2D([], [], color="#1f4e79", lw=1.6),
        plt.Line2D([], [], color="#b35806", lw=1.6),
    ]
    left.legend(handles, ["elevation", "achievable rate"], loc="lower center", fontsize=8)

    right = axes[1]
    right.plot(times, cumulative, lw=1.7, color="#1f4e79")
    for intervals, colour, label in (
        (acquisition, "#d6604d", "acquisition"),
        (resync, "#9970ab", "re-sync"),
        (idle, "#5aae61", "idle"),
    ):
        for position, (start, stop) in enumerate(intervals):
            right.axvspan(
                start, stop, color=colour, alpha=0.35,
                label=label if position == 0 else None,
            )
    right.set_xlabel("time from mask rise, s")
    right.set_ylabel("cumulative volume, Mbit")
    right.set_title("volume charged to each term")
    right.legend(loc="upper left", fontsize=8)

    figure.suptitle(
        f"shortfall {item.shortfall_bits:.0f} Mbit of a booked "
        f"{item.nominal_bits:.0f} Mbit; residual {item.residual_bits:.2e} Mbit",
        fontsize=9,
    )
    figure.tight_layout()
    target = ROOT / _bootstrap.SCREENSHOTS / "pass_geometry.png"
    figure.savefig(target)
    plt.close(figure)
    print(f"wrote {_bootstrap.relative(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
