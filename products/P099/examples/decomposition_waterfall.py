"""Plot the shortfall decomposition over the whole committed campaign.

The left panel is the campaign waterfall: what was booked, what the geometry
could carry, and what each of acquisition, re-synchronisation and idle took
away, ending at what the record accounts for. The right panel is the share of
the shortfall each term carries at each station, which is where a held-out
station's difficulty comes from.

Writes ``screenshots/decomposition_waterfall.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()
_bootstrap.style()

import matplotlib.pyplot as plt  # noqa: E402

from passyield.baselines import accounted_yield  # noqa: E402
from passyield.dataset import (  # noqa: E402
    STATIONS,
    events_of,
    load_passes,
    station_by_name,
    window_of,
)
from passyield.decompose import decompose_pass  # noqa: E402
from passyield.geometry import satellite  # noqa: E402
from passyield.report import format_accounted  # noqa: E402

#: The four terms, in the order the waterfall subtracts them.
TERMS = ("geometry", "acquisition", "resync", "idle")

#: Colours, one per term.
COLOURS = {
    "geometry": "#4393c3",
    "acquisition": "#d6604d",
    "resync": "#9970ab",
    "idle": "#5aae61",
}


def main() -> int:
    """Draw the campaign waterfall and the per-station shares."""
    print(f"fast path (PASSYIELD_FAST=1): {_bootstrap.FAST}")
    record = load_passes()[: _bootstrap.record_limit()]
    satrec = satellite()
    decompositions = [
        decompose_pass(
            satrec, station_by_name(row.station), window_of(row), events_of(row)
        )
        for row in record
    ]
    accounted = accounted_yield(
        decompositions, seed=_bootstrap.SEED,
        resamples=_bootstrap.BOOTSTRAP_RESAMPLES, confidence=_bootstrap.CONFIDENCE,
    )
    print(format_accounted(accounted))

    by_station: dict[str, dict[str, float]] = {}
    for station in STATIONS:
        rows = [
            item for item, row in zip(decompositions, record, strict=True)
            if row.station == station.name
        ]
        total = sum(item.shortfall_bits for item in rows)
        by_station[station.name] = {
            "geometry": sum(item.geometry_bits for item in rows) / total,
            "acquisition": sum(item.acquisition_bits for item in rows) / total,
            "resync": sum(item.resync_bits for item in rows) / total,
            "idle": sum(item.idle_bits for item in rows) / total,
        }

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))

    left = axes[0]
    scale = 1.0e-6
    booked = accounted.nominal_mbit * scale
    position = booked
    left.bar(0, booked, color="0.55", width=0.65)
    left.annotate(f"{booked:.1f}", xy=(0, booked), xytext=(0, 4),
                  textcoords="offset points", ha="center", fontsize=7.5)
    for index, term in enumerate(TERMS, start=1):
        height = accounted.shortfall_share[term] * accounted.shortfall_mbit * scale
        left.bar(index, -height, bottom=position, color=COLOURS[term], width=0.65)
        position -= height
        left.annotate(f"-{height:.1f}", xy=(index, position), xytext=(0, -11),
                      textcoords="offset points", ha="center", fontsize=7.5)
    realised = accounted.realised_mbit * scale
    left.bar(len(TERMS) + 1, realised, color="#1a9850", width=0.65)
    left.annotate(f"{realised:.1f}", xy=(len(TERMS) + 1, realised), xytext=(0, 4),
                  textcoords="offset points", ha="center", fontsize=7.5)
    left.set_xticks(range(len(TERMS) + 2))
    left.set_xticklabels(
        ["booked\n(baseline 1)", "geometry", "acquisition", "re-sync", "idle",
         "accounted\n(baseline 3)"],
        fontsize=7.5,
    )
    left.set_ylabel("campaign volume, Tbit")
    left.set_title(
        f"{accounted.passes} passes: booked {booked:.1f} against accounted {realised:.1f} Tbit"
    )

    right = axes[1]
    names = [station.name for station in STATIONS]
    bottom = np.zeros(len(names))
    for term in TERMS:
        values = np.asarray([by_station[name][term] for name in names])
        right.bar(names, values, bottom=bottom, color=COLOURS[term], label=term, width=0.72)
        bottom += values
    right.set_ylabel("share of the station's shortfall")
    right.set_title("where each station loses its yield")
    right.legend(fontsize=7.5, ncol=4, loc="upper center")
    right.set_ylim(0.0, 1.25)
    right.tick_params(axis="x", labelrotation=35, labelsize=7.5)

    figure.suptitle(
        f"residual of the identity: {100.0 * accounted.residual_share:.2e} per cent of the "
        "shortfall, reported and not assumed",
        fontsize=9,
    )
    figure.tight_layout()
    target = ROOT / _bootstrap.SCREENSHOTS / "decomposition_waterfall.png"
    figure.savefig(target)
    plt.close(figure)
    print(f"wrote {_bootstrap.relative(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
