"""Plot the headline output: the shortfall as a distribution, not a point.

The left panel is the analytic shortfall law of one pass with its quantiles
marked, the direct Monte Carlo of the same laws over it, the accounted
shortfall of that pass as a line, and baseline 1's prediction of zero as
another. The right panel is the empirical law over the whole committed
campaign, which is what a scheduler would book against.

Writes ``screenshots/shortfall_distribution.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()
_bootstrap.style()

import matplotlib.pyplot as plt  # noqa: E402

from passyield.baselines import (  # noqa: E402
    accounted_yield,
    analytic_shortfall_distribution,
    booked_shortfall_mbit,
    fit_event_model,
    monte_carlo_shortfall_distribution,
)
from passyield.dataset import (  # noqa: E402
    events_of,
    load_passes,
    parse_intervals,
    station_by_name,
    window_of,
)
from passyield.decompose import decompose_pass  # noqa: E402
from passyield.geometry import satellite  # noqa: E402
from passyield.report import REPORTED_QUANTILES, format_distribution  # noqa: E402

#: The pass drawn in the left panel.
PASS_ID = 5

#: Monte Carlo draws for the cross-check. Sized for the compute budget.
DRAWS = 100000


def main() -> int:
    """Draw both panels and print the quantiles that go with them."""
    print(f"fast path (PASSYIELD_FAST=1): {_bootstrap.FAST}")
    record = load_passes()[: _bootstrap.record_limit()]
    index = {row.pass_id: row for row in record}
    row = index.get(PASS_ID, record[0])
    satrec = satellite()
    station = station_by_name(row.station)
    window = window_of(row)
    item = decompose_pass(satrec, station, window, events_of(row))

    model = fit_event_model(
        acquisition_s=[entry.acquisition_s for entry in record],
        outage_counts=[entry.outage_count for entry in record],
        window_durations_s=[entry.duration_s for entry in record],
        outage_durations_s=[
            duration for entry in record
            for _start, duration in parse_intervals(entry.resync_intervals)
        ],
        idle_totals_s=[entry.idle_total_s for entry in record],
    )
    analytic = analytic_shortfall_distribution(
        satrec, station, window, model, item.nominal_bits, item.capacity_bits
    )
    direct = monte_carlo_shortfall_distribution(
        satrec, station, window, model, item.nominal_bits, item.capacity_bits,
        seed=_bootstrap.SEED, draws=DRAWS,
    )
    print(format_distribution(analytic))
    print()
    print(format_distribution(direct))

    decompositions = [
        decompose_pass(
            satrec, station_by_name(entry.station), window_of(entry), events_of(entry)
        )
        for entry in record
    ]
    accounted = accounted_yield(
        decompositions, seed=_bootstrap.SEED,
        resamples=_bootstrap.BOOTSTRAP_RESAMPLES, confidence=_bootstrap.CONFIDENCE,
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))

    left = axes[0]
    left.plot(analytic.grid_mbit, analytic.mass, lw=1.5, color="#1f4e79",
              label="baseline 2, convolution")
    left.plot(direct.grid_mbit, direct.mass, lw=1.0, color="#b35806", alpha=0.85,
              label=f"monte carlo, {DRAWS} draws")
    for level in (0.05, 0.5, 0.95):
        left.axvline(analytic.quantile(level), ls=":", lw=0.9, color="0.45")
    left.axvline(item.shortfall_bits, lw=1.6, color="#1a9850",
                 label="accounted shortfall of this pass")
    left.axvline(booked_shortfall_mbit(), lw=1.6, color="#d73027",
                 label="baseline 1 prediction")
    left.set_xlabel("shortfall, Mbit")
    left.set_ylabel("probability mass per bin")
    left.set_title(f"{row.station}, pass {row.pass_id}: the output is a distribution")
    left.legend(fontsize=7.5, loc="upper right")

    right = axes[1]
    empirical = accounted.empirical_distribution
    right.plot(empirical.grid_mbit, np.cumsum(empirical.mass), lw=1.7, color="#1f4e79")
    for level in REPORTED_QUANTILES:
        value = empirical.quantile(level)
        right.plot([value], [level], marker="o", ms=3.2, color="#b35806")
        right.annotate(
            f"{level:.2f}", xy=(value, level), xytext=(5, -3),
            textcoords="offset points", fontsize=7, color="0.3",
        )
    right.axvline(booked_shortfall_mbit(), lw=1.6, color="#d73027",
                  label="baseline 1 prediction")
    right.set_xlabel("per-pass shortfall, Mbit")
    right.set_ylabel("cumulative probability")
    right.set_title(f"campaign law over {accounted.passes} passes (baseline 3)")
    right.legend(fontsize=7.5, loc="lower right")

    figure.suptitle(
        "a schedule can only be written against a random variable by booking a "
        "quantile of it",
        fontsize=9,
    )
    figure.tight_layout()
    target = ROOT / _bootstrap.SCREENSHOTS / "shortfall_distribution.png"
    figure.savefig(target)
    plt.close(figure)
    print(f"wrote {_bootstrap.relative(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
