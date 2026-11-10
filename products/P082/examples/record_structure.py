"""The synthetic record: its seasonal and diurnal cycle and its spatial structure.

A reader has to be able to see what the record IS before trusting a number
measured on it. Writes ``screenshots/record_structure.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

from _common import (  # noqa: E402
    EMPIRICAL,
    FITTED,
    INDEPENDENT,
    NEUTRAL,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from sitediverse.climatology import estimate_climatology  # noqa: E402
from sitediverse.field import (  # noqa: E402
    EPOCHS_PER_DAY,
    EPOCHS_PER_YEAR,
    RECORD_SPECS,
    REGIME_COUNT,
    generate_record,
    true_correlation_at,
)
from sitediverse.geometry import along_cross_matrices, separation_matrix  # noqa: E402


def main() -> int:
    style()
    record = generate_record(RECORD_SPECS["anisotropic"])
    climatology = estimate_climatology(record.cloud)
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))

    day = np.arange(EPOCHS_PER_YEAR) / EPOCHS_PER_DAY
    truth = norm.cdf(record.threshold_true[:EPOCHS_PER_YEAR])
    estimated = climatology.rate_series()[:EPOCHS_PER_YEAR]
    for station, colour in ((0, EMPIRICAL), (8, INDEPENDENT)):
        axes[0].plot(
            day, truth[:, station], color=colour, linewidth=1.0,
            label=f"{record.site_names[station]} declared",
        )
        axes[0].plot(
            day, estimated[:, station], color=colour, linewidth=0.8,
            linestyle=":", label=f"{record.site_names[station]} estimated",
        )
    axes[0].set_xlabel("day of the declared year")
    axes[0].set_ylabel("marginal cloud probability")
    axes[0].set_title("equation (6): seasonal and diurnal")
    axes[0].legend(loc="upper right", fontsize=7)

    hour = np.arange(EPOCHS_PER_DAY) * 3
    for station, colour in ((0, EMPIRICAL), (8, INDEPENDENT)):
        series = record.cloud[:, station].reshape(-1, EPOCHS_PER_DAY).mean(axis=0)
        axes[1].plot(
            hour, series, color=colour, marker="o", linewidth=1.2,
            label=record.site_names[station],
        )
    axes[1].set_xlabel("hour of the declared day")
    axes[1].set_ylabel("observed cloud rate")
    axes[1].set_title("the diurnal cycle, as observed")
    axes[1].legend(loc="upper right")

    separation = separation_matrix(record.sites)
    along, cross = along_cross_matrices(record.sites)
    rows, cols = np.triu_indices(record.site_count, 1)
    grid = np.linspace(50.0, 7400.0, 400)
    for regime, style_name in zip(
        range(REGIME_COUNT), ("-", "--", "-.", ":"), strict=True
    ):
        truth_pairs = true_correlation_at(
            record.spec, regime, along[rows, cols], cross[rows, cols]
        )
        axes[2].scatter(
            separation[rows, cols], truth_pairs, s=5, color=NEUTRAL, alpha=0.3
        )
        axes[2].plot(
            grid,
            record.spec.nugget * np.exp(-grid / record.spec.along_km[regime]),
            color=FITTED,
            linestyle=style_name,
            linewidth=1.0,
            label=f"regime {regime}: along {record.spec.along_km[regime]:.0f} km",
        )
        axes[2].plot(
            grid,
            record.spec.nugget * np.exp(-grid / record.spec.cross_km[regime]),
            color=INDEPENDENT,
            linestyle=style_name,
            linewidth=1.0,
        )
    axes[2].set_xscale("log")
    axes[2].set_xlabel("great-circle separation, km")
    axes[2].set_ylabel("true latent correlation")
    axes[2].set_title("equation (4): anisotropic, four regimes")
    axes[2].legend(loc="upper right", fontsize=7)

    path = save(figure, "record_structure.png")
    print(f"written: {path}")
    print(f"  epochs                         : {record.epochs}")
    print(f"  declared marginal range        : "
          f"{float(norm.cdf(record.threshold_true).min()):.6f} to "
          f"{float(norm.cdf(record.threshold_true).max()):.6f}")
    print(f"  observed cloud rate range      : "
          f"{float(record.cloud.mean(axis=0).min()):.6f} to "
          f"{float(record.cloud.mean(axis=0).max()):.6f}")
    worst = float(
        np.max(np.abs(climatology.rate_series() - norm.cdf(record.threshold_true)))
    )
    print(f"  worst climatology error        : {worst:.6f}")
    print("  the grey points are the TRUE pair correlations; the lines are the")
    print("  along-axis and cross-axis limits of equation (4) for each regime, which")
    print("  bracket them because every pair's baseline lies between the two axes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
