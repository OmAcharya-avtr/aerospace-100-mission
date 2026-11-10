"""The learned model's uncertainty output, and its measured coverage.

The interval is an 80 per cent quantile-regression band. Its in-sample coverage
is near nominal and its out-of-sample coverage is NOT, which is the point of
measuring it. Writes ``screenshots/learned_interval.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    EMPIRICAL,
    LEARNED,
    NEUTRAL,
    WARN,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from sitediverse.field import RECORD_SPECS, generate_record  # noqa: E402
from sitediverse.learned import NOMINAL_COVERAGE  # noqa: E402
from sitediverse.splits import CELL_NAMES  # noqa: E402
from sitediverse.study import run_study  # noqa: E402


def main() -> int:
    style()
    record = generate_record(RECORD_SPECS["anisotropic"])
    study = run_study(record, "anisotropic", margin_replicates=100, margin_block=64)
    table = study.table
    prediction = study.learned_prediction

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    cell = table.mask(1)
    order = np.argsort(table.separation_km()[cell])
    separation = table.separation_km()[cell][order]
    lower = np.minimum(prediction.lower, prediction.upper)[cell][order]
    upper = np.maximum(prediction.lower, prediction.upper)[cell][order]
    median = prediction.median[cell][order]
    target = table.target[cell][order]
    axes[0].fill_between(
        separation, lower, upper, color=LEARNED, alpha=0.2,
        label=f"{NOMINAL_COVERAGE:.0%} quantile band",
    )
    axes[0].plot(separation, median, color=LEARNED, linewidth=1.2, label="median")
    axes[0].scatter(
        separation, target, s=10, color=EMPIRICAL, marker="x",
        label="implied correlation, equation (11)",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("great-circle separation, km")
    axes[0].set_ylabel("latent correlation")
    axes[0].set_title("held-out SITES: the interval output")
    axes[0].legend(loc="upper right")

    positions = np.arange(len(CELL_NAMES))
    coverage = [
        float(np.mean(prediction.covers(table.target)[table.mask(code)]))
        for code in range(len(CELL_NAMES))
    ]
    axes[1].axhline(
        NOMINAL_COVERAGE, color=NEUTRAL, linestyle="--", linewidth=1.2,
        label=f"nominal {NOMINAL_COVERAGE:.2f}",
    )
    axes[1].bar(
        positions,
        coverage,
        0.55,
        color=[
            LEARNED if value >= NOMINAL_COVERAGE - 0.05 else WARN
            for value in coverage
        ],
    )
    for index, value in enumerate(coverage):
        axes[1].text(index, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(CELL_NAMES, rotation=12, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("measured coverage")
    axes[1].set_title("the interval is UNDER-covered out of sample")
    axes[1].legend(loc="lower left")

    path = save(figure, "learned_interval.png")
    print(f"written: {path}")
    print(f"  nominal coverage               : {NOMINAL_COVERAGE:.4f}")
    for code, name in enumerate(CELL_NAMES):
        mask = table.mask(code)
        print(f"  {name:16s} rows {int(mask.sum()):4d}  coverage "
              f"{coverage[code]:.4f}  median width "
              f"{float(np.median(prediction.interval_width[mask])):.4f}  "
              f"crossings {int(np.count_nonzero(prediction.crossed[mask]))}")
    print(f"  quantile crossings, all rows   : {prediction.crossing_count} of "
          f"{prediction.rows}")
    print(f"  medians clamped                : {prediction.clamp_count} of "
          f"{prediction.rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
