"""The learned delivery-time predictor against the closed form and a lookup table.

Produces two figures from one fit, because the fit is the expensive part:

``screenshots/predictor_benchmark.png``
    Median and 90th-percentile absolute relative error per predictor, split by
    regime, with the regime where the closed form is exact separated from the
    regime where it is an extension.
``screenshots/calibration_curve.png``
    Measured interval coverage against nominal, with the median interval width
    beside it, because coverage without a width is not a result.

The configuration matches ``validation/validate_predictors.py`` exactly, so the
figures and the committed validation numbers cannot drift apart.
"""

from __future__ import annotations

from _common import ANALYTIC, LEARNED, LOOKUP, TAIL, bootstrap, save, style

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from relfiledelivery import (  # noqa: E402
    build_predictors,
    calibration_table,
    fit_split,
    generate_table,
    run_benchmark,
)

TRAIN_ROWS_PER_REGIME = 1500
TEST_ROWS_PER_REGIME = 600
TRAIN_SEED = 100_000
TEST_SEED = 900_000
NOMINAL = 0.90
NOMINALS = (0.50, 0.80, 0.90)

COLOURS = {
    "analytic": ANALYTIC,
    "analytic_mean": "#d08c9b",
    "mean_lookup": LOOKUP,
    "learned_quantile": LEARNED,
}


def main() -> tuple[str, str]:
    train = fit_split(generate_table(TRAIN_ROWS_PER_REGIME, TRAIN_SEED))
    test = fit_split(generate_table(TEST_ROWS_PER_REGIME, TEST_SEED))
    predictors = build_predictors()
    rows = run_benchmark(predictors, train, test, nominal=NOMINAL)

    names = [predictor.name for predictor in predictors]
    regimes = ["no_outage", "correlated_outage"]
    by_key = {(row.predictor, row.regime): row for row in rows}

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    width = 0.2
    for axis, regime in zip(axes, regimes, strict=True):
        positions = np.arange(len(names))
        medians = [by_key[(name, regime)].median_abs_rel_error for name in names]
        p90s = [by_key[(name, regime)].p90_abs_rel_error for name in names]
        colours = [COLOURS[name] for name in names]
        axis.bar(positions - width / 2, medians, width * 2, color=colours, label="median")
        axis.bar(
            positions + width * 1.5,
            p90s,
            width * 2,
            color=colours,
            alpha=0.42,
            hatch="//",
            label="90th percentile",
        )
        for index, value in enumerate(medians):
            axis.text(index - width / 2, value * 1.05, f"{value:.3f}", ha="center", fontsize=7)
        axis.set_xticks(positions + width / 2)
        axis.set_xticklabels(names, fontsize=7, rotation=12)
        axis.set_yscale("log")
        axis.set_title(
            f"{regime}  ({by_key[(names[0], regime)].rows} held-out scenarios)", fontsize=9
        )
        axis.legend(fontsize=7)
    axes[0].set_ylabel("absolute relative error of the point prediction")
    figure.suptitle(
        "Delivery-time prediction: the closed form is exact in the left regime and an "
        "extension in the right one",
        fontsize=10,
    )
    figure.subplots_adjust(wspace=0.08, top=0.87)
    benchmark_path = save(figure, "predictor_benchmark")

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for predictor in predictors:
        table = calibration_table(predictor, test, NOMINALS)
        nominals = [row[0] for row in table]
        coverages = [row[1] for row in table]
        widths = [row[2] for row in table]
        axes[0].plot(
            nominals,
            coverages,
            "o-",
            color=COLOURS[predictor.name],
            label=predictor.name,
        )
        axes[1].plot(
            nominals, widths, "o-", color=COLOURS[predictor.name], label=predictor.name
        )
    axes[0].plot([0.4, 1.0], [0.4, 1.0], color=TAIL, linestyle="--", label="calibrated")
    axes[0].set_xlabel("nominal interval mass")
    axes[0].set_ylabel("measured coverage")
    axes[0].set_xlim(0.45, 0.95)
    axes[0].set_ylim(0.15, 1.0)
    axes[0].set_title("coverage against nominal: below the line is over-confident", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left")
    axes[1].set_xlabel("nominal interval mass")
    axes[1].set_ylabel("median (upper - lower) / observed")
    axes[1].set_yscale("log")
    axes[1].set_title("the price of that coverage", fontsize=9)
    axes[1].legend(fontsize=7, loc="upper left")
    figure.suptitle(
        f"Uncertainty output, {len(test)} held-out scenarios: coverage without a width is "
        "not a result",
        fontsize=10,
    )
    figure.subplots_adjust(wspace=0.32, top=0.86)
    calibration_path = save(figure, "calibration_curve")
    return benchmark_path, calibration_path


if __name__ == "__main__":
    for path in main():
        print(path)
