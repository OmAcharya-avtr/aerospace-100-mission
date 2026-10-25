"""The stratified coverage table, rendered so the miscalibration is visible.

What to notice: the marginal coverage of a conformally calibrated predictor
lands near the nominal level, and the cells do not. Split conformal prediction
promises marginal coverage for exchangeable data and promises nothing
conditional on a feature; the rows here are grouped by object, so the promise
it makes is the only one it keeps. A product that reported the marginal figure
alone would be hiding that.
"""

from __future__ import annotations

import numpy as np

from _common import bootstrap, is_fast, load_dataset, save, style

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402

from orbitdrift import run_benchmark  # noqa: E402
from orbitdrift.benchmark import run_extrapolation_benchmark  # noqa: E402

NOMINAL = 0.90
_ESTIMATORS = 40 if is_fast() else None
"""Boosting stages for the learned model: reduced in probe mode only."""
COLD = (
    "rule_of_thumb",
    "analytic_linear",
    "analytic_power_altitude",
    "analytic_power_altitude_conformal",
    "learned_gbr_quantile",
    "learned_gbr_conformal",
)
EXTRAPOLATION = (
    "own_curve_power",
    "own_curve_power_conformal",
    "own_curve_linear_conformal",
    "own_curve_drag_exponential_conformal",
    "learned_with_curve_features",
    "learned_with_curve_features_conformal",
)


def _grid(axis, result, names, title: str):
    strata = [cell.stratum for cell in result.coverage_by_altitude[names[0]]]
    strata += [cell.stratum for cell in result.coverage_by_age[names[0]]]
    matrix = np.empty((len(names), len(strata)))
    for row_index, name in enumerate(names):
        cells = list(result.coverage_by_altitude[name]) + list(result.coverage_by_age[name])
        for column_index, cell in enumerate(cells):
            matrix[row_index, column_index] = cell.coverage

    image = axis.imshow(
        matrix, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto", interpolation="nearest"
    )
    axis.set_xticks(range(len(strata)))
    axis.set_xticklabels(strata, rotation=45, ha="right", fontsize=6.8)
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(
        [name.replace("_", " ") for name in names], fontsize=6.8
    )
    for row_index, name in enumerate(names):
        cells = list(result.coverage_by_altitude[name]) + list(result.coverage_by_age[name])
        for column_index, cell in enumerate(cells):
            axis.text(
                column_index,
                row_index,
                f"{cell.coverage:.2f}\n{cell.verdict}",
                ha="center",
                va="center",
                fontsize=5.6,
                color="#111111",
            )
    axis.set_title(title, fontsize=9)
    return image


def main() -> None:
    table = load_dataset()
    cold = run_benchmark(
        table, nominal_coverage=NOMINAL, learned_estimators=_ESTIMATORS
    )
    extrapolation = run_extrapolation_benchmark(
        table, nominal_coverage=NOMINAL, learned_estimators=_ESTIMATORS
    )

    figure, axes = plt.subplots(2, 1, figsize=(9.5, 7.6))
    image = _grid(
        axes[0],
        cold,
        COLD,
        "cold prediction: coverage by altitude band then by epoch age",
    )
    _grid(
        axes[1],
        extrapolation,
        EXTRAPOLATION,
        "extrapolation: coverage by altitude band then by epoch age",
    )
    bar = figure.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    bar.set_label(f"measured coverage (nominal {NOMINAL:.2f})")

    figure.suptitle(
        "orbitdrift: measured interval coverage, stratified. "
        "'under' and 'over' mark cells inconsistent with the nominal level.",
        fontsize=10,
    )
    print(f"wrote {save(figure, 'coverage_table')}")
    for label, result, names in (
        ("cold", cold, COLD),
        ("extrapolation", extrapolation, EXTRAPOLATION),
    ):
        print(f"  {label}:")
        for name in names:
            cells = list(result.coverage_by_altitude[name]) + list(result.coverage_by_age[name])
            bad = sum(1 for cell in cells if cell.verdict in ("under", "over"))
            under = sum(1 for cell in cells if cell.verdict == "under")
            print(
                f"    {name:40s} marginal {result.row(name).metrics.coverage:.4f}, "
                f"{bad} of {len(cells)} cells inconsistent ({under} under)"
            )


if __name__ == "__main__":
    main()
