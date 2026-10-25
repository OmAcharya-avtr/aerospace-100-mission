"""The baseline-against-learned benchmark, plotted as coverage against width.

What to notice: the vertical dashed line is the nominal 90 per cent coverage.
A predictor to the LEFT of it under-covers, and narrowing an interval always
moves a point left and down, so a low score to the left of the line is bought
rather than earned. The only predictors on or to the right of the line on the
extrapolation panel are analytic.
"""

from __future__ import annotations

from _common import SERIES, bootstrap, is_fast, load_dataset, save, style

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402

from orbitdrift import run_benchmark  # noqa: E402
from orbitdrift.benchmark import run_extrapolation_benchmark  # noqa: E402

NOMINAL = 0.90
_ESTIMATORS = 40 if is_fast() else None
"""Boosting stages for the learned model: reduced in probe mode only."""


def _panel(axis, result, title: str) -> None:
    for row in result.rows:
        colour = SERIES[0] if row.kind == "analytic" else SERIES[1]
        marker = "o" if "conformal" in row.name else "s"
        axis.errorbar(
            row.metrics.coverage,
            row.metrics.median_log_width,
            xerr=[
                [row.metrics.coverage - row.metrics.coverage_ci[0]],
                [row.metrics.coverage_ci[1] - row.metrics.coverage],
            ],
            fmt=marker,
            markersize=6.0,
            color=colour,
            ecolor="#b0b0b0",
            linewidth=0.8,
            capsize=2.0,
        )
        axis.annotate(
            row.name.replace("analytic_", "").replace("own_curve_", "").replace("learned_", ""),
            (row.metrics.coverage, row.metrics.median_log_width),
            textcoords="offset points",
            xytext=(6, 3),
            fontsize=6.4,
            color="#333333",
        )
    axis.axvline(NOMINAL, color="#444444", linestyle="--", linewidth=1.0)
    axis.set_xlabel("measured coverage on held-out objects")
    axis.set_ylabel(r"median $\log(\mathrm{upper}/\mathrm{lower})$")
    axis.set_title(title)


def main() -> None:
    table = load_dataset()
    cold = run_benchmark(
        table, nominal_coverage=NOMINAL, learned_estimators=_ESTIMATORS
    )
    extrapolation = run_extrapolation_benchmark(
        table, nominal_coverage=NOMINAL, learned_estimators=_ESTIMATORS
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    _panel(axes[0], cold, "cold prediction from elements and epoch age")
    _panel(axes[1], extrapolation, "extrapolation beyond a two-day fitting horizon")

    handles = [
        plt.Line2D([], [], color=SERIES[0], marker="s", linestyle="", label="analytic"),
        plt.Line2D([], [], color=SERIES[1], marker="s", linestyle="", label="learned"),
        plt.Line2D([], [], color="#666666", marker="o", linestyle="", label="conformal"),
        plt.Line2D([], [], color="#666666", marker="s", linestyle="", label="nominal interval"),
        plt.Line2D([], [], color="#444444", linestyle="--", label=f"nominal {NOMINAL:.2f}"),
    ]
    axes[0].legend(handles=handles, fontsize=7.5, loc="upper left")

    figure.suptitle(
        "orbitdrift: coverage must be passed before width is comparable. "
        "Error bars are Clopper-Pearson.",
        fontsize=10,
    )
    figure.tight_layout()
    print(f"wrote {save(figure, 'predictor_benchmark')}")
    for label, result in (("cold", cold), ("extrapolation", extrapolation)):
        print(f"  {label}:")
        for line in result.verdict_line().splitlines():
            print(f"    {line}")
    print(f"  rows: {cold.n_test_rows} cold, {extrapolation.n_test_target_rows} extrapolation")
    print(f"  objects held out: {cold.n_test_objects}")


if __name__ == "__main__":
    main()
