"""Realised interpolation error against equation (3) and the independent Bernstein bound."""

from __future__ import annotations

import numpy as np

from _common import BOUND, CHANNEL_COLOURS, INDEPENDENT, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from telemgap import dataset, policy  # noqa: E402

SLOTS = np.arange(1, 25)


def main() -> int:
    style()
    grid = dataset.CADENCE_S * np.arange(6000, dtype=np.float64)
    figure, axes = plt.subplots(2, 2, figsize=(9.8, 6.4), sharex=True)
    for model, axis in zip(dataset.MODELS, axes.ravel(), strict=True):
        truth = model.evaluate(grid)
        measured = []
        taylor = []
        bernstein = []
        for slots in SLOTS:
            duration = (slots + 1) * dataset.CADENCE_S
            starts = np.linspace(0, grid.size - slots - 3, 200).astype(np.int64)
            worst = 0.0
            for start in starts:
                stop = start + slots + 1
                fraction = (grid[start : stop + 1] - grid[start]) / (grid[stop] - grid[start])
                chord = truth[start] + fraction * (truth[stop] - truth[start])
                worst = max(worst, float(np.max(np.abs(chord - truth[start : stop + 1]))))
            measured.append(worst)
            taylor.append(policy.linear_error_bound(model.curvature_bound, duration))
            bernstein.append(
                policy.linear_error_bound(
                    policy.bernstein_curvature(model.amplitude_bound, model.bandwidth_hz),
                    duration,
                )
            )
        axis.loglog(
            SLOTS,
            measured,
            color=CHANNEL_COLOURS[model.name],
            marker="o",
            markersize=3,
            linewidth=1.0,
            label="measured, noiseless truth",
        )
        axis.loglog(SLOTS, taylor, color=BOUND, linewidth=1.2, label="equation (3), analytic M2")
        axis.loglog(
            SLOTS,
            bernstein,
            color=INDEPENDENT,
            linewidth=1.0,
            linestyle="--",
            label="equation (3) with the Bernstein M2 of equation (8)",
        )
        axis.set_title(
            f"{model.name}: top frequency {model.bandwidth_hz:.4g} Hz, "
            f"analytic M2 {model.curvature_bound:.3g}/s^2",
            fontsize=9,
        )
        axis.set_ylabel(f"max |error| [{model.name.split('_')[-1]}]")
        axis.legend(fontsize=7, loc="upper left")
    for axis in axes[-1]:
        axis.set_xlabel("gap length [cadence intervals]")
    figure.suptitle(
        "telemgap.policy: equation (3) is a bound, not an estimate. The measured curve "
        "never crosses it,\nand the Bernstein bound of equation (8) -- which shares no "
        "mathematics with equation (3) -- brackets it from above.",
        fontsize=10,
    )
    figure.subplots_adjust(hspace=0.32, wspace=0.27)
    path = save(figure, "interpolation_error_vs_gap")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
