"""The curvature estimator's transfer function, and the channel it cannot see."""

from __future__ import annotations

import numpy as np

from _common import BOUND, CHANNEL_COLOURS, GRID, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from telemgap import dataset, policy  # noqa: E402

FREQUENCIES = np.logspace(-4, np.log10(0.49), 400)
HALF_WINDOWS = (3, 5, 7, 11)


def main() -> int:
    style()
    figure, axes = plt.subplots(
        2, 1, figsize=(9.0, 6.2), gridspec_kw={"height_ratios": [1.5, 1.0]}
    )
    for half_window in HALF_WINDOWS:
        transfer = [policy.estimator_transfer(half_window, value) for value in FREQUENCIES]
        bandwidth = policy.estimator_bandwidth_fraction(half_window)
        axes[0].semilogx(
            FREQUENCIES,
            transfer,
            linewidth=1.6 if half_window == policy.DEFAULT_HALF_WINDOW else 0.9,
            label=f"m = {half_window}, half power at fT = {bandwidth:.4f}",
        )
    axes[0].axhline(0.5, color=GRID, linewidth=0.8)
    for model in dataset.MODELS:
        for _amplitude, frequency, _phase in model.terms:
            axes[0].axvline(
                frequency * dataset.CADENCE_S,
                color=CHANNEL_COLOURS[model.name],
                linewidth=0.8,
                linestyle=":",
            )
    axes[0].set_xlabel("f T  [cycles per sample]")
    axes[0].set_ylabel("estimated curvature / true curvature")
    axes[0].set_ylim(-0.05, 1.1)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_title(
        "telemgap.policy equation (9): the quadratic-fit second-derivative estimator "
        "attenuates what it cannot resolve.\nDotted verticals are the spectral lines of "
        "the four generated channels."
    )

    rows = []
    for model in dataset.MODELS:
        estimate = policy.estimate_curvature(
            model.evaluate(dataset.CADENCE_S * np.arange(6000, dtype=np.float64)),
            dataset.CADENCE_S,
            span=4.0 * max(model.amplitude_bound, 1e-9),
            declared_sigma=1e-9,
        )
        rows.append((model.name, estimate.value, model.curvature_bound))
    positions = np.arange(len(rows))
    axes[1].bar(
        positions - 0.18,
        [row[2] for row in rows],
        width=0.34,
        color=BOUND,
        label="analytic, equation (12)",
    )
    axes[1].bar(
        positions + 0.18,
        [row[1] for row in rows],
        width=0.34,
        color=CHANNEL_COLOURS["battery_voltage"],
        label="estimated, equation (9), noiseless",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([row[0] for row in rows], rotation=12, ha="right", fontsize=8)
    axes[1].set_ylabel("curvature bound [unit / s^2]")
    axes[1].legend(fontsize=8)
    for position, (_name, estimated, analytic) in zip(positions, rows, strict=True):
        axes[1].text(
            position,
            max(estimated, analytic) * 1.6,
            f"{estimated / analytic:.3g}x",
            ha="center",
            fontsize=8,
        )
    axes[1].set_title(
        "estimated over analytic, on the noiseless channel: the near-Nyquist term of "
        "gyro_rate is invisible to the estimator",
        fontsize=9,
    )
    figure.subplots_adjust(hspace=0.55)
    path = save(figure, "curvature_estimator")
    print(f"written: {path}")
    for name, estimated, analytic in rows:
        print(f"  {name:18s} estimated {estimated:.6g}  analytic {analytic:.6g}  "
              f"ratio {estimated / analytic:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
