"""Calibration of the crossing-time distribution, complete against censored.

What to notice: the diagonal is perfect calibration. Every model is off it, and
each is off it differently on the two halves of the population. The learned
forecaster is closest to the diagonal on complete trajectories and furthest
from it on censored ones, which is exactly what a complete-case fit does.
"""

from __future__ import annotations

import numpy as np

from _common import MODEL_COLOURS, add_src_to_path, save, style

add_src_to_path()

import matplotlib.pyplot as plt  # noqa: E402

from degradepredict.benchmark import run_head_to_head  # noqa: E402
from degradepredict.generate import DEFAULT_LIMIT, generate_population  # noqa: E402
from degradepredict.metrics import calibration_curve, ks_uniform, pit_values  # noqa: E402

MODELS = ("linear", "exponential", "powerlaw-fade", "learned-ipcw", "learned-complete-case")


def main() -> None:
    style()
    train = generate_population(1200, seed=20261016)
    test = generate_population(400, seed=20261017)
    truth = np.array([tr.remaining_true_days for tr in test])
    event = np.array([tr.event for tr in test])
    results = run_head_to_head(train, test, DEFAULT_LIMIT)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    for ax, mask, label in (
        (axes[0], event, f"complete ({int(event.sum())} units)"),
        (axes[1], ~event, f"right-censored ({int((~event).sum())} units)"),
    ):
        ax.plot([0, 1], [0, 1], color="#1a1a1a", lw=1.0, ls="--", label="calibrated")
        for name in MODELS:
            pit = pit_values(results[name]["run"].forecasts, truth)[mask]
            nominal, empirical = calibration_curve(pit, n_bins=20)
            ax.plot(nominal, empirical, color=MODEL_COLOURS[name], lw=1.3, marker=".",
                    ms=3, label=name)
        ax.set_xlabel("nominal probability level")
        ax.set_ylabel("empirical fraction of PIT values below")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=10)
    axes[0].legend(fontsize=7.5, loc="lower right")

    cx = axes[2]
    x = np.arange(len(MODELS))
    ks_c = [ks_uniform(pit_values(results[m]["run"].forecasts, truth)[event])["ks_statistic"]
            for m in MODELS]
    ks_r = [ks_uniform(pit_values(results[m]["run"].forecasts, truth)[~event])["ks_statistic"]
            for m in MODELS]
    cx.bar(x - 0.18, ks_c, 0.36, color="#1f6fb4", label="complete")
    cx.bar(x + 0.18, ks_r, 0.36, color="#d9700b", label="right-censored")
    floor = ks_uniform(np.random.default_rng(20261016).uniform(0, 1, int(event.sum())))
    cx.axhline(floor["ks_statistic"], color="#1a1a1a", lw=1.0, ls=":",
               label=f"sampling floor at n={int(event.sum())}")
    cx.set_xticks(x)
    cx.set_xticklabels(MODELS, rotation=22, ha="right", fontsize=7.5)
    cx.set_ylabel("Kolmogorov-Smirnov distance from uniform")
    cx.set_title("PIT uniformity: lower is better", fontsize=10)
    cx.legend(fontsize=7.5)
    fig.suptitle(
        "Calibration of the predicted crossing-time distribution, reported separately "
        "for complete and censored units",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    print(f"wrote {save(fig, 'calibration_curves.png')}")


if __name__ == "__main__":
    main()
