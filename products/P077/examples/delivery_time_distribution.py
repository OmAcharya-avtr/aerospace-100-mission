"""Where the mean lands on an intermittent link, and where the runs land.

Produces ``screenshots/delivery_time_distribution.png``: the measured
delivery-time distribution for one scenario per regime, with the quantiles
marked, the tail shaded, and the mean drawn as a line so that the gap between
the mean and the mass is visible rather than asserted.
"""

from __future__ import annotations

from _common import ANALYTIC, MONTE_CARLO, OUTAGE, TAIL, bootstrap, save, style

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from relfiledelivery import (  # noqa: E402
    FileTransferSpec,
    LinkConditions,
    OutageModel,
    TransferMode,
    run_ensemble,
    uniform_windows,
)

RUNS = 3000
SPEC = FileTransferSpec(120 * 512 - 7, 512, 12)

SCENARIOS = {
    "no outages, continuous contact": LinkConditions(5.0e5, 1.0e4, 0.8, 0.08),
    "correlated outages, continuous": LinkConditions(
        5.0e5, 1.0e4, 0.8, 0.05, outage=OutageModel(mean_up_s=1.5, mean_down_s=1.0)
    ),
    "correlated outages, six contacts": LinkConditions(
        5.0e5,
        1.0e4,
        0.8,
        0.05,
        outage=OutageModel(mean_up_s=1.5, mean_down_s=1.0),
        windows=uniform_windows(6, 4.0, 10.0),
    ),
}


def main() -> str:
    figure, axes = plt.subplots(3, 1, figsize=(7.4, 8.2))
    for axis, (name, link) in zip(axes, SCENARIOS.items(), strict=True):
        distribution = run_ensemble(SPEC, link, TransferMode.DEFERRED_NAK, RUNS, 606)
        times = distribution.delivery_times_s
        axis.hist(times, bins=90, color=MONTE_CARLO, alpha=0.85, label=f"{RUNS} runs")
        q95 = distribution.quantiles[0.95]
        axis.axvspan(
            q95,
            float(times.max()),
            color=OUTAGE,
            alpha=0.9,
            zorder=0,
            label="tail above q0.95",
        )
        for level, style_ in ((0.50, "-"), (0.95, "--")):
            axis.axvline(
                distribution.quantiles[level],
                color=TAIL,
                linestyle=style_,
                linewidth=1.3,
                label=f"q{level:.2f} = {distribution.quantiles[level]:.2f} s",
            )
        axis.axvline(
            distribution.mean_s,
            color=ANALYTIC,
            linewidth=2.0,
            label=f"mean = {distribution.mean_s:.2f} s",
        )
        within = float(np.mean(np.abs(times - distribution.mean_s) <= 0.05 * distribution.mean_s))
        axis.set_title(
            f"{name}  --  mean/median {distribution.mean_over_median:.3f}, "
            f"{within:.1%} of runs within 5% of the mean"
        )
        axis.set_ylabel("runs")
        axis.legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1].set_xlabel("delivery time, seconds")
    figure.suptitle(
        "Delivery time over an intermittent link is multimodal; the mean is not a location",
        fontsize=10,
    )
    figure.subplots_adjust(hspace=0.42, top=0.93)
    return save(figure, "delivery_time_distribution")


if __name__ == "__main__":
    print(main())
