"""How delivery time and overhead move with the outage duty cycle.

Produces ``screenshots/outage_sensitivity.png``: the median, 95th percentile
and maximum delivery time against the outage duty cycle, with the closed-form
extension drawn alongside, and the retransmission overhead on the second panel.
The point of the figure is the widening gap between the median and the 95th
percentile: the cost of an outage is not that the transfer is slower on
average, it is that the spread grows faster than the centre.
"""

from __future__ import annotations

from _common import ANALYTIC, MONTE_CARLO, TAIL, bootstrap, save, style

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from relfiledelivery import (  # noqa: E402
    FileTransferSpec,
    LinkConditions,
    OutageModel,
    TransferMode,
    extended_expected_delivery_time,
    run_ensemble,
)

RUNS = 800
SPEC = FileTransferSpec(120 * 512 - 7, 512, 12)
MEAN_DOWN_S = 1.0
DUTIES = np.array([0.98, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40])


def main() -> str:
    medians, q95s, maxima, extensions, overheads_median, overheads_q95 = ([] for _ in range(6))
    for duty in DUTIES:
        mean_up = MEAN_DOWN_S * duty / (1.0 - duty)
        link = LinkConditions(
            5.0e5,
            1.0e4,
            0.8,
            0.05,
            outage=OutageModel(mean_up_s=mean_up, mean_down_s=MEAN_DOWN_S),
        )
        distribution = run_ensemble(SPEC, link, TransferMode.DEFERRED_NAK, RUNS, 31337)
        medians.append(distribution.quantiles[0.50])
        q95s.append(distribution.quantiles[0.95])
        maxima.append(distribution.tail["max_s"])
        extensions.append(
            extended_expected_delivery_time(SPEC, link, TransferMode.DEFERRED_NAK)
        )
        overheads_median.append(float(np.median(distribution.overhead_ratios)))
        overheads_q95.append(float(np.quantile(distribution.overhead_ratios, 0.95)))

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))

    axes[0].plot(DUTIES, medians, "o-", color=MONTE_CARLO, label="measured q0.50")
    axes[0].plot(DUTIES, q95s, "s--", color=TAIL, label="measured q0.95")
    axes[0].plot(DUTIES, maxima, "^:", color=TAIL, alpha=0.7, label=f"maximum of {RUNS} runs")
    axes[0].plot(DUTIES, extensions, "d-", color=ANALYTIC, label="closed-form extension")
    axes[0].set_xlabel("outage duty cycle (fraction of time the link is up)")
    axes[0].set_ylabel("delivery time, s")
    axes[0].set_yscale("log")
    axes[0].invert_xaxis()
    axes[0].set_title("delivery time against outage duty cycle", fontsize=9)
    axes[0].legend(fontsize=7)

    spread = np.array(q95s) / np.array(medians)
    twin = axes[0].twinx()
    twin.grid(False)
    twin.plot(DUTIES, spread, color="#7a5195", linewidth=1.0, alpha=0.8)
    twin.set_ylabel("q0.95 / q0.50", color="#7a5195", fontsize=8)
    twin.tick_params(axis="y", labelcolor="#7a5195", labelsize=7)

    axes[1].plot(DUTIES, overheads_median, "o-", color=MONTE_CARLO, label="measured q0.50")
    axes[1].plot(DUTIES, overheads_q95, "s--", color=TAIL, label="measured q0.95")
    axes[1].axhline(
        SPEC.minimum_overhead_ratio,
        color=ANALYTIC,
        linewidth=1.2,
        label=f"lossless floor {SPEC.minimum_overhead_ratio:.4f}",
    )
    axes[1].set_xlabel("outage duty cycle")
    axes[1].set_ylabel("wire bytes / file bytes")
    axes[1].invert_xaxis()
    axes[1].set_title("retransmission overhead against outage duty cycle", fontsize=9)
    axes[1].legend(fontsize=7)

    figure.suptitle(
        f"Mean outage {MEAN_DOWN_S:.0f} s, {SPEC.pdu_count} PDUs, 5% residual loss, "
        f"{RUNS} runs per point",
        fontsize=10,
    )
    figure.subplots_adjust(wspace=0.42, top=0.86)
    return save(figure, "outage_sensitivity")


if __name__ == "__main__":
    print(main())
