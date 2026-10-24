"""The three transfer modes on one scenario, as quantiles and completion.

Produces ``screenshots/mode_comparison.png``: median and 95th-percentile
delivery time per mode on the left, completion fraction and retransmission
overhead on the right, for a continuous lossy link and for the same link with
correlated outages and contact windows.
"""

from __future__ import annotations

from _common import MODE_COLOURS, bootstrap, save, style

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from relfiledelivery import (  # noqa: E402
    ALL_MODES,
    FileTransferSpec,
    LinkConditions,
    OutageModel,
    run_ensemble,
    uniform_windows,
)

RUNS = 800
SPEC = FileTransferSpec(200 * 1024 + 37, 1024, 12)

SCENARIOS = {
    "continuous, 8% loss": LinkConditions(1.0e6, 1.0e4, 1.2, 0.08),
    "outages + 8 contacts": LinkConditions(
        1.0e6,
        1.0e4,
        1.2,
        0.08,
        outage=OutageModel(mean_up_s=6.0, mean_down_s=3.0),
        windows=uniform_windows(8, 8.0, 16.0),
    ),
}


def main() -> str:
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
    width = 0.34
    for row, (name, link) in enumerate(SCENARIOS.items()):
        distributions = {
            mode: run_ensemble(SPEC, link, mode, RUNS, 4242) for mode in ALL_MODES
        }
        positions = np.arange(len(ALL_MODES))
        time_axis, right_axis = axes[row]

        medians = [
            d.quantiles.get(0.50, 0.0) if d.completed else 0.0 for d in distributions.values()
        ]
        q95s = [
            d.quantiles.get(0.95, 0.0) if d.completed else 0.0 for d in distributions.values()
        ]
        colours = [MODE_COLOURS[mode.value] for mode in ALL_MODES]
        time_axis.bar(positions - width / 2, medians, width, color=colours, label="q0.50")
        time_axis.bar(
            positions + width / 2,
            q95s,
            width,
            color=colours,
            alpha=0.45,
            hatch="//",
            label="q0.95",
        )
        for index, median in enumerate(medians):
            if median == 0.0:
                time_axis.text(
                    index, 0.02 * max(q95s + [1.0]), "never\ncompleted", ha="center", fontsize=7
                )
        time_axis.set_xticks(positions)
        time_axis.set_xticklabels([mode.value for mode in ALL_MODES], fontsize=7)
        time_axis.set_ylabel("delivery time, s")
        time_axis.set_title(f"{name}: delivery time", fontsize=9)
        time_axis.legend(fontsize=7)

        completion = [d.completion_fraction for d in distributions.values()]
        overhead = [float(np.median(d.overhead_ratios)) for d in distributions.values()]
        right_axis.bar(positions - width / 2, completion, width, color=colours, label="completed")
        twin = right_axis.twinx()
        twin.grid(False)
        twin.bar(
            positions + width / 2,
            overhead,
            width,
            color=colours,
            alpha=0.45,
            hatch="..",
            label="overhead",
        )
        twin.set_ylabel("median overhead ratio")
        twin.set_ylim(1.0, max(overhead) * 1.12)
        right_axis.set_xticks(positions)
        right_axis.set_xticklabels([mode.value for mode in ALL_MODES], fontsize=7)
        right_axis.set_ylabel("completion fraction")
        right_axis.set_ylim(0.0, 1.08)
        right_axis.set_title(f"{name}: completion and overhead", fontsize=9)

    figure.suptitle(
        f"Three retransmission disciplines, {RUNS} runs each, "
        f"{SPEC.pdu_count} PDUs of {SPEC.pdu_payload_bytes} bytes",
        fontsize=10,
    )
    figure.subplots_adjust(hspace=0.45, wspace=0.38, top=0.91)
    return save(figure, "mode_comparison")


if __name__ == "__main__":
    print(main())
