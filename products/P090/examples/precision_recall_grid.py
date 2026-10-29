"""The rule baseline's per-class precision and recall over a swept corruption rate.

What to notice: recall is roughly flat across the grid and precision is not.
Precision moves because the class balance moves, and a screen's precision is a
property of the population it is run on, not of the screen. The loud class keeps
a recall near 1 because a physical bound violation is checked against a number
rather than estimated.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import CLASS_COLOURS, HISTORY_LENGTH, OBJECTS, bootstrap, save, style

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from tlecheck import benchmark, dataset  # noqa: E402
from tlecheck.consequence import CLASSES  # noqa: E402

RATES = (0.1, 0.25, 0.4, 0.5, 0.75, 1.0)


def main() -> None:
    style()
    precision: dict[str, list[float]] = {label: [] for label in CLASSES}
    recall: dict[str, list[float]] = {label: [] for label in CLASSES}
    for rate in RATES:
        records = dataset.split(
            "test", objects=OBJECTS, corruption_rate=rate, history_length=HISTORY_LENGTH
        )
        prepared = benchmark.prepare(records, silent_threshold_km=1.0)
        scores = benchmark.per_class(prepared.measured_classes, prepared.rule_classes)
        for label in CLASSES:
            precision[label].append(scores[label].precision)
            recall[label].append(scores[label].recall)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), sharey=True)
    for label in CLASSES:
        axes[0].plot(
            RATES,
            precision[label],
            marker="o",
            markersize=4,
            color=CLASS_COLOURS[label],
            label=label,
        )
        axes[1].plot(
            RATES,
            recall[label],
            marker="o",
            markersize=4,
            color=CLASS_COLOURS[label],
            label=label,
        )
    axes[0].set_title("precision against measured consequence", loc="left")
    axes[1].set_title("recall against measured consequence", loc="left")
    for axis in axes:
        axis.set_xlabel("fraction of objects corrupted")
        axis.set_ylim(0.0, 1.05)
    axes[0].set_ylabel("rate")
    axes[1].legend(fontsize=7.5, loc="lower right")
    figure.subplots_adjust(wspace=0.08)
    print(save(figure, "precision_recall_grid"))


if __name__ == "__main__":
    main()
