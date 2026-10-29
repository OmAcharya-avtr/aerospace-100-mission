"""Per-class detection recall and precision, isolated against combined.

What to notice: every class is detected perfectly when it is the only change
in the pair, and three loud classes lose ground when every class is injected
at once. The cause is named in the figure: an unrelated removal and an
unrelated addition in the same pair can be paired as a rename.
"""

from __future__ import annotations

from _common import LOUD, OPTIMAL, SILENT, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tmtcdiff.changes import Severity  # noqa: E402
from tmtcdiff.diff import DiffConfig, diff_dictionaries  # noqa: E402
from tmtcdiff.metrics import per_class_metrics  # noqa: E402
from tmtcdiff.synth import (  # noqa: E402
    INJECTABLE,
    baseline_dictionary,
    change_pair,
    inject_changes,
)

SIZES = (30, 120)
SEEDS = tuple(5200 + 11 * index for index in range(10))


def main() -> None:
    style()
    config = DiffConfig()

    isolated_truth: list = []
    isolated_detected: list = []
    for change_class in INJECTABLE:
        for size in SIZES:
            for seed in SEEDS:
                old = baseline_dictionary(size, seed)
                pair = inject_changes(old, (change_class,), seed + 1)
                result = diff_dictionaries(pair.old, pair.new, config)
                isolated_truth.extend(pair.truth_keys)
                isolated_detected.extend((c.change_class, c.name) for c in result.changes)
    combined_truth: list = []
    combined_detected: list = []
    for size in SIZES:
        for seed in SEEDS:
            pair = change_pair(size, seed)
            result = diff_dictionaries(pair.old, pair.new, config)
            combined_truth.extend(pair.truth_keys)
            combined_detected.extend((c.change_class, c.name) for c in result.changes)

    isolated = per_class_metrics(isolated_truth, isolated_detected)
    combined = per_class_metrics(combined_truth, combined_detected)
    classes = sorted(INJECTABLE, key=lambda c: (c.severity.rank, c.value))
    positions = np.arange(len(classes))

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 6.4), sharey=True)
    for panel, (title, metrics) in zip(
        axes,
        (
            ("isolated: one injected class per pair", isolated),
            ("combined: every class in the same pair", combined),
        ),
        strict=True,
    ):
        recalls = [metrics[c].recall if c in metrics else np.nan for c in classes]
        precisions = [metrics[c].precision if c in metrics else np.nan for c in classes]
        colours = [
            SILENT
            if c.severity is Severity.SILENT
            else (LOUD if c.severity is Severity.LOUD else "#8a8a8a")
            for c in classes
        ]
        panel.barh(positions + 0.19, recalls, height=0.36, color=colours, label="recall")
        panel.barh(
            positions - 0.19,
            precisions,
            height=0.36,
            color="none",
            edgecolor=OPTIMAL,
            linewidth=1.3,
            label="precision",
        )
        panel.set_yticks(positions)
        panel.set_yticklabels([c.value for c in classes], fontsize=8)
        panel.set_xlim(0.0, 1.05)
        panel.set_xlabel("metric")
        panel.set_title(title, loc="left")
        panel.invert_yaxis()
        panel.legend(loc="lower left", fontsize=8)
    figure.suptitle(
        "tmtcdiff per-class detection, bar colour is the severity tier\n"
        "silent red, loud amber, advisory grey; precision is the open outline",
        x=0.01,
        ha="left",
    )
    figure.subplots_adjust(top=0.88, wspace=0.08)
    print(f"written: {save(figure, 'change_class_detection')}")


if __name__ == "__main__":
    main()
