"""The change-class taxonomy as a figure: 21 classes in three tiers of visibility.

What to notice: the silent tier is the largest of the three. A dictionary
upgrade has more ways to change meaning quietly than to break loudly, and the
quiet ones are the ones a text diff will not rank for you.
"""

from __future__ import annotations

from _common import ADVISORY, LOUD, SILENT, TIER_COLOURS, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from tmtcdiff.changes import (  # noqa: E402
    ADVISORY_CLASSES,
    CLASS_NOTES,
    LOUD_CLASSES,
    SILENT_CLASSES,
)


def main() -> None:
    style()
    figure, axes = plt.subplots(figsize=(10.5, 7.0))
    tiers = (
        ("silent: resolves, runs, means something else", SILENT_CLASSES, SILENT),
        ("loud: stops resolving, fails visibly", LOUD_CLASSES, LOUD),
        ("advisory: nothing existing changes", ADVISORY_CLASSES, ADVISORY),
    )
    row = 0
    labels: list[str] = []
    for title, classes, colour in tiers:
        axes.text(
            -0.02,
            -row - 0.4,
            title,
            fontsize=10,
            fontweight="bold",
            color=colour,
            va="center",
            ha="left",
            transform=axes.transData,
        )
        row += 1
        labels.append("")
        for change_class in classes:
            axes.barh(-row, 1.0, height=0.68, color=colour, alpha=0.18)
            axes.text(
                0.01,
                -row,
                change_class.value,
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color=colour,
            )
            axes.text(
                0.30,
                -row,
                CLASS_NOTES[change_class],
                va="center",
                ha="left",
                fontsize=8.2,
                color="#303030",
            )
            row += 1
            labels.append(change_class.value)
        row += 0.4

    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(-row, 0.2)
    axes.set_yticks([])
    axes.set_xticks([])
    axes.grid(False)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.set_title(
        "tmtcdiff change-class taxonomy, ordered by failure visibility\n"
        f"{len(SILENT_CLASSES)} silent, {len(LOUD_CLASSES)} loud, "
        f"{len(ADVISORY_CLASSES)} advisory; within-tier order is a declared convention",
        loc="left",
    )
    _ = TIER_COLOURS
    print(f"written: {save(figure, 'taxonomy_map')}")


if __name__ == "__main__":
    main()
