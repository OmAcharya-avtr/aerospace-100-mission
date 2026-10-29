"""Impact classification of corpus references, against corpus coverage.

What to notice: the silently changed references are the largest actionable
group, and the undecidable group never goes to zero. A corpus that builds
mnemonics at run time has references no static tool can classify, and those
are reported rather than assumed harmless.
"""

from __future__ import annotations

from _common import IMPACT_COLOURS, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tmtcdiff.diff import DiffConfig, diff_dictionaries  # noqa: E402
from tmtcdiff.impact import ImpactClass, assess_impact  # noqa: E402
from tmtcdiff.synth import change_pair, synthetic_corpus  # noqa: E402

SEEDS = tuple(8500 + 19 * index for index in range(6))
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def main() -> None:
    style()
    config = DiffConfig()
    classes = sorted(ImpactClass, key=lambda c: c.rank)
    counts = {impact_class: [] for impact_class in classes}
    coverages = []
    for fraction in FRACTIONS:
        totals = {impact_class: 0 for impact_class in classes}
        coverage = 0.0
        for seed in SEEDS:
            pair = change_pair(80, seed)
            result = diff_dictionaries(pair.old, pair.new, config)
            changed = [change.name for change in pair.truth]
            others = [
                name for name in pair.old.item_names if name not in changed
            ]
            take = int(round(fraction * len(others)))
            corpus = synthetic_corpus(
                pair.old, seed + 3, names=[*changed, *others[:take]], include_unreadable=True
            )
            report = assess_impact(result, corpus, pair.old, pair.new)
            for impact_class, count in report.class_counts.items():
                totals[impact_class] += count
            coverage += report.verdict.coverage_fraction
        for impact_class in classes:
            counts[impact_class].append(totals[impact_class] / len(SEEDS))
        coverages.append(coverage / len(SEEDS))

    positions = np.arange(len(FRACTIONS))
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    bottom = np.zeros(len(FRACTIONS))
    for impact_class in classes:
        values = np.array(counts[impact_class], dtype=float)
        axes[0].bar(
            positions,
            values,
            bottom=bottom,
            width=0.6,
            color=IMPACT_COLOURS[impact_class.value],
            label=impact_class.value,
        )
        bottom += values
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([f"{coverage:.2f}" for coverage in coverages])
    axes[0].set_xlabel("measured coverage of the dictionary by the corpus")
    axes[0].set_ylabel("references per corpus, mean over 6 seeds")
    axes[0].set_title("impact classes, ranked bottom to top", loc="left")
    axes[0].legend(fontsize=8, loc="upper left")

    actionable = np.array(
        [
            counts[ImpactClass.SILENTLY_CHANGED][index]
            + counts[ImpactClass.BROKEN][index]
            + counts[ImpactClass.RENAMED_AWAY][index]
            for index in range(len(FRACTIONS))
        ]
    )
    silent_share = np.array(counts[ImpactClass.SILENTLY_CHANGED]) / np.maximum(actionable, 1e-9)
    axes[1].plot(
        coverages,
        silent_share,
        marker="o",
        color=IMPACT_COLOURS["silently_changed"],
        label="silently changed share of actionable",
    )
    undecidable_share = np.array(counts[ImpactClass.UNDECIDABLE]) / np.maximum(
        sum(np.array(counts[c]) for c in classes), 1e-9
    )
    axes[1].plot(
        coverages,
        undecidable_share,
        marker="s",
        color=IMPACT_COLOURS["undecidable"],
        label="undecidable share of all references",
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("measured coverage of the dictionary by the corpus")
    axes[1].set_ylabel("fraction")
    axes[1].set_title("the two shares a reader should check first", loc="left")
    axes[1].legend(fontsize=8, loc="center right")
    figure.suptitle(
        "tmtcdiff impact report: silently changed references dominate the actionable set",
        x=0.01,
        ha="left",
    )
    figure.subplots_adjust(top=0.86, wspace=0.26)
    print(f"written: {save(figure, 'impact_breakdown')}")


if __name__ == "__main__":
    main()
