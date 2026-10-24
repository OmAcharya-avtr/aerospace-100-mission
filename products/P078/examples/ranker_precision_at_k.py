"""Precision at k: the learned ranker against the two free orderings.

Four orderings on the same held-out cases. The specification predicted that the
irreversibility-flag ordering would match or beat the learned model; what this
figure shows is what was measured, with the paired bootstrap interval drawn so
a reader can see which margins are real.

The right-hand panel is the ablation. Four of the nine defect families edit the
time column, so inter-command spacing carries the signature of the injection
process rather than information about commanding. Withholding the three timing
features removes that advantage, and the ablated bar is the conservative number.
"""

from __future__ import annotations

import _common
from _common import FLAG, LEARNED, RANDOM, RESOURCE, save, style

_common.bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cmdguard.benchmark import assert_disjoint, ranker_comparison  # noqa: E402
from cmdguard.dataset import build_corpus  # noqa: E402

# The same seed ranges as validation/validate_ranker.py, so the figure and the
# committed transcript carry the same numbers rather than two nearby sets.
TRAIN_SEEDS = range(0, 70)
TEST_SEEDS = range(100_000, 100_035)
KS = (5, 10, 20)
TIMING_FEATURES = ("dt_prev_s", "dt_next_s", "neighbours_within_60s")
SERIES = (
    ("learned", "learned ranker", LEARNED),
    ("irreversibility_flag", "irreversibility-flag ordering", FLAG),
    ("resource_draw", "resource-draw ordering", RESOURCE),
    ("random_control", "random control", RANDOM),
)


def panel(axis, comparison, title: str) -> None:
    positions = np.arange(len(KS))
    width = 0.2
    for index, (key, label, colour) in enumerate(SERIES):
        values = [comparison.scores[(key, k)].mean_precision for k in KS]
        axis.bar(positions + (index - 1.5) * width, values, width=width,
                 color=colour, label=label)
        for position, value in zip(positions + (index - 1.5) * width, values, strict=True):
            axis.annotate(f"{value:.3f}", (position, value), ha="center", va="bottom",
                          fontsize=6.2, color=colour)
    axis.set_xticks(positions)
    axis.set_xticklabels([f"k = {k}" for k in KS])
    axis.set_ylabel("mean precision at k over held-out cases")
    axis.set_ylim(0.0, 0.34)
    axis.set_title(title, fontsize=9.5)


def main() -> None:
    style()
    n_train, n_test = assert_disjoint(TRAIN_SEEDS, TEST_SEEDS)
    train = build_corpus(TRAIN_SEEDS, include_clean=False)
    test = build_corpus(TEST_SEEDS, include_clean=False)
    full = ranker_comparison(train, test, ks=KS, random_seed=4711)
    ablated = ranker_comparison(
        train, test, ks=KS, drop_features=TIMING_FEATURES, random_seed=4711
    )

    figure = plt.figure(figsize=(12.0, 7.2))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.55, wspace=0.22)

    panel(figure.add_subplot(grid[0, 0]), full,
          "all twenty features")
    panel(figure.add_subplot(grid[0, 1]), ablated,
          "three timing-shape features withheld\n"
          f"({', '.join(TIMING_FEATURES)})")

    axis = figure.add_subplot(grid[1, :])
    labels: list[str] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    colours: list[str] = []
    for name, comparison in (("all features", full), ("timing withheld", ablated)):
        for k in KS:
            for key, colour in (("irreversibility_flag", FLAG), ("resource_draw", RESOURCE)):
                mean, low, high = comparison.paired_bootstrap("learned", key, k)
                short = "flag" if key == "irreversibility_flag" else "draw"
                labels.append(f"{name}\nk={k} vs {short}")
                means.append(mean)
                lows.append(mean - low)
                highs.append(high - mean)
                colours.append(colour)
    positions = np.arange(len(labels))
    axis.bar(positions, means, color=colours, width=0.6)
    axis.errorbar(positions, means, yerr=[lows, highs], fmt="none", ecolor="#333333",
                  capsize=3, linewidth=1.0)
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=7.0, rotation=30, ha="right")
    axis.set_ylabel("learned minus baseline")
    axis.set_title(
        "paired difference in precision at k, by case, with a percentile bootstrap "
        "95 per cent interval\n"
        "a bar whose interval crosses zero is not distinguished from its baseline on "
        "this corpus",
        fontsize=9.5,
    )

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=colour, label=label,
                   markersize=8)
        for _, label, colour in SERIES
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
                  bbox_to_anchor=(0.5, -0.02))
    figure.suptitle(
        f"risk ranking for review triage: {n_train} training seeds, {n_test} held-out "
        f"seeds, {len(test)} held-out cases\n"
        f"mean {full.mean_involved_per_case:.2f} steps involved in a violation out of "
        f"{full.mean_steps_per_case:.1f} steps per case",
        fontsize=11,
    )
    print(f"wrote {save(figure, 'ranker_precision_at_k')}")
    for k in KS:
        for key, _, _ in SERIES:
            print(f"k={k:<3d} {key:<22s} full {full.scores[(key, k)].mean_precision:.6f} "
                  f"ablated {ablated.scores[(key, k)].mean_precision:.6f}")
        print(f"k={k:<3d} best (all features)  : {full.best_at(k)}")
        print(f"k={k:<3d} best (timing withheld): {ablated.best_at(k)}")


if __name__ == "__main__":
    main()
