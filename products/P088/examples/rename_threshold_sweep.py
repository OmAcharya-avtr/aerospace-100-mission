"""Rename false-match rate and recall against the acceptance threshold.

What to notice: the two curves cross nowhere useful below 0.80. Every
threshold under 0.80 admits false matches at one dictionary size or another,
and recall holds at 1.0 until 0.85, so the lowest zero-false-match threshold
is the one the library ships. The vertical line is that choice, and it was
computed from this curve rather than placed on it.
"""

from __future__ import annotations

from _common import GREEDY, OPTIMAL, SILENT, THRESHOLD, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tmtcdiff.changes import ChangeClass  # noqa: E402
from tmtcdiff.diff import DEFAULT_RENAME_THRESHOLD  # noqa: E402
from tmtcdiff.rename import STRATEGIES, match_renames  # noqa: E402
from tmtcdiff.synth import rename_pair  # noqa: E402

THRESHOLDS = np.round(np.arange(0.40, 0.96, 0.05), 2)
SIZES = (20, 50, 100, 200, 400)
SEEDS = tuple(4100 + 7 * index for index in range(12))


def main() -> None:
    style()
    pools = {}
    for size in SIZES:
        for seed in SEEDS:
            pair = rename_pair(size, seed, renames=max(1, size // 10), removals=2, additions=2)
            removed = [
                pair.old.parameter_map[name]
                for name in sorted(set(pair.old.parameter_map) - set(pair.new.parameter_map))
            ]
            added = [
                pair.new.parameter_map[name]
                for name in sorted(set(pair.new.parameter_map) - set(pair.old.parameter_map))
            ]
            truth = {c.name: c.new_name for c in pair.truth_for(ChangeClass.RENAMED)}
            pools[(size, seed)] = (truth, match_renames(removed, added, threshold=0.0))
    greedy_pools = {}
    for size in SIZES:
        for seed in SEEDS:
            pair = rename_pair(size, seed, renames=max(1, size // 10), removals=2, additions=2)
            removed = [
                pair.old.parameter_map[name]
                for name in sorted(set(pair.old.parameter_map) - set(pair.new.parameter_map))
            ]
            added = [
                pair.new.parameter_map[name]
                for name in sorted(set(pair.new.parameter_map) - set(pair.old.parameter_map))
            ]
            greedy_pools[(size, seed)] = match_renames(
                removed, added, threshold=0.0, strategy="greedy"
            )

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    shades = np.linspace(0.35, 1.0, len(SIZES))
    for size, shade in zip(SIZES, shades, strict=True):
        fmr = []
        recall = []
        for threshold in THRESHOLDS:
            accepted = 0
            correct = 0
            injected = 0
            for seed in SEEDS:
                truth, candidates = pools[(size, seed)]
                taken = [c for c in candidates if c.score >= threshold]
                accepted += len(taken)
                correct += sum(1 for c in taken if truth.get(c.old_name) == c.new_name)
                injected += len(truth)
            fmr.append((accepted - correct) / accepted if accepted else np.nan)
            recall.append(correct / injected if injected else np.nan)
        axes[0].plot(
            THRESHOLDS, fmr, marker="o", markersize=3.2, color=SILENT, alpha=shade,
            label=f"n={size}"
        )
        axes[1].plot(
            THRESHOLDS, recall, marker="o", markersize=3.2, color=OPTIMAL, alpha=shade,
            label=f"n={size}"
        )

    greedy_fmr = []
    for threshold in THRESHOLDS:
        accepted = 0
        correct = 0
        for size in SIZES:
            for seed in SEEDS:
                truth, _ = pools[(size, seed)]
                taken = [c for c in greedy_pools[(size, seed)] if c.score >= threshold]
                accepted += len(taken)
                correct += sum(1 for c in taken if truth.get(c.old_name) == c.new_name)
        greedy_fmr.append((accepted - correct) / accepted if accepted else np.nan)
    axes[0].plot(
        THRESHOLDS,
        greedy_fmr,
        linestyle="--",
        color=GREEDY,
        linewidth=1.4,
        label="greedy baseline, pooled",
    )

    for panel, title, ylabel in (
        (axes[0], "false-match rate by dictionary size", "false matches / accepted pairs"),
        (axes[1], "recall by dictionary size", "correct / injected renames"),
    ):
        panel.axvline(DEFAULT_RENAME_THRESHOLD, color=THRESHOLD, linewidth=1.1, linestyle=":")
        panel.text(
            DEFAULT_RENAME_THRESHOLD - 0.012,
            0.52,
            f"shipped default {DEFAULT_RENAME_THRESHOLD:.2f}",
            transform=panel.get_xaxis_transform(),
            rotation=90,
            fontsize=8,
            ha="right",
            va="center",
            color=THRESHOLD,
        )
        panel.set_xlabel("acceptance threshold on the combined similarity score")
        panel.set_ylabel(ylabel)
        panel.set_title(title, loc="left")
        panel.legend(fontsize=8, loc="lower left")
    axes[0].set_ylim(-0.01, 0.26)
    axes[1].set_ylim(0.6, 1.02)
    figure.suptitle(
        "tmtcdiff rename detection: the threshold was selected from this grid, "
        f"not chosen ({len(SIZES)} sizes x {len(SEEDS)} seeds x {len(THRESHOLDS)} thresholds)",
        x=0.01,
        ha="left",
    )
    figure.subplots_adjust(top=0.86, wspace=0.26)
    _ = STRATEGIES
    print(f"written: {save(figure, 'rename_threshold_sweep')}")


if __name__ == "__main__":
    main()
