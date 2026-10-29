"""Silent changes against loud ones, across generated dictionary pairs.

What to notice: the silent count is not a small correction to the loud count.
On these pairs the silent tier is the larger of the two, and every silent
change leaves a working procedure pointing at a different quantity.
"""

from __future__ import annotations

from _common import ADVISORY, LOUD, SILENT, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tmtcdiff.diff import DiffConfig, diff_dictionaries  # noqa: E402
from tmtcdiff.synth import change_pair  # noqa: E402

SIZES = (30, 60, 120, 240, 400)
SEEDS = tuple(6300 + 13 * index for index in range(8))


def main() -> None:
    style()
    config = DiffConfig()
    silent = []
    loud = []
    advisory = []
    resolving = []
    for size in SIZES:
        tier_counts = np.zeros(3)
        still_resolving = 0
        total_silent = 0
        for seed in SEEDS:
            pair = change_pair(size, seed)
            result = diff_dictionaries(pair.old, pair.new, config)
            tier_counts += (len(result.silent), len(result.loud), len(result.advisory))
            for change in result.silent:
                total_silent += 1
                if pair.new.resolve(change.name) is not None:
                    still_resolving += 1
        silent.append(tier_counts[0] / len(SEEDS))
        loud.append(tier_counts[1] / len(SEEDS))
        advisory.append(tier_counts[2] / len(SEEDS))
        resolving.append(still_resolving / total_silent if total_silent else np.nan)

    positions = np.arange(len(SIZES))
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    axes[0].bar(positions - 0.26, silent, width=0.25, color=SILENT, label="silent")
    axes[0].bar(positions, loud, width=0.25, color=LOUD, label="loud")
    axes[0].bar(positions + 0.26, advisory, width=0.25, color=ADVISORY, label="advisory")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([str(size) for size in SIZES])
    axes[0].set_xlabel("parameters in the dictionary")
    axes[0].set_ylabel("changes per pair, mean over 8 seeds")
    axes[0].set_title("changes by tier, all 21 classes injected once each", loc="left")
    axes[0].legend(fontsize=8)

    axes[1].bar(positions, resolving, width=0.5, color=SILENT)
    axes[1].axhline(1.0, color="#1a1a1a", linewidth=1.0, linestyle=":")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([str(size) for size in SIZES])
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_xlabel("parameters in the dictionary")
    axes[1].set_ylabel("fraction whose old name still resolves")
    axes[1].set_title(
        "why the tier is called silent: the reference still works", loc="left"
    )
    for position, value in zip(positions, resolving, strict=True):
        axes[1].text(position, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    figure.suptitle(
        "tmtcdiff: the silent tier is the larger one, and it does not announce itself",
        x=0.01,
        ha="left",
    )
    figure.subplots_adjust(top=0.86, wspace=0.26)
    print(f"written: {save(figure, 'silent_versus_loud')}")


if __name__ == "__main__":
    main()
