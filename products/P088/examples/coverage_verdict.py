"""What the tool cannot determine, as a function of the corpus it is given.

What to notice: the undetermined fraction does not go to zero as the corpus
grows. Run-time-constructed references stay undecidable however many files are
supplied, and a dictionary item no file mentions stays outside the assessment.
Both are returned by the library as part of the coverage verdict.
"""

from __future__ import annotations

from _common import SILENT, UNDETERMINED, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tmtcdiff.diff import DiffConfig, diff_dictionaries  # noqa: E402
from tmtcdiff.impact import assess_impact  # noqa: E402
from tmtcdiff.synth import change_pair, synthetic_corpus  # noqa: E402

SEEDS = tuple(8500 + 19 * index for index in range(6))
FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SIZE = 80


def main() -> None:
    style()
    config = DiffConfig()
    coverage = []
    undecidable = []
    unreferenced = []
    determinable = []
    reasons = []
    for fraction in FRACTIONS:
        values = np.zeros(5)
        for seed in SEEDS:
            pair = change_pair(SIZE, seed)
            result = diff_dictionaries(pair.old, pair.new, config)
            changed = [change.name for change in pair.truth]
            others = [name for name in pair.old.item_names if name not in changed]
            take = int(round(fraction * len(others)))
            corpus = synthetic_corpus(
                pair.old, seed + 3, names=[*changed, *others[:take]], include_unreadable=True
            )
            report = assess_impact(result, corpus, pair.old, pair.new)
            verdict = report.verdict
            values += (
                verdict.coverage_fraction,
                1.0 - verdict.decidable_fraction,
                verdict.items_unreferenced / max(1, verdict.dictionary_items),
                1.0 if verdict.determinable else 0.0,
                len(verdict.reasons),
            )
        values /= len(SEEDS)
        coverage.append(values[0])
        undecidable.append(values[1])
        unreferenced.append(values[2])
        determinable.append(values[3])
        reasons.append(values[4])

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    axes[0].plot(coverage, undecidable, marker="o", color=UNDETERMINED,
                 label="references constructed at run time")
    axes[0].plot(coverage, unreferenced, marker="s", color=SILENT,
                 label="dictionary items the corpus never mentions")
    axes[0].set_xlabel("measured coverage of the dictionary by the corpus")
    axes[0].set_ylabel("fraction")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("two things that stay undetermined", loc="left")
    axes[0].legend(fontsize=8)

    positions = np.arange(len(FRACTIONS))
    axes[1].bar(positions, reasons, width=0.55, color=UNDETERMINED)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f"{value:.2f}" for value in coverage])
    axes[1].set_xlabel("measured coverage of the dictionary by the corpus")
    axes[1].set_ylabel("reasons returned in the coverage verdict")
    axes[1].set_title(
        f"verdict reasons; fully determinable in {sum(determinable) / len(determinable):.2f} "
        "of cases",
        loc="left",
    )
    for position, value in zip(positions, reasons, strict=True):
        axes[1].text(position, value + 0.03, f"{value:.1f}", ha="center", fontsize=8)
    figure.suptitle(
        "tmtcdiff coverage verdict: the uncovered part of the question is a returned value",
        x=0.01,
        ha="left",
    )
    figure.subplots_adjust(top=0.86, wspace=0.26)
    print(f"written: {save(figure, 'coverage_verdict')}")


if __name__ == "__main__":
    main()
