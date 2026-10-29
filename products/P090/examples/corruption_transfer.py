"""Where the rules win: a corruption family the learned model has not seen.

What to notice: the learned model loses to the rule baseline on two of the five
withheld corruption families, and recovers when the family is restored to its
training set. The drop is the unseen-family effect and nothing else. A real
corruption is not drawn from the distribution a model was trained on, and none
of the rule screens was fitted to a corruption distribution at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import LEARNED, MAJORITY, RULES, bootstrap, save, style

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from tlecheck import benchmark, dataset  # noqa: E402
from tlecheck.consequence import SILENTLY_WRONG  # noqa: E402
from tlecheck.learned import LearnedClassifier  # noqa: E402

CONFIGURATION = "elements_and_history"


def main() -> None:
    style()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
    from _bootstrap import prepare_splits

    splits = prepare_splits()
    train, test = splits["train"], splits["test"]
    positive = test.measured_classes == SILENTLY_WRONG
    train_corruption = np.array([record.corruption for record in train.records])
    test_corruption = np.array([record.corruption for record in test.records])
    full = LearnedClassifier().fit(train.matrices[CONFIGURATION], train.measured_classes)

    labels: list[str] = []
    rules: list[float] = []
    held: list[float] = []
    everything: list[float] = []
    for corruption in dataset.CORRUPTION_FAMILIES[1:]:
        keep = train_corruption != corruption
        take = (test_corruption == corruption) | (test_corruption == "none")
        subset = positive[take]
        if subset.sum() < 5 or (~subset).sum() < 5:
            continue
        model = LearnedClassifier().fit(
            train.matrices[CONFIGURATION][keep], train.measured_classes[keep]
        )

        def area(scores: np.ndarray, mask: np.ndarray = subset) -> float:
            try:
                return benchmark.area_under_curve(scores, mask)
            except ValueError:
                return float("nan")

        labels.append(corruption)
        rules.append(area(test.rule_scores[take]))
        held.append(area(model.predict(test.matrices[CONFIGURATION][take]).silent_probability))
        everything.append(
            area(full.predict(test.matrices[CONFIGURATION][take]).silent_probability)
        )

    figure, axis = plt.subplots(figsize=(9.6, 4.2))
    positions = np.arange(len(labels))
    width = 0.27
    axis.bar(positions - width, rules, width, color=RULES, label="rule baseline, equation (15)")
    axis.bar(positions, held, width, color=LEARNED, label="learned, family withheld from training")
    axis.bar(
        positions + width,
        everything,
        width,
        color="#8fbf8f",
        label="learned, family in training",
    )
    axis.axhline(0.5, color=MAJORITY, linewidth=1.2, linestyle=":")
    axis.text(-0.45, 0.52, "chance", color=MAJORITY, fontsize=7.5)
    for index in range(len(labels)):
        if rules[index] > held[index]:
            axis.annotate(
                f"rules win\n{rules[index] - held[index]:+.3f}",
                xy=(index - width / 2, max(rules[index], held[index]) + 0.02),
                ha="center",
                fontsize=7.0,
                color=RULES,
            )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=7.6, rotation=12, ha="right")
    axis.set_ylim(0.0, 1.12)
    axis.set_ylabel("area under the operating characteristic, silent class")
    axis.set_title(
        "Transfer to an unseen corruption family, test split", loc="left"
    )
    axis.legend(fontsize=7.2, loc="lower left")
    print(save(figure, "corruption_transfer"))


if __name__ == "__main__":
    main()
