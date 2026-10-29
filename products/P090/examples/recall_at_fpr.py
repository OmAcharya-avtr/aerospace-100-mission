"""The headline comparison: rules, three learned configurations, majority baseline.

What to notice: the learned curves sit above the rule curve on this corpus, which
is not what the product specification predicted and is published as measured.
The majority-class baseline is the diagonal, by construction: its score is a
constant. The vertical line is the declared false-positive rate at which the
headline recall is quoted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import LEARNED, MAJORITY, RULES, RULES_LIGHT, THRESHOLD, bootstrap, save, style

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from tlecheck import benchmark  # noqa: E402
from tlecheck.consequence import SILENTLY_WRONG  # noqa: E402
from tlecheck.learned import CONFIGURATIONS, LearnedClassifier, MajorityClassifier  # noqa: E402

TARGET_FPR = 0.05
LEARNED_COLOURS = {
    "elements_and_history": LEARNED,
    "elements_only": "#8fbf8f",
    "with_rule_score": "#1f6f3f",
}


def curve(scores: np.ndarray, positive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = benchmark.operating_characteristic(scores, positive)
    return (
        np.array([0.0] + [point.false_positive_rate for point in points] + [1.0]),
        np.array([0.0] + [point.recall for point in points] + [1.0]),
    )


def main() -> None:
    style()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
    from _bootstrap import prepare_splits

    splits = prepare_splits()
    train, test = splits["train"], splits["test"]
    positive = test.measured_classes == SILENTLY_WRONG

    figure, axes = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    axis = axes[0]
    fpr, tpr = curve(test.rule_scores, positive)
    axis.plot(fpr, tpr, color=RULES, linewidth=1.8, label="rules, equation (15) swept")
    point = benchmark.point_operating_point(test.rule_classes, positive)
    axis.plot(
        [point.false_positive_rate],
        [point.recall],
        marker="s",
        markersize=7,
        color=RULES,
        label="rules at their own threshold",
    )
    bars = []
    for name, configuration in CONFIGURATIONS.items():
        feature_configuration = str(configuration["feature_configuration"])
        model = LearnedClassifier(**configuration)
        model.fit(train.matrices[feature_configuration], train.measured_classes)
        prediction = model.predict(test.matrices[feature_configuration])
        learned_fpr, learned_tpr = curve(prediction.silent_probability, positive)
        axis.plot(
            learned_fpr,
            learned_tpr,
            color=LEARNED_COLOURS[name],
            linewidth=1.5,
            label=f"learned {name}",
        )
        operating = benchmark.recall_at_fpr(
            prediction.silent_probability, positive, TARGET_FPR
        )
        bars.append((f"learned\n{name}", operating.recall, LEARNED_COLOURS[name]))
    axis.plot([0.0, 1.0], [0.0, 1.0], color=MAJORITY, linewidth=1.2, linestyle=":",
              label="majority-class baseline (chance)")
    axis.axvline(TARGET_FPR, color=THRESHOLD, linewidth=1.2, linestyle="--")
    axis.text(TARGET_FPR + 0.01, 0.06, "declared 0.05", color=THRESHOLD, fontsize=7.5)
    axis.set_xlabel("false-positive rate on the silently-wrong class")
    axis.set_ylabel("recall on the silently-wrong class")
    axis.set_title("Operating characteristics, test split", loc="left")
    axis.legend(fontsize=6.8, loc="lower right")

    axis = axes[1]
    swept = benchmark.recall_at_fpr(test.rule_scores, positive, TARGET_FPR)
    majority = MajorityClassifier().fit(train.measured_classes)
    majority_point = benchmark.point_operating_point(
        majority.predict(len(test.records)).labels, positive
    )
    rows = [
        ("rules\nswept", swept.recall, RULES),
        ("rules\nown threshold", point.recall, RULES_LIGHT),
        *bars,
        ("majority\nbaseline", majority_point.recall, MAJORITY),
    ]
    axis.bar(
        np.arange(len(rows)),
        [value for _name, value, _colour in rows],
        color=[colour for _name, _value, colour in rows],
    )
    for index, (_name, value, _colour) in enumerate(rows):
        axis.text(index, value + 0.015, f"{value:.3f}", ha="center", fontsize=7.0)
    axis.set_xticks(np.arange(len(rows)))
    axis.set_xticklabels([name for name, _value, _colour in rows], fontsize=6.8)
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("recall at a false-positive rate of 0.05")
    axis.set_title("Recall at the declared operating point", loc="left")

    figure.subplots_adjust(wspace=0.28)
    print(save(figure, "recall_at_fpr"))


if __name__ == "__main__":
    main()
