"""Per-class precision, recall and surfaced rate: rules, learned, majority guess."""

from __future__ import annotations

import numpy as np

from _common import LEARNED, MAJORITY, RULES, RULES_LIGHT, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from telemgap import benchmark, classify, dataset, features  # noqa: E402
from telemgap.learned import (  # noqa: E402
    CLASS_NAMES,
    CONFIGURATIONS,
    LearnedClassifier,
    MajorityClassifier,
)

SAMPLES = 12_000
METRICS = ("precision", "recall", "surfaced_rate")


def main() -> int:
    style()
    train = dataset.training_record(samples=SAMPLES)
    test = dataset.test_record(samples=SAMPLES)
    train_matrix, _ = features.build(train.record)
    train_labels = features.truth_labels(train.value_truth)
    test_matrix, _ = features.build(test.record)
    test_labels = features.truth_labels(test.value_truth)

    tables: dict[str, benchmark.ScoreTable] = {}
    for mode in classify.MODES:
        report = classify.classify(test.record, mode=mode)
        predicted = benchmark.rule_labels(test.record, report).reshape(-1)
        tables[f"rules {mode}"] = benchmark.score(test_labels, predicted, mode)
    model = LearnedClassifier(**CONFIGURATIONS["unweighted"]).fit(train_matrix, train_labels)
    tables["learned unweighted"] = benchmark.score(
        test_labels, model.predict(test_matrix).labels, "learned"
    )
    majority = MajorityClassifier().fit(train_labels)
    tables["majority guess"] = benchmark.score(
        test_labels, majority.predict(test_matrix).labels, "majority"
    )

    colours = {
        "rules specified": RULES_LIGHT,
        "rules strengthened": RULES,
        "learned unweighted": LEARNED,
        "majority guess": MAJORITY,
    }
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 4.4), sharey=True)
    width = 0.2
    for axis, metric in zip(axes, METRICS, strict=True):
        positions = np.arange(len(CLASS_NAMES))
        for offset, (label, table) in enumerate(tables.items()):
            values = [getattr(table[name], metric) for name in CLASS_NAMES]
            axis.bar(
                positions + (offset - 1.5) * width,
                [0.0 if not np.isfinite(value) else value for value in values],
                width=width,
                color=colours[label],
                label=label,
            )
            for position, value in zip(positions, values, strict=True):
                if not np.isfinite(value):
                    axis.text(
                        position + (offset - 1.5) * width,
                        0.02,
                        "n/a",
                        ha="center",
                        fontsize=6,
                        rotation=90,
                    )
        axis.set_xticks(positions)
        axis.set_xticklabels(CLASS_NAMES, rotation=12, ha="right")
        axis.set_ylim(0.0, 1.08)
        axis.set_xlabel(metric.replace("_", " "), fontsize=10)
    axes[0].set_ylabel("value")
    axes[0].legend(fontsize=7, loc="lower left")
    figure.suptitle(
        "telemgap.benchmark: the rules, the learned model and a majority-class guess on "
        f"{test_labels.size} held-out rows.\nThe rules abstain, which costs them recall "
        "and is counted in the surfaced rate; the majority guess finds no anomaly at all.",
        fontsize=10,
    )
    figure.subplots_adjust(wspace=0.12)
    path = save(figure, "rules_versus_learned")
    print(f"written: {path}")
    for label, table in tables.items():
        out = ", ".join(
            f"{name} P {table[name].precision:.4f} R {table[name].recall:.4f}"
            for name in CLASS_NAMES
        )
        print(f"  {label:<20s} {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
