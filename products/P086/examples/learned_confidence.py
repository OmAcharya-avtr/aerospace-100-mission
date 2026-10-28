"""Both uncertainty outputs on the ambiguous residual set, and the majority line."""

from __future__ import annotations

import numpy as np

from _common import GRID, LEARNED, MAJORITY, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from telemgap import benchmark, classify, dataset, features  # noqa: E402
from telemgap.learned import (  # noqa: E402
    CONFIGURATIONS,
    LearnedClassifier,
    MajorityClassifier,
    dispersion_error_auc,
    expected_calibration_error,
)

SAMPLES = 12_000
THRESHOLDS = np.linspace(0.34, 0.99, 30)


def ambiguous(generated: dataset.GeneratedRecord) -> np.ndarray:
    """Mask of the rows the strengthened rules left undecided."""
    report = classify.classify(generated.record)
    return benchmark.rule_labels(generated.record, report).reshape(-1) == benchmark.UNDECIDED


def main() -> int:
    style()
    train = dataset.training_record(samples=SAMPLES)
    calibration = dataset.calibration_record(samples=SAMPLES)
    train_matrix, _ = features.build(train.record)
    train_labels = features.truth_labels(train.value_truth)
    calibration_matrix, _ = features.build(calibration.record)
    calibration_labels = features.truth_labels(calibration.value_truth)
    train_mask = ambiguous(train)
    calibration_mask = ambiguous(calibration)

    model = LearnedClassifier(**CONFIGURATIONS["ambiguous_only"]).fit(
        train_matrix[train_mask], train_labels[train_mask]
    )
    majority = MajorityClassifier().fit(train_labels[train_mask])
    prediction = model.predict(calibration_matrix[calibration_mask])
    truth = calibration_labels[calibration_mask]
    correct = prediction.labels == truth
    ece, centres, confidence, accuracy = expected_calibration_error(
        prediction.probability, correct
    )
    auc = dispersion_error_auc(prediction.dispersion, ~correct)
    baseline = float((truth == majority.label).mean())

    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.8))
    finite = np.isfinite(confidence)
    axes[0].plot([0, 1], [0, 1], color=GRID, linewidth=1.0, label="perfect calibration")
    axes[0].plot(
        confidence[finite],
        accuracy[finite],
        color=LEARNED,
        marker="o",
        markersize=4,
        linewidth=1.1,
        label=f"measured, ECE {ece:.4f}",
    )
    axes[0].set_xlabel("mean vote share in bin")
    axes[0].set_ylabel("realised accuracy")
    axes[0].set_xlim(0.3, 1.02)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(fontsize=7, loc="lower right")
    axes[0].set_title("vote share against realised accuracy", fontsize=9)

    bins = np.linspace(0.0, max(float(prediction.dispersion.max()), 1e-6), 24)
    axes[1].hist(
        prediction.dispersion[correct],
        bins=bins,
        color=LEARNED,
        alpha=0.75,
        label=f"correct ({int(correct.sum())})",
    )
    axes[1].hist(
        prediction.dispersion[~correct],
        bins=bins,
        color=MAJORITY,
        alpha=0.75,
        label=f"wrong ({int((~correct).sum())})",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("per-tree dispersion of the chosen class")
    axes[1].set_ylabel("rows")
    axes[1].legend(fontsize=7)
    axes[1].set_title(f"dispersion, AUC against wrong {auc:.4f}", fontsize=9)

    accuracies = []
    rates = []
    for threshold in THRESHOLDS:
        gated = model.predict(
            calibration_matrix[calibration_mask], abstain_threshold=float(threshold)
        )
        decided = gated.labels != -1
        rates.append(float(1.0 - decided.mean()))
        accuracies.append(
            float((gated.labels[decided] == truth[decided]).mean()) if decided.any() else np.nan
        )
    axes[2].plot(rates, accuracies, color=LEARNED, marker="o", markersize=3, linewidth=1.1)
    axes[2].axhline(
        baseline,
        color=MAJORITY,
        linestyle="--",
        linewidth=1.1,
        label=f"majority-class guess {baseline:.4f}",
    )
    axes[2].set_xlabel("abstention rate")
    axes[2].set_ylabel("accuracy on the rows it decided")
    axes[2].legend(fontsize=7, loc="lower right")
    axes[2].set_title("what abstention buys", fontsize=9)

    figure.suptitle(
        "telemgap.learned: the 'ambiguous_only' configuration on "
        f"{truth.size} rows of an independent third split.\nThe majority-class guess is "
        "the line the model has to beat to be worth anything on this population.",
        fontsize=10,
    )
    figure.subplots_adjust(wspace=0.33)
    path = save(figure, "learned_confidence")
    print(f"written: {path}")
    print(f"  rows {truth.size}  accuracy {float(correct.mean()):.6f}  majority {baseline:.6f}")
    print(f"  ECE {ece:.6f}  dispersion AUC {auc:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
