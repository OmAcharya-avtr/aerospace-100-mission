"""Isolation confusion matrices: classical GLR bank against the classifier.

Three panels: the two 8x8 confusion matrices, printed complete rather than
summarised, and the signature Gram matrix that explains the classical bank's
worst confusion.  Every count is annotated on the cell.

This runs a **reduced** campaign so that it finishes in about a minute.  The
published matrices come from ``validation/isolation_confusion.py``, which uses
three times the data.

    python examples/confusion_matrices.py
"""

from __future__ import annotations

import numpy as np
from _plotstyle import save

import matplotlib.pyplot as plt  # noqa: E402  (backend set in _plotstyle)

from fdiscope import (  # noqa: E402
    BenchmarkConfig,
    FaultClassifier,
    build_default_bank,
    class_labels,
    confusion_report,
    evaluate_isolation,
    harvest_training_rows,
    run_scenarios,
    sample_scenarios,
)

N_TRAIN = 80
N_TEST = 80
N_TREES = 100
SHORT = {
    "none": "none",
    "sensor_bias": "s-bias",
    "sensor_drift": "s-drift",
    "sensor_stuck": "s-stuck",
    "sensor_dropout": "s-drop",
    "actuator_loss_of_effectiveness": "a-LOE",
    "actuator_stuck": "a-stuck",
    "actuator_runaway": "a-run",
}


def draw_matrix(ax, matrix, labels, title):
    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    ax.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                ax.text(
                    j,
                    i,
                    str(int(matrix[i, j])),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if normalised[i, j] > 0.55 else "black",
                )


def main() -> None:
    cfg = BenchmarkConfig()
    bank = build_default_bank(cfg)
    train = sample_scenarios(N_TRAIN, 1000)
    test = sample_scenarios(N_TEST, 5000)
    train_runs = run_scenarios(train, cfg)
    test_runs = run_scenarios(test, cfg)
    x, y = harvest_training_rows(train, train_runs, cfg)
    clf = FaultClassifier(n_estimators=N_TREES, random_state=0).fit(x, y)

    outcomes = evaluate_isolation(test, test_runs, cfg, bank, clf)
    labels = class_labels()
    short = [SHORT[label] for label in labels]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    reports = {}
    for ax, (name, outcome) in zip(axes[:2], outcomes.items(), strict=True):
        report = confusion_report(outcome.truth, outcome.predicted, labels)
        reports[name] = report
        draw_matrix(ax, report.matrix, short, f"{name}: accuracy {report.accuracy:.3f}")

    ax = axes[2]
    gram = np.abs(bank.gram())
    fault_names = [SHORT[f.value] for f in bank.faults]
    ax.imshow(gram, cmap="Reds", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(fault_names)), fault_names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(fault_names)), fault_names, fontsize=7)
    ax.set_title("|cos| between fault signatures")
    ax.grid(False)
    for i in range(gram.shape[0]):
        for j in range(gram.shape[1]):
            ax.text(
                j,
                i,
                f"{gram[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if gram[i, j] > 0.55 else "black",
            )

    fig.tight_layout()
    path = save(fig, "confusion_matrices.png")
    print(f"saved {path}")
    print(f"reduced campaign: {N_TRAIN} training / {N_TEST} held-out scenarios, {N_TREES} trees")
    print("NOT the published numbers; see validation/isolation_confusion.py")
    for name, report in reports.items():
        print(f"\n=== {name}")
        print(report.to_text(15))
    off = gram - np.eye(gram.shape[0])
    i, j = np.unravel_index(int(np.argmax(off)), off.shape)
    print(
        f"\nworst signature pair: {bank.faults[i].value} and {bank.faults[j].value}, "
        f"|cos| = {gram[i, j]:.4f}"
    )


if __name__ == "__main__":
    main()
