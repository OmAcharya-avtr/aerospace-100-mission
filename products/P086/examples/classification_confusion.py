"""Confusion matrices of the two rule baselines, side by side, on the held-out split."""

from __future__ import annotations

import numpy as np

from _common import bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from telemgap import benchmark, classify, dataset  # noqa: E402
from telemgap.learned import CLASS_NAMES  # noqa: E402


def main() -> int:
    style()
    generated = dataset.test_record(samples=12_000)
    record = generated.record
    truth = generated.value_truth.reshape(-1)
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    columns = [*CLASS_NAMES, "undecided"]
    for axis, mode in zip(axes, classify.MODES, strict=True):
        report = classify.classify(record, mode=mode)
        predicted = benchmark.rule_labels(record, report).reshape(-1)
        table = benchmark.score(truth, predicted, mode)
        matrix = table.confusion.astype(np.float64)
        normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
        image = axis.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{int(matrix[row, column])}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if normalised[row, column] > 0.5 else "#222222",
                )
        axis.set_xticks(range(len(columns)))
        axis.set_xticklabels(columns, rotation=16, ha="right")
        axis.set_yticks(range(len(CLASS_NAMES)))
        axis.set_yticklabels(CLASS_NAMES)
        axis.set_xlabel("rule verdict")
        axis.set_ylabel("ground truth")
        axis.grid(False)
        axis.set_title(
            f"{mode} baseline\n"
            f"corrupt R {table['corrupt'].recall:.4f}, surfaced "
            f"{table['corrupt'].surfaced_rate:.4f}; misparsed R "
            f"{table['misparsed'].recall:.4f}",
            fontsize=9,
        )
    figure.colorbar(image, ax=axes, shrink=0.85, label="row-normalised share")
    figure.suptitle(
        "telemgap.classify: the specified rule baseline and the strengthened one, "
        f"{truth.size} frame-channel rows of the held-out split",
        fontsize=10,
    )
    path = save(figure, "classification_confusion")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
