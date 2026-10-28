"""What each withheld field costs, as a verdict-count matrix."""

from __future__ import annotations

import numpy as np

from _common import VERDICT_COLOURS, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from telemgap import classify, dataset  # noqa: E402
from telemgap.determinability import LIMIT_KEYS  # noqa: E402

SAMPLES = 12_000
VERDICTS = ("missing", "corrupt", "misparsed", "undecided")


def main() -> int:
    style()
    generated = dataset.test_record(samples=SAMPLES)
    record = generated.record
    variants = {
        "complete": record,
        "no validity": record.without_validity(),
        "no raw words": record.without_raw_words(),
        "neither": record.without_validity().without_raw_words(),
    }
    counts = {}
    determinable = {}
    for label, variant in variants.items():
        report = classify.classify(variant)
        counts[label] = report.counts
        determinable[label] = {
            key: report.determinability[key].determinable for key in LIMIT_KEYS
        }

    figure, axes = plt.subplots(
        1, 2, figsize=(10.2, 4.2), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )
    positions = np.arange(len(variants))
    bottom = np.zeros(len(variants))
    for verdict in VERDICTS:
        values = np.array([counts[label][verdict] for label in variants], dtype=np.float64)
        axes[0].bar(
            positions,
            values,
            bottom=bottom,
            color=VERDICT_COLOURS[verdict],
            width=0.6,
            label=verdict,
        )
        bottom += values
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(list(variants), rotation=12, ha="right")
    axes[0].set_ylabel("anomalies by verdict")
    axes[0].legend(fontsize=8)
    axes[0].set_title(
        "withholding a field moves verdicts into 'undecided',\nit does not make them "
        "disappear",
        fontsize=9,
    )

    matrix = np.array(
        [[1.0 if determinable[label][key] else 0.0 for key in LIMIT_KEYS] for label in variants]
    )
    axes[1].imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[1].text(
                column,
                row,
                "yes" if matrix[row, column] else "NO",
                ha="center",
                va="center",
                fontsize=7,
                color="#222222",
            )
    axes[1].set_xticks(range(len(LIMIT_KEYS)))
    axes[1].set_xticklabels(LIMIT_KEYS, rotation=38, ha="right", fontsize=7)
    axes[1].set_yticks(range(len(variants)))
    axes[1].set_yticklabels(list(variants))
    axes[1].grid(False)
    axes[1].set_title(
        "the seven determinability limits, per record variant\n"
        "three are never determinable: they are structural",
        fontsize=9,
    )
    figure.suptitle(
        "telemgap.determinability: the first output of every call, computed from the "
        f"record supplied.\n{SAMPLES} frames of the held-out split, four variants of the "
        "same record.",
        fontsize=10,
    )
    figure.subplots_adjust(wspace=0.3)
    path = save(figure, "determinability")
    print(f"written: {path}")
    for label in variants:
        print(f"  {label:<14s} {counts[label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
