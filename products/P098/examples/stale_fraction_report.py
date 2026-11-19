"""The headline output: the stale fraction of an assurance case, by method.

Writes ``screenshots/stale_fraction.png``.

What to notice: the closure comparison sits exactly on the ground truth, while
both timestamp variants sit well away from it -- and the plain timestamp
baseline's aggregate error is SMALLER than its per-item error, because its
missed staleness and its false alarms partly cancel in the aggregate.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from evidencestale.corpus import MAIN_CORPUS, load_corpus  # noqa: E402
from evidencestale.report import build_report, render_report  # noqa: E402


def main() -> int:
    """Build the headline report and draw it."""
    print("P098 EvidenceStale -- headline stale fraction")
    print(f"fast path (EVIDENCESTALE_FAST=1): {_bootstrap.FAST}")
    print(f"corpus: {MAIN_CORPUS}")
    print("")
    corpus = load_corpus(ROOT / MAIN_CORPUS)
    report = build_report(
        corpus,
        "complete",
        resamples=_bootstrap.EFFECTIVE_RESAMPLES,
        seed=_bootstrap.SEED,
    )
    print(render_report(report))

    names = [
        "ground truth\n(baseline 3)",
        "closure hash\n(baseline 2)",
        "timestamp\n(baseline 1)",
        "timestamp depth 1\n(baseline 1b)",
    ]
    values = [
        report.truth.stale_fraction,
        report.closure_hash.stale_fraction,
        report.timestamp.stale_fraction,
        report.timestamp_depth1.stale_fraction,
    ]
    colours = [
        _bootstrap.COLOUR_TRUTH,
        _bootstrap.COLOUR_CLOSURE,
        _bootstrap.COLOUR_TIMESTAMP,
        _bootstrap.COLOUR_TIMESTAMP,
    ]

    figure, axes = plt.subplots(figsize=_bootstrap.FIGURE_SIZE)
    bars = axes.bar(names, values, color=colours, width=0.6)
    interval = report.cluster_interval
    axes.axhspan(
        interval.low,
        interval.high,
        color=_bootstrap.COLOUR_TRUTH,
        alpha=0.12,
        zorder=0,
        label=(
            f"ground-truth {interval.confidence:.0%} cluster bootstrap "
            f"[{interval.low:.3f}, {interval.high:.3f}]"
        ),
    )
    axes.axhline(
        report.truth.stale_fraction,
        color=_bootstrap.COLOUR_TRUTH,
        linewidth=1.0,
        linestyle="--",
        zorder=1,
    )
    for bar, value in zip(bars, values, strict=True):
        axes.annotate(
            f"{value:.4f}",
            (bar.get_x() + bar.get_width() / 2.0, value),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
        )
    axes.set_ylabel("stale fraction of the assurance case (dimensionless)")
    axes.set_ylim(0.0, 1.0)
    axes.set_title(
        f"Stale fraction by method -- {len(corpus.evidence)} evidence items, "
        f"{len(corpus.true_graph.nodes)} artifacts, audit revision "
        f"{corpus.evaluation_time}"
    )
    axes.grid(axis="y", **_bootstrap.GRID_STYLE)
    axes.legend(loc="upper left", fontsize=9)
    figure.text(
        0.01,
        -0.04,
        "Research-grade. Not flight-qualified, not certified, not approved for "
        "operational aerospace use.",
        fontsize=8,
        color="#555555",
    )
    saved = _bootstrap.save(figure, "stale_fraction.png")
    plt.close(figure)
    print("")
    print(f"figure written to {saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
