"""The two error directions of each method, never summed into one number.

Writes ``screenshots/error_directions.png``.

What to notice: the depth-1 timestamp variant trades missed staleness for false
alarms rather than removing either, and the closure comparison has a zero bar
in both directions on complete declarations and a missed-staleness bar only
once 30 per cent of the dependency edges are hidden.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from evidencestale.baselines import baseline_closure_hash, ground_truth  # noqa: E402
from evidencestale.corpus import MAIN_CORPUS, load_corpus  # noqa: E402
from evidencestale.metrics import error_directions  # noqa: E402
from evidencestale.report import build_report  # noqa: E402


def main() -> int:
    """Measure both error directions for every method and draw them."""
    print("P098 EvidenceStale -- error directions by method")
    print(f"fast path (EVIDENCESTALE_FAST=1): {_bootstrap.FAST}")
    corpus = load_corpus(ROOT / MAIN_CORPUS)
    report = build_report(
        corpus,
        "complete",
        resamples=_bootstrap.EFFECTIVE_RESAMPLES,
        seed=_bootstrap.SEED,
    )
    truth = ground_truth(
        corpus.evidence, corpus.history, corpus.true_graph, corpus.evaluation_time
    )
    partial = baseline_closure_hash(
        corpus.evidence, corpus.history, corpus.declared_graph, corpus.evaluation_time
    )
    rows = [
        ("timestamp\n(baseline 1)", report.errors("baseline1_timestamp")),
        ("timestamp depth 1\n(baseline 1b)", report.errors("baseline1b_timestamp_depth1")),
        ("closure hash\n(baseline 2)", report.errors("baseline2_closure_hash")),
        (
            "closure hash,\n30% edges hidden",
            error_directions(partial.as_array(), truth.as_array()),
        ),
    ]
    print("")
    print(f"{'method':<34} {'missed':>8} {'alarm':>8} {'agree':>8}")
    for name, errors in rows:
        label = name.replace("\n", " ")
        print(
            f"{label:<34} {errors.missed_staleness_rate:>8.4f} "
            f"{errors.false_alarm_rate:>8.4f} {errors.agreement:>8.4f}"
        )

    positions = np.arange(len(rows), dtype=float)
    width = 0.36
    figure, axes = plt.subplots(figsize=_bootstrap.FIGURE_SIZE)
    missed = [errors.missed_staleness_rate for _name, errors in rows]
    alarms = [errors.false_alarm_rate for _name, errors in rows]
    axes.bar(
        positions - width / 2.0,
        missed,
        width,
        color=_bootstrap.COLOUR_MISSED,
        label="missed staleness: evidence cited that no longer describes the system",
    )
    axes.bar(
        positions + width / 2.0,
        alarms,
        width,
        color=_bootstrap.COLOUR_ALARM,
        label="false alarm: work re-run that did not need re-running",
    )
    for position, value in zip(positions - width / 2.0, missed, strict=True):
        axes.annotate(
            f"{value:.3f}",
            (position, value),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
        )
    for position, value in zip(positions + width / 2.0, alarms, strict=True):
        axes.annotate(
            f"{value:.3f}",
            (position, value),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
        )
    axes.set_xticks(positions)
    axes.set_xticklabels([name for name, _errors in rows], fontsize=9)
    axes.set_ylabel("error rate (dimensionless)")
    axes.set_ylim(0.0, 1.0)
    axes.set_title(
        "The two error directions are different failures and are never summed"
    )
    axes.grid(axis="y", **_bootstrap.GRID_STYLE)
    axes.legend(loc="upper center", fontsize=8)
    figure.text(
        0.01,
        -0.06,
        "Research-grade. Not flight-qualified, not certified, not approved for "
        "operational aerospace use.",
        fontsize=8,
        color="#555555",
    )
    saved = _bootstrap.save(figure, "error_directions.png")
    plt.close(figure)
    print("")
    print(f"figure written to {saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
