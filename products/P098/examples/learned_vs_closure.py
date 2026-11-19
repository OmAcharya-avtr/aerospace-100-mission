"""The learned staleness-risk model against the baselines, on both splits.

Writes ``screenshots/learned_vs_closure.png``.

What to notice: the closure comparison reaches an area under the ROC curve of
1.0000 on complete declarations and the learned model does not come near it on
either split -- which is the result the specification predicted and the result
this product publishes. The right panel is the one case in which a model could
have helped, an incomplete dependency declaration, and the model does not beat
the degraded closure comparison there either.
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

from evidencestale.benchmark import run_benchmark  # noqa: E402
from evidencestale.corpus import MAIN_CORPUS, load_corpus  # noqa: E402

METHODS = (
    ("baseline1_timestamp", "timestamp\n(baseline 1)", _bootstrap.COLOUR_TIMESTAMP),
    ("baseline1b_timestamp_depth1", "timestamp\ndepth 1", _bootstrap.COLOUR_TIMESTAMP),
    ("baseline2_closure_hash", "closure hash\n(baseline 2)", _bootstrap.COLOUR_CLOSURE),
    ("learned_forest", "learned\nforest", _bootstrap.COLOUR_LEARNED),
)


def main() -> int:
    """Run both benchmarks and draw the comparison."""
    print("P098 EvidenceStale -- learned model against the baselines")
    print(f"fast path (EVIDENCESTALE_FAST=1): {_bootstrap.FAST}")
    print(f"grouped folds: {_bootstrap.EFFECTIVE_SPLITS}")
    corpus = load_corpus(ROOT / MAIN_CORPUS)
    results = {
        view: run_benchmark(corpus, view, n_group_splits=_bootstrap.EFFECTIVE_SPLITS)
        for view in ("complete", "partial")
    }
    print("")
    for view, result in results.items():
        print(f"{view} dependency declarations, fit seconds {result.fit_seconds:.3f}")
        print(f"  {'method':<30} {'artifact auc':>13} {'time auc':>10}")
        for key, _label, _colour in METHODS:
            print(
                f"  {key:<30} {result.summary(key, 'artifact')['auc']:>13.4f} "
                f"{result.summary(key, 'time')['auc']:>10.4f}"
            )

    figure, axes_pair = plt.subplots(
        1, 2, figsize=(11.0, 5.0), sharey=True
    )
    positions = np.arange(len(METHODS), dtype=float)
    width = 0.36
    titles = {
        "complete": "complete dependency declarations",
        "partial": "30 per cent of dependency edges hidden",
    }
    for axes, view in zip(axes_pair, ("complete", "partial"), strict=True):
        result = results[view]
        artifact = [result.summary(key, "artifact")["auc"] for key, _l, _c in METHODS]
        time_scores = [result.summary(key, "time")["auc"] for key, _l, _c in METHODS]
        colours = [colour for _key, _label, colour in METHODS]
        axes.bar(
            positions - width / 2.0,
            artifact,
            width,
            color=colours,
            label="held-out artifacts",
        )
        axes.bar(
            positions + width / 2.0,
            time_scores,
            width,
            color=colours,
            alpha=0.55,
            hatch="//",
            label="held-out time",
        )
        for position, value in zip(positions - width / 2.0, artifact, strict=True):
            axes.annotate(
                f"{value:.3f}",
                (position, value),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7,
            )
        for position, value in zip(positions + width / 2.0, time_scores, strict=True):
            axes.annotate(
                f"{value:.3f}",
                (position, value),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7,
            )
        axes.axhline(0.5, color="#888888", linewidth=0.8, linestyle=":")
        axes.set_xticks(positions)
        axes.set_xticklabels([label for _key, label, _colour in METHODS], fontsize=8)
        axes.set_title(titles[view], fontsize=10)
        axes.grid(axis="y", **_bootstrap.GRID_STYLE)
        axes.set_ylim(0.0, 1.08)
    axes_pair[0].set_ylabel("area under the ROC curve (dimensionless)")
    axes_pair[0].legend(loc="upper left", fontsize=8)
    figure.suptitle(
        "The exact method wins on both splits, and the learned model is published "
        "saying so",
        fontsize=11,
    )
    figure.text(
        0.01,
        -0.03,
        "Research-grade. Not flight-qualified, not certified, not approved for "
        "operational aerospace use. The dotted line is chance.",
        fontsize=8,
        color="#555555",
    )
    saved = _bootstrap.save(figure, "learned_vs_closure.png")
    plt.close(figure)
    print("")
    print(f"figure written to {saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
