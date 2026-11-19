"""The independently derived bracket on the stale fraction, over a grid.

Writes ``screenshots/bound_bracket.png``.

What to notice: the measured stale fraction stays inside a bracket whose
endpoints are computed from four integers -- the change count, the maximum
number of direct dependents, the declared impact depth and the artifact count
-- with no reachability search, no content and no digest. The upper bound rises
towards 1 as the change count grows, which is a property of the bound and not
of the measurement.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from evidencestale.baselines import ground_truth  # noqa: E402
from evidencestale.bounds import moore_bound, stale_fraction_bracket  # noqa: E402
from evidencestale.corpus import MAIN_CORPUS, load_corpus  # noqa: E402
from evidencestale.replay import changed_sets_by_item  # noqa: E402

#: Evaluation revisions. Every evidence item is produced in revisions 1..600,
#: so the evidence set is complete at every point on this grid.
GRID = (600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000)


def main() -> int:
    """Compute the bracket over the grid and draw it."""
    print("P098 EvidenceStale -- independent bracket on the stale fraction")
    print(f"fast path (EVIDENCESTALE_FAST=1): {_bootstrap.FAST}")
    corpus = load_corpus(ROOT / MAIN_CORPUS)
    items = corpus.evidence
    artifacts = len(corpus.true_graph.nodes)
    degree = corpus.declared_max_dependents
    depth = corpus.declared_max_impact_depth
    print("")
    print(f"artifacts N              : {artifacts}")
    print(f"max direct dependents d  : {degree}")
    print(f"declared impact depth L  : {depth}")
    print(f"M(d, L)                  : {moore_bound(degree, depth)}")
    print("")
    print(f"{'revision':>9} {'k':>4} {'lower':>9} {'measured':>9} {'upper':>9} {'holds':>6}")

    revisions: list[int] = []
    lowers: list[float] = []
    measured: list[float] = []
    uppers: list[float] = []
    changes: list[int] = []
    for revision in GRID:
        subset = [item for item in items if item.produced_at <= revision]
        truth = ground_truth(subset, corpus.history, corpus.true_graph, revision)
        changed = changed_sets_by_item(
            corpus.history,
            [(item.identifier, item.produced_at) for item in subset],
            revision,
        )
        bracket = stale_fraction_bracket(
            subset,
            changed,
            corpus.true_graph.dependencies(),
            measured=truth.stale_fraction,
            max_out_degree=degree,
            depth_limit=depth,
            artifact_count=artifacts,
        )
        revisions.append(revision)
        lowers.append(bracket.lower)
        measured.append(bracket.measured)
        uppers.append(bracket.upper)
        changes.append(bracket.changed_count)
        print(
            f"{revision:>9} {bracket.changed_count:>4} {bracket.lower:>9.6f} "
            f"{bracket.measured:>9.6f} {bracket.upper:>9.6f} {str(bracket.holds):>6}"
        )

    figure, axes = plt.subplots(figsize=_bootstrap.FIGURE_SIZE)
    axes.fill_between(
        revisions,
        lowers,
        uppers,
        color=_bootstrap.COLOUR_CLOSURE,
        alpha=0.15,
        label="bracket from the Moore bound and a union bound (no reachability search)",
    )
    axes.plot(
        revisions,
        uppers,
        color=_bootstrap.COLOUR_CLOSURE,
        linewidth=1.2,
        linestyle="--",
        label="upper bound  m*min(N, k*M(d,L))/E",
    )
    axes.plot(
        revisions,
        lowers,
        color=_bootstrap.COLOUR_CLOSURE,
        linewidth=1.2,
        linestyle=":",
        label="lower bound  depth-1 count / E",
    )
    axes.plot(
        revisions,
        measured,
        color=_bootstrap.COLOUR_TRUTH,
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="measured stale fraction (closure comparison, exact here)",
    )
    for revision, value, count in zip(revisions, uppers, changes, strict=True):
        axes.annotate(
            f"k={count}",
            (revision, value),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=7,
            color="#555555",
        )
    axes.set_xlabel("audit revision")
    axes.set_ylabel("stale fraction (dimensionless)")
    axes.set_ylim(0.0, 1.05)
    axes.set_title(
        f"The measured fraction stays inside an independently derived bracket "
        f"(N={artifacts}, d={degree}, L={depth}, M={moore_bound(degree, depth)})"
    )
    axes.grid(**_bootstrap.GRID_STYLE)
    axes.legend(loc="upper left", fontsize=8)
    figure.text(
        0.01,
        -0.05,
        "Research-grade. Not flight-qualified, not certified, not approved for "
        "operational aerospace use.",
        fontsize=8,
        color="#555555",
    )
    saved = _bootstrap.save(figure, "bound_bracket.png")
    plt.close(figure)
    print("")
    print(f"figure written to {saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
