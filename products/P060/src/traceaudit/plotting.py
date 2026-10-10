"""Matplotlib figures for an audit result.

Matplotlib is an optional dependency (``pip install traceaudit[plot]``) and is
imported inside each function so that the package imports without it.  The Agg
backend is forced; nothing here opens a window.
"""

from __future__ import annotations

from pathlib import Path

from .findings import CODE_DESCRIPTIONS, AuditResult
from .testreports import Outcome

_OUTCOME_ORDER = (
    Outcome.PASSED,
    Outcome.FAILED,
    Outcome.ERRORED,
    Outcome.SKIPPED,
    Outcome.XFAILED,
    Outcome.UNKNOWN,
)
_OUTCOME_VALUE = {o: i for i, o in enumerate(_OUTCOME_ORDER)}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_trace_matrix(result: AuditResult, path: str | Path, *, title: str | None = None) -> Path:
    """Requirement-by-test grid, coloured by outcome.

    One row per unique declared requirement, one column per claiming test
    case.  A requirement with no claiming test is an empty row, which is the
    visual form of finding TA001.
    """
    plt = _pyplot()
    reqs = list(result.matrix.requirement_to_tests)
    tests = [c.node_id for c in result.report.cases if c.claims]
    grid = [[-1.0] * max(len(tests), 1) for _ in range(max(len(reqs), 1))]
    for i, req in enumerate(reqs):
        for node in result.matrix.requirement_to_tests[req]:
            if node in tests:
                grid[i][tests.index(node)] = float(
                    _OUTCOME_VALUE[result.matrix.outcome_by_test[node]]
                )
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(tests) + 4.0),
                                    max(3.0, 0.36 * len(reqs) + 2.0)))
    cmap = plt.get_cmap("viridis", len(_OUTCOME_ORDER))
    masked = [[None if v < 0 else v for v in row] for row in grid]
    import numpy as np  # noqa: PLC0415 - optional plotting dependency

    array = np.ma.masked_invalid(
        np.array([[float("nan") if v is None else v for v in row] for row in masked])
    )
    cmap = cmap.with_extremes(bad="#eeeeee")
    im = ax.imshow(array, cmap=cmap, vmin=-0.5, vmax=len(_OUTCOME_ORDER) - 0.5, aspect="auto")
    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels([t.split("::")[-1] for t in tests], rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(reqs)))
    ax.set_yticklabels(reqs, fontsize=8)
    ax.set_xlabel("claiming test case")
    ax.set_ylabel("declared requirement")
    ax.set_title(title or "requirement-to-test trace matrix (grey = no claim)")
    cbar = fig.colorbar(im, ax=ax, ticks=range(len(_OUTCOME_ORDER)))
    cbar.ax.set_yticklabels([o.value for o in _OUTCOME_ORDER], fontsize=7)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_coverage_figures(
    results: dict[str, AuditResult], path: str | Path, *, title: str | None = None
) -> Path:
    """Grouped bar chart of the coverage figures for several audits.

    The denominator is printed on each group, because a coverage percentage
    without its denominator is the thing this product exists to refuse.
    """
    plt = _pyplot()
    labels = list(results)
    kinds = ["nominal coverage", "executed coverage", "passing coverage"]
    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 4.0, 4.4))
    width = 0.26
    for k, kind in enumerate(kinds):
        xs, ys = [], []
        for i, label in enumerate(labels):
            figure = next((f for f in results[label].coverage if f.label == kind), None)
            xs.append(i + (k - 1) * width)
            ys.append(0.0 if figure is None or figure.percent is None else figure.percent)
        ax.bar(xs, ys, width=width, label=kind)
    for i, label in enumerate(labels):
        denominator = results[label].matrix.n_requirements
        ax.text(i, 103.0, f"n = {denominator}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("coverage [%]")
    ax.set_title(title or "the same requirements, three definitions of covered")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_finding_breakdown(
    results: dict[str, AuditResult], path: str | Path, *, title: str | None = None
) -> Path:
    """Stacked bar of finding counts per code, for several audits."""
    plt = _pyplot()
    labels = list(results)
    codes = list(CODE_DESCRIPTIONS)
    fig, ax = plt.subplots(figsize=(1.8 * len(labels) + 5.0, 4.4))
    bottoms = [0.0] * len(labels)
    for code in codes:
        heights = [float(results[label].counts_by_code().get(code, 0)) for label in labels]
        ax.bar(labels, heights, bottom=bottoms, label=f"{code} {CODE_DESCRIPTIONS[code]}")
        bottoms = [b + h for b, h in zip(bottoms, heights, strict=True)]
    ax.set_ylabel("findings")
    ax.set_ylim(0, max(bottoms + [1.0]) * 1.55)
    ax.set_title(title or "findings by code")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_confusion(
    matrix: dict[str, int], path: str | Path, *, title: str | None = None, subtitle: str = ""
) -> Path:
    """2x2 confusion matrix for the assertion heuristic.

    Keys expected: ``tp``, ``fp``, ``fn``, ``tn``, where the positive class is
    "the heuristic says this test has no assertion".
    """
    plt = _pyplot()
    import numpy as np  # noqa: PLC0415 - optional plotting dependency

    array = np.array([[matrix["tp"], matrix["fp"]], [matrix["fn"], matrix["tn"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    im = ax.imshow(array, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(array[i, j])}", ha="center", va="center",
                    fontsize=18, color="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["truly empty", "truly asserts"], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["flagged empty", "not flagged"], fontsize=9)
    ax.set_xlabel("hand label")
    ax.set_ylabel("heuristic verdict")
    ax.set_title(title or "assertion heuristic vs hand labels")
    if subtitle:
        ax.text(0.5, -0.22, subtitle, transform=ax.transAxes, ha="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
