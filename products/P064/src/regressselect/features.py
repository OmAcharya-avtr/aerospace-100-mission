"""Change features for the learned selector.

One feature row describes a (change, test node) pair. The features are the
four families the product scope names -- files touched, churn, test-module
distance and historical co-failure -- plus a lexical name-overlap term,
because in a numerical repository ``tests/test_quat.py`` naming
``sampleproj/quat.py`` is real signal and leaving it out would understate the
baselines it is being compared against.

Units are given per feature in :data:`FEATURE_NAMES`. Churn features are
``log1p`` transformed because churn is heavy-tailed in lines and an untransformed
tail dominates a tree split threshold. Distance is in import-graph hops, with
unreachable encoded as :data:`DISTANCE_CAP` *and* flagged separately, so the
model can distinguish "far" from "no path", which are different claims.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .history import History, Revision
from .importgraph import ImportGraph
from .inventory import module_of_node
from .prioritise import CoFailurePrioritiser

__all__ = [
    "FEATURE_NAMES",
    "DISTANCE_CAP",
    "feature_row",
    "build_design_matrix",
    "revision_features",
]

DISTANCE_CAP = 8

FEATURE_NAMES: tuple[str, ...] = (
    "n_changed_modules",            # count
    "log1p_total_churn",            # log1p(lines)
    "log1p_max_churn",              # log1p(lines)
    "reachable",                    # 0 or 1
    "distance_hops",                # import-graph hops, DISTANCE_CAP when unreachable
    "inverse_distance",             # 1 / (1 + hops), dimensionless
    "log1p_nearest_churn",          # log1p(lines) of the nearest changed module
    "cofail_max",                   # dimensionless probability
    "cofail_mean",                  # dimensionless probability
    "test_base_failure_rate",       # dimensionless probability
    "name_overlap",                 # dimensionless fraction in [0, 1]
)


def _basename_tokens(dotted: str) -> set[str]:
    leaf = dotted.rsplit(".", 1)[-1]
    if leaf.startswith("test_"):
        leaf = leaf[len("test_") :]
    return {part for part in leaf.split("_") if part}


def _name_overlap(test_module: str, changed_modules: Sequence[str]) -> float:
    left = _basename_tokens(test_module)
    if not left:
        return 0.0
    best = 0.0
    for mod in changed_modules:
        right = _basename_tokens(mod)
        if not right:
            continue
        best = max(best, len(left & right) / len(left | right))
    return best


def feature_row(
    node_id: str,
    changed_modules: Sequence[str],
    churn: dict[str, int],
    graph: ImportGraph,
    prioritiser: CoFailurePrioritiser,
) -> list[float]:
    """Feature vector for one (change, test node) pair, ordered as FEATURE_NAMES."""
    test_module = module_of_node(node_id)
    if test_module not in graph.edges:
        raise ValueError(f"test module {test_module!r} is not in the import graph")
    distances = {m: graph.distance(test_module, m) for m in changed_modules}
    finite = {m: d for m, d in distances.items() if d is not None}
    reachable = 1.0 if finite else 0.0
    if finite:
        nearest = min(finite, key=lambda m: (finite[m], m))
        hops = float(finite[nearest])
        nearest_churn = float(churn.get(nearest, 0))
    else:
        hops = float(DISTANCE_CAP)
        nearest_churn = 0.0
    rates = [prioritiser.conditional_rate(m, node_id) for m in changed_modules] or [0.0]
    total_churn = float(sum(churn.values()))
    max_churn = float(max(churn.values())) if churn else 0.0
    return [
        float(len(changed_modules)),
        float(np.log1p(total_churn)),
        float(np.log1p(max_churn)),
        reachable,
        hops,
        1.0 / (1.0 + hops),
        float(np.log1p(nearest_churn)),
        float(max(rates)),
        float(sum(rates) / len(rates)),
        float(prioritiser.score(node_id, [])),
        _name_overlap(test_module, changed_modules),
    ]


def build_design_matrix(
    history: History,
    graph: ImportGraph,
    prioritiser: CoFailurePrioritiser,
    node_ids: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Design matrix for every (revision, test node) pair in ``history``.

    Returns ``(X, y, index)`` where ``X`` has shape ``(n_pairs, len(FEATURE_NAMES))``,
    ``y`` is 1 where that test failed at that revision, and ``index`` lists the
    ``(revision_id, node_id)`` pair behind each row so a prediction can be
    traced back.
    """
    nodes = list(node_ids) if node_ids is not None else list(history.test_ids)
    if not nodes:
        raise ValueError("no test nodes to build features for")
    rows: list[list[float]] = []
    labels: list[int] = []
    index: list[tuple[str, str]] = []
    for rev in history.revisions:
        failed = set(rev.failed_tests)
        for node in nodes:
            rows.append(feature_row(node, rev.changed_modules, rev.churn, graph, prioritiser))
            labels.append(1 if node in failed else 0)
            index.append((rev.revision, node))
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), index


def revision_features(
    revision: Revision,
    nodes: Sequence[str],
    graph: ImportGraph,
    prioritiser: CoFailurePrioritiser,
) -> np.ndarray:
    """Feature matrix for one revision over ``nodes`` (rows follow ``nodes``)."""
    return np.asarray(
        [
            feature_row(n, revision.changed_modules, revision.churn, graph, prioritiser)
            for n in nodes
        ],
        dtype=float,
    )
