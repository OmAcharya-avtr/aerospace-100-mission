"""Graph and churn features for the learned staleness-risk model.

WHAT IS DELIBERATELY NOT A FEATURE, AND WHY
-------------------------------------------
No feature here reads any artifact content, any digest, or any event after the
evidence was produced. That restriction is the whole design, and it is not a
handicap imposed to make the baseline win:

An indicator "some artifact in the declared closure has an event after the
production revision" would make the model a reimplementation of baseline 2 over
the declared graph, so a forest given that feature would score at or near
baseline 2 by construction and the comparison would measure nothing. It is
measured anyway, as a named ablation, in ``validation/validate_learned_model.py``
and reported in ``MODEL_CARD.md`` -- because the honest way to state "that
feature would be a leak" is to show what it does.

So the features are: the shape of the subject's declared dependency
neighbourhood, how much the artifacts in it had churned BEFORE the evidence was
produced, and how long ago the evidence was produced. These are exactly the
quantities available when the dependency declarations are incomplete, which is
the only circumstance in which a learned model could add anything over an exact
method, and that is the case the validation set measures.

UNITS
-----
``age`` is in revisions. The churn counts are counts of events. The two rates
are events per revision. Everything else is a count of artifacts or edges.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .graph import DEFAULT_DEPTH_LIMIT, DependencyGraph
from .model import EvidenceItem
from .replay import ChangeHistory

#: Feature names, in the column order :func:`feature_matrix` produces.
FEATURE_NAMES: tuple[str, ...] = (
    "age",
    "closure_size",
    "closure_depth",
    "direct_dependencies",
    "direct_dependents",
    "closure_leaf_count",
    "subject_writes_before",
    "subject_events_before",
    "closure_writes_before",
    "closure_max_writes_before",
    "closure_write_rate_before",
    "subject_touch_share_before",
)


def feature_matrix(
    items: Sequence[EvidenceItem],
    history: ChangeHistory,
    graph: DependencyGraph,
    evaluation_time: int,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
) -> np.ndarray:
    """Feature matrix of shape ``(len(items), len(FEATURE_NAMES))``, dtype float64.

    Parameters
    ----------
    items:
        Evidence items, one row each, in order.
    history:
        The change history. Only events STRICTLY BEFORE an item's production
        revision are read for that item's churn features.
    graph:
        The graph view the features are computed on -- the declared subset when
        measuring the incomplete-declaration case, the true graph otherwise.
    evaluation_time:
        Audit revision, used only for ``age``.
    depth_limit:
        Passed to the closure computation.

    Returns
    -------
    numpy.ndarray
    """
    if not items:
        raise ValueError(
            "feature_matrix() needs at least one evidence item; an empty matrix has no "
            "column meaning and would be silently accepted by the estimator."
        )
    dependencies = graph.dependencies()
    dependents = graph.dependents()
    closures = {
        item.subject: graph.closure(item.subject, depth_limit=depth_limit)
        for item in items
    }
    depths = {
        item.subject: graph.closure_depth(item.subject, depth_limit=depth_limit)
        for item in items
    }
    writes_cache: dict[int, dict[str, int]] = {}
    events_cache: dict[int, dict[str, int]] = {}

    rows: list[list[float]] = []
    for item in items:
        produced = item.produced_at
        if produced not in writes_cache:
            writes_cache[produced] = history.write_count_before(produced)
            events_cache[produced] = history.event_count_before(produced)
        writes = writes_cache[produced]
        events = events_cache[produced]
        closure = closures[item.subject]
        leaf_count = sum(1 for name in closure if not dependencies[name])
        closure_writes = sum(writes[name] for name in closure)
        closure_max = max(writes[name] for name in closure)
        denominator = max(1, produced)
        subject_events = events[item.subject]
        touch_share = (
            0.0
            if subject_events == 0
            else (subject_events - writes[item.subject]) / subject_events
        )
        rows.append(
            [
                float(evaluation_time - produced),
                float(len(closure)),
                float(depths[item.subject]),
                float(len(dependencies[item.subject])),
                float(len(dependents[item.subject])),
                float(leaf_count),
                float(writes[item.subject]),
                float(subject_events),
                float(closure_writes),
                float(closure_max),
                closure_writes / denominator,
                touch_share,
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def leak_feature(
    items: Sequence[EvidenceItem],
    history: ChangeHistory,
    graph: DependencyGraph,
    evaluation_time: int,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
) -> np.ndarray:
    """The ablation feature: did any declared-closure artifact have an event after?

    Shape ``(len(items), 1)``. This reproduces the information baseline 2 uses,
    so a model given it is not a risk model. It exists so the claim "that would
    be a leak" can be measured instead of asserted.
    """
    closures = {
        item.subject: graph.closure(item.subject, depth_limit=depth_limit)
        for item in items
    }
    by_artifact: dict[str, list[int]] = {name: [] for name in history.artifacts}
    for event in history.events:
        by_artifact[event.artifact].append(event.time)
    rows: list[list[float]] = []
    for item in items:
        touched = any(
            any(item.produced_at < time <= evaluation_time for time in by_artifact[name])
            for name in closures[item.subject]
        )
        rows.append([float(touched)])
    return np.asarray(rows, dtype=np.float64)
