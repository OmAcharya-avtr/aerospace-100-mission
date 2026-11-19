"""The learned model against the baselines, on identical held-out test items.

ORDER OF WORK, WHICH IS BINDING IN THIS MISSION
-----------------------------------------------
Both classical baselines are implemented and validated against the
by-construction ground truth FIRST -- that is
``validation/validate_baselines.py`` -- and only then is the learned model
benchmarked, here and in ``validation/validate_learned_model.py``, on the same
corpus and the same splits. An under-strength baseline inflates an apparent
learned win, and the baseline that this specification names as the thing being
refuted is also implemented in its strongest honest form (the depth-1 variant)
so the comparison cannot flatter the model by weakening its opponent.

THE PRE-REGISTERED EXPECTATION
------------------------------
The specification states the expected result: the closure comparison is exact
under its stated conditions, and the learned model cannot beat it. That is
reported as the finding whichever way it comes out, and nothing here is retuned
after a score is seen.

THE ONE CASE WHERE A LEARNED MODEL COULD ADD ANYTHING
-----------------------------------------------------
If the dependency declarations are incomplete, the closure comparison is no
longer exact: it cannot see a change that arrives through an undeclared edge,
so it makes missed-staleness errors and no false alarms. :func:`run_benchmark`
is therefore run twice, once on the complete edge set and once on the declared
subset, with the ground truth always over the complete set. Both results are
reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .baselines import (
    baseline_closure_hash,
    baseline_timestamp,
    baseline_timestamp_depth1,
    ground_truth,
)
from .corpus import Corpus
from .evaluation import (
    Score,
    Split,
    artifact_splits,
    score_binary,
    score_probabilistic,
    summarise,
    time_splits,
)
from .features import feature_matrix, leak_feature
from .learned import StalenessRiskModel

#: Decision threshold applied to the model's probability. Reported wherever a
#: thresholded figure appears, because a threshold chosen after seeing a score
#: is a tuning step.
THRESHOLD = 0.5


@dataclass
class BenchmarkResult:
    """All scores from one benchmark run, under one graph view."""

    declarations: str
    n_items: int
    n_positive: int
    scores: list[Score] = field(default_factory=list)
    fit_seconds: float = 0.0
    importances: dict[str, float] = field(default_factory=dict)
    degenerate_folds: int = 0

    def by(self, method: str, split_kind: str) -> list[Score]:
        """Scores of one method on one split kind."""
        return [
            score
            for score in self.scores
            if score.method == method and score.split_kind == split_kind
        ]

    def summary(self, method: str, split_kind: str) -> dict[str, float]:
        """Mean figures of one method on one split kind."""
        return summarise(self.by(method, split_kind))

    def methods(self) -> tuple[str, ...]:
        """Method names present, in a stable order."""
        order: list[str] = []
        for score in self.scores:
            if score.method not in order:
                order.append(score.method)
        return tuple(order)


def run_benchmark(
    corpus: Corpus,
    declarations: str = "complete",
    *,
    n_group_splits: int = 5,
    include_leak_ablation: bool = False,
) -> BenchmarkResult:
    """Score every method on held-out artifacts and held-out time.

    Parameters
    ----------
    corpus:
        The corpus to score over.
    declarations:
        ``"complete"`` or ``"partial"``. The baselines and the model features
        both see this view; the ground truth is always the complete one.
    n_group_splits:
        Folds for the grouped split.
    include_leak_ablation:
        Also fit a model with the post-production event indicator appended,
        which reproduces baseline 2's information and is reported under the
        method name ``learned_leak_ablation``.

    Returns
    -------
    BenchmarkResult
    """
    import time

    view = corpus.graph(declarations)
    items = corpus.evidence
    truth = ground_truth(
        items, corpus.history, corpus.true_graph, corpus.evaluation_time
    ).as_array()
    deterministic = {
        "baseline1_timestamp": baseline_timestamp(
            items, corpus.history, view, corpus.evaluation_time
        ).as_array(),
        "baseline1b_timestamp_depth1": baseline_timestamp_depth1(
            items, corpus.history, view, corpus.evaluation_time
        ).as_array(),
        "baseline2_closure_hash": baseline_closure_hash(
            items, corpus.history, view, corpus.evaluation_time
        ).as_array(),
    }
    features = feature_matrix(
        items, corpus.history, view, corpus.evaluation_time
    )
    leak = (
        np.hstack(
            [features, leak_feature(items, corpus.history, view, corpus.evaluation_time)]
        )
        if include_leak_ablation
        else None
    )

    splits: list[Split] = artifact_splits(corpus.subjects(), n_splits=n_group_splits)
    splits += time_splits([item.produced_at for item in items])

    result = BenchmarkResult(
        declarations=declarations,
        n_items=len(items),
        n_positive=sum(truth),
    )
    importance_sum = np.zeros(features.shape[1], dtype=np.float64)
    importance_folds = 0
    elapsed = 0.0
    for split in splits:
        for method, decisions in deterministic.items():
            result.scores.append(score_binary(method, split, decisions, truth))
        train_labels = [truth[index] for index in split.train]
        model = StalenessRiskModel()
        start = time.perf_counter()
        model.fit(features[split.train], train_labels)
        elapsed += time.perf_counter() - start
        if model.degenerate:
            result.degenerate_folds += 1
        prediction = model.predict_with_confidence(features[split.test])
        result.scores.append(
            score_probabilistic(
                "learned_forest",
                split,
                prediction.probability,
                truth,
                threshold=THRESHOLD,
            )
        )
        importances = model.feature_importances()
        if importances.size == features.shape[1]:
            importance_sum += importances
            importance_folds += 1
        if leak is not None:
            leak_model = StalenessRiskModel()
            leak_model.fit(leak[split.train], train_labels)
            leak_prediction = leak_model.predict_with_confidence(leak[split.test])
            result.scores.append(
                score_probabilistic(
                    "learned_leak_ablation",
                    split,
                    leak_prediction.probability,
                    truth,
                    threshold=THRESHOLD,
                )
            )
    result.fit_seconds = elapsed
    if importance_folds:
        from .features import FEATURE_NAMES

        mean = importance_sum / importance_folds
        result.importances = {
            name: float(value) for name, value in zip(FEATURE_NAMES, mean, strict=True)
        }
    return result


def confidence_table(
    corpus: Corpus,
    declarations: str = "complete",
    bins: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> list[tuple[float, float, int, float]]:
    """Agreement with the truth within bands of the model's confidence output.

    One grouped-fold pass: every item is in exactly one test fold, so each item
    is predicted by a model that never saw its subject artifact. Returns rows of
    ``(band low, band high, items, agreement)``. A confidence output is only
    worth having if agreement rises with it, and this is the measurement of
    whether it does.
    """
    view = corpus.graph(declarations)
    items = corpus.evidence
    truth = ground_truth(
        items, corpus.history, corpus.true_graph, corpus.evaluation_time
    ).as_array()
    features = feature_matrix(items, corpus.history, view, corpus.evaluation_time)
    confidences = np.zeros(len(items), dtype=np.float64)
    correct = np.zeros(len(items), dtype=np.int64)
    for split in artifact_splits(corpus.subjects(), n_splits=5):
        model = StalenessRiskModel().fit(
            features[split.train], [truth[index] for index in split.train]
        )
        prediction = model.predict_with_confidence(features[split.test])
        decisions = prediction.decide(THRESHOLD)
        for position, index in enumerate(split.test):
            confidences[index] = prediction.confidence[position]
            correct[index] = int(decisions[position] == truth[index])
    rows: list[tuple[float, float, int, float]] = []
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidences >= low) & (
            (confidences < high) if high < 1.0 else (confidences <= high)
        )
        count = int(mask.sum())
        agreement = float(correct[mask].mean()) if count else float("nan")
        rows.append((float(low), float(high), count, agreement))
    return rows
