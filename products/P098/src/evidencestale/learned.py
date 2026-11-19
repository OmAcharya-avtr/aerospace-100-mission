"""The learned staleness-risk model, with a confidence output.

ARCHITECTURE AND WHY IT IS SMALL
--------------------------------
``sklearn.ensemble.RandomForestClassifier``, 200 trees, ``max_depth=6``,
``n_jobs=1``, fixed ``random_state``. Three hard constraints decide this, not
preference: the corpus has 480 evidence items over 240 subject artifacts, so a
model with many parameters would memorise subjects; the build environment has
two cores shared by five concurrent builds and a 60 s budget per fit; and a
forest gives a vote spread across trees, which is a conservative but honest
uncertainty output without training a second model.

``max_depth=6`` caps a tree at 64 leaves against at most 480 rows, which is the
loosest cap that still forces leaves to average several items.

THE CONFIDENCE OUTPUT IS NOT A POINT ESTIMATE
---------------------------------------------
:meth:`StalenessRiskModel.predict_with_confidence` returns, per item:

- ``probability`` -- the forest's mean vote for "stale", in [0, 1];
- ``spread`` -- the standard deviation of the per-tree votes, in [0, 0.5], which
  is the forest's own disagreement and is the uncertainty output;
- ``confidence`` -- ``max(p, 1 - p)``, the margin of the thresholded decision.

``spread`` is a measure of model disagreement on the training distribution. It
is **not** a calibrated credible interval and nothing here claims it is: a
forest whose trees all agree on a wrong answer reports a small spread. That
failure mode is listed in ``MODEL_CARD.md``.

NO MODEL IS SERIALISED. Nothing in this product writes a joblib or pickle
artifact, so there is no stored-format compatibility surface; the
``scikit-learn`` ceiling in ``pyproject.toml`` is declared anyway because the
measured metrics depend on the estimator's own defaults and tie-breaking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

#: Hyperparameters. Fixed before the first score was looked at, and not
#: retuned afterwards: the build guide forbids retuning to manufacture a win,
#: and the numbers in MODEL_CARD.md were not retuned.
N_ESTIMATORS = 200
MAX_DEPTH = 6
RANDOM_STATE = 980098


@dataclass(frozen=True)
class RiskPrediction:
    """Per-item model output.

    Attributes
    ----------
    probability:
        Forest mean vote for "stale", dimensionless, in [0, 1].
    spread:
        Standard deviation of per-tree votes, dimensionless, in [0, 0.5].
    confidence:
        ``max(p, 1 - p)``, the margin of the thresholded decision, in [0.5, 1].
    """

    probability: np.ndarray
    spread: np.ndarray
    confidence: np.ndarray

    def decide(self, threshold: float = 0.5) -> np.ndarray:
        """Thresholded 0/1 decisions. The threshold is reported wherever used."""
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must lie in (0, 1), got {threshold}")
        return (self.probability >= threshold).astype(np.int64)


class StalenessRiskModel:
    """A random forest over graph and churn features, with a confidence output."""

    def __init__(
        self,
        n_estimators: int = N_ESTIMATORS,
        max_depth: int = MAX_DEPTH,
        random_state: int = RANDOM_STATE,
    ) -> None:
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be at least 1, got {n_estimators}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be at least 1, got {max_depth}")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._forest: RandomForestClassifier | None = None
        self._single_class: int | None = None

    def fit(self, features: np.ndarray, labels: Sequence[int]) -> StalenessRiskModel:
        """Fit the forest. Returns ``self``.

        A training set with one class present is accepted and recorded, because
        a grouped or time split can produce one; the model then predicts that
        class with probability 1 and a spread of 0, and the scoring code reports
        the degenerate fold rather than dropping it.
        """
        matrix = np.asarray(features, dtype=np.float64)
        target = np.asarray([int(bool(value)) for value in labels], dtype=np.int64)
        if matrix.ndim != 2:
            raise ValueError(
                f"features must be a 2-D array of shape (items, columns), got shape "
                f"{matrix.shape}"
            )
        if matrix.shape[0] != target.size:
            raise ValueError(
                f"features has {matrix.shape[0]} rows and labels has {target.size} "
                f"entries; they must be aligned item by item."
            )
        if target.size == 0:
            raise ValueError("fit() needs at least one training item")
        classes = np.unique(target)
        if classes.size == 1:
            self._forest = None
            self._single_class = int(classes[0])
            return self
        self._single_class = None
        self._forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=1,
        )
        self._forest.fit(matrix, target)
        return self

    @property
    def degenerate(self) -> bool:
        """Whether the training set had a single class, so no forest was fitted."""
        return self._single_class is not None

    def predict_with_confidence(self, features: np.ndarray) -> RiskPrediction:
        """Probability, per-tree spread and decision margin for each row."""
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(
                f"features must be a 2-D array, got shape {matrix.shape}"
            )
        if self._forest is None and self._single_class is None:
            raise ValueError(
                "predict_with_confidence() was called before fit(); fit the model on a "
                "training split first."
            )
        if self._forest is None:
            value = float(self._single_class)
            probability = np.full(matrix.shape[0], value, dtype=np.float64)
            spread = np.zeros(matrix.shape[0], dtype=np.float64)
            return RiskPrediction(
                probability=probability,
                spread=spread,
                confidence=np.maximum(probability, 1.0 - probability),
            )
        votes = np.stack(
            [tree.predict(matrix).astype(np.float64) for tree in self._forest.estimators_]
        )
        probability = self._forest.predict_proba(matrix)[:, 1].astype(np.float64)
        spread = votes.std(axis=0, ddof=0)
        return RiskPrediction(
            probability=probability,
            spread=spread,
            confidence=np.maximum(probability, 1.0 - probability),
        )

    def feature_importances(self) -> np.ndarray:
        """Impurity-based importances, or an empty array for a degenerate fit."""
        if self._forest is None:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(self._forest.feature_importances_, dtype=np.float64)
