"""Learned fault detector and isolator over residual-window features.

A single `sklearn.ensemble.RandomForestClassifier` maps the sixteen features of
:mod:`fdiscope.features` to one of the eight classes of
:class:`fdiscope.faults.FaultType`.  It is deliberately the *whole* AI content
of this package: the classical chi-squared test, CUSUM bank and GLR isolator
are implemented, validated and benchmarked first, and the model is measured
against them on the same held-out scenarios.

Why a forest and not something larger
-------------------------------------
PyTorch is not available in the target environment, the compute budget is two
CPU cores and three minutes, and the input is sixteen hand-designed scalar
features from a window -- not a sequence a recurrent or convolutional model
could exploit.  A random forest is the right size of hammer, trains in under a
second, and gives a class posterior directly.  A model this small also cannot
hide behind capacity: if it beats the classical tests, it is because the
features carry information the tests throw away.

Confidence output
-----------------
``predict_proba`` gives the fraction of trees voting for each class, and
:meth:`FaultClassifier.predict_with_confidence` returns the winning class and
that fraction.  The **detection** score is ``1 - P(NONE)``, which is what the
ROC curves sweep.

The vote fraction is an ensemble-agreement heuristic, not a calibrated
probability, and this package measures its calibration rather than assuming
it: ``validation/isolation_confusion.py`` reports the empirical accuracy of
predictions bucketed by confidence, which is the reliability diagram in table
form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier

from .faults import FAULT_CLASSES, FaultType
from .features import N_FEATURES, feature_names

__all__ = ["FaultClassifier", "ClassifierPrediction"]


@dataclass(frozen=True)
class ClassifierPrediction:
    """Batch prediction.

    Attributes
    ----------
    classes : list of FaultType
        Winning class per row.
    confidence : ndarray, shape (n,)
        Vote fraction for the winning class, in ``(0, 1]``.
    proba : ndarray, shape (n, 8)
        Full posterior over :data:`fdiscope.faults.FAULT_CLASSES`.
    detection_score : ndarray, shape (n,)
        ``1 - P(NONE)``, the score used for detection ROC curves.
    """

    classes: list[FaultType]
    confidence: NDArray[np.float64]
    proba: NDArray[np.float64]
    detection_score: NDArray[np.float64]


class FaultClassifier:
    """Random-forest detector/isolator over residual-window features.

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    max_depth : int or None
        Tree depth cap.
    min_samples_leaf : int
        Minimum samples per leaf; the main regulariser here.
    random_state : int
        Seed.  Fixed seeds give bit-identical models, checked by
        ``tests/test_classifier.py``.
    class_weight : str or None
        Passed to scikit-learn; ``"balanced"`` because the fault-free class
        has many more training windows than any single fault class.

    Notes
    -----
    ``n_jobs`` is fixed at 1: the target environment has two cores and the
    benchmark scripts are already the long pole.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = 12,
        min_samples_leaf: int = 2,
        random_state: int = 0,
        class_weight: str | None = "balanced",
    ) -> None:
        if int(n_estimators) < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")
        if max_depth is not None and int(max_depth) < 1:
            raise ValueError(f"max_depth must be >= 1 or None, got {max_depth}")
        if int(min_samples_leaf) < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")
        self.model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
            class_weight=class_weight,
            n_jobs=1,
        )
        self._fitted = False

    @staticmethod
    def _as_features(x: ArrayLike) -> NDArray[np.float64]:
        a = np.atleast_2d(np.asarray(x, dtype=float))
        if a.ndim != 2 or a.shape[1] != N_FEATURES:
            raise ValueError(f"features must be (n, {N_FEATURES}), got shape {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError("features must be finite")
        return a

    def fit(self, x: ArrayLike, y: ArrayLike) -> FaultClassifier:
        """Fit on feature rows and integer class indices.

        Parameters
        ----------
        x : array_like, shape (n, 16)
            Feature rows from :func:`fdiscope.features.feature_matrix`.
        y : array_like of int, shape (n,)
            Class indices into :data:`fdiscope.faults.FAULT_CLASSES`.

        Returns
        -------
        FaultClassifier
            ``self``, fitted.
        """
        feats = self._as_features(x)
        labels = np.asarray(y, dtype=int).reshape(-1)
        if labels.size != feats.shape[0]:
            raise ValueError(f"got {feats.shape[0]} feature rows and {labels.size} labels")
        if labels.size < 2:
            raise ValueError("need at least 2 training rows")
        if labels.min() < 0 or labels.max() >= len(FAULT_CLASSES):
            raise ValueError(f"labels must lie in [0, {len(FAULT_CLASSES)})")
        self.model.fit(feats, labels)
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("classifier is not fitted; call fit() first")

    def predict_proba(self, x: ArrayLike) -> NDArray[np.float64]:
        """Posterior over all eight classes, shape ``(n, 8)``.

        Classes absent from the training set get exactly zero probability, so
        the returned array always has eight columns in
        :data:`fdiscope.faults.FAULT_CLASSES` order regardless of what the
        training set contained.
        """
        self._check_fitted()
        feats = self._as_features(x)
        raw = self.model.predict_proba(feats)
        out = np.zeros((feats.shape[0], len(FAULT_CLASSES)))
        for col, cls in enumerate(self.model.classes_):
            out[:, int(cls)] = raw[:, col]
        return out

    def predict_with_confidence(self, x: ArrayLike) -> ClassifierPrediction:
        """Winning class, its vote fraction, the posterior and the ROC score."""
        proba = self.predict_proba(x)
        idx = np.argmax(proba, axis=1)
        none_col = FAULT_CLASSES.index(FaultType.NONE)
        return ClassifierPrediction(
            classes=[FAULT_CLASSES[int(i)] for i in idx],
            confidence=proba[np.arange(proba.shape[0]), idx],
            proba=proba,
            detection_score=1.0 - proba[:, none_col],
        )

    def detection_score(self, x: ArrayLike) -> NDArray[np.float64]:
        """``1 - P(NONE)`` per row, the detection statistic."""
        none_col = FAULT_CLASSES.index(FaultType.NONE)
        return 1.0 - self.predict_proba(x)[:, none_col]

    def feature_importances(self) -> dict[str, float]:
        """Impurity-based feature importances, keyed by feature name.

        Impurity importance is biased toward high-cardinality features and is
        reported here only as a coarse indication of what the model leans on;
        no causal reading should be taken from it.
        """
        self._check_fitted()
        return dict(zip(feature_names(), self.model.feature_importances_, strict=True))
