"""The learned fault classifier with calibrated confidence. Implemented LAST.

Written after the threshold-logic baseline in ``baseline.py`` and after the
whole deterministic core, in that order, as the mission's AI rule requires.
The model exists to answer one question the baseline cannot: what is the
probability that this observation vector came from each declared fault, well
enough calibrated that a policy can act on the number.

Architecture, with the reason for each choice:

- ``StandardScaler`` then multinomial logistic regression (scikit-learn
  ``LogisticRegression``, lbfgs, L2). The generator is an isotropic Gaussian
  per class with a shared covariance, so the Bayes-optimal decision boundary
  between any two classes is linear and the Bayes-optimal posterior is exactly
  multinomial-logistic. A linear model is therefore the correctly specified
  choice, not a simplification; a tree ensemble would approximate the same
  boundary with axis-aligned steps.
- ``CalibratedClassifierCV`` with isotonic regression on a cross-validated
  split, as an optional wrapper. Isotonic calibration is Zadrozny and Elkan,
  "Transforming Classifier Scores into Accurate Multiclass Probability
  Estimates", KDD 2002; the sigmoid alternative is Platt, "Probabilistic
  Outputs for Support Vector Machines and Comparisons to Regularized
  Likelihood Methods", Advances in Large Margin Classifiers, MIT Press, 1999.
  Both are exposed because which one wins is an empirical question this package
  measures rather than assumes.

Confidence output: ``predict_proba`` gives the full posterior and
``confidence`` gives the top-label probability. Both are dimensionless in
[0, 1]. The policy in ``policy.py`` consumes the top-label probability.

Validity range and the ceiling that matters: two faults in the same declared
ambiguity class have identical observation distributions by construction (see
``data.py``), so the top-label posterior for a member of a class of size ``m``
cannot exceed ``1/m`` plus sampling noise, and no amount of model capacity
changes that. The accuracy ceiling of ANY classifier on this problem is
therefore below 1, and the benchmark reports the ceiling alongside the measured
accuracy so the gap to it can be read directly.

Compute: one fit on 7 features and a few thousand rows, under a second on two
cores. No GPU, no PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import NOMINAL_LABEL, ObservationSet
from .spec import SafeModeSpec

__all__ = ["LearnedFaultClassifier", "accuracy_ceiling"]


def accuracy_ceiling(spec: SafeModeSpec, observations: ObservationSet) -> float:
    """Best accuracy any classifier can reach on this declaration's data.

    Within a declared ambiguity class of size ``m`` the observation
    distributions are identical, so the best possible rule gets ``1/m`` of
    those samples right. Outside the classes the separation is finite but the
    ceiling here ignores overlap between distinct signatures, so the returned
    value is an UPPER bound on achievable accuracy and not a tight one. It is
    the ceiling imposed by the declared observability model alone.

    Returns:
        Dimensionless in (0, 1].
    """
    counts = observations.counts()
    total = sum(counts.values())
    if total == 0:
        raise ValueError("no observations")
    class_size: dict[str, int] = {}
    for group in spec.observability.indistinguishable_groups():
        for fault_id in group:
            class_size[fault_id] = len(group)
    best = 0.0
    for label, count in counts.items():
        best += count / class_size.get(label, 1)
    return float(best / total)


@dataclass
class LearnedFaultClassifier:
    """Multinomial logistic fault classifier with an optional calibration wrapper.

    Attributes:
        calibration: ``"none"``, ``"isotonic"`` or ``"sigmoid"``.
        regularisation: inverse L2 strength ``C`` of the logistic regression,
            > 0, dimensionless.
        cv: cross-validation folds used by the calibration wrapper, >= 2.
        random_state: seed for the estimator's own randomness.
    """

    calibration: str = "isotonic"
    regularisation: float = 1.0
    cv: int = 5
    random_state: int = 0
    _pipeline: Pipeline | CalibratedClassifierCV | None = field(default=None, init=False)
    _classes: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if self.calibration not in ("none", "isotonic", "sigmoid"):
            raise ValueError(
                f"calibration must be 'none', 'isotonic' or 'sigmoid', got {self.calibration!r}"
            )
        if isinstance(self.regularisation, bool) or not isinstance(
            self.regularisation, int | float
        ):
            raise TypeError("regularisation must be a real number")
        if self.regularisation <= 0.0:
            raise ValueError(f"regularisation must be > 0, got {self.regularisation}")
        if self.cv < 2:
            raise ValueError(f"cv must be >= 2, got {self.cv}")

    @property
    def classes(self) -> tuple[str, ...]:
        """Class labels in ``predict_proba`` column order. Empty before fitting."""
        return self._classes

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` has run."""
        return self._pipeline is not None

    def _base(self) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=float(self.regularisation),
                        max_iter=2000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> LearnedFaultClassifier:
        """Fit on ``(n, n_channels)`` features and ``(n,)`` string labels.

        Raises:
            ValueError: fewer than two distinct labels, or a class with fewer
                members than ``cv`` when a calibration wrapper is requested.
        """
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels).astype(str)
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {features.shape}")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels have different lengths")
        unique, counts = np.unique(labels, return_counts=True)
        if unique.size < 2:
            raise ValueError(f"need at least two classes, got {unique.size}")
        if self.calibration != "none" and counts.min() < self.cv:
            raise ValueError(
                f"class {unique[int(counts.argmin())]!r} has {int(counts.min())} samples, "
                f"fewer than cv={self.cv}; lower cv or pass calibration='none'"
            )
        estimator: Pipeline | CalibratedClassifierCV = self._base()
        if self.calibration != "none":
            estimator = CalibratedClassifierCV(
                estimator, method=self.calibration, cv=self.cv, ensemble=True
            )
        estimator.fit(features, labels)
        self._pipeline = estimator
        self._classes = tuple(str(c) for c in estimator.classes_)
        return self

    def _require_fitted(self) -> Pipeline | CalibratedClassifierCV:
        if self._pipeline is None:
            raise RuntimeError("the classifier is not fitted; call fit() first")
        return self._pipeline

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Posterior over :attr:`classes`, shape ``(n, K)``, rows summing to 1."""
        estimator = self._require_fitted()
        features = np.atleast_2d(np.asarray(features, dtype=float))
        return np.asarray(estimator.predict_proba(features), dtype=float)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Top-label prediction per row, shape ``(n,)``, dtype object."""
        proba = self.predict_proba(features)
        return np.array(
            [self._classes[int(index)] for index in proba.argmax(axis=1)], dtype=object
        )

    def confidence(self, features: np.ndarray) -> np.ndarray:
        """Top-label probability per row, dimensionless in [0, 1]."""
        return self.predict_proba(features).max(axis=1)

    def predict_with_confidence(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Labels and top-label probabilities in one pass."""
        proba = self.predict_proba(features)
        index = proba.argmax(axis=1)
        return (
            np.array([self._classes[int(i)] for i in index], dtype=object),
            proba[np.arange(proba.shape[0]), index],
        )

    def fault_probability(self, features: np.ndarray) -> np.ndarray:
        """Total posterior mass on the fault classes, dimensionless in [0, 1].

        One minus the posterior on the no-fault class. This is the quantity a
        detection threshold is swept over when matching a false-alarm rate.
        """
        proba = self.predict_proba(features)
        if NOMINAL_LABEL not in self._classes:
            return np.ones(proba.shape[0])
        column = self._classes.index(NOMINAL_LABEL)
        return 1.0 - proba[:, column]
