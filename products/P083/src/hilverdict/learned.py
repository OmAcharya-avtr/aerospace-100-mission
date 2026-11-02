"""The learned component, and the narrow role the measurement leaves it.

What it is
----------
A gradient-boosted tree classifier over the cheap log features of
:mod:`hilverdict.features`, trained to predict whether a deadline miss is
consequential. Gradient boosting with the logistic loss: Friedman (2001),
"Greedy function approximation: a gradient boosting machine", *Annals of
Statistics* 29(5), 1189-1232, implemented by
``sklearn.ensemble.GradientBoostingClassifier``. PyTorch is not installed in
the measured environment and is not a dependency of this package.

Why it exists at all
--------------------
Baseline 2 -- direct re-simulation -- is exact and is the reference, and it
costs one plant integration per miss. Baseline 1 -- the analytic certificate --
is a proof and costs essentially nothing. The learned component is for the case
where the certificate leaves a miss undecided and re-simulation is
unaffordable, which is the only case in which a prediction is worth anything
here at all.

Its hyperparameters are declared, not tuned. There is no tuning loop anywhere
in this repository: a search over hyperparameters scored on the same held-out
runs that then report the result would make the reported result a selection
artifact, and the honest alternative to a tuning loop with a third split is
declaring the hyperparameters and reporting what they give.

The asymmetric cost, built in
-----------------------------
Dismissing a consequential miss is not symmetric with re-simulating an
inconsequential one, so the fit upweights the consequential class by a
declared factor and the decision threshold is chosen to reach a declared
recall target rather than to maximise accuracy. Accuracy is not reported
anywhere in this package; it is the wrong number for a problem whose positive
class is a few per cent of the sample.

The uncertainty output
----------------------
:meth:`MissClassifier.predict_with_uncertainty` returns the predicted
probability, the decision at the current threshold, the signed margin to that
threshold, and an **abstention** flag for probabilities inside a declared band
around it. An abstention is not a dismissal: it is a referral to
re-simulation, which is the only honest thing a classifier can do with a miss
it cannot call when the cost of being wrong is asymmetric. Calibration of the
probabilities is measured, not assumed: see
:meth:`MissClassifier.reliability`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .features import FEATURE_NAMES

__all__ = [
    "DEFAULT_SPEC",
    "ClassifierSpec",
    "MissClassifier",
    "Prediction",
    "Reliability",
]


@dataclass(frozen=True)
class ClassifierSpec:
    """Declared hyperparameters. There is no tuning loop in this repository.

    Attributes
    ----------
    n_estimators, learning_rate, max_depth, min_samples_leaf, subsample
        Passed straight to ``GradientBoostingClassifier``.
    positive_weight
        Sample weight applied to the consequential class, reflecting the
        declared asymmetry of the two errors.
    seed
        ``random_state``. Declared, so a refit reproduces.
    abstention_band
        Half-width of the probability band around the decision threshold
        inside which the model abstains and refers the miss to re-simulation.
    """

    n_estimators: int = 150
    learning_rate: float = 0.08
    max_depth: int = 3
    min_samples_leaf: int = 4
    subsample: float = 0.9
    positive_weight: float = 8.0
    seed: int = 20260301
    abstention_band: float = 0.05

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be at least 1, got {self.n_estimators}")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError(
                f"learning_rate must lie in (0, 1], got {self.learning_rate}"
            )
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be at least 1, got {self.max_depth}")
        if self.positive_weight <= 0.0:
            raise ValueError(
                f"positive_weight must be positive, got {self.positive_weight}"
            )
        if not 0.0 <= self.abstention_band < 0.5:
            raise ValueError(
                f"abstention_band must lie in [0, 0.5), got {self.abstention_band}"
            )


#: The declared specification every reported number uses.
DEFAULT_SPEC = ClassifierSpec()


@dataclass(frozen=True)
class Prediction:
    """The model's output for a batch of misses.

    Attributes
    ----------
    probability
        Predicted probability of being consequential, in [0, 1].
    flagged
        Decision at the model's current threshold: True means do not dismiss.
    margin
        ``probability - threshold``. Signed, so its magnitude is how far from
        the decision the model was.
    abstained
        True where ``|margin|`` is inside the declared abstention band. An
        abstention is a referral to re-simulation, never a dismissal.
    threshold
        The threshold the decision used.
    """

    probability: np.ndarray
    flagged: np.ndarray
    margin: np.ndarray
    abstained: np.ndarray
    threshold: float

    @property
    def referred(self) -> np.ndarray:
        """Misses that must be re-simulated: flagged or abstained."""
        return self.flagged | self.abstained


@dataclass(frozen=True)
class Reliability:
    """Measured calibration of the predicted probabilities.

    Attributes
    ----------
    edges
        Bin edges in probability.
    predicted
        Mean predicted probability per non-empty bin.
    observed
        Observed consequential rate per non-empty bin.
    counts
        Number of misses per non-empty bin.
    brier
        Brier score over the whole sample: mean squared difference between
        probability and outcome. Lower is better; the all-base-rate predictor
        scores ``p(1-p)``.
    """

    edges: np.ndarray
    predicted: np.ndarray
    observed: np.ndarray
    counts: np.ndarray
    brier: float

    @property
    def worst_gap(self) -> float:
        """Largest absolute difference between predicted and observed, over bins."""
        if self.predicted.size == 0:
            return float("nan")
        return float(np.max(np.abs(self.predicted - self.observed)))


@dataclass
class MissClassifier:
    """A fitted classifier over the cheap log features.

    The class is mutable because fitting mutates it; every number it reports is
    a function of the declared spec and the data it was given.
    """

    spec: ClassifierSpec = DEFAULT_SPEC
    threshold: float = 0.5
    _model: GradientBoostingClassifier | None = field(default=None, repr=False)
    _trained_on: int = field(default=0, repr=False)

    def fit(self, features: np.ndarray, consequential: np.ndarray) -> MissClassifier:
        """Fit on ``features`` and boolean labels ``consequential``.

        Raises when either class is absent: a classifier fitted on one class
        predicts that class with probability 1 and would be reported as
        achieving a recall of 1 or 0 for reasons that have nothing to do with
        the problem.
        """
        matrix = np.asarray(features, dtype=float)
        labels = np.asarray(consequential).astype(int)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have shape (n, {len(FEATURE_NAMES)}), got "
                f"{matrix.shape}"
            )
        if labels.shape[0] != matrix.shape[0]:
            raise ValueError(
                f"got {matrix.shape[0]} feature rows and {labels.shape[0]} labels"
            )
        present = set(np.unique(labels).tolist())
        if present != {0, 1}:
            raise ValueError(
                "the training set must contain both consequential and "
                f"inconsequential misses; it contains only {sorted(present)}"
            )
        weights = np.where(labels == 1, self.spec.positive_weight, 1.0)
        model = GradientBoostingClassifier(
            n_estimators=self.spec.n_estimators,
            learning_rate=self.spec.learning_rate,
            max_depth=self.spec.max_depth,
            min_samples_leaf=self.spec.min_samples_leaf,
            subsample=self.spec.subsample,
            random_state=self.spec.seed,
        )
        model.fit(matrix, labels, sample_weight=weights)
        self._model = model
        self._trained_on = int(matrix.shape[0])
        return self

    @property
    def fitted(self) -> bool:
        """True once :meth:`fit` has run."""
        return self._model is not None

    @property
    def trained_on(self) -> int:
        """Number of misses the model was fitted on."""
        return self._trained_on

    def _require_fitted(self) -> GradientBoostingClassifier:
        if self._model is None:
            raise ValueError(
                "the classifier has not been fitted; call fit(features, labels) first"
            )
        return self._model

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Probability of being consequential, shape ``(n,)``."""
        model = self._require_fitted()
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have shape (n, {len(FEATURE_NAMES)}), got "
                f"{matrix.shape}"
            )
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)

    def predict_with_uncertainty(self, features: np.ndarray) -> Prediction:
        """Probability, decision, margin and abstention. The model's only interface.

        There is no point-prediction method on this class on purpose: a bare
        label would hide the margin, and the margin is the only thing that
        makes an asymmetric decision rule possible.
        """
        probability = self.predict_proba(features)
        margin = probability - self.threshold
        return Prediction(
            probability=probability,
            flagged=probability >= self.threshold,
            margin=margin,
            abstained=np.abs(margin) < self.spec.abstention_band,
            threshold=float(self.threshold),
        )

    def set_threshold(self, value: float) -> MissClassifier:
        """Set the decision threshold, in probability units."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold must lie in [0, 1], got {value}")
        self.threshold = float(value)
        return self

    @property
    def feature_importance(self) -> dict[str, float]:
        """Impurity-based importance per feature; sums to 1 over the features.

        Impurity importance is biased towards high-cardinality features and is
        reported as a description of the fitted model, never as evidence about
        the plant.
        """
        model = self._require_fitted()
        return dict(
            zip(FEATURE_NAMES, (float(v) for v in model.feature_importances_), strict=True)
        )

    def reliability(
        self, features: np.ndarray, consequential: np.ndarray, bins: int = 5
    ) -> Reliability:
        """Measured calibration of the probabilities on held-out misses."""
        if bins < 2:
            raise ValueError(f"bins must be at least 2, got {bins}")
        probability = self.predict_proba(features)
        truth = np.asarray(consequential).astype(float)
        if truth.shape != probability.shape:
            raise ValueError(
                f"got {probability.shape[0]} predictions and {truth.shape[0]} labels"
            )
        edges = np.linspace(0.0, 1.0, bins + 1)
        which = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
        predicted: list[float] = []
        observed: list[float] = []
        counts: list[int] = []
        for index in range(bins):
            mask = which == index
            if not np.any(mask):
                continue
            predicted.append(float(np.mean(probability[mask])))
            observed.append(float(np.mean(truth[mask])))
            counts.append(int(np.count_nonzero(mask)))
        return Reliability(
            edges=edges,
            predicted=np.array(predicted, dtype=float),
            observed=np.array(observed, dtype=float),
            counts=np.array(counts, dtype=int),
            brier=float(np.mean((probability - truth) ** 2)),
        )
