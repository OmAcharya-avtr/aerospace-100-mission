"""The learned anomaly classifier, and the majority-class guess it must beat.

Scope
-----
The product specification is explicit about what the learned component is for:
*the ambiguous cases the rules leave undecided*. It is trained and scored on
two populations, and both are reported:

1. **Every ``(frame, channel)`` pair**, so that the claim "the rules win on
   every class they cover" is a measurement rather than an assertion.
2. **The ambiguous residual set only** -- the pairs the rules left
   ``undecided`` -- where the specification says the learned model's only
   defensible contribution lies, and where it is scored against a
   majority-class guess so that a useless model cannot look useful.

Architecture
------------
A random forest over the fifteen features of :mod:`telemgap.features`. The
choice is deliberate and is not a performance claim: the features are
low-dimensional, mixed continuous and binary, and the decision structure the
rules encode is a shallow conjunction of thresholds, which is the hypothesis
class a forest represents natively. A gradient-boosted ensemble was not used
because it has no uncertainty output that survives the per-member decomposition
this module reports, and no neural model was used because PyTorch is not
installed in the measured environment and nothing was installed to get it.

Class imbalance, and three declared configurations
--------------------------------------------------
Clean pairs outnumber anomalous ones by about 140 to 1, and how that imbalance
is handled changes the answer far more than anything else about the model. So
it is not chosen once and defended; three configurations are declared, measured
and all reported:

``recall_weighted``
    Clean rows subsampled to :attr:`LearnedClassifier.clean_ratio` times the
    anomaly count and ``class_weight="balanced_subsample"``. Favours recall.
``unweighted``
    No subsampling and no class weighting. Favours precision.
``ambiguous_only``
    Trained on the ambiguous residual rows alone, unweighted. This is the model
    the product specification actually describes -- *a learned anomaly
    classifier over frame-level features for the ambiguous cases the rules
    leave undecided* -- and it is the one whose score against a majority-class
    guess over the same population decides whether the learned component earns
    its place.

**Scoring never subsamples and never reweights**: every reported precision and
recall is measured on the complete held-out population, because a precision
measured on a rebalanced test set is not a precision.

Uncertainty output
------------------
Two numbers per prediction, which are different things and are labelled
differently:

``probability``
    The forest's ``predict_proba`` for the predicted class. It is a vote share,
    and a vote share is only a probability to the extent that it is calibrated.
    Its calibration is measured on an independent third split in
    ``validation/validate_learned.py`` and reported as an expected calibration
    error; it is not assumed and it is not post-hoc corrected.

``dispersion``
    The standard deviation, across the forest's individual trees, of the
    predicted probability of the chosen class. This is an ensemble
    disagreement measure, not a confidence interval -- the trees are not
    independent draws from a posterior. Whether it carries any information
    about correctness is measured (as the area under the ROC curve of
    dispersion against the indicator of being wrong) and reported even when the
    answer is that it carries almost none.

``abstain``
    A declared probability threshold below which the model returns
    ``undecided`` rather than a class. Reported at several thresholds so the
    precision-recall trade is visible rather than chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .features import FEATURE_NAMES

#: Class codes, matching :data:`telemgap.dataset.VALUE_CLASSES`.
CLASS_NAMES = ("clean", "corrupt", "misparsed")

#: Code returned instead of a class when the model abstains.
ABSTAIN = -1

#: Default abstention threshold on the predicted-class vote share.
DEFAULT_ABSTAIN_THRESHOLD = 0.60

#: The three declared configurations, as keyword arguments to
#: :class:`LearnedClassifier`. All three are measured and all three reported.
CONFIGURATIONS: dict[str, dict[str, object]] = {
    "recall_weighted": {
        "subsample": True,
        "clean_ratio": 10.0,
        "class_weight": "balanced_subsample",
    },
    "unweighted": {"subsample": False, "class_weight": None},
    "ambiguous_only": {"subsample": False, "class_weight": None},
}


@dataclass
class Prediction:
    """Predicted classes with both uncertainty outputs.

    Attributes
    ----------
    labels
        ``(n,)`` int64 class codes, or :data:`ABSTAIN` where the model
        abstained.
    probability
        ``(n,)`` vote share of the predicted class, in ``[0, 1]``.
    dispersion
        ``(n,)`` standard deviation across trees of the predicted class's
        probability.
    proba
        ``(n, 3)`` full vote-share matrix.
    abstain_threshold
        The threshold applied, or ``0.0`` when none was.
    """

    labels: np.ndarray
    probability: np.ndarray
    dispersion: np.ndarray
    proba: np.ndarray
    abstain_threshold: float

    @property
    def abstained(self) -> np.ndarray:
        """Boolean mask of the abstained rows."""
        return self.labels == ABSTAIN

    @property
    def abstention_rate(self) -> float:
        """Fraction of rows the model declined to classify."""
        return float(self.abstained.mean()) if self.labels.size else 0.0


@dataclass
class LearnedClassifier:
    """Random forest over :data:`telemgap.features.FEATURE_NAMES`.

    Parameters
    ----------
    n_estimators
        Trees in the forest. The default of 120 is sized to the compute budget
        of this product: training and the per-tree uncertainty decomposition
        together must finish well inside the three-minute limit on two cores,
        and the measured times are recorded in the validation output.
    max_depth, min_samples_leaf
        Capacity controls. Fixed rather than tuned; no hyperparameter search
        was run, and that is stated in MODEL_CARD.md rather than hidden.
    clean_ratio
        Clean training rows retained per anomalous training row. Ignored when
        ``subsample`` is false.
    subsample
        Whether to subsample the clean class at all.
    class_weight
        Passed to ``RandomForestClassifier``. ``None`` means no reweighting.
    seed
        Seed for both the forest and the clean-class subsample.
    """

    n_estimators: int = 120
    max_depth: int = 12
    min_samples_leaf: int = 4
    clean_ratio: float = 10.0
    subsample: bool = True
    class_weight: str | None = "balanced_subsample"
    seed: int = 20261028
    forest: RandomForestClassifier | None = None
    training_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_estimators < 2:
            raise ValueError(f"n_estimators must be at least 2, got {self.n_estimators!r}")
        if self.clean_ratio <= 0.0:
            raise ValueError(f"clean_ratio must be positive, got {self.clean_ratio!r}")
        if self.class_weight not in (None, "balanced", "balanced_subsample"):
            raise ValueError(
                "class_weight must be None, 'balanced' or 'balanced_subsample', got "
                f"{self.class_weight!r}"
            )

    def _subsample(self, labels: np.ndarray) -> np.ndarray:
        """Row indices retained for training: all anomalies, a seeded clean subset."""
        if not self.subsample:
            return np.arange(labels.size)
        rng = np.random.default_rng(self.seed)
        anomalous = np.flatnonzero(labels != 0)
        clean = np.flatnonzero(labels == 0)
        wanted = int(min(clean.size, round(self.clean_ratio * max(anomalous.size, 1))))
        chosen = rng.choice(clean, size=wanted, replace=False) if wanted else np.empty(0, int)
        return np.sort(np.concatenate((anomalous, chosen)))

    def fit(self, features: np.ndarray, labels: np.ndarray) -> LearnedClassifier:
        """Fit the forest. Returns ``self``.

        Raises ``ValueError`` if the feature width does not match
        :data:`telemgap.features.FEATURE_NAMES`, or if fewer than two classes
        are present after subsampling.
        """
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have {len(FEATURE_NAMES)} columns matching "
                f"FEATURE_NAMES, got shape {features.shape}"
            )
        if features.shape[0] != labels.size:
            raise ValueError(
                f"features has {features.shape[0]} rows but labels has {labels.size}"
            )
        rows = self._subsample(labels)
        selected = labels[rows]
        if np.unique(selected).size < 2:
            raise ValueError(
                "training data contains fewer than two classes after subsampling; "
                f"present classes {np.unique(selected).tolist()}"
            )
        self.forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.seed,
            n_jobs=2,
        ).fit(features[rows], selected)
        self.training_counts = {
            CLASS_NAMES[code]: int((selected == code).sum())
            for code in range(len(CLASS_NAMES))
        }
        self.training_counts["rows"] = int(rows.size)
        return self

    def _require_fitted(self) -> RandomForestClassifier:
        if self.forest is None:
            raise RuntimeError("LearnedClassifier.fit must be called before predicting")
        return self.forest

    def predict(
        self, features: np.ndarray, abstain_threshold: float = 0.0
    ) -> Prediction:
        """Predict classes with both uncertainty outputs.

        Parameters
        ----------
        features
            ``(n, 15)`` feature matrix from :func:`telemgap.features.build`.
        abstain_threshold
            Vote share below which :data:`ABSTAIN` is returned. ``0.0``
            disables abstention.

        Notes
        -----
        The per-tree dispersion is accumulated by Welford's online algorithm
        rather than by stacking every tree's probability matrix, because the
        stacked array is 150 times the size of the output for no benefit.
        """
        forest = self._require_fitted()
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have {len(FEATURE_NAMES)} columns, got shape {features.shape}"
            )
        if not 0.0 <= abstain_threshold <= 1.0:
            raise ValueError(
                f"abstain_threshold must lie in [0, 1], got {abstain_threshold!r}"
            )
        classes = np.asarray(forest.classes_, dtype=np.int64)
        proba = np.zeros((features.shape[0], len(CLASS_NAMES)), dtype=np.float64)
        mean = np.zeros((features.shape[0], classes.size), dtype=np.float64)
        m2 = np.zeros_like(mean)
        for count, tree in enumerate(forest.estimators_, start=1):
            member = tree.predict_proba(features)
            delta = member - mean
            mean += delta / count
            m2 += delta * (member - mean)
        variance = m2 / max(len(forest.estimators_) - 1, 1)
        for position, code in enumerate(classes):
            proba[:, int(code)] = mean[:, position]
        chosen = np.argmax(proba, axis=1)
        probability = proba[np.arange(proba.shape[0]), chosen]
        chosen_position = np.searchsorted(classes, chosen)
        chosen_position = np.clip(chosen_position, 0, classes.size - 1)
        dispersion = np.sqrt(
            np.maximum(variance[np.arange(variance.shape[0]), chosen_position], 0.0)
        )
        labels = chosen.astype(np.int64)
        if abstain_threshold > 0.0:
            labels = np.where(probability < abstain_threshold, ABSTAIN, labels)
        return Prediction(
            labels=labels,
            probability=probability,
            dispersion=dispersion,
            proba=proba,
            abstain_threshold=float(abstain_threshold),
        )

    @property
    def feature_importances(self) -> dict[str, float]:
        """Mean decrease in impurity per feature. Reported, not interpreted causally."""
        forest = self._require_fitted()
        return dict(
            zip(
                FEATURE_NAMES,
                (float(value) for value in forest.feature_importances_),
                strict=True,
            )
        )


@dataclass
class MajorityClassifier:
    """The guess a learned model must beat to be worth anything.

    Predicts the most frequent class of its training labels, with a vote share
    equal to that class's training frequency as its confidence output, so that
    it can be scored by exactly the same code path as the forest.
    """

    label: int = 0
    frequency: float = 0.0

    def fit(self, labels: np.ndarray) -> MajorityClassifier:
        """Record the most frequent class of ``labels``. Returns ``self``."""
        labels = np.asarray(labels, dtype=np.int64)
        if labels.size == 0:
            raise ValueError("MajorityClassifier.fit requires at least one label")
        counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        self.label = int(np.argmax(counts))
        self.frequency = float(counts[self.label] / labels.size)
        return self

    def predict(self, features: np.ndarray, abstain_threshold: float = 0.0) -> Prediction:
        """Predict the majority class everywhere, with a constant confidence."""
        n = int(np.asarray(features).shape[0])
        proba = np.zeros((n, len(CLASS_NAMES)), dtype=np.float64)
        proba[:, self.label] = self.frequency
        labels = np.full(n, self.label, dtype=np.int64)
        if abstain_threshold > self.frequency:
            labels = np.full(n, ABSTAIN, dtype=np.int64)
        return Prediction(
            labels=labels,
            probability=np.full(n, self.frequency, dtype=np.float64),
            dispersion=np.zeros(n, dtype=np.float64),
            proba=proba,
            abstain_threshold=float(abstain_threshold),
        )


def expected_calibration_error(
    probability: np.ndarray, correct: np.ndarray, bins: int = 10
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Expected calibration error of a confidence output, with its reliability curve.

    Parameters
    ----------
    probability
        ``(n,)`` confidence in the predicted class.
    correct
        ``(n,)`` boolean, whether that prediction was right.
    bins
        Equal-width bins over ``[0, 1]``.

    Returns
    -------
    (ece, bin_centres, bin_confidence, bin_accuracy)
        ``ece`` is the population-weighted mean absolute gap between mean
        confidence and realised accuracy. Empty bins contribute nothing and
        are returned as ``nan``.
    """
    probability = np.asarray(probability, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    if probability.shape != correct.shape:
        raise ValueError(
            f"probability {probability.shape} and correct {correct.shape} must match"
        )
    if bins < 2:
        raise ValueError(f"bins must be at least 2, got {bins!r}")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    confidence = np.full(bins, np.nan)
    accuracy = np.full(bins, np.nan)
    total = 0.0
    for bin_index in range(bins):
        members = index == bin_index
        if not members.any():
            continue
        confidence[bin_index] = float(probability[members].mean())
        accuracy[bin_index] = float(correct[members].mean())
        total += members.mean() * abs(confidence[bin_index] - accuracy[bin_index])
    return float(total), centres, confidence, accuracy


def dispersion_error_auc(dispersion: np.ndarray, wrong: np.ndarray) -> float:
    """Area under the ROC curve of ``dispersion`` as a predictor of being wrong.

    Computed by the rank-sum identity, so no sklearn metric is involved and the
    result is well defined for ties. Returns ``nan`` when either class is
    empty, which is the honest answer rather than 0.5.
    """
    dispersion = np.asarray(dispersion, dtype=np.float64)
    wrong = np.asarray(wrong, dtype=bool)
    if dispersion.shape != wrong.shape:
        raise ValueError(f"dispersion {dispersion.shape} and wrong {wrong.shape} must match")
    positives = dispersion[wrong]
    negatives = dispersion[~wrong]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate((positives, negatives)), kind="stable")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    combined = np.concatenate((positives, negatives))
    unique, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(unique.size, dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    rank_sum = float(ranks[: positives.size].sum())
    return float(
        (rank_sum - positives.size * (positives.size + 1) / 2.0)
        / (positives.size * negatives.size)
    )
