"""The learned screening classifier, and the baselines it has to beat.

The specification this product is built against predicts in writing that the
explicit rule screens of :mod:`tlecheck.verdict` will match or beat this model
on every class they cover, and that its only defensible contribution is the
*silently wrong* class. That prediction is the experiment; this module is one
arm of it. The result is reported either way.

Architecture
------------
``sklearn.ensemble.RandomForestClassifier``, 160 trees, maximum depth 12,
``min_samples_leaf`` 2, ``random_state`` fixed. A tree ensemble is chosen for
three reasons that are about this problem and not about fashion: the features
are a mixture of indicators, bounded angles and heavy-tailed z-scores on wildly
different scales, which trees do not care about; the sample size is a few
hundred objects per split, which rules out anything that needs more; and the
per-member votes give an uncertainty output for free. No neural network is used
and none is available in this environment.

Uncertainty output
------------------
Two numbers, both required by the mission's AI product rules and both reported:

(19) Ensemble dispersion, the standard deviation across member trees of the
     predicted probability of the predicted class:

         u_disp = std_j( p_j(c_hat | x) )

(20) Decision margin, the gap between the two largest class probabilities of
     the ensemble:

         u_margin = p_(1) - p_(2)

     A margin near zero is the model saying it cannot separate two classes for
     this element set, which is information a point prediction destroys.

Baselines
---------
:class:`MajorityClassifier` predicts the training majority class for every
input and reports that class's training prevalence as its probability. Its
operating characteristic is therefore the chance diagonal, which is the point:
a model that cannot beat it has learned nothing, and the specification requires
it to be scored against so that a useless model cannot look useful.

The rule baseline is not in this module. It is :func:`tlecheck.verdict.classify`
and equation (15), it was implemented and validated before this file existed,
and it is the operational default.

Compute budget
--------------
Measured on the two cores this repository was built on: fitting all three
declared configurations on the training split and scoring the test split takes
under ten seconds wall clock. The exact figure is in
``validation/validate_learned_output.txt`` in a table declared unstable. No
training here needs a GPU and none is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .consequence import BENIGN, CLASSES, SILENTLY_WRONG, WILL_FAIL_LOUDLY

RANDOM_STATE = 20261029

CONFIGURATIONS: dict[str, dict[str, object]] = {
    "elements_and_history": {"feature_configuration": "elements_and_history"},
    "elements_only": {"feature_configuration": "elements_only"},
    "with_rule_score": {"feature_configuration": "with_rule_score"},
}

CLASS_ORDER: tuple[str, ...] = (BENIGN, SILENTLY_WRONG, WILL_FAIL_LOUDLY)


@dataclass(frozen=True)
class Prediction:
    """Model output for a batch, with both uncertainty outputs.

    Attributes
    ----------
    labels:
        Predicted class names, shape ``(n,)``.
    probabilities:
        Shape ``(n, 3)``, columns in :data:`CLASS_ORDER`.
    silent_probability:
        The ``silently_wrong`` column, which is the score used for recall at a
        fixed false-positive rate.
    dispersion:
        Equation (19), shape ``(n,)``.
    margin:
        Equation (20), shape ``(n,)``.
    """

    labels: np.ndarray
    probabilities: np.ndarray
    silent_probability: np.ndarray
    dispersion: np.ndarray
    margin: np.ndarray


class LearnedClassifier:
    """Random forest over element and history features.

    Parameters
    ----------
    feature_configuration:
        Name of the declared feature configuration this instance expects. It
        is carried so that a fitted model records which features it was fitted
        on and cannot be given a different matrix by accident.
    trees, max_depth, min_samples_leaf, random_state:
        Architecture. Defaults are the measured configuration.
    class_weight:
        Passed to scikit-learn. ``"balanced"`` is the default because the
        benign class is the majority and an unweighted forest on this corpus
        learns to predict it.
    """

    def __init__(
        self,
        feature_configuration: str = "elements_and_history",
        trees: int = 160,
        max_depth: int = 12,
        min_samples_leaf: int = 2,
        random_state: int = RANDOM_STATE,
        class_weight: str | None = "balanced",
    ) -> None:
        if trees < 1:
            raise ValueError(f"trees must be at least 1, got {trees}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be at least 1, got {max_depth}")
        self.feature_configuration = feature_configuration
        self.forest = RandomForestClassifier(
            n_estimators=trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=1,
        )
        self.classes_: tuple[str, ...] = ()
        self.n_features_: int = 0

    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> LearnedClassifier:
        """Fit on a design matrix and an array of class-name strings.

        Raises
        ------
        ValueError
            If the matrix is not two-dimensional, the lengths disagree, or a
            label is not one of :data:`tlecheck.consequence.CLASSES`.
        """
        matrix = np.asarray(matrix, dtype=float)
        labels = np.asarray(labels)
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be two-dimensional, got shape {matrix.shape}")
        if matrix.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{matrix.shape[0]} rows and {labels.shape[0]} labels supplied"
            )
        unknown = set(np.unique(labels)) - set(CLASSES)
        if unknown:
            raise ValueError(f"labels contain unknown classes {sorted(unknown)}")
        self.forest.fit(matrix, labels)
        self.classes_ = tuple(str(name) for name in self.forest.classes_)
        self.n_features_ = matrix.shape[1]
        return self

    def predict(self, matrix: np.ndarray) -> Prediction:
        """Predict with both uncertainty outputs.

        Raises
        ------
        ValueError
            If the model is unfitted or the column count disagrees with the fit.
        """
        if not self.classes_:
            raise ValueError("the classifier is not fitted")
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_:
            raise ValueError(
                f"matrix has shape {matrix.shape}, expected (n, {self.n_features_})"
            )
        probabilities = np.zeros((matrix.shape[0], len(CLASS_ORDER)), dtype=float)
        ensemble = self.forest.predict_proba(matrix)
        index_of = {name: index for index, name in enumerate(self.classes_)}
        for column, name in enumerate(CLASS_ORDER):
            if name in index_of:
                probabilities[:, column] = ensemble[:, index_of[name]]

        members = np.stack(
            [tree.predict_proba(matrix) for tree in self.forest.estimators_], axis=0
        )
        best = np.argmax(probabilities, axis=1)
        dispersion = np.zeros(matrix.shape[0], dtype=float)
        for row in range(matrix.shape[0]):
            name = CLASS_ORDER[best[row]]
            if name in index_of:
                dispersion[row] = float(np.std(members[:, row, index_of[name]]))

        ordered = np.sort(probabilities, axis=1)
        margin = ordered[:, -1] - ordered[:, -2]
        labels = np.array([CLASS_ORDER[index] for index in best], dtype=object)
        return Prediction(
            labels=labels,
            probabilities=probabilities,
            silent_probability=probabilities[:, CLASS_ORDER.index(SILENTLY_WRONG)],
            dispersion=dispersion,
            margin=margin,
        )


class MajorityClassifier:
    """Predicts the training majority class for every input.

    The probability it reports for that class is its training prevalence, a
    constant, so its operating characteristic over any score threshold is the
    chance diagonal. This is the control that stops a useless model from
    looking useful.
    """

    def __init__(self) -> None:
        self.majority: str = ""
        self.prevalence: dict[str, float] = {}

    def fit(self, labels: np.ndarray) -> MajorityClassifier:
        """Record the majority class and every class prevalence.

        Raises
        ------
        ValueError
            If ``labels`` is empty.
        """
        labels = np.asarray(labels)
        if labels.size == 0:
            raise ValueError("labels must be non-empty")
        names, counts = np.unique(labels, return_counts=True)
        self.majority = str(names[int(np.argmax(counts))])
        self.prevalence = {
            str(name): float(count) / float(labels.size)
            for name, count in zip(names, counts, strict=True)
        }
        return self

    def predict(self, rows: int) -> Prediction:
        """Predict the majority class ``rows`` times.

        Raises
        ------
        ValueError
            If unfitted, or ``rows`` is negative.
        """
        if not self.majority:
            raise ValueError("the majority classifier is not fitted")
        if rows < 0:
            raise ValueError(f"rows must be non-negative, got {rows}")
        probabilities = np.tile(
            np.array([self.prevalence.get(name, 0.0) for name in CLASS_ORDER]), (rows, 1)
        )
        return Prediction(
            labels=np.array([self.majority] * rows, dtype=object),
            probabilities=probabilities,
            silent_probability=probabilities[:, CLASS_ORDER.index(SILENTLY_WRONG)],
            dispersion=np.zeros(rows, dtype=float),
            margin=np.zeros(rows, dtype=float),
        )
