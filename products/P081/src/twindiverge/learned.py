"""The learned residual-signature classifier, and its confidence.

What it is
----------
A gradient-boosted decision-tree classifier (Friedman 2001, "Greedy function
approximation: a gradient boosting machine", Annals of Statistics 29(5),
1189-1232, as implemented by ``sklearn.ensemble.GradientBoostingClassifier``)
over the sixteen residual-signature features of :mod:`twindiverge.features`. It
maps a residual time series to a responsible parameter, which is the mapping
the specification names.

**There is no point-prediction method on this class.** Every prediction comes
back as a :class:`Prediction` carrying the full class-probability vector, the
chosen label, the confidence -- the largest class probability -- and an
abstention flag set when that confidence falls below the declared threshold.
A caller who wants only a label has to take it out of an object that is also
telling it how sure the model is.

Two declared variants
---------------------
``variant="parameter"`` (five classes)
    The specified architecture: the five parameters, and nothing else. It has
    no way to say "the telemetry cannot tell", and what it does instead inside
    the unidentifiable subspace is the headline measurement of this product.

``variant="with_refusal"`` (six classes)
    An ablation. The same classifier with a sixth class for "the change lies in
    the unidentifiable subspace", trained with ``scale`` cases included. It
    exists to answer a fair objection to the headline -- that the five-class
    model was never given the option of refusing -- and to test whether having
    the option generalises. It does not generalise automatically: a refusal
    class trained on pure-scale cases is tested in
    ``validation/validate_learned_model.py`` on the ``equivalent`` pairs, where
    the residual is large and the ambiguity is in the attribution rather than
    in the detection, and the measured result is reported there.

What the confidence is and is not
---------------------------------
It is the largest of the class probabilities the boosted ensemble produces by
the multinomial logistic transform of its raw scores. It is a model output, not
a probability derived from the physics: nothing in it knows the rank of the
sensitivity matrix. Its calibration is measured, in :mod:`twindiverge.scoring`,
with a Brier score and a reliability table, separately for each case
population, and the measured result inside the unidentifiable subspace is that
it stays high where the analytic path refuses. That contrast is the point.

Compute budget
--------------
Declared hyperparameters: 150 boosting stages, depth 3, learning rate 0.1, no
subsampling, fixed ``random_state``. On the measured environment (2 cores) a
fit on 600 cases of the ``full`` campaign takes a few seconds and the whole
learned study in ``validation/validate_learned_model.py`` is reported with its
wall-clock time in its own output file. Nothing here is near the three-minute
budget the build guide sets.

Reproducibility
---------------
``random_state`` is declared and fixed, the feature matrix is a deterministic
function of the case seeds, and the class order is fixed by
:attr:`ResidualClassifier.labels` rather than taken from the order the labels
happen to appear in the training set. No model artifact is serialised or
committed: every number in this repository comes from a model refitted in the
script that reports it, so there is no pickle whose format would need a
dependency bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .plant import PARAMETER_NAMES
from .synthetic import UNIDENTIFIABLE

__all__ = [
    "ABSTAIN_THRESHOLD",
    "LEARNING_RATE",
    "MAX_DEPTH",
    "RANDOM_STATE",
    "STAGES",
    "VARIANTS",
    "Prediction",
    "ResidualClassifier",
]

#: Declared number of boosting stages.
STAGES: Final[int] = 150

#: Declared tree depth.
MAX_DEPTH: Final[int] = 3

#: Declared learning rate.
LEARNING_RATE: Final[float] = 0.1

#: Declared random state. Fitting is deterministic given this value within one
#: scikit-learn minor version; it is not part of scikit-learn's public API
#: contract across versions, which is why ``pyproject.toml`` bounds the
#: dependency at the minor version.
RANDOM_STATE: Final[int] = 20261109

#: Declared confidence below which a prediction is flagged as an abstention.
#: It is a flag and not a refusal: the label is still reported, because a
#: classifier that silently dropped its answer would be unmeasurable.
ABSTAIN_THRESHOLD: Final[float] = 0.5

#: The two declared variants. See the module docstring.
VARIANTS: Final[tuple[str, str]] = ("parameter", "with_refusal")


@dataclass(frozen=True)
class Prediction:
    """One prediction: the whole probability vector, not a label.

    ``probabilities`` is indexed by :attr:`ResidualClassifier.labels`.
    """

    labels: tuple[int, ...]
    probabilities: np.ndarray
    label: int
    confidence: float

    @property
    def abstains(self) -> bool:
        """Whether the confidence falls below :data:`ABSTAIN_THRESHOLD`.

        A thresholded boolean. Tests assert the confidence to tolerance and
        allow this flag to differ only when the confidence sits within
        tolerance of the threshold.
        """
        return self.confidence < ABSTAIN_THRESHOLD

    @property
    def label_name(self) -> str:
        """The chosen label as it is printed in reports."""
        if self.label == UNIDENTIFIABLE:
            return "UNIDENTIFIABLE"
        return PARAMETER_NAMES[self.label]

    @property
    def margin(self) -> float:
        """Gap between the largest and the second largest class probability.

        Reported beside the confidence because a 0.6 against 0.1 and a 0.6
        against 0.39 are different states of knowledge and the maximum alone
        does not distinguish them.
        """
        ordered = np.sort(self.probabilities)[::-1]
        return float(ordered[0] - ordered[1]) if ordered.size > 1 else float(ordered[0])


class ResidualClassifier:
    """Gradient-boosted classifier over residual-signature features.

    ``labels`` is fixed at construction from ``variant``, not inferred from the
    training set, so a training population that happens to contain no example
    of one class still produces a probability column for it and the column
    order of :meth:`predict_probabilities` is stable across runs.
    """

    def __init__(
        self,
        variant: str = "parameter",
        stages: int = STAGES,
        max_depth: int = MAX_DEPTH,
        learning_rate: float = LEARNING_RATE,
        random_state: int = RANDOM_STATE,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
        if stages < 1:
            raise ValueError(f"stages must be at least 1, got {stages!r}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be at least 1, got {max_depth!r}")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError(f"learning_rate must lie in (0, 1], got {learning_rate!r}")
        self.variant = variant
        self.labels: tuple[int, ...] = tuple(range(len(PARAMETER_NAMES)))
        if variant == "with_refusal":
            self.labels = (*self.labels, UNIDENTIFIABLE)
        self._model = GradientBoostingClassifier(
            n_estimators=stages,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self._fitted = False
        self._seen: tuple[int, ...] = ()

    def fit(self, features: np.ndarray, labels: np.ndarray) -> ResidualClassifier:
        """Fit on a feature matrix of shape ``(cases, 16)`` and integer labels."""
        matrix = np.atleast_2d(np.asarray(features, dtype=float))
        targets = np.asarray(labels, dtype=int)
        if targets.shape != (matrix.shape[0],):
            raise ValueError(
                f"{matrix.shape[0]} feature rows but {targets.shape} labels"
            )
        unknown = sorted(set(targets.tolist()) - set(self.labels))
        if unknown:
            raise ValueError(
                f"variant {self.variant!r} has label set {self.labels} and the "
                f"training data contains {unknown}; a 'parameter' variant cannot be "
                "trained on unidentifiable-subspace cases, which is the point of the "
                "'with_refusal' variant"
            )
        if len(set(targets.tolist())) < 2:
            raise ValueError(
                "at least two distinct labels are needed to fit a classifier; got "
                f"{sorted(set(targets.tolist()))}"
            )
        self._model.fit(matrix, targets)
        self._seen = tuple(int(value) for value in self._model.classes_)
        self._fitted = True
        return self

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Class probabilities in :attr:`labels` order, shape ``(cases, len(labels))``.

        A class absent from the training set gets a column of exact zeros, and
        the remaining columns are renormalised, so the rows always sum to 1.
        """
        if not self._fitted:
            raise ValueError("the classifier has not been fitted")
        matrix = np.atleast_2d(np.asarray(features, dtype=float))
        raw = self._model.predict_proba(matrix)
        output = np.zeros((matrix.shape[0], len(self.labels)))
        for column, label in enumerate(self._seen):
            output[:, self.labels.index(label)] = raw[:, column]
        totals = output.sum(axis=1, keepdims=True)
        return output / np.where(totals > 0.0, totals, 1.0)

    def predict(self, features: np.ndarray) -> tuple[Prediction, ...]:
        """Predictions for a stack of feature rows.

        There is deliberately no method that returns bare labels. See the
        module docstring.
        """
        probabilities = self.predict_probabilities(features)
        predictions = []
        for row in probabilities:
            index = int(np.argmax(row))
            predictions.append(
                Prediction(
                    labels=self.labels,
                    probabilities=row.copy(),
                    label=self.labels[index],
                    confidence=float(row[index]),
                )
            )
        return tuple(predictions)

    def predict_one(self, features: np.ndarray) -> Prediction:
        """Prediction for a single feature row of shape ``(16,)``."""
        array = np.asarray(features, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"expected a single feature row, got shape {array.shape}")
        return self.predict(array[None, :])[0]

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._fitted

    @property
    def feature_importances(self) -> np.ndarray:
        """Impurity-based feature importances, shape ``(16,)``.

        Impurity-based importances are biased towards high-cardinality
        features and are reported here only as a description of what the
        fitted trees used, never as evidence about the physics. The physics is
        baseline 1's business.
        """
        if not self._fitted:
            raise ValueError("the classifier has not been fitted")
        return np.asarray(self._model.feature_importances_, dtype=float)
