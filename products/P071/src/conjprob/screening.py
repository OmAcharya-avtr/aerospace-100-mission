"""Learned screening classifier, and the comparison it has to survive.

The operational problem is triage: thousands of screened pairs, a budget for
running the full computation on a few of them, and a strongly asymmetric cost,
because a missed high-Pc event is not the same kind of mistake as a wasted
computation. The right figure of merit is therefore **recall at a fixed
false-positive rate**, not accuracy and not the area under the curve.

This module provides that metric, a random-forest classifier over the cheap
features of :mod:`conjprob.features`, and a comparison harness that scores the
classifier against the analytic screens of :mod:`conjprob.screens` on identical
held-out data at identical operating points.

The classifier exposes an uncertainty output: :meth:`ScreeningClassifier.predict_with_uncertainty`
returns the ensemble mean probability and the standard deviation of the
per-tree probabilities. That standard deviation is a **dispersion measure
across the ensemble, not a confidence interval**; the distinction is the subject
of Wager, S., Hastie, T., and Efron, B. (2014), "Confidence intervals for random
forests: the jackknife and the infinitesimal jackknife", *Journal of Machine
Learning Research* 15, 1625-1651. What is measured here instead is calibration:
the Brier score (Brier, G. W., 1950, "Verification of forecasts expressed in
terms of probability", *Monthly Weather Review* 78(1), 1-3) and a reliability
table of observed frequency against predicted probability.

Model: random forest (Breiman, L., 2001, "Random forests", *Machine Learning*
45(1), 5-32) as implemented by scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier

from .features import FEATURE_NAMES
from .screens import all_screen_scores


@dataclass(frozen=True)
class OperatingPoint:
    """Performance of one score at one false-positive-rate target.

    Attributes
    ----------
    name
        Screen or model name.
    fpr_target
        Requested false-positive rate, dimensionless.
    fpr_achieved
        False-positive rate actually attained at the chosen threshold. It can
        differ from the target because the threshold is taken from the
        empirical distribution of negative scores, which is discrete.
    threshold
        Score threshold; samples with ``score >= threshold`` are flagged.
    recall
        Fraction of positives flagged, dimensionless. The headline number.
    precision
        Fraction of flagged samples that are positive, dimensionless.
    n_positive, n_negative
        Class counts in the evaluation set.
    """

    name: str
    fpr_target: float
    fpr_achieved: float
    threshold: float
    recall: float
    precision: float
    n_positive: int
    n_negative: int


def recall_at_fpr(
    y_true: ArrayLike, scores: ArrayLike, fpr_target: float, *, name: str = "score"
) -> OperatingPoint:
    """Recall at the threshold whose false-positive rate is closest to the target.

    The threshold is the **smallest distinct negative score** whose
    false-positive rate is at or below the target, which maximises recall
    subject to the constraint and makes ``fpr_achieved <= fpr_target`` hold
    exactly, ties included. Using the ``(1 - fpr_target)`` quantile instead
    does not: the quantile index is ``ceil((1 - f)(n - 1))`` while the
    constraint needs ``ceil(n (1 - f))``, and the difference lets the achieved
    rate exceed the target by one sample. A screen with many tied scores can
    end up well below the target, and that is reported in ``fpr_achieved``
    rather than hidden. If no negative score satisfies the constraint, the
    threshold is placed just above the largest negative score, giving a
    false-positive rate of zero.

    Parameters
    ----------
    y_true
        Boolean or 0/1 labels.
    scores
        Scores in which larger means more likely to be positive.
    fpr_target
        Target false-positive rate in ``(0, 1)``.
    name
        Label carried into the returned :class:`OperatingPoint`.

    Raises
    ------
    ValueError
        If either class is empty, the inputs have different lengths, or the
        target is outside ``(0, 1)``.
    """
    labels = np.asarray(y_true).astype(bool).reshape(-1)
    values = np.asarray(scores, dtype=float).reshape(-1)
    if labels.size != values.size:
        raise ValueError(
            f"y_true and scores must have the same length, got {labels.size} and {values.size}"
        )
    if not 0.0 < fpr_target < 1.0:
        raise ValueError(f"fpr_target must lie in (0, 1), got {fpr_target!r}")
    positives = values[labels]
    negatives = values[~labels]
    if positives.size == 0 or negatives.size == 0:
        raise ValueError(
            "both classes must be present; got "
            f"{positives.size} positive and {negatives.size} negative samples"
        )
    ordered = np.sort(negatives)
    distinct = np.unique(ordered)
    rates = (negatives.size - np.searchsorted(ordered, distinct, side="left")) / negatives.size
    acceptable = np.nonzero(rates <= fpr_target)[0]
    if acceptable.size > 0:
        threshold = float(distinct[acceptable[0]])
    else:
        threshold = float(np.nextafter(distinct[-1], np.inf))
    flagged = values >= threshold
    achieved = float(np.mean(negatives >= threshold))
    recall = float(np.mean(positives >= threshold))
    precision = float(np.mean(labels[flagged])) if np.any(flagged) else 0.0
    return OperatingPoint(
        name=name,
        fpr_target=float(fpr_target),
        fpr_achieved=achieved,
        threshold=threshold,
        recall=recall,
        precision=precision,
        n_positive=int(positives.size),
        n_negative=int(negatives.size),
    )


def brier_score(y_true: ArrayLike, probabilities: ArrayLike) -> float:
    """Mean squared error of predicted probabilities, dimensionless in ``[0, 1]``.

    Lower is better. A model that always predicts the base rate ``p`` scores
    ``p (1 - p)``, which is the reference the measured value should be compared
    against.
    """
    labels = np.asarray(y_true).astype(float).reshape(-1)
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if labels.size != probs.size:
        raise ValueError("y_true and probabilities must have the same length")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    return float(np.mean((probs - labels) ** 2))


def reliability_table(
    y_true: ArrayLike, probabilities: ArrayLike, *, n_bins: int = 10
) -> list[tuple[float, float, float, int]]:
    """Observed frequency against predicted probability, one row per bin.

    Returns a list of ``(bin_low, mean_predicted, observed_frequency, count)``.
    Empty bins are omitted. Units: all probabilities dimensionless.
    """
    labels = np.asarray(y_true).astype(float).reshape(-1)
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins!r}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    rows: list[tuple[float, float, float, int]] = []
    for bin_id in range(n_bins):
        mask = index == bin_id
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        rows.append(
            (
                float(edges[bin_id]),
                float(np.mean(probs[mask])),
                float(np.mean(labels[mask])),
                count,
            )
        )
    return rows


class ScreeningClassifier:
    """Random-forest screen over the cheap conjunction features.

    The model predicts whether the Foster-Estes Pc exceeds the declared
    threshold. It is given every feature the analytic screens use, so it is
    never short of information relative to them.

    Parameters
    ----------
    n_estimators
        Trees in the forest. The measured recall at a 1 per cent false-positive
        rate on the nominal regime is 0.99046 at 25 trees, rises to 0.99284 at
        200, is unchanged at 300 and falls to 0.99213 at 400, so the default of
        300 sits on the flat part and is not the maximum of the table by more
        than the noise. The measurement is check 7 of
        ``validation/validate_screening_output.txt``.
    max_depth
        Maximum tree depth, or ``None`` for unlimited.
    min_samples_leaf
        Minimum samples in a leaf. Raised above one to keep the per-tree
        probabilities from collapsing to zero and one, which would make the
        uncertainty output useless.
    random_state
        Seed. Training is deterministic given the seed, the data and the
        scikit-learn version.
    n_jobs
        Worker processes. Default 2, matching the build container.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 4,
        random_state: int = 0,
        n_jobs: int = 2,
    ) -> None:
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be at least 1, got {n_estimators!r}")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be at least 1, got {min_samples_leaf!r}")
        self.forest = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.feature_names: tuple[str, ...] = FEATURE_NAMES
        self._fitted = False

    def fit(self, features: ArrayLike, labels: ArrayLike) -> ScreeningClassifier:
        """Fit the forest. Returns self.

        Raises
        ------
        ValueError
            If the feature matrix has the wrong width or either class is absent.
        """
        matrix = self._check_matrix(features)
        y = np.asarray(labels).astype(int).reshape(-1)
        if y.size != matrix.shape[0]:
            raise ValueError("features and labels must have the same number of rows")
        if np.unique(y).size < 2:
            raise ValueError("training labels must contain both classes")
        self.forest.fit(matrix, y)
        self._fitted = True
        return self

    def _check_matrix(self, features: ArrayLike) -> NDArray[np.float64]:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have shape (n, {len(FEATURE_NAMES)}), got {matrix.shape}"
            )
        return matrix

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("ScreeningClassifier.fit must be called before prediction")

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        """Probability that Pc exceeds the threshold, shape ``(n,)``, dimensionless."""
        self._require_fit()
        matrix = self._check_matrix(features)
        return self.forest.predict_proba(matrix)[:, 1]

    def predict_with_uncertainty(
        self, features: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Ensemble mean probability and per-tree standard deviation.

        Returns ``(mean, dispersion)``, both shape ``(n,)`` and dimensionless.
        ``dispersion`` is the standard deviation of the individual trees'
        predicted probabilities. It is a disagreement measure across the
        ensemble and is **not** a confidence interval; its usefulness is
        established, or not, by the calibration measurement in
        ``validation/validate_screening_output.txt``, where it is shown to rank
        errors.
        """
        self._require_fit()
        matrix = self._check_matrix(features)
        per_tree = np.stack(
            [tree.predict_proba(matrix)[:, 1] for tree in self.forest.estimators_]
        )
        return per_tree.mean(axis=0), per_tree.std(axis=0, ddof=1)

    def score_samples(self, features: ArrayLike) -> NDArray[np.float64]:
        """Screening score, larger meaning more suspicious. Equal to the probability."""
        return self.predict_proba(features)

    def feature_importances(self) -> dict[str, float]:
        """Mean decrease in impurity per feature, dimensionless, summing to one."""
        self._require_fit()
        return dict(
            zip(self.feature_names, map(float, self.forest.feature_importances_), strict=True)
        )


def compare_screens(
    y_true: ArrayLike,
    model_scores: ArrayLike,
    features: ArrayLike,
    fpr_targets: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> dict[str, list[OperatingPoint]]:
    """Score the learned model and every analytic screen at the same operating points.

    Returns a dictionary keyed by ``"learned"`` and by each name in
    :data:`conjprob.screens.SCREEN_NAMES`, each value a list of
    :class:`OperatingPoint` in ``fpr_targets`` order.
    """
    scores = dict(all_screen_scores(features))
    scores["learned"] = np.asarray(model_scores, dtype=float).reshape(-1)
    results: dict[str, list[OperatingPoint]] = {}
    for name, values in scores.items():
        results[name] = [
            recall_at_fpr(y_true, values, target, name=name) for target in fpr_targets
        ]
    return results
