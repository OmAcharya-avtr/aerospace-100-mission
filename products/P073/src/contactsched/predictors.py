"""Conflict prediction: which requests will be starved, before scheduling.

The task
--------
Binary classification over requests. Positive means the exact method, with no
fairness constraints, will deliver fewer contacts than the request declared it
needs. A planner who knows this at submission time can renegotiate -- shorten
the contact, widen the allowed antennas, lower the minimum count -- instead of
discovering a silent drop after the schedule is published.

The baselines come first, and they are built to win
---------------------------------------------------
Three contention ratios are available before scheduling: ``ratio_max``,
``ratio_weighted`` and ``demand_over_usable`` (see
:mod:`contactsched.contention`). Each is used two ways:

* :class:`RatioRankingBaseline` ranks requests by the raw ratio with no fitting
  at all. It can be measured by area under the ROC curve and by average
  precision, which are rank statistics, and it uses no training data.
* :class:`RatioLogisticBaseline` fits a one-feature logistic regression on the
  ratio, so it emits a probability and can be measured by Brier score and
  calibration alongside the learned model. This is the *strong* form of the
  baseline and it is the one the learned model has to beat to claim anything.

A monotone one-feature fit cannot change the ranking, so the two agree exactly
on the rank statistics and differ only on the calibration ones; that identity is
asserted in ``tests/test_predictors.py`` rather than assumed.

Which ratio is the baseline was decided by measurement, not by reading the
specification. The specification names a demand-over-supply ratio *per
station-window*. Measured on 725 held-out requests from 80 instances, that ratio
is the weakest of the three: ``ratio_max`` scores 0.6010 and ``ratio_weighted``
0.6228 area under the ROC curve, and on a smaller 108-row split both fall below
0.5, which is anti-correlation and not merely weakness. The per-request
``demand_over_usable`` ratio scores 0.9230 on the same held-out rows. The per-bin
ratio aggregates supply that is abundant in total with competition that is
pairwise: a request that loses its only window to one higher-priority neighbour
sits in a bin whose total supply looks ample. ``demand_over_usable`` is therefore
the default baseline feature here and the one the learned model has to beat,
because taking the weaker ratio as the baseline would have inflated the learned
margin. All three are reported in
``validation/validate_predictors_output.txt``, including the per-bin ratios'
failure. This is the build guide's instruction to validate the baseline before
benchmarking the learned model, and it changed the published comparison.

The learned model
-----------------
:class:`LearnedConflictPredictor` is a
:class:`sklearn.ensemble.HistGradientBoostingClassifier` over the whole feature
set. Its ``predict_proba`` output is the confidence this package exposes, and
that confidence is *measured*, not asserted: :func:`reliability_table` bins the
predictions and reports the empirical positive rate in each bin, and
:func:`expected_calibration_error` reduces that table to one number. Where the
model is miscalibrated, the validation output says so.

Reproducibility
---------------
Every estimator takes an explicit ``random_state``. The scikit-learn version is
bounded on both sides in ``pyproject.toml`` because these are properties of
fitted trees, not of a documented API, and a minor release that changes the
binning or the split order moves them without failing a test.

References
----------
Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision
tree. *NeurIPS 30*. The histogram-based boosting algorithm scikit-learn's
``HistGradientBoostingClassifier`` implements.

Brier, G. W. (1950). Verification of forecasts expressed in terms of
probability. *Monthly Weather Review* 78(1), 1-3.

Niculescu-Mizil, A. and Caruana, R. (2005). Predicting good probabilities with
supervised learning. *ICML 22*. Why a boosted ensemble's raw scores are
typically over-confident at the extremes, which is what the reliability table
here is checking for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import FEATURE_NAMES


@dataclass
class RatioRankingBaseline:
    """Rank requests by one contention ratio. No fitting, no parameters.

    Attributes
    ----------
    feature : str
        Name of the feature to rank on; must be in
        :data:`contactsched.features.FEATURE_NAMES`.

    Notes
    -----
    :meth:`predict_proba` returns the ratio squashed into ``[0, 1]`` by a fixed
    monotone map, ``r / (1 + r)``. That preserves the ranking exactly, so the
    rank statistics are unaffected, and it is **not** a calibrated probability.
    Its Brier score is reported anyway, labelled as uncalibrated, because a
    reader comparing Brier scores needs to see what an uncalibrated score costs.
    """

    feature: str = "demand_over_usable"

    def __post_init__(self) -> None:
        if self.feature not in FEATURE_NAMES:
            raise ValueError(f"unknown feature {self.feature!r}; expected one of {FEATURE_NAMES}")
        self._column = FEATURE_NAMES.index(self.feature)

    @property
    def name(self) -> str:
        """Label used in result tables."""
        return f"ratio-ranking[{self.feature}]"

    def fit(self, x: np.ndarray, y: np.ndarray) -> RatioRankingBaseline:
        """No-op, present so every predictor has the same interface."""
        del x, y
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """Raw ratio, higher meaning more likely to be starved."""
        return np.asarray(x, dtype=float)[:, self._column]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Uncalibrated pseudo-probability ``r / (1 + r)``, shape ``(n,)``."""
        raw = np.clip(self.score(x), 0.0, None)
        return raw / (1.0 + raw)


@dataclass
class RatioLogisticBaseline:
    """One-feature logistic regression on a contention ratio.

    This is the baseline the learned model must beat. It is fitted on the same
    training instances, emits a probability, and has one coefficient and one
    intercept.

    Attributes
    ----------
    feature : str
        Feature to regress on.
    random_state : int
        Passed to the solver for determinism.
    """

    feature: str = "demand_over_usable"
    random_state: int = 0
    _model: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.feature not in FEATURE_NAMES:
            raise ValueError(f"unknown feature {self.feature!r}; expected one of {FEATURE_NAMES}")
        self._column = FEATURE_NAMES.index(self.feature)

    @property
    def name(self) -> str:
        """Label used in result tables."""
        return f"ratio-logistic[{self.feature}]"

    def fit(self, x: np.ndarray, y: np.ndarray) -> RatioLogisticBaseline:
        """Fit the single coefficient on the training rows."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        column = np.asarray(x, dtype=float)[:, [self._column]]
        self._model = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=int(self.random_state), max_iter=1000),
        ).fit(column, np.asarray(y, dtype=int))
        return self

    def _require_fitted(self) -> object:
        if self._model is None:
            raise RuntimeError(f"{self.name} has not been fitted; call fit first")
        return self._model

    def score(self, x: np.ndarray) -> np.ndarray:
        """Predicted probability, which is also the ranking score."""
        return self.predict_proba(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Probability of being starved, shape ``(n,)``."""
        model = self._require_fitted()
        column = np.asarray(x, dtype=float)[:, [self._column]]
        return np.asarray(model.predict_proba(column))[:, 1]


@dataclass
class LearnedConflictPredictor:
    """Gradient-boosted classifier over the full pre-scheduling feature set.

    Attributes
    ----------
    max_iter : int
        Boosting iterations. Kept at 200 for the two-core compute budget.
    learning_rate : float
        Shrinkage.
    max_leaf_nodes : int
        Tree size.
    min_samples_leaf : int
        Leaf size floor, which is the main regulariser here given the dataset is
        a few thousand rows.
    l2_regularization : float
        Leaf-value penalty.
    random_state : int
        Seed.
    """

    max_iter: int = 200
    learning_rate: float = 0.06
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 25
    l2_regularization: float = 1.0
    random_state: int = 0
    _model: object | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        """Label used in result tables."""
        return "learned-hgb"

    def fit(self, x: np.ndarray, y: np.ndarray) -> LearnedConflictPredictor:
        """Fit on the training design matrix."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        self._model = HistGradientBoostingClassifier(
            max_iter=int(self.max_iter),
            learning_rate=float(self.learning_rate),
            max_leaf_nodes=int(self.max_leaf_nodes),
            min_samples_leaf=int(self.min_samples_leaf),
            l2_regularization=float(self.l2_regularization),
            early_stopping=False,
            random_state=int(self.random_state),
        ).fit(np.asarray(x, dtype=float), np.asarray(y, dtype=int))
        return self

    def _require_fitted(self) -> object:
        if self._model is None:
            raise RuntimeError("learned-hgb has not been fitted; call fit first")
        return self._model

    def score(self, x: np.ndarray) -> np.ndarray:
        """Predicted probability, which is also the ranking score."""
        return self.predict_proba(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Probability of being starved, shape ``(n,)``. This is the confidence output."""
        model = self._require_fitted()
        return np.asarray(model.predict_proba(np.asarray(x, dtype=float)))[:, 1]


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve. ``nan`` when one class is absent."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true, dtype=int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision, the area under the precision-recall curve."""
    from sklearn.metrics import average_precision_score

    y = np.asarray(y_true, dtype=int)
    if y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, np.asarray(scores, dtype=float)))


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error of the probability forecast, dimensionless.

    Lower is better. This is only meaningful for a predictor that claims to emit
    probabilities; for :class:`RatioRankingBaseline` it is reported and labelled
    uncalibrated.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: labels {y.shape}, probabilities {p.shape}")
    return float(np.mean((p - y) ** 2))


def reliability_table(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> list[tuple[float, float, float, int]]:
    """Calibration table: predicted against observed, per probability bin.

    Returns
    -------
    list of tuple
        ``(bin_low, mean_predicted, observed_rate, count)`` for every non-empty
        bin, with equal-width bins on ``[0, 1]``.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows: list[tuple[float, float, float, int]] = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        if high >= 1.0:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append((float(low), float(p[mask].mean()), float(y[mask].mean()), count))
    return rows


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Count-weighted mean absolute gap between predicted and observed rates.

    Zero means perfectly calibrated on this binning. The value depends on the
    number of bins, so the bin count is always reported next to it.
    """
    table = reliability_table(y_true, probabilities, bins)
    total = sum(row[3] for row in table)
    if total == 0:
        return float("nan")
    return float(sum(abs(row[1] - row[2]) * row[3] for row in table) / total)


def threshold_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Precision, recall, F1 and accuracy at a fixed decision threshold.

    Returns
    -------
    dict
        Keys ``threshold``, ``precision``, ``recall``, ``f1``, ``accuracy``,
        ``true_positives``, ``false_positives``, ``false_negatives``,
        ``true_negatives``. Precision and recall are zero when their denominator
        is zero, which is the convention scikit-learn uses with
        ``zero_division=0``.
    """
    y = np.asarray(y_true, dtype=int)
    predicted = (np.asarray(probabilities, dtype=float) >= float(threshold)).astype(int)
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / len(y)) if len(y) else float("nan"),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_negatives": float(tn),
    }


def best_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Threshold maximising F1 on the data given, chosen on training rows only.

    Selecting a threshold on the data it is then evaluated on is a leak. Callers
    must pass training labels here and apply the returned threshold to the test
    rows; ``validation/validate_predictors.py`` does exactly that and says so.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    grid = np.unique(np.clip(np.round(probabilities, 3), 0.0, 1.0))
    best, best_f1 = 0.5, -1.0
    for threshold in grid:
        f1 = threshold_metrics(y_true, probabilities, float(threshold))["f1"]
        if f1 > best_f1:
            best, best_f1 = float(threshold), f1
    return best
