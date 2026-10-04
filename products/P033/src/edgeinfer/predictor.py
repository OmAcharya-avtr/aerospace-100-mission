"""Learned latency and peak-memory predictors, and the baseline comparison.

Order of work, which is not negotiable in this package: the analytic cost
model in :mod:`edgeinfer.analytic` is implemented and validated first, and
this module exists only to be measured against it. The comparison is run by
``validation/validate_predictor.py`` and its result is reported in
``README.md`` whichever way it falls.

Model
-----
:class:`LatencyPredictor` wraps a scikit-learn
:class:`~sklearn.ensemble.RandomForestRegressor` fitted on
``log10(latency / s)`` from the graph features in :mod:`edgeinfer.features`.

* **Why log-target.** Latency spans orders of magnitude over the model
  population, so a squared-error fit on raw seconds would be decided by the
  largest graph alone. Predicting the logarithm makes the loss a relative
  error, which is what a budget margin is expressed in.
* **Why a random forest.** The target has thresholds in it (the roofline
  ridge, a cache capacity, a kernel-selection switch in the runtime) that a
  linear model cannot represent, and the population is a few hundred points,
  which is too small for anything with many parameters. A forest also gives
  the uncertainty output below for free.
* **Uncertainty output.** The prediction's uncertainty is the standard
  deviation of the per-tree predictions, converted back through the
  exponential. This is the *ensemble disagreement*, which Breiman 2001
  ("Random Forests", *Machine Learning* 45(1), 5-32) relates to the
  variance component of the forest's error. It is not a calibrated
  prediction interval: it says nothing about the irreducible measurement
  noise, and
  ``validation/validate_predictor.py`` reports its empirical coverage so that
  the gap is a measured number rather than a claim.

Metrics
-------
Reported on a held-out split, for the baseline and the learned model on the
same split, as:

* median absolute relative error and 90th-percentile absolute relative error
  --- the tail of the *error*, for the same reason the latency tail is
  reported separately from the median;
* Spearman rank correlation, because the first use of a predictor is ranking
  candidate models against each other;
* the fraction of graphs whose budget verdict the predictor gets right, which
  is the decision the user actually makes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from edgeinfer.features import FEATURE_NAMES

__all__ = [
    "LatencyPredictor",
    "PredictionWithUncertainty",
    "PredictorMetrics",
    "compare_predictors",
    "evaluate_predictions",
    "split_indices",
]


@dataclass(frozen=True)
class PredictionWithUncertainty:
    """A point prediction with its ensemble-disagreement uncertainty.

    Attributes
    ----------
    value
        Point prediction in the target's unit (seconds for latency, bytes for
        memory).
    uncertainty
        Ensemble standard deviation propagated back through the exponential,
        same unit. **Not** a calibrated interval; see the module docstring.
    log10_value, log10_uncertainty
        The same quantities in the space the forest actually fits.
    n_estimators
        Trees behind the estimate.
    """

    value: float
    uncertainty: float
    log10_value: float
    log10_uncertainty: float
    n_estimators: int

    @property
    def relative_uncertainty(self) -> float:
        """``uncertainty / value`` [dimensionless]."""
        return self.uncertainty / self.value if self.value else float("nan")


@dataclass(frozen=True)
class PredictorMetrics:
    """Held-out metrics for one predictor.

    Attributes
    ----------
    label
        Predictor name.
    n_test
        Held-out sample count.
    median_abs_rel_error, p90_abs_rel_error
        Median and 90th percentile of ``|pred - truth| / truth``
        [dimensionless].
    max_abs_rel_error
        Worst single relative error [dimensionless].
    spearman
        Spearman rank correlation of prediction against truth
        [dimensionless, -1 to 1].
    log10_rmse
        Root mean squared error in ``log10`` space [dimensionless].
    """

    label: str
    n_test: int
    median_abs_rel_error: float
    p90_abs_rel_error: float
    max_abs_rel_error: float
    spearman: float
    log10_rmse: float

    def summary_lines(self) -> list[str]:
        """Human-readable metrics, every number with its unit."""
        return [
            f"predictor               : {self.label}",
            f"n (held out)            : {self.n_test}",
            f"median |rel error|      : {self.median_abs_rel_error * 100:.2f} %",
            f"p90 |rel error|         : {self.p90_abs_rel_error * 100:.2f} %",
            f"max |rel error|         : {self.max_abs_rel_error * 100:.2f} %",
            f"Spearman rho            : {self.spearman:.4f}",
            f"RMSE in log10 space     : {self.log10_rmse:.4f}",
        ]


def split_indices(
    n: int, test_fraction: float = 0.3, seed: int = 20260401
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/test split indices.

    Uses :func:`sklearn.model_selection.train_test_split` with a fixed
    ``random_state`` so the split is identical across runs, which is what
    makes the pinned regression outputs in ``tests/test_regression.py``
    meaningful.

    Parameters
    ----------
    n
        Population size, >= 4.
    test_fraction
        Held-out fraction, in (0, 1).
    seed
        Split seed.
    """
    if n < 4:
        raise ValueError(f"need at least 4 samples to split, got {n}")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must lie in (0, 1), got {test_fraction}")
    train, test = train_test_split(
        np.arange(n), test_size=test_fraction, random_state=seed, shuffle=True
    )
    return np.sort(train), np.sort(test)


class LatencyPredictor:
    """Random-forest regressor on ``log10(target)`` with an uncertainty output.

    Parameters
    ----------
    n_estimators
        Trees, >= 2. Default 120: on the 120-model population used here a
        forest of this size fits in well under a second on one CPU core, which
        is inside this repository's compute budget, and the ensemble spread is
        stable enough to report.
    max_depth
        Tree depth cap, or ``None`` for unlimited. Default 8, which keeps
        leaves from holding single samples on a population of a few hundred.
    min_samples_leaf
        Minimum samples per leaf, >= 1.
    seed
        ``random_state`` for the forest.
    target_name
        What is being predicted, carried into reports, e.g.
        ``"p50 latency [s]"``.
    """

    def __init__(
        self,
        n_estimators: int = 120,
        max_depth: int | None = 8,
        min_samples_leaf: int = 2,
        seed: int = 20260401,
        target_name: str = "latency [s]",
    ) -> None:
        if n_estimators < 2:
            raise ValueError(f"n_estimators must be >= 2, got {n_estimators}")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be >= 1 or None, got {max_depth}")
        self.n_estimators = int(n_estimators)
        self.target_name = target_name
        self._forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=1,  # one core; n_jobs=-1 on a shared host costs more than it saves
        )
        self._fitted = False

    def fit(self, features: np.ndarray, target: np.ndarray) -> LatencyPredictor:
        """Fit on ``log10(target)``.

        Parameters
        ----------
        features
            Shape ``(n, len(FEATURE_NAMES))``.
        target
            Shape ``(n,)``, strictly positive, in the unit named by
            ``target_name``.
        """
        features = np.asarray(features, dtype=float)
        target = np.asarray(target, dtype=float)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have shape (n, {len(FEATURE_NAMES)}), got {features.shape}"
            )
        if features.shape[0] != target.shape[0]:
            raise ValueError(
                f"features has {features.shape[0]} rows but target has {target.shape[0]}"
            )
        if np.any(target <= 0):
            raise ValueError(
                "target must be strictly positive: the predictor fits log10(target)"
            )
        self._forest.fit(features, np.log10(target))
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("LatencyPredictor.fit() must be called before predicting")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Point predictions in the target unit. Shape ``(n,)``."""
        self._check_fitted()
        features = np.atleast_2d(np.asarray(features, dtype=float))
        return np.power(10.0, self._forest.predict(features))

    def predict_with_uncertainty(
        self, features: np.ndarray
    ) -> list[PredictionWithUncertainty]:
        """Point predictions with ensemble-disagreement uncertainty.

        The uncertainty on the target is propagated from the ``log10`` space
        by the first-order relation ``u(y) = y * ln(10) * u(log10 y)``, which
        is the standard propagation of a standard uncertainty through a
        monotonic transform (GUM, JCGM 100:2008 §5.1.2, with the sensitivity
        coefficient ``dy/d log10 y = y ln 10``). It is a first-order
        approximation and understates the uncertainty when the ensemble spread
        is large.
        """
        self._check_fitted()
        features = np.atleast_2d(np.asarray(features, dtype=float))
        per_tree = np.vstack(
            [tree.predict(features) for tree in self._forest.estimators_]
        )
        log_mean = per_tree.mean(axis=0)
        log_std = per_tree.std(axis=0, ddof=1)
        value = np.power(10.0, log_mean)
        return [
            PredictionWithUncertainty(
                value=float(v),
                uncertainty=float(v * np.log(10.0) * s),
                log10_value=float(m),
                log10_uncertainty=float(s),
                n_estimators=self.n_estimators,
            )
            for v, s, m in zip(value, log_std, log_mean, strict=True)
        ]

    @property
    def feature_importances(self) -> dict[str, float]:
        """Impurity-based feature importances, keyed by feature name.

        Impurity importance is biased towards high-cardinality features
        (Strobl, Boulesteix, Zeileis & Hothorn 2007, "Bias in random forest
        variable importance measures", *BMC Bioinformatics* 8:25), so it is
        reported as a diagnostic and never as evidence that a feature matters
        physically.
        """
        self._check_fitted()
        return dict(
            zip(FEATURE_NAMES, (float(v) for v in self._forest.feature_importances_),
                strict=True)
        )


def evaluate_predictions(
    predicted: np.ndarray, truth: np.ndarray, label: str
) -> PredictorMetrics:
    """Held-out metrics for one set of predictions.

    Parameters
    ----------
    predicted, truth
        Same length, strictly positive.
    label
        Predictor name for the report.
    """
    from scipy.stats import spearmanr

    predicted = np.asarray(predicted, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if predicted.shape != truth.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {truth.shape}")
    if predicted.size < 2:
        raise ValueError("need at least 2 points to evaluate")
    if np.any(truth <= 0) or np.any(predicted <= 0):
        raise ValueError("predicted and truth must be strictly positive")
    rel = np.abs(predicted - truth) / truth
    rho = spearmanr(predicted, truth).statistic
    return PredictorMetrics(
        label=label,
        n_test=int(predicted.size),
        median_abs_rel_error=float(np.median(rel)),
        p90_abs_rel_error=float(np.quantile(rel, 0.90)),
        max_abs_rel_error=float(np.max(rel)),
        spearman=float(rho),
        log10_rmse=float(np.sqrt(np.mean((np.log10(predicted) - np.log10(truth)) ** 2))),
    )


def compare_predictors(
    analytic: np.ndarray,
    learned: np.ndarray,
    truth: np.ndarray,
    analytic_label: str = "analytic roofline (calibrated)",
    learned_label: str = "learned random forest",
) -> tuple[PredictorMetrics, PredictorMetrics, str]:
    """Evaluate both predictors on the same held-out truth and state the winner.

    Returns
    -------
    (analytic_metrics, learned_metrics, verdict)
        ``verdict`` is one line naming which predictor has the lower median
        absolute relative error, or stating that the difference is inside the
        bootstrap uncertainty of the difference, in which case the honest
        reading is **no measurable advantage**.

    Notes
    -----
    The comparison is on the median absolute relative error. The uncertainty
    of the *difference* between two medians on the same test set is obtained
    by a paired bootstrap over the test points (Efron & Tibshirani 1993 §13),
    using 2000 resamples with a fixed seed. If zero lies inside one standard
    deviation of the difference, no advantage is claimed.
    """
    analytic = np.asarray(analytic, dtype=float)
    learned = np.asarray(learned, dtype=float)
    truth = np.asarray(truth, dtype=float)
    m_a = evaluate_predictions(analytic, truth, analytic_label)
    m_l = evaluate_predictions(learned, truth, learned_label)

    rel_a = np.abs(analytic - truth) / truth
    rel_l = np.abs(learned - truth) / truth
    rng = np.random.default_rng(20260401)
    idx = rng.integers(0, truth.size, size=(2000, truth.size))
    diffs = np.median(rel_a[idx], axis=1) - np.median(rel_l[idx], axis=1)
    diff = m_a.median_abs_rel_error - m_l.median_abs_rel_error
    sd = float(np.std(diffs, ddof=1))

    if abs(diff) <= sd:
        verdict = (
            f"no measurable advantage: median |rel error| differs by "
            f"{diff * 100:+.2f} pp, inside the paired-bootstrap standard deviation of "
            f"{sd * 100:.2f} pp (2000 resamples, seed 20260401)"
        )
    elif diff > 0:
        verdict = (
            f"{learned_label} wins: median |rel error| {m_l.median_abs_rel_error * 100:.2f} % "
            f"vs {m_a.median_abs_rel_error * 100:.2f} % for {analytic_label}, a difference of "
            f"{diff * 100:.2f} pp against a bootstrap sd of {sd * 100:.2f} pp"
        )
    else:
        verdict = (
            f"{analytic_label} wins: median |rel error| "
            f"{m_a.median_abs_rel_error * 100:.2f} % vs "
            f"{m_l.median_abs_rel_error * 100:.2f} % for {learned_label}, a difference of "
            f"{-diff * 100:.2f} pp against a bootstrap sd of {sd * 100:.2f} pp"
        )
    return m_a, m_l, verdict
