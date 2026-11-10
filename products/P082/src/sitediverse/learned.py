"""The learned spatio-temporal correlation model, with an interval output.

Implemented and benchmarked AFTER the analytic baselines of
``sitediverse.independence``, ``sitediverse.correlation`` and
``sitediverse.empirical``, which are validated against the generator's own
ground truth before anything here is fitted. That order is not a courtesy: an
under-strength baseline inflates an apparent learned margin, and this mission
has measured that happening.

**(35) What is learned.** The map from the eight pair-regime features of
equation (33) to the implied latent correlation of equation (11). It replaces
``rho_0 exp(-d / L)`` with a learned function of separation, of the along-axis
and cross-axis decomposition of equation (3), of the seasonal regime and of the
marginal rates. Everything downstream -- the copula of equation (8), the network
Monte Carlo of equation (10) -- is shared with the baselines, so the comparison
isolates the correlation model and nothing else.

**(36) The architecture.** Three gradient-boosted regression trees
(``sklearn.ensemble.GradientBoostingRegressor``, the algorithm of Friedman 2001,
*Annals of Statistics* 29, 1189), fitted with the quantile loss at the declared
levels ``(0.1, 0.5, 0.9)``. The median model is the point estimate; the outer
two give an interval. Gradient boosting rather than a neural network because
PyTorch is not installed in the measured environment, because the training set
is 420 rows in the declared configuration and a tree ensemble is the right
capacity for that, and because the quantile loss gives a prediction interval
directly rather than through a separate calibration step.

**(37) The uncertainty output, and what it is not.** ``predict`` returns a
``CorrelationPrediction`` carrying ``lower``, ``median`` and ``upper``. There is
no method on this class that returns a bare number: a caller who wants a point
estimate reads ``.median`` and in doing so has the interval in hand. The
interval is an 80 per cent quantile-regression interval on the CONDITIONAL
DISTRIBUTION OF THE TARGET given the features, as estimated from the training
rows. It is therefore:

- an estimate of where a future implied correlation with these features would
  fall, which is what a network designer wants;
- NOT a confidence interval on the true correlation, because it contains the
  estimation noise of equation (11)'s target as well as the model's own error;
- NOT calibrated by construction. Its empirical coverage on both hold-out cells
  is MEASURED and reported in ``validation/validate_learned_model_output.txt``,
  and the measured coverage is below the nominal 80 per cent on at least one
  cell. An uncertainty output whose coverage is not measured is decoration, so
  the measurement is reported whichever way it comes out.

A monotonicity note. Three quantile models fitted independently can cross -- the
0.9 model can return a value below the 0.5 model at some feature vector --
because nothing in the fitting couples them. Crossings are COUNTED and reported
rather than repaired by sorting, because the count is a diagnostic of the fit
and a sorted interval hides it. ``CorrelationPrediction.crossed`` carries the
flag per row.

**(38) Hyperparameters, all declared.** 240 trees of depth 3, learning rate
0.05, subsample 0.9, ``random_state`` 12345. Chosen once, against the TRAIN cell
only, and not tuned against either hold-out cell -- which would make both
hold-out scores meaningless. The validation output records that the
configuration is the declared one and that no search was run against a hold-out
cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

#: Declared quantile levels of equation (36).
QUANTILES: tuple[float, float, float] = (0.1, 0.5, 0.9)

#: Nominal coverage of the interval, ``QUANTILES[2] - QUANTILES[0]``.
NOMINAL_COVERAGE: float = QUANTILES[2] - QUANTILES[0]

#: Declared hyperparameters of equation (38).
TREES: int = 240
MAX_DEPTH: int = 3
LEARNING_RATE: float = 0.05
SUBSAMPLE: float = 0.9
RANDOM_STATE: int = 12345

#: Feature names, repeated here so that a report can label an importance vector
#: without importing ``sitediverse.features``, which imports the copula and the
#: climatology. Asserted equal to ``features.FEATURE_NAMES`` by the test suite.
FEATURE_NAMES_NOTE: tuple[str, ...] = (
    "separation_km",
    "along_km",
    "cross_km",
    "anisotropy_ratio",
    "regime",
    "rate_min",
    "rate_max",
    "rate_product",
)

#: Clamp applied to a predicted correlation before it is handed to the copula.
#: A tree ensemble is not constrained to return a correlation, and a value at or
#: beyond +/-1 is not one. The clamp is reported whenever it binds.
RHO_CLAMP: tuple[float, float] = (-0.95, 0.995)


@dataclass(frozen=True)
class CorrelationPrediction:
    """A predicted correlation as an interval. There is no bare-value method.

    Attributes
    ----------
    lower, median, upper:
        ``(rows,)`` quantile predictions at the declared levels.
    clamped:
        ``(rows,)`` boolean, whether the median had to be clamped into
        ``RHO_CLAMP`` before use.
    crossed:
        ``(rows,)`` boolean, whether the three quantiles came out of order.
    """

    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray
    clamped: np.ndarray
    crossed: np.ndarray

    @property
    def rows(self) -> int:
        """Number of predicted rows."""
        return int(self.median.size)

    @property
    def interval_width(self) -> np.ndarray:
        """``upper - lower`` per row. Negative where the quantiles crossed."""
        return self.upper - self.lower

    @property
    def crossing_count(self) -> int:
        """Number of rows whose quantiles came out of order."""
        return int(np.count_nonzero(self.crossed))

    @property
    def clamp_count(self) -> int:
        """Number of rows whose median was clamped."""
        return int(np.count_nonzero(self.clamped))

    def covers(self, truth: np.ndarray) -> np.ndarray:
        """``(rows,)`` boolean: whether each interval contains ``truth``.

        Uses the min and max of the two outer quantiles, so a crossed row is
        treated as the interval between them rather than as empty. The crossing
        is reported separately and is not silently absorbed here: a reader
        looking at a coverage figure also has ``crossing_count``.
        """
        target = np.asarray(truth, dtype=float)
        if target.shape != (self.rows,):
            raise ValueError(
                f"truth must have shape ({self.rows},), got {target.shape}"
            )
        low = np.minimum(self.lower, self.upper)
        high = np.maximum(self.lower, self.upper)
        return (target >= low) & (target <= high)

    def coverage(self, truth: np.ndarray) -> float:
        """Fraction of rows whose interval contains ``truth``."""
        return float(np.mean(self.covers(truth)))


@dataclass(frozen=True)
class TrainingReport:
    """What a fit actually did. Reported, not inferred from the configuration.

    Attributes
    ----------
    rows, feature_count:
        Shape of the training matrix.
    fit_seconds:
        Wall-clock time to fit all three quantile models.
    quantiles:
        The declared levels.
    trees, max_depth, learning_rate, subsample, random_state:
        The declared hyperparameters that were used.
    tuned_against_holdout:
        Always ``False``. Carried as a field rather than as a sentence in a
        docstring so that the validation output can print it.
    """

    rows: int
    feature_count: int
    fit_seconds: float
    quantiles: tuple[float, float, float]
    trees: int
    max_depth: int
    learning_rate: float
    subsample: float
    random_state: int
    tuned_against_holdout: bool = False


class LearnedCorrelationModel:
    """Equation (35): the learned correlation model of equations (36) to (38)."""

    def __init__(
        self,
        quantiles: tuple[float, float, float] = QUANTILES,
        trees: int = TREES,
        max_depth: int = MAX_DEPTH,
        learning_rate: float = LEARNING_RATE,
        subsample: float = SUBSAMPLE,
        random_state: int = RANDOM_STATE,
    ) -> None:
        if len(quantiles) != 3:
            raise ValueError(
                f"exactly 3 quantile levels are required, got {len(quantiles)}"
            )
        if not all(0.0 < level < 1.0 for level in quantiles):
            raise ValueError(f"every quantile level must lie in (0, 1), got {quantiles}")
        if not quantiles[0] < quantiles[1] < quantiles[2]:
            raise ValueError(
                f"quantile levels must be strictly increasing, got {quantiles}"
            )
        if trees < 1:
            raise ValueError(f"trees must be at least 1, got {trees!r}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be at least 1, got {max_depth!r}")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError(
                f"learning_rate must lie in (0, 1], got {learning_rate!r}"
            )
        if not 0.0 < subsample <= 1.0:
            raise ValueError(f"subsample must lie in (0, 1], got {subsample!r}")
        self.quantiles = tuple(float(level) for level in quantiles)
        self.trees = int(trees)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.random_state = int(random_state)
        self._models: list[GradientBoostingRegressor] = []
        self._report: TrainingReport | None = None

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return bool(self._models)

    @property
    def report(self) -> TrainingReport:
        """The training report. Raises before a fit."""
        if self._report is None:
            raise RuntimeError("the model has not been fitted; call fit() first")
        return self._report

    def fit(self, features: np.ndarray, target: np.ndarray) -> LearnedCorrelationModel:
        """Fit the three quantile models of equation (36).

        Parameters
        ----------
        features:
            ``(rows, 8)`` the equation (33) features.
        target:
            ``(rows,)`` implied latent correlations.

        Returns
        -------
        LearnedCorrelationModel
            ``self``, so that a fit can be chained.
        """
        import time

        matrix = np.asarray(features, dtype=float)
        values = np.asarray(target, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"features must be two-dimensional, got {matrix.ndim}D")
        if values.shape != (matrix.shape[0],):
            raise ValueError(
                f"target must have shape ({matrix.shape[0]},), got {values.shape}"
            )
        if matrix.shape[0] < 2 * self.max_depth + 2:
            raise ValueError(
                f"{matrix.shape[0]} training rows is too few for trees of depth "
                f"{self.max_depth}; at least {2 * self.max_depth + 2} are needed"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("features contain a non-finite value")
        if not np.all(np.isfinite(values)):
            raise ValueError("target contains a non-finite value")
        started = time.perf_counter()
        self._models = []
        for level in self.quantiles:
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=level,
                n_estimators=self.trees,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=self.random_state,
            )
            model.fit(matrix, values)
            self._models.append(model)
        self._report = TrainingReport(
            rows=int(matrix.shape[0]),
            feature_count=int(matrix.shape[1]),
            fit_seconds=float(time.perf_counter() - started),
            quantiles=self.quantiles,
            trees=self.trees,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        return self

    def predict(self, features: np.ndarray) -> CorrelationPrediction:
        """Equation (37): the interval prediction. No bare-value variant exists."""
        if not self.fitted:
            raise RuntimeError("the model has not been fitted; call fit() first")
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"features must be two-dimensional, got {matrix.ndim}D")
        expected = self.report.feature_count
        if matrix.shape[1] != expected:
            raise ValueError(
                f"features must have {expected} columns to match the fit, got "
                f"{matrix.shape[1]}"
            )
        raw = [model.predict(matrix) for model in self._models]
        lower, median, upper = raw
        clamped_median = np.clip(median, RHO_CLAMP[0], RHO_CLAMP[1])
        return CorrelationPrediction(
            lower=lower,
            median=clamped_median,
            upper=upper,
            clamped=clamped_median != median,
            crossed=~((lower <= median) & (median <= upper)),
        )

    def feature_importance(self) -> np.ndarray:
        """Impurity-based importance of the MEDIAN model, one entry per feature.

        Impurity-based importance is biased toward high-cardinality features and
        is reported here as a diagnostic only. No claim in this product rests on
        it; the claim that the anisotropy features matter rests on the measured
        hold-out scores against a model that cannot see them.
        """
        if not self.fitted:
            raise RuntimeError("the model has not been fitted; call fit() first")
        return np.asarray(self._models[1].feature_importances_, dtype=float)
