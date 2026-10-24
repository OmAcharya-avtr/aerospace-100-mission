"""The analytic asymmetry-bias estimators. These are implemented first.

Three of them, in increasing order of cleverness and none of them learned:

* :class:`SymmetricAssumption` predicts zero bias. This is what every
  conventional two-way implementation does, and it is the thing any corrector
  has to beat to be worth its existence. It is not a strawman: when the path
  really is reciprocal it is the right answer and has no variance.
* :class:`MedianFilterEstimator` predicts
  ``(median(fwd_c) - median(rev_c)) / 2`` from the centred dwells. Robust to
  the heavy right tail of queueing delay, needs no training, and is the
  estimator the specification for this product predicted would match the
  learned model.
* :class:`MeanDifferenceEstimator` predicts
  ``(mean(fwd_c) - mean(rev_c)) / 2``. The target is a *mean* over the window,
  so the mean difference is the natural plug-in estimator; it is included
  because leaving it out would have made the median filter look better than it
  is, and the comparison is the point.

What none of them can do
------------------------
All three work on centred dwells, so none has access to the static asymmetry
``f_ab - f_ba``: :class:`SymmetricAssumption` ignores it and the other two
cannot see it. The static part is why the residual of even a perfect queueing
estimator is not zero, and it is reported separately in the benchmark.

Fitting
-------
None of the three has parameters. :meth:`fit` exists so they share an interface
with the learned corrector, and what it actually does is **measure the
residual standard deviation on the training split**, which is the number
:mod:`timetransfer.uncertainty` needs for its asymmetry term. That number is a
measurement, not a datasheet figure, and it is measured on data the estimator
was not scored on.
"""

from __future__ import annotations

import numpy as np

from ..errors import ConfigurationError, InsufficientDataError
from .features import FEATURE_NAMES

__all__ = [
    "AnalyticEstimator",
    "MeanDifferenceEstimator",
    "MedianFilterEstimator",
    "SymmetricAssumption",
]

_MEAN_DIFFERENCE_COLUMN = FEATURE_NAMES.index("mean_difference_s")
_MEDIAN_DIFFERENCE_COLUMN = FEATURE_NAMES.index("median_difference_s")


class AnalyticEstimator:
    """Base class: a closed-form predictor plus a measured residual spread.

    Subclasses implement :meth:`predict`. :meth:`fit` records the residual
    standard deviation and median absolute residual on whatever split it is
    given, and :attr:`residual_std_s` is what the uncertainty budget consumes.
    """

    name = "analytic"

    def __init__(self) -> None:
        self._residual_std_s: float | None = None
        self._residual_mad_s: float | None = None
        self._n_fit = 0

    @staticmethod
    def _check(features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ConfigurationError(
                f"features must have shape (n, {len(FEATURE_NAMES)}), got {x.shape}"
            )
        if not np.all(np.isfinite(x)):
            raise ConfigurationError("features must be finite")
        return x

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predicted asymmetry bias [s], one per row."""
        raise NotImplementedError

    def fit(self, features: np.ndarray, target_s: np.ndarray) -> AnalyticEstimator:
        """Measure the residual spread on ``(features, target_s)``.

        Returns ``self`` so it chains like a scikit-learn estimator.
        """
        x = self._check(features)
        y = np.asarray(target_s, dtype=np.float64).reshape(-1)
        if y.size != x.shape[0]:
            raise ConfigurationError(
                f"target has {y.size} entries for {x.shape[0]} feature rows"
            )
        if y.size < 4:
            raise InsufficientDataError(
                f"residual spread needs at least 4 rows, got {y.size}",
                n_usable=int(y.size),
                n_required=4,
            )
        resid = y - self.predict(x)
        self._residual_std_s = float(np.std(resid, ddof=1))
        self._residual_mad_s = float(np.median(np.abs(resid - np.median(resid))))
        self._n_fit = int(y.size)
        return self

    @property
    def residual_std_s(self) -> float:
        """Measured residual standard deviation [s] from :meth:`fit`."""
        if self._residual_std_s is None:
            raise ConfigurationError(
                f"{self.name}: residual_std_s is only available after fit(); it is a "
                f"measurement, not a constant"
            )
        return self._residual_std_s

    @property
    def residual_mad_s(self) -> float:
        """Measured median absolute residual [s] from :meth:`fit`."""
        if self._residual_mad_s is None:
            raise ConfigurationError(f"{self.name}: residual_mad_s needs fit() first")
        return self._residual_mad_s

    def predict_with_uncertainty(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prediction [s] and a per-row standard uncertainty [s].

        An analytic estimator has no per-row uncertainty of its own, so the
        measured training residual standard deviation is broadcast to every
        row. That is stated rather than dressed up: the interval is
        homoscedastic because the estimator is.
        """
        x = self._check(features)
        mean = self.predict(x)
        return mean, np.full(x.shape[0], self.residual_std_s, dtype=np.float64)


class SymmetricAssumption(AnalyticEstimator):
    """Predict zero asymmetry bias: the conventional two-way assumption."""

    name = "symmetric_assumption"

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = self._check(features)
        return np.zeros(x.shape[0], dtype=np.float64)


class MedianFilterEstimator(AnalyticEstimator):
    """Predict ``(median(fwd_c) - median(rev_c)) / 2`` [s]."""

    name = "median_filter"

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = self._check(features)
        return 0.5 * x[:, _MEDIAN_DIFFERENCE_COLUMN]


class MeanDifferenceEstimator(AnalyticEstimator):
    """Predict ``(mean(fwd_c) - mean(rev_c)) / 2`` [s]."""

    name = "mean_difference"

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = self._check(features)
        return 0.5 * x[:, _MEAN_DIFFERENCE_COLUMN]
