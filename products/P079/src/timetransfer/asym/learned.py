"""The learned delay-asymmetry corrector, and the uncertainty it reports.

What it is
----------
A gradient-boosted regression tree ensemble (Friedman, "Greedy function
approximation: a gradient boosting machine", *Annals of Statistics*
29(5):1189-1232, 2001; implementation
:class:`sklearn.ensemble.GradientBoostingRegressor`) mapping the sixteen window
features of :mod:`timetransfer.asym.features` to the window's mean asymmetry
bias in seconds.

A ridge regression (:class:`sklearn.linear_model.Ridge`) on standardised
features is provided alongside as the linear learned baseline, because the
median-filtered and mean-difference estimators are both linear in one feature
and the honest question is whether anything beyond a linear combination of the
sixteen helps.

Uncertainty output
------------------
Three models are trained: the squared-error model for the point prediction and
two quantile models at ``alpha = 0.1587`` and ``alpha = 0.8413``, whose
difference is a one-sigma interval under a symmetric error law. The reported
per-row standard uncertainty is ``(upper - lower) / 2``. Quantile regression
for gradient boosting is the pinball-loss variant described in Friedman 2001
section 4.3 and implemented by scikit-learn as ``loss="quantile"``.

This is a *dispersion estimate*, not a calibrated interval, and the difference
matters. ``validation/validate_asymmetry_benchmark.py`` measures the coverage
of the resulting band on the held-out split and reports it; where it is not
calibrated, the number is published as measured and MODEL_CARD.md says so.

Compute budget
--------------
Three ensembles of 200 trees at depth 3 over fewer than 2000 rows and 16
features. Measured on the two-core container this product was built in, the
fit is in the low seconds; the measured figure is in
``validation/validate_asymmetry_benchmark_output.txt`` and in the benchmark
table, which is declared unstable.

Determinism
-----------
``random_state`` is set on every estimator and no subsampling is used, so a
refit on the same split reproduces the same model. That is asserted by
``tests/test_reproducibility.py`` on the prediction vector, not on a pickle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..errors import ConfigurationError, InsufficientDataError
from .features import FEATURE_NAMES

__all__ = ["LearnedAsymmetryCorrector", "RidgeAsymmetryCorrector", "TrainingConfig"]


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters, fixed in code so that a run is reproducible.

    Attributes
    ----------
    n_estimators:
        Boosting stages, >= 10.
    max_depth:
        Tree depth, >= 1.
    learning_rate:
        Shrinkage, > 0.
    min_samples_leaf:
        Minimum rows per leaf, >= 1.
    random_state:
        Seed for every estimator.
    sigma_quantiles:
        The two quantiles whose half-difference is reported as one sigma.
    """

    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    min_samples_leaf: int = 8
    random_state: int = 20261024
    sigma_quantiles: tuple[float, float] = (0.1587, 0.8413)

    def __post_init__(self) -> None:
        if int(self.n_estimators) < 10:
            raise ConfigurationError(f"n_estimators must be >= 10, got {self.n_estimators!r}")
        if int(self.max_depth) < 1:
            raise ConfigurationError(f"max_depth must be >= 1, got {self.max_depth!r}")
        if not (float(self.learning_rate) > 0.0):
            raise ConfigurationError(f"learning_rate must be > 0, got {self.learning_rate!r}")
        if int(self.min_samples_leaf) < 1:
            raise ConfigurationError(
                f"min_samples_leaf must be >= 1, got {self.min_samples_leaf!r}"
            )
        lo, hi = (float(v) for v in self.sigma_quantiles)
        if not (0.0 < lo < hi < 1.0):
            raise ConfigurationError(
                f"sigma_quantiles must satisfy 0 < lo < hi < 1, got {self.sigma_quantiles!r}"
            )


def _check_xy(features: np.ndarray, target_s: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ConfigurationError(
            f"features must have shape (n, {len(FEATURE_NAMES)}), got {x.shape}"
        )
    if not np.all(np.isfinite(x)):
        raise ConfigurationError("features must be finite")
    if target_s is None:
        return x, np.zeros(0, dtype=np.float64)
    y = np.asarray(target_s, dtype=np.float64).reshape(-1)
    if y.size != x.shape[0]:
        raise ConfigurationError(f"target has {y.size} entries for {x.shape[0]} feature rows")
    if not np.all(np.isfinite(y)):
        raise ConfigurationError("target must be finite")
    return x, y


class LearnedAsymmetryCorrector:
    """Gradient-boosted corrector with a quantile-pair dispersion estimate.

    Parameters
    ----------
    config:
        Hyperparameters; the default is the configuration every number in this
        repository was measured with.
    """

    name = "learned_gbr"

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = TrainingConfig() if config is None else config
        self._mean: GradientBoostingRegressor | None = None
        self._lower: GradientBoostingRegressor | None = None
        self._upper: GradientBoostingRegressor | None = None
        self._residual_std_s: float | None = None
        self._n_fit = 0

    def _make(self, loss: str, alpha: float | None) -> GradientBoostingRegressor:
        cfg = self.config
        kwargs: dict[str, object] = {
            "loss": loss,
            "n_estimators": int(cfg.n_estimators),
            "max_depth": int(cfg.max_depth),
            "learning_rate": float(cfg.learning_rate),
            "min_samples_leaf": int(cfg.min_samples_leaf),
            "random_state": int(cfg.random_state),
            "subsample": 1.0,
        }
        if alpha is not None:
            kwargs["alpha"] = float(alpha)
        return GradientBoostingRegressor(**kwargs)  # type: ignore[arg-type]

    def fit(self, features: np.ndarray, target_s: np.ndarray) -> LearnedAsymmetryCorrector:
        """Train the point model and the two quantile models."""
        x, y = _check_xy(features, target_s)
        if y.size < 4 * int(self.config.min_samples_leaf):
            raise InsufficientDataError(
                f"training needs at least {4 * int(self.config.min_samples_leaf)} rows for "
                f"min_samples_leaf={self.config.min_samples_leaf}, got {y.size}",
                n_usable=int(y.size),
                n_required=4 * int(self.config.min_samples_leaf),
            )
        lo, hi = (float(v) for v in self.config.sigma_quantiles)
        self._mean = self._make("squared_error", None).fit(x, y)
        self._lower = self._make("quantile", lo).fit(x, y)
        self._upper = self._make("quantile", hi).fit(x, y)
        self._residual_std_s = float(np.std(y - self._mean.predict(x), ddof=1))
        self._n_fit = int(y.size)
        return self

    def _require_fit(self) -> GradientBoostingRegressor:
        if self._mean is None:
            raise ConfigurationError(f"{self.name}: call fit() before predict()")
        return self._mean

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predicted asymmetry bias [s], one per row."""
        model = self._require_fit()
        x, _ = _check_xy(features, None)
        return np.asarray(model.predict(x), dtype=np.float64)

    def predict_with_uncertainty(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prediction [s] and a per-row standard uncertainty [s].

        The uncertainty is half the width of the quantile pair, floored at zero
        because the two quantile models are fitted independently and can cross
        on a row neither saw -- which is reported here rather than hidden, and
        counted by :attr:`crossing_fraction` on the last call.
        """
        model = self._require_fit()
        assert self._lower is not None and self._upper is not None
        x, _ = _check_xy(features, None)
        mean = np.asarray(model.predict(x), dtype=np.float64)
        lower = np.asarray(self._lower.predict(x), dtype=np.float64)
        upper = np.asarray(self._upper.predict(x), dtype=np.float64)
        raw = 0.5 * (upper - lower)
        self.crossing_fraction = float(np.mean(raw < 0.0))
        sigma = np.maximum(raw, 0.0)
        return mean, sigma

    @property
    def residual_std_s(self) -> float:
        """In-sample residual standard deviation [s] from :meth:`fit`.

        In-sample, and labelled so. The uncertainty budget uses the held-out
        figure reported by :mod:`timetransfer.asym.metrics`, not this one.
        """
        if self._residual_std_s is None:
            raise ConfigurationError(f"{self.name}: residual_std_s needs fit() first")
        return self._residual_std_s

    @property
    def feature_importances(self) -> np.ndarray:
        """Impurity-based feature importances of the point model."""
        model = self._require_fit()
        return np.asarray(model.feature_importances_, dtype=np.float64)


class RidgeAsymmetryCorrector:
    """Standardised ridge regression on the sixteen features.

    Parameters
    ----------
    alpha:
        Ridge penalty, > 0.
    """

    name = "learned_ridge"

    def __init__(self, *, alpha: float = 1.0) -> None:
        if not (float(alpha) > 0.0):
            raise ConfigurationError(f"alpha must be > 0, got {alpha!r}")
        self._pipeline = Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(alpha=float(alpha)))]
        )
        self._fitted = False
        self._residual_std_s: float | None = None

    def fit(self, features: np.ndarray, target_s: np.ndarray) -> RidgeAsymmetryCorrector:
        """Fit the pipeline."""
        x, y = _check_xy(features, target_s)
        if y.size < len(FEATURE_NAMES) + 2:
            raise InsufficientDataError(
                f"ridge needs at least {len(FEATURE_NAMES) + 2} rows, got {y.size}",
                n_usable=int(y.size),
                n_required=len(FEATURE_NAMES) + 2,
            )
        self._pipeline.fit(x, y)
        self._fitted = True
        self._residual_std_s = float(np.std(y - self._pipeline.predict(x), ddof=1))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predicted asymmetry bias [s]."""
        if not self._fitted:
            raise ConfigurationError(f"{self.name}: call fit() before predict()")
        x, _ = _check_xy(features, None)
        return np.asarray(self._pipeline.predict(x), dtype=np.float64)

    def predict_with_uncertainty(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prediction [s] and the homoscedastic in-sample residual spread [s]."""
        if self._residual_std_s is None:
            raise ConfigurationError(f"{self.name}: call fit() first")
        mean = self.predict(features)
        return mean, np.full(mean.size, self._residual_std_s, dtype=np.float64)

    @property
    def coefficients(self) -> np.ndarray:
        """Ridge coefficients on the standardised features."""
        if not self._fitted:
            raise ConfigurationError(f"{self.name}: call fit() first")
        return np.asarray(self._pipeline.named_steps["ridge"].coef_, dtype=np.float64)
