"""The learned tail predictor: gradient-boosted trees with quantile heads.

Architecture
------------
Three gradient-boosting ensembles over the same thirteen probe features
(:mod:`latencynet.features`):

* a squared-error ensemble for the point prediction of ``ln q_p``;
* a quantile-loss ensemble at ``alpha = (1 - level) / 2`` for the lower
  interval endpoint;
* a quantile-loss ensemble at ``alpha = (1 + level) / 2`` for the upper
  endpoint.

Gradient boosting of regression trees: Friedman (2001), "Greedy Function
Approximation: A Gradient Boosting Machine", *Annals of Statistics* 29(5):
1189-1232. Quantile loss: Koenker & Bassett (1978), "Regression Quantiles",
*Econometrica* 46(1): 33-50, as implemented by
``sklearn.ensemble.GradientBoostingRegressor(loss="quantile")``.

Why trees and not a neural network: PyTorch is not available in this build
environment, the training sets here are a few hundred pipelines, and the
target is a smooth low-dimensional function of thirteen features. A boosted
tree ensemble is the right size of model for that, and it trains in about a
second on one core.

Why this model could help at all
--------------------------------
It cannot beat the analytic baseline on its own ground. Where stage latencies
are independent, equation (2) of :mod:`latencynet.analytic` is exact and the
only error left is Fenton-Wilkinson approximation error plus probe noise; a
tree ensemble fitted to a few hundred noisy targets will not improve on an
identity. Where the stages are *dependent*, the independence-mode analytic
baseline is systematically wrong -- it under-states the total variance by
``2 sum_{i<j} Cov_ij`` and therefore under-predicts the tail -- and the
dependence features give a learned model something to correct with. That is
the hypothesis the comparison in ``validation/validate_model_comparison.py``
tests, and the README reports which regime the data actually falls in.

Reproducibility: ``random_state`` is fixed and gradient boosting is
deterministic at ``subsample = 1.0``, so a fit is reproducible from
``(records, p, hyperparameters, random_state)``.

Compute: three fits of 300 trees at depth 3 over at most a few hundred rows
and thirteen features, about 1.2 s on one core, measured. No GPU, no
PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .dataset import PipelineRecord, feature_matrix, log_target
from .predictors import IntervalPrediction, TailPredictor


@dataclass(frozen=True)
class BoostingHyperparameters:
    """Fixed hyperparameters of the learned predictor.

    300 trees of depth 3 at learning rate 0.05, with a minimum of 5 samples
    per leaf. These are fixed in the source and were not tuned against the
    test split. ``validation/validate_model_comparison.py`` reports a
    six-point capacity sweep over ``(n_estimators, max_depth,
    learning_rate)`` so that the comparison's conclusion can be checked
    against the possibility that it is an artefact of one setting; the sweep
    is reported in full, not reduced to its best entry.
    """

    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.05
    min_samples_leaf: int = 5
    random_state: int = 20260403

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("learning_rate must lie in (0, 1]")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")


class LearnedTailPredictor(TailPredictor):
    """Gradient-boosted tail predictor with native quantile-loss intervals.

    Parameters
    ----------
    hyperparameters:
        Fixed ensemble settings; see :class:`BoostingHyperparameters`.
    native_interval_level:
        The nominal coverage the quantile heads are trained for. The quantile
        heads are fitted at this level during :meth:`fit`, so asking
        :meth:`predict_log_interval` for a different level raises rather than
        silently returning an interval for the wrong level.
    """

    name = "learned_gbt"
    uses_dependence_features = True

    def __init__(
        self,
        hyperparameters: BoostingHyperparameters | None = None,
        native_interval_level: float = 0.9,
    ) -> None:
        if not (0.0 < float(native_interval_level) < 1.0):
            raise ValueError(
                f"native_interval_level must lie strictly in (0, 1), got {native_interval_level!r}"
            )
        self.hyperparameters = hyperparameters or BoostingHyperparameters()
        self.native_interval_level = float(native_interval_level)
        self.p: float | None = None
        self._point: GradientBoostingRegressor | None = None
        self._lower: GradientBoostingRegressor | None = None
        self._upper: GradientBoostingRegressor | None = None

    def _make(self, loss: str, alpha: float | None = None) -> GradientBoostingRegressor:
        hp = self.hyperparameters
        kwargs: dict[str, object] = {
            "loss": loss,
            "n_estimators": hp.n_estimators,
            "max_depth": hp.max_depth,
            "learning_rate": hp.learning_rate,
            "min_samples_leaf": hp.min_samples_leaf,
            "random_state": hp.random_state,
            "subsample": 1.0,
        }
        if alpha is not None:
            kwargs["alpha"] = alpha
        return GradientBoostingRegressor(**kwargs)  # type: ignore[arg-type]

    def fit(self, records: tuple[PipelineRecord, ...], p: float) -> LearnedTailPredictor:
        """Fit the point ensemble and the two quantile heads on ``records``."""
        if not (0.0 < float(p) < 1.0):
            raise ValueError(f"p must lie strictly in (0, 1), got {p!r}")
        if len(records) < 20:
            raise ValueError(
                f"the learned predictor needs at least 20 training pipelines, got {len(records)}"
            )
        self.p = float(p)
        x = feature_matrix(records)
        y = log_target(records, self.p)
        tail = 0.5 * (1.0 - self.native_interval_level)
        self._point = self._make("squared_error").fit(x, y)
        self._lower = self._make("quantile", alpha=tail).fit(x, y)
        self._upper = self._make("quantile", alpha=1.0 - tail).fit(x, y)
        return self

    def _require_fit(self) -> GradientBoostingRegressor:
        if self._point is None:
            raise RuntimeError("call fit(records, p) before predicting")
        return self._point

    def predict_log(self, records: tuple[PipelineRecord, ...]) -> np.ndarray:
        """Point predictions of ``ln q_p``, shape ``(n,)``."""
        return np.asarray(self._require_fit().predict(feature_matrix(records)), dtype=float)

    def predict_log_interval(
        self, records: tuple[PipelineRecord, ...], level: float = 0.9
    ) -> IntervalPrediction:
        """Interval from the two quantile heads, at the level they were fitted for."""
        if abs(float(level) - self.native_interval_level) > 1e-12:
            raise ValueError(
                f"quantile heads were fitted for level {self.native_interval_level}; "
                f"refit with native_interval_level={level} to ask for that level"
            )
        self._require_fit()
        assert self._lower is not None and self._upper is not None
        x = feature_matrix(records)
        point = np.asarray(self._point.predict(x), dtype=float)  # type: ignore[union-attr]
        lower = np.asarray(self._lower.predict(x), dtype=float)
        upper = np.asarray(self._upper.predict(x), dtype=float)
        # Quantile heads are fitted independently and can cross on a few rows;
        # order them rather than reporting a negative-width interval.
        lo = np.minimum(lower, upper)
        hi = np.maximum(lower, upper)
        return IntervalPrediction(point, lo, hi, float(level))

    def feature_importances(self) -> np.ndarray:
        """Impurity-based feature importances of the point ensemble, shape ``(13,)``.

        Impurity importances are biased toward high-cardinality features
        (Strobl et al. 2007, *BMC Bioinformatics* 8:25). Used here only to
        report which features the model leaned on, never as evidence of a
        causal effect.
        """
        return np.asarray(self._require_fit().feature_importances_, dtype=float)
