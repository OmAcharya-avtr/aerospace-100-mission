"""Learned crossing-time forecaster: gradient-boosted quantile regression.

What it is
----------
One :class:`sklearn.ensemble.GradientBoostingRegressor` with
``loss="quantile"`` per quantile level, fitted on the log of the remaining life
in days, on the lag and shape features of :mod:`degradepredict.features`. The
set of fitted quantiles is the predictive distribution; there is no separate
point estimate and no separate uncertainty model, so the uncertainty output is
intrinsic rather than bolted on.

Why a separate model per level, and what that costs: the levels are fitted
independently and can cross. Crossings are removed by a cumulative maximum in
:class:`degradepredict.forecast.QuantileForecast`, following Chernozhukov,
Fernandez-Val and Galichon (2010), Econometrica 78(3):1093-1125.

Censoring
---------
The target is observed only for units whose crossing happened inside the
follow-up. Censored units are not dropped silently: they enter through inverse
probability of censoring weights (:mod:`degradepredict.censoring`), which
re-weight the uncensored units by the reciprocal of their probability of having
been followed long enough. ``weighting="none"`` reproduces the naive
complete-case fit and exists so that the survivorship bias can be measured
rather than asserted; ``validation/validate_headtohead.py`` reports both.

What this does **not** do: it does not implement censored quantile regression
(Portnoy 2003) or Buckley-James imputation (1979). IPCW on a complete-case fit
is the weakest of the three principled options and is chosen because it is
twenty lines of NumPy on top of stock scikit-learn. A product that needed the
best available estimator should use `lifelines` or `scikit-survival`.

References
----------
- Friedman, J. H. (2001), "Greedy function approximation: a gradient boosting
  machine", *Annals of Statistics* 29(5):1189-1232.
- Koenker, R. and Bassett, G. (1978), "Regression quantiles", *Econometrica*
  46(1):33-50. The pinball loss that ``loss="quantile"`` minimises.

Compute
-------
Sized for two cores and a three-minute budget: with the shipped defaults
(7 levels, 300 trees, depth 3) a fit on roughly 500 uncensored training rows and
19 features takes a few seconds. The measured figure is recorded in
``validation/validate_headtohead_output.txt`` and in MODEL_CARD.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .censoring import ipcw_weights
from .features import FEATURE_NAMES, feature_matrix, window_features
from .forecast import QuantileForecast
from .limits import LimitSpec

__all__ = ["DEFAULT_LEVELS", "LearnedCrossingForecaster"]

#: Quantile levels fitted by default. Chosen so that the 0.90 central interval
#: used throughout the repository is read directly off two fitted levels rather
#: than interpolated.
DEFAULT_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

_MIN_REMAINING_DAYS = 1.0


@dataclass
class LearnedCrossingForecaster:
    """Quantile gradient boosting on log remaining life, with IPCW for censoring.

    Parameters
    ----------
    limit:
        The declared limit the features are computed against.
    levels:
        Quantile levels to fit, strictly increasing inside (0, 1).
    weighting:
        ``"ipcw"`` (default) or ``"none"`` for the naive complete-case fit.
    n_estimators, max_depth, learning_rate, min_samples_leaf, subsample:
        Passed through to :class:`~sklearn.ensemble.GradientBoostingRegressor`.
    weight_cap:
        Cap on an individual IPCW weight; see
        :func:`degradepredict.censoring.ipcw_weights`.
    random_state:
        Seed for the boosting subsample draws. Fixed for reproducibility.
    """

    limit: LimitSpec
    levels: tuple[float, ...] = DEFAULT_LEVELS
    weighting: str = "ipcw"
    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.05
    min_samples_leaf: int = 10
    subsample: float = 0.9
    weight_cap: float = 20.0
    random_state: int = 0
    models_: dict[float, GradientBoostingRegressor] = field(default_factory=dict, repr=False)
    n_train_: int = 0
    n_train_used_: int = 0
    feature_names_: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        lv = np.asarray(self.levels, dtype=float)
        if lv.ndim != 1 or lv.size < 2:
            raise ValueError("need at least two quantile levels")
        if np.any(lv <= 0) or np.any(lv >= 1) or not np.all(np.diff(lv) > 0):
            raise ValueError("levels must be strictly increasing inside (0, 1)")
        if self.weighting not in ("ipcw", "none"):
            raise ValueError(f"weighting must be 'ipcw' or 'none', got {self.weighting!r}")
        if self.subsample <= 0 or self.subsample > 1:
            raise ValueError(f"subsample must be in (0, 1], got {self.subsample!r}")

    @property
    def fitted(self) -> bool:
        """True once :meth:`fit` has completed."""
        return bool(self.models_)

    def fit(self, trajectories) -> LearnedCrossingForecaster:
        """Fit one quantile model per level on the uncensored training units.

        Parameters
        ----------
        trajectories:
            Sequence of :class:`degradepredict.generate.Trajectory`. Censored
            units contribute through the IPCW weights, not through a target.

        Raises
        ------
        ValueError
            If fewer than ``2 * len(levels)`` uncensored units are available,
            which is too few to fit the outer quantiles at all.
        """
        trajectories = list(trajectories)
        self.n_train_ = len(trajectories)
        x = feature_matrix(trajectories, self.limit)
        durations = np.array([tr.remaining_observed_days for tr in trajectories], dtype=float)
        events = np.array([tr.event for tr in trajectories], dtype=bool)
        if self.weighting == "ipcw":
            weights = ipcw_weights(durations, events, weight_cap=self.weight_cap)
        else:
            weights = events.astype(float)
        keep = weights > 0
        self.n_train_used_ = int(keep.sum())
        if self.n_train_used_ < 2 * len(self.levels):
            raise ValueError(
                f"only {self.n_train_used_} uncensored training units; need at least "
                f"{2 * len(self.levels)} to fit {len(self.levels)} quantile levels"
            )
        y = np.log(np.maximum(durations[keep], _MIN_REMAINING_DAYS))
        self.models_ = {}
        for q in self.levels:
            gbm = GradientBoostingRegressor(
                loss="quantile",
                alpha=float(q),
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_samples_leaf=self.min_samples_leaf,
                subsample=self.subsample,
                random_state=self.random_state,
            )
            gbm.fit(x[keep], y, sample_weight=weights[keep])
            self.models_[float(q)] = gbm
        return self

    def predict_matrix(self, x: np.ndarray) -> np.ndarray:
        """Predicted remaining-life quantiles in days, shape ``(n, len(levels))``."""
        if not self.fitted:
            raise ValueError("call fit() before predicting")
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"expected {len(self.feature_names_)} features, got {x.shape[1]}"
            )
        cols = [self.models_[float(q)].predict(x) for q in self.levels]
        return np.exp(np.column_stack(cols))

    def predict(self, trajectory) -> QuantileForecast:
        """Predictive distribution over remaining life for one trajectory."""
        x = window_features(trajectory.times_days, trajectory.values_percent, self.limit)
        values = self.predict_matrix(x.reshape(1, -1))[0]
        return QuantileForecast(
            levels=np.asarray(self.levels, dtype=float),
            values_days=values,
            model=f"learned-{self.weighting}",
        )

    def predict_many(self, trajectories) -> list[QuantileForecast]:
        """Predictive distributions for a sequence of trajectories."""
        trajectories = list(trajectories)
        x = feature_matrix(trajectories, self.limit)
        mat = self.predict_matrix(x)
        return [
            QuantileForecast(
                levels=np.asarray(self.levels, dtype=float),
                values_days=row,
                model=f"learned-{self.weighting}",
            )
            for row in mat
        ]

    def feature_importance(self) -> dict[str, float]:
        """Mean impurity-based importance over the fitted levels, summing to 1.

        Impurity importances are biased toward high-cardinality features
        (Strobl, C., Boulesteix, A.-L., Zeileis, A. and Hothorn, T. (2007),
        "Bias in random forest variable importance measures", BMC
        Bioinformatics 8:25). They are reported for inspection only and no
        conclusion in this repository rests on them.
        """
        if not self.fitted:
            raise ValueError("call fit() before asking for importances")
        imp = np.mean([m.feature_importances_ for m in self.models_.values()], axis=0)
        return dict(zip(self.feature_names_, (imp / imp.sum()).tolist(), strict=True))
