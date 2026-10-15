"""The learned component: duration and energy predictors with calibrated intervals.

Implemented last, after the analytic baseline (:mod:`activityplan.features`) and
after both planners (:mod:`activityplan.greedy`, :mod:`activityplan.exact`), as
the mission's build order requires.

What is learned
---------------
A random-forest regression (Breiman, L., "Random forests", Machine Learning
45(1), 2001) of activity duration [s] and of activity energy [W*h] on the ten
features declared in :data:`activityplan.features.FEATURE_NAMES`. The forest is
a point predictor; the interval comes from a separate calibration step.

Why conformal and not the forest's own spread
---------------------------------------------
The tree-to-tree standard deviation of a random forest is not a prediction
interval and has no coverage guarantee (Wager, S., Hastie, T. and Efron, B.,
"Confidence intervals for random forests: the jackknife and the infinitesimal
jackknife", JMLR 15, 2014). **Split conformal prediction** does have one, it is
distribution-free, and it is finite-sample:

    Papadopoulos, H., Proedrou, K., Vovk, V. and Gammerman, A., "Inductive
    confidence machines for regression", ECML 2002, LNCS 2430, pages 345-356.

    Vovk, V., Gammerman, A. and Shafer, G., *Algorithmic Learning in a Random
    World*, Springer, 2005.

    Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L.,
    "Distribution-free predictive inference for regression", Journal of the
    American Statistical Association 113(523), 2018, pages 1094-1111.

The procedure, exactly as implemented
-------------------------------------
1. Fit the point regressor ``f`` on a **proper training set**.
2. On a **disjoint calibration set** of size ``n``, form nonconformity scores
   ``R_i``. In ``mode="absolute"`` these are ``|y_i - f(x_i)|``; in
   ``mode="normalised"`` they are ``|y_i - f(x_i)| / (s(x_i) + beta)`` where
   ``s`` is a second forest fitted to the **out-of-bag** absolute residuals of
   the training set and ``beta`` is a positive regulariser.
3. Take ``q`` as the ``ceil((n + 1) * (1 - alpha)) / n`` empirical quantile of
   ``R``. If that index exceeds ``n`` the interval is infinite, which is the
   honest answer when the calibration set is too small for the requested level,
   and :attr:`ConformalRegressor.level_attainable` reports it as ``False``.
4. The interval for a new ``x`` is ``f(x) ± q`` (absolute) or
   ``f(x) ± q * (s(x) + beta)`` (normalised).

Guarantee and its limits
------------------------
Under exchangeability of the calibration and test points, the interval covers
with probability at least ``1 - alpha`` **marginally** -- averaged over the
test distribution. It carries **no conditional guarantee**: coverage within a
subgroup, for instance within the longest third of activities, can be well
below ``1 - alpha``. With the heteroscedastic noise the generator in
:mod:`activityplan.features` declares, the absolute variant must undercover long
activities, and ``validation/validate_predictor.py`` measures by how much. That
is the entire reason the normalised variant is implemented, and the measured
comparison between the two is in the README.

Compute
-------
Both forests are fitted with ``n_jobs=1``. On the 2-core container used for this
build, the whole predictor fits in a fraction of a second on the shipped
1400-activity dataset; see the README compute-budget table for measured wall
clocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .features import ActivityRequest, analytic_duration_s, analytic_energy_wh, feature_matrix

__all__ = [
    "ConformalRegressor",
    "DurationEnergyPredictor",
    "Interval",
    "coverage",
    "mean_interval_width",
]


@dataclass(frozen=True)
class Interval:
    """A point prediction with a lower and an upper bound, same unit as the target."""

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        if not (self.point.shape == self.lower.shape == self.upper.shape):
            raise ValueError(
                "point, lower and upper must have the same shape, got "
                f"{self.point.shape}, {self.lower.shape}, {self.upper.shape}"
            )

    @property
    def width(self) -> np.ndarray:
        """``upper - lower`` elementwise."""
        return self.upper - self.lower


class ConformalRegressor:
    """Random-forest point prediction with a split-conformal interval.

    Parameters
    ----------
    alpha
        Miscoverage level, in ``(0, 1)``. The interval targets ``1 - alpha``
        marginal coverage.
    mode
        ``"absolute"`` for a constant-width interval, ``"normalised"`` for one
        scaled by a learned local error magnitude. See the module docstring.
    n_estimators, min_samples_leaf, random_state
        Passed to :class:`sklearn.ensemble.RandomForestRegressor`. ``n_jobs`` is
        fixed at 1 and not exposed.
    beta
        Additive regulariser on the learned scale, as a fraction of the mean
        out-of-bag absolute residual. Only used in ``"normalised"`` mode. A
        strictly positive value is required, otherwise a point with a predicted
        scale of zero would get a zero-width interval.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        mode: str = "absolute",
        n_estimators: int = 200,
        min_samples_leaf: int = 2,
        random_state: int = 0,
        beta: float = 0.1,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if mode not in ("absolute", "normalised"):
            raise ValueError(f"mode must be 'absolute' or 'normalised', got {mode!r}")
        if beta <= 0.0:
            raise ValueError(f"beta must be strictly positive, got {beta!r}")
        self.alpha = float(alpha)
        self.mode = mode
        self.beta = float(beta)
        self._regressor = RandomForestRegressor(
            n_estimators=int(n_estimators),
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
            # Out-of-bag predictions are needed only by the normalised variant,
            # which fits its scale model on them. Requesting them in absolute
            # mode costs nothing but warns on small forests, so it is
            # conditional.
            oob_score=(mode == "normalised"),
            bootstrap=True,
            n_jobs=1,
        )
        self._scale_model: RandomForestRegressor | None = None
        self._scale_offset = 0.0
        self.quantile_: float = math.nan
        self.n_calibration_: int = 0
        self.level_attainable: bool = False

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_calibration: np.ndarray,
        y_calibration: np.ndarray,
    ) -> ConformalRegressor:
        """Fit the point model on the training split and calibrate on the other.

        The two splits must be disjoint; that is the caller's responsibility and
        is what the conformal guarantee rests on.
        :class:`DurationEnergyPredictor` performs the split itself.

        Raises ``ValueError`` for an empty split or mismatched lengths.
        """
        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        x_calibration = np.asarray(x_calibration, dtype=float)
        y_calibration = np.asarray(y_calibration, dtype=float)
        if x_train.ndim != 2 or x_calibration.ndim != 2:
            raise ValueError("x_train and x_calibration must be 2-D")
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x_train and y_train must have the same number of rows")
        if x_calibration.shape[0] != y_calibration.shape[0]:
            raise ValueError("x_calibration and y_calibration must have the same number of rows")
        if x_train.shape[0] < 2 or x_calibration.shape[0] < 1:
            raise ValueError(
                "need at least 2 training rows and 1 calibration row, got "
                f"{x_train.shape[0]} and {x_calibration.shape[0]}"
            )

        self._regressor.fit(x_train, y_train)

        if self.mode == "normalised":
            oob = np.asarray(self._regressor.oob_prediction_, dtype=float)
            residuals = np.abs(y_train - oob)
            self._scale_offset = self.beta * float(residuals.mean())
            if self._scale_offset <= 0.0:
                self._scale_offset = 1e-9
            self._scale_model = RandomForestRegressor(
                n_estimators=self._regressor.n_estimators,
                min_samples_leaf=self._regressor.min_samples_leaf,
                random_state=int(self._regressor.random_state) + 1,
                n_jobs=1,
            )
            self._scale_model.fit(x_train, residuals)

        scores = np.abs(y_calibration - self._regressor.predict(x_calibration))
        if self.mode == "normalised":
            scores = scores / self._local_scale(x_calibration)
        n = int(scores.size)
        self.n_calibration_ = n
        rank = math.ceil((n + 1) * (1.0 - self.alpha))
        if rank > n:
            self.quantile_ = math.inf
            self.level_attainable = False
        else:
            self.quantile_ = float(np.sort(scores)[rank - 1])
            self.level_attainable = True
        return self

    def _local_scale(self, x: np.ndarray) -> np.ndarray:
        if self._scale_model is None:
            return np.ones(x.shape[0], dtype=float)
        predicted = np.maximum(0.0, self._scale_model.predict(x))
        return predicted + self._scale_offset

    def predict(self, x: np.ndarray) -> Interval:
        """Point prediction and interval for the rows of ``x``.

        Raises ``ValueError`` if :meth:`fit` has not been called.
        """
        if math.isnan(self.quantile_):
            raise ValueError("ConformalRegressor.predict called before fit")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be 2-D")
        point = np.asarray(self._regressor.predict(x), dtype=float)
        half_width = self.quantile_ * (
            self._local_scale(x) if self.mode == "normalised" else np.ones(x.shape[0])
        )
        return Interval(point=point, lower=point - half_width, upper=point + half_width)

    def predict_point(self, x: np.ndarray) -> np.ndarray:
        """Point prediction only, without forming the interval."""
        return np.asarray(self._regressor.predict(np.asarray(x, dtype=float)), dtype=float)


@dataclass
class DurationEnergyPredictor:
    """Conformal predictors for activity duration [s] and energy [W*h].

    Both targets use the same features and the same train/calibration split, so
    an interval on one and an interval on the other are calibrated on the same
    exchangeable sample. They are **not jointly calibrated**: the probability
    that both intervals cover simultaneously is not ``1 - alpha`` and is not
    bounded below by anything this package computes beyond the union bound
    ``1 - 2*alpha``. That is stated in ``MODEL_CARD.md`` under failure cases.

    Parameters
    ----------
    alpha
        Per-target miscoverage level.
    mode
        ``"absolute"`` or ``"normalised"``; see :class:`ConformalRegressor`.
    calibration_fraction
        Fraction of the fitting data held out for calibration, in ``(0, 1)``.
    random_state
        Seeds both forests and the train/calibration split.
    """

    alpha: float = 0.1
    mode: str = "absolute"
    calibration_fraction: float = 0.35
    n_estimators: int = 200
    min_samples_leaf: int = 2
    random_state: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError(
                f"calibration_fraction must be in (0, 1), got {self.calibration_fraction!r}"
            )
        self.duration = ConformalRegressor(
            alpha=self.alpha,
            mode=self.mode,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.energy = ConformalRegressor(
            alpha=self.alpha,
            mode=self.mode,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state + 101,
        )
        self.n_train_: int = 0
        self.n_calibration_: int = 0

    def fit(
        self, x: np.ndarray, duration_s: np.ndarray, energy_wh: np.ndarray
    ) -> DurationEnergyPredictor:
        """Split ``x`` into proper-training and calibration parts and fit both targets.

        The split is a seeded permutation, so it is reproducible from
        ``random_state`` alone.
        """
        x = np.asarray(x, dtype=float)
        duration_s = np.asarray(duration_s, dtype=float)
        energy_wh = np.asarray(energy_wh, dtype=float)
        n = x.shape[0]
        if duration_s.shape[0] != n or energy_wh.shape[0] != n:
            raise ValueError("x, duration_s and energy_wh must have the same number of rows")
        n_calibration = int(round(self.calibration_fraction * n))
        if n_calibration < 1 or n - n_calibration < 2:
            raise ValueError(
                f"{n} rows cannot be split into a usable train/calibration pair at "
                f"calibration_fraction {self.calibration_fraction!r}"
            )
        order = np.random.default_rng(self.random_state).permutation(n)
        calibration_index = order[:n_calibration]
        train_index = order[n_calibration:]
        self.n_train_ = int(train_index.size)
        self.n_calibration_ = int(calibration_index.size)
        self.duration.fit(
            x[train_index],
            duration_s[train_index],
            x[calibration_index],
            duration_s[calibration_index],
        )
        self.energy.fit(
            x[train_index],
            energy_wh[train_index],
            x[calibration_index],
            energy_wh[calibration_index],
        )
        return self

    def predict_requests(
        self, requests: tuple[ActivityRequest, ...] | list[ActivityRequest]
    ) -> tuple[Interval, Interval]:
        """``(duration_interval [s], energy_interval [W*h])`` for the requests."""
        x = feature_matrix(list(requests))
        return self.duration.predict(x), self.energy.predict(x)


def coverage(y: np.ndarray, interval: Interval) -> float:
    """Empirical fraction of ``y`` inside ``[lower, upper]``, inclusive.

    Inclusive because the conformal bound is attained with equality at the
    calibration quantile, and an exclusive test would reject exactly the point
    the construction is designed to admit.
    """
    y = np.asarray(y, dtype=float)
    if y.shape != interval.point.shape:
        raise ValueError(f"y shape {y.shape} does not match interval shape {interval.point.shape}")
    inside = (y >= interval.lower) & (y <= interval.upper)
    return float(np.mean(inside))


def mean_interval_width(interval: Interval) -> float:
    """Mean of ``upper - lower``, in the unit of the target."""
    return float(np.mean(interval.width))


def analytic_predictions(
    requests: tuple[ActivityRequest, ...] | list[ActivityRequest],
) -> tuple[np.ndarray, np.ndarray]:
    """``(duration_s, energy_wh)`` from the analytic model, for the same requests.

    The baseline the learned model is scored against, evaluated on the same rows
    so the comparison is paired.
    """
    duration = np.array([analytic_duration_s(request) for request in requests], dtype=float)
    energy = np.array([analytic_energy_wh(request) for request in requests], dtype=float)
    return duration, energy
