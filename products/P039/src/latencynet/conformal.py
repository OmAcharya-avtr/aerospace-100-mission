"""Split-conformal prediction intervals, for any of the three predictors.

The problem with native intervals
---------------------------------
Each of the three models produces an interval from its own assumptions, and
each set of assumptions can be wrong in a different way:

* the analytic model's GUM interval covers probe sampling uncertainty only,
  and nothing of the Fenton-Wilkinson approximation error or of ignored stage
  dependence;
* the OLS interval is exact under linearity and Gaussian homoscedastic error,
  which is an assumption about the data, not a fact about it;
* the quantile heads of the learned model are themselves estimates with no
  finite-sample coverage guarantee.

Split conformal prediction
--------------------------
Fit the model on a training split, compute absolute residuals on a disjoint
calibration split of ``m`` pipelines, and take

    k = ceil((m + 1) (1 - alpha))
    d = the k-th smallest absolute calibration residual
    interval(x) = y_hat(x) +/- d

Under exchangeability of the calibration and test pipelines alone -- no
assumption whatever about the model, the features or the noise -- this gives
marginal coverage at least ``1 - alpha``, and at most ``1 - alpha + 1/(m + 1)``
when the residual distribution is continuous. Sources: Vovk, Gammerman &
Shafer (2005), *Algorithmic Learning in a Random World*, Ch. 2; Lei, G'Sell,
Rinaldo, Tibshirani & Wasserman (2018), "Distribution-Free Predictive
Inference for Regression", *JASA* 113(523): 1094-1111, Sec. 2.2.

Validity range and what it costs
--------------------------------
The guarantee is *marginal*: coverage holds averaged over pipelines, not
conditionally on a particular pipeline. A conformal interval has the same
width everywhere, so it over-covers easy pipelines and under-covers hard
ones. It also inherits the exchangeability assumption: here the calibration
and test pipelines are drawn from the same generator in the same regime, so
exchangeability holds by construction. On a real fleet it would not hold
across, say, a hardware revision, and the guarantee would go with it.

With ``m = 60`` calibration pipelines the achievable coverage is quantised in
steps of ``1/(m + 1) = 1.6 %``, which is reported alongside the measured
coverage so that a 1 % discrepancy is not mistaken for a defect.
"""

from __future__ import annotations

import math

import numpy as np

from .dataset import PipelineRecord, log_target
from .predictors import IntervalPrediction, TailPredictor


def conformal_radius(abs_residuals: np.ndarray, level: float) -> float:
    """The split-conformal radius ``d`` for nominal coverage ``level``.

    Returns ``inf`` when ``m`` is too small for the required order statistic
    to exist, i.e. when ``ceil((m + 1) * level) > m``. That is the honest
    answer: with four calibration points you cannot certify 90 % coverage.
    """
    r = np.asarray(abs_residuals, dtype=float).ravel()
    if r.size == 0:
        raise ValueError("abs_residuals must be non-empty")
    if np.any(r < 0.0) or not np.all(np.isfinite(r)):
        raise ValueError("abs_residuals must be finite and non-negative")
    if not (0.0 < float(level) < 1.0):
        raise ValueError(f"level must lie strictly in (0, 1), got {level!r}")
    m = r.size
    k = math.ceil((m + 1) * float(level))
    if k > m:
        return float("inf")
    return float(np.sort(r)[k - 1])


def coverage_quantisation(n_calibration: int) -> float:
    """Smallest resolvable coverage step, ``1 / (m + 1)``, dimensionless."""
    if not isinstance(n_calibration, (int, np.integer)) or n_calibration < 1:
        raise ValueError(f"n_calibration must be an integer >= 1, got {n_calibration!r}")
    return 1.0 / (int(n_calibration) + 1.0)


class ConformalPredictor(TailPredictor):
    """Wrap a :class:`~latencynet.predictors.TailPredictor` with a conformal interval.

    Parameters
    ----------
    base:
        The predictor to wrap. Its point predictions are used unchanged; only
        the interval is replaced.
    level:
        Nominal coverage. Fixed at construction because the calibration
        residual quantile depends on it.

    The wrapper's :meth:`fit` fits the base model on ``records``;
    :meth:`calibrate` must then be called on a *disjoint* set of pipelines.
    Calling :meth:`predict_log_interval` before calibration raises.
    """

    def __init__(self, base: TailPredictor, level: float = 0.9) -> None:
        if not isinstance(base, TailPredictor):
            raise TypeError("base must be a TailPredictor")
        if not (0.0 < float(level) < 1.0):
            raise ValueError(f"level must lie strictly in (0, 1), got {level!r}")
        self.base = base
        self.level = float(level)
        self.name = f"{base.name}+conformal"
        self.uses_dependence_features = base.uses_dependence_features
        self.radius: float | None = None
        self.n_calibration: int | None = None
        self.p: float | None = None

    @classmethod
    def from_fitted(
        cls, base: TailPredictor, p: float, level: float = 0.9
    ) -> ConformalPredictor:
        """Wrap a base predictor that has already been fitted for ``p``.

        Used by :mod:`latencynet.compare`, where every model is fitted once
        and then wrapped, so that the conformal interval is calibrated against
        exactly the fit whose point predictions it will be attached to.
        """
        wrapper = cls(base, level=level)
        if not (0.0 < float(p) < 1.0):
            raise ValueError(f"p must lie strictly in (0, 1), got {p!r}")
        wrapper.p = float(p)
        return wrapper

    def fit(self, records: tuple[PipelineRecord, ...], p: float) -> ConformalPredictor:
        """Fit the base predictor. Does not calibrate."""
        self.base.fit(records, p)
        self.p = float(p)
        return self

    def calibrate(self, records: tuple[PipelineRecord, ...]) -> ConformalPredictor:
        """Compute the conformal radius on a disjoint calibration split."""
        if self.p is None:
            raise RuntimeError("call fit(records, p) before calibrate(records)")
        if not records:
            raise ValueError("calibration records must be non-empty")
        pred = self.base.predict_log(records)
        truth = log_target(records, self.p)
        self.radius = conformal_radius(np.abs(truth - pred), self.level)
        self.n_calibration = len(records)
        return self

    def predict_log(self, records: tuple[PipelineRecord, ...]) -> np.ndarray:
        """The base predictor's point predictions, unchanged."""
        return self.base.predict_log(records)

    def predict_log_interval(
        self, records: tuple[PipelineRecord, ...], level: float | None = None
    ) -> IntervalPrediction:
        """Conformal interval ``y_hat +/- d`` at the construction-time level."""
        if self.radius is None:
            raise RuntimeError("call calibrate(records) before predict_log_interval")
        if level is not None and abs(float(level) - self.level) > 1e-12:
            raise ValueError(
                f"this wrapper is calibrated for level {self.level}; "
                f"construct another ConformalPredictor for level {level}"
            )
        point = self.base.predict_log(records)
        d = self.radius
        return IntervalPrediction(point, point - d, point + d, self.level)
