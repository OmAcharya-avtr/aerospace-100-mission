"""The predictor interface, and the analytic baseline wrapped in it.

Every model in this package predicts the same thing: the natural log of an
end-to-end latency quantile of a pipeline it has never seen, together with an
interval. Three models implement it:

1. :class:`AnalyticTailPredictor` -- baseline 1, the sum-of-stages model of
   :mod:`latencynet.analytic`. It is not fitted: it reads the probe trace's
   per-stage moments and evaluates equations (1)-(3). Its interval is the GUM
   propagation of probe sampling uncertainty.
2. :class:`latencynet.linear.LinearTailPredictor` -- baseline 2, ordinary
   least squares on the thirteen probe features with the exact Student-t
   prediction interval.
3. :class:`latencynet.learned.LearnedTailPredictor` -- the learned model,
   gradient-boosted regression trees with quantile-loss interval heads.

All three can additionally be wrapped by
:class:`latencynet.conformal.ConformalPredictor`, which replaces the native
interval with a split-conformal one calibrated on held-out pipelines. That
makes the coverage comparison fair: the same calibration procedure for every
model, so what is left to compare is interval *width* at matched coverage.

Log space is the working space throughout, so an interval is reported as
``(log_lower, log_point, log_upper)`` and exponentiated only for display.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from .analytic import SumOfStagesModel
from .dataset import PipelineRecord


@dataclass(frozen=True)
class IntervalPrediction:
    """Log-space point prediction and interval for a batch of pipelines.

    Attributes
    ----------
    log_point:
        Point prediction of ``ln q_p``, shape ``(n,)``.
    log_lower, log_upper:
        Interval endpoints in log space, shape ``(n,)``.
    level:
        Nominal coverage of the interval, dimensionless in ``(0, 1)``.
    """

    log_point: np.ndarray
    log_lower: np.ndarray
    log_upper: np.ndarray
    level: float

    def __post_init__(self) -> None:
        if not (self.log_point.shape == self.log_lower.shape == self.log_upper.shape):
            raise ValueError("log_point, log_lower and log_upper must share a shape")
        if not (0.0 < float(self.level) < 1.0):
            raise ValueError(f"level must lie strictly in (0, 1), got {self.level!r}")

    @property
    def log_width(self) -> np.ndarray:
        """Interval width in log space, which is a relative width in seconds."""
        return self.log_upper - self.log_lower

    def seconds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exponentiate to ``(lower_s, point_s, upper_s)``."""
        return np.exp(self.log_lower), np.exp(self.log_point), np.exp(self.log_upper)


class TailPredictor(abc.ABC):
    """Predict ``ln q_p`` for unseen pipelines, with an interval."""

    name: str = "unnamed"
    #: Whether this predictor sees the dependence-bearing probe features.
    uses_dependence_features: bool = False

    @abc.abstractmethod
    def fit(self, records: tuple[PipelineRecord, ...], p: float) -> TailPredictor:
        """Fit for tail probability ``p``. Returns ``self``."""

    @abc.abstractmethod
    def predict_log(self, records: tuple[PipelineRecord, ...]) -> np.ndarray:
        """Point predictions of ``ln q_p``, shape ``(n,)``."""

    @abc.abstractmethod
    def predict_log_interval(
        self, records: tuple[PipelineRecord, ...], level: float = 0.9
    ) -> IntervalPrediction:
        """Native prediction interval at nominal coverage ``level``."""


class AnalyticTailPredictor(TailPredictor):
    """Baseline 1: the analytic sum-of-stages model, as a predictor.

    Parameters
    ----------
    assume_independent:
        ``True`` gives the specified baseline -- variances add, covariances
        ignored. ``False`` uses the probe covariance matrix, which makes
        equation (2) exact and is reported as a diagnostic rather than as the
        baseline.

    There is nothing to fit: ``fit`` records the tail probability and returns.
    That is the point of a baseline -- no training data, no hyperparameters,
    no seed.
    """

    def __init__(self, assume_independent: bool = True) -> None:
        self.model = SumOfStagesModel(assume_independent=assume_independent)
        self.assume_independent = bool(assume_independent)
        self.name = "analytic_sum_indep" if assume_independent else "analytic_sum_cov"
        self.uses_dependence_features = not assume_independent
        self.p: float | None = None

    def fit(self, records: tuple[PipelineRecord, ...], p: float) -> AnalyticTailPredictor:
        """Record the tail probability. The analytic model has no parameters."""
        if not (0.0 < float(p) < 1.0):
            raise ValueError(f"p must lie strictly in (0, 1), got {p!r}")
        self.p = float(p)
        return self

    def _require_p(self) -> float:
        if self.p is None:
            raise RuntimeError("call fit(records, p) before predicting")
        return self.p

    def predict_log(self, records: tuple[PipelineRecord, ...]) -> np.ndarray:
        """Point predictions of ``ln q_p``, shape ``(n,)``."""
        p = self._require_p()
        out = np.empty(len(records), dtype=float)
        for i, rec in enumerate(records):
            cov = None if self.assume_independent else rec.probe.covariance_matrix()
            out[i] = np.log(
                self.model.predict_quantile(
                    np.asarray(rec.probe.stage_mean_s),
                    np.asarray(rec.probe.stage_std_s),
                    p,
                    cov,
                )
            )
        return out

    def predict_log_interval(
        self, records: tuple[PipelineRecord, ...], level: float = 0.9
    ) -> IntervalPrediction:
        """GUM interval from probe sampling uncertainty only. See module docs."""
        p = self._require_p()
        n = len(records)
        point = np.empty(n, dtype=float)
        lower = np.empty(n, dtype=float)
        upper = np.empty(n, dtype=float)
        for i, rec in enumerate(records):
            cov = None if self.assume_independent else rec.probe.covariance_matrix()
            lo, pt, hi = self.model.predict_quantile_interval(
                np.asarray(rec.probe.stage_mean_s),
                np.asarray(rec.probe.stage_std_s),
                np.asarray(rec.probe.stage_m4_s4),
                rec.probe.n_probe,
                p,
                level=level,
                stage_cov_s2=cov,
            )
            lower[i], point[i], upper[i] = np.log(lo), np.log(pt), np.log(hi)
        return IntervalPrediction(point, lower, upper, float(level))
