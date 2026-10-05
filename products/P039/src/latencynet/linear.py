"""Baseline 2: ordinary least squares with the exact prediction interval.

Model
-----
``y = X beta + eps`` with ``eps ~ N(0, sigma^2 I)``, ``y = ln q_p``, and ``X``
the thirteen probe features of :mod:`latencynet.features` plus an intercept.
Fitted by ``numpy.linalg.lstsq`` on standardised columns, which is a linear
reparameterisation and so leaves every prediction and interval unchanged while
improving the conditioning of ``X'X``.

Prediction interval
-------------------
For a new feature row ``x0`` the exact interval on a *future observation* is

    y_hat +/- t_{(1+level)/2, n-p} * s * sqrt(1 + x0' (X'X)^-1 x0)

with ``s^2 = RSS / (n - p)``. Source: Draper & Smith (1998), *Applied
Regression Analysis*, 3rd ed., Sec. 1.4; Seber & Lee (2003), *Linear
Regression Analysis*, 2nd ed., Sec. 5.3.

Validity range: the interval is exact only under the model's own assumptions
-- linearity in the features, homoscedastic Gaussian errors, and a correctly
specified mean function. None of those is guaranteed here. Working in log
space makes the homoscedasticity assumption more nearly true than it would be
in seconds, and the ``+ 1`` term under the square root is what makes this a
prediction interval rather than a confidence interval on the mean response.
Whether the assumptions hold well enough is answered by the measured coverage
in ``validation/validate_interval_coverage.py``, not asserted here.

Units: ``y`` is a log of seconds; coefficients are per unit of each feature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .dataset import PipelineRecord, feature_matrix, log_target
from .predictors import IntervalPrediction, TailPredictor


@dataclass(frozen=True)
class OLSFit:
    """A fitted OLS model on standardised features.

    Attributes
    ----------
    coef:
        Coefficients on the standardised design including the intercept,
        shape ``(1 + d,)``.
    centre, scale:
        Column means and standard deviations used to standardise, shape
        ``(d,)``. A zero-variance column is given scale 1.0 and so is left at
        its centred value of zero, which removes it from the fit rather than
        dividing by zero.
    xtx_inv:
        Inverse of ``X'X`` on the standardised design with intercept, shape
        ``(1 + d, 1 + d)``.
    residual_std:
        ``s = sqrt(RSS / (n - p))``, in units of ``y``.
    dof:
        ``n - p`` residual degrees of freedom.
    n_obs, n_params:
        ``n`` and ``p``.
    """

    coef: np.ndarray
    centre: np.ndarray
    scale: np.ndarray
    xtx_inv: np.ndarray
    residual_std: float
    dof: int
    n_obs: int
    n_params: int


def _design(x: np.ndarray, centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (np.asarray(x, dtype=float) - centre) / scale
    return np.hstack([np.ones((z.shape[0], 1), dtype=float), z])


def ols_fit(x: np.ndarray, y: np.ndarray) -> OLSFit:
    """Fit OLS of ``y`` on ``x`` with an intercept.

    Requires at least two more observations than parameters, so that the
    residual variance has at least two degrees of freedom.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got shape {x.shape}")
    if x.shape[0] != y.size:
        raise ValueError(f"x has {x.shape[0]} rows but y has {y.size} entries")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must be finite")
    n, d = x.shape
    n_params = d + 1
    if n < n_params + 2:
        raise ValueError(
            f"need at least {n_params + 2} observations for {n_params} parameters, got {n}"
        )
    centre = x.mean(axis=0)
    scale = x.std(axis=0, ddof=1)
    scale = np.where(scale > 0.0, scale, 1.0)
    design = _design(x, centre, scale)
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = n - n_params
    s = math.sqrt(float(resid @ resid) / dof)
    xtx_inv = np.linalg.pinv(design.T @ design)
    return OLSFit(
        coef=coef,
        centre=centre,
        scale=scale,
        xtx_inv=xtx_inv,
        residual_std=s,
        dof=int(dof),
        n_obs=int(n),
        n_params=int(n_params),
    )


def ols_predict(fit: OLSFit, x: np.ndarray) -> np.ndarray:
    """Point predictions, shape ``(n,)``, in units of ``y``."""
    return _design(x, fit.centre, fit.scale) @ fit.coef


def ols_prediction_interval(
    fit: OLSFit, x: np.ndarray, level: float = 0.9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Student-t prediction interval. Returns ``(lower, point, upper)``.

    Half-width ``t * s * sqrt(1 + x0' (X'X)^-1 x0)`` as in the module
    docstring.
    """
    if not (0.0 < float(level) < 1.0):
        raise ValueError(f"level must lie strictly in (0, 1), got {level!r}")
    design = _design(x, fit.centre, fit.scale)
    point = design @ fit.coef
    leverage = np.einsum("ij,jk,ik->i", design, fit.xtx_inv, design)
    t = float(stats.t.ppf(0.5 * (1.0 + float(level)), fit.dof))
    half = t * fit.residual_std * np.sqrt(1.0 + np.maximum(leverage, 0.0))
    return point - half, point, point + half


class LinearTailPredictor(TailPredictor):
    """Baseline 2 as a :class:`~latencynet.predictors.TailPredictor`."""

    name = "linear_ols"
    uses_dependence_features = True

    def __init__(self) -> None:
        self.fit_result: OLSFit | None = None
        self.p: float | None = None

    def fit(self, records: tuple[PipelineRecord, ...], p: float) -> LinearTailPredictor:
        """Fit OLS of ``ln q_p`` on the probe features of ``records``."""
        if not (0.0 < float(p) < 1.0):
            raise ValueError(f"p must lie strictly in (0, 1), got {p!r}")
        self.p = float(p)
        self.fit_result = ols_fit(feature_matrix(records), log_target(records, self.p))
        return self

    def _require_fit(self) -> OLSFit:
        if self.fit_result is None:
            raise RuntimeError("call fit(records, p) before predicting")
        return self.fit_result

    def predict_log(self, records: tuple[PipelineRecord, ...]) -> np.ndarray:
        """Point predictions of ``ln q_p``, shape ``(n,)``."""
        return ols_predict(self._require_fit(), feature_matrix(records))

    def predict_log_interval(
        self, records: tuple[PipelineRecord, ...], level: float = 0.9
    ) -> IntervalPrediction:
        """Exact Student-t prediction interval under the OLS assumptions."""
        lower, point, upper = ols_prediction_interval(
            self._require_fit(), feature_matrix(records), level=level
        )
        return IntervalPrediction(point, lower, upper, float(level))
