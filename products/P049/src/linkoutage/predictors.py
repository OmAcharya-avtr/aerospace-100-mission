"""Short-horizon outage forecasters: three baselines, then one learned model.

Order matters and is not negotiable. The baselines below were implemented and
validated before the learned model existed, and they are evaluated on exactly
the same held-out rows. If a baseline wins, the baseline winning is the
published result.

The three baselines
-------------------
:class:`ConstantRatePredictor`
    Outputs the training base rate for every row. On a rare-event problem this
    is a strong baseline and leaving it out is the standard way to fool
    oneself: it is *perfectly calibrated* by construction, its Brier score
    equals the problem's uncertainty term, and any model with worse
    reliability than it is worse than useless even if its AUC looks good. It
    has zero resolution, which is the one thing a real forecaster must beat.

:class:`AnalyticLcrPredictor`
    The level-crossing-rate predictor. Fits the lognormal AR(1) channel from
    the training record alone -- scintillation index from the variance of
    ``ln I``, lag-one correlation from the autocorrelation of the standardised
    log amplitude -- and then evaluates the *exact* conditional expected number
    of down-crossings in the horizon given the present log amplitude, via
    :func:`linkoutage.channel.conditional_onset_probability`. No training
    labels are used at all, only the amplitude record and the threshold. This
    is the physics baseline: if a learned model cannot beat a two-parameter
    channel fit, it has learned nothing about the channel.

:class:`LogisticBaseline`
    Standardised features into an L2-regularised logistic regression. The
    linear-model baseline.

The learned model
-----------------
:class:`RandomForestOutageClassifier` -- a random forest over the 14 features
of :mod:`linkoutage.features`, with the **spread of the per-tree votes** as its
uncertainty output, so the forecast is a probability with a stated dispersion
rather than a point estimate. A forest is used rather than a gradient-boosted
ensemble because the per-tree vote distribution is a usable uncertainty, and
rather than a neural network because PyTorch is unavailable in the build
environment and a 14-feature tabular problem does not need one.

:class:`PlattCalibrated` wraps any of the above with a one-parameter logistic
recalibration fitted on the held-out calibration split. Recalibration is
reported separately from the raw model, never folded into it.

Persistence
-----------
Models are stored with :mod:`joblib` (``.joblib``), never as a pickled
framework checkpoint, and every model carries the seed and configuration
needed to regenerate it exactly
(``validation/regenerate_outage_model.py``).

Units
-----
Probabilities dimensionless; horizons in samples; thresholds in the amplitude
unit of the record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .channel import (
    amplitude_threshold_to_gaussian_level,
    conditional_onset_probability,
    sigma_ln_i_from_si,
)

__all__ = [
    "AnalyticLcrPredictor",
    "ConstantRatePredictor",
    "LogisticBaseline",
    "PlattCalibrated",
    "RandomForestOutageClassifier",
    "SupportsOutageProbability",
    "estimate_channel_parameters",
]


@runtime_checkable
class SupportsOutageProbability(Protocol):
    """Anything that turns a feature matrix into onset probabilities."""

    name: str

    def predict_proba_onset(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        ...


@dataclass
class ConstantRatePredictor:
    """Forecast the training base rate, always.

    Attributes
    ----------
    rate:
        The fitted base rate.
    n_train, n_positive_train:
        Behind the rate, so its own sampling error is available.
    """

    name: str = "constant base rate"
    rate: float = float("nan")
    n_train: int = 0
    n_positive_train: int = 0

    def fit(self, y: ArrayLike) -> ConstantRatePredictor:
        """Fit the base rate from training labels only."""
        yy = np.asarray(y, dtype=np.float64).ravel()
        if yy.size == 0:
            raise ValueError("y is empty")
        if not np.all(np.isin(yy, (0.0, 1.0))):
            raise ValueError("y must contain only 0 and 1")
        self.rate = float(yy.mean())
        self.n_train = int(yy.size)
        self.n_positive_train = int(yy.sum())
        return self

    def predict_proba_onset(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if not math.isfinite(self.rate):
            raise RuntimeError("ConstantRatePredictor.fit has not been called")
        n = int(np.asarray(x).shape[0])
        return np.full(n, self.rate, dtype=np.float64)

    def rate_standard_error(self) -> float:
        """Binomial standard error of the fitted rate.

        The honest uncertainty of a constant forecast, and the yardstick for
        whether a difference in mean forecast between two models is resolvable
        at all.
        """
        if self.n_train <= 0:
            return float("nan")
        p = self.rate
        return math.sqrt(p * (1.0 - p) / self.n_train)


@dataclass(frozen=True)
class ChannelParameterEstimate:
    """Channel parameters recovered from an amplitude record.

    Attributes
    ----------
    si:
        Scintillation index from ``var(ln I)``: ``SI = exp(var) - 1``, the
        inverse of ``var = ln(1 + SI)``.
    rho:
        Lag-one autocorrelation of the standardised log amplitude.
    sigma_ln_i:
        ``sqrt(var(ln I))``.
    mean_irradiance:
        Sample mean of ``a**2``.
    n_samples:
        Record length used.
    """

    si: float
    rho: float
    sigma_ln_i: float
    mean_irradiance: float
    n_samples: int

    def describe(self) -> str:
        return (
            f"estimated from {self.n_samples} samples: SI={self.si!r}, "
            f"sigma_lnI={self.sigma_ln_i!r}, rho={self.rho!r}, "
            f"E[I]={self.mean_irradiance!r}"
        )


def estimate_channel_parameters(amplitude: ArrayLike) -> ChannelParameterEstimate:
    """Fit the two-parameter lognormal AR(1) channel to an amplitude record.

    ``ln I = 2 ln a``; ``var(ln I) = ln(1 + SI)`` inverts to
    ``SI = exp(var(ln I)) - 1``; ``rho`` is the lag-one sample autocorrelation
    of the standardised ``ln I``. Uses no labels, so it may legitimately be fitted
    on the training segment of a record and applied to the test segment.
    """
    a = np.asarray(amplitude, dtype=np.float64).ravel()
    if a.size < 3:
        raise ValueError(f"amplitude must have at least 3 samples, got {a.size}")
    if np.any(a <= 0.0) or not np.all(np.isfinite(a)):
        raise ValueError("amplitude must be finite and strictly positive")
    log_i = 2.0 * np.log(a)
    var = float(log_i.var())
    if var <= 0.0:
        raise ValueError(
            "log-irradiance variance is zero; the record is constant and no fading "
            "channel can be identified from it"
        )
    centred = log_i - log_i.mean()
    rho = float(np.dot(centred[:-1], centred[1:]) / np.dot(centred, centred))
    rho = min(max(rho, 0.0), 1.0 - 1e-12)
    return ChannelParameterEstimate(
        si=float(math.expm1(var)),
        rho=rho,
        sigma_ln_i=math.sqrt(var),
        mean_irradiance=float(np.mean(a * a)),
        n_samples=int(a.size),
    )


@dataclass
class AnalyticLcrPredictor:
    """Conditional level-crossing-rate forecaster, no labels used.

    The forecast is
    ``1 - exp(-m(x_t))`` with ``m`` the exact conditional expected number of
    down-crossings in the horizon under the fitted AR(1) lognormal channel
    (see :func:`linkoutage.channel.conditional_onset_probability`). It is
    evaluated on a grid of ``x_t`` and linearly interpolated, because it is a
    function of the single scalar ``x_t`` and nothing else.

    Attributes
    ----------
    threshold, horizon_samples:
        The prediction problem.
    estimate:
        The fitted channel parameters.
    grid, grid_probability:
        The tabulated forecast function.
    """

    threshold: float
    horizon_samples: int
    name: str = "analytic LCR (fit)"
    estimate: ChannelParameterEstimate | None = None
    grid: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    grid_probability: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    level: float = float("nan")
    n_grid: int = 193
    grid_span: float = 6.0

    def fit(
        self, amplitude: ArrayLike, *, estimate: ChannelParameterEstimate | None = None
    ) -> AnalyticLcrPredictor:
        """Fit the channel from a *training segment* of amplitude and tabulate.

        Parameters
        ----------
        amplitude:
            Training segment of the amplitude record. No labels.
        estimate:
            Pre-computed parameters, to bypass the fit (used by the tests to
            feed exact values).
        """
        est = estimate if estimate is not None else estimate_channel_parameters(amplitude)
        self.estimate = est
        self.level = amplitude_threshold_to_gaussian_level(
            self.threshold, est.si, mean_irradiance=est.mean_irradiance
        )
        self.grid = np.linspace(-self.grid_span, self.grid_span, int(self.n_grid))
        self.grid_probability = np.asarray(
            conditional_onset_probability(
                self.grid,
                level=self.level,
                rho=est.rho,
                horizon_samples=int(self.horizon_samples),
            ),
            dtype=np.float64,
        )
        return self

    def gaussian_from_amplitude(self, amplitude: ArrayLike) -> NDArray[np.float64]:
        """Map amplitude to the AR(1) level ``x`` using the fitted parameters.

        ``x = (2 ln a - ln E[I] + sigma**2 / 2) / sigma``.
        """
        if self.estimate is None:
            raise RuntimeError("AnalyticLcrPredictor.fit has not been called")
        a = np.asarray(amplitude, dtype=np.float64).ravel()
        sigma = sigma_ln_i_from_si(self.estimate.si)
        return (
            2.0 * np.log(a) - math.log(self.estimate.mean_irradiance) + sigma * sigma / 2.0
        ) / sigma

    def predict_proba_from_gaussian(self, gaussian: ArrayLike) -> NDArray[np.float64]:
        """Forecast from the AR(1) level directly."""
        if self.estimate is None:
            raise RuntimeError("AnalyticLcrPredictor.fit has not been called")
        g = np.asarray(gaussian, dtype=np.float64).ravel()
        return np.asarray(
            np.interp(g, self.grid, self.grid_probability), dtype=np.float64
        )

    def predict_proba_onset(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forecast from the feature matrix.

        Uses column 0 (``log_amp_last``) only, which is the whole state the
        analytic model conditions on. Passing the feature matrix keeps the
        baseline interchangeable with the learned model in the evaluation
        loop; it does not give the baseline access to the other 13 columns.
        """
        xx = np.asarray(x, dtype=np.float64)
        if xx.ndim != 2 or xx.shape[1] < 1:
            raise ValueError(f"x must be a 2-D feature matrix, got shape {xx.shape}")
        return self.predict_proba_from_gaussian(
            self.gaussian_from_amplitude(np.exp(xx[:, 0]))
        )


@dataclass
class LogisticBaseline:
    """Standardised features into L2 logistic regression."""

    name: str = "logistic regression"
    c: float = 1.0
    max_iter: int = 2000
    pipeline: Pipeline | None = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> LogisticBaseline:
        xx = np.asarray(x, dtype=np.float64)
        yy = np.asarray(y, dtype=np.int64).ravel()
        if xx.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {xx.shape}")
        if xx.shape[0] != yy.size:
            raise ValueError(f"x has {xx.shape[0]} rows but y has {yy.size}")
        if np.unique(yy).size < 2:
            raise ValueError("y must contain both classes to fit a logistic regression")
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(C=self.c, max_iter=self.max_iter, solver="lbfgs"),
                ),
            ]
        )
        self.pipeline.fit(xx, yy)
        return self

    def predict_proba_onset(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.pipeline is None:
            raise RuntimeError("LogisticBaseline.fit has not been called")
        return np.asarray(self.pipeline.predict_proba(np.asarray(x, dtype=np.float64))[:, 1])

    def coefficients(self) -> NDArray[np.float64]:
        """Coefficients on the standardised features, so they are comparable."""
        if self.pipeline is None:
            raise RuntimeError("LogisticBaseline.fit has not been called")
        return np.asarray(self.pipeline.named_steps["logit"].coef_[0], dtype=np.float64)


@dataclass
class RandomForestOutageClassifier:
    """Random forest with the per-tree vote spread as its uncertainty output.

    Attributes
    ----------
    n_estimators, min_samples_leaf, max_depth, seed:
        Forest configuration. ``min_samples_leaf`` is deliberately large (20
        by default): with a 2.6 % base rate a leaf of one sample emits a
        probability of 0 or 1, and those are the forecasts that destroy a
        Brier score.
    """

    name: str = "random forest"
    n_estimators: int = 200
    min_samples_leaf: int = 20
    max_depth: int | None = None
    seed: int = 4901
    n_jobs: int = 2
    model: RandomForestClassifier | None = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> RandomForestOutageClassifier:
        xx = np.asarray(x, dtype=np.float64)
        yy = np.asarray(y, dtype=np.int64).ravel()
        if xx.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {xx.shape}")
        if xx.shape[0] != yy.size:
            raise ValueError(f"x has {xx.shape[0]} rows but y has {yy.size}")
        if np.unique(yy).size < 2:
            raise ValueError("y must contain both classes to fit a forest")
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        self.model.fit(xx, yy)
        return self

    def predict_proba_onset(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.model is None:
            raise RuntimeError("RandomForestOutageClassifier.fit has not been called")
        return np.asarray(self.model.predict_proba(np.asarray(x, dtype=np.float64))[:, 1])

    def predict_with_uncertainty(
        self, x: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(probability, standard_deviation_across_trees)``.

        The dispersion is the standard deviation of the per-tree probability
        estimates. It measures **ensemble disagreement**, which is an
        epistemic-uncertainty proxy: it is large where the training data were
        sparse and the trees extrapolate differently. It is *not* a confidence
        interval on the event: a well-determined probability of 0.5 has small
        tree spread and large outcome uncertainty. Both are reported, and the
        coverage of the spread is measured in
        ``validation/validate_outage_classifier.py``.
        """
        if self.model is None:
            raise RuntimeError("RandomForestOutageClassifier.fit has not been called")
        xx = np.asarray(x, dtype=np.float64)
        per_tree = np.stack(
            [np.asarray(t.predict_proba(xx))[:, 1] for t in self.model.estimators_], axis=0
        )
        return (
            np.asarray(per_tree.mean(axis=0), dtype=np.float64),
            np.asarray(per_tree.std(axis=0, ddof=1), dtype=np.float64),
        )

    def feature_importance(self) -> NDArray[np.float64]:
        """Mean impurity decrease per feature.

        Biased towards high-cardinality features; useful for ordering, not for
        causal reading.
        """
        if self.model is None:
            raise RuntimeError("RandomForestOutageClassifier.fit has not been called")
        return np.asarray(self.model.feature_importances_, dtype=np.float64)


@dataclass
class PlattCalibrated:
    """One-parameter logistic recalibration of another forecaster's output.

    Fitted on a held-out calibration split, never on the training split, and
    reported as a separate row so that the raw model's calibration is visible.
    """

    base: SupportsOutageProbability
    name: str = ""
    max_iter: int = 1000
    _model: LogisticRegression | None = None

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.base.name} + Platt"

    def fit(self, x: ArrayLike, y: ArrayLike) -> PlattCalibrated:
        p = np.asarray(self.base.predict_proba_onset(np.asarray(x, dtype=np.float64)))
        yy = np.asarray(y, dtype=np.int64).ravel()
        if np.unique(yy).size < 2:
            raise ValueError("calibration split must contain both classes")
        eps = 1e-12
        logit = np.log(np.clip(p, eps, 1.0 - eps) / np.clip(1.0 - p, eps, 1.0 - eps))
        if float(np.max(logit) - np.min(logit)) < 1e-12:
            raise ValueError(
                "the base forecaster emits a single value; Platt scaling is "
                "unidentified (this is the expected failure for the constant-rate "
                "baseline, which needs no recalibration)"
            )
        self._model = LogisticRegression(C=1e6, max_iter=self.max_iter, solver="lbfgs")
        self._model.fit(logit.reshape(-1, 1), yy)
        return self

    def predict_proba_onset(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._model is None:
            raise RuntimeError("PlattCalibrated.fit has not been called")
        p = np.asarray(self.base.predict_proba_onset(np.asarray(x, dtype=np.float64)))
        eps = 1e-12
        logit = np.log(np.clip(p, eps, 1.0 - eps) / np.clip(1.0 - p, eps, 1.0 - eps))
        return np.asarray(self._model.predict_proba(logit.reshape(-1, 1))[:, 1])
