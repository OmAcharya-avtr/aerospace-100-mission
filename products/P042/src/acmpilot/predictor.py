"""Channel prediction over the feedback horizon, with a confidence that gates rate.

What is being predicted
-----------------------
Given the delayed SNR reports ``snr[n-d], snr[n-d-1], ..., snr[n-d-L+1]`` the
predictor estimates the distribution of ``snr[n]`` --- the SNR the slot will
actually have. MODCOD selection then uses a **lower** summary of that
distribution, not its centre, so that an uncertain prediction automatically
produces a conservative rate choice. A predictor that emits only a point
estimate cannot do that, and in this application that makes it useless: the cost
of being wrong is asymmetric (a total slot loss against a partial one, see
:mod:`acmpilot.accounting`), so the decision needs a width, not just a mean.

Two predictors are implemented, analytic first
----------------------------------------------
:class:`GaussMarkovPredictor` is **not learned**. For the lognormal channel of
:mod:`acmpilot.channel`, ``snr_dB`` is an affine function of a stationary AR(1)
driver, so the minimum-mean-square-error predictor at horizon ``d`` is exactly
linear:

    E[x[n] | x[n-d]] = mu + rho**d * (x[n-d] - mu)                        (1)
    Var[x[n] | x[n-d]] = s2 * (1 - rho**(2d))                             (2)

with ``mu``, ``s2`` the mean and variance of the dB series and ``rho`` its lag-1
correlation. Equations (1)-(2) are the standard conditional moments of a
stationary Gaussian AR(1) process. All three parameters are **estimated from the
training sample path**, not taken from the channel configuration, so this
predictor has no oracle advantage over the learned one. Its confidence output is
the constant conditional standard deviation from (2) --- it is a
*homoscedastic* predictor, and the single thing a learned model could add is
telling the difference between a quiet moment and a volatile one.

:class:`QuantilePredictor` is learned: three
``sklearn.ensemble.GradientBoostingRegressor`` models with the quantile loss at
10%, 50% and 90%, on lagged features. Its confidence output is the
interquantile spread ``q90 - q10``, which is state-dependent, so unlike (2) it
can widen in the middle of a fade and narrow in a calm stretch. Gradient
boosting is used because PyTorch is unavailable in the build environment and
because the sample counts here (tens of thousands) are in the range where
boosted trees are a reasonable choice.

The gate
--------
Both predictors expose the same selection rule:

    snr_for_selection = centre - gate_k * spread                          (3)

``gate_k = 0`` selects on the centre and is deliberately available, because it
shows how much of the predictor's value is the gate rather than the point
estimate. For a Gaussian predictive distribution and ``spread = q90 - q10``,
``gate_k = 0.5`` is close to selecting on the 10th percentile.

Calibration is reported, not assumed
------------------------------------
Mean absolute error says nothing about whether the gate is trustworthy. What
matters is whether the stated interval covers the truth at the stated rate:
nominal 80% coverage for ``[q10, q90]``, nominal 10% exceedance below ``q10``.
``validation/validate_predictor.py`` reports measured coverage and the
below-``q10`` rate at every feedback delay, next to the error. A predictor with
a good MAE and 40% exceedance below its own 10th percentile would be dangerous
here, and that failure is invisible in an error metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .modcod import ModcodTable

#: Nominal quantile levels of the learned predictor.
QUANTILES: tuple[float, float, float] = (0.1, 0.5, 0.9)


def make_lag_features(
    snr_db: np.ndarray, *, delay_slots: int, n_lags: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Supervised dataset for horizon prediction from delayed reports.

    For each target slot ``n`` the feature row is the ``n_lags`` most recent
    reports available at ``n`` --- that is ``snr[n-d]`` down to
    ``snr[n-d-n_lags+1]`` --- followed by their ``n_lags - 1`` first differences,
    which give the model the local trend explicitly rather than making it
    rediscover a subtraction.

    Parameters
    ----------
    snr_db
        True SNR per slot, dB.
    delay_slots
        Feedback delay in whole slots, >= 0.
    n_lags
        Number of past reports per feature row, >= 1.

    Returns
    -------
    (features, target, slot_index)
        ``features`` shape ``(n_valid, 2*n_lags - 1)``, ``target`` shape
        ``(n_valid,)`` dB, ``slot_index`` the slot each row predicts, so a caller
        can line predictions up with a sample path.
    """
    x = np.asarray(snr_db, dtype=float)
    if delay_slots < 0:
        raise ValueError(f"delay_slots must be >= 0, got {delay_slots}")
    if n_lags < 1:
        raise ValueError(f"n_lags must be >= 1, got {n_lags}")
    first = delay_slots + n_lags - 1
    if first >= x.size:
        raise ValueError(
            f"need > {first} slots for delay_slots={delay_slots}, n_lags={n_lags}, "
            f"got {x.size}"
        )
    slots = np.arange(first, x.size)
    lags = np.stack([x[slots - delay_slots - j] for j in range(n_lags)], axis=1)
    if n_lags > 1:
        diffs = lags[:, :-1] - lags[:, 1:]
        features = np.concatenate([lags, diffs], axis=1)
    else:
        features = lags
    return features, x[slots], slots


@runtime_checkable
class ChannelPredictor(Protocol):
    """Predictor emitting a centre and a spread, both in dB."""

    name: str
    learned: bool

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(centre_db, spread_db)``, each shape ``(n_rows,)``."""
        ...


@dataclass
class GaussMarkovPredictor:
    """Analytic AR(1) MMSE predictor, equations (1)-(2). **Not learned.**

    Attributes are estimated by :meth:`fit` from a training sample path.

    Parameters
    ----------
    delay_slots
        Horizon ``d`` in slots, >= 0.
    spread_sigmas
        The reported ``spread`` is ``spread_sigmas * sqrt(Var)`` from (2). The
        default 2.5631 makes the spread equal to a nominal 80% Gaussian
        interval width (``2 * 1.28155``), so it is directly comparable to the
        learned predictor's ``q90 - q10``.
    """

    delay_slots: int
    spread_sigmas: float = 2.5631
    learned: bool = False
    mean_db: float = float("nan")
    std_db: float = float("nan")
    rho: float = float("nan")

    @property
    def name(self) -> str:
        """Label for tables and plot legends."""
        return "analytic AR(1) MMSE (not learned)"

    def fit(self, snr_db: np.ndarray) -> GaussMarkovPredictor:
        """Estimate ``mu``, ``s2`` and ``rho`` from a training dB series."""
        x = np.asarray(snr_db, dtype=float)
        if x.size < 16:
            raise ValueError(f"need >= 16 training samples, got {x.size}")
        self.mean_db = float(x.mean())
        self.std_db = float(x.std(ddof=1))
        centred = x - self.mean_db
        denom = float(np.dot(centred, centred))
        self.rho = float(np.dot(centred[:-1], centred[1:]) / denom) if denom > 0 else 0.0
        return self

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict from the feature matrix of :func:`make_lag_features`.

        Only column 0 --- the most recent report ``snr[n-d]`` --- is used, which
        is what (1) says is sufficient for an AR(1) process.
        """
        if not np.isfinite(self.mean_db):
            raise RuntimeError("GaussMarkovPredictor.fit must be called before predict")
        f = np.atleast_2d(np.asarray(features, dtype=float))
        last = f[:, 0]
        rho_d = self.rho ** max(self.delay_slots, 0)
        centre = self.mean_db + rho_d * (last - self.mean_db)
        var = (self.std_db**2) * max(0.0, 1.0 - rho_d**2)
        spread = np.full(centre.shape, self.spread_sigmas * float(np.sqrt(var)))
        return centre, spread


@dataclass
class QuantilePredictor:
    """Learned quantile predictor: three gradient-boosting models. **Learned.**

    Parameters
    ----------
    delay_slots
        Horizon in slots, recorded for provenance; the features already encode it.
    n_estimators, max_depth, learning_rate, subsample
        ``GradientBoostingRegressor`` hyperparameters. The defaults are sized for
        the 2-core build budget: three fits over ~30k rows in a few seconds each.
    random_state
        Seed; the fit is deterministic given the data and this value.
    """

    delay_slots: int
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    subsample: float = 0.8
    random_state: int = 20261006
    learned: bool = True
    models: dict[float, GradientBoostingRegressor] | None = None

    @property
    def name(self) -> str:
        """Label for tables and plot legends."""
        return "learned quantile GBR"

    def fit(self, features: np.ndarray, target: np.ndarray) -> QuantilePredictor:
        """Fit one model per level in :data:`QUANTILES`."""
        f = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)
        if f.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {f.shape}")
        if f.shape[0] != y.size:
            raise ValueError(f"{f.shape[0]} feature rows but {y.size} targets")
        self.models = {}
        for q in QUANTILES:
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=self.random_state,
            )
            model.fit(f, y)
            self.models[q] = model
        return self

    def predict_quantiles(self, features: np.ndarray) -> dict[float, np.ndarray]:
        """Predicted q10, q50, q90 in dB, keyed by level.

        The three models are fitted independently, so their outputs can cross at
        a few rows. They are sorted per row here, which is the standard
        non-crossing repair and is reported as applied rather than hidden.
        """
        if self.models is None:
            raise RuntimeError("QuantilePredictor.fit must be called before predict")
        f = np.atleast_2d(np.asarray(features, dtype=float))
        raw = np.stack([self.models[q].predict(f) for q in QUANTILES], axis=1)
        ordered = np.sort(raw, axis=1)
        return {q: ordered[:, i] for i, q in enumerate(QUANTILES)}

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(q50, q90 - q10)`` in dB."""
        q = self.predict_quantiles(features)
        return q[0.5], q[0.9] - q[0.1]

    def crossing_fraction(self, features: np.ndarray) -> float:
        """Fraction of rows where the raw quantile models were out of order."""
        if self.models is None:
            raise RuntimeError("QuantilePredictor.fit must be called before predict")
        f = np.atleast_2d(np.asarray(features, dtype=float))
        raw = np.stack([self.models[q].predict(f) for q in QUANTILES], axis=1)
        return float(np.mean(np.any(np.diff(raw, axis=1) < 0, axis=1)))

    def save(self, path: str | Path) -> None:
        """Persist with joblib. No model binaries are tracked in this repository."""
        if self.models is None:
            raise RuntimeError("nothing to save; call fit first")
        joblib.dump(
            {
                "delay_slots": self.delay_slots,
                "quantiles": QUANTILES,
                "models": self.models,
                "hyperparameters": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "subsample": self.subsample,
                    "random_state": self.random_state,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> QuantilePredictor:
        """Restore a predictor written by :meth:`save`."""
        payload = joblib.load(path)
        obj = cls(delay_slots=int(payload["delay_slots"]), **payload["hyperparameters"])
        obj.models = payload["models"]
        return obj


@dataclass(frozen=True)
class PredictivePolicy:
    """MODCOD selection driven by a predictor and its confidence gate, eq (3).

    This is a :class:`acmpilot.policy.Policy`: it is causal, it never reads the
    true SNR, and it is scored by exactly the same accounting as the baselines.

    Parameters
    ----------
    predictor
        A fitted :class:`ChannelPredictor`.
    delay_slots, n_lags
        Must match the feature construction the predictor was fitted with.
    gate_k
        Confidence gate strength in equation (3), >= 0. ``0`` selects on the
        centre and is the ablation that shows what the gate is worth.
    label
        Optional override for the legend label.
    """

    predictor: ChannelPredictor
    delay_slots: int
    n_lags: int = 8
    gate_k: float = 0.5
    label: str | None = None
    causal: bool = True

    def __post_init__(self) -> None:
        if self.gate_k < 0:
            raise ValueError(f"gate_k must be >= 0, got {self.gate_k}")

    @property
    def name(self) -> str:
        """Label for tables and plot legends."""
        if self.label is not None:
            return self.label
        return f"{self.predictor.name}, gate k={self.gate_k:g}"

    def select(
        self, table: ModcodTable, observed_db: np.ndarray, true_db: np.ndarray
    ) -> np.ndarray:
        """Select per slot from the gated prediction. ``true_db`` is **not read**.

        ``observed_db`` is the already-delayed report series, so the features are
        rebuilt from it with ``delay_slots=0``: shifting twice would double the
        delay. Slots before the first valid feature row fall back to MODCOD 0.
        """
        del true_db
        obs = np.asarray(observed_db, dtype=float)
        features, _, slots = make_lag_features(obs, delay_slots=0, n_lags=self.n_lags)
        centre, spread = self.predictor.predict(features)
        gated = centre - self.gate_k * spread
        chosen = np.zeros(obs.size, dtype=int)
        chosen[slots] = np.maximum(np.atleast_1d(table.best_supported(gated)), 0)
        return chosen


def calibration_report(
    predictor: QuantilePredictor, features: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    """Measured calibration of the learned predictor's stated quantiles.

    Returns
    -------
    dict
        ``coverage_80`` (fraction of targets inside ``[q10, q90]``, nominal
        0.80), ``below_q10`` (nominal 0.10), ``above_q90`` (nominal 0.10),
        ``pinball_q10/q50/q90`` (the quantile loss each model was trained on,
        dB), ``mae_q50`` and ``rmse_q50`` (dB), ``mean_spread_db``,
        ``crossing_fraction``, ``n_rows``.
    """
    q = predictor.predict_quantiles(features)
    y = np.asarray(target, dtype=float)
    out: dict[str, float] = {
        "coverage_80": float(np.mean((y >= q[0.1]) & (y <= q[0.9]))),
        "below_q10": float(np.mean(y < q[0.1])),
        "above_q90": float(np.mean(y > q[0.9])),
        "mae_q50": float(np.mean(np.abs(y - q[0.5]))),
        "rmse_q50": float(np.sqrt(np.mean((y - q[0.5]) ** 2))),
        "mean_spread_db": float(np.mean(q[0.9] - q[0.1])),
        "crossing_fraction": predictor.crossing_fraction(features),
        "n_rows": float(y.size),
    }
    for level in QUANTILES:
        err = y - q[level]
        out[f"pinball_q{int(level * 100)}"] = float(
            np.mean(np.maximum(level * err, (level - 1.0) * err))
        )
    return out


def analytic_calibration_report(
    predictor: GaussMarkovPredictor, features: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    """The same calibration fields for the analytic predictor, equations (1)-(2).

    Its nominal 80% interval is ``centre +/- spread/2``, where ``spread`` is
    ``2 * 1.28155`` conditional standard deviations by construction.
    """
    centre, spread = predictor.predict(features)
    y = np.asarray(target, dtype=float)
    lo, hi = centre - spread / 2.0, centre + spread / 2.0
    return {
        "coverage_80": float(np.mean((y >= lo) & (y <= hi))),
        "below_q10": float(np.mean(y < lo)),
        "above_q90": float(np.mean(y > hi)),
        "mae_q50": float(np.mean(np.abs(y - centre))),
        "rmse_q50": float(np.sqrt(np.mean((y - centre) ** 2))),
        "mean_spread_db": float(np.mean(spread)),
        "crossing_fraction": 0.0,
        "n_rows": float(y.size),
    }
