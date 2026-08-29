r"""Learned adaptive process-noise tuning, with a calibrated confidence output.

What the model does
-------------------
Given a sliding window of filter innovations — quantities available **online,
without truth** — predict ``log₁₀`` of the multiplier that should be applied to
the nominal process-noise matrix ``Q₀``. The features are the seven
dimensionless statistics documented in
:func:`navbench.adaptive.innovation_window_features`; the target is
``log₁₀(s_true)``, the ratio between the process noise that actually generated
the trajectory and ``Q₀``.

Because the applied multiplier is itself a feature, the mapping is a *closed
loop*: the model sees "I am currently running at scale ``s_applied`` and these
are the innovation statistics that resulted", and outputs the scale it believes
is right. Training draws ``s_applied`` and ``s_true`` independently, so the
regression is well posed over the whole plane rather than only along the
diagonal that a naive self-play data collection would visit.

Architecture and why it is small
--------------------------------
An ensemble of five ``sklearn.neural_network.MLPRegressor`` networks
(2 hidden layers, 32 and 16 units, ``tanh``), differing only in
``random_state`` and in the bootstrap resample they see. Seven inputs, one
output. The problem has seven informative features and an irreducible noise
floor set by the window length (see :mod:`navbench.adaptive`), so a larger model
would fit noise. Training takes seconds on 2 CPU cores; PyTorch is not
available in this environment and is not needed.

Uncertainty output — required, and calibrated
---------------------------------------------
The ensemble reports a mean ``μ`` and a spread ``σ`` in ``log₁₀`` units. Raw
ensemble spread systematically *under*-covers, because the members share
architecture, features and (mostly) data. A single scalar calibration factor
``z`` is therefore fitted on a held-out split so that
``[μ − zσ, μ + zσ]`` attains the requested nominal coverage; both the raw and
the calibrated coverage are reported by
``validation/v5_adaptive_q.py`` and in ``MODEL_CARD.md``. This is a
recalibration of an ensemble interval, not a conformal guarantee: it is fitted
on the same synthetic distribution the model is trained on, and it carries no
coverage guarantee off that distribution.

Honesty
-------
This is a *learned tuner*, not a learned filter. It cannot invent information
that is not in the innovations, and the statistical floor derived in
:mod:`navbench.adaptive` applies to it exactly as it does to the classical
schemes. Whether it beats them is an empirical question answered — either way —
in ``validation/VALIDATION.md`` and ``MODEL_CARD.md``.

References
----------
* Mehra, R. K. (1972), "Approaches to adaptive filtering", *IEEE Transactions
  on Automatic Control* **17**(5), 693–698 — the classical framing this model
  is benchmarked against.
* Lakshminarayanan, B., Pritzel, A. and Blundell, C. (2017), "Simple and
  Scalable Predictive Uncertainty Estimation using Deep Ensembles", *NeurIPS
  30* — the deep-ensemble uncertainty estimate used here in miniature.
* Bar-Shalom, Rong Li & Kirubarajan (2001), §5.4 and §11 — consistency and
  adaptive estimation background.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .adaptive import QAdapter, innovation_window_features
from .kf import KalmanFilter
from .models import ConstantVelocity

__all__ = ["QScalePrediction", "QScaleEnsemble", "LearnedQAdapter", "generate_training_data"]

#: Bounds of the log-uniform scale prior used for training data.
LOG10_SCALE_RANGE: tuple[float, float] = (-1.5, 1.5)


@dataclass(frozen=True)
class QScalePrediction:
    """A point estimate with its confidence interval, in linear scale units.

    Attributes
    ----------
    scale : float
        Point estimate ``10^μ`` of the process-noise multiplier.
    log10_mean : float
        Ensemble mean in ``log₁₀`` units.
    log10_std : float
        Raw ensemble standard deviation in ``log₁₀`` units.
    low, high : float
        Calibrated interval on the multiplier (linear units).
    """

    scale: float
    log10_mean: float
    log10_std: float
    low: float
    high: float


def generate_training_data(
    n_episodes: int = 600,
    n_steps: int = 160,
    window: int = 30,
    seed: int = 20260801,
    model: ConstantVelocity | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Simulate episodes and extract ``(features, log10 s_true, log10 s_applied)``.

    Each episode draws ``log₁₀ s_true`` and ``log₁₀ s_applied`` independently and
    uniformly from :data:`LOG10_SCALE_RANGE`, generates a constant-velocity
    trajectory whose true process-noise PSD is ``s_true · q₀``, and runs the
    linear Kalman filter with ``s_applied · Q₀``. Non-overlapping windows of
    ``window`` innovations are taken after a burn-in of ``2·window`` steps so
    the filter covariance has settled.

    Deterministic for a fixed ``seed``. Roughly ``n_episodes ×
    ⌊(n_steps − 2·window)/window⌋`` rows.
    """
    if n_episodes < 1 or n_steps < 3 * window:
        raise ValueError(
            f"need n_episodes >= 1 and n_steps >= 3*window ({3 * window}), "
            f"got {n_episodes}, {n_steps}"
        )
    cv = model if model is not None else ConstantVelocity(dt=1.0, q_psd=0.1, sigma_pos=5.0, dim=2)
    rng = np.random.default_rng(seed)
    lo, hi = LOG10_SCALE_RANGE
    feats: list[NDArray[np.float64]] = []
    targets: list[float] = []
    applied: list[float] = []
    burn = 2 * window
    for _ in range(n_episodes):
        log_true = float(rng.uniform(lo, hi))
        log_app = float(rng.uniform(lo, hi))
        s_true, s_app = 10.0 ** log_true, 10.0 ** log_app
        x0 = np.array([0.0, 10.0, 0.0, -5.0])
        _, zs = cv.simulate(x0, n_steps, rng, q_true_scale=s_true)
        kf = KalmanFilter(
            f=cv.f(), q=cv.q(s_app), h=cv.h(), r=cv.r(), x=x0.copy(), p=np.diag([100.0, 25.0] * 2)
        )
        nus, ss = [], []
        for k in range(n_steps):
            kf.predict()
            info = kf.update(zs[k])
            nus.append(info.innovation)
            ss.append(info.innovation_cov)
        nus_a, ss_a = np.array(nus), np.array(ss)
        k0 = burn
        while k0 + window <= n_steps:
            feats.append(
                innovation_window_features(nus_a[k0:k0 + window], ss_a[k0:k0 + window], s_app)
            )
            targets.append(log_true)
            applied.append(log_app)
            k0 += window
    return np.array(feats), np.array(targets), np.array(applied)


@dataclass
class QScaleEnsemble:
    """Bootstrap ensemble of MLP regressors predicting ``log₁₀ s``.

    Parameters
    ----------
    n_members : int
        Ensemble size.
    hidden : tuple of int
        Hidden layer sizes.
    max_iter : int
        Maximum L-BFGS/Adam iterations per member.
    seed : int
        Base seed; member ``i`` uses ``seed + i``.
    """

    n_members: int = 5
    hidden: tuple[int, ...] = (32, 16)
    max_iter: int = 600
    seed: int = 0
    members: list[MLPRegressor] = field(default_factory=list, repr=False)
    scaler: StandardScaler | None = field(default=None, repr=False)
    calibration_z: float = field(default=1.959963984540054, repr=False)
    raw_coverage: float = field(default=float("nan"), repr=False)
    calibrated_coverage: float = field(default=float("nan"), repr=False)

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` has completed."""
        return bool(self.members) and self.scaler is not None

    def fit(
        self, x: ArrayLike, y: ArrayLike, calibration_fraction: float = 0.25,
        nominal_coverage: float = 0.95,
    ) -> "QScaleEnsemble":
        """Fit the ensemble and calibrate its interval width.

        The last ``calibration_fraction`` of the (shuffled) rows is held out
        from *every* member and used only to fit the scalar interval width
        ``z``. Both raw (``z = 1.96``) and calibrated coverage are stored.
        """
        xa = np.atleast_2d(np.asarray(x, dtype=float))
        ya = np.asarray(y, dtype=float).reshape(-1)
        if xa.shape[0] != ya.size:
            raise ValueError(f"x has {xa.shape[0]} rows but y has {ya.size}")
        if not 0.0 < calibration_fraction < 0.5:
            raise ValueError(f"calibration_fraction must be in (0, 0.5), got {calibration_fraction}")
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(xa.shape[0])
        n_cal = max(16, int(round(calibration_fraction * xa.shape[0])))
        cal_idx, fit_idx = order[:n_cal], order[n_cal:]
        self.scaler = StandardScaler().fit(xa[fit_idx])
        xs = self.scaler.transform(xa[fit_idx])
        yf = ya[fit_idx]
        self.members = []
        for i in range(self.n_members):
            boot = np.random.default_rng(self.seed + 1000 + i).integers(
                0, xs.shape[0], size=xs.shape[0]
            )
            mlp = MLPRegressor(
                hidden_layer_sizes=tuple(self.hidden),
                activation="tanh",
                solver="adam",
                alpha=1e-3,
                learning_rate_init=5e-3,
                max_iter=self.max_iter,
                random_state=self.seed + i,
                early_stopping=False,
            )
            mlp.fit(xs[boot], yf[boot])
            self.members.append(mlp)
        mu, sd = self._raw_predict(xa[cal_idx])
        resid = np.abs(ya[cal_idx] - mu)
        safe_sd = np.maximum(sd, 1e-6)
        self.raw_coverage = float(np.mean(resid <= 1.959963984540054 * safe_sd))
        self.calibration_z = float(np.quantile(resid / safe_sd, nominal_coverage))
        self.calibrated_coverage = float(np.mean(resid <= self.calibration_z * safe_sd))
        return self

    def _raw_predict(self, x: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not self.is_fitted:
            raise RuntimeError("QScaleEnsemble is not fitted; call fit() first")
        assert self.scaler is not None
        xs = self.scaler.transform(np.atleast_2d(np.asarray(x, dtype=float)))
        preds = np.array([m.predict(xs) for m in self.members])
        return preds.mean(axis=0), preds.std(axis=0, ddof=1)

    def predict(self, features: ArrayLike) -> QScalePrediction:
        """Predict for one feature vector, returning point estimate and interval."""
        mu, sd = self._raw_predict(np.atleast_2d(np.asarray(features, dtype=float)))
        m, s = float(mu[0]), float(sd[0])
        half = self.calibration_z * s
        return QScalePrediction(
            scale=float(10.0 ** m),
            log10_mean=m,
            log10_std=s,
            low=float(10.0 ** (m - half)),
            high=float(10.0 ** (m + half)),
        )

    def predict_batch(self, x: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Vectorised ``(mean, std)`` in ``log₁₀`` units."""
        return self._raw_predict(x)


@dataclass
class LearnedQAdapter(QAdapter):
    """Online process-noise scale adapter driven by :class:`QScaleEnsemble`.

    Parameters
    ----------
    ensemble : QScaleEnsemble
        A fitted ensemble.
    window : int
        Innovation window length; must match the training window.
    smoothing : float
        First-order smoothing on the applied scale, in ``(0, 1]``.
    s_min, s_max : float
        Clipping bounds. Defaults match the training prior, so the adapter never
        extrapolates outside the domain it was trained on; :attr:`extrapolating`
        reports when the raw prediction wanted to.
    initial : float
        Scale used until the first full window.
    """

    ensemble: QScaleEnsemble = field(default_factory=QScaleEnsemble)
    window: int = 30
    smoothing: float = 0.5
    s_min: float = 10.0 ** LOG10_SCALE_RANGE[0]
    s_max: float = 10.0 ** LOG10_SCALE_RANGE[1]
    initial: float = 1.0
    name: str = "learned"
    _nu: deque = field(default_factory=deque, repr=False)
    _s: deque = field(default_factory=deque, repr=False)
    _scale: float = field(default=1.0, repr=False)
    _pred: QScalePrediction | None = field(default=None, repr=False)
    extrapolating: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.window < 4:
            raise ValueError(f"window must be >= 4, got {self.window}")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError(f"smoothing must be in (0, 1], got {self.smoothing}")
        if not self.ensemble.is_fitted:
            raise RuntimeError("LearnedQAdapter needs a fitted QScaleEnsemble")
        self._nu = deque(maxlen=int(self.window))
        self._s = deque(maxlen=int(self.window))
        self._scale = float(self.initial)

    def reset(self) -> None:
        """Clear the innovation window and restore the initial scale."""
        self._nu.clear()
        self._s.clear()
        self._scale = float(self.initial)
        self._pred = None
        self.extrapolating = False

    def observe(
        self,
        innovation: ArrayLike,
        innovation_cov: ArrayLike,
        gain: ArrayLike,
        *,
        f: ArrayLike | None = None,
        h: ArrayLike | None = None,
        p_post: ArrayLike | None = None,
        q0: ArrayLike | None = None,
        r: ArrayLike | None = None,
    ) -> None:
        """Absorb one innovation and, once the window is full, re-predict."""
        del gain, f, h, p_post, q0, r
        self._nu.append(np.asarray(innovation, dtype=float).reshape(-1))
        self._s.append(np.atleast_2d(np.asarray(innovation_cov, dtype=float)))
        if len(self._nu) < self._nu.maxlen:
            return
        feats = innovation_window_features(np.array(self._nu), np.array(self._s), self._scale)
        pred = self.ensemble.predict(feats)
        self._pred = pred
        raw = pred.scale
        self.extrapolating = bool(raw < self.s_min or raw > self.s_max)
        target = float(np.clip(raw, self.s_min, self.s_max))
        self._scale = (1.0 - self.smoothing) * self._scale + self.smoothing * target

    @property
    def scale(self) -> float:
        """Current smoothed multiplier."""
        return float(np.clip(self._scale, self.s_min, self.s_max))

    @property
    def prediction(self) -> QScalePrediction | None:
        """Most recent raw prediction, or ``None`` before the first full window."""
        return self._pred

    @property
    def confidence(self) -> tuple[float, float] | None:
        """Calibrated interval on the multiplier, or ``None`` before the first window."""
        return None if self._pred is None else (self._pred.low, self._pred.high)
