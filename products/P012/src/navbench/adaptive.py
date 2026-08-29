"""Adaptive process-noise (Q) tuning: a classical baseline and a learned tuner.

THE PROBLEM.  A Kalman filter's process-noise covariance ``Q`` encodes how
much the truth is expected to deviate from the assumed dynamics.  It is
almost never known.  Set it too small and the filter becomes over-confident:
NEES climbs above its chi-squared bound and the estimate lags manoeuvres.
Set it too large and the filter throws away information: NEES falls below the
bound and RMS error rises.  Both failures are invisible to an RMS-only
assessment, which is why :mod:`navbench.consistency` is the scoring tool here.

Everything below tunes a **single scalar** ``λ`` with ``Q = λ Q_nominal``.
Restricting both methods to the same one-dimensional knob is deliberate: it
makes the classical/learned comparison a like-for-like test of *how well the
scale is inferred*, not a test of how many parameters each method is allowed.

CLASSICAL BASELINE — innovation-based adaptive estimation (IAE)

Mehra, R. K. (1970), "On the identification of variances and adaptive Kalman
filtering", *IEEE Transactions on Automatic Control* 15(2), 175-184, and
Mehra, R. K. (1972), "Approaches to adaptive filtering", *IEEE Transactions
on Automatic Control* 17(5), 693-698, derive ``Q`` from the sample covariance
of the innovation sequence.  The compact form used in practice is

    Ĉ = (1/N) Σ_{j∈window} ν_j ν_jᵀ                 (sample innovation covariance)
    Q̂ = K Ĉ Kᵀ                                      (Mehra 1972; Mohamed &
                                                     Schwarz 1999, "Adaptive
                                                     Kalman filtering for
                                                     INS/GPS", J. Geodesy 73,
                                                     193-203, Eq. (12))

with ``K`` the Kalman gain in force over the window.  :class:`MehraAdaptiveQ`
implements exactly that, and additionally projects ``Q̂`` onto the scalar knob
by ``λ = tr(Q̂)/tr(Q_nominal)`` so that it competes on equal terms with the
learned tuner.  The projection is *this package's* choice, not Mehra's, and
is flagged as such.

Known weakness of IAE, stated because it matters for the comparison: ``Q̂ =
K Ĉ Kᵀ`` has rank at most ``m`` (the measurement dimension) and is biased
whenever the window is short compared with the filter's settling time.  It
also has no notion of confidence.

LEARNED TUNER

:class:`LearnedAdaptiveQ` regresses ``log₁₀ λ`` from a fixed set of scale-free
innovation statistics using a bootstrap ensemble of
``sklearn.ensemble.GradientBoostingRegressor``.  The ensemble spread is the
model's **confidence output** — required for AI products in this portfolio —
and the feature bounding box gives an explicit extrapolation flag.

Features (all computed from a sliding window of innovations ``ν`` and the
filter's own reported ``S``; all dimensionless):

    1. log10(mean NIS / m)                      — the primary signal
    2. log10(tr Ĉ / tr S̄)                        — trace ratio
    3. lag-1 autocorrelation of the scalarised normalised innovation
    4. log10(sample variance of normalised innovation, channel 0)
    5. log10(sample variance of normalised innovation, channel 1 or 0 again
       when m = 1)
    6. fraction of |normalised innovation| > 2

Both methods are benchmarked in ``validation/v6_adaptive_q_benchmark.py``
against a **fixed hand-tuned Q** on the same held-out runs.  Whichever wins,
wins; the result is reported as measured.

COMPUTE BUDGET.  Dataset generation and training together take well under a
minute on 2 cores with ``n_jobs=1`` throughout (scikit-learn's gradient
boosting is single-threaded by construction; no ``n_jobs`` is passed because
the estimator does not accept one).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import GradientBoostingRegressor

from .kf import KalmanFilter, symmetrize

__all__ = [
    "N_FEATURES",
    "FEATURE_NAMES",
    "innovation_features",
    "MehraAdaptiveQ",
    "AdaptiveQPrediction",
    "LearnedAdaptiveQ",
    "AdaptiveRunResult",
    "run_adaptive_kf",
    "generate_adaptive_dataset",
]

#: Number of features consumed by :class:`LearnedAdaptiveQ`.
N_FEATURES = 6
FEATURE_NAMES = (
    "log10_mean_nis_per_dof",
    "log10_trace_ratio",
    "lag1_autocorr",
    "log10_var_norm_ch0",
    "log10_var_norm_ch1",
    "frac_abs_gt_2",
)

_LOG_FLOOR = 1e-12


def innovation_features(
    innovations: ArrayLike, innovation_covs: ArrayLike
) -> NDArray[np.float64]:
    """Scale-free statistics of one innovation window.

    Parameters
    ----------
    innovations : array_like, shape (N, m)
        Innovation sequence over the window; rows with NaN are dropped.
    innovation_covs : array_like, shape (N, m, m)
        The filter's own reported innovation covariance at the same steps.

    Returns
    -------
    ndarray, shape (6,)
        The features listed in the module docstring, in the order of
        :data:`FEATURE_NAMES`.

    Raises
    ------
    ValueError
        If fewer than 3 finite innovations remain (the lag-1 autocorrelation
        would be meaningless), or on shape mismatch.
    """
    v = np.atleast_2d(np.asarray(innovations, dtype=float))
    s = np.asarray(innovation_covs, dtype=float)
    if v.ndim != 2:
        raise ValueError(f"innovations must be 2-D (N, m), got shape {v.shape}")
    n, m = v.shape
    if s.shape != (n, m, m):
        raise ValueError(f"innovation_covs must have shape ({n}, {m}, {m}), got {s.shape}")
    keep = np.all(np.isfinite(v), axis=1) & np.all(np.isfinite(s), axis=(1, 2))
    v, s = v[keep], s[keep]
    if v.shape[0] < 3:
        raise ValueError(f"need at least 3 finite innovations in a window, got {v.shape[0]}")

    norm = np.zeros_like(v)
    nis_vals = np.zeros(v.shape[0])
    for k in range(v.shape[0]):
        sk = symmetrize(s[k])
        try:
            chol = np.linalg.cholesky(sk)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"innovation covariance at window index {k} is not positive definite"
            ) from exc
        y = np.linalg.solve(chol, v[k])
        norm[k] = y
        nis_vals[k] = float(y @ y)

    c_hat = (v.T @ v) / v.shape[0]
    s_bar = np.mean(s, axis=0)
    trace_ratio = float(np.trace(c_hat)) / max(float(np.trace(s_bar)), _LOG_FLOOR)

    flat = np.sum(norm, axis=1)
    flat = flat - np.mean(flat)
    denom = float(flat @ flat)
    lag1 = float(flat[:-1] @ flat[1:]) / denom if denom > 0.0 else 0.0

    var0 = float(np.var(norm[:, 0]))
    var1 = float(np.var(norm[:, 1])) if m > 1 else var0
    frac = float(np.mean(np.abs(norm) > 2.0))

    return np.array(
        [
            np.log10(max(float(np.mean(nis_vals)) / m, _LOG_FLOOR)),
            np.log10(max(trace_ratio, _LOG_FLOOR)),
            lag1,
            np.log10(max(var0, _LOG_FLOOR)),
            np.log10(max(var1, _LOG_FLOOR)),
            frac,
        ]
    )


@dataclass
class MehraAdaptiveQ:
    """Classical innovation-based adaptive estimation of ``Q`` (Mehra 1970/1972).

    Parameters
    ----------
    q_nominal : array_like, shape (n, n)
        The nominal process-noise covariance that ``λ`` scales.
    min_scale, max_scale : float
        Clip bounds on ``λ``.  Clipping is mandatory in practice: the IAE
        estimator is unbounded and a short window can produce an arbitrarily
        large or near-zero ``λ``.  Defaults span 6 octaves either way.

    Notes
    -----
    ``estimate_q`` implements the published estimator ``Q̂ = K Ĉ Kᵀ``.
    ``estimate_scale`` is this package's scalar projection ``tr(Q̂)/tr(Q_nom)``,
    added so the classical scheme tunes the same one knob as the learned one.
    """

    q_nominal: ArrayLike
    min_scale: float = 1.0 / 64.0
    max_scale: float = 64.0
    _q_nom: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        q = np.atleast_2d(np.asarray(self.q_nominal, dtype=float))
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise ValueError(f"q_nominal must be square, got shape {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError("q_nominal must be finite")
        if float(np.trace(q)) <= 0.0:
            raise ValueError("q_nominal must have positive trace")
        lo, hi = float(self.min_scale), float(self.max_scale)
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0.0 or hi <= lo:
            raise ValueError(f"require 0 < min_scale < max_scale, got {lo!r}, {hi!r}")
        self.min_scale, self.max_scale = lo, hi
        self._q_nom = symmetrize(q)

    @property
    def dim(self) -> int:
        """State dimension of ``q_nominal``."""
        return int(self._q_nom.shape[0])

    def estimate_q(self, innovations: ArrayLike, gains: ArrayLike) -> NDArray[np.float64]:
        """``Q̂ = K Ĉ Kᵀ`` over a window (Mehra 1972; Mohamed & Schwarz 1999 Eq. 12).

        Parameters
        ----------
        innovations : array_like, shape (N, m)
        gains : array_like, shape (N, n, m) or (n, m)
            Kalman gains over the window; the mean gain is used when a series
            is supplied.
        """
        v = np.atleast_2d(np.asarray(innovations, dtype=float))
        if v.ndim != 2:
            raise ValueError(f"innovations must be 2-D (N, m), got shape {v.shape}")
        keep = np.all(np.isfinite(v), axis=1)
        v = v[keep]
        if v.shape[0] < 2:
            raise ValueError(f"need at least 2 finite innovations, got {v.shape[0]}")
        g = np.asarray(gains, dtype=float)
        if g.ndim == 3:
            g = g[keep] if g.shape[0] == keep.size else g
            k_mat = np.mean(g, axis=0)
        elif g.ndim == 2:
            k_mat = g
        else:
            raise ValueError(f"gains must be 2-D or 3-D, got shape {g.shape}")
        n, m = k_mat.shape
        if n != self.dim:
            raise ValueError(f"gain has {n} rows but q_nominal is {self.dim}x{self.dim}")
        if m != v.shape[1]:
            raise ValueError(f"gain has {m} columns but innovations have {v.shape[1]}")
        c_hat = (v.T @ v) / v.shape[0]
        return symmetrize(k_mat @ c_hat @ k_mat.T)

    def estimate_scale(self, innovations: ArrayLike, gains: ArrayLike) -> float:
        """Scalar ``λ = clip(tr(Q̂)/tr(Q_nominal))`` for the window."""
        q_hat = self.estimate_q(innovations, gains)
        lam = float(np.trace(q_hat)) / float(np.trace(self._q_nom))
        if not np.isfinite(lam) or lam <= 0.0:
            return self.min_scale
        return float(np.clip(lam, self.min_scale, self.max_scale))


@dataclass(frozen=True)
class AdaptiveQPrediction:
    """A learned ``Q``-scale prediction with its uncertainty.

    Attributes
    ----------
    log10_scale : float
        Ensemble-mean prediction of ``log₁₀ λ``.
    log10_std : float
        Ensemble standard deviation of ``log₁₀ λ`` — the model's uncertainty
        output, in decades.
    scale : float
        ``10**log10_scale``, clipped to the model's training range.
    confidence : float
        ``exp(−log10_std)`` in ``(0, 1]``: 1.0 when the ensemble agrees
        exactly, 0.37 at one decade of disagreement.  A monotone reparameter-
        isation of the spread, not a calibrated probability — see MODEL_CARD.
    extrapolating : bool
        True when any feature lies outside the min/max box seen in training.
    """

    log10_scale: float
    log10_std: float
    scale: float
    confidence: float
    extrapolating: bool


class LearnedAdaptiveQ:
    """Bootstrap ensemble of gradient-boosted trees predicting ``log₁₀ λ``.

    Parameters
    ----------
    n_members : int
        Ensemble size, ≥ 2.  Each member is fitted on an independent bootstrap
        resample with its own ``random_state``.
    n_estimators, max_depth, learning_rate, subsample : passthrough
        ``sklearn.ensemble.GradientBoostingRegressor`` hyper-parameters.
    random_state : int
        Master seed; member ``i`` uses ``random_state + i``.
    min_scale, max_scale : float
        Clip bounds applied to the returned ``scale``.

    Notes
    -----
    Gradient boosting is single-threaded in scikit-learn, so this respects the
    ``n_jobs = 1`` budget of the build environment by construction.
    """

    def __init__(
        self,
        n_members: int = 5,
        n_estimators: int = 150,
        max_depth: int = 3,
        learning_rate: float = 0.06,
        subsample: float = 0.85,
        random_state: int = 20260812,
        min_scale: float = 1.0 / 64.0,
        max_scale: float = 64.0,
    ) -> None:
        if int(n_members) < 2:
            raise ValueError(f"n_members must be >= 2 for an ensemble spread, got {n_members!r}")
        if int(n_estimators) < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators!r}")
        lo, hi = float(min_scale), float(max_scale)
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0.0 or hi <= lo:
            raise ValueError(f"require 0 < min_scale < max_scale, got {lo!r}, {hi!r}")
        self.n_members = int(n_members)
        self.random_state = int(random_state)
        self.min_scale, self.max_scale = lo, hi
        self._params = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
        }
        self._members: list[GradientBoostingRegressor] = []
        self._lo: NDArray[np.float64] | None = None
        self._hi: NDArray[np.float64] | None = None
        self.n_train: int = 0

    @property
    def fitted(self) -> bool:
        """True once :meth:`fit` has completed."""
        return bool(self._members)

    def fit(self, x: ArrayLike, y: ArrayLike) -> LearnedAdaptiveQ:
        """Fit the ensemble.

        Parameters
        ----------
        x : array_like, shape (N, 6)
            Feature matrix from :func:`innovation_features`.
        y : array_like, shape (N,)
            Targets ``log₁₀(q_true/q_nominal)``.
        """
        xm = np.atleast_2d(np.asarray(x, dtype=float))
        ym = np.asarray(y, dtype=float).ravel()
        if xm.ndim != 2 or xm.shape[1] != N_FEATURES:
            raise ValueError(f"x must have shape (N, {N_FEATURES}), got {xm.shape}")
        if ym.size != xm.shape[0]:
            raise ValueError(f"y must have {xm.shape[0]} elements, got {ym.size}")
        if not (np.all(np.isfinite(xm)) and np.all(np.isfinite(ym))):
            raise ValueError("x and y must be finite")
        if xm.shape[0] < 20:
            raise ValueError(f"need at least 20 training samples, got {xm.shape[0]}")
        rng = np.random.default_rng(self.random_state)
        self._members = []
        for i in range(self.n_members):
            idx = rng.integers(0, xm.shape[0], size=xm.shape[0])
            model = GradientBoostingRegressor(
                random_state=self.random_state + i, **self._params
            )
            model.fit(xm[idx], ym[idx])
            self._members.append(model)
        self._lo = xm.min(axis=0)
        self._hi = xm.max(axis=0)
        self.n_train = int(xm.shape[0])
        return self

    def predict(self, features: ArrayLike) -> AdaptiveQPrediction:
        """Predict for a single feature vector, returning mean, spread and flags."""
        if not self.fitted:
            raise RuntimeError("LearnedAdaptiveQ is not fitted; call fit() first")
        f = np.asarray(features, dtype=float).ravel()
        if f.size != N_FEATURES:
            raise ValueError(f"features must have {N_FEATURES} elements, got {f.size}")
        if not np.all(np.isfinite(f)):
            raise ValueError("features must be finite")
        preds = np.array([float(m.predict(f[None, :])[0]) for m in self._members])
        mean = float(np.mean(preds))
        std = float(np.std(preds, ddof=1))
        scale = float(np.clip(10.0**mean, self.min_scale, self.max_scale))
        assert self._lo is not None and self._hi is not None
        extrap = bool(np.any(f < self._lo) or np.any(f > self._hi))
        return AdaptiveQPrediction(
            log10_scale=mean,
            log10_std=std,
            scale=scale,
            confidence=float(np.exp(-std)),
            extrapolating=extrap,
        )

    def predict_batch(self, features: ArrayLike) -> tuple[NDArray, NDArray]:
        """Vectorised ``(mean, std)`` of ``log₁₀ λ`` for a feature matrix."""
        if not self.fitted:
            raise RuntimeError("LearnedAdaptiveQ is not fitted; call fit() first")
        f = np.atleast_2d(np.asarray(features, dtype=float))
        if f.ndim != 2 or f.shape[1] != N_FEATURES:
            raise ValueError(f"features must have shape (N, {N_FEATURES}), got {f.shape}")
        preds = np.array([m.predict(f) for m in self._members])
        return preds.mean(axis=0), preds.std(axis=0, ddof=1)


@dataclass(frozen=True)
class AdaptiveRunResult:
    """Outcome of one adaptive-filter run.

    Attributes
    ----------
    states : ndarray, shape (N, n)  — posterior estimates.
    covariances : ndarray, shape (N, n, n)
    innovations : ndarray, shape (N, m)
    innovation_covs : ndarray, shape (N, m, m)
    scales : ndarray, shape (N,) — the ``λ`` in force at each step.
    confidences : ndarray, shape (N,) — NaN for non-learned tuners.
    """

    states: NDArray[np.float64]
    covariances: NDArray[np.float64]
    innovations: NDArray[np.float64]
    innovation_covs: NDArray[np.float64]
    scales: NDArray[np.float64]
    confidences: NDArray[np.float64]


def run_adaptive_kf(
    *,
    f: ArrayLike,
    h: ArrayLike,
    q_nominal: ArrayLike,
    r: ArrayLike,
    x0: ArrayLike,
    p0: ArrayLike,
    measurements: ArrayLike,
    tuner: str = "fixed",
    model: LearnedAdaptiveQ | None = None,
    window: int = 40,
    update_every: int = 20,
    min_scale: float = 1.0 / 64.0,
    max_scale: float = 64.0,
) -> AdaptiveRunResult:
    """Run a linear KF whose ``Q`` scale is re-estimated every ``update_every`` steps.

    Parameters
    ----------
    tuner : {'fixed', 'mehra', 'learned'}
        ``'fixed'`` keeps ``λ = 1`` (the hand-tuned nominal); ``'mehra'`` uses
        :class:`MehraAdaptiveQ`; ``'learned'`` uses ``model``.
    model : LearnedAdaptiveQ, optional
        Required when ``tuner='learned'``; must already be fitted.
    window : int
        Number of past steps used by the estimators, ≥ 5.
    update_every : int
        Re-estimation cadence in steps, ≥ 1.

    Notes
    -----
    The scale in force at step ``k`` is always estimated from data strictly
    before ``k``, so the run is causal and no future information leaks into
    the tuning.
    """
    if tuner not in ("fixed", "mehra", "learned"):
        raise ValueError(f"tuner must be 'fixed', 'mehra' or 'learned', got {tuner!r}")
    if tuner == "learned":
        if model is None:
            raise ValueError("tuner='learned' requires a fitted model")
        if not model.fitted:
            raise RuntimeError("the supplied LearnedAdaptiveQ is not fitted")
    win = int(window)
    cadence = int(update_every)
    if win < 5:
        raise ValueError(f"window must be >= 5, got {window!r}")
    if cadence < 1:
        raise ValueError(f"update_every must be >= 1, got {update_every!r}")

    z = np.atleast_2d(np.asarray(measurements, dtype=float))
    q_nom = symmetrize(np.atleast_2d(np.asarray(q_nominal, dtype=float)))
    kf = KalmanFilter(f, h, q_nom, r, x0, p0)
    n_steps = z.shape[0]
    if z.shape[1] != kf.m:
        raise ValueError(f"measurements must have {kf.m} columns, got {z.shape[1]}")

    mehra = MehraAdaptiveQ(q_nom, min_scale=min_scale, max_scale=max_scale)
    states = np.zeros((n_steps, kf.n))
    covs = np.zeros((n_steps, kf.n, kf.n))
    innov = np.full((n_steps, kf.m), np.nan)
    innov_cov = np.zeros((n_steps, kf.m, kf.m))
    gains = np.zeros((n_steps, kf.n, kf.m))
    scales = np.ones(n_steps)
    confid = np.full(n_steps, np.nan)
    scale = 1.0
    conf = np.nan

    for k in range(n_steps):
        if tuner != "fixed" and k >= win and k % cadence == 0:
            lo = k - win
            w_innov = innov[lo:k]
            w_cov = innov_cov[lo:k]
            if np.all(np.isfinite(w_innov)):
                if tuner == "mehra":
                    scale = mehra.estimate_scale(w_innov, gains[lo:k])
                    conf = np.nan
                else:
                    assert model is not None
                    feats = innovation_features(w_innov, w_cov)
                    pred = model.predict(feats)
                    scale = float(np.clip(pred.scale, min_scale, max_scale))
                    conf = pred.confidence
        kf.predict(q=scale * q_nom)
        out = kf.update(z[k])
        states[k] = out["x"]  # type: ignore[assignment]
        covs[k] = out["p"]  # type: ignore[assignment]
        innov[k] = out["innovation"]  # type: ignore[assignment]
        innov_cov[k] = out["innovation_cov"]  # type: ignore[assignment]
        gains[k] = out["gain"]  # type: ignore[assignment]
        scales[k] = scale
        confid[k] = conf
    return AdaptiveRunResult(states, covs, innov, innov_cov, scales, confid)


def generate_adaptive_dataset(
    *,
    n_runs: int,
    n_steps: int = 400,
    dt: float = 1.0,
    q_nominal_psd: float = 0.05,
    sigma_z: float = 3.0,
    log10_scale_range: tuple[float, float] = (-1.5, 1.5),
    window: int = 40,
    stride: int = 20,
    seed: int = 20260812,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate a supervised dataset of (innovation features → log₁₀ λ_true).

    Each run draws a true acceleration PSD ``q̃_true = q̃_nom · 10^u`` with
    ``u`` uniform on ``log10_scale_range``, simulates a 1-D CWNA
    constant-velocity truth with position-only measurements, runs a KF with
    the **nominal** ``Q``, and emits one feature vector per sliding window.

    Parameters
    ----------
    n_runs : int
        Number of independent runs, ≥ 1.
    n_steps : int
        Steps per run.
    dt : float
        Sample interval [s].
    q_nominal_psd : float
        Nominal acceleration PSD [m²/s³] — the hand-tuned baseline.
    sigma_z : float
        Position measurement noise [m].
    log10_scale_range : (float, float)
        Range of the true log-scale, inclusive.
    window, stride : int
        Sliding-window length and step, in filter steps.
    seed : int
        Master seed; run ``i`` uses ``seed + i``, making the dataset
        deterministic and re-generable run-by-run.

    Returns
    -------
    (features, targets, run_index) with shapes (K, 6), (K,), (K,).
    """
    from .models import constant_velocity_cwna, simulate_linear_system

    runs = int(n_runs)
    if runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs!r}")
    steps = int(n_steps)
    if steps < window + stride:
        raise ValueError(f"n_steps must be >= window + stride = {window + stride}, got {steps}")
    lo_u, hi_u = float(log10_scale_range[0]), float(log10_scale_range[1])
    if not (np.isfinite(lo_u) and np.isfinite(hi_u)) or hi_u <= lo_u:
        raise ValueError(
            f"log10_scale_range must be increasing and finite, got {log10_scale_range!r}"
        )

    f_mat, q_nom = constant_velocity_cwna(dt, q_nominal_psd)
    h = np.array([[1.0, 0.0]])
    r = np.array([[float(sigma_z) ** 2]])
    p0 = np.diag([100.0, 10.0])

    feats: list[NDArray[np.float64]] = []
    targets: list[float] = []
    run_idx: list[int] = []
    for i in range(runs):
        rng = np.random.default_rng(int(seed) + i)
        u = float(rng.uniform(lo_u, hi_u))
        _, q_true = constant_velocity_cwna(dt, q_nominal_psd * 10.0**u)
        _, z = simulate_linear_system(
            f_mat, h, q_true, r, np.array([0.0, 1.0]), steps, rng
        )
        kf = KalmanFilter(f_mat, h, q_nom, r, np.array([0.0, 0.0]), p0)
        res = kf.run(z)
        for start in range(window, steps - window + 1, stride):
            w_innov = res.innovation[start : start + window]
            w_cov = res.innovation_cov[start : start + window]
            if not np.all(np.isfinite(w_innov)):
                continue
            feats.append(innovation_features(w_innov, w_cov))
            targets.append(u)
            run_idx.append(i)
    if not feats:
        raise ValueError("no windows were produced; check n_steps, window and stride")
    return np.array(feats), np.array(targets), np.array(run_idx)
