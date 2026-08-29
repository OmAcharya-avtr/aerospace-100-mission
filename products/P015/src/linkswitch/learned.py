"""Learned switching policy: predict imminent optical outage, with a confidence output.

The two classical baselines (:mod:`linkswitch.policies`) were built and
validated first. This module adds a supervised predictor of the event

    "the optical channel is down at any step in [t, t + H - 1]"

from a causal window of recent link telemetry, and turns that prediction into a
channel selection by a cost-derived probability threshold.

Model
-----
An ensemble of ``sklearn.ensemble.HistGradientBoostingClassifier`` members, each
fitted to an independent bootstrap resample with its own ``random_state``.
Gradient-boosted decision trees are used because the feature set is small,
tabular and heterogeneous in scale, and because they train in seconds on two
CPU cores. PyTorch is not available in this environment, so no recurrent or
convolutional sequence model was considered.

Confidence output
-----------------
:meth:`OutagePredictor.predict_outage` returns ``(p_mean, p_std)``: the mean
predicted outage probability across ensemble members and the standard deviation
across them. ``p_mean`` is the model's own probabilistic output (its calibration
is measured, not assumed -- see ``MODEL_CARD.md``); ``p_std`` is deep-ensemble
epistemic spread in the sense of Lakshminarayanan, Pritzel & Blundell, "Simple
and Scalable Predictive Uncertainty Estimation using Deep Ensembles",
*NeurIPS 30*, 2017. ``p_std`` is **not** a calibrated error bar on ``p_mean``.

Decision rule
-------------
With optical rate ``R_o`` and RF rate ``R_r``, committing the next step to the
optical channel has expected reward ``R_o (1 - p)`` and to RF ``R_r p_rf``, so
the indifference point is ``p* = 1 - R_r p_rf / R_o``. That value is *derived*,
not tuned. A validation-tuned ``p*`` is also reported separately in
``validation/VALIDATION.md``; it is never selected on the test seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier

from .policies import Policy, shift_causal
from .scenario import LinkTrace

__all__ = [
    "FEATURE_NAMES",
    "LearnedSwitchPolicy",
    "OutagePredictor",
    "TelemetryFeatureConfig",
    "make_features",
    "make_labels",
]

#: Names of the causal telemetry features, in column order.
FEATURE_NAMES: tuple[str, ...] = (
    "margin_last_db",
    "diff1_db",
    "diff4_db",
    "mean_short_db",
    "std_short_db",
    "min_short_db",
    "frac_below_zero_short",
    "mean_long_db",
    "std_long_db",
    "rf_margin_last_db",
)


@dataclass(frozen=True)
class TelemetryFeatureConfig:
    """Window lengths for the causal telemetry features.

    Attributes
    ----------
    short_window : int
        Short rolling window [samples], >= 2. Captures the fade-scale dynamics
        (a few scintillation correlation times).
    long_window : int
        Long rolling window [samples], > ``short_window``. Its mean and standard
        deviation give the model an estimate of the *current* turbulence
        strength, which is the only thing a memoryless threshold cannot see.
    """

    short_window: int = 16
    long_window: int = 128

    def __post_init__(self) -> None:
        if not isinstance(self.short_window, (int, np.integer)) or self.short_window < 2:
            raise ValueError(f"short_window must be an integer >= 2, got {self.short_window!r}")
        if not isinstance(self.long_window, (int, np.integer)):
            raise TypeError("long_window must be an integer")
        if self.long_window <= self.short_window:
            raise ValueError(
                f"long_window ({self.long_window}) must exceed short_window "
                f"({self.short_window})"
            )


def _causal_moments(x: NDArray[np.float64], w: int) -> tuple[NDArray, NDArray]:
    """Causal rolling mean and (population) standard deviation over ``w`` samples.

    Window at index ``t`` covers ``x[max(0, t-w+1) : t+1]`` -- partial at the
    start, never using future samples. Units follow ``x``.
    """
    n = x.size
    idx = np.arange(n)
    lo = np.maximum(idx - w + 1, 0)
    cnt = (idx - lo + 1).astype(float)
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    mean = (c1[idx + 1] - c1[lo]) / cnt
    var = np.maximum((c2[idx + 1] - c2[lo]) / cnt - mean * mean, 0.0)
    return mean, np.sqrt(var)


def _causal_min(x: NDArray[np.float64], w: int) -> NDArray[np.float64]:
    """Causal rolling minimum over ``w`` samples (edge-padded at the start)."""
    pad = np.concatenate((np.full(w - 1, x[0]), x))
    view = np.lib.stride_tricks.sliding_window_view(pad, w)
    return view.min(axis=1)


def _causal_frac_below(x: NDArray[np.float64], w: int, level: float = 0.0) -> NDArray:
    """Causal rolling fraction of samples below ``level`` over ``w`` samples [-]."""
    ind = (x < level).astype(float)
    mean, _ = _causal_moments(ind, w)
    return mean


def make_features(
    trace: LinkTrace, config: TelemetryFeatureConfig | None = None
) -> NDArray[np.float64]:
    """Build the causal feature matrix for one trace.

    Row ``t`` uses only telemetry from steps ``<= t-1``: the raw series is
    delayed one step (:func:`linkswitch.policies.shift_causal`) before any
    window is applied, matching the one-step decision latency the classical
    baselines also face.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_steps, len(FEATURE_NAMES))``, dtype float64. Columns are
        described by :data:`FEATURE_NAMES`; margins are in dB, fractions
        dimensionless.
    """
    if not isinstance(trace, LinkTrace):
        raise TypeError(f"trace must be a LinkTrace, got {type(trace)!r}")
    cfg = config or TelemetryFeatureConfig()
    v = shift_causal(trace.optical_telemetry_db)
    rf = shift_causal(trace.rf_margin_db)
    n = v.size
    if n <= cfg.long_window:
        raise ValueError(
            f"trace has {n} steps but long_window is {cfg.long_window}; "
            "use a longer trace or a shorter window"
        )
    d1 = np.empty(n)
    d1[0] = 0.0
    d1[1:] = v[1:] - v[:-1]
    d4 = np.empty(n)
    d4[:4] = 0.0
    d4[4:] = v[4:] - v[:-4]
    m_s, s_s = _causal_moments(v, cfg.short_window)
    m_l, s_l = _causal_moments(v, cfg.long_window)
    return np.column_stack(
        (
            v,
            d1,
            d4,
            m_s,
            s_s,
            _causal_min(v, cfg.short_window),
            _causal_frac_below(v, cfg.short_window),
            m_l,
            s_l,
            rf,
        )
    )


def make_labels(trace: LinkTrace, horizon: int) -> NDArray[np.bool_]:
    """Label ``y[t] = optical down at any step in [t, t+horizon-1]``.

    Near the end of the trace the window is truncated to the available samples.
    ``horizon = 1`` reduces to "optical is down at step ``t``", which is exactly
    the event the analytic fixed threshold is optimal against.
    """
    if not isinstance(trace, LinkTrace):
        raise TypeError(f"trace must be a LinkTrace, got {type(trace)!r}")
    h = int(horizon)
    if h < 1:
        raise ValueError(f"horizon must be >= 1, got {h}")
    down = ~np.asarray(trace.optical_up, dtype=bool)
    n = down.size
    if h == 1:
        return down.copy()
    # Rolling forward OR via a reversed cumulative maximum on the integer view.
    c = np.concatenate(([0], np.cumsum(down.astype(np.int64))))
    idx = np.arange(n)
    hi = np.minimum(idx + h, n)
    return (c[hi] - c[idx]) > 0


class OutagePredictor:
    """Bootstrap ensemble of gradient-boosted trees predicting imminent optical outage.

    Parameters
    ----------
    horizon : int
        Prediction horizon ``H`` [samples], >= 1.
    n_members : int
        Ensemble size, >= 2 (a spread needs at least two members).
    max_iter : int
        Boosting iterations per member.
    max_leaf_nodes : int
        Tree size per member.
    max_bins : int
        Histogram bins per feature, 2-255. Fewer bins train faster at some cost
        in resolution; 64 was chosen to fit the two-core compute budget.
    learning_rate : float
        Boosting learning rate.
    random_state : int
        Base seed; member ``k`` uses ``random_state + k`` for both its bootstrap
        resample and its own tree randomness.
    feature_config : TelemetryFeatureConfig or None
        Window lengths; the same configuration must be used at fit and predict
        time (enforced by storing it on the instance).
    """

    def __init__(
        self,
        horizon: int = 4,
        n_members: int = 5,
        max_iter: int = 120,
        max_leaf_nodes: int = 15,
        max_bins: int = 64,
        learning_rate: float = 0.1,
        random_state: int = 0,
        feature_config: TelemetryFeatureConfig | None = None,
    ) -> None:
        if int(horizon) < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if int(n_members) < 2:
            raise ValueError(f"n_members must be >= 2 to produce a spread, got {n_members}")
        if int(max_iter) < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if not (2 <= int(max_bins) <= 255):
            raise ValueError(f"max_bins must be in [2, 255], got {max_bins}")
        if not (0.0 < float(learning_rate) <= 1.0):
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        self.horizon = int(horizon)
        self.n_members = int(n_members)
        self.max_iter = int(max_iter)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.max_bins = int(max_bins)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self.feature_config = feature_config or TelemetryFeatureConfig()
        self.members_: list[HistGradientBoostingClassifier] = []
        self.fitted_: bool = False
        self.n_train_rows_: int = 0
        self.train_positive_rate_: float = float("nan")

    def dataset(self, traces: list[LinkTrace]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Stack features and labels for a list of traces."""
        if not traces:
            raise ValueError("traces must be a non-empty list of LinkTrace")
        xs = [make_features(t, self.feature_config) for t in traces]
        ys = [make_labels(t, self.horizon) for t in traces]
        return np.vstack(xs), np.concatenate(ys)

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.bool_],
        max_rows: int | None = 60000,
        rng: np.random.Generator | None = None,
    ) -> OutagePredictor:
        """Fit the ensemble.

        ``max_rows`` subsamples the training matrix (without replacement) before
        fitting, to keep the fit inside the documented compute budget; ``None``
        uses every row. Each member then draws its own bootstrap resample of the
        retained rows.
        """
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=bool)
        if xa.ndim != 2 or xa.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"x must have shape (n, {len(FEATURE_NAMES)}), got {xa.shape}"
            )
        if ya.shape != (xa.shape[0],):
            raise ValueError(f"y must have shape ({xa.shape[0]},), got {ya.shape}")
        if ya.all() or not ya.any():
            raise ValueError(
                "training labels contain a single class; the scenario produces no "
                "outages (or nothing but outages) and no classifier can be fitted"
            )
        gen = rng if rng is not None else np.random.default_rng(self.random_state)
        if max_rows is not None and xa.shape[0] > int(max_rows):
            keep = gen.choice(xa.shape[0], size=int(max_rows), replace=False)
            xa, ya = xa[keep], ya[keep]
        self.n_train_rows_ = int(xa.shape[0])
        self.train_positive_rate_ = float(ya.mean())
        self.members_ = []
        n = xa.shape[0]
        for k in range(self.n_members):
            member_rng = np.random.default_rng(self.random_state + 1009 * k)
            boot = member_rng.integers(0, n, size=n)
            if ya[boot].all() or not ya[boot].any():  # pragma: no cover - very unlikely
                boot = np.arange(n)
            clf = HistGradientBoostingClassifier(
                max_iter=self.max_iter,
                max_leaf_nodes=self.max_leaf_nodes,
                max_bins=self.max_bins,
                learning_rate=self.learning_rate,
                early_stopping=False,
                random_state=self.random_state + k,
            )
            clf.fit(xa[boot], ya[boot])
            self.members_.append(clf)
        self.fitted_ = True
        return self

    def predict_outage(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict ``(p_mean, p_std)`` of imminent optical outage.

        ``p_mean`` is the ensemble-mean probability [-] in [0, 1]; ``p_std`` is
        the standard deviation across members [-], an epistemic spread, not a
        calibrated error bar.
        """
        if not self.fitted_:
            raise RuntimeError("OutagePredictor.predict_outage called before fit()")
        xa = np.asarray(x, dtype=float)
        if xa.ndim != 2 or xa.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"x must have shape (n, {len(FEATURE_NAMES)}), got {xa.shape}")
        probs = np.stack([m.predict_proba(xa)[:, 1] for m in self.members_], axis=0)
        return probs.mean(axis=0), probs.std(axis=0)


@dataclass(frozen=True)
class LearnedSwitchPolicy(Policy):
    """Baseline-3 policy: drop to RF when predicted outage probability exceeds ``p_star``.

    Parameters
    ----------
    predictor : OutagePredictor
        A fitted predictor.
    p_star : float
        Decision threshold on the predicted outage probability [-] in (0, 1).
        The cost-derived value is ``1 - rate_rf / rate_optical``.
    """

    predictor: OutagePredictor
    p_star: float
    name: str = "learned"

    def __post_init__(self) -> None:
        if not isinstance(self.predictor, OutagePredictor):
            raise TypeError("predictor must be an OutagePredictor")
        if not self.predictor.fitted_:
            raise ValueError("predictor must be fitted before building a policy")
        p = float(self.p_star)
        if not (0.0 < p < 1.0) or math.isnan(p):
            raise ValueError(f"p_star must be in (0, 1), got {self.p_star}")

    def confidence(self, trace: LinkTrace) -> tuple[NDArray, NDArray]:
        """Return ``(p_mean, p_std)`` for every step of ``trace``."""
        return self.predictor.predict_outage(make_features(trace, self.predictor.feature_config))

    def _decide(self, optical_telemetry_prev_db, rf_margin_prev_db, trace):
        """Select optical wherever the predicted outage probability is at most ``p_star``."""
        p_mean, _ = self.confidence(trace)
        return p_mean <= float(self.p_star)
