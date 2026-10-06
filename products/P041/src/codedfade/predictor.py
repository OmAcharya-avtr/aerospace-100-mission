"""Learned fade-duration exceedance predictor, and the analytic baselines it must beat.

The question
------------
At the instant the received amplitude crosses below a threshold, how long will the
fade last? Specifically, ``P(T > t_target | state observable at the crossing)``.
That probability is what sizes an interleaver: a depth that covers the mean fade
leaves roughly ``exp(-1)`` of fades under-covered, and a designer needs the tail,
conditioned on what the receiver can actually see.

Baselines, implemented and validated first
------------------------------------------
Both are the level-crossing-rate / mean-fade-duration result of ``fade.py`` closed
with the memoryless assumption of equation (16).

``analytic_configured``
    ``P(T > t) = exp(-t / MFD)`` with ``MFD`` from :func:`markov_mean_fade_duration`,
    i.e. the **exact** mean fade duration of the sampled Gauss-Markov path, computed
    from the true scintillation index and correlation time. This baseline is given
    the channel configuration that the learned model is not given. That is
    deliberate: it is the strongest honest form of the analytic result.

``analytic_observed``
    The same expression with ``MFD`` estimated from a trailing window of the
    received amplitude only -- the information the receiver actually has. The local
    log-amplitude standard deviation gives the scintillation index and the local
    lag-1 autocorrelation gives the correlation time; both feed
    :func:`markov_mean_fade_duration`.

Neither baseline uses the state at the crossing, because level-crossing theory does
not supply a conditional distribution. **That is exactly the gap the learned model
has to fill, and the only way it can win.**

The learned model
-----------------
A bagged ensemble of decision trees (scikit-learn ``RandomForestClassifier``) on the
causal features listed in :func:`crossing_features`. The ensemble's per-tree
predicted probabilities give the required **uncertainty output**: the reported
confidence is the standard deviation across trees of the predicted probability, not
a point estimate. Calibration is reported as a reliability table and an expected
calibration error, because a probability that is accurate on average and wrong in
every bin is useless for sizing.

PyTorch is unavailable in this build environment; everything here is scikit-learn
and numpy.

Honesty note
------------
``analytic_observed`` is handed to the model as a feature. The model therefore
starts from the baseline's own answer and can only improve on it by extracting
information from the state. If it does not improve, the published result is that
the analytic result wins, and that is what the README says.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .channel import ChannelConfig, generate_amplitude
from .fade import (
    exponential_exceedance,
    fade_runs,
    lognormal_standard_level,
    markov_mean_fade_duration,
)

#: Number of trailing samples used to form the observable state at a crossing.
WINDOW = 32

FEATURE_NAMES: tuple[str, ...] = (
    "log_depth_at_crossing",
    "log_depth_previous",
    "descent_slope_1",
    "descent_slope_4",
    "window_log_std",
    "window_lag1_autocorr",
    "window_min_log_margin",
    "window_fraction_below",
    "analytic_observed_exceedance",
    "analytic_observed_mfd_samples",
)


@dataclass(frozen=True)
class CrossingDataset:
    """Feature matrix and labels for fade-duration exceedance.

    Attributes
    ----------
    features:
        float64 array ``(n_events, len(FEATURE_NAMES))``.
    labels:
        uint8 array ``(n_events,)``; 1 where the fade exceeded the target duration.
    durations_samples:
        int64 array ``(n_events,)``; the realised fade duration in samples.
    analytic_configured:
        float64 array ``(n_events,)``; the configured-baseline exceedance
        probability for each event.
    group:
        int64 array ``(n_events,)``; the path seed each event came from, so that
        splits never mix events from one realisation across train and test.
    target_samples:
        The exceedance target ``t_target`` in samples.
    """

    features: np.ndarray
    labels: np.ndarray
    durations_samples: np.ndarray
    analytic_configured: np.ndarray
    group: np.ndarray
    target_samples: int


def _local_mfd_samples(
    window: np.ndarray, threshold: float, min_lc: float = 1.5
) -> tuple[float, float]:
    """Mean fade duration in samples from a trailing window of amplitude only.

    Returns ``(mfd_samples, lc_estimate)``. The window's log-amplitude standard
    deviation gives the log-irradiance variance (``ln I = 2 ln a``), hence the
    scintillation index through ``SI = exp(sigma_lnI**2) - 1``; its lag-1
    autocorrelation gives ``L_c = -1/ln(rho)``. Both are then fed to the exact
    sampled-Gauss-Markov expression of equation (15) with ``fs = 1``.
    """
    log_a = np.log(window)
    sigma_lni = float(2.0 * np.std(log_a))
    if not np.isfinite(sigma_lni) or sigma_lni <= 1e-6:
        return float("nan"), float("nan")
    si = float(np.expm1(sigma_lni * sigma_lni))
    centred = log_a - log_a.mean()
    denom = float(np.dot(centred, centred))
    rho = float(np.dot(centred[:-1], centred[1:]) / denom) if denom > 0 else 0.0
    rho = min(max(rho, 1e-6), 1.0 - 1e-6)
    lc = max(-1.0 / np.log(rho), min_lc)
    try:
        u = lognormal_standard_level(threshold, si)
        mfd = markov_mean_fade_duration(u, lc, 1.0)
    except (ValueError, ZeroDivisionError):
        return float("nan"), lc
    if not np.isfinite(mfd):
        return float("nan"), lc
    return float(mfd), lc


def crossing_features(
    amplitude: np.ndarray, threshold: float, target_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract one feature row per complete fade, using only causal information.

    The window is the ``WINDOW`` samples **ending at the crossing sample**, so
    nothing after the crossing enters the features.

    Parameters
    ----------
    amplitude:
        Amplitude series, dimensionless, strictly positive.
    threshold:
        Amplitude threshold, > 0.
    target_samples:
        ``t_target`` in samples, >= 1.

    Returns
    -------
    ``(features, durations_samples, start_indices)``.
    """
    a = np.asarray(amplitude, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"amplitude must be 1-D, got shape {a.shape}")
    if np.any(a <= 0):
        raise ValueError("amplitude must be strictly positive")
    if not threshold > 0:
        raise ValueError(f"threshold must be > 0, got {threshold!r}")
    if target_samples < 1:
        raise ValueError(f"target_samples must be >= 1, got {target_samples!r}")

    below = a < threshold
    starts, lengths = fade_runs(below)
    keep = (starts >= WINDOW) & (starts + lengths < a.size)
    starts, lengths = starts[keep], lengths[keep]
    log_thr = float(np.log(threshold))
    rows = np.empty((starts.size, len(FEATURE_NAMES)), dtype=np.float64)
    log_a = np.log(a)

    for row, (s, _length) in enumerate(zip(starts.tolist(), lengths.tolist(), strict=True)):
        win = a[s - WINDOW + 1 : s + 1]
        log_win = log_a[s - WINDOW + 1 : s + 1]
        mfd, lc = _local_mfd_samples(win, threshold)
        if not np.isfinite(mfd):
            mfd = float(target_samples)
        exceed = float(exponential_exceedance(float(target_samples), max(mfd, 1e-6)))
        rows[row] = (
            log_a[s] - log_thr,
            log_a[s - 1] - log_thr,
            log_a[s] - log_a[s - 1],
            log_a[s] - log_a[s - 4],
            float(np.std(log_win)),
            float(np.corrcoef(log_win[:-1], log_win[1:])[0, 1])
            if np.std(log_win) > 0
            else 0.0,
            float(np.min(log_win) - log_thr),
            float(np.mean(win < threshold)),
            exceed,
            float(np.log10(max(mfd, 1e-6))),
        )
    rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
    return rows, lengths.astype(np.int64), starts.astype(np.int64)


def build_dataset(
    seeds: list[int],
    n_samples: int,
    threshold: float,
    target_samples: int,
    scintillation_index: float = 0.6,
    correlation_time_s: float = 2.0e-4,
    sample_rate_hz: float = 1.0e6,
) -> CrossingDataset:
    """Build a crossing dataset from seeded lognormal Gauss-Markov paths.

    One path per seed. All paths share the channel parameters, so the model sees a
    single regime; the generalisation claim is across **realisations**, not across
    turbulence conditions, and the README says so.
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")
    feats: list[np.ndarray] = []
    durs: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for seed in seeds:
        cfg = ChannelConfig(
            scintillation_index=scintillation_index,
            correlation_time_s=correlation_time_s,
            sample_rate_hz=sample_rate_hz,
            marginal="lognormal",
            kernel="exp",
            seed=int(seed),
        )
        a = generate_amplitude(cfg, n_samples)
        f, d, _ = crossing_features(a, threshold, target_samples)
        if f.size == 0:
            continue
        feats.append(f)
        durs.append(d)
        groups.append(np.full(d.size, int(seed), dtype=np.int64))
    if not feats:
        raise RuntimeError("no complete fades found; lower the threshold or lengthen the path")
    features = np.concatenate(feats, axis=0)
    durations = np.concatenate(durs)
    group = np.concatenate(groups)
    labels = (durations > target_samples).astype(np.uint8)

    cfg = ChannelConfig(
        scintillation_index=scintillation_index,
        correlation_time_s=correlation_time_s,
        sample_rate_hz=sample_rate_hz,
        marginal="lognormal",
        kernel="exp",
        seed=int(seeds[0]),
    )
    u = lognormal_standard_level(threshold, scintillation_index)
    mfd_s = markov_mean_fade_duration(u, correlation_time_s, sample_rate_hz)
    p_conf = float(
        exponential_exceedance(target_samples / sample_rate_hz, mfd_s)
    )
    return CrossingDataset(
        features=features,
        labels=labels,
        durations_samples=durations,
        analytic_configured=np.full(durations.size, p_conf, dtype=np.float64),
        group=group,
        target_samples=int(target_samples),
    )


@dataclass(frozen=True)
class Prediction:
    """Probability with an uncertainty, not a point estimate.

    Attributes
    ----------
    probability:
        Ensemble-mean ``P(T > t_target)``, dimensionless.
    uncertainty:
        Standard deviation of the per-tree probabilities, dimensionless. Large
        values mean the ensemble disagrees and the probability should not be
        trusted to size anything.
    """

    probability: np.ndarray
    uncertainty: np.ndarray


class FadeExceedancePredictor:
    """Bagged-tree classifier for ``P(T > t_target | state)`` with an uncertainty output.

    Parameters
    ----------
    n_estimators:
        Number of trees. 160 fits in a few seconds on one core at the dataset
        sizes used here.
    max_depth:
        Tree depth cap; shallow trees keep the per-tree probabilities from
        collapsing to 0/1, which would make the uncertainty output useless.
    seed:
        ``random_state`` for the forest.
    """

    def __init__(
        self, n_estimators: int = 160, max_depth: int = 8, seed: int = 0
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_leaf=20,
            random_state=int(seed),
            n_jobs=1,
        )
        self._fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray) -> FadeExceedancePredictor:
        """Fit the forest. ``labels`` must contain both classes."""
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels).reshape(-1)
        if x.ndim != 2 or x.shape[0] != y.size:
            raise ValueError(f"shape mismatch: features {x.shape}, labels {y.shape}")
        if np.unique(y).size < 2:
            raise ValueError("labels must contain both classes")
        self.model.fit(x, y)
        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> Prediction:
        """Ensemble-mean probability and per-tree standard deviation."""
        if not self._fitted:
            raise RuntimeError("predictor is not fitted; call fit() first")
        x = np.asarray(features, dtype=np.float64)
        per_tree = np.stack(
            [est.predict_proba(x)[:, 1] for est in self.model.estimators_], axis=0
        )
        return Prediction(
            probability=per_tree.mean(axis=0), uncertainty=per_tree.std(axis=0)
        )

    @property
    def feature_importances(self) -> np.ndarray:
        """Gini importances, aligned with :data:`FEATURE_NAMES`."""
        if not self._fitted:
            raise RuntimeError("predictor is not fitted; call fit() first")
        return np.asarray(self.model.feature_importances_, dtype=np.float64)


def brier_score(probability: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error of a probabilistic forecast. Lower is better."""
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {y.shape}")
    return float(np.mean((p - y) ** 2))


def log_loss_score(probability: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Negative log likelihood per sample, nats. Lower is better."""
    p = np.clip(np.asarray(probability, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def roc_auc(probability: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve by the rank (Mann-Whitney) identity.

    Returns ``nan`` if either class is absent.
    """
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(labels).astype(np.int64)
    n_pos = int(np.count_nonzero(y == 1))
    n_neg = int(np.count_nonzero(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(p.size, dtype=np.float64)
    ranks[order] = np.arange(1, p.size + 1, dtype=np.float64)
    # average ranks within ties
    sorted_p = p[order]
    i = 0
    while i < sorted_p.size:
        j = i
        while j + 1 < sorted_p.size and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def reliability_table(
    probability: np.ndarray, labels: np.ndarray, bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibration table: ``(bin_mean_prediction, bin_observed_frequency, bin_count)``.

    Bins are equal-width on ``[0, 1]``. Empty bins carry ``nan``.
    """
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins!r}")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    mean_p = np.full(bins, np.nan)
    obs = np.full(bins, np.nan)
    count = np.zeros(bins, dtype=np.int64)
    for b in range(bins):
        sel = idx == b
        count[b] = int(np.count_nonzero(sel))
        if count[b]:
            mean_p[b] = float(p[sel].mean())
            obs[b] = float(y[sel].mean())
    return mean_p, obs, count


def expected_calibration_error(
    probability: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    """Count-weighted mean ``|predicted - observed|`` over the reliability bins."""
    mean_p, obs, count = reliability_table(probability, labels, bins)
    sel = count > 0
    if not np.any(sel):
        return float("nan")
    w = count[sel] / count[sel].sum()
    return float(np.sum(w * np.abs(mean_p[sel] - obs[sel])))


def empirical_exceedance_baseline(train_labels: np.ndarray) -> float:
    """State-blind constant baseline: the measured exceedance frequency.

    This is the third baseline and the one that matters most for an honest
    comparison. ``analytic_configured`` closes the level-crossing result with the
    exponential assumption of equation (16); this one keeps the same state-blind
    structure but replaces the assumption with the measured frequency on the
    training paths. Any claim for the learned model must be made against **this**
    number, not against the exponential closure, because beating a miscalibrated
    constant is not evidence that state carries information.
    """
    y = np.asarray(train_labels, dtype=np.float64).reshape(-1)
    if y.size == 0:
        raise ValueError("train_labels must be non-empty")
    return float(y.mean())
