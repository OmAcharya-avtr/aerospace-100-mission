"""Lag and shape features extracted from one telemetry window.

Every feature is computed from the observation window ``[0, t0]`` alone. None
uses the crossing time, the follow-up duration or the censoring indicator, so no
label information can reach the learned model through the feature vector.

The feature vector deliberately **includes** a linear extrapolation
time-to-limit, ``log_ttl_linear``, which is the linear baseline's point
estimate. A practitioner would compute it, and withholding it would make the
head-to-head comparison flattering rather than informative. What the learned
model is **not** given is any baseline's *interval*, its residual variance
scaling, or any other baseline output beyond that single extrapolation; a
stacked model that is given all three baselines' full predictive distributions
is not implemented here.

Units are stated per feature in :data:`FEATURE_UNITS`. Time is days, the metric
is in the units of the declared limit.
"""

from __future__ import annotations

import numpy as np

from .limits import LimitSpec, margin

__all__ = ["FEATURE_NAMES", "FEATURE_UNITS", "feature_matrix", "window_features"]

_EPS = 1e-9
_RECENT = 10

FEATURE_NAMES: tuple[str, ...] = (
    "origin_days",
    "n_samples",
    "cadence_days",
    "margin_last",
    "margin_smoothed",
    "fraction_consumed",
    "slope_full",
    "slope_recent",
    "slope_ratio",
    "curvature",
    "resid_sd",
    "log_ttl_linear",
    "log_ttl_recent",
    "log_rate_exponential",
    "log_ttl_exponential",
    "powerlaw_resid_ratio",
    "lag1_rate",
    "lag4_rate",
    "lag8_rate",
)

FEATURE_UNITS: dict[str, str] = {
    "origin_days": "day",
    "n_samples": "count",
    "cadence_days": "day",
    "margin_last": "metric unit",
    "margin_smoothed": "metric unit",
    "fraction_consumed": "dimensionless",
    "slope_full": "metric unit per day",
    "slope_recent": "metric unit per day",
    "slope_ratio": "dimensionless",
    "curvature": "metric unit per day squared",
    "resid_sd": "metric unit",
    "log_ttl_linear": "log day",
    "log_ttl_recent": "log day",
    "log_rate_exponential": "log metric unit per day",
    "log_ttl_exponential": "log day",
    "powerlaw_resid_ratio": "dimensionless",
    "lag1_rate": "metric unit per day",
    "lag4_rate": "metric unit per day",
    "lag8_rate": "metric unit per day",
}


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of ``y`` on ``x``; 0.0 if ``x`` has no spread."""
    xb = x.mean()
    sxx = float(np.sum((x - xb) ** 2))
    if sxx <= _EPS:
        return 0.0
    return float(np.sum((x - xb) * (y - y.mean())) / sxx)


def _safe_log(value: float) -> float:
    return float(np.log(max(float(value), _EPS)))


def window_features(
    times_days: np.ndarray, values: np.ndarray, limit: LimitSpec
) -> np.ndarray:
    """Feature vector for one window, shape ``(len(FEATURE_NAMES),)``, all finite.

    Raises
    ------
    ValueError
        If the window has fewer than three samples, is not finite, or the times
        are not strictly increasing.
    """
    t = np.asarray(times_days, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.ndim != 1 or t.shape != y.shape:
        raise ValueError("times and values must be one-dimensional and the same length")
    if t.size < 3:
        raise ValueError(f"need at least three samples, got {t.size}")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
        raise ValueError("times and values must all be finite")
    if not np.all(np.diff(t) > 0):
        raise ValueError("times must be strictly increasing")

    n = int(t.size)
    t0 = float(t[-1])
    cadence = float(np.median(np.diff(t)))
    m = margin(y, limit)
    margin_last = float(m[-1])
    margin_smoothed = float(np.mean(m[-3:]))
    margin_first = float(m[0])
    consumed = (margin_first - margin_smoothed) / max(abs(margin_first), _EPS)

    slope_full = _ols_slope(t, y)
    k = min(_RECENT, n)
    slope_recent = _ols_slope(t[-k:], y[-k:])
    slope_ratio = slope_recent / slope_full if abs(slope_full) > _EPS else 0.0
    curvature = (slope_recent - slope_full) / max(t0, _EPS)

    inter = y.mean() - slope_full * t.mean()
    resid = y - (inter + slope_full * t)
    resid_sd = float(np.sqrt(np.sum(resid**2) / max(n - 2, 1)))

    decline_full = max(-slope_full, _EPS) if limit.direction == "below" else max(slope_full, _EPS)
    decline_recent = (
        max(-slope_recent, _EPS) if limit.direction == "below" else max(slope_recent, _EPS)
    )
    log_ttl_linear = _safe_log(max(margin_smoothed, 0.0) / decline_full)
    log_ttl_recent = _safe_log(max(margin_smoothed, 0.0) / decline_recent)

    positive = np.all(y > 0) and limit.value > 0
    if positive:
        log_rate = _ols_slope(t, np.log(y))
        rate = max(-log_rate, _EPS)
        log_ttl_exp = _safe_log(max(float(np.log(max(y[-1], _EPS) / limit.value)), 0.0) / rate)
    else:
        log_rate = 0.0
        log_ttl_exp = log_ttl_linear

    x_pl = np.power(t, 0.55)
    slope_pl = _ols_slope(x_pl, y)
    inter_pl = y.mean() - slope_pl * x_pl.mean()
    rss_pl = float(np.sum((y - (inter_pl + slope_pl * x_pl)) ** 2))
    rss_lin = float(np.sum(resid**2))
    resid_ratio = rss_pl / max(rss_lin, _EPS)

    def lag_rate(step: int) -> float:
        j = min(step, n - 1)
        return float((y[-1] - y[-1 - j]) / max(t[-1] - t[-1 - j], _EPS))

    out = np.array(
        [
            t0,
            float(n),
            cadence,
            margin_last,
            margin_smoothed,
            float(consumed),
            slope_full,
            slope_recent,
            float(slope_ratio),
            float(curvature),
            resid_sd,
            log_ttl_linear,
            log_ttl_recent,
            float(log_rate),
            log_ttl_exp,
            float(np.clip(resid_ratio, 0.0, 1e6)),
            lag_rate(1),
            lag_rate(4),
            lag_rate(8),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(out)):
        bad = [FEATURE_NAMES[i] for i in np.flatnonzero(~np.isfinite(out))]
        raise ValueError(f"non-finite features produced: {bad}")
    return out


def feature_matrix(trajectories, limit: LimitSpec) -> np.ndarray:
    """Stack :func:`window_features` over an iterable of trajectories.

    Returns shape ``(n_trajectories, len(FEATURE_NAMES))``.
    """
    rows = [window_features(tr.times_days, tr.values_percent, limit) for tr in trajectories]
    if not rows:
        raise ValueError("no trajectories supplied")
    return np.vstack(rows)
