"""Causal features over recent per-stage latencies, and the label.

Prediction problem
------------------
At iteration ``i``, using only iterations ``<= i``, predict

    y[i] = 1 if any of d[i+1] .. d[i+H] exceeds the deadline, else 0

for a horizon ``H`` (``horizon``, default 3). The one-step-ahead version
(``H = 1``) is a special case. The window form is used because the point of
the prediction is to shed load *before* the overrun, and a flag that only ever
fires one iteration ahead has a lead time of exactly one by construction,
which makes "lead time" a meaningless metric. With ``H > 1`` the lead time is
measured, not assumed.

Causality
---------
Every feature at index ``i`` is a function of ``stage_s[:i+1]`` only. The
first ``warmup`` rows (default 32, the longest window) are dropped because
their windows are incomplete, and the last ``H`` rows are dropped because
their labels are not yet determined. :func:`build_dataset` returns the index
array so a caller can map rows back to iterations and check this.

Features
--------
18 features, all in seconds or dimensionless; see :data:`FEATURE_NAMES`.
``headroom_frac`` and ``util_last`` are normalised by the deadline so a model
trained at one period is not nonsense at another — though no claim is made
that it transfers; see ``MODEL_CARD.md``.
"""

from __future__ import annotations

import numpy as np

from ..errors import ConfigurationError

__all__ = ["FEATURE_NAMES", "WARMUP", "build_dataset", "make_features", "make_labels"]

FEATURE_NAMES: tuple[str, ...] = (
    "total_last_s",
    "total_lag1_s",
    "total_lag2_s",
    "total_mean8_s",
    "total_max8_s",
    "total_std8_s",
    "total_ewma_s",
    "total_slope4_s",
    "total_mean32_s",
    "total_max32_s",
    "overruns_in_8",
    "iters_since_overrun",
    "headroom_frac",
    "util_last",
    "sense_last_s",
    "estimate_last_s",
    "control_last_s",
    "actuate_last_s",
)

WARMUP = 32
_EWMA_ALPHA = 0.30


def _rolling(x: np.ndarray, window: int, fn) -> np.ndarray:
    """``fn`` over a trailing window of length ``window``, aligned to the right.

    Entries before the window is full are filled with the statistic over the
    partial window, which is only ever used for rows the caller drops.
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    if n >= window:
        view = np.lib.stride_tricks.sliding_window_view(x, window)
        out[window - 1 :] = fn(view, axis=1)
    for i in range(min(window - 1, n)):
        out[i] = fn(x[: i + 1][None, :], axis=1)[0]
    return out


def make_features(stage_s: np.ndarray, deadline_s: float) -> np.ndarray:
    """Build the ``(n, 18)`` causal feature matrix.

    Parameters
    ----------
    stage_s:
        ``(n, 4)`` per-stage durations [s], finite and >= 0.
    deadline_s:
        Relative deadline [s], > 0.

    Returns
    -------
    numpy.ndarray
        ``(n, 18)`` float64, column order :data:`FEATURE_NAMES`. Row ``i``
        uses only ``stage_s[:i+1]``.
    """
    st = np.asarray(stage_s, dtype=np.float64)
    if st.ndim != 2 or st.shape[1] != 4:
        raise ConfigurationError(f"stage_s must have shape (n, 4), got {st.shape}")
    if not np.all(np.isfinite(st)) or np.any(st < 0.0):
        raise ValueError("stage_s must be finite and >= 0 s")
    if not (deadline_s > 0.0):
        raise ConfigurationError(f"deadline_s must be > 0, got {deadline_s!r}")
    n = st.shape[0]
    if n < 3:
        raise ConfigurationError(f"need at least 3 iterations, got {n}")
    totals = st.sum(axis=1)

    lag1 = np.concatenate(([totals[0]], totals[:-1]))
    lag2 = np.concatenate(([totals[0], totals[0]], totals[:-2]))
    mean8 = _rolling(totals, 8, np.mean)
    max8 = _rolling(totals, 8, np.max)
    std8 = _rolling(totals, 8, lambda v, axis: np.std(v, axis=axis, ddof=0))
    mean32 = _rolling(totals, WARMUP, np.mean)
    max32 = _rolling(totals, WARMUP, np.max)

    ewma = np.empty(n, dtype=np.float64)
    acc = totals[0]
    for i in range(n):
        acc = _EWMA_ALPHA * totals[i] + (1.0 - _EWMA_ALPHA) * acc
        ewma[i] = acc

    slope4 = np.empty(n, dtype=np.float64)
    slope4[:4] = 0.0
    slope4[4:] = totals[4:] - totals[:-4]

    over = (totals > deadline_s).astype(np.float64)
    over8 = _rolling(over, 8, np.sum)

    since = np.empty(n, dtype=np.float64)
    counter = float(WARMUP)
    for i in range(n):
        counter = 0.0 if over[i] else min(counter + 1.0, 1000.0)
        since[i] = counter

    headroom = (deadline_s - totals) / deadline_s
    util = totals / deadline_s

    cols = [
        totals,
        lag1,
        lag2,
        mean8,
        max8,
        std8,
        ewma,
        slope4,
        mean32,
        max32,
        over8,
        since,
        headroom,
        util,
        st[:, 0],
        st[:, 1],
        st[:, 2],
        st[:, 3],
    ]
    out = np.column_stack(cols)
    if out.shape[1] != len(FEATURE_NAMES):  # pragma: no cover - guards the constant
        raise AssertionError(
            f"built {out.shape[1]} columns but FEATURE_NAMES has {len(FEATURE_NAMES)}"
        )
    return out


def make_labels(totals_s: np.ndarray, deadline_s: float, *, horizon: int = 3) -> np.ndarray:
    """``y[i] = any(totals_s[i+1 : i+1+horizon] > deadline_s)``.

    The last ``horizon`` entries are ``False`` by convention and are dropped
    by :func:`build_dataset`.

    Parameters
    ----------
    totals_s:
        ``(n,)`` iteration durations [s].
    deadline_s:
        Relative deadline [s], > 0.
    horizon:
        Look-ahead ``H`` in iterations, >= 1.
    """
    d = np.asarray(totals_s, dtype=np.float64).ravel()
    if not (deadline_s > 0.0):
        raise ConfigurationError(f"deadline_s must be > 0, got {deadline_s!r}")
    if horizon < 1:
        raise ConfigurationError(f"horizon must be >= 1, got {horizon!r}")
    n = d.size
    over = d > deadline_s
    y = np.zeros(n, dtype=bool)
    for h in range(1, horizon + 1):
        if h < n:
            y[: n - h] |= over[h:]
    y[max(n - horizon, 0) :] = False
    return y


def build_dataset(
    stage_s: np.ndarray,
    deadline_s: float,
    *,
    horizon: int = 3,
    warmup: int = WARMUP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features, labels and the source index for the usable rows.

    Parameters
    ----------
    stage_s:
        ``(n, 4)`` per-stage durations [s].
    deadline_s:
        Relative deadline [s].
    horizon:
        Label look-ahead in iterations.
    warmup:
        Rows to drop at the start, >= 1; must be at least as large as the
        longest feature window (32) for every feature to be complete.

    Returns
    -------
    tuple
        ``(X, y, index)`` with ``X`` of shape ``(m, 18)``, ``y`` boolean of
        length ``m``, and ``index`` the original iteration numbers.
    """
    st = np.asarray(stage_s, dtype=np.float64)
    if warmup < 1:
        raise ConfigurationError(f"warmup must be >= 1, got {warmup!r}")
    n = st.shape[0]
    if n <= warmup + horizon:
        raise ConfigurationError(
            f"trace of {n} iterations is too short for warmup {warmup} and horizon {horizon}"
        )
    x = make_features(st, deadline_s)
    y = make_labels(st.sum(axis=1), deadline_s, horizon=horizon)
    hi = n - horizon
    idx = np.arange(warmup, hi, dtype=np.int64)
    return x[warmup:hi], y[warmup:hi], idx
