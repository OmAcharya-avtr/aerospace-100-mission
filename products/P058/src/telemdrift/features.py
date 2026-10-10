"""Windowed features for the learned detector.

Eight features over a trailing window of ``W`` samples. All are dimensionless
because the channel is standardised upstream (see :mod:`telemdrift.streams`).
Each is chosen to carry information one of the analytic detectors uses, so the
learned detector has at least the opportunity to match them:

===  ===================================  =============================================
#    Feature                              Analytic counterpart
===  ===================================  =============================================
0    window mean                          CUSUM / EWMA level statistic
1    window standard deviation            variance-change evidence
2    second-half mean minus first-half     CUSUM one-step increment, localised
3    log ratio of half-window std devs    variance-step evidence
4    least-squares slope per sample        drift-ramp evidence
5    lag-one autocorrelation              dependence, and transient shape
6    interquartile range                  scale, robust to a single outlier
7    range (max minus min)                scale, maximally sensitive to an outlier
===  ===================================  =============================================

Features 6 and 7 are included as a pair on purpose: their *difference in
behaviour* is the only signal in the window that distinguishes a short transient
spike (range jumps, IQR does not) from a genuine variance step (both move). Whether
the learned detector actually exploits that is measured, not assumed, in
``validation/validate_transient.py``.

Causality
---------
Every feature is a function of the trailing window only, so the batch
computation in :func:`window_features` is exactly equal to what an online
detector would have computed at each sample. The benchmark uses the batch path
for speed; ``tests/test_features.py`` pins the equality.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ["FEATURE_NAMES", "WINDOW", "window_features", "window_features_single"]

#: Default trailing window length in samples.
WINDOW = 50

FEATURE_NAMES: tuple[str, ...] = (
    "mean",
    "std",
    "half_mean_diff",
    "log_half_std_ratio",
    "slope",
    "lag1_autocorr",
    "iqr",
    "range",
)

_EPS = 1e-12


def _features_from_windows(win: np.ndarray) -> np.ndarray:
    """Compute the eight features for a 2-D array of shape ``(n_windows, W)``."""
    w = win.shape[1]
    half = w // 2
    mean = win.mean(axis=1)
    std = win.std(axis=1)
    m1 = win[:, :half].mean(axis=1)
    m2 = win[:, half:].mean(axis=1)
    s1 = win[:, :half].std(axis=1)
    s2 = win[:, half:].std(axis=1)
    centred_t = np.arange(w, dtype=float) - (w - 1) / 2.0
    slope = (win * centred_t).sum(axis=1) / (centred_t**2).sum()
    a = win[:, :-1]
    b = win[:, 1:]
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a_c**2).mean(axis=1) * (b_c**2).mean(axis=1)) + _EPS
    lag1 = (a_c * b_c).mean(axis=1) / denom
    q25, q75 = np.quantile(win, [0.25, 0.75], axis=1)
    rng = win.max(axis=1) - win.min(axis=1)
    return np.column_stack(
        [
            mean,
            std,
            m2 - m1,
            np.log((s2 + _EPS) / (s1 + _EPS)),
            slope,
            lag1,
            q75 - q25,
            rng,
        ]
    )


def window_features(stream: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Features for every trailing window of ``stream``.

    Parameters
    ----------
    stream:
        1-D array of standardised samples.
    window:
        Trailing window length in samples, ``>= 4``.

    Returns
    -------
    np.ndarray
        Shape ``(len(stream) - window + 1, 8)``. Row ``i`` is computed from
        ``stream[i : i + window]`` and is therefore available to an online
        detector at stream index ``i + window - 1``.
    """
    x = np.asarray(stream, dtype=float)
    if x.ndim != 1:
        raise ValueError("stream must be 1-D")
    if window < 4:
        raise ValueError("window must be >= 4 samples")
    if x.size < window:
        return np.empty((0, len(FEATURE_NAMES)))
    return _features_from_windows(sliding_window_view(x, window))


def window_features_single(window_values: np.ndarray) -> np.ndarray:
    """Features for one window, shape ``(1, 8)``. Reference path for the tests."""
    x = np.asarray(window_values, dtype=float).reshape(1, -1)
    if x.shape[1] < 4:
        raise ValueError("window must be >= 4 samples")
    return _features_from_windows(x)
