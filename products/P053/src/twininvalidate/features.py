"""Windowed residual features for the learned drift classifier.

Nine features are computed over a sliding window of the normalised residual.
They were chosen before any model was fitted, each to be the obvious summary of
one of the injected change kinds, so that the learned model has a genuine
chance against the analytic baselines rather than being set up to lose:

==  ======================  ==================================================
#   feature                 what it is there for
==  ======================  ==================================================
0   window mean             a step in the residual mean (parameter_step)
1   window standard dev.    a change in residual spread (noise_variance)
2   mean |z|                spread, robust to the tail
3   max |z|                 single large excursions
4   lag-1 autocorrelation   residual colouring, which a process-noise change
                            produces and a mean shift does not
5   excess kurtosis         tail shape, a scale-mixture signature
6   trend                   the OLS slope of z against sample index, times the
                            window length: a ramp (slow_ramp)
7   fraction |z| > 2        tail mass
8   mean z^2                the sufficient statistic for a scale change
==  ======================  ==================================================

All features are dimensionless, because ``z`` is. Features 0, 1, 2, 8 are the
ones the analytic baselines already use in one form or another; 4, 5, 6 are the
ones they do not, and are where any win for the learned model has to come from.

Window-fill latency
-------------------
No feature vector exists until ``window`` samples have arrived. The learned
monitor therefore cannot alarm earlier than ``window`` samples after it starts,
which is a structural floor on its detection delay that the recursive analytic
detectors do not have. The floor is reported, not hidden: see the README and
``validation/validate_classifier.py``.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DEFAULT_WINDOW = 50
"""Declared feature window length in samples (2.5 s at the reference 20 Hz)."""

FEATURE_NAMES: tuple[str, ...] = (
    "mean",
    "std",
    "mean_abs",
    "max_abs",
    "lag1_autocorr",
    "excess_kurtosis",
    "trend",
    "frac_abs_gt_2",
    "mean_square",
)
"""Names of the nine features, in column order."""

N_FEATURES = len(FEATURE_NAMES)
"""Number of features, 9."""


def window_features(z: np.ndarray, window: int = DEFAULT_WINDOW) -> np.ndarray:
    """Sliding-window features of a residual batch.

    Parameters
    ----------
    z:
        Normalised residuals, shape ``(n_runs, n_samples)``, dimensionless.
    window:
        Window length in samples; must be at least 4 (the trend, the
        autocorrelation and the kurtosis are meaningless below that) and at
        most ``n_samples``.

    Returns
    -------
    Array of shape ``(n_runs, n_samples - window + 1, 9)``. Element
    ``[r, j, :]`` describes the window covering samples ``j .. j+window-1`` of
    run ``r``, so a decision made from it is available at sample
    ``j + window - 1``.
    """
    arr = np.asarray(z, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"residuals must be 1-D or 2-D, got ndim={arr.ndim}")
    if window < 4:
        raise ValueError(f"window must be at least 4 samples, got {window}")
    if window > arr.shape[1]:
        raise ValueError(
            f"window {window} exceeds stream length {arr.shape[1]}; nothing to compute"
        )
    win = sliding_window_view(arr, window, axis=1)  # (n_runs, n_windows, window)
    n_runs, n_windows, _ = win.shape

    mean = win.mean(axis=2)
    centred_sq = ((win - mean[:, :, None]) ** 2).mean(axis=2)
    std = np.sqrt(centred_sq)
    mean_abs = np.abs(win).mean(axis=2)
    max_abs = np.abs(win).max(axis=2)
    mean_square = (win**2).mean(axis=2)

    safe_var = np.where(centred_sq > 1e-12, centred_sq, 1.0)
    dev = win - mean[:, :, None]
    lag1 = (dev[:, :, :-1] * dev[:, :, 1:]).mean(axis=2) / safe_var
    lag1 = np.where(centred_sq > 1e-12, lag1, 0.0)

    m4 = (dev**4).mean(axis=2)
    kurt = np.where(centred_sq > 1e-12, m4 / safe_var**2 - 3.0, 0.0)

    # OLS slope against the centred sample index, times the window length, so
    # the feature is the total rise across the window in units of z.
    t = np.arange(window, dtype=float)
    t = t - t.mean()
    denom = float((t * t).sum())
    trend = (dev * t).sum(axis=2) / denom * float(window)

    frac2 = (np.abs(win) > 2.0).mean(axis=2)

    out = np.empty((n_runs, n_windows, N_FEATURES))
    out[:, :, 0] = mean
    out[:, :, 1] = std
    out[:, :, 2] = mean_abs
    out[:, :, 3] = max_abs
    out[:, :, 4] = lag1
    out[:, :, 5] = kurt
    out[:, :, 6] = trend
    out[:, :, 7] = frac2
    out[:, :, 8] = mean_square
    return out


def flatten_features(features: np.ndarray) -> np.ndarray:
    """Reshape ``(n_runs, n_windows, 9)`` to ``(n_runs * n_windows, 9)``."""
    f = np.asarray(features, dtype=float)
    if f.ndim != 3 or f.shape[2] != N_FEATURES:
        raise ValueError(f"expected shape (n_runs, n_windows, {N_FEATURES}), got {f.shape}")
    return f.reshape(-1, N_FEATURES)
