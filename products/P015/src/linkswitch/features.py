"""Causal rolling features and imminent-outage labels for the learned policy.

Features are built from the log-irradiance series so they are scale-free
with respect to the mean-normalisation of ``I``. All statistics at step
``t`` use only samples up to and including ``t`` (causal — no lookahead),
so the same function is safe for both training and live decision-making.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = ["FEATURE_NAMES", "rolling_features", "label_imminent_outage"]

FEATURE_NAMES = ("log_i", "roll_mean", "roll_std", "roll_min", "slope")


def rolling_features(irradiance: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling feature matrix, shape (n_steps, 5).

    Columns (see ``FEATURE_NAMES``):
    0. log_i      -- ln I(t)
    1. roll_mean  -- mean of ln I over the trailing window (min_periods=1)
    2. roll_std   -- std of ln I over the trailing window (0 for a single sample)
    3. roll_min   -- min of ln I over the trailing window
    4. slope      -- (ln I(t) - ln I(t - window)) / window; 0 for t < window
    """
    irradiance = np.asarray(irradiance, dtype=float)
    if irradiance.ndim != 1 or irradiance.shape[0] < 1:
        raise ValueError("irradiance must be a non-empty 1-D array")
    if np.any(irradiance <= 0.0) or not np.all(np.isfinite(irradiance)):
        raise ValueError("irradiance must be finite and > 0 everywhere")
    if not isinstance(window, (int, np.integer)) or isinstance(window, bool) or window < 1:
        raise ValueError(f"window must be a positive integer, got {window!r}")

    log_i = np.log(irradiance)
    s = pd.Series(log_i)
    roll_mean = s.rolling(window=window, min_periods=1).mean().to_numpy()
    roll_std = s.rolling(window=window, min_periods=1).std(ddof=0).fillna(0.0).to_numpy()
    roll_min = s.rolling(window=window, min_periods=1).min().to_numpy()

    n = log_i.shape[0]
    shifted = np.empty(n)
    shifted[:window] = log_i[0]  # no lookback available yet: zero slope
    if n > window:
        shifted[window:] = log_i[: n - window]
    slope = (log_i - shifted) / float(window)
    slope[:window] = 0.0

    return np.column_stack([log_i, roll_mean, roll_std, roll_min, slope])


def label_imminent_outage(
    irradiance: np.ndarray, tau_phys: float, horizon: int
) -> np.ndarray:
    """Boolean label: does an outage (irradiance < tau_phys) occur within the
    next ``horizon`` steps, strictly after the current one?

    ``label[t] = any(irradiance[t+1 : t+1+horizon] < tau_phys)``. Near the
    end of the series, fewer than ``horizon`` future samples are available;
    the label uses whatever future remains (documented edge effect — the
    last ``horizon`` labels are conservative/under-informed and are excluded
    from training by callers where practical).
    """
    irradiance = np.asarray(irradiance, dtype=float)
    tau_phys = float(tau_phys)
    if not (math.isfinite(tau_phys) and tau_phys > 0.0):
        raise ValueError(f"tau_phys must be finite and > 0, got {tau_phys!r}")
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")

    n = irradiance.shape[0]
    below = irradiance < tau_phys
    label = np.zeros(n, dtype=bool)
    # Reverse cumulative "any outage within the next `horizon` steps" via a
    # sliding OR implemented through a min-style running window (O(n)).
    # Simple and fully vectorisable for the modest n_steps used here.
    csum = np.concatenate(([0], np.cumsum(below.astype(np.int64))))
    for t in range(n):
        hi = min(n, t + 1 + horizon)
        label[t] = (csum[hi] - csum[t + 1]) > 0
    return label
