"""Vectorised persistence-run detection over many independent monitoring windows.

The whole package expresses its operating point as a *window false-alarm
probability*: the probability that a detector raises at least one alarm during
one nominal monitoring window of ``W`` samples, with the detector state reset at
the start of the window.  That quantity is a Bernoulli trial per window, so an
estimate from ``M`` independent windows has the ordinary binomial standard
error ``sqrt(p(1-p)/M)``.  A per-sample alarm rate measured on one long run does
*not* have that property, because consecutive samples of a control-chart
statistic are dependent; this module exists so the window formulation can be
evaluated at Monte-Carlo scale without a Python loop over samples.

No equations from the literature appear here -- this is bookkeeping.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["first_run_index", "any_run", "run_counter_trace"]


def first_run_index(breach: NDArray[np.bool_], persistence: int) -> NDArray[np.int64]:
    """Index of the sample at which a run of ``persistence`` breaches completes.

    Parameters
    ----------
    breach
        Boolean array of shape ``(n_windows, window_length)``; ``True`` marks a
        sample whose value is outside the limit (or whose score exceeds the
        threshold).  Dimensionless.
    persistence
        Number of *consecutive* breaching samples required to raise an alarm
        (the debounce count).  ``persistence = 1`` means raise immediately.
        Dimensionless, >= 1.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_windows,)``, dtype int64.  Element ``i`` is the zero-based
        sample index at which the alarm is raised in window ``i``, or ``-1`` if
        no run of the required length completes inside the window.

    Notes
    -----
    Cost is ``O(window_length)`` NumPy operations on length-``n_windows``
    vectors, independent of ``persistence``.
    """
    breach = np.asarray(breach, dtype=bool)
    if breach.ndim != 2:
        raise ValueError(f"breach must be 2-D (n_windows, window_length), got shape {breach.shape}")
    if not isinstance(persistence, (int, np.integer)) or persistence < 1:
        raise ValueError(f"persistence must be an integer >= 1, got {persistence!r}")

    n_windows, length = breach.shape
    counter = np.zeros(n_windows, dtype=np.int64)
    out = np.full(n_windows, -1, dtype=np.int64)
    open_ = np.ones(n_windows, dtype=bool)
    for t in range(length):
        col = breach[:, t]
        counter = np.where(col, counter + 1, 0)
        fired = open_ & (counter >= persistence)
        if fired.any():
            out[fired] = t
            open_ &= ~fired
            if not open_.any():
                break
    return out


def any_run(breach: NDArray[np.bool_], persistence: int) -> NDArray[np.bool_]:
    """``True`` where a window contains a completed run of ``persistence`` breaches."""
    return first_run_index(breach, persistence) >= 0


def run_counter_trace(breach: NDArray[np.bool_], persistence: int) -> NDArray[np.int64]:
    """Per-sample persistence counter for a single window, for inspection and tests.

    Parameters
    ----------
    breach
        Boolean array of shape ``(window_length,)``.
    persistence
        Debounce count, >= 1.

    Returns
    -------
    numpy.ndarray
        Shape ``(window_length,)``, the counter value *after* each sample is
        processed; it saturates at ``persistence``.
    """
    breach = np.asarray(breach, dtype=bool)
    if breach.ndim != 1:
        raise ValueError(f"breach must be 1-D, got shape {breach.shape}")
    if persistence < 1:
        raise ValueError(f"persistence must be >= 1, got {persistence!r}")
    counter = 0
    out = np.zeros(breach.size, dtype=np.int64)
    for t, b in enumerate(breach):
        counter = min(counter + 1, persistence) if b else 0
        out[t] = counter
    return out
