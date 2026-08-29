"""Scoring a channel selection against a link trace.

All policies produce the same object -- a boolean array ``use_optical`` of one
element per decision step -- and are scored by the same function, so no policy
can gain an advantage from its own accounting.

Throughput model (units in brackets):

* step ``t`` delivers ``rate_optical_bps`` [bit/s] if the optical channel is
  selected and up, ``rate_rf_bps`` if the RF channel is selected and up, and
  zero otherwise;
* every step at which the selection differs from the previous step starts a
  handover guard of ``switch_penalty_steps`` samples during which the delivered
  rate is zero, on either channel.

Assumptions: a fixed-rate optical link with a hard forward-error-correction
threshold (no rate adaptation, no partial credit near threshold), instantaneous
and error-free knowledge of which channel was selected, and no buffering or
retransmission across the guard interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["PolicyMetrics", "evaluate_selection"]


@dataclass(frozen=True)
class PolicyMetrics:
    """Scored performance of one channel selection over one trace.

    Attributes
    ----------
    throughput_bps : float
        Time-averaged delivered payload rate [bit/s].
    outage_fraction : float
        Fraction of steps delivering zero payload [-] in [0, 1].
    n_switches : int
        Number of channel changes [-].
    switches_per_s : float
        Channel changes per second [1/s].
    optical_fraction : float
        Fraction of steps on which the optical channel was selected [-].
    guard_fraction : float
        Fraction of steps lost to handover guard intervals [-].
    duration_s : float
        Trace duration [s].
    """

    throughput_bps: float
    outage_fraction: float
    n_switches: int
    switches_per_s: float
    optical_fraction: float
    guard_fraction: float
    duration_s: float


def evaluate_selection(
    use_optical: NDArray[np.bool_],
    optical_up: NDArray[np.bool_],
    rf_up: NDArray[np.bool_],
    *,
    rate_optical_bps: float,
    rate_rf_bps: float,
    dt_s: float,
    switch_penalty_steps: int = 0,
) -> PolicyMetrics:
    """Score a channel selection.

    Parameters
    ----------
    use_optical : numpy.ndarray of bool
        Selected channel per step (``True`` = optical).
    optical_up, rf_up : numpy.ndarray of bool
        True channel availability per step, same length.
    rate_optical_bps, rate_rf_bps : float
        Payload rates while up [bit/s], > 0.
    dt_s : float
        Step duration [s], > 0.
    switch_penalty_steps : int
        Guard length in samples following each change, >= 0.

    Returns
    -------
    PolicyMetrics
    """
    sel = np.asarray(use_optical, dtype=bool)
    o_up = np.asarray(optical_up, dtype=bool)
    r_up = np.asarray(rf_up, dtype=bool)
    if sel.ndim != 1 or sel.size == 0:
        raise ValueError("use_optical must be a non-empty 1-D array")
    if o_up.shape != sel.shape or r_up.shape != sel.shape:
        raise ValueError(
            f"shape mismatch: use_optical {sel.shape}, optical_up {o_up.shape}, "
            f"rf_up {r_up.shape}"
        )
    if not (np.isfinite(rate_optical_bps) and rate_optical_bps > 0):
        raise ValueError(f"rate_optical_bps must be > 0, got {rate_optical_bps}")
    if not (np.isfinite(rate_rf_bps) and rate_rf_bps > 0):
        raise ValueError(f"rate_rf_bps must be > 0, got {rate_rf_bps}")
    if not (np.isfinite(dt_s) and dt_s > 0):
        raise ValueError(f"dt_s must be > 0, got {dt_s}")
    g = int(switch_penalty_steps)
    if g < 0:
        raise ValueError(f"switch_penalty_steps must be >= 0, got {g}")

    n = sel.size
    idx = np.arange(n)
    changed = np.empty(n, dtype=bool)
    changed[0] = False
    changed[1:] = sel[1:] != sel[:-1]

    if g > 0:
        chg_pos = np.where(changed, idx, -(n + g + 1))
        last_chg = np.maximum.accumulate(chg_pos)
        blocked = (idx - last_chg) < g
    else:
        blocked = np.zeros(n, dtype=bool)

    rate = np.where(sel, np.where(o_up, rate_optical_bps, 0.0), np.where(r_up, rate_rf_bps, 0.0))
    rate = np.where(blocked, 0.0, rate)

    n_sw = int(np.count_nonzero(changed))
    duration = n * float(dt_s)
    return PolicyMetrics(
        throughput_bps=float(rate.mean()),
        outage_fraction=float(np.count_nonzero(rate == 0.0) / n),
        n_switches=n_sw,
        switches_per_s=float(n_sw / duration),
        optical_fraction=float(np.count_nonzero(sel) / n),
        guard_fraction=float(np.count_nonzero(blocked) / n),
        duration_s=duration,
    )
