"""Switching policies for a hybrid RF-optical link.

Every policy maps a :class:`~linkswitch.scenario.LinkTrace` to a boolean array
``use_optical`` with one entry per decision step. Causality is enforced by the
base class: the decision for step ``t`` may only depend on telemetry up to and
including step ``t-1``. Step 0 has no telemetry and defaults to the optical
channel (the high-rate channel), which is what a real terminal does at
acquisition.

The two classical baselines in this module (:class:`FixedThresholdPolicy` and
:class:`HysteresisPolicy`) were implemented and validated **before** the learned
policy in :mod:`linkswitch.learned`, and the learned policy is benchmarked
against them on identical seeded traces.

Hysteresis with two thresholds is the standard remedy for chattering in a
threshold-triggered switch; see e.g. the general treatment of hysteretic
switching in K. J. Astrom and R. M. Murray, *Feedback Systems: An Introduction
for Scientists and Engineers*, Princeton University Press, 2008, Ch. 3
(relay/hysteresis nonlinearity). It is used here in its textbook form and no
originality is claimed for it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .scenario import LinkTrace

__all__ = [
    "AlwaysOpticalPolicy",
    "AlwaysRfPolicy",
    "ClairvoyantPolicy",
    "FixedThresholdPolicy",
    "HysteresisPolicy",
    "Policy",
    "shift_causal",
]


def shift_causal(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Delay a telemetry series by one step: ``y[t] = x[t-1]``, ``y[0] = x[0]``.

    The step-0 value is duplicated rather than invented; the corresponding
    decision is overridden to "optical" by :meth:`Policy.select` anyway.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("input must be a non-empty 1-D array")
    out = np.empty_like(arr)
    out[0] = arr[0]
    out[1:] = arr[:-1]
    return out


class Policy(ABC):
    """Base class for channel-selection policies.

    Subclasses implement :meth:`_decide`, which receives the *already delayed*
    telemetry so that causality cannot be violated by accident.
    """

    name: str = "policy"

    @abstractmethod
    def _decide(
        self,
        optical_telemetry_prev_db: NDArray[np.float64],
        rf_margin_prev_db: NDArray[np.float64],
        trace: LinkTrace,
    ) -> NDArray[np.bool_]:
        """Return ``use_optical`` given one-step-delayed telemetry."""

    def select(self, trace: LinkTrace) -> NDArray[np.bool_]:
        """Return the boolean ``use_optical`` selection for ``trace``."""
        if not isinstance(trace, LinkTrace):
            raise TypeError(f"trace must be a LinkTrace, got {type(trace)!r}")
        tele = shift_causal(trace.optical_telemetry_db)
        rf = shift_causal(trace.rf_margin_db)
        sel = np.asarray(self._decide(tele, rf, trace), dtype=bool)
        if sel.shape != trace.optical_margin_db.shape:
            raise ValueError(
                f"{type(self).__name__}._decide returned shape {sel.shape}, "
                f"expected {trace.optical_margin_db.shape}"
            )
        sel = sel.copy()
        sel[0] = True  # acquisition default: start on the high-rate channel
        return sel


class AlwaysOpticalPolicy(Policy):
    """Reference: never switch, always use the optical channel."""

    name = "always_optical"

    def _decide(self, optical_telemetry_prev_db, rf_margin_prev_db, trace):
        """Always select the optical channel."""
        return np.ones(optical_telemetry_prev_db.size, dtype=bool)


class AlwaysRfPolicy(Policy):
    """Reference: never switch, always use the RF channel."""

    name = "always_rf"

    def _decide(self, optical_telemetry_prev_db, rf_margin_prev_db, trace):
        """Always select the RF channel."""
        return np.zeros(optical_telemetry_prev_db.size, dtype=bool)


class FixedThresholdPolicy(Policy):
    """Baseline 1: use the optical channel while the last measured margin >= T.

    Parameters
    ----------
    threshold_db : float
        Switching threshold ``T`` on the measured optical margin [dB]. The
        myopically optimal value for a stationary channel and zero handover
        guard is given by
        :func:`linkswitch.analytic.optimal_fixed_threshold_db`.
    """

    name = "fixed_threshold"

    def __init__(self, threshold_db: float) -> None:
        t = float(threshold_db)
        if math.isnan(t):
            raise ValueError("threshold_db must not be NaN")
        self.threshold_db = t

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FixedThresholdPolicy(threshold_db={self.threshold_db:.4g})"

    def _decide(self, optical_telemetry_prev_db, rf_margin_prev_db, trace):
        """Select optical wherever the delayed measured margin reaches the threshold."""
        return optical_telemetry_prev_db >= self.threshold_db


class HysteresisPolicy(Policy):
    """Baseline 2: two-threshold hysteretic switch (anti-chatter).

    Leave the optical channel when the measured margin falls to or below
    ``lower_db``; return to it only when the margin rises to or above
    ``upper_db``. Between the two thresholds the current channel is held, which
    suppresses the rapid switching a single threshold produces when the margin
    dithers around it.

    Parameters
    ----------
    lower_db, upper_db : float
        Drop-to-RF and return-to-optical thresholds [dB]; ``upper_db`` must be
        ``>= lower_db``. Equal thresholds reduce this policy to
        :class:`FixedThresholdPolicy` up to the boundary convention.
    start_optical : bool
        Channel assumed before the first telemetry sample.
    """

    name = "hysteresis"

    def __init__(self, lower_db: float, upper_db: float, start_optical: bool = True) -> None:
        lo = float(lower_db)
        up = float(upper_db)
        if math.isnan(lo) or math.isnan(up):
            raise ValueError("lower_db and upper_db must not be NaN")
        if up < lo:
            raise ValueError(f"upper_db ({up}) must be >= lower_db ({lo})")
        self.lower_db = lo
        self.upper_db = up
        self.start_optical = bool(start_optical)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"HysteresisPolicy(lower_db={self.lower_db:.4g}, upper_db={self.upper_db:.4g})"
        )

    def _decide(self, optical_telemetry_prev_db, rf_margin_prev_db, trace):
        """Run the two-threshold state machine over the delayed telemetry."""
        v = optical_telemetry_prev_db
        n = v.size
        # +1 = commanded optical, -1 = commanded RF, 0 = hold previous state.
        event = np.where(v >= self.upper_db, 1, np.where(v <= self.lower_db, -1, 0))
        idx = np.where(event != 0, np.arange(n), -1)
        last = np.maximum.accumulate(idx)
        held = event[np.clip(last, 0, None)]
        return np.where(last >= 0, held > 0, self.start_optical)


class ClairvoyantPolicy(Policy):
    """Non-causal upper reference: knows the true optical state at step ``t``.

    Not a deployable policy. It bounds what any predictor could achieve when the
    handover guard is zero, and is reported only as context.
    """

    name = "clairvoyant"

    def _decide(self, optical_telemetry_prev_db, rf_margin_prev_db, trace):
        """Select optical exactly when the optical channel is truly up (non-causal)."""
        return np.asarray(trace.optical_up, dtype=bool)
