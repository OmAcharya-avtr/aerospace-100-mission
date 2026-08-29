"""Switching policies: fixed-threshold, hysteresis, and learned predictive.

Every policy is a small class exposing ``select_channels(telemetry) ->
np.ndarray[bool]`` — ``True`` at step ``t`` means "select optical",
``False`` means "select RF". All policies are causal: the decision at step
``t`` uses only ``telemetry`` data up to and including step ``t`` (the
learned policy's rolling-window features never look ahead).
"""

from __future__ import annotations

import math

import numpy as np

from .features import rolling_features
from .scenario import Telemetry

__all__ = ["FixedThresholdPolicy", "HysteresisPolicy", "LearnedPolicy"]


class FixedThresholdPolicy:
    """Switch to RF whenever irradiance falls below a single threshold ``tau``.

    No memory: ``select_optical(t) = irradiance(t) >= tau``.
    """

    name = "fixed_threshold"

    def __init__(self, tau: float):
        tau = float(tau)
        if not (math.isfinite(tau) and tau > 0.0):
            raise ValueError(f"tau must be finite and > 0, got {tau!r}")
        self.tau = tau

    def select_channels(self, telemetry: Telemetry) -> np.ndarray:
        return telemetry.irradiance >= self.tau


class HysteresisPolicy:
    """Two thresholds to avoid chatter: switch away below ``tau_low``, switch
    back only above ``tau_high`` (``tau_high >= tau_low``).

    Currently optical: switch to RF when irradiance < tau_low.
    Currently RF: switch to optical when irradiance > tau_high.
    """

    name = "hysteresis"

    def __init__(self, tau_low: float, tau_high: float):
        tau_low = float(tau_low)
        tau_high = float(tau_high)
        if not (math.isfinite(tau_low) and tau_low > 0.0):
            raise ValueError(f"tau_low must be finite and > 0, got {tau_low!r}")
        if not (math.isfinite(tau_high) and tau_high > 0.0):
            raise ValueError(f"tau_high must be finite and > 0, got {tau_high!r}")
        if tau_high < tau_low:
            raise ValueError(
                f"tau_high ({tau_high}) must be >= tau_low ({tau_low}) or the policy "
                "would never return to optical after a marginal dip"
            )
        self.tau_low = tau_low
        self.tau_high = tau_high

    def select_channels(self, telemetry: Telemetry) -> np.ndarray:
        irr = telemetry.irradiance
        n = irr.shape[0]
        out = np.empty(n, dtype=bool)
        on_optical = True
        for t in range(n):
            if on_optical and irr[t] < self.tau_low:
                on_optical = False
            elif not on_optical and irr[t] > self.tau_high:
                on_optical = True
            out[t] = on_optical
        return out


class LearnedPolicy:
    """Preemptively switch away when a trained model predicts imminent outage.

    Switch-away side (proactive): currently optical, switch to RF when the
    model's ``P(outage within the next `horizon` steps)`` >=
    ``confidence_threshold``.
    Switch-back side: currently RF, switch to optical only once BOTH the
    observed irradiance has recovered above ``tau_phys`` AND the model's
    outage confidence has dropped back below ``confidence_threshold``.
    Requiring both (rather than irradiance alone) matters because the
    predictor is inherently forward-looking: a purely irradiance-triggered
    return would often fire on the very next step after a proactive
    switch-away (irradiance had not dropped yet — that was the point of
    switching early), causing the policy to flap every step whenever the
    model keeps predicting danger while the current sample still looks
    fine.
    """

    name = "learned"

    def __init__(self, model, tau_phys: float, confidence_threshold: float, window: int):
        tau_phys = float(tau_phys)
        confidence_threshold = float(confidence_threshold)
        if not (math.isfinite(tau_phys) and tau_phys > 0.0):
            raise ValueError(f"tau_phys must be finite and > 0, got {tau_phys!r}")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold!r}"
            )
        if not isinstance(window, (int, np.integer)) or isinstance(window, bool) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window!r}")
        self.model = model
        self.tau_phys = tau_phys
        self.confidence_threshold = confidence_threshold
        self.window = int(window)

    def outage_confidence(self, telemetry: Telemetry) -> np.ndarray:
        """P(outage within the prediction horizon) at every step, in [0, 1]."""
        features = rolling_features(telemetry.irradiance, self.window)
        return self.model.predict_proba(features)

    def select_channels(self, telemetry: Telemetry) -> np.ndarray:
        irr = telemetry.irradiance
        n = irr.shape[0]
        p_outage = self.outage_confidence(telemetry)
        out = np.empty(n, dtype=bool)
        on_optical = True
        for t in range(n):
            if on_optical and p_outage[t] >= self.confidence_threshold:
                on_optical = False
            elif (
                not on_optical
                and irr[t] >= self.tau_phys
                and p_outage[t] < self.confidence_threshold
            ):
                on_optical = True
            out[t] = on_optical
        return out
