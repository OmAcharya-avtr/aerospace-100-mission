"""Hybrid RF-optical link scenario: geometry-free parameter bundle plus trace generator.

A *scenario* fixes both channels, the data rates, the sampling interval and the
switching penalty. :func:`simulate_trace` turns a scenario and a seed into a
:class:`LinkTrace`, which is the only object the switching policies see.

**The channel statistics are simulated, not measured.** See ``DATASET_CARD.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .channels import (
    OpticalChannelParams,
    RainChannelParams,
    ar1_rho,
    simulate_optical_margin_db,
    simulate_rain_attenuation_db,
)

__all__ = ["HybridLinkScenario", "LinkTrace", "simulate_trace"]


@dataclass(frozen=True)
class LinkTrace:
    """One realisation of the hybrid link.

    Attributes
    ----------
    dt_s : float
        Sample interval [s].
    optical_margin_db : numpy.ndarray
        True optical margin above the optical FEC threshold [dB]. ``>= 0``
        means the optical channel carries data.
    optical_telemetry_db : numpy.ndarray
        Margin as *measured* by the receiver [dB]. Equals
        ``optical_margin_db`` when telemetry noise is zero. This is the only
        optical quantity a causal policy may use.
    rf_margin_db : numpy.ndarray
        True RF margin above the RF demodulator threshold [dB].
    rain_rate_mm_per_h : numpy.ndarray
        Simulated rain rate [mm/h].
    optical_up : numpy.ndarray of bool
        ``optical_margin_db >= 0``.
    rf_up : numpy.ndarray of bool
        ``rf_margin_db >= 0``.
    seed : int
        Seed used to generate the trace.
    """

    dt_s: float
    optical_margin_db: NDArray[np.float64]
    optical_telemetry_db: NDArray[np.float64]
    rf_margin_db: NDArray[np.float64]
    rain_rate_mm_per_h: NDArray[np.float64]
    optical_up: NDArray[np.bool_]
    rf_up: NDArray[np.bool_]
    seed: int

    @property
    def n_steps(self) -> int:
        """Number of samples in the trace [-]."""
        return int(self.optical_margin_db.size)

    @property
    def duration_s(self) -> float:
        """Trace duration [s]."""
        return self.n_steps * self.dt_s


@dataclass(frozen=True)
class HybridLinkScenario:
    """Everything needed to generate and score one hybrid-link experiment.

    Attributes
    ----------
    optical : OpticalChannelParams
        Optical (scintillation) channel parameters.
    rf : RainChannelParams
        RF (rain-fade) channel parameters.
    dt_s : float
        Telemetry / decision interval [s], > 0.
    n_steps : int
        Samples per trace, >= 2.
    rate_optical_bps : float
        Optical payload rate while the optical link is up [bit/s], > 0.
    rate_rf_bps : float
        RF payload rate while the RF link is up [bit/s], > 0 and normally
        ``< rate_optical_bps``.
    switch_penalty_steps : int
        Samples of zero throughput following every change of selected channel
        (radio re-acquisition / handover guard), >= 0.
    """

    optical: OpticalChannelParams = field(default_factory=OpticalChannelParams)
    rf: RainChannelParams = field(
        default_factory=lambda: RainChannelParams(k_itu=0.10, alpha_itu=1.10)
    )
    dt_s: float = 2.0e-3
    n_steps: int = 8000
    rate_optical_bps: float = 1.0e9
    rate_rf_bps: float = 2.5e8
    switch_penalty_steps: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.optical, OpticalChannelParams):
            raise TypeError("optical must be an OpticalChannelParams")
        if not isinstance(self.rf, RainChannelParams):
            raise TypeError("rf must be a RainChannelParams")
        if not (math.isfinite(self.dt_s) and self.dt_s > 0.0):
            raise ValueError(f"dt_s must be > 0, got {self.dt_s}")
        if not isinstance(self.n_steps, (int, np.integer)) or self.n_steps < 2:
            raise ValueError(f"n_steps must be an integer >= 2, got {self.n_steps!r}")
        if not (math.isfinite(self.rate_optical_bps) and self.rate_optical_bps > 0.0):
            raise ValueError(f"rate_optical_bps must be > 0, got {self.rate_optical_bps}")
        if not (math.isfinite(self.rate_rf_bps) and self.rate_rf_bps > 0.0):
            raise ValueError(f"rate_rf_bps must be > 0, got {self.rate_rf_bps}")
        if not isinstance(self.switch_penalty_steps, (int, np.integer)):
            raise TypeError("switch_penalty_steps must be an integer")
        if self.switch_penalty_steps < 0:
            raise ValueError(
                f"switch_penalty_steps must be >= 0, got {self.switch_penalty_steps}"
            )

    @property
    def rho(self) -> float:
        """Lag-1 correlation of the optical margin at the decision rate [-]."""
        return ar1_rho(self.dt_s, self.optical.correlation_time_s)

    @property
    def rate_ratio(self) -> float:
        """``rate_rf / rate_optical`` [-]: the myopic switch-indifference level."""
        return self.rate_rf_bps / self.rate_optical_bps


def simulate_trace(scenario: HybridLinkScenario, seed: int) -> LinkTrace:
    """Generate one seeded realisation of a hybrid link.

    The optical and RF channels are driven by **independent** streams of the
    same seeded generator, so they are statistically independent. That is a
    modelling assumption: in reality cloud and rain are correlated with
    turbulence and with each other (README "Limitations").
    """
    if not isinstance(scenario, HybridLinkScenario):
        raise TypeError(f"scenario must be a HybridLinkScenario, got {type(scenario)!r}")
    if not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {seed!r}")
    rng = np.random.default_rng(int(seed))
    true_db, meas_db = simulate_optical_margin_db(
        scenario.optical, scenario.n_steps, scenario.dt_s, rng
    )
    rate_mm, atten_db = simulate_rain_attenuation_db(
        scenario.rf, scenario.n_steps, scenario.dt_s, rng
    )
    rf_margin = scenario.rf.clear_sky_margin_db - atten_db
    return LinkTrace(
        dt_s=scenario.dt_s,
        optical_margin_db=true_db,
        optical_telemetry_db=meas_db,
        rf_margin_db=rf_margin,
        rain_rate_mm_per_h=rate_mm,
        optical_up=true_db >= 0.0,
        rf_up=rf_margin >= 0.0,
        seed=int(seed),
    )
