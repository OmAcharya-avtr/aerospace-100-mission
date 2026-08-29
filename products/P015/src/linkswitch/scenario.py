"""Scenario configuration and synchronised dual-channel telemetry generation.

Combines the optical channel (``optical.py``) and RF channel (``rf.py``)
into one seeded, reproducible telemetry generator. **Every field in the
resulting telemetry is simulated; no field-measured data is used.**
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .optical import OpticalParams, simulate_ar1_log_irradiance
from .rf import RFParams, rf_path_attenuation_db, simulate_rain_indicator

__all__ = ["SwitchCost", "ScenarioConfig", "Telemetry", "generate_telemetry"]


@dataclass(frozen=True)
class SwitchCost:
    """Physical switch-over cost.

    downtime_steps : number of time steps of zero delivered throughput
        incurred every time the active channel changes (beam re-acquisition
        / RF re-lock time). Must be >= 0. A hybrid RF/FSO handover overhead
        is a recognised practical concern; see Kaushal & Kaddoum (2017)
        survey (general discussion, no specific numeric value reproduced
        here — ``downtime_steps`` is a configuration input).
    """

    downtime_steps: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.downtime_steps, (int, np.integer))
            or isinstance(self.downtime_steps, bool)
            or self.downtime_steps < 0
        ):
            raise ValueError(f"downtime_steps must be a non-negative integer, "
                              f"got {self.downtime_steps!r}")


@dataclass(frozen=True)
class ScenarioConfig:
    """Bundles optical + RF channel configuration and switch cost."""

    optical: OpticalParams = field(default_factory=OpticalParams)
    rf: RFParams = field(default_factory=RFParams)
    switch_cost: SwitchCost = field(default_factory=SwitchCost)


@dataclass(frozen=True)
class Telemetry:
    """Synchronised per-step dual-channel telemetry, all arrays length n_steps.

    irradiance : mean-normalised optical irradiance I(t), > 0.
    opt_available : physical optical availability, I(t) >= tau_phys.
    rain_rate_mm_hr : simulated instantaneous rain rate (0 when not raining).
    rf_atten_db : simulated RF path attenuation due to rain, dB (0 when clear).
    rf_available : physical RF availability, SNR_rf(t) >= snr_min_db.
    """

    irradiance: np.ndarray
    opt_available: np.ndarray
    rain_rate_mm_hr: np.ndarray
    rf_atten_db: np.ndarray
    rf_available: np.ndarray

    @property
    def n_steps(self) -> int:
        return int(self.irradiance.shape[0])


def generate_telemetry(config: ScenarioConfig, n_steps: int, seed: int) -> Telemetry:
    """Generate one seeded, reproducible telemetry realisation.

    Optical and RF channels are driven by independent RNG streams derived
    from ``seed`` (turbulence-induced optical scintillation and rain-induced
    RF fading are physically distinct phenomena with no shared driver in
    this simulator).
    """
    if not isinstance(n_steps, (int, np.integer)) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    ss = np.random.SeedSequence(int(seed))
    opt_seed, rf_seed = ss.spawn(2)
    rng_opt = np.random.default_rng(opt_seed)
    rng_rf = np.random.default_rng(rf_seed)

    opt = config.optical
    if opt.fading_model != "lognormal":
        raise ValueError(
            "generate_telemetry only wires up fading_model='lognormal' for the temporal "
            f"AR(1) process; got {opt.fading_model!r}. Use "
            "optical.sample_gamma_gamma_irradiance directly for gamma-gamma."
        )
    irradiance = simulate_ar1_log_irradiance(rng_opt, n_steps, opt.sigma_i2, opt.coherence_steps)
    opt_available = irradiance >= opt.tau_phys

    rf = config.rf
    raining = simulate_rain_indicator(rng_rf, n_steps, rf.p_rain, rf.mean_event_steps)
    n_rain = int(np.count_nonzero(raining))
    rain_rate = np.zeros(n_steps)
    if n_rain > 0:
        rain_rate[raining] = rng_rf.lognormal(
            mean=math.log(rf.r_med_mm_hr), sigma=rf.rate_sigma, size=n_rain
        )
    rf_atten = np.zeros(n_steps)
    if n_rain > 0:
        rf_atten[raining] = rf_path_attenuation_db(
            rain_rate[raining], rf.k, rf.alpha, rf.path_length_km, rf.reduction_length_km
        )
    rf_snr_db = rf.snr_clear_db - rf_atten
    rf_available = rf_snr_db >= rf.snr_min_db

    return Telemetry(
        irradiance=irradiance,
        opt_available=opt_available,
        rain_rate_mm_hr=rain_rate,
        rf_atten_db=rf_atten,
        rf_available=rf_available,
    )
