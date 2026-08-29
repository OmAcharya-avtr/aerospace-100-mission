"""RF channel: simulated two-state rain process and rain-fade attenuation.

**All rain/attenuation data here is SIMULATED. No field-measured rain-gauge
or link data is used anywhere in this package.**

Two-state rain occurrence process
------------------------------------
Rain occurrence is modelled as a stationary two-state (clear / rain) discrete
Markov chain. For a chain with self-transition probability of the rain
state ``p_stay`` and stationary rain probability ``p_rain``, elementary
two-state Markov chain algebra gives:

    p_rain_to_clear = 1 - p_stay                     (leaving rain each step)
    mean rain-event length (steps) = 1 / p_rain_to_clear   (geometric holding time)
    p_clear_to_rain = p_rain * p_rain_to_clear / (1 - p_rain)   (stationarity: pi = pi P)

so the chain is parameterised directly by the two physically meaningful
numbers (``p_rain``, ``mean_event_steps``) rather than by raw transition
probabilities. This is a coarse, purely synthetic occurrence model: it is
NOT derived from any measured rain-event-duration climatology.

Rain rate given "raining"
----------------------------
Conditional on the rain state, the instantaneous rain rate R (mm/hr) is
drawn from a lognormal distribution with median ``r_med_mm_hr`` and log-space
spread ``rate_sigma``. ITU-R P.837 defines the standard point rainfall-rate
statistic used in link design, R_0.01 (rate exceeded 0.01% of an average
year) — it is referenced here only as the *concept* motivating why a rain
rate distribution with a heavy right tail is used; this package does not
implement the ITU-R P.837 global rain-zone tables, and ``r_med_mm_hr`` /
``rate_sigma`` are illustrative user-supplied parameters, not looked-up
climatological values.

Source (concept only): ITU-R Recommendation P.837, "Characteristics of
precipitation for propagation modelling."

Specific rain attenuation
-----------------------------
    gamma_R(R) = k * R^alpha    [dB/km],  R in mm/hr

is the ITU-R P.838 power-law form for specific attenuation due to rain.
Source (functional form only): ITU-R Recommendation P.838, "Specific
attenuation model for rain for use in prediction methods." **The (k, alpha)
values used as defaults in this package are illustrative placeholders, not
verified against the current ITU-R P.838 coefficient tables** (those tables
depend on frequency, polarization and temperature); supply your own (k,
alpha) for a specific, verified frequency/polarization.

Path attenuation
--------------------
    A(R) = gamma_R(R) * L_eff(R),   L_eff(R) = L_km / (1 + L_km / L0_km)

is a SIMPLIFIED effective-path-length reduction factor in the spirit of the
ITU-R P.618 concept that a real path is not uniformly rained on end-to-end
(so specific attenuation should not simply be multiplied by the full
geometric path length). This is not a reproduction of the exact ITU-R
P.618-13 procedure (which uses a rain-rate- and frequency-dependent
reduction factor derived from a specific empirical fit) — ``L0_km`` here is
a free "attenuation correlation length" configuration parameter, not a table
lookup. Source (concept only): ITU-R Recommendation P.618, "Propagation data
and prediction methods required for the design of Earth-space
telecommunication systems."

RF link margin and availability
------------------------------------
    SNR_rf(t) [dB] = snr_clear_db - A(t)
    available(t)   = SNR_rf(t) >= snr_min_db

a standard fixed-margin link-budget decision (Friis-style dB bookkeeping;
not tied to a specific citation beyond standard link-budget practice).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "rain_markov_transition_probs",
    "simulate_rain_indicator",
    "rain_specific_attenuation_db_per_km",
    "rf_path_attenuation_db",
    "RFParams",
]


def rain_markov_transition_probs(p_rain: float, mean_event_steps: float) -> tuple[float, float]:
    """Two-state Markov transition probabilities from (p_rain, mean_event_steps).

    Returns (p_clear_to_rain, p_rain_to_clear). See module docstring for the
    derivation. Raises ValueError for parameters outside the valid range.
    """
    p_rain = float(p_rain)
    mean_event_steps = float(mean_event_steps)
    if not (0.0 < p_rain < 1.0):
        raise ValueError(f"p_rain must be in (0, 1), got {p_rain!r}")
    if not (math.isfinite(mean_event_steps) and mean_event_steps >= 1.0):
        raise ValueError(f"mean_event_steps must be finite and >= 1, got {mean_event_steps!r}")
    p_rain_to_clear = 1.0 / mean_event_steps
    p_clear_to_rain = p_rain * p_rain_to_clear / (1.0 - p_rain)
    if p_clear_to_rain > 1.0:
        raise ValueError(
            f"p_rain={p_rain:g} is too large for mean_event_steps={mean_event_steps:g}: "
            f"implied p_clear_to_rain={p_clear_to_rain:g} > 1"
        )
    return p_clear_to_rain, p_rain_to_clear


def simulate_rain_indicator(
    rng: np.random.Generator, n_steps: int, p_rain: float, mean_event_steps: float
) -> np.ndarray:
    """Simulate a boolean two-state rain-occurrence time series.

    Stationary chain: the initial state is drawn from the stationary
    distribution (Bernoulli(p_rain)) so the series has no warm-up transient.
    """
    if not isinstance(n_steps, (int, np.integer)) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    p_clear_to_rain, p_rain_to_clear = rain_markov_transition_probs(p_rain, mean_event_steps)
    u = rng.random(n_steps)
    raining = np.empty(n_steps, dtype=bool)
    raining[0] = rng.random() < p_rain
    for t in range(1, n_steps):
        if raining[t - 1]:
            raining[t] = u[t] >= p_rain_to_clear  # stays raining w.p. 1 - p_rain_to_clear
        else:
            raining[t] = u[t] < p_clear_to_rain
    return raining


def rain_specific_attenuation_db_per_km(rain_rate_mm_hr: np.ndarray | float, k: float,
                                         alpha: float) -> np.ndarray:
    """ITU-R P.838 specific attenuation gamma_R = k * R^alpha [dB/km]."""
    r = np.asarray(rain_rate_mm_hr, dtype=float)
    if np.any(r < 0.0) or not np.all(np.isfinite(r)):
        raise ValueError("rain_rate_mm_hr must be finite and >= 0")
    k = float(k)
    alpha = float(alpha)
    if not (math.isfinite(k) and k > 0.0):
        raise ValueError(f"k must be finite and > 0, got {k!r}")
    if not (math.isfinite(alpha) and alpha > 0.0):
        raise ValueError(f"alpha must be finite and > 0, got {alpha!r}")
    return k * np.power(r, alpha)


def rf_path_attenuation_db(
    rain_rate_mm_hr: np.ndarray | float, k: float, alpha: float, path_length_km: float,
    reduction_length_km: float,
) -> np.ndarray:
    """Total path attenuation A = gamma_R(R) * L_eff [dB]; see module docstring."""
    path_length_km = float(path_length_km)
    reduction_length_km = float(reduction_length_km)
    if not (math.isfinite(path_length_km) and path_length_km > 0.0):
        raise ValueError(f"path_length_km must be finite and > 0, got {path_length_km!r}")
    if not (math.isfinite(reduction_length_km) and reduction_length_km > 0.0):
        raise ValueError(
            f"reduction_length_km must be finite and > 0, got {reduction_length_km!r}"
        )
    gamma_r = rain_specific_attenuation_db_per_km(rain_rate_mm_hr, k, alpha)
    l_eff = path_length_km / (1.0 + path_length_km / reduction_length_km)
    return gamma_r * l_eff


@dataclass(frozen=True)
class RFParams:
    """Configuration for the RF channel (rain occurrence + rain-fade attenuation).

    Attributes
    ----------
    p_rain : stationary fraction of time raining, in (0, 1).
    mean_event_steps : mean rain-event duration in steps, >= 1.
    r_med_mm_hr, rate_sigma : lognormal rain-rate-when-raining parameters.
    k, alpha : ITU-R P.838 specific-attenuation coefficients (illustrative).
    path_length_km : RF path length, > 0.
    reduction_length_km : simplified path-reduction-factor length scale, > 0.
    snr_clear_db : clear-sky RF link SNR margin (dB).
    snr_min_db : minimum SNR (dB) for the RF link to be considered available.
    rate_mbps : delivered rate (Mb/s) when the RF link is available, > 0.
    """

    p_rain: float = 0.04
    mean_event_steps: float = 20.0
    r_med_mm_hr: float = 8.0
    rate_sigma: float = 0.7
    k: float = 0.07
    alpha: float = 1.10
    path_length_km: float = 5.0
    reduction_length_km: float = 20.0
    snr_clear_db: float = 25.0
    snr_min_db: float = 6.0
    rate_mbps: float = 150.0

    def __post_init__(self) -> None:
        rain_markov_transition_probs(self.p_rain, self.mean_event_steps)  # validates
        if not (math.isfinite(self.r_med_mm_hr) and self.r_med_mm_hr > 0.0):
            raise ValueError(f"r_med_mm_hr must be finite and > 0, got {self.r_med_mm_hr!r}")
        if not (math.isfinite(self.rate_sigma) and self.rate_sigma > 0.0):
            raise ValueError(f"rate_sigma must be finite and > 0, got {self.rate_sigma!r}")
        rain_specific_attenuation_db_per_km(1.0, self.k, self.alpha)  # validates k, alpha
        if not (math.isfinite(self.path_length_km) and self.path_length_km > 0.0):
            raise ValueError(f"path_length_km must be finite and > 0, got {self.path_length_km!r}")
        if not (math.isfinite(self.reduction_length_km) and self.reduction_length_km > 0.0):
            raise ValueError(
                f"reduction_length_km must be finite and > 0, got {self.reduction_length_km!r}"
            )
        if not math.isfinite(self.snr_clear_db):
            raise ValueError(f"snr_clear_db must be finite, got {self.snr_clear_db!r}")
        if not math.isfinite(self.snr_min_db):
            raise ValueError(f"snr_min_db must be finite, got {self.snr_min_db!r}")
        if self.snr_min_db > self.snr_clear_db:
            raise ValueError(
                f"snr_min_db ({self.snr_min_db}) must not exceed snr_clear_db "
                f"({self.snr_clear_db}); the link would never be available"
            )
        if not (math.isfinite(self.rate_mbps) and self.rate_mbps > 0.0):
            raise ValueError(f"rate_mbps must be finite and > 0, got {self.rate_mbps!r}")

    @property
    def margin_db(self) -> float:
        """Clear-sky margin above the minimum-SNR availability floor, dB."""
        return self.snr_clear_db - self.snr_min_db
