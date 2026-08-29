"""Named reference scenarios and the fixed seed partition used for all evidence.

Two scenarios are defined.

**Scenario A - stationary scintillation only.** No slow obscuration, no
telemetry noise, no handover guard, RF always available. The optical margin is
then exactly Gaussian AR(1), which is the setting in which the closed forms of
:mod:`linkswitch.analytic` are *exact*. Scenario A exists so that the analytic
optimal threshold can be checked against Monte Carlo without any modelling
approximation in between.

**Scenario B - operational hybrid link.** Scintillation *plus* slow obscuration
events, noisy telemetry, a rain-faded RF channel and a non-zero handover guard.
The closed forms do not apply here; this is the scenario on which the three
switching policies are compared.

Seed partition (disjoint by construction, asserted in the test suite):

===================  ==================  ==========================
Purpose              Seeds               Used by
===================  ==================  ==========================
Predictor training   1000-1020           ``OutagePredictor.fit``
Policy tuning        3000-3039           threshold / ``p_star`` search
Held-out test        5000-5099           all reported comparisons
Horizon study        7000-7039           ``v3_horizon_sensitivity.py``
Figures / examples   9000-9009           ``examples/``
===================  ==================  ==========================

Parameter provenance
--------------------
The optical numbers describe a 3 km horizontal 1550 nm terrestrial link in
moderate turbulence: ``sigma_I^2 = 0.5`` is inside the weak-fluctuation range,
and the correlation time is the Fresnel-crossing time at a 3 m/s transverse
wind (:func:`linkswitch.channels.fresnel_crossing_time`).

The RF numbers are **not climatological**. ``k_itu`` and ``alpha_itu`` are round
placeholders, not values from Recommendation ITU-R P.838-3, and the rain
occurrence probability, median rate and correlation time were chosen so that RF
outages are actually exercised inside a 24-second Monte Carlo trial. Real rain
events are rarer and much longer (see Recommendation ITU-R P.837 for rainfall
rate statistics and ITU-R P.618 for the attenuation prediction method); with
realistic values almost every trial would be rain-free and the RF-outage
behaviour of the policies would be untested. This is a deliberate,
compute-budget-driven distortion and it is the single largest reason the
absolute throughput numbers in this package must not be read as predictions of
any real link.

Nothing here is measured. See ``DATASET_CARD.md``.
"""

from __future__ import annotations

from .channels import OpticalChannelParams, RainChannelParams, fresnel_crossing_time
from .scenario import HybridLinkScenario

__all__ = [
    "FIGURE_SEEDS",
    "HORIZON_SEEDS",
    "REFERENCE_PATH_M",
    "REFERENCE_WAVELENGTH_M",
    "REFERENCE_WIND_MPS",
    "TEST_SEEDS",
    "TRAIN_SEEDS",
    "TUNE_SEEDS",
    "scenario_a_stationary",
    "scenario_b_operational",
]

#: Optical link geometry the reference correlation time is derived from.
REFERENCE_WAVELENGTH_M = 1.55e-6
REFERENCE_PATH_M = 3.0e3
REFERENCE_WIND_MPS = 3.0

TRAIN_SEEDS = tuple(range(1000, 1021))
TUNE_SEEDS = tuple(range(3000, 3040))
TEST_SEEDS = tuple(range(5000, 5100))
HORIZON_SEEDS = tuple(range(7000, 7040))
FIGURE_SEEDS = tuple(range(9000, 9010))


def _correlation_time_s() -> float:
    """Fresnel-crossing time of the reference geometry [s]."""
    return fresnel_crossing_time(REFERENCE_WAVELENGTH_M, REFERENCE_PATH_M, REFERENCE_WIND_MPS)


def scenario_a_stationary(n_steps: int = 20000, dt_s: float = 2.0e-3) -> HybridLinkScenario:
    """Scenario A: pure scintillation, noiseless telemetry, no guard, RF always up.

    The optical margin is exactly Gaussian with
    ``mean = optical.margin_mean_db`` and ``std = optical.margin_std_db`` and
    lag-1 correlation ``scenario.rho``, so
    :func:`linkswitch.analytic.optimal_fixed_threshold_db` and
    :func:`linkswitch.analytic.expected_throughput_fixed_threshold` are exact.
    """
    optical = OpticalChannelParams(
        mean_margin_db=3.0,
        sigma_i2=0.5,
        correlation_time_s=_correlation_time_s(),
        telemetry_noise_db=0.0,
        obscuration_prob=0.0,
    )
    rf = RainChannelParams(
        k_itu=0.10,  # placeholder, NOT an ITU-R P.838-3 value
        alpha_itu=1.10,  # placeholder, NOT an ITU-R P.838-3 value
        effective_path_km=5.0,
        clear_sky_margin_db=6.0,
        rain_prob=0.0,  # RF never fades: isolates the optical decision problem
    )
    return HybridLinkScenario(
        optical=optical,
        rf=rf,
        dt_s=dt_s,
        n_steps=n_steps,
        rate_optical_bps=1.0e9,
        rate_rf_bps=2.5e8,
        switch_penalty_steps=0,
    )


def scenario_b_operational(n_steps: int = 12000, dt_s: float = 2.0e-3) -> HybridLinkScenario:
    """Scenario B: scintillation + obscuration + rain-faded RF + handover guard.

    The 2 ms guard (one sample) models a hot-standby RF chain in which both
    radios are already powered and a switch costs only a protocol reframing
    interval. A cold-standby architecture would need tens of milliseconds and,
    as shown in ``validation/VALIDATION.md``, would make switching at
    scintillation timescales counter-productive.
    """
    optical = OpticalChannelParams(
        mean_margin_db=6.0,
        sigma_i2=0.5,
        correlation_time_s=_correlation_time_s(),
        telemetry_noise_db=0.2,
        obscuration_prob=0.10,
        obscuration_median_db=8.0,
        obscuration_sigma_ln=0.6,
        obscuration_time_s=3.0,
    )
    rf = RainChannelParams(
        k_itu=0.10,  # placeholder, NOT an ITU-R P.838-3 value
        alpha_itu=1.10,  # placeholder, NOT an ITU-R P.838-3 value
        effective_path_km=5.0,
        clear_sky_margin_db=6.0,
        rain_prob=0.15,
        median_rain_rate_mm_per_h=12.0,
        rain_rate_sigma_ln=0.8,
        rain_correlation_time_s=40.0,
    )
    return HybridLinkScenario(
        optical=optical,
        rf=rf,
        dt_s=dt_s,
        n_steps=n_steps,
        rate_optical_bps=1.0e9,
        rate_rf_bps=2.5e8,
        switch_penalty_steps=1,
    )
