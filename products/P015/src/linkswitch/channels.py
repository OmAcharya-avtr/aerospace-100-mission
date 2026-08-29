"""Channel models for a hybrid RF-optical link: optical scintillation fading and RF rain fade.

**The fading models in this module are SIMULATED, not measured.** No parameter,
trace or statistic produced here has been validated against field data of any
kind. See ``DATASET_CARD.md``.

Optical channel
---------------
Mean-normalised irradiance ``I`` is modelled as **lognormal**, the weak-fluctuation
result of first-order Rytov theory:

    ln I ~ N(-sigma_z^2 / 2, sigma_z^2),    sigma_z^2 = ln(1 + sigma_I^2)

with ``sigma_I^2`` the scintillation index (normalised irradiance variance).
The ``-sigma_z^2/2`` mean makes ``E[I] = 1``.

Source: L. C. Andrews and R. L. Phillips, *Laser Beam Propagation through Random
Media*, 2nd ed., SPIE Press, 2005, Ch. 8-9 (lognormal irradiance PDF and its
parameterisation by the scintillation index).
Units: ``sigma_I^2`` and ``sigma_z`` dimensionless; ``I`` dimensionless.
Validity: weak fluctuations, Rytov variance ``sigma_R^2 <~ 1``. Beyond that the
lognormal law *underestimates* deep fades and gamma-gamma (Al-Habash, Andrews &
Phillips, *Opt. Eng.* 40(8), 1554-1562, 2001) should be used instead. This
module deliberately follows the lognormal lineage already established in this
portfolio (P001 BeamTwin, P010 BERBench) so that fade statistics are comparable;
gamma-gamma is **not** implemented (README "Limitations").

Temporal structure is added as a stationary first-order Gauss-Markov (AR(1))
process on ``ln I`` with correlation time equal to the Fresnel-zone crossing
time under the Taylor frozen-flow hypothesis:

    t_F = sqrt(lambda * L) / v_perp        [s]
    rho = exp(-dt / t_F)                   [-]

Sources: G. I. Taylor, "The spectrum of turbulence", *Proc. R. Soc. Lond. A*
164(919), 476-490, 1938 (frozen-flow hypothesis); Andrews & Phillips 2005,
Ch. 12 (temporal statistics of optical scintillation; the Fresnel scale
sqrt(lambda L) swept past the aperture at the transverse wind speed sets the
scintillation quasi-frequency). The AR(1) form itself is a modelling choice: it
reproduces the correct one-point marginal and a single correlation time, but not
the full Kolmogorov temporal power spectrum (README "Limitations").

RF channel
----------
Rain-induced specific attenuation follows the power law

    gamma_R = k * R^alpha                  [dB/km],  R in mm/h

with ``k`` and ``alpha`` frequency-, polarisation- and temperature-dependent
coefficients tabulated in **Recommendation ITU-R P.838-3** (2005), "Specific
attenuation model for rain for use in prediction methods". The coefficients are
*not* reproduced in this package: the caller must supply values taken from the
Recommendation. The defaults used in the shipped examples are round placeholder
numbers, explicitly flagged as such.

Path attenuation is ``A_rain = gamma_R * L_eff`` with ``L_eff`` an effective
path length; the slant-path effective-length methodology is
**Recommendation ITU-R P.618** ("Propagation data and prediction methods
required for the design of Earth-space telecommunication systems"). This module
does not implement the P.618 reduction-factor geometry; ``L_eff`` is an input.

Slow optical obscuration
------------------------
Scintillation alone does not justify a hybrid architecture: its fades last a
few Fresnel-crossing times, which is comparable to a radio handover guard. The
event that does justify one is a slow, deep obscuration -- cloud, fog or haze
crossing the path -- lasting seconds to minutes. This module adds an optional
excess attenuation ``A_obs(t)`` [dB] on top of the scintillation, generated as a
slow correlated excursion process (see :func:`simulate_lognormal_excursion`).

Fog and haze attenuation of near-infrared beams is strongly wavelength- and
visibility-dependent; the standard empirical treatment is V. Kim, B. McArthur
and E. Korevaar, "Comparison of laser beam propagation at 785 nm and 1550 nm in
fog and haze for optical wireless communications", *Proc. SPIE* 4214, 26-37,
2001. **The temporal model used here is a modelling choice with no standard
behind it**: only the order of magnitude of the attenuation depth is taken from
the literature, and the event durations, onset shapes and occurrence rates are
free parameters chosen by the user. Nothing about it has been compared with
measured transmissometer or cloud-cover data.

The rain time series is generated as a slowly correlated Gaussian process
transformed to a lognormal rain rate, with no rain below an excursion
threshold. That construction follows the *style* of
**Recommendation ITU-R P.1853** ("Time series synthesis of tropospheric
impairments"), which synthesises rain attenuation by low-pass filtering a white
Gaussian process and applying a lognormal transform. **This is a simplified
surrogate, not an implementation of P.1853**: none of the Recommendation's
calibrated filter or distribution parameters are used, and the resulting
statistics have not been checked against P.837 rainfall-rate maps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

__all__ = [
    "DB_PER_NEPER",
    "OpticalChannelParams",
    "RainChannelParams",
    "ar1_rho",
    "fresnel_crossing_time",
    "lognormal_sigma_ln",
    "rain_specific_attenuation_db_per_km",
    "simulate_ar1_unit",
    "simulate_lognormal_excursion",
    "simulate_optical_margin_db",
    "simulate_rain_attenuation_db",
    "sigma_i2_rytov_plane",
]

#: 10 / ln(10): converts a natural-log irradiance deviation to decibels of
#: *optical received power* (power is proportional to irradiance, so 10 log10).
DB_PER_NEPER = 10.0 / math.log(10.0)

#: Rytov variance above which the lognormal model is not trustworthy
#: (Andrews & Phillips 2005, Ch. 9). Used only for warnings/validation flags.
WEAK_REGIME_LIMIT = 1.0


def _positive(name: str, value: float) -> float:
    """Validate a strictly positive finite float, raising an actionable error."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {v}")
    return v


def _nonnegative(name: str, value: float) -> float:
    """Validate a non-negative finite float, raising an actionable error."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    if v < 0.0:
        raise ValueError(f"{name} must be >= 0, got {v}")
    return v


def sigma_i2_rytov_plane(cn2: float, wavelength_m: float, path_length_m: float) -> float:
    """Plane-wave Rytov variance = weak-fluctuation scintillation index.

    ``sigma_R^2 = 1.23 * Cn^2 * k^(7/6) * L^(11/6)``, ``k = 2*pi/lambda``.

    Source: Andrews & Phillips, *Laser Beam Propagation through Random Media*,
    2nd ed., SPIE Press, 2005 (standard Kolmogorov plane-wave result).
    Units: ``cn2`` [m^(-2/3)], ``wavelength_m`` [m], ``path_length_m`` [m];
    returns a dimensionless variance.
    Assumptions: Kolmogorov spectrum, no inner/outer scale, constant Cn^2 along
    a horizontal homogeneous path, point receiver (no aperture averaging).
    Validity: the returned value equals the scintillation index only for
    ``sigma_R^2 <~ 1``.

    This is provided as a convenience for parameterising a scenario; the
    portfolio's dedicated turbulence products (P003 ScintiNet, P001 BeamTwin)
    cover aperture averaging and wave-optics simulation and are not duplicated
    here.
    """
    c = _nonnegative("cn2", cn2)
    lam = _positive("wavelength_m", wavelength_m)
    ell = _positive("path_length_m", path_length_m)
    k = 2.0 * math.pi / lam
    return 1.23 * c * k ** (7.0 / 6.0) * ell ** (11.0 / 6.0)


def lognormal_sigma_ln(sigma_i2: float) -> float:
    """Standard deviation of ``ln I`` for lognormal irradiance [dimensionless].

    ``sigma_z = sqrt(ln(1 + sigma_I^2))`` (Andrews & Phillips 2005, Ch. 8).
    Units: input and output dimensionless. Validity: weak fluctuations.
    """
    s = _nonnegative("sigma_i2", sigma_i2)
    return math.sqrt(math.log1p(s))


def fresnel_crossing_time(wavelength_m: float, path_length_m: float, wind_mps: float) -> float:
    """Fresnel-zone crossing time ``t_F = sqrt(lambda L) / v_perp`` [s].

    Source: Taylor frozen-flow hypothesis (Taylor 1938) applied to the
    Fresnel scale sqrt(lambda L), the dominant scale for weak-fluctuation
    scintillation (Andrews & Phillips 2005, Ch. 12).
    Units: [m], [m], [m/s] -> [s].
    Assumptions: single transverse wind speed, frozen turbulence, plane wave.
    Validity: order-of-magnitude estimate of the scintillation correlation
    time; the true temporal autocorrelation is not a single exponential.
    """
    lam = _positive("wavelength_m", wavelength_m)
    ell = _positive("path_length_m", path_length_m)
    v = _positive("wind_mps", wind_mps)
    return math.sqrt(lam * ell) / v


def ar1_rho(dt_s: float, tau_s: float) -> float:
    """Lag-1 correlation of a first-order Gauss-Markov process, ``exp(-dt/tau)``.

    Units: [s], [s] -> dimensionless in (0, 1).
    """
    dt = _positive("dt_s", dt_s)
    tau = _positive("tau_s", tau_s)
    return math.exp(-dt / tau)


def simulate_ar1_unit(
    n: int, rho: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Stationary zero-mean unit-variance AR(1) sequence.

    ``z[t] = rho * z[t-1] + sqrt(1 - rho^2) * eps[t]``, ``eps ~ N(0, 1)``,
    initialised from the stationary distribution so the whole sequence is
    stationary (no burn-in transient).

    Parameters
    ----------
    n : int
        Number of samples, >= 1.
    rho : float
        Lag-1 correlation in [0, 1).
    rng : numpy.random.Generator
        Seeded generator.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``, dimensionless.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be an integer >= 1, got {n!r}")
    r = float(rho)
    if not (0.0 <= r < 1.0):
        raise ValueError(f"rho must satisfy 0 <= rho < 1, got {r}")
    eps = rng.standard_normal(int(n))
    z0 = rng.standard_normal()
    b = [math.sqrt(1.0 - r * r)]
    a = [1.0, -r]
    out, _ = lfilter(b, a, eps, zi=[r * z0])
    return np.asarray(out, dtype=float)


def simulate_lognormal_excursion(
    n_steps: int,
    dt_s: float,
    prob: float,
    median: float,
    sigma_ln: float,
    tau_s: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Slow intermittent lognormal excursion process (zero most of the time).

    A stationary unit AR(1) process ``g(t)`` with correlation time ``tau_s`` is
    thresholded at ``g_thr = Phi^{-1}(1 - prob)``. Where ``g > g_thr`` the
    output is ``median * exp(sigma_ln * (g - g_thr))``; elsewhere it is exactly
    zero. The long-run fraction of non-zero samples is ``prob`` by construction,
    the value at onset is ``median``, and events have a mean duration set by
    ``tau_s``.

    Used for both the rain rate [mm/h] and the optical obscuration depth [dB].
    This is a **surrogate** generator, in the spirit of (but not an
    implementation of) the filtered-Gaussian/lognormal synthesis of
    Recommendation ITU-R P.1853; its statistics have not been fitted to any
    measured dataset.

    Parameters
    ----------
    n_steps : int
        Samples, >= 1.
    dt_s : float
        Sample interval [s], > 0.
    prob : float
        Long-run fraction of non-zero samples [-] in [0, 1).
    median : float
        Output value at the onset threshold, in the units of the output, > 0.
    sigma_ln : float
        Standard deviation of the log-excess [-], > 0.
    tau_s : float
        Correlation time of the driving process [s], > 0.
    rng : numpy.random.Generator
        Seeded generator.
    """
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps must be an integer >= 1, got {n_steps!r}")
    dt = _positive("dt_s", dt_s)
    if not (0.0 <= prob < 1.0):
        raise ValueError(f"prob must be in [0, 1), got {prob}")
    out = np.zeros(int(n_steps), dtype=float)
    if prob == 0.0:
        return out
    _positive("median", median)
    _positive("sigma_ln", sigma_ln)
    from scipy.stats import norm  # local import: keeps module import cheap

    g = simulate_ar1_unit(int(n_steps), ar1_rho(dt, tau_s), rng)
    g_thr = float(norm.isf(prob))
    active = g > g_thr
    out[active] = median * np.exp(sigma_ln * (g[active] - g_thr))
    return out


@dataclass(frozen=True)
class OpticalChannelParams:
    """Parameters of the simulated optical channel.

    Attributes
    ----------
    mean_margin_db : float
        Mean received-power margin above the optical receiver's FEC threshold,
        in the absence of scintillation [dB]. Positive means the link closes
        on average.
    sigma_i2 : float
        Scintillation index (normalised irradiance variance) [-], > 0.
    correlation_time_s : float
        Scintillation correlation time [s], > 0 (e.g. from
        :func:`fresnel_crossing_time`).
    turbulence_drift_sigma : float
        Standard deviation of ``ln`` of the slow turbulence-strength modulation
        [-]. ``0.0`` gives a strictly stationary channel; positive values make
        ``sigma_I^2`` wander slowly, which is the non-stationary regime where a
        history-based predictor can in principle beat a memoryless threshold.
    turbulence_drift_time_s : float
        Correlation time of that slow modulation [s], > 0.
    telemetry_noise_db : float
        1-sigma Gaussian noise on the *measured* margin reported by the receiver
        [dB], >= 0. Does not affect the true margin.
    obscuration_prob : float
        Long-run fraction of time with non-zero slow obscuration (cloud, fog,
        haze) on the optical path [-] in [0, 1). ``0.0`` disables it and gives a
        pure scintillation channel, which is the regime the closed forms in
        :mod:`linkswitch.analytic` describe exactly.
    obscuration_median_db : float
        Excess attenuation at event onset [dB], > 0. Order of magnitude from
        visibility-based fog/haze attenuation (Kim, McArthur & Korevaar 2001);
        the temporal behaviour is a modelling choice, not a standard.
    obscuration_sigma_ln : float
        Standard deviation of the log excess attenuation within an event [-].
    obscuration_time_s : float
        Correlation time of the obscuration process [s], > 0. Must be well
        separated from ``correlation_time_s`` for the two effects to be
        distinguishable from telemetry.
    """

    mean_margin_db: float = 3.0
    sigma_i2: float = 0.5
    correlation_time_s: float = 6.8e-3
    turbulence_drift_sigma: float = 0.0
    turbulence_drift_time_s: float = 2.0
    telemetry_noise_db: float = 0.0
    obscuration_prob: float = 0.0
    obscuration_median_db: float = 10.0
    obscuration_sigma_ln: float = 0.5
    obscuration_time_s: float = 3.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean_margin_db):
            raise ValueError(f"mean_margin_db must be finite, got {self.mean_margin_db}")
        _positive("sigma_i2", self.sigma_i2)
        _positive("correlation_time_s", self.correlation_time_s)
        _nonnegative("turbulence_drift_sigma", self.turbulence_drift_sigma)
        _positive("turbulence_drift_time_s", self.turbulence_drift_time_s)
        _nonnegative("telemetry_noise_db", self.telemetry_noise_db)
        if not (0.0 <= self.obscuration_prob < 1.0):
            raise ValueError(
                f"obscuration_prob must be in [0, 1), got {self.obscuration_prob}"
            )
        _positive("obscuration_median_db", self.obscuration_median_db)
        _positive("obscuration_sigma_ln", self.obscuration_sigma_ln)
        _positive("obscuration_time_s", self.obscuration_time_s)
        if self.obscuration_prob > 0.0 and self.obscuration_time_s < 10.0 * (
            self.correlation_time_s
        ):
            raise ValueError(
                "obscuration_time_s must be >= 10x correlation_time_s so that the two "
                f"processes are separable; got {self.obscuration_time_s} vs "
                f"{self.correlation_time_s}"
            )

    @property
    def sigma_ln(self) -> float:
        """``sigma_z`` of ``ln I`` [-]."""
        return lognormal_sigma_ln(self.sigma_i2)

    @property
    def margin_std_db(self) -> float:
        """Standard deviation of the scintillation-only optical margin [dB].

        Valid as the margin standard deviation only when
        ``obscuration_prob = 0`` and ``turbulence_drift_sigma = 0``.
        """
        return DB_PER_NEPER * self.sigma_ln

    @property
    def margin_mean_db(self) -> float:
        """Mean of the scintillation-only optical margin [dB].

        ``E[10 log10 I] = (10/ln10) * E[ln I] = -(10/ln10) * sigma_z^2 / 2``,
        so the mean margin sits *below* ``mean_margin_db`` by that amount even
        though ``E[I] = 1``. Valid only when ``obscuration_prob = 0`` and
        ``turbulence_drift_sigma = 0``.
        """
        return self.mean_margin_db - DB_PER_NEPER * 0.5 * self.sigma_ln**2

    @property
    def weak_regime_valid(self) -> bool:
        """True when ``sigma_I^2 < 1`` (lognormal model regarded as usable)."""
        return self.sigma_i2 < WEAK_REGIME_LIMIT


def simulate_optical_margin_db(
    params: OpticalChannelParams,
    n_steps: int,
    dt_s: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate the true and measured optical margin time series.

    Returns ``(true_margin_db, measured_margin_db)``, both shape ``(n_steps,)``.

    Model (units in brackets):
    ``m(t) = mean_margin_db + (10/ln10) * ln I(t)`` [dB], with ``ln I`` a
    stationary Gauss-Markov process of marginal ``N(-sigma_z(t)^2/2,
    sigma_z(t)^2)`` and lag-1 correlation ``exp(-dt/tau)``.

    When ``turbulence_drift_sigma > 0``, ``sigma_z`` is multiplied by
    ``exp(u(t) - turbulence_drift_sigma^2/2)`` with ``u`` a slow AR(1) of
    standard deviation ``turbulence_drift_sigma``; the ``-s^2/2`` term keeps
    ``E[exp(u - s^2/2)] = 1`` so the *mean* turbulence strength is unchanged.
    This modulated-Gaussian construction preserves the instantaneous marginal
    only when the modulation is slow compared with ``correlation_time_s``;
    that condition is checked and violations raise ``ValueError``.

    When ``obscuration_prob > 0`` a slow excess attenuation ``A_obs(t)`` [dB]
    from :func:`simulate_lognormal_excursion` is subtracted from the margin. The
    result is then **not** Gaussian and the closed forms of
    :mod:`linkswitch.analytic` no longer apply.
    """
    if not isinstance(params, OpticalChannelParams):
        raise TypeError(f"params must be OpticalChannelParams, got {type(params)!r}")
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps must be an integer >= 1, got {n_steps!r}")
    dt = _positive("dt_s", dt_s)
    rho = ar1_rho(dt, params.correlation_time_s)
    z = simulate_ar1_unit(int(n_steps), rho, rng)

    sigma_z = params.sigma_ln
    if params.turbulence_drift_sigma > 0.0:
        if params.turbulence_drift_time_s < 10.0 * params.correlation_time_s:
            raise ValueError(
                "turbulence_drift_time_s must be >= 10x correlation_time_s for the "
                f"slow-modulation approximation to hold; got "
                f"{params.turbulence_drift_time_s} vs {params.correlation_time_s}"
            )
        rho_u = ar1_rho(dt, params.turbulence_drift_time_s)
        u = simulate_ar1_unit(int(n_steps), rho_u, rng) * params.turbulence_drift_sigma
        scale = np.exp(u - 0.5 * params.turbulence_drift_sigma**2)
        sigma_t = sigma_z * scale
    else:
        sigma_t = np.full(int(n_steps), sigma_z, dtype=float)

    ln_i = -0.5 * sigma_t**2 + sigma_t * z
    true_margin = params.mean_margin_db + DB_PER_NEPER * ln_i
    if params.obscuration_prob > 0.0:
        true_margin = true_margin - simulate_lognormal_excursion(
            int(n_steps),
            dt,
            params.obscuration_prob,
            params.obscuration_median_db,
            params.obscuration_sigma_ln,
            params.obscuration_time_s,
            rng,
        )
    if params.telemetry_noise_db > 0.0:
        measured = true_margin + params.telemetry_noise_db * rng.standard_normal(int(n_steps))
    else:
        measured = true_margin.copy()
    return true_margin, measured


def rain_specific_attenuation_db_per_km(
    rain_rate_mm_per_h: NDArray[np.float64] | float, k: float, alpha: float
) -> NDArray[np.float64]:
    """Specific rain attenuation ``gamma_R = k * R^alpha`` [dB/km].

    Source: Recommendation ITU-R P.838-3 (2005), "Specific attenuation model
    for rain for use in prediction methods". ``k`` and ``alpha`` depend on
    frequency, polarisation tilt and path elevation and are tabulated in that
    Recommendation; **they are not reproduced here and must be supplied by the
    caller**.
    Units: ``R`` [mm/h], returns [dB/km].
    Validity: the P.838-3 regression covers roughly 1-1000 GHz; ``R >= 0``.
    """
    kk = _positive("k", k)
    aa = _positive("alpha", alpha)
    r = np.asarray(rain_rate_mm_per_h, dtype=float)
    if np.any(r < 0.0):
        raise ValueError("rain_rate_mm_per_h must be >= 0 everywhere")
    return kk * np.power(r, aa)


@dataclass(frozen=True)
class RainChannelParams:
    """Parameters of the simulated RF (rain-faded) channel.

    ``k_itu`` and ``alpha_itu`` have **no defaults**: they must be taken from
    Recommendation ITU-R P.838-3 for the frequency, polarisation and elevation
    of interest.

    Attributes
    ----------
    k_itu, alpha_itu : float
        ITU-R P.838-3 power-law coefficients, ``gamma_R = k R^alpha`` [dB/km].
    effective_path_km : float
        Effective rain path length [km] (ITU-R P.618 methodology), > 0.
    clear_sky_margin_db : float
        RF margin above the RF demodulator threshold in the absence of rain
        [dB].
    rain_prob : float
        Long-run fraction of time with non-zero rain rate [-], in [0, 1).
    median_rain_rate_mm_per_h : float
        Median rain rate *conditional on raining* [mm/h], > 0.
    rain_rate_sigma_ln : float
        Standard deviation of ``ln R`` conditional on raining [-], > 0.
    rain_correlation_time_s : float
        Correlation time of the underlying slow Gaussian process [s], > 0.
    """

    k_itu: float
    alpha_itu: float
    effective_path_km: float = 5.0
    clear_sky_margin_db: float = 6.0
    rain_prob: float = 0.05
    median_rain_rate_mm_per_h: float = 6.0
    rain_rate_sigma_ln: float = 1.0
    rain_correlation_time_s: float = 120.0

    def __post_init__(self) -> None:
        _positive("k_itu", self.k_itu)
        _positive("alpha_itu", self.alpha_itu)
        _positive("effective_path_km", self.effective_path_km)
        if not math.isfinite(self.clear_sky_margin_db):
            raise ValueError("clear_sky_margin_db must be finite")
        if not (0.0 <= self.rain_prob < 1.0):
            raise ValueError(f"rain_prob must be in [0, 1), got {self.rain_prob}")
        _positive("median_rain_rate_mm_per_h", self.median_rain_rate_mm_per_h)
        _positive("rain_rate_sigma_ln", self.rain_rate_sigma_ln)
        _positive("rain_correlation_time_s", self.rain_correlation_time_s)


def simulate_rain_attenuation_db(
    params: RainChannelParams,
    n_steps: int,
    dt_s: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate ``(rain_rate_mm_per_h, rain_attenuation_db)`` time series.

    A stationary unit AR(1) process ``g(t)`` with correlation time
    ``rain_correlation_time_s`` is thresholded at the ``1 - rain_prob``
    quantile of the standard normal; where it exceeds the threshold the rain
    rate is ``median_rain_rate * exp(rain_rate_sigma_ln * (g - g_thr))`` [mm/h],
    elsewhere zero. Attenuation is ``k R^alpha * effective_path_km`` [dB]
    (ITU-R P.838-3 power law, ITU-R P.618 effective-path formulation).

    This is a **simplified surrogate** in the spirit of ITU-R P.1853 time-series
    synthesis, not an implementation of it, and it has not been checked against
    ITU-R P.837 rainfall-rate statistics.
    """
    if not isinstance(params, RainChannelParams):
        raise TypeError(f"params must be RainChannelParams, got {type(params)!r}")
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps must be an integer >= 1, got {n_steps!r}")
    dt = _positive("dt_s", dt_s)
    rate = simulate_lognormal_excursion(
        int(n_steps),
        dt,
        params.rain_prob,
        params.median_rain_rate_mm_per_h,
        params.rain_rate_sigma_ln,
        params.rain_correlation_time_s,
        rng,
    )
    atten = rain_specific_attenuation_db_per_km(rate, params.k_itu, params.alpha_itu)
    return rate, atten * params.effective_path_km
