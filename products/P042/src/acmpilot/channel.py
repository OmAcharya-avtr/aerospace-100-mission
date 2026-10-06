"""Temporally correlated optical fading channel with an emergent fade duration.

What this module is for
-----------------------
Adaptive coding and modulation under feedback delay is only interesting if the
channel has *memory*: if successive slots were independent, a stale channel
measurement would carry no information at all and the only sensible policy
would be a fixed conservative rate. This module therefore generates amplitude
sample paths whose **temporal correlation time is a stated input** and whose
**fade duration and level-crossing rate are emergent sample statistics**, never
parameters.

The driver process
------------------
Both marginal families below are driven by a first-order Gauss-Markov
(Ornstein-Uhlenbeck, discretely sampled) process, which is the standard
minimal model for a filtered-Gaussian sample path:

    z[n+1] = rho * z[n] + sqrt(1 - rho**2) * w[n],   w[n] ~ N(0, 1)       (1)
    rho    = exp(-dt / tau_c)                                            (2)

``z`` is unit-variance and stationary, and its autocovariance is

    R_z(k) = rho**|k| = exp(-|k| dt / tau_c)                             (3)

so ``tau_c`` is the 1/e correlation time of the driver in seconds. Equation
(1) is the exact discrete-time sampling of the OU stochastic differential
equation; it is not an approximation of it.

Lognormal amplitude (weak turbulence)
-------------------------------------
With log-amplitude ``chi = sigma_chi * z``, irradiance normalised to unit mean
is

    I[n] = exp(2 * chi[n] - 2 * sigma_chi**2)                             (4)

and the scintillation index of (4) is the standard lognormal result

    sigma_I**2 = exp(4 * sigma_chi**2) - 1                                (5)

Equations (4)-(5) are the weak-turbulence lognormal irradiance model used
throughout the free-space-optics literature; see Andrews & Phillips, *Laser
Beam Propagation through Random Media* (SPIE Press, 2nd ed., 2005) for the
derivation and its validity range. Validity: weak fluctuations, roughly
``sigma_I**2 < 1``; the model is known to underpredict the deep-fade tail in
moderate-to-strong turbulence, which is the reason the gamma-gamma family
below is also provided.

Gamma-gamma amplitude (weak to strong turbulence)
-------------------------------------------------
The gamma-gamma irradiance model of Al-Habash, Andrews & Phillips, "Mathematical
model for the irradiance probability density function of a laser beam
propagating through turbulent media", *Optical Engineering* 40(8), 2001, writes
the normalised irradiance as a product of two independent unit-mean gamma
variates representing large-scale and small-scale scattering:

    I = X_large * X_small,
    X_large  ~ Gamma(shape=alpha, scale=1/alpha)
    X_small  ~ Gamma(shape=beta,  scale=1/beta)                           (6)

whose scintillation index is

    sigma_I**2 = 1/alpha + 1/beta + 1/(alpha*beta)                        (7)

Equation (7) is checked numerically in ``validation/validate_channel.py``
rather than asserted.

To give (6) a temporal structure, two **independent** Gauss-Markov drivers
``z_l``, ``z_s`` from (1) are mapped to the gamma marginals through the
Gaussian copula

    X = Gamma_ppf(Phi(z), shape, scale)                                   (8)

This preserves the gamma marginals exactly (so (7) still holds) but the
irradiance autocorrelation is a monotone *transform* of (3) rather than (3)
itself, so the measured 1/e correlation time of a gamma-gamma irradiance path
is shorter than ``tau_c``. That discrepancy is measured and reported in
``validation/validate_channel.py``; it is a property of the copula
construction, not a bug, and ``tau_c`` must be read as the correlation time of
the *driver*. A joint model with an exactly prescribed gamma-gamma
autocorrelation is outside the scope of this package.

Nothing here is a measured turbulence spectrum
----------------------------------------------
A real turbulence temporal spectrum is set by the transverse wind speed and the
aperture-filtered Kolmogorov spatial spectrum, and it is not a single
exponential. ``tau_c`` is an engineering knob standing in for that, and every
number this package produces is a property of this simulator.

From irradiance to SNR
----------------------
The electrical SNR per symbol is

    SNR_dB[n] = SNR0_dB + 10 * detector_exponent * log10(I[n])            (9)

``detector_exponent = 1`` for a coherent (homodyne/heterodyne) receiver, where
electrical SNR is proportional to received optical power; ``= 2`` for
thermal-noise-limited direct detection, where the photocurrent is proportional
to ``I`` and electrical SNR to ``I**2``. The default is 1 because the MODCOD
set in :mod:`acmpilot.modcod` is a coherent set. ``SNR0_dB`` is the SNR that
would be received at the mean irradiance ``E[I] = 1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import special, stats

Marginal = Literal["lognormal", "gamma-gamma"]


def gauss_markov_path(
    n_samples: int,
    *,
    dt_s: float,
    tau_c_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unit-variance stationary Gauss-Markov sample path, equations (1)-(2).

    Parameters
    ----------
    n_samples
        Number of samples (slots) to generate, must be >= 1.
    dt_s
        Sample (slot) interval in seconds, must be > 0.
    tau_c_s
        Driver correlation time in seconds (the 1/e point of equation (3)),
        must be > 0.
    rng
        NumPy generator; the path is fully determined by its state.

    Returns
    -------
    ndarray, shape (n_samples,), dimensionless
        Zero-mean unit-variance path. The first sample is drawn from the
        stationary distribution, so there is no burn-in transient.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if dt_s <= 0:
        raise ValueError(f"dt_s must be > 0 s, got {dt_s}")
    if tau_c_s <= 0:
        raise ValueError(f"tau_c_s must be > 0 s, got {tau_c_s}")

    rho = float(np.exp(-dt_s / tau_c_s))
    innovation = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    noise = rng.standard_normal(n_samples)
    path = np.empty(n_samples, dtype=float)
    path[0] = noise[0]
    # Stationary AR(1) recursion. n_samples is <= a few 1e5 here, and the
    # recursion is inherently sequential, so an explicit loop is used rather
    # than a cumulative-product trick that loses precision as rho -> 1.
    for i in range(1, n_samples):
        path[i] = rho * path[i - 1] + innovation * noise[i]
    return path


def lognormal_irradiance(driver: np.ndarray, *, sigma_i2: float) -> np.ndarray:
    """Unit-mean lognormal irradiance from a unit-variance driver, equations (4)-(5).

    Parameters
    ----------
    driver
        Unit-variance Gauss-Markov path from :func:`gauss_markov_path`.
    sigma_i2
        Target scintillation index (dimensionless), must be > 0. Inverting
        equation (5) gives ``sigma_chi**2 = log1p(sigma_i2) / 4``.

    Returns
    -------
    ndarray, dimensionless
        Irradiance normalised so that the ensemble mean is exactly 1.
    """
    if sigma_i2 <= 0:
        raise ValueError(f"sigma_i2 must be > 0, got {sigma_i2}")
    sigma_chi2 = float(np.log1p(sigma_i2)) / 4.0
    sigma_chi = float(np.sqrt(sigma_chi2))
    return np.exp(2.0 * sigma_chi * np.asarray(driver, dtype=float) - 2.0 * sigma_chi2)


def gamma_gamma_shapes(sigma_i2: float, *, ratio: float = 4.0) -> tuple[float, float]:
    """Solve equation (7) for ``(alpha, beta)`` at a stated shape ratio.

    Equation (7) is one equation in two unknowns, so a second condition is
    required. Here ``alpha = ratio * beta``, with ``ratio > 1`` reflecting the
    usual ordering ``alpha > beta`` (large-scale scattering less severe than
    small-scale). With ``a = alpha``, ``b = beta``, ``a = r*b``:

        sigma_i2 = 1/(r b) + 1/b + 1/(r b**2)

    which is a quadratic in ``1/b`` solved in closed form below.

    Parameters
    ----------
    sigma_i2
        Scintillation index, must be > 0.
    ratio
        ``alpha / beta``, must be >= 1.

    Returns
    -------
    (alpha, beta)
        Gamma shape parameters, dimensionless.
    """
    if sigma_i2 <= 0:
        raise ValueError(f"sigma_i2 must be > 0, got {sigma_i2}")
    if ratio < 1:
        raise ValueError(f"ratio must be >= 1, got {ratio}")
    r = float(ratio)
    # (1/r) u**2 + (1 + 1/r) u - sigma_i2 = 0 with u = 1/b
    qa = 1.0 / r
    qb = 1.0 + 1.0 / r
    qc = -float(sigma_i2)
    u = (-qb + np.sqrt(qb * qb - 4.0 * qa * qc)) / (2.0 * qa)
    beta = 1.0 / float(u)
    return r * beta, beta


def gamma_gamma_irradiance(
    driver_large: np.ndarray,
    driver_small: np.ndarray,
    *,
    sigma_i2: float,
    ratio: float = 4.0,
) -> np.ndarray:
    """Unit-mean gamma-gamma irradiance by Gaussian copula, equations (6)-(8).

    Parameters
    ----------
    driver_large, driver_small
        Two **independent** unit-variance Gauss-Markov paths of equal length.
    sigma_i2
        Scintillation index (dimensionless), must be > 0.
    ratio
        ``alpha / beta``, passed to :func:`gamma_gamma_shapes`.

    Returns
    -------
    ndarray, dimensionless
        Irradiance with exact gamma-gamma marginal and unit mean. The
        correlation time of this series is shorter than that of its drivers;
        see the module docstring.
    """
    big = np.asarray(driver_large, dtype=float)
    small = np.asarray(driver_small, dtype=float)
    if big.shape != small.shape:
        raise ValueError(f"driver shapes must match, got {big.shape} and {small.shape}")
    alpha, beta = gamma_gamma_shapes(sigma_i2, ratio=ratio)
    u_big = special.ndtr(big)
    u_small = special.ndtr(small)
    eps = np.finfo(float).tiny
    x_big = stats.gamma.ppf(np.clip(u_big, eps, 1.0 - 1e-15), a=alpha, scale=1.0 / alpha)
    x_small = stats.gamma.ppf(np.clip(u_small, eps, 1.0 - 1e-15), a=beta, scale=1.0 / beta)
    return x_big * x_small


@dataclass(frozen=True)
class ChannelConfig:
    """Channel configuration. All times in seconds, all SNR in dB.

    Attributes
    ----------
    slot_s
        Slot (sample) interval, seconds. Default 1e-3 s.
    tau_c_s
        Driver correlation time, seconds. Default 10e-3 s.
    sigma_i2
        Scintillation index, dimensionless. Default 0.5.
    mean_snr_db
        SNR per symbol at the mean irradiance ``E[I] = 1``, dB.
    marginal
        ``"lognormal"`` or ``"gamma-gamma"``.
    gg_ratio
        ``alpha / beta`` for the gamma-gamma marginal.
    detector_exponent
        1 for coherent detection, 2 for thermal-limited direct detection; see
        equation (9).
    """

    slot_s: float = 1e-3
    tau_c_s: float = 10e-3
    sigma_i2: float = 0.5
    mean_snr_db: float = 14.0
    marginal: Marginal = "lognormal"
    gg_ratio: float = 4.0
    detector_exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.slot_s <= 0:
            raise ValueError(f"slot_s must be > 0 s, got {self.slot_s}")
        if self.tau_c_s <= 0:
            raise ValueError(f"tau_c_s must be > 0 s, got {self.tau_c_s}")
        if self.sigma_i2 <= 0:
            raise ValueError(f"sigma_i2 must be > 0, got {self.sigma_i2}")
        if self.marginal not in ("lognormal", "gamma-gamma"):
            raise ValueError(
                f"marginal must be 'lognormal' or 'gamma-gamma', got {self.marginal!r}"
            )
        if self.detector_exponent <= 0:
            raise ValueError(
                f"detector_exponent must be > 0, got {self.detector_exponent}"
            )

    @property
    def rho(self) -> float:
        """Driver lag-1 correlation coefficient, equation (2), dimensionless."""
        return float(np.exp(-self.slot_s / self.tau_c_s))

    def delay_slots(self, tau_s: float) -> int:
        """Feedback delay in whole slots, ``round(tau_s / slot_s)``, >= 0."""
        if tau_s < 0:
            raise ValueError(f"tau_s must be >= 0 s, got {tau_s}")
        return int(round(tau_s / self.slot_s))


def irradiance_path(config: ChannelConfig, n_slots: int, seed: int) -> np.ndarray:
    """Unit-mean irradiance sample path for ``config``, reproducible from ``seed``.

    Returns
    -------
    ndarray, shape (n_slots,), dimensionless
    """
    rng = np.random.default_rng(seed)
    if config.marginal == "lognormal":
        driver = gauss_markov_path(
            n_slots, dt_s=config.slot_s, tau_c_s=config.tau_c_s, rng=rng
        )
        return lognormal_irradiance(driver, sigma_i2=config.sigma_i2)
    d_large = gauss_markov_path(
        n_slots, dt_s=config.slot_s, tau_c_s=config.tau_c_s, rng=rng
    )
    d_small = gauss_markov_path(
        n_slots, dt_s=config.slot_s, tau_c_s=config.tau_c_s, rng=rng
    )
    return gamma_gamma_irradiance(
        d_large, d_small, sigma_i2=config.sigma_i2, ratio=config.gg_ratio
    )


def snr_db_path(config: ChannelConfig, n_slots: int, seed: int) -> np.ndarray:
    """SNR-per-symbol sample path in dB, equation (9).

    Returns
    -------
    ndarray, shape (n_slots,), dB
    """
    irradiance = irradiance_path(config, n_slots, seed)
    return config.mean_snr_db + 10.0 * config.detector_exponent * np.log10(irradiance)


def correlation_time_s(series: np.ndarray, *, dt_s: float) -> float:
    """Measured 1/e correlation time of ``series``, seconds.

    The sample autocorrelation of the mean-removed series is scanned for the
    first lag at which it drops below ``1/e``, and that crossing is linearly
    interpolated between the bracketing lags. This is a **measurement** of a
    sample path, not an inversion of equation (3); for a lognormal path driven
    at ``tau_c`` it recovers ``tau_c`` because the log of (4) is affine in the
    driver, and for a gamma-gamma path it does not. Returns ``nan`` if the
    autocorrelation never crosses ``1/e`` within the series.
    """
    x = np.asarray(series, dtype=float)
    if x.size < 8:
        raise ValueError(f"series needs >= 8 samples, got {x.size}")
    if dt_s <= 0:
        raise ValueError(f"dt_s must be > 0 s, got {dt_s}")
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        raise ValueError("series is constant; correlation time is undefined")
    max_lag = min(x.size - 1, 4096)
    target = float(np.exp(-1.0))
    prev = 1.0
    for lag in range(1, max_lag + 1):
        acf = float(np.dot(x[:-lag], x[lag:]) / denom)
        if acf < target:
            frac = (prev - target) / (prev - acf) if prev != acf else 0.0
            return (lag - 1 + frac) * dt_s
        prev = acf
    return float("nan")


def fade_statistics(
    series_db: np.ndarray, threshold_db: float, *, dt_s: float
) -> dict[str, float]:
    """Emergent fade statistics of a dB series below ``threshold_db``.

    Fades are maximal runs of consecutive samples with ``series_db <
    threshold_db``. Runs touching either end of the series are **excluded from
    the duration statistics** because they are right-censored, but they are
    still counted in ``outage_fraction``. ``level_crossing_rate_hz`` counts
    downward crossings only, divided by the total observation time, which is the
    convention that makes ``mean_fade_duration_s * level_crossing_rate_hz``
    approximately equal to ``outage_fraction``.

    Returns
    -------
    dict
        ``outage_fraction`` (dimensionless), ``n_fades`` (count of complete
        fades), ``mean_fade_duration_s``, ``median_fade_duration_s``,
        ``max_fade_duration_s`` (seconds, ``nan`` if no complete fade),
        ``level_crossing_rate_hz`` (downward crossings per second).
    """
    x = np.asarray(series_db, dtype=float)
    if x.size < 2:
        raise ValueError(f"series needs >= 2 samples, got {x.size}")
    if dt_s <= 0:
        raise ValueError(f"dt_s must be > 0 s, got {dt_s}")
    below = x < threshold_db
    total_time = x.size * dt_s
    down = int(np.count_nonzero(~below[:-1] & below[1:]))
    padded = np.concatenate(([False], below, [False]))
    change = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = change[0::2], change[1::2]
    lengths = ends - starts
    complete = np.ones(starts.size, dtype=bool)
    if starts.size:
        complete &= starts > 0
        complete &= ends < x.size
    durations = lengths[complete] * dt_s
    return {
        "outage_fraction": float(np.count_nonzero(below) / x.size),
        "n_fades": float(durations.size),
        "mean_fade_duration_s": float(durations.mean()) if durations.size else float("nan"),
        "median_fade_duration_s": (
            float(np.median(durations)) if durations.size else float("nan")
        ),
        "max_fade_duration_s": float(durations.max()) if durations.size else float("nan"),
        "level_crossing_rate_hz": down / total_time,
    }


def rice_level_crossing_rate_hz(
    *,
    sigma_i2: float,
    tau_c_s: float,
    threshold_db: float,
    mean_snr_db: float,
    slot_s: float = 1e-3,
) -> float:
    """Analytic downward level-crossing rate for the lognormal channel, Hz.

    Rice's formula for a stationary differentiable Gaussian process ``chi`` with
    variance ``sigma_chi**2`` and derivative variance ``sigma_dot**2`` gives the
    downward crossing rate of a level ``chi_th`` as

        N = (1 / (2*pi)) * (sigma_dot / sigma_chi)
            * exp(-chi_th**2 / (2 * sigma_chi**2))                      (10)

    (S. O. Rice, "Mathematical analysis of random noise", *Bell System
    Technical Journal* 23(3) 1944 and 24(1) 1945.) The Gauss-Markov driver of
    equation (1) is **not** differentiable in continuous time --- the
    continuous-time crossing rate of an Ornstein-Uhlenbeck process diverges ---
    so (10) is applied with the derivative variance implied by the *discrete*
    sampling at interval ``dt``:

        sigma_dot**2 = 2 * sigma_chi**2 * (1 - rho) / dt**2,
        rho          = exp(-dt / tau_c)                                   (11)

    The prediction is therefore **sampling-rate dependent by construction**: a
    path sampled twice as fast has more measurable crossings, and both (10)-(11)
    and the sample statistic move together. Within that stated convention the
    agreement with the measured crossing rate is tight, and
    ``validation/validate_channel.py`` reports the measured ratio.

    Here the level in dB maps to log-amplitude through equations (4) and (9)
    with ``detector_exponent = 1``:

        chi_th = (ln(10) / 20) * (threshold_db - mean_snr_db) + sigma_chi**2   (12)

    Parameters
    ----------
    sigma_i2
        Scintillation index, dimensionless, > 0.
    tau_c_s
        Driver correlation time, seconds, > 0.
    threshold_db
        Level whose downward crossings are counted, dB.
    mean_snr_db
        SNR at the mean irradiance, dB, from equation (9).
    slot_s
        Sampling interval ``dt`` of equation (11), seconds, > 0.

    Returns
    -------
    float
        Downward crossings per second.
    """
    if sigma_i2 <= 0:
        raise ValueError(f"sigma_i2 must be > 0, got {sigma_i2}")
    if tau_c_s <= 0:
        raise ValueError(f"tau_c_s must be > 0 s, got {tau_c_s}")
    if slot_s <= 0:
        raise ValueError(f"slot_s must be > 0 s, got {slot_s}")
    sigma_chi2 = float(np.log1p(sigma_i2)) / 4.0
    chi_th = (np.log(10.0) / 20.0) * (threshold_db - mean_snr_db) + sigma_chi2
    rho = float(np.exp(-slot_s / tau_c_s))
    # Equation (11): discrete-sampling derivative standard deviation over
    # sigma_chi, which is what (10) needs as a ratio.
    ratio = float(np.sqrt(2.0 * (1.0 - rho)) / slot_s)
    rate = (ratio / (2.0 * np.pi)) * np.exp(-(chi_th * chi_th) / (2.0 * sigma_chi2))
    return float(rate)
