"""Optical (FSO) channel: simulated fading irradiance and outage threshold.

**All fading here is SIMULATED. No field-measured turbulence or link data is
used anywhere in this package.**

Lognormal irradiance model (weak-to-moderate turbulence)
----------------------------------------------------------
Mean-normalised irradiance ``E[I] = 1``:

    ln I ~ N(mu_z, sigma_z^2),  mu_z = -sigma_z^2 / 2
    sigma_z^2 = ln(1 + sigma_I^2)

where sigma_I^2 = Var[I]/E[I]^2 is the scintillation index.

Source: L. C. Andrews and R. L. Phillips, *Laser Beam Propagation through
Random Media*, 2nd ed., SPIE Press, 2005 (Ch. 8-9). Validity: weak-fluctuation
regime, generally accepted for sigma_I^2 < ~1; a ``UserWarning`` is raised
above that. Units: irradiance is dimensionless (mean-normalised); sigma_I^2
is dimensionless.

Temporal correlation (AR(1) / Gauss-Markov)
--------------------------------------------
A single lognormal draw per step has no time structure. To generate a
*time series* with a specified coherence time, the standardised log-irradiance

    Z_t = (ln I_t - mu_z) / sigma_z

is propagated as a discrete AR(1) process

    Z_t = rho * Z_{t-1} + sqrt(1 - rho^2) * eps_t,   eps_t ~ N(0, 1) i.i.d.

which is stationary with ``Z_t ~ N(0, 1)`` marginally and lag-1 correlation
exactly ``rho`` for every t (elementary property of a stationary AR(1)
process; this is also the exact discrete-time transition law of an
Ornstein-Uhlenbeck process sampled at fixed intervals, i.e. a standard
Gauss-Markov construction). We set ``rho = exp(-dt / tau_c)`` for a
user-supplied coherence time ``tau_c`` (in units of steps) so ``rho`` decays
smoothly to 0 as the sample spacing grows relative to ``tau_c``.

**This AR(1) temporal model is an ENGINEERING APPROXIMATION.** It reproduces
the correct marginal lognormal PDF exactly and gives the fading process a
tunable coherence time, but it is not derived from a measured or published
temporal power spectrum of atmospheric scintillation (the real process has
structure set by wind speed, aperture averaging and inner/outer turbulence
scale that a single-pole AR(1) model does not capture). Treat it as a coarse,
clearly-labelled simulation device, not a validated temporal model.

Gamma-gamma irradiance model (moderate-to-strong turbulence)
--------------------------------------------------------------
``I = X_a * X_b``, with ``X_a ~ Gamma(alpha, 1/alpha)`` and
``X_b ~ Gamma(beta, 1/beta)`` independent, giving ``E[I] = 1`` and

    sigma_I^2 = 1/alpha + 1/beta + 1/(alpha*beta)

Source: M. A. Al-Habash, L. C. Andrews and R. L. Phillips, "Mathematical
model for the irradiance probability density function of a laser beam
propagating through turbulent media," *Optical Engineering* 40(8),
1554-1562, 2001 (large-scale/small-scale eddy decomposition). alpha and beta
are supplied directly by the caller (they are not derived here from a Rytov
variance / path-geometry formula — that closed-form mapping exists in
Al-Habash et al. 2001 but is not reproduced in this package to avoid
transcribing numeric constants that were not independently re-derived here).
Only i.i.d. per-step sampling is provided for gamma-gamma (no AR(1) temporal
correlation) — see ``DATASET_CARD.md`` / README limitations.

Outage threshold from link margin
------------------------------------
For a mean-normalised irradiance, a receiver power margin of ``margin_db``
decibels relative to the mean corresponds linearly to

    tau = 10^(-margin_db / 10)

by the definition of the decibel (a ratio in dB is 10*log10 of the linear
power ratio). This is a definitional identity, not a physical model.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SIGMA_I2_WEAK_LIMIT",
    "lognormal_sigma_z",
    "validate_sigma_i2",
    "irradiance_threshold_from_margin_db",
    "sample_lognormal_irradiance",
    "simulate_ar1_log_irradiance",
    "gamma_gamma_sigma_i2",
    "sample_gamma_gamma_irradiance",
    "validate_gamma_gamma_params",
    "OpticalParams",
]

#: Weak-fluctuation validity limit for the scintillation index (Andrews &
#: Phillips 2005): lognormal statistics are questionable beyond this.
SIGMA_I2_WEAK_LIMIT = 1.0


def validate_sigma_i2(sigma_i2: float) -> float:
    """Validate a scintillation index; warn if outside the weak-fluctuation range."""
    sigma_i2 = float(sigma_i2)
    if not math.isfinite(sigma_i2) or sigma_i2 <= 0.0:
        raise ValueError(f"sigma_i2 must be a finite value > 0, got {sigma_i2!r}")
    if sigma_i2 > SIGMA_I2_WEAK_LIMIT:
        warnings.warn(
            f"sigma_i2={sigma_i2:g} exceeds the weak-fluctuation validity limit "
            f"(~{SIGMA_I2_WEAK_LIMIT:g}); the lognormal model is an extrapolation "
            "here (Andrews & Phillips 2005, Ch. 9).",
            UserWarning,
            stacklevel=3,
        )
    return sigma_i2


def lognormal_sigma_z(sigma_i2: float) -> float:
    """Log-amplitude std dev sigma_z from the scintillation index sigma_I^2.

    sigma_z^2 = ln(1 + sigma_I^2). Source: Andrews & Phillips 2005, Ch. 9.
    """
    sigma_i2 = validate_sigma_i2(sigma_i2)
    return math.sqrt(math.log1p(sigma_i2))


def irradiance_threshold_from_margin_db(margin_db: float) -> float:
    """Mean-normalised irradiance below which the optical link is in outage.

    tau = 10^(-margin_db / 10). ``margin_db`` is the receiver power margin
    (dB) relative to the mean (no-fade) irradiance level; must be >= 0.
    """
    margin_db = float(margin_db)
    if not math.isfinite(margin_db) or margin_db < 0.0:
        raise ValueError(f"margin_db must be a finite value >= 0, got {margin_db!r}")
    return 10.0 ** (-margin_db / 10.0)


def sample_lognormal_irradiance(
    rng: np.random.Generator, size: int, sigma_i2: float
) -> np.ndarray:
    """Draw i.i.d. mean-normalised lognormal irradiance samples (no time structure)."""
    sigma_z = lognormal_sigma_z(sigma_i2)
    return rng.lognormal(mean=-0.5 * sigma_z * sigma_z, sigma=sigma_z, size=size)


def simulate_ar1_log_irradiance(
    rng: np.random.Generator,
    n_steps: int,
    sigma_i2: float,
    coherence_steps: float,
) -> np.ndarray:
    """Correlated lognormal irradiance time series via a standardised AR(1) process.

    Parameters
    ----------
    n_steps : number of time steps to generate, >= 1.
    sigma_i2 : scintillation index (see module docstring).
    coherence_steps : coherence time in units of the sample interval, > 0.
        ``rho = exp(-1/coherence_steps)``; ``coherence_steps -> inf`` gives
        rho -> 1 (near-constant fades), ``coherence_steps -> 0`` gives
        rho -> 0 (i.i.d. per step).

    Returns
    -------
    np.ndarray, shape (n_steps,) of mean-normalised irradiance I_t > 0.
    """
    if not isinstance(n_steps, (int, np.integer)) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    coherence_steps = float(coherence_steps)
    if not math.isfinite(coherence_steps) or coherence_steps <= 0.0:
        raise ValueError(f"coherence_steps must be finite and > 0, got {coherence_steps!r}")
    sigma_z = lognormal_sigma_z(sigma_i2)
    rho = math.exp(-1.0 / coherence_steps)

    eps = rng.standard_normal(n_steps)
    z = np.empty(n_steps)
    z[0] = eps[0]  # Z_0 ~ N(0,1) directly (stationary initial condition)
    one_minus_rho2 = math.sqrt(max(0.0, 1.0 - rho * rho))
    for t in range(1, n_steps):
        z[t] = rho * z[t - 1] + one_minus_rho2 * eps[t]

    log_i = -0.5 * sigma_z * sigma_z + sigma_z * z
    return np.exp(log_i)


def gamma_gamma_sigma_i2(alpha: float, beta: float) -> float:
    """Scintillation index of the gamma-gamma model: 1/a + 1/b + 1/(a*b).

    Source: Al-Habash, Andrews & Phillips, Opt. Eng. 40(8), 1554 (2001).
    """
    alpha, beta = validate_gamma_gamma_params(alpha, beta)
    return 1.0 / alpha + 1.0 / beta + 1.0 / (alpha * beta)


def validate_gamma_gamma_params(alpha: float, beta: float) -> tuple[float, float]:
    alpha = float(alpha)
    beta = float(beta)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be finite and > 0, got {alpha!r}")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"beta must be finite and > 0, got {beta!r}")
    return alpha, beta


def sample_gamma_gamma_irradiance(
    rng: np.random.Generator, size: int, alpha: float, beta: float
) -> np.ndarray:
    """Draw i.i.d. mean-normalised gamma-gamma irradiance samples.

    I = X_a * X_b, X_a ~ Gamma(alpha, 1/alpha), X_b ~ Gamma(beta, 1/beta),
    each shape/scale chosen so E[X_a] = E[X_b] = 1 and hence E[I] = 1.
    No temporal correlation is modelled (see module docstring).
    """
    alpha, beta = validate_gamma_gamma_params(alpha, beta)
    if not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size < 1:
        raise ValueError(f"size must be a positive integer, got {size!r}")
    x_a = rng.gamma(shape=alpha, scale=1.0 / alpha, size=size)
    x_b = rng.gamma(shape=beta, scale=1.0 / beta, size=size)
    return x_a * x_b


@dataclass(frozen=True)
class OpticalParams:
    """Configuration for the optical channel.

    Attributes
    ----------
    sigma_i2 : scintillation index (lognormal model), > 0.
    coherence_steps : AR(1) coherence time in steps, > 0.
    margin_db : link power margin (dB) defining the outage threshold, >= 0.
    rate_mbps : delivered rate (Mb/s) when the optical link is available, > 0.
    fading_model : "lognormal" (temporally correlated) is the only model
        wired into scenario telemetry generation; "gamma_gamma" is available
        as a standalone i.i.d. sampler (see ``sample_gamma_gamma_irradiance``).
    """

    sigma_i2: float = 0.25
    coherence_steps: float = 5.0
    margin_db: float = 6.0
    rate_mbps: float = 1000.0
    fading_model: str = "lognormal"

    def __post_init__(self) -> None:
        validate_sigma_i2(self.sigma_i2)
        if not (math.isfinite(self.coherence_steps) and self.coherence_steps > 0):
            raise ValueError(
                f"coherence_steps must be finite and > 0, got {self.coherence_steps!r}"
            )
        irradiance_threshold_from_margin_db(self.margin_db)  # validates margin_db
        if not (math.isfinite(self.rate_mbps) and self.rate_mbps > 0):
            raise ValueError(f"rate_mbps must be finite and > 0, got {self.rate_mbps!r}")
        if self.fading_model not in ("lognormal", "gamma_gamma"):
            raise ValueError(
                f"fading_model must be 'lognormal' or 'gamma_gamma', got {self.fading_model!r}"
            )

    @property
    def sigma_z(self) -> float:
        return lognormal_sigma_z(self.sigma_i2)

    @property
    def tau_phys(self) -> float:
        """Physical outage threshold (mean-normalised irradiance)."""
        return irradiance_threshold_from_margin_db(self.margin_db)
