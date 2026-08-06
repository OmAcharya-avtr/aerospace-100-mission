"""Fade statistics for FSO Monte Carlo samples and analytic baselines.

References
----------
- L. C. Andrews and R. L. Phillips, *Laser Beam Propagation through Random
  Media*, 2nd ed., SPIE Press, 2005 - lognormal fade probability.
- A. Agresti and B. A. Coull, "Approximate is better than 'exact' for
  interval estimation of binomial proportions", The American Statistician
  52(2), 1998 - Wilson score interval used for Monte Carlo confidence
  bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

_LN10 = math.log(10.0)


@dataclass(frozen=True)
class FadeEstimate:
    """Monte Carlo fade-probability estimate with Wilson 95 % interval."""

    probability: float
    ci_low: float
    ci_high: float
    n_samples: int
    n_fades: int


def fade_probability(samples_dbm: np.ndarray, sensitivity_dbm: float) -> FadeEstimate:
    """Estimate P(received power < sensitivity) from Monte Carlo samples.

    Confidence bounds: Wilson score interval at 95 % (z = 1.96), which
    remains well-behaved for zero observed fades (Agresti & Coull 1998).
    Units: samples_dbm and sensitivity_dbm in dBm.
    """
    samples = np.asarray(samples_dbm, dtype=float)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples_dbm must be a non-empty 1-D array")
    if not math.isfinite(sensitivity_dbm):
        raise ValueError(f"sensitivity_dbm must be finite, got {sensitivity_dbm}")
    n = int(samples.size)
    k = int(np.count_nonzero(samples < sensitivity_dbm))
    p_hat = k / n
    z = 1.959963984540054  # 97.5th percentile of the standard normal
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return FadeEstimate(
        probability=p_hat,
        ci_low=max(0.0, centre - half),
        ci_high=min(1.0, centre + half),
        n_samples=n,
        n_fades=k,
    )


def analytic_fade_probability_lognormal(margin_db: float, sigma_ln: float) -> float:
    """Closed-form fade probability for scintillation-only lognormal fading.

    With mean-normalised irradiance X = I / <I>, ln X ~ N(-sigma_ln^2/2,
    sigma_ln^2), and a deterministic margin M [dB] above the threshold, a
    fade occurs when X < 10^(-M/10):

        P_fade = Phi( (ln 10^(-M/10) + sigma_ln^2 / 2) / sigma_ln )
               = Q( (M * ln10/10 - sigma_ln^2/2) / sigma_ln )

    where Phi is the standard normal CDF and Q = 1 - Phi. This is the
    standard lognormal fade/threshold probability (Andrews & Phillips 2005,
    Ch. 11, probability of fade below threshold for lognormal irradiance).
    Units: margin_db [dB], sigma_ln dimensionless (std of ln I).
    Validity: scintillation only (no jitter), weak-fluctuation regime.
    Edge case sigma_ln = 0: returns 1.0 if margin_db < 0 else 0.0.
    """
    if not math.isfinite(margin_db):
        raise ValueError(f"margin_db must be finite, got {margin_db}")
    if sigma_ln < 0.0 or not math.isfinite(sigma_ln):
        raise ValueError(f"sigma_ln must be >= 0 and finite, got {sigma_ln}")
    if sigma_ln == 0.0:
        return 1.0 if margin_db < 0.0 else 0.0
    ln_threshold = -margin_db * _LN10 / 10.0
    return float(norm.cdf((ln_threshold + 0.5 * sigma_ln**2) / sigma_ln))


def margin_percentiles(
    samples_dbm: np.ndarray,
    sensitivity_dbm: float,
    percentiles: tuple[float, ...] = (1.0, 5.0, 50.0, 95.0, 99.0),
) -> dict[str, float]:
    """Percentiles of the instantaneous fade margin (received - sensitivity).

    Units: dB. Percentiles given in percent, keys formatted 'p01', 'p05', ...
    """
    samples = np.asarray(samples_dbm, dtype=float)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples_dbm must be a non-empty 1-D array")
    for q in percentiles:
        if not (0.0 <= q <= 100.0):
            raise ValueError(f"percentiles must be in [0, 100], got {q}")
    margins = samples - sensitivity_dbm
    values = np.percentile(margins, percentiles)
    return {f"p{q:02.0f}": float(v) for q, v in zip(percentiles, values)}


def margin_moments(samples_dbm: np.ndarray, sensitivity_dbm: float) -> dict[str, float]:
    """Mean and variance of the instantaneous margin [dB, dB^2]."""
    samples = np.asarray(samples_dbm, dtype=float)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples_dbm must be a non-empty 1-D array")
    margins = samples - sensitivity_dbm
    return {
        "mean_db": float(np.mean(margins)),
        "variance_db2": float(np.var(margins)),
        "std_db": float(np.std(margins)),
    }


__all__ = [
    "FadeEstimate",
    "analytic_fade_probability_lognormal",
    "fade_probability",
    "margin_moments",
    "margin_percentiles",
]
