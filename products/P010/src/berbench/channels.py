"""Channel models: AWGN and lognormal fading (weak-turbulence FSO).

Lognormal irradiance model
--------------------------
Under weak atmospheric turbulence the received optical irradiance I is well
modelled as lognormal (first-order Rytov theory):

    ln I ~ N(mu_z, sigma_z^2)

Mean-normalised irradiance E[I] = 1 requires mu_z = -sigma_z^2 / 2, and the
scintillation index (normalised irradiance variance) is

    sigma_I^2 = Var[I] / E[I]^2 = exp(sigma_z^2) - 1
    =>  sigma_z^2 = ln(1 + sigma_I^2)

Sources:
- L. C. Andrews and R. L. Phillips, "Laser Beam Propagation through Random
  Media", 2nd ed., SPIE Press, 2005 (Ch. 8-9: weak-fluctuation lognormal
  model and scintillation index).
- X. Zhu and J. M. Kahn, "Free-space optical communication through
  atmospheric turbulence channels", IEEE Trans. Commun. 50(8), 1293-1300,
  2002 (lognormal fading BER analysis for FSO links).

Validity: the lognormal model is a *weak-fluctuation* result, generally
accepted for sigma_I^2 < ~1 (equivalently Rytov variance < ~1). For stronger
turbulence, gamma-gamma or negative-exponential statistics are required and
are OUT OF SCOPE for this package. A UserWarning is emitted for
sigma_I^2 > 1.

Units: irradiance is mean-normalised (dimensionless); sigma_I^2 is
dimensionless.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from ._math import gauss_hermite

__all__ = [
    "CHANNELS",
    "validate_channel",
    "lognormal_sigma_z",
    "lognormal_irradiance_nodes",
    "sample_lognormal_irradiance",
]

CHANNELS = ("awgn", "lognormal")

#: Weak-fluctuation validity limit for the scintillation index (Andrews &
#: Phillips 2005, Ch. 9): lognormal statistics are questionable beyond this.
SIGMA_I2_WEAK_LIMIT = 1.0


def validate_channel(channel: str, sigma_i2: float | None) -> None:
    """Validate the channel name / parameter combination.

    Raises ValueError on unknown channel, on lognormal without a positive
    scintillation index, or on non-finite sigma_i2. Emits UserWarning when
    sigma_i2 exceeds the weak-fluctuation validity limit (1.0).
    """
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; expected one of {CHANNELS}")
    if channel == "lognormal":
        if sigma_i2 is None:
            raise ValueError("lognormal channel requires sigma_i2 (scintillation index) > 0")
        sigma_i2 = float(sigma_i2)
        if not math.isfinite(sigma_i2) or sigma_i2 <= 0.0:
            raise ValueError(f"sigma_i2 must be a finite value > 0, got {sigma_i2!r}")
        if sigma_i2 > SIGMA_I2_WEAK_LIMIT:
            warnings.warn(
                f"sigma_i2={sigma_i2:g} exceeds the weak-fluctuation validity limit "
                f"(~{SIGMA_I2_WEAK_LIMIT:g}); the lognormal model is not trustworthy here "
                "(Andrews & Phillips 2005, Ch. 9). Results are extrapolations.",
                UserWarning,
                stacklevel=3,
            )
    elif sigma_i2 is not None:
        raise ValueError("sigma_i2 is only meaningful for channel='lognormal'")


def lognormal_sigma_z(sigma_i2: float) -> float:
    """Log-amplitude-domain std dev sigma_z from scintillation index sigma_I^2.

    sigma_z^2 = ln(1 + sigma_I^2); mean-normalised so mu_z = -sigma_z^2/2.
    Source: Andrews & Phillips 2005, Eq. (9.7)-(9.11) region (lognormal pdf
    parameterisation via the scintillation index).
    """
    if sigma_i2 <= 0:
        raise ValueError(f"sigma_i2 must be > 0, got {sigma_i2}")
    return math.sqrt(math.log1p(sigma_i2))


def lognormal_irradiance_nodes(
    sigma_i2: float, n_nodes: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes (I_i) and normalised weights for E_I[f(I)].

    With ln I ~ N(-sigma_z^2/2, sigma_z^2) the expectation over the lognormal
    pdf transforms to a Gauss-Hermite sum:

        E[f(I)] = (1/sqrt(pi)) * sum_i w_i f( exp(sqrt(2) sigma_z x_i
                                                - sigma_z^2 / 2) )

    Returns (irradiance_nodes, weights) with weights already divided by
    sqrt(pi) so that ``E[f(I)] ~ sum_i weights_i * f(I_i)``.

    This is the standard technique for averaging conditional BER over the
    lognormal fading pdf (Zhu & Kahn 2002, Sec. III).
    """
    sigma_z = lognormal_sigma_z(sigma_i2)
    x, w = gauss_hermite(n_nodes)
    irr = np.exp(math.sqrt(2.0) * sigma_z * x - 0.5 * sigma_z * sigma_z)
    return irr, w / math.sqrt(math.pi)


def sample_lognormal_irradiance(
    rng: np.random.Generator, size: int, sigma_i2: float
) -> np.ndarray:
    """Draw i.i.d. mean-normalised lognormal irradiance samples.

    ln I ~ N(-sigma_z^2/2, sigma_z^2), sigma_z^2 = ln(1 + sigma_I^2), E[I] = 1.
    Per-symbol i.i.d. sampling gives the ergodic (long-term average) BER;
    real turbulence is correlated over ~ms coherence times, which does not
    change the average BER, only its short-term statistics.
    """
    sigma_z = lognormal_sigma_z(sigma_i2)
    return rng.lognormal(mean=-0.5 * sigma_z * sigma_z, sigma=sigma_z, size=size)
