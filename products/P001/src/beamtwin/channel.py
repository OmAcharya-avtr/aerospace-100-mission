"""Stochastic atmospheric channel for FSO links.

Models two random effects on received power:

1. Scintillation - lognormal irradiance fluctuations with scintillation
   index derived from the plane-wave Rytov variance (weak-fluctuation
   regime). Source: L. C. Andrews and R. L. Phillips, *Laser Beam
   Propagation through Random Media*, 2nd ed., SPIE Press, 2005.
2. Pointing jitter - Gaussian per-axis angular jitter producing a random
   transverse displacement at the receiver and hence a random pointing-loss
   factor (point-receiver Gaussian-beam model, see beamtwin.budget).

The combined received power is sampled by vectorised, seeded Monte Carlo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .budget import LinkBudget, LinkParams, compute_budget, dbm_from_watts, watts_from_dbm


def rytov_variance_plane_wave(cn2: float, wavelength_m: float, range_m: float) -> float:
    """Plane-wave Rytov variance.

    sigma_R^2 = 1.23 * Cn^2 * k^(7/6) * L^(11/6),   k = 2 pi / lambda

    Source: Andrews & Phillips 2005, Ch. 8 (weak-fluctuation scintillation
    index for a plane wave; the classical Rytov result).
    Units: cn2 [m^-2/3], wavelength_m [m], range_m [m], result dimensionless.
    Assumptions: homogeneous Kolmogorov turbulence along a horizontal path,
    constant Cn^2. Validity: weak-fluctuation regime sigma_R^2 < 1; beyond
    that the lognormal model underestimates fades (saturation regime).
    """
    if cn2 < 0.0 or not math.isfinite(cn2):
        raise ValueError(f"cn2 must be >= 0 and finite, got {cn2}")
    if wavelength_m <= 0.0 or range_m <= 0.0:
        raise ValueError("wavelength_m and range_m must be > 0")
    k = 2.0 * math.pi / wavelength_m
    return 1.23 * cn2 * k ** (7.0 / 6.0) * range_m ** (11.0 / 6.0)


def lognormal_sigma_ln(scintillation_index: float) -> float:
    """Standard deviation of ln(I) for a lognormal irradiance model.

    For lognormal irradiance with scintillation index sigma_I^2
    (= var(I)/E[I]^2):  sigma_ln^2 = ln(1 + sigma_I^2).

    Source: Andrews & Phillips 2005, Ch. 8 (lognormal PDF parameterisation).
    In the weak regime sigma_I^2 ~= sigma_R^2 (plane wave).
    Units: dimensionless.
    """
    if scintillation_index < 0.0 or not math.isfinite(scintillation_index):
        raise ValueError(f"scintillation_index must be >= 0, got {scintillation_index}")
    return math.sqrt(math.log1p(scintillation_index))


def mean_pointing_loss_fraction(sigma_disp_m: float, beam_radius_m: float) -> float:
    """Closed-form average pointing-loss factor under zero-bias Gaussian jitter.

    With per-axis displacement x, y ~ N(0, sigma_d^2) and point-receiver loss
    L_p = exp(-2 (x^2+y^2) / w^2):

        E[L_p] = 1 / (1 + 4 sigma_d^2 / w^2)

    Derivation: E[exp(-c x^2)] = (1 + 2 c sigma^2)^(-1/2) for x ~ N(0,
    sigma^2) (Gaussian integral), applied per axis with c = 2/w^2. Consistent
    with the beta-distributed pointing-loss model of Farid & Hranilovic,
    J. Lightwave Technol. 25(7), 2007 in the point-receiver limit.
    Units: lengths [m], result dimensionless in (0, 1].
    Assumptions: zero bias, independent identical per-axis jitter.
    """
    if sigma_disp_m < 0.0 or not math.isfinite(sigma_disp_m):
        raise ValueError(f"sigma_disp_m must be >= 0, got {sigma_disp_m}")
    if beam_radius_m <= 0.0:
        raise ValueError(f"beam_radius_m must be > 0, got {beam_radius_m}")
    return 1.0 / (1.0 + 4.0 * sigma_disp_m**2 / beam_radius_m**2)


@dataclass(frozen=True)
class ChannelParams:
    """Stochastic channel parameters.

    Attributes
    ----------
    cn2 : refractive-index structure parameter [m^-2/3], >= 0.
        Typical: 1e-16 (calm night) to 1e-13 (strong daytime turbulence)
        (Andrews & Phillips 2005, Ch. 12).
    pointing_jitter_rad : per-axis RMS angular jitter [rad], >= 0.
    """

    cn2: float = 1e-15
    pointing_jitter_rad: float = 0.0

    def __post_init__(self) -> None:
        if not (isinstance(self.cn2, (int, float)) and math.isfinite(self.cn2)):
            raise TypeError(f"cn2 must be a finite number, got {self.cn2!r}")
        if self.cn2 < 0.0:
            raise ValueError(f"cn2 must be >= 0, got {self.cn2}")
        if self.cn2 > 1e-11:
            raise ValueError(
                f"cn2={self.cn2} m^-2/3 is far above physical values (<= ~1e-12); check units"
            )
        j = self.pointing_jitter_rad
        if not (isinstance(j, (int, float)) and math.isfinite(j)):
            raise TypeError(f"pointing_jitter_rad must be a finite number, got {j!r}")
        if j < 0.0:
            raise ValueError(f"pointing_jitter_rad must be >= 0, got {j}")


@dataclass(frozen=True)
class ChannelModel:
    """Derived stochastic-channel quantities for a given link."""

    rytov_variance: float
    scintillation_index: float
    sigma_ln: float
    sigma_disp_m: float
    beam_radius_at_rx_m: float
    weak_regime_valid: bool


def build_channel_model(link: LinkParams, channel: ChannelParams) -> ChannelModel:
    """Derive scintillation and jitter statistics for a link/channel pair.

    weak_regime_valid is False when sigma_R^2 >= 1 (lognormal weak-
    fluctuation model outside its validity range; results then
    underestimate deep fades - Andrews & Phillips 2005, Ch. 9).
    """
    s_r2 = rytov_variance_plane_wave(channel.cn2, link.wavelength_m, link.range_m)
    # Weak regime: sigma_I^2 ~= sigma_R^2 (plane wave, Andrews & Phillips 2005).
    sigma_i2 = s_r2
    sigma_ln = lognormal_sigma_ln(sigma_i2)
    from .budget import beam_radius  # local import to avoid cycle noise

    w_rx = beam_radius(link.wavelength_m, link.beam_waist_radius_m, link.range_m)
    sigma_d = channel.pointing_jitter_rad * link.range_m
    return ChannelModel(
        rytov_variance=s_r2,
        scintillation_index=sigma_i2,
        sigma_ln=sigma_ln,
        sigma_disp_m=sigma_d,
        beam_radius_at_rx_m=w_rx,
        weak_regime_valid=s_r2 < 1.0,
    )


@dataclass(frozen=True)
class MonteCarloResult:
    """Monte Carlo samples of received power.

    samples_dbm : received power samples [dBm], shape (n_samples,).
    budget : deterministic budget used as the mean-power reference.
    model : derived channel statistics.
    n_samples, seed : reproducibility record.
    """

    samples_dbm: np.ndarray
    budget: LinkBudget
    model: ChannelModel
    n_samples: int
    seed: int


def sample_received_power_dbm(
    link: LinkParams,
    channel: ChannelParams,
    n_samples: int = 100_000,
    seed: int = 0,
) -> MonteCarloResult:
    """Vectorised seeded Monte Carlo of instantaneous received power.

    Model: P = P_det_nopoint * X_scint * L_p(r), where
      - P_det_nopoint is the deterministic received power with the static
        pointing term removed (pointing is re-sampled including bias),
      - X_scint = exp(Z), Z ~ N(-sigma_ln^2/2, sigma_ln^2) so E[X] = 1
        (lognormal irradiance, Andrews & Phillips 2005, Ch. 8),
      - L_p = exp(-2 r^2 / w^2) with r^2 = x^2 + y^2,
        x ~ N(bias*L, sigma_d^2), y ~ N(0, sigma_d^2) (bias along x WLOG).

    Assumptions: scintillation and jitter independent; point receiver
    (a << w); weak-fluctuation regime for the lognormal law.
    Units: result samples in dBm.
    """
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}")
    if n_samples > 20_000_000:
        raise ValueError(f"n_samples={n_samples} exceeds the 2e7 memory/compute budget")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    budget = compute_budget(link)
    model = build_channel_model(link, channel)
    rng = np.random.default_rng(seed)

    # Deterministic power with the static pointing loss removed (re-sampled below).
    p0_dbm = budget.received_power_dbm + budget.pointing_loss_db
    p0_w = watts_from_dbm(p0_dbm)

    if model.sigma_ln > 0.0:
        z = rng.normal(-0.5 * model.sigma_ln**2, model.sigma_ln, size=n_samples)
        scint = np.exp(z)
    else:
        scint = np.ones(n_samples)

    bias_m = link.pointing_bias_rad * link.range_m
    if model.sigma_disp_m > 0.0:
        x = rng.normal(bias_m, model.sigma_disp_m, size=n_samples)
        y = rng.normal(0.0, model.sigma_disp_m, size=n_samples)
        r2 = x * x + y * y
    else:
        r2 = np.full(n_samples, bias_m**2)
    l_point = np.exp(-2.0 * r2 / model.beam_radius_at_rx_m**2)

    p_w = p0_w * scint * l_point
    samples_dbm = 10.0 * np.log10(np.maximum(p_w, 1e-300) / 1e-3)
    return MonteCarloResult(
        samples_dbm=samples_dbm,
        budget=budget,
        model=model,
        n_samples=n_samples,
        seed=seed,
    )


__all__ = [
    "ChannelModel",
    "ChannelParams",
    "MonteCarloResult",
    "build_channel_model",
    "dbm_from_watts",
    "lognormal_sigma_ln",
    "mean_pointing_loss_fraction",
    "rytov_variance_plane_wave",
    "sample_received_power_dbm",
]
