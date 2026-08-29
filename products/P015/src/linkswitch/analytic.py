"""Closed-form (semi-analytic) model of the fixed-threshold policy's expected
throughput, used to derive the optimal fixed switching threshold directly
from the channel statistics, independent of the Monte Carlo simulator.

Setup
-----
The optical channel's standardised log-irradiance ``Z_t = (ln I_t - mu_z) /
sigma_z`` is a stationary AR(1) Gaussian process with lag-1 correlation
``rho`` (see ``optical.py``). A fixed-threshold policy at threshold ``tau``
selects optical iff ``I_t >= tau``, i.e. iff ``Z_t >= z_th`` where
``z_th = (ln tau - mu_z) / sigma_z``.

Throughput term
----------------
Physical optical outage occurs at ``Z_t < z_phys`` (the link-margin
threshold). If the policy threshold is above the physical one
(``z_th >= z_phys``) the policy never selects optical while it is
physically down, so the fraction of time optical is *usefully* selected is
``P(Z_t >= z_th) = 1 - Phi(z_th)``. If the policy threshold is below the
physical one (``z_th < z_phys``), the policy still selects optical whenever
``Z_t >= z_th``, but delivers nothing while ``z_th <= Z_t < z_phys`` (a
physical outage the policy failed to react to), so the useful fraction is
capped at ``1 - Phi(z_phys)`` regardless of how far below z_phys the
threshold sits. Both cases combine into

    g_opt(z_th) = 1 - Phi(max(z_th, z_phys))

The RF contribution is ``R_rf * p_rf_avail * Phi(z_th)`` (fraction of time
RF is selected, times the probability RF itself is not in a rain fade at
that moment; optical scintillation and RF rain fades are independent
physical phenomena in this simulator, so this product form is exact under
that independence assumption).

Switch-cost term
------------------
Every threshold up-crossing or down-crossing of a stationary Gauss-Markov
process triggers one switch. For a bivariate normal pair
``(Z_t, Z_{t+1})`` with correlation ``rho``, the per-step *down*-crossing
probability is ``P(Z_t >= z_th, Z_{t+1} < z_th)``, and by the time-reversal
symmetry of a stationary AR(1) Gaussian process the up-crossing probability
equals it, giving a total per-step crossing (switch) probability

    N(z_th; rho) = 2 * [Phi(z_th) - Phi_2(z_th, z_th; rho)]

where ``Phi_2`` is the standard bivariate normal CDF. This is a standard
level-crossing-style result for a discrete-time stationary Gaussian
sequence (cf. S. O. Rice, "Mathematical Analysis of Random Noise," Bell
Syst. Tech. J. 23-24, 1944-45, for the continuous-time analogue; H. Cramer
and M. R. Leadbetter, *Stationary and Related Stochastic Processes*, Wiley,
1967, Ch. 10, for crossing theory of stationary Gaussian processes) applied
here directly to the discrete AR(1) chain via the exact bivariate normal
joint law — no continuum/derivative approximation is needed because the
process is already discrete-time.

Each switch costs ``downtime_steps`` steps of zero throughput. This
package approximates the *expected* rate lost per switch as
``downtime_steps * R_opt`` (a deliberately simple, slightly conservative
proxy — the exact expected loss depends on which direction the switch goes
and what the destination channel would have delivered; the Monte Carlo
simulator in ``simulate.py`` computes the exact cost and is the ground
truth that this closed form is checked against in ``validation/``).

Objective
------------
    J(z_th) = R_opt * g_opt(z_th) + R_rf * p_rf_avail * Phi(z_th)
              - downtime_steps * R_opt * N(z_th; rho)

The optimal threshold ``z_th*`` maximises ``J``. This module finds it by
bounded scalar minimisation of ``-J`` (``optimal_threshold_analytic``) and,
independently, by a fine grid search over the same closed form
(``optimal_threshold_grid``) as an internal sanity check that the optimizer
converged to the true optimum of ``J`` rather than a numerical artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.stats import multivariate_normal, norm

from .optical import OpticalParams
from .rf import RFParams, rf_path_attenuation_db

__all__ = [
    "bivariate_normal_cdf",
    "crossing_probability",
    "z_to_irradiance",
    "irradiance_to_z",
    "p_rf_available_estimate",
    "expected_throughput_analytic",
    "OptimalThresholdResult",
    "optimal_threshold_analytic",
    "optimal_threshold_grid",
]


def bivariate_normal_cdf(a: float, b: float, rho: float) -> float:
    """P(X <= a, Y <= b) for standard bivariate normal (X, Y) with corr rho."""
    if not (-1.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [-1, 1], got {rho!r}")
    cov = [[1.0, rho], [rho, 1.0]]
    return float(multivariate_normal(mean=[0.0, 0.0], cov=cov).cdf([a, b]))


def crossing_probability(z_th: float, rho: float) -> float:
    """Per-step level-crossing (switch) probability at threshold z_th; see module docstring."""
    phi = float(norm.cdf(z_th))
    phi2 = bivariate_normal_cdf(z_th, z_th, rho)
    return max(0.0, 2.0 * (phi - phi2))


def z_to_irradiance(z: float, mu_z: float, sigma_z: float) -> float:
    return math.exp(mu_z + sigma_z * z)


def irradiance_to_z(tau: float, mu_z: float, sigma_z: float) -> float:
    tau = float(tau)
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError(f"tau must be finite and > 0, got {tau!r}")
    return (math.log(tau) - mu_z) / sigma_z


def p_rf_available_estimate(rf: RFParams, n_mc: int = 20_000, seed: int = 12345) -> float:
    """Steady-state RF availability probability, via a fast deterministic
    Monte Carlo draw straight from the RF rain/attenuation model (not the
    Markov *sequence* — just its stationary rain probability), independent
    of any telemetry episode.

    P(available) = (1 - p_rain) + p_rain * P(A(R) too small to break margin | raining)

    The second term is estimated by drawing ``n_mc`` i.i.d. rain rates from
    the model's own conditional lognormal distribution (this uses the exact
    same functions as telemetry generation, so it is internally consistent
    with the simulator, not a separate approximation of it).
    """
    if not isinstance(n_mc, (int, np.integer)) or isinstance(n_mc, bool) or n_mc < 1:
        raise ValueError(f"n_mc must be a positive integer, got {n_mc!r}")
    rng = np.random.default_rng(int(seed))
    rain_rate = rng.lognormal(mean=math.log(rf.r_med_mm_hr), sigma=rf.rate_sigma, size=n_mc)
    atten = rf_path_attenuation_db(rain_rate, rf.k, rf.alpha, rf.path_length_km,
                                    rf.reduction_length_km)
    snr = rf.snr_clear_db - atten
    p_avail_given_rain = float(np.mean(snr >= rf.snr_min_db))
    return (1.0 - rf.p_rain) * 1.0 + rf.p_rain * p_avail_given_rain


def expected_throughput_analytic(
    z_th: float,
    z_phys: float,
    rho: float,
    r_opt: float,
    r_rf_eff: float,
    downtime_steps: float,
) -> float:
    """J(z_th): closed-form expected per-step throughput; see module docstring."""
    g_opt = 1.0 - float(norm.cdf(max(z_th, z_phys)))
    g_rf = float(norm.cdf(z_th))
    switch_term = downtime_steps * r_opt * crossing_probability(z_th, rho)
    return r_opt * g_opt + r_rf_eff * g_rf - switch_term


@dataclass(frozen=True)
class OptimalThresholdResult:
    z_th: float
    tau: float
    objective: float
    z_phys: float
    rho: float


def optimal_threshold_analytic(
    optical: OpticalParams,
    rf: RFParams,
    downtime_steps: float,
    z_bounds: tuple[float, float] = (-8.0, 8.0),
) -> OptimalThresholdResult:
    """Find z_th* = argmax J(z_th) by bounded scalar minimisation of -J."""
    sigma_z = optical.sigma_z
    mu_z = -0.5 * sigma_z * sigma_z
    z_phys = irradiance_to_z(optical.tau_phys, mu_z, sigma_z)
    rho = math.exp(-1.0 / optical.coherence_steps)
    r_rf_eff = rf.rate_mbps * p_rf_available_estimate(rf)

    def neg_j(z: float) -> float:
        return -expected_throughput_analytic(z, z_phys, rho, optical.rate_mbps, r_rf_eff,
                                              downtime_steps)

    res = optimize.minimize_scalar(neg_j, bounds=z_bounds, method="bounded",
                                    options={"xatol": 1e-8})
    z_star = float(res.x)
    return OptimalThresholdResult(
        z_th=z_star,
        tau=z_to_irradiance(z_star, mu_z, sigma_z),
        objective=-float(res.fun),
        z_phys=z_phys,
        rho=rho,
    )


def optimal_threshold_grid(
    optical: OpticalParams,
    rf: RFParams,
    downtime_steps: float,
    n_points: int = 4001,
    z_bounds: tuple[float, float] = (-8.0, 8.0),
) -> OptimalThresholdResult:
    """Independent grid-search cross-check of the same closed form J(z_th)."""
    sigma_z = optical.sigma_z
    mu_z = -0.5 * sigma_z * sigma_z
    z_phys = irradiance_to_z(optical.tau_phys, mu_z, sigma_z)
    rho = math.exp(-1.0 / optical.coherence_steps)
    r_rf_eff = rf.rate_mbps * p_rf_available_estimate(rf)

    zs = np.linspace(z_bounds[0], z_bounds[1], n_points)
    js = np.array([
        expected_throughput_analytic(z, z_phys, rho, optical.rate_mbps, r_rf_eff, downtime_steps)
        for z in zs
    ])
    i_best = int(np.argmax(js))
    z_star = float(zs[i_best])
    return OptimalThresholdResult(
        z_th=z_star,
        tau=z_to_irradiance(z_star, mu_z, sigma_z),
        objective=float(js[i_best]),
        z_phys=z_phys,
        rho=rho,
    )
