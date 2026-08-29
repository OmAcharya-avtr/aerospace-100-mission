"""Closed-form results for the fixed-threshold switching policy.

Setting
-------
The optical margin ``m(t)`` [dB] is stationary Gaussian with mean ``mu`` and
standard deviation ``sigma``, and lag-1 correlation ``rho`` (this is exactly the
first-order Gauss-Markov lognormal-irradiance model of
:mod:`linkswitch.channels`: ``10 log10 I`` is Gaussian because ``ln I`` is).
The optical link carries ``R_o`` [bit/s] when ``m >= b`` (``b = 0`` dB by
definition of the margin) and nothing otherwise; the RF link carries ``R_r``
[bit/s] whenever it is up, which happens independently with probability
``p_rf``.

A causal policy sees ``m(t-1)`` and must pick the channel used at step ``t``.

Optimal myopic threshold
------------------------
Conditional on ``m(t-1) = v``, ``m(t) ~ N(mu + rho (v - mu), sigma^2 (1 - rho^2))``
(standard bivariate-normal conditioning; e.g. Papoulis & Pillai, *Probability,
Random Variables and Stochastic Processes*, 4th ed., McGraw-Hill, 2002, Ch. 6).
The expected step reward is

    optical:  R_o * P(m(t) >= b | v) = R_o * Phi( (mu + rho (v - mu) - b) / sigma_c )
    RF:       R_r * p_rf

with ``sigma_c = sigma sqrt(1 - rho^2)``. The optical reward is strictly
increasing in ``v`` for ``rho > 0``, so the optimal rule is a threshold rule and
the threshold ``T*`` is where the two rewards are equal:

    Phi((mu + rho (T* - mu) - b) / sigma_c) = q,   q = R_r p_rf / R_o
    T* = mu + (b + sigma_c * z_q - mu) / rho,      z_q = Phi^{-1}(q)

Validity and assumptions
------------------------
* Exact only for a **zero switching penalty**. With a non-zero handover guard
  the problem becomes a partially observed Markov decision process and the
  myopic threshold is no longer optimal; :func:`optimal_fixed_threshold_db` is
  then a reference point, not the optimum (quantified in
  ``validation/VALIDATION.md``).
* Exact only for **noiseless telemetry**. With measurement noise the sufficient
  statistic is the posterior mean of ``m(t-1)``, not the raw sample.
* Assumes a stationary channel (``turbulence_drift_sigma = 0``) and independence
  between the optical and RF channels.
* ``q`` must lie in ``(0, 1)``: if ``R_r p_rf >= R_o`` the RF channel is never
  worse and the optimal policy is "always RF" (``T* = +inf``).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import multivariate_normal, norm

__all__ = [
    "expected_throughput_fixed_threshold",
    "optical_outage_probability",
    "optimal_fixed_threshold_db",
]


def _check_stats(mean_db: float, std_db: float, rho: float) -> tuple[float, float, float]:
    """Validate ``(mu, sigma, rho)`` for the Gaussian AR(1) margin model."""
    mu = float(mean_db)
    sd = float(std_db)
    r = float(rho)
    if not math.isfinite(mu):
        raise ValueError(f"mean_margin_db must be finite, got {mu}")
    if not math.isfinite(sd) or sd <= 0.0:
        raise ValueError(f"std_margin_db must be finite and > 0, got {sd}")
    if not (0.0 < r < 1.0):
        raise ValueError(f"rho must satisfy 0 < rho < 1, got {r}")
    return mu, sd, r


def optical_outage_probability(
    mean_margin_db: float, std_margin_db: float, outage_margin_db: float = 0.0
) -> float:
    """Stationary probability that the optical margin is below threshold [-].

    ``P = Phi((b - mu) / sigma)``. Units: all arguments in dB.
    Source: Gaussian margin implied by lognormal irradiance
    (Andrews & Phillips 2005, Ch. 8); identical in form to the lognormal
    probability-of-fade expression used in P001 BeamTwin.
    """
    mu = float(mean_margin_db)
    sd = float(std_margin_db)
    if not math.isfinite(mu):
        raise ValueError(f"mean_margin_db must be finite, got {mu}")
    if not math.isfinite(sd) or sd <= 0.0:
        raise ValueError(f"std_margin_db must be finite and > 0, got {sd}")
    return float(norm.cdf((float(outage_margin_db) - mu) / sd))


def optimal_fixed_threshold_db(
    mean_margin_db: float,
    std_margin_db: float,
    rho: float,
    rate_optical_bps: float,
    rate_rf_bps: float,
    p_rf_up: float = 1.0,
    outage_margin_db: float = 0.0,
) -> float:
    """Myopically optimal fixed switching threshold ``T*`` [dB].

    ``T* = mu + (b + sigma sqrt(1-rho^2) Phi^{-1}(q) - mu) / rho`` with
    ``q = R_r p_rf / R_o``. Returns ``+inf`` when ``q >= 1`` (always use RF)
    and ``-inf`` when ``q <= 0`` (always use optical).

    Units: dB in, dB out; rates in bit/s; ``rho`` and ``p_rf_up`` dimensionless.
    Validity: see the module docstring -- zero switching penalty, noiseless
    telemetry, stationary channel, independent RF channel.
    """
    mu, sd, r = _check_stats(mean_margin_db, std_margin_db, rho)
    if not (np.isfinite(rate_optical_bps) and rate_optical_bps > 0):
        raise ValueError(f"rate_optical_bps must be > 0, got {rate_optical_bps}")
    if not (np.isfinite(rate_rf_bps) and rate_rf_bps > 0):
        raise ValueError(f"rate_rf_bps must be > 0, got {rate_rf_bps}")
    if not (0.0 <= p_rf_up <= 1.0):
        raise ValueError(f"p_rf_up must be in [0, 1], got {p_rf_up}")
    q = float(rate_rf_bps) * float(p_rf_up) / float(rate_optical_bps)
    if q >= 1.0:
        return math.inf
    if q <= 0.0:
        return -math.inf
    sigma_c = sd * math.sqrt(1.0 - r * r)
    z_q = float(norm.ppf(q))
    return float(mu + (float(outage_margin_db) + sigma_c * z_q - mu) / r)


def expected_throughput_fixed_threshold(
    threshold_db: float,
    mean_margin_db: float,
    std_margin_db: float,
    rho: float,
    rate_optical_bps: float,
    rate_rf_bps: float,
    p_rf_up: float = 1.0,
    outage_margin_db: float = 0.0,
) -> float:
    """Steady-state expected delivered throughput of a fixed-threshold policy [bit/s].

    ``E[R] = R_o * P(m(t-1) >= T, m(t) >= b) + R_r p_rf * P(m(t-1) < T)``.
    The joint term is a bivariate-normal orthant probability with correlation
    ``rho``, evaluated with :func:`scipy.stats.multivariate_normal.cdf`.

    Assumes zero switching penalty (so steps are scored independently) and an
    RF channel independent of the optical one. Units: dB in, bit/s out.
    """
    mu, sd, r = _check_stats(mean_margin_db, std_margin_db, rho)
    if not (0.0 <= p_rf_up <= 1.0):
        raise ValueError(f"p_rf_up must be in [0, 1], got {p_rf_up}")
    t = float(threshold_db)
    b = float(outage_margin_db)
    if math.isinf(t):
        # T = +inf: never optical. T = -inf: always optical.
        if t > 0:
            return float(rate_rf_bps * p_rf_up)
        return float(rate_optical_bps * (1.0 - norm.cdf((b - mu) / sd)))
    a1 = (t - mu) / sd
    a2 = (b - mu) / sd
    cov = np.array([[1.0, r], [r, 1.0]])
    phi2 = float(multivariate_normal(mean=[0.0, 0.0], cov=cov).cdf([a1, a2]))
    p_joint = 1.0 - float(norm.cdf(a1)) - float(norm.cdf(a2)) + phi2
    p_below = float(norm.cdf(a1))
    return float(rate_optical_bps * p_joint + rate_rf_bps * p_rf_up * p_below)
