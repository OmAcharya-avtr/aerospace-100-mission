"""Jitter-to-pointing-loss conversion for a Gaussian beam.

Theory
------
Far-field intensity of a fundamental Gaussian beam versus off-axis
angle ``theta``::

    I(theta) = I0 * exp(-2 * theta^2 / theta_div^2)

where ``theta_div`` is the 1/e^2 *half-angle* divergence [rad]
(Siegman 1986, "Lasers", ch. 17). The instantaneous normalized
pointing loss for a boresight error ``theta`` is therefore
``L(theta) = exp(-2 theta^2 / theta_div^2)`` (point-receiver /
far-field approximation: receiver aperture much smaller than the beam
footprint).

For zero-mean Gaussian jitter with per-axis standard deviation
``sigma_theta`` on two independent axes, the radial error is
Rayleigh-distributed, ``p(theta) = (theta / sigma^2)
exp(-theta^2 / (2 sigma^2))``, and the *average* power loss has the
closed form (derivation: Gaussian integral; equivalent to the
``gamma^2 / (gamma^2 + 1)`` result of Farid & Hranilovic 2007,
J. Lightwave Technol. 25(7):1702-1710, in the point-receiver limit;
see also Andrews & Phillips 2005, "Laser Beam Propagation through
Random Media", 2nd ed., ch. 12 on pointing-error statistics)::

    <L> = integral_0^inf exp(-2 theta^2 / theta_div^2) p(theta) dtheta
        = 1 / (1 + 4 sigma_theta^2 / theta_div^2)

Assumptions and validity range:
- Gaussian TEM00 far field; no truncation, obscuration, or turbulence.
- Independent, zero-mean, equal-variance Gaussian jitter on both axes
  (no static bias); for biased or anisotropic jitter the closed form
  does not apply.
- Point receiver in the far field.

The formula is verified against Monte Carlo in
``validation/val_pointing.py`` and ``tests/test_pointing.py``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["pointing_loss_avg", "pointing_loss_avg_mc"]


def pointing_loss_avg(sigma_theta: float, theta_div: float) -> float:
    """Average Gaussian-beam pointing loss under radial Gaussian jitter.

    ``<L> = 1 / (1 + 4 (sigma_theta / theta_div)^2)`` — see module
    docstring for derivation, sources, and assumptions.

    Parameters
    ----------
    sigma_theta : float
        Per-axis RMS pointing jitter [rad], >= 0 (zero-mean Gaussian,
        identical on both axes).
    theta_div : float
        Beam 1/e^2 half-angle divergence [rad], > 0.

    Returns
    -------
    float
        Mean normalized received power ``<L>`` in (0, 1]; 1 means no
        loss. Convert to dB via ``-10 * log10(<L>)``.

    Raises
    ------
    ValueError
        If ``sigma_theta < 0`` or ``theta_div <= 0`` or inputs are
        non-finite.
    """
    if not (math.isfinite(sigma_theta) and math.isfinite(theta_div)):
        raise ValueError("sigma_theta and theta_div must be finite")
    if sigma_theta < 0:
        raise ValueError(f"sigma_theta must be >= 0 rad, got {sigma_theta}")
    if theta_div <= 0:
        raise ValueError(f"theta_div must be > 0 rad, got {theta_div}")
    r = sigma_theta / theta_div
    return 1.0 / (1.0 + 4.0 * r * r)


def pointing_loss_avg_mc(
    sigma_theta: float,
    theta_div: float,
    n_samples: int = 200_000,
    seed: int = 0,
) -> float:
    """Monte Carlo estimate of the average pointing loss (cross-check).

    Draws ``n_samples`` independent (theta_x, theta_y) ~ N(0, sigma^2)
    pairs and averages ``exp(-2 (theta_x^2 + theta_y^2) / theta_div^2)``.
    Used by the validation suite to verify :func:`pointing_loss_avg`;
    statistical error scales as ``1/sqrt(n_samples)``.

    Parameters
    ----------
    sigma_theta : float
        Per-axis RMS jitter [rad], >= 0.
    theta_div : float
        1/e^2 half-angle divergence [rad], > 0.
    n_samples : int
        Number of Monte Carlo draws, >= 1000.
    seed : int
        RNG seed (numpy default_rng) for reproducibility.

    Returns
    -------
    float
        Monte Carlo mean of the normalized received power.
    """
    if sigma_theta < 0 or theta_div <= 0:
        raise ValueError("require sigma_theta >= 0 and theta_div > 0")
    if n_samples < 1000:
        raise ValueError(f"n_samples must be >= 1000, got {n_samples}")
    rng = np.random.default_rng(seed)
    tx = rng.normal(0.0, sigma_theta, n_samples)
    ty = rng.normal(0.0, sigma_theta, n_samples)
    return float(np.mean(np.exp(-2.0 * (tx**2 + ty**2) / theta_div**2)))
