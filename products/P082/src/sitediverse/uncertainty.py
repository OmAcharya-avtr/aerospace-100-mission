"""Climatology sampling uncertainty, propagated and reported beside every figure.

The specification requires that the record's own sampling uncertainty is
propagated and reported beside every availability figure. This module does that,
and it defines the "climatology-uncertainty share" precisely, because an
undefined share is worse than none.

**(27) Propagating the marginal uncertainty into the naive product.** The naive
product of equation (14) is a product of estimated rates. With independent
estimation errors of standard deviation ``s_i`` on ``qbar_i``, first-order
propagation in logarithms gives

    sd( P_naive ) / P_naive  =  sqrt( sum_i ( s_i / qbar_i )^2 )

the relative standard deviation adding in quadrature. The ``s_i`` come from
equation (13), so they already carry the serial-dependence inflation. Two
approximations are declared: the propagation is first order, and the ``s_i`` are
treated as independent across stations, which they are NOT -- the stations'
cloud series are correlated, which is the whole subject. The error this makes is
bounded in one direction and MEASURED in the other: the correlated case has a
LARGER variance, so the figure below is a lower bound on the naive product's
sampling uncertainty, and ``validation/validate_uncertainty.py`` compares it
against a block bootstrap of the same product, which makes no independence
assumption. The bootstrap figure is the one the reports quote; the analytic
figure is quoted beside it so that the size of the independence approximation is
visible.

**(28) The climatology-uncertainty share.** For a model estimate ``P_model`` and
the empirical reference ``P_hat`` with bootstrap standard error ``s_boot``, the
share of the discrepancy attributable to the record's finite length is defined
here as

    share  =  s_boot^2 / ( s_boot^2 + ( P_model - P_hat )^2 )

A share near one says the model and the record disagree by no more than the
record's own sampling noise, so the disagreement is not evidence of anything. A
share near zero says the disagreement is structural. The definition is a
variance decomposition with the squared discrepancy standing in for the
structural variance, and it has three properties worth stating because they are
what make it usable and what limit it:

1. It is exactly ``1.0`` for the empirical reference against itself, by
   construction, since the discrepancy is zero. That is not a claim that the
   reference is certain; it is a statement that the reference's only error is
   sampling error.
2. It is NOT a p-value and must not be read as one. A share of 0.5 corresponds
   to a discrepancy of exactly one bootstrap standard error, which for a
   two-sided 95 per cent test is nowhere near significant.
3. It uses the squared discrepancy, which is a biased estimate of the squared
   structural error: ``E[(P_model - P_hat)^2] = bias^2 + s_boot^2``, so the
   share is biased TOWARD zero -- it understates how much of the discrepancy is
   sampling. The bias-corrected form subtracts ``s_boot^2`` from the numerator
   of the discrepancy, and ``climatology_share_corrected`` below computes it;
   both are reported, and the corrected form can exceed one, which is itself
   informative and is not clipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def naive_product_relative_sd(
    rates: np.ndarray, standard_errors: np.ndarray
) -> float:
    """Equation (27): relative standard deviation of the naive product.

    Parameters
    ----------
    rates:
        ``(n,)`` estimated marginal rates, strictly positive.
    standard_errors:
        ``(n,)`` standard errors of those rates, non-negative.

    Returns
    -------
    float
        Relative standard deviation, dimensionless.
    """
    values = np.asarray(rates, dtype=float)
    errors = np.asarray(standard_errors, dtype=float)
    if values.shape != errors.shape:
        raise ValueError(
            f"rates and standard_errors must have the same shape, got "
            f"{values.shape} and {errors.shape}"
        )
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"inputs must be non-empty 1-D, got shape {values.shape}")
    if np.any(values <= 0.0):
        raise ValueError(
            "every rate must be strictly positive to propagate in logarithms; "
            f"got a minimum of {float(np.min(values)):.6e}"
        )
    if np.any(errors < 0.0):
        raise ValueError("standard errors must be non-negative")
    return float(np.sqrt(np.sum((errors / values) ** 2)))


def climatology_share(discrepancy: float, bootstrap_sd: float) -> float:
    """Equation (28): the share of a discrepancy explained by sampling noise.

    Parameters
    ----------
    discrepancy:
        ``P_model - P_hat``. Its sign does not matter.
    bootstrap_sd:
        Bootstrap standard error of the empirical reference, non-negative.

    Returns
    -------
    float
        In ``[0, 1]``. ``1.0`` when the discrepancy is zero, ``0.0`` when the
        bootstrap standard error is zero and the discrepancy is not, and
        ``nan`` when both are zero, which carries no information at all.
    """
    gap = float(discrepancy)
    noise = float(bootstrap_sd)
    if noise < 0.0:
        raise ValueError(f"bootstrap_sd must be non-negative, got {noise!r}")
    denominator = noise**2 + gap**2
    if denominator == 0.0:
        return float("nan")
    return noise**2 / denominator


def climatology_share_corrected(discrepancy: float, bootstrap_sd: float) -> float:
    """The bias-corrected form of equation (28). May exceed 1; not clipped.

    ``E[(P_model - P_hat)^2] = bias^2 + s_boot^2``, so subtracting ``s_boot^2``
    from the squared discrepancy gives an unbiased estimate of the squared
    structural error. Where that estimate is negative -- which happens whenever
    the observed discrepancy is smaller than the sampling noise, and is the
    expected case for a model that is right -- the share exceeds one. That is
    reported rather than clipped, because a clipped 1.0 is indistinguishable
    from a discrepancy of exactly zero and the two mean different things.
    """
    gap = float(discrepancy)
    noise = float(bootstrap_sd)
    if noise < 0.0:
        raise ValueError(f"bootstrap_sd must be non-negative, got {noise!r}")
    structural = gap**2 - noise**2
    denominator = noise**2 + structural
    if denominator == 0.0:
        return float("nan")
    return noise**2 / denominator


@dataclass(frozen=True)
class FigureWithUncertainty:
    """One availability figure with its climatology sampling uncertainty.

    Every availability figure this product reports is carried in this form, so
    that no number can be quoted without the record's sampling uncertainty
    beside it.

    Attributes
    ----------
    label:
        What the figure is.
    unavailability:
        The joint unavailability, primary.
    sampling_sd:
        Standard error from the record's own sampling uncertainty.
    source:
        How ``sampling_sd`` was obtained: ``"block-bootstrap"``,
        ``"analytic-propagation"``, ``"monte-carlo"`` or ``"none"``. A figure
        whose uncertainty was not computed says ``"none"`` and carries
        ``nan``, rather than carrying a zero that reads as certainty.
    """

    label: str
    unavailability: float
    sampling_sd: float
    source: str

    @property
    def availability(self) -> float:
        """``1 - unavailability``. Derived, never primary; see equation (14)."""
        return 1.0 - self.unavailability

    @property
    def relative_sd(self) -> float:
        """``sampling_sd / unavailability``, or ``inf`` at an unavailability of 0."""
        if self.unavailability == 0.0:
            return float("inf")
        return self.sampling_sd / self.unavailability

    def render(self) -> str:
        """``value +/- sd`` to six decimals, or ``value (sd not computed)``."""
        if not np.isfinite(self.sampling_sd):
            return f"{self.unavailability:.6f} (sampling sd not computed)"
        return f"{self.unavailability:.6f} +/- {self.sampling_sd:.6f}"
