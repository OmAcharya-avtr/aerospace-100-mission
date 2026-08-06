"""First-order (linear) uncertainty propagation for the link margin.

Method
------
Given independent input uncertainties ``sigma_i`` (1-sigma) on selected
``LinkBudget`` fields, the margin variance is approximated by the standard
first-order (delta-method) formula

    sigma_margin^2 ~= sum_i (d margin / d x_i)^2 * sigma_i^2

(JCGM 100:2008, "Guide to the Expression of Uncertainty in Measurement",
Sec. 5.1, uncorrelated inputs). Partial derivatives are computed by central
finite differences.

Linearity assumption and when it breaks
---------------------------------------
The formula is exact only if margin is linear in each input over roughly
+/- 3 sigma. It degrades when:

- The margin is strongly curved in an input across the sigma range, e.g.
  pointing loss is QUADRATIC in ``pointing_error_rad``; at zero nominal
  pointing error the first derivative vanishes and linear propagation
  predicts zero contribution even though jitter clearly costs margin.
- A sigma is large relative to the scale over which the derivative changes
  (e.g. sigma_range comparable to the range itself).
- An input sits near a hard physical bound (range -> 0, efficiency -> 1),
  where the Gaussian input model itself is questionable.

For such cases use :func:`monte_carlo_margin` and compare; the validation
suite does exactly this cross-check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from .core import LinkBudget

__all__ = ["MarginUncertainty", "propagate_margin_sigma", "monte_carlo_margin"]

_NUMERIC_FIELDS = {
    "tx_power_dbm",
    "wavelength_nm",
    "beam_divergence_rad",
    "range_km",
    "rx_aperture_diameter_m",
    "rx_sensitivity_dbm",
    "tx_optics_efficiency",
    "rx_optics_efficiency",
    "pointing_error_rad",
    "atmos_attenuation_db_per_km",
}


@dataclass(frozen=True)
class MarginUncertainty:
    """Result of first-order propagation.

    Attributes:
        margin_db: Nominal link margin [dB].
        sigma_margin_db: 1-sigma uncertainty on the margin [dB].
        partials: ``{field: d(margin_db)/d(field)}`` in dB per field unit.
        contributions_db: ``{field: |partial| * sigma}`` — each input's
            1-sigma contribution to the margin [dB].
    """

    margin_db: float
    sigma_margin_db: float
    partials: dict[str, float]
    contributions_db: dict[str, float]


def _check_sigmas(budget: LinkBudget, sigmas: dict[str, float]) -> None:
    valid = {f.name for f in fields(budget)} & _NUMERIC_FIELDS
    for name, sigma in sigmas.items():
        if name not in valid:
            raise ValueError(
                f"Unknown or non-numeric field {name!r}; valid fields: {sorted(valid)}."
            )
        if not (sigma >= 0.0) or not math.isfinite(sigma):
            raise ValueError(f"sigma for {name!r} must be finite and >= 0, got {sigma}.")


def _margin(budget: LinkBudget) -> float:
    return budget.compute().margin_db


def propagate_margin_sigma(
    budget: LinkBudget,
    sigmas: dict[str, float],
    rel_step: float = 1e-4,
) -> MarginUncertainty:
    """Propagate 1-sigma input uncertainties to the link margin (first order).

    Args:
        budget: Nominal link budget (validated).
        sigmas: Map of field name -> 1-sigma uncertainty in that field's own
            units (dBm, rad, km, dB/km, dimensionless efficiency, ...).
        rel_step: Relative central-difference step. The absolute step per
            input is ``h = rel_step * max(|x|, sigma, tiny)`` so zero-valued
            inputs (e.g. pointing_error_rad = 0) still get a finite step.

    Returns:
        MarginUncertainty with sigma, per-input partials and contributions.

    Raises:
        ValueError: Unknown field name, negative sigma, or a perturbed input
            leaving its physical domain (reduce sigma or rel_step).
    """
    _check_sigmas(budget, sigmas)
    m0 = _margin(budget)
    partials: dict[str, float] = {}
    contributions: dict[str, float] = {}
    var = 0.0
    for name, sigma in sigmas.items():
        x0 = float(getattr(budget, name))
        h = rel_step * max(abs(x0), sigma, 1e-12)
        # Central difference; fall back to a one-sided difference when a
        # perturbed point leaves the physical domain (e.g. pointing error 0).
        try:
            m_plus = _margin(budget.replace(**{name: x0 + h}))
        except ValueError:
            m_plus = None
        try:
            m_minus = _margin(budget.replace(**{name: x0 - h}))
        except ValueError:
            m_minus = None
        if m_plus is not None and m_minus is not None:
            d = (m_plus - m_minus) / (2.0 * h)
        elif m_plus is not None:
            d = (m_plus - m0) / h  # forward difference at a lower bound
        elif m_minus is not None:
            d = (m0 - m_minus) / h  # backward difference at an upper bound
        else:
            raise ValueError(
                f"Finite-difference steps on {name!r} left the physical domain in both "
                f"directions (x0={x0}, h={h}). Reduce rel_step or the sigma."
            )
        partials[name] = d
        contributions[name] = abs(d) * sigma
        var += (d * sigma) ** 2
    return MarginUncertainty(
        margin_db=m0,
        sigma_margin_db=math.sqrt(var),
        partials=partials,
        contributions_db=contributions,
    )


def monte_carlo_margin(
    budget: LinkBudget,
    sigmas: dict[str, float],
    n_samples: int = 20_000,
    seed: int = 0,
    max_redraws: int = 100,
) -> np.ndarray:
    """Monte Carlo distribution of the link margin under Gaussian input errors.

    Each selected input is drawn independently from
    ``Normal(nominal, sigma)``. Draws that violate physical bounds (e.g. a
    negative range, |pointing offset| is taken since only the radial
    magnitude matters) are redrawn per-sample up to ``max_redraws`` times
    (truncated-Gaussian behaviour; negligible bias when bounds are many
    sigma away — keep sigmas small relative to the nominal values).

    Args:
        budget: Nominal link budget.
        sigmas: Map of field name -> 1-sigma uncertainty (same units as field).
        n_samples: Number of Monte Carlo samples (runtime is O(n)).
        seed: RNG seed (numpy default_rng) for exact reproducibility.
        max_redraws: Redraw attempts per sample before raising.

    Returns:
        numpy array of shape ``(n_samples,)`` of margin_db values.

    Raises:
        ValueError: Bad sigma spec, or a sample could not be made physical
            within ``max_redraws`` redraws (sigmas too large).
    """
    _check_sigmas(budget, sigmas)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}.")
    rng = np.random.default_rng(seed)
    names = sorted(sigmas)
    out = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        for attempt in range(max_redraws + 1):
            changes = {}
            for name in names:
                val = rng.normal(float(getattr(budget, name)), sigmas[name])
                if name == "pointing_error_rad":
                    val = abs(val)  # radial magnitude
                changes[name] = val
            try:
                out[i] = _margin(budget.replace(**changes))
                break
            except ValueError:
                if attempt == max_redraws:
                    raise ValueError(
                        f"Could not draw a physical sample after {max_redraws} redraws; "
                        f"sigmas {sigmas} are too large relative to nominal values."
                    ) from None
    return out
