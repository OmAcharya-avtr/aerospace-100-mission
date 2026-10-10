"""Murphy decomposition of the Brier score, and its exact binned extension.

The classical decomposition (Murphy 1973) applies to a forecast that takes
finitely many distinct values. Let the distinct values be indexed by ``k``,
with ``n_k`` cases, mean forecast ``f_k`` (which equals the forecast value
itself) and observed relative frequency ``o_k``, and let ``obar`` be the
overall base rate. Then, exactly,

    BS = REL - RES + UNC
    REL = sum_k (n_k / n) (f_k - o_k)**2        reliability, lower is better
    RES = sum_k (n_k / n) (o_k - obar)**2       resolution, higher is better
    UNC = obar * (1 - obar)                     uncertainty, a property of the
                                                outcomes alone

Applying the same three formulas to *binned* continuous forecasts does not
give an identity, because the forecast is no longer constant inside a bin.
Writing ``f_i = f_k + d_i`` with ``sum_{i in k} d_i = 0`` and using
``o_i in {0, 1}`` so that ``mean_{i in k}(f_k - o_i)**2 = (f_k - o_k)**2 +
o_k (1 - o_k)``, together with the law of total variance

    sum_k (n_k / n) o_k (1 - o_k) = UNC - RES,

gives the exact five-term identity this module implements:

    BS = REL - RES + UNC + WBV - 2 * WBC
    WBV = (1 / n) sum_i (f_i - f_k(i))**2             within-bin variance
    WBC = (1 / n) sum_i (f_i - f_k(i)) o_i            within-bin covariance
        = sum_k (n_k / n) cov_k(f, o)

``WBV`` and ``WBC`` are exactly zero when the forecast is constant within
every occupied bin, which is when the three-term identity holds. They are the
part of the score that a binned decomposition silently drops, and they are
reported rather than discarded. The derivation above is elementary algebra and
is checked to machine precision in ``tests/test_properties.py`` and in
``validation/validate_decomposition_identity.py``.

References
----------
Murphy, A. H. (1973). "A new vector partition of the probability score."
*Journal of Applied Meteorology* 12(4), 595-600.
doi:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2

Murphy, A. H. (1986). "A new decomposition of the Brier score: formulation
and interpretation." *Monthly Weather Review* 114(12), 2671-2673.
doi:10.1175/1520-0493(1986)114<2671:ANDOTB>2.0.CO;2

Hersbach, H. (2000). "Decomposition of the continuous ranked probability
score for ensemble prediction systems." *Weather and Forecasting* 15(5),
559-570. doi:10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2
(the same partition in the continuous case)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from .binning import assign_bins
from .scores import brier_score, check_forecasts

__all__ = [
    "BrierDecomposition",
    "binned_decomposition",
    "murphy_decomposition",
]


@dataclass(frozen=True)
class BrierDecomposition:
    """A decomposition of the Brier score. All quantities are dimensionless.

    Attributes
    ----------
    brier
        The Brier score of the forecast, computed directly from the samples.
    reliability, resolution, uncertainty
        The three Murphy terms, ``REL``, ``RES`` and ``UNC``.
    within_bin_variance, within_bin_covariance
        ``WBV`` and ``WBC``. Exactly 0.0 for ``murphy_decomposition``.
    base_rate
        Sample base rate ``obar``, i.e. ``outcomes.mean()``.
    n_samples
        Number of forecast-outcome pairs.
    strategy
        ``"exact"`` for the distinct-value grouping, otherwise the binning
        strategy name.
    n_bins_requested, n_bins_occupied
        Requested bin count (0 for ``"exact"``) and the number of bins that
        received at least one sample.
    bin_counts, bin_mean_forecast, bin_observed_frequency
        Per-occupied-bin arrays, in increasing order of forecast value.
    bin_edges
        Bin edges for a binned decomposition, or the distinct forecast values
        for the exact one.
    """

    brier: float
    base_rate: float
    reliability: float
    resolution: float
    uncertainty: float
    within_bin_variance: float
    within_bin_covariance: float
    n_samples: int
    strategy: str
    n_bins_requested: int
    n_bins_occupied: int
    bin_counts: np.ndarray = field(repr=False)
    bin_mean_forecast: np.ndarray = field(repr=False)
    bin_observed_frequency: np.ndarray = field(repr=False)
    bin_edges: np.ndarray = field(repr=False)

    @property
    def three_term_sum(self) -> float:
        """``REL - RES + UNC``: the quantity usually printed as "the" decomposition."""
        return float(self.reliability - self.resolution + self.uncertainty)

    @property
    def five_term_sum(self) -> float:
        """``REL - RES + UNC + WBV - 2 WBC``: equals ``brier`` to machine precision."""
        return float(
            self.three_term_sum + self.within_bin_variance - 2.0 * self.within_bin_covariance
        )

    @property
    def identity_residual(self) -> float:
        """``brier - five_term_sum``. Zero to machine precision, always."""
        return float(self.brier - self.five_term_sum)

    @property
    def three_term_residual(self) -> float:
        """``brier - three_term_sum``: the part a three-term report drops.

        Zero for ``murphy_decomposition``; nonzero in general for
        ``binned_decomposition``, with sign and size set by ``WBV - 2 WBC``.
        """
        return float(self.brier - self.three_term_sum)

    @property
    def brier_skill_vs_climatology(self) -> float:
        """``(RES - REL) / UNC``: skill against the constant base-rate forecast.

        Identical to ``1 - BS / UNC`` only when the three-term identity is
        exact; this property uses the three Murphy terms, so for a binned
        decomposition it differs from the directly computed skill score by
        ``three_term_residual / UNC``.
        """
        if self.uncertainty == 0.0:
            raise ValueError(
                "uncertainty is exactly 0 (all outcomes identical), so skill against "
                "climatology is undefined on this sample"
            )
        return float((self.resolution - self.reliability) / self.uncertainty)

    def report(self) -> str:
        """Multi-line human-readable summary. No units: every term is dimensionless."""
        lines = [
            f"Brier decomposition ({self.strategy}, n = {self.n_samples})",
            f"  base rate                 : {self.base_rate:.12f}",
            f"  bins requested / occupied : {self.n_bins_requested} / {self.n_bins_occupied}",
            f"  BS  (direct)              : {self.brier:.12f}",
            f"  REL reliability           : {self.reliability:.12f}",
            f"  RES resolution            : {self.resolution:.12f}",
            f"  UNC uncertainty           : {self.uncertainty:.12f}",
            f"  WBV within-bin variance   : {self.within_bin_variance:.12f}",
            f"  WBC within-bin covariance : {self.within_bin_covariance:.12f}",
            f"  REL - RES + UNC           : {self.three_term_sum:.12f}",
            f"  + WBV - 2 WBC             : {self.five_term_sum:.12f}",
            f"  identity residual         : {self.identity_residual:+.3e}",
            f"  three-term residual       : {self.three_term_residual:+.3e}",
        ]
        return "\n".join(lines)


def _decompose(
    f: np.ndarray,
    o: np.ndarray,
    labels: np.ndarray,
    n_bins_requested: int,
    strategy: str,
    bin_edges: np.ndarray,
    group_values: np.ndarray | None = None,
) -> BrierDecomposition:
    """Shared core. ``labels`` holds a bin index per sample; empty bins are dropped.

    ``group_values``, when given, is the exact forecast value of each group,
    used instead of the group mean. The exact decomposition passes it so that
    ``WBV`` and ``WBC`` are identically 0.0 rather than the 1e-32 and 1e-17
    that a floating-point mean of identical values leaves behind.
    """
    n = f.size
    obar = float(o.mean())
    uncertainty = obar * (1.0 - obar)

    n_slots = int(labels.max()) + 1 if labels.size else 1
    raw_counts = np.bincount(labels, minlength=n_slots).astype(np.int64)
    sum_f = np.bincount(labels, weights=f, minlength=n_slots)
    sum_o = np.bincount(labels, weights=o, minlength=n_slots)
    occupied = raw_counts > 0
    counts = raw_counts[occupied]
    nk = counts.astype(float)
    obs_freq = sum_o[occupied] / nk
    if group_values is None:
        mean_f = sum_f[occupied] / nk
    else:
        mean_f = np.asarray(group_values, dtype=float)[occupied]

    weights = nk / n
    reliability = float(np.sum(weights * (mean_f - obs_freq) ** 2))
    resolution = float(np.sum(weights * (obs_freq - obar) ** 2))

    # Within-bin terms, accumulated against each sample's own group value.
    per_sample_fk = np.empty(n_slots, dtype=float)
    per_sample_fk[occupied] = mean_f
    dev = f - per_sample_fk[labels]
    wbv = float(np.sum(dev * dev)) / n
    wbc = float(np.sum(dev * o)) / n

    return BrierDecomposition(
        brier=brier_score(f, o),
        base_rate=obar,
        reliability=float(reliability),
        resolution=float(resolution),
        uncertainty=float(uncertainty),
        within_bin_variance=float(wbv),
        within_bin_covariance=float(wbc),
        n_samples=int(n),
        strategy=strategy,
        n_bins_requested=int(n_bins_requested),
        n_bins_occupied=int(counts.size),
        bin_counts=counts,
        bin_mean_forecast=mean_f,
        bin_observed_frequency=obs_freq,
        bin_edges=bin_edges,
    )


def murphy_decomposition(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
) -> BrierDecomposition:
    """Exact Murphy (1973) decomposition, grouping by distinct forecast value.

    No binning and no tolerance: two forecasts share a group only if their
    floating-point values are bit-identical. ``WBV`` and ``WBC`` are therefore
    exactly 0.0 and ``BS = REL - RES + UNC`` holds to machine precision.

    The cost of exactness is that a continuous forecast gives one group per
    sample, which makes ``REL`` equal to the Brier score of a perfectly sharp
    forecast and ``RES`` meaningless. Use this on forecasts that genuinely take
    few values (a lookup table, a quantised model output, a decision rule) and
    ``binned_decomposition`` otherwise.

    Parameters
    ----------
    forecasts, outcomes
        See :func:`calibaudit.scores.check_forecasts`.
    """
    f, o = check_forecasts(forecasts, outcomes)
    values, labels = np.unique(f, return_inverse=True)
    return _decompose(f, o, labels, 0, "exact", values, group_values=values)


def binned_decomposition(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: str = "equal_width",
) -> BrierDecomposition:
    """Binned decomposition with the exact five-term identity.

    ``REL``, ``RES`` and ``UNC`` are the usual binned estimates; ``WBV`` and
    ``WBC`` are the two terms that make the identity exact and that a
    three-term report drops. ``three_term_residual`` is how much was dropped.

    Parameters
    ----------
    forecasts, outcomes
        See :func:`calibaudit.scores.check_forecasts`.
    n_bins
        Number of bins, at least 1.
    strategy
        ``"equal_width"`` for fixed-width bins on [0, 1], or ``"equal_mass"``
        for bins at sample quantiles of the forecast. See
        :mod:`calibaudit.binning`.
    """
    f, o = check_forecasts(forecasts, outcomes)
    labels, edges = assign_bins(f, n_bins=n_bins, strategy=strategy)
    return _decompose(f, o, labels, n_bins, strategy, edges)
