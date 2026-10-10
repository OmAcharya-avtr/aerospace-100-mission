"""Reliability diagrams with bootstrap bands, and a check that the bands work.

A reliability diagram plots the observed relative frequency of the outcome
against the mean forecast, one point per bin. The point estimate is cheap; the
hard part is knowing which departures from the diagonal are real, which is
what the bands are for.

The bands here are percentile bootstrap intervals over the ``(f_i, o_i)``
pairs: resample ``n`` pairs with replacement, recompute each bin's observed
frequency, and take percentiles across replicates. **The bin edges are fixed
once, from the original sample, and reused for every replicate.** Recomputing
quantile edges inside the loop would give each replicate a different set of
bins, and the percentile across replicates would then mix bins rather than
bound one. Equal-width edges do not depend on the sample at all, so for that
strategy the distinction does not arise.

A bootstrap band is not a confidence band for the whole curve: the stated
level is pointwise, per bin, and a curve with ``B`` bins has roughly ``B``
chances to step outside one. ``band_coverage`` measures both the pointwise
rate and the all-bins-simultaneously rate against the known calibration curve
of a synthetic spec, and the gap between them is reported rather than
explained away.

References
----------
Murphy, A. H. and Winkler, R. L. (1977). "Reliability of subjective
probability forecasts of precipitation and temperature." *Journal of the Royal
Statistical Society Series C* 26(1), 41-47. doi:10.2307/2346866

Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
Chapman and Hall. (the percentile interval, chapter 13)

Bröcker, J. and Smith, L. A. (2007). "Increasing the reliability of
reliability diagrams." *Weather and Forecasting* 22(3), 651-661.
doi:10.1175/WAF993.1 (consistency bands under the calibrated null)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from .binning import bin_edges
from .scores import check_forecasts

__all__ = [
    "BandCoverage",
    "ReliabilityCurve",
    "band_coverage",
    "bootstrap_reliability",
    "reliability_curve",
]


def _labels_from_edges(f: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n_bins = edges.size - 1
    if n_bins == 1:
        return np.zeros(f.shape, dtype=np.int64)
    labels = np.digitize(f, edges[1:-1], right=False).astype(np.int64)
    return np.clip(labels, 0, n_bins - 1)


@dataclass(frozen=True)
class ReliabilityCurve:
    """One reliability diagram. All quantities dimensionless.

    Attributes
    ----------
    bin_index
        Index into ``edges`` of each occupied bin, ascending.
    mean_forecast, observed_frequency, counts
        Per-occupied-bin mean forecast, observed relative frequency, and
        sample count.
    lower, upper
        Pointwise bootstrap band on ``observed_frequency``, or ``None`` when
        no bootstrap was requested.
    level
        Nominal pointwise coverage of the band, e.g. 0.9.
    n_bootstrap
        Number of bootstrap replicates, 0 when no band was computed.
    edges
        The ``n_bins + 1`` bin edges actually used.
    """

    bin_index: np.ndarray = field(repr=False)
    mean_forecast: np.ndarray = field(repr=False)
    observed_frequency: np.ndarray = field(repr=False)
    counts: np.ndarray = field(repr=False)
    edges: np.ndarray = field(repr=False)
    n_samples: int
    n_bins: int
    strategy: str
    level: float = 0.0
    n_bootstrap: int = 0
    lower: np.ndarray | None = field(default=None, repr=False)
    upper: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_occupied(self) -> int:
        return int(self.bin_index.size)

    def table(self) -> str:
        """Fixed-width table of the diagram, one line per occupied bin."""
        band = self.lower is not None
        head = f"{'bin':>4} {'lo_edge':>8} {'hi_edge':>8} {'n':>7} {'mean_f':>9} {'obs':>9}"
        if band:
            head += f" {'band_lo':>9} {'band_hi':>9} {'gap':>9}"
        lines = [head, "-" * len(head)]
        for j in range(self.n_occupied):
            k = int(self.bin_index[j])
            row = (
                f"{k:>4d} {self.edges[k]:>8.4f} {self.edges[k + 1]:>8.4f} "
                f"{int(self.counts[j]):>7d} {self.mean_forecast[j]:>9.6f} "
                f"{self.observed_frequency[j]:>9.6f}"
            )
            if band:
                gap = self.mean_forecast[j] - self.observed_frequency[j]
                row += (
                    f" {self.lower[j]:>9.6f} {self.upper[j]:>9.6f} {gap:>+9.6f}"
                )
            lines.append(row)
        return "\n".join(lines)

    def diagonal_excluded(self) -> np.ndarray:
        """Boolean mask of bins whose band excludes the mean forecast.

        These are the bins where the diagram says the forecaster is
        miscalibrated at the stated pointwise level.

        Raises
        ------
        ValueError
            If no band was computed.
        """
        if self.lower is None or self.upper is None:
            raise ValueError("no bootstrap band on this curve; call bootstrap_reliability")
        return (self.mean_forecast < self.lower) | (self.mean_forecast > self.upper)


def _summarise(
    f: np.ndarray, o: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = _labels_from_edges(f, edges)
    n_slots = edges.size - 1
    raw_counts = np.bincount(labels, minlength=n_slots).astype(np.int64)
    sum_f = np.bincount(labels, weights=f, minlength=n_slots)
    sum_o = np.bincount(labels, weights=o, minlength=n_slots)
    mask = raw_counts > 0
    occupied = np.flatnonzero(mask).astype(np.int64)
    counts = raw_counts[mask]
    nk = counts.astype(float)
    return occupied, sum_f[mask] / nk, sum_o[mask] / nk, counts


def reliability_curve(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: str = "equal_width",
) -> ReliabilityCurve:
    """Reliability diagram without bands."""
    f, o = check_forecasts(forecasts, outcomes)
    edges = bin_edges(f, n_bins=n_bins, strategy=strategy)
    occupied, mean_f, obs, counts = _summarise(f, o, edges)
    return ReliabilityCurve(
        bin_index=occupied,
        mean_forecast=mean_f,
        observed_frequency=obs,
        counts=counts,
        edges=edges,
        n_samples=int(f.size),
        n_bins=int(n_bins),
        strategy=str(strategy),
    )


def bootstrap_reliability(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: str = "equal_width",
    n_bootstrap: int = 1000,
    level: float = 0.9,
    seed: int = 0,
) -> ReliabilityCurve:
    """Reliability diagram with pointwise percentile bootstrap bands.

    Bins with no samples in a given replicate contribute nothing to that
    replicate, so a bin that is frequently empty gets a band computed from
    fewer than ``n_bootstrap`` values; the band is ``nan`` if it is empty in
    every replicate, which cannot happen for a bin occupied in the original
    sample but can happen for a bin holding one sample if that sample is never
    drawn.

    Raises
    ------
    ValueError
        If ``level`` is outside (0, 1) or ``n_bootstrap < 2``.
    """
    f, o = check_forecasts(forecasts, outcomes)
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level!r}")
    if n_bootstrap < 2:
        raise ValueError(f"n_bootstrap must be at least 2, got {n_bootstrap!r}")

    edges = bin_edges(f, n_bins=n_bins, strategy=strategy)
    occupied, mean_f, obs, counts = _summarise(f, o, edges)
    labels = _labels_from_edges(f, edges)

    rng = np.random.default_rng(int(seed))
    n = f.size
    n_slots = edges.size - 1
    reps = np.full((int(n_bootstrap), occupied.size), np.nan, dtype=float)
    for r in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        lab_r = labels[idx]
        cnt_r = np.bincount(lab_r, minlength=n_slots).astype(float)
        sum_r = np.bincount(lab_r, weights=o[idx], minlength=n_slots)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(cnt_r > 0, sum_r / np.maximum(cnt_r, 1.0), np.nan)
        reps[r] = means[occupied]

    alpha = (1.0 - float(level)) / 2.0
    with np.errstate(invalid="ignore"):
        lower = np.nanquantile(reps, alpha, axis=0)
        upper = np.nanquantile(reps, 1.0 - alpha, axis=0)
    return ReliabilityCurve(
        bin_index=occupied,
        mean_forecast=mean_f,
        observed_frequency=obs,
        counts=counts,
        edges=edges,
        n_samples=int(n),
        n_bins=int(n_bins),
        strategy=str(strategy),
        level=float(level),
        n_bootstrap=int(n_bootstrap),
        lower=lower,
        upper=upper,
    )


@dataclass(frozen=True)
class BandCoverage:
    """Measured coverage of bootstrap reliability bands against a known curve."""

    spec_name: str
    n_samples: int
    n_bins: int
    strategy: str
    level: float
    n_bootstrap: int
    n_replicates: int
    pointwise_coverage: float
    pointwise_n: int
    simultaneous_coverage: float
    per_bin_coverage: np.ndarray = field(repr=False)
    per_bin_n: np.ndarray = field(repr=False)

    def report(self) -> str:
        """Multi-line summary."""
        return "\n".join(
            [
                f"bootstrap band coverage, spec = {self.spec_name}, "
                f"n = {self.n_samples}, bins = {self.n_bins}, strategy = {self.strategy}",
                f"  nominal pointwise level : {self.level:.3f}",
                f"  measured pointwise      : {self.pointwise_coverage:.4f} "
                f"over {self.pointwise_n} bin-replicates",
                f"  measured simultaneous   : {self.simultaneous_coverage:.4f} "
                f"over {self.n_replicates} replicates",
                f"  bootstrap replicates    : {self.n_bootstrap}",
            ]
        )


def band_coverage(
    spec,
    *,
    n_samples: int,
    n_bins: int = 10,
    strategy: str = "equal_width",
    level: float = 0.9,
    n_bootstrap: int = 200,
    n_replicates: int = 100,
    seed: int = 0,
) -> BandCoverage:
    """Measure whether the bands cover the spec's true calibration curve.

    For each replicate, draw a fresh sample from ``spec``, compute bands, and
    ask whether each bin's band contains the exact conditional probability
    ``E[o | f]`` evaluated at that bin's mean forecast. Pointwise coverage
    pools over bins and replicates; simultaneous coverage counts a replicate
    as covered only when every occupied bin is covered.

    The exact conditional probability is evaluated at the bin's *mean
    forecast*, which is itself an approximation when the calibration map is
    curved inside the bin: for a convex map the bin's true mean outcome is not
    ``g^{-1}(fbar_k)``. The resulting discrepancy is a property of binning,
    not of the bootstrap, and it grows with bin width.
    """
    from .synthetic import calibration_map, sample_forecast

    hits_total = 0
    n_total = 0
    simultaneous = 0
    per_bin_hits = np.zeros(int(n_bins), dtype=np.int64)
    per_bin_n = np.zeros(int(n_bins), dtype=np.int64)
    for r in range(int(n_replicates)):
        s = sample_forecast(spec, int(n_samples), seed=int(seed) * 9973 + r + 1)
        curve = bootstrap_reliability(
            s.forecasts,
            s.outcomes,
            n_bins=int(n_bins),
            strategy=strategy,
            n_bootstrap=int(n_bootstrap),
            level=float(level),
            seed=int(seed) * 104729 + r + 1,
        )
        truth = calibration_map(spec, curve.mean_forecast)
        inside = (truth >= curve.lower) & (truth <= curve.upper)
        hits_total += int(np.sum(inside))
        n_total += int(inside.size)
        if bool(np.all(inside)):
            simultaneous += 1
        for j, k in enumerate(curve.bin_index):
            per_bin_n[int(k)] += 1
            per_bin_hits[int(k)] += int(inside[j])
    with np.errstate(invalid="ignore", divide="ignore"):
        per_bin = np.where(per_bin_n > 0, per_bin_hits / np.maximum(per_bin_n, 1), np.nan)
    return BandCoverage(
        spec_name=spec.name,
        n_samples=int(n_samples),
        n_bins=int(n_bins),
        strategy=str(strategy),
        level=float(level),
        n_bootstrap=int(n_bootstrap),
        n_replicates=int(n_replicates),
        pointwise_coverage=float(hits_total / n_total) if n_total else float("nan"),
        pointwise_n=int(n_total),
        simultaneous_coverage=float(simultaneous / int(n_replicates)),
        per_bin_coverage=per_bin,
        per_bin_n=per_bin_n,
    )
