"""Bootstrap intervals, including the two-level one baseline 3 needs.

The sustained figure this product reports is an average over equilibrium samples
from several soak runs, and it has two sources of spread that are not the same
size and must not be pooled as though they were: variation between samples
inside one run, and variation between runs. Pooling them understates the
interval, because samples inside a run are strongly dependent and runs are what
a reader would actually repeat.

So the interval is a two-level (cluster, or hierarchical) bootstrap: resample
the RUNS with replacement, then resample the samples within each drawn run with
replacement, recompute the statistic, repeat. See Efron & Tibshirani (1993), *An
Introduction to the Bootstrap*, chapters 6 and 13 (the percentile interval), and
Davison & Hinkley (1997), *Bootstrap Methods and their Application*, section
3.8 on resampling hierarchical data. The point estimate is the unweighted mean
of the per-run means, which is the quantity whose spread across repetitions the
interval describes.

**The honest limitation of a three-run cluster bootstrap, stated here and
repeated wherever a number from it is quoted.** With three runs there are
3**3 = 27 distinct multisets the run level can draw, so the run-level
resampling distribution is supported on at most 27 points however many
resamples are requested. The interval is therefore granular, and it is not a
substitute for more runs. :attr:`BootstrapResult.n_distinct_group_draws`
carries that number so it can be printed beside the interval, and
:func:`t_interval_of_group_means` provides an interval derived from entirely
different mathematics -- a Student-t interval on the per-run means, two degrees
of freedom -- as an independent cross-check rather than as a replacement.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

#: Resamples used wherever this module is called with the default. 20000 costs a
#: few tenths of a second for the array sizes here on two cores.
DEFAULT_RESAMPLES: int = 20000

#: Coverage of every interval in this module.
DEFAULT_CONFIDENCE: float = 0.95


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate with a percentile bootstrap interval.

    Attributes
    ----------
    point:
        The statistic on the observed data, not the mean of the resamples.
    ci_low, ci_high:
        Percentile interval at :attr:`confidence`.
    confidence:
        Coverage.
    n_resamples:
        Resamples actually drawn.
    standard_error:
        Standard deviation of the resampled statistic, ddof=1.
    bias:
        Mean of the resampled statistic minus :attr:`point`. A large bias
        relative to the standard error is a reason to distrust a percentile
        interval and is reported rather than hidden.
    n_groups, n_samples:
        Number of clusters and total samples the statistic was computed over.
    n_distinct_group_draws:
        ``n_groups ** n_groups`` for a two-level bootstrap -- the number of
        distinct multisets the cluster level can draw -- or ``None`` for a
        single-level bootstrap.
    """

    point: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    standard_error: float
    bias: float
    n_groups: int
    n_samples: int
    n_distinct_group_draws: int | None = None

    @property
    def half_width(self) -> float:
        """Half the width of the interval, in the units of the statistic."""
        return 0.5 * (self.ci_high - self.ci_low)

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies inside the closed interval."""
        return self.ci_low <= value <= self.ci_high

    def __str__(self) -> str:
        return f"{self.point:.6g} [{self.ci_low:.6g}, {self.ci_high:.6g}]"


def _percentiles(confidence: float) -> tuple[float, float]:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    alpha = 0.5 * (1.0 - confidence)
    return 100.0 * alpha, 100.0 * (1.0 - alpha)


def _finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> BootstrapResult:
    """Single-level percentile bootstrap of the mean of ``values``.

    Non-finite entries are dropped before resampling. Raises on an empty or
    single-element sample: an interval over one point is not an interval.
    """
    sample = _finite(values)
    if sample.size < 2:
        raise ValueError(
            f"bootstrap_mean needs at least 2 finite samples, got {sample.size}; "
            "an empty or single-point input cannot produce an interval"
        )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, sample.size, size=(n_resamples, sample.size))
    statistics = sample[draws].mean(axis=1)
    low, high = _percentiles(confidence)
    point = float(sample.mean())
    return BootstrapResult(
        point=point,
        ci_low=float(np.percentile(statistics, low)),
        ci_high=float(np.percentile(statistics, high)),
        confidence=confidence,
        n_resamples=n_resamples,
        standard_error=float(np.std(statistics, ddof=1)),
        bias=float(statistics.mean() - point),
        n_groups=1,
        n_samples=int(sample.size),
    )


def _cluster_statistics(
    groups: Sequence[np.ndarray],
    rng: np.random.Generator,
    n_resamples: int,
    reducer: Callable[[np.ndarray], float],
) -> np.ndarray:
    n_groups = len(groups)
    statistics = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        chosen = rng.integers(0, n_groups, size=n_groups)
        means = np.empty(n_groups, dtype=float)
        for slot, group_index in enumerate(chosen):
            group = groups[group_index]
            picks = rng.integers(0, group.size, size=group.size)
            means[slot] = group[picks].mean()
        statistics[index] = reducer(means)
    return statistics


def bootstrap_cluster_mean(
    groups: Sequence[Sequence[float] | np.ndarray],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> BootstrapResult:
    """Two-level bootstrap of the unweighted mean of per-group means.

    Parameters
    ----------
    groups:
        One sequence of samples per run. Non-finite entries are dropped per
        group.
    n_resamples, confidence, seed:
        As elsewhere in this module.

    Raises
    ------
    ValueError
        If fewer than two groups survive, or any surviving group is empty. Two
        groups is already a weak basis for a run-level interval and the result
        says so through :attr:`BootstrapResult.n_distinct_group_draws`; one
        group is not a basis at all.
    """
    cleaned = [_finite(group) for group in groups]
    cleaned = [group for group in cleaned if group.size > 0]
    if len(cleaned) < 2:
        raise ValueError(
            f"a run-level bootstrap needs at least 2 non-empty groups, got {len(cleaned)}; "
            "the between-run spread is not estimable from one run"
        )
    rng = np.random.default_rng(seed)
    statistics = _cluster_statistics(cleaned, rng, n_resamples, lambda means: float(means.mean()))
    low, high = _percentiles(confidence)
    point = float(np.mean([group.mean() for group in cleaned]))
    return BootstrapResult(
        point=point,
        ci_low=float(np.percentile(statistics, low)),
        ci_high=float(np.percentile(statistics, high)),
        confidence=confidence,
        n_resamples=n_resamples,
        standard_error=float(np.std(statistics, ddof=1)),
        bias=float(statistics.mean() - point),
        n_groups=len(cleaned),
        n_samples=int(sum(group.size for group in cleaned)),
        n_distinct_group_draws=len(cleaned) ** len(cleaned),
    )


def cluster_resampled_means(
    groups: Sequence[Sequence[float] | np.ndarray],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> np.ndarray:
    """Draws of the unweighted mean of per-group means, two-level resampling.

    Exposed so that two statistics derived from the same quantity -- the
    sustained fraction and the overstatement, which are reciprocals of each
    other shifted by one -- can be computed from the SAME resample draws rather
    than from two independent bootstraps that would not agree with each other.

    Raises
    ------
    ValueError
        If fewer than two non-empty groups survive.
    """
    cleaned = [group for group in (_finite(g) for g in groups) if group.size]
    if len(cleaned) < 2:
        raise ValueError(
            f"a run-level bootstrap needs at least 2 non-empty groups, got {len(cleaned)}"
        )
    rng = np.random.default_rng(seed)
    return _cluster_statistics(cleaned, rng, n_resamples, lambda means: float(means.mean()))


def observed_cluster_mean(groups: Sequence[Sequence[float] | np.ndarray]) -> float:
    """Unweighted mean of per-group means over the observed data."""
    cleaned = [group for group in (_finite(g) for g in groups) if group.size]
    if not cleaned:
        raise ValueError("no non-empty group; an empty input set has no mean")
    return float(np.mean([group.mean() for group in cleaned]))


def percentile_interval(
    statistics: np.ndarray, *, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    """Percentile interval of a resampled statistic at ``confidence``."""
    low, high = _percentiles(confidence)
    return float(np.percentile(statistics, low)), float(np.percentile(statistics, high))


def bootstrap_cluster_ratio(
    numerator_groups: Sequence[Sequence[float] | np.ndarray],
    denominator_groups: Sequence[Sequence[float] | np.ndarray],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> BootstrapResult:
    """Two-level bootstrap of the ratio of two unweighted means of group means.

    The two sides are resampled independently, which is the right model here:
    the burst runs and the soak runs are separate executions and nothing pairs a
    particular burst with a particular soak.

    Raises
    ------
    ValueError
        If either side has fewer than two non-empty groups, or if any resample
        produces a zero denominator, which would make the ratio undefined.
    """
    numerators = [group for group in (_finite(g) for g in numerator_groups) if group.size]
    denominators = [group for group in (_finite(g) for g in denominator_groups) if group.size]
    for name, side in (("numerator", numerators), ("denominator", denominators)):
        if len(side) < 2:
            raise ValueError(
                f"{name} side needs at least 2 non-empty groups for a run-level "
                f"interval, got {len(side)}"
            )
    rng = np.random.default_rng(seed)
    top = _cluster_statistics(numerators, rng, n_resamples, lambda means: float(means.mean()))
    bottom = _cluster_statistics(denominators, rng, n_resamples, lambda means: float(means.mean()))
    if np.any(bottom == 0.0):
        raise ValueError("a resample produced a zero denominator; the ratio is undefined")
    statistics = top / bottom
    low, high = _percentiles(confidence)
    point_top = float(np.mean([group.mean() for group in numerators]))
    point_bottom = float(np.mean([group.mean() for group in denominators]))
    if point_bottom == 0.0:
        raise ValueError("the observed denominator is zero; the ratio is undefined")
    point = point_top / point_bottom
    return BootstrapResult(
        point=point,
        ci_low=float(np.percentile(statistics, low)),
        ci_high=float(np.percentile(statistics, high)),
        confidence=confidence,
        n_resamples=n_resamples,
        standard_error=float(np.std(statistics, ddof=1)),
        bias=float(statistics.mean() - point),
        n_groups=min(len(numerators), len(denominators)),
        n_samples=int(
            sum(g.size for g in numerators) + sum(g.size for g in denominators)
        ),
        n_distinct_group_draws=min(len(numerators), len(denominators))
        ** min(len(numerators), len(denominators)),
    )


def t_interval_of_group_means(
    groups: Sequence[Sequence[float] | np.ndarray],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float, float]:
    """Student-t interval on the mean of per-group means. An independent check.

    This shares no machinery with the bootstrap above: it is the classical
    normal-theory interval ``mean +/- t_{1-alpha/2, k-1} * s / sqrt(k)`` over the
    ``k`` per-run means, with ``s`` the sample standard deviation (ddof=1). It is
    reported beside the bootstrap interval as a bound derived from different
    mathematics, as the mission's build guide requires, and NOT as the headline
    interval: with ``k = 3`` the t multiplier is 4.303 and the interval is wide.

    Returns
    -------
    (point, low, high)

    Raises
    ------
    ValueError
        If fewer than two non-empty groups survive.
    """
    means = [float(_finite(group).mean()) for group in groups if _finite(group).size]
    k = len(means)
    if k < 2:
        raise ValueError(f"a t interval over group means needs at least 2 groups, got {k}")
    from scipy import stats

    point = float(np.mean(means))
    spread = float(np.std(means, ddof=1))
    multiplier = float(stats.t.ppf(0.5 * (1.0 + confidence), df=k - 1))
    half = multiplier * spread / math.sqrt(k)
    return point, point - half, point + half
