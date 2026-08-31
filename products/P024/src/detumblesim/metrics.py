"""Confidence intervals for Monte Carlo policy comparisons.

Only two things are computed here, and both are ordinary Student-t intervals:
a marginal interval on the mean of one metric, and an interval on the mean of
the **paired** per-scenario difference between two policies.  The paired
interval is the one that licenses a claim that one policy beats another,
because every scenario is simulated under both policies with identical
initial conditions, orbit and hardware.

Interpreting them
-----------------
- A paired-difference interval that excludes zero means the difference is
  resolved at this sample size.
- A paired-difference interval that contains zero means **this experiment
  cannot tell the two policies apart**.  It does not mean they are equal.
- Marginal intervals that overlap say even less: overlapping marginals are
  compatible with a clearly resolved paired difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


@dataclass(frozen=True)
class Interval:
    """Mean with a two-sided Student-t confidence interval.

    Attributes
    ----------
    mean : float
        Sample mean (units of the input).
    ci_low, ci_high : float
        Interval endpoints.
    n : int
        Number of samples.
    ci_level : float
        Nominal coverage, e.g. 0.95.
    std : float
        Sample standard deviation (ddof=1); NaN when ``n == 1``.
    """

    mean: float
    ci_low: float
    ci_high: float
    n: int
    ci_level: float
    std: float

    @property
    def half_width(self) -> float:
        """Half-width of the interval (same units as ``mean``)."""
        return 0.5 * (self.ci_high - self.ci_low)

    @property
    def excludes_zero(self) -> bool:
        """True if the interval does not contain zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


def mean_ci(values: ArrayLike, ci_level: float = 0.95) -> Interval:
    """Student-t confidence interval on the mean of ``values``.

    With a single sample the interval collapses to the point estimate and the
    standard deviation is NaN; that is reported, not hidden.

    Raises
    ------
    ValueError
        If ``values`` is empty, contains non-finite entries, or ``ci_level``
        is not strictly between 0 and 1.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("values must be non-empty")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite; drop or impute failures first")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must lie in (0, 1), got {ci_level}")
    m = float(v.mean())
    n = int(v.size)
    if n == 1:
        return Interval(m, m, m, 1, ci_level, float("nan"))
    sd = float(v.std(ddof=1))
    if sd == 0.0:
        return Interval(m, m, m, n, ci_level, 0.0)
    tcrit = float(stats.t.ppf(0.5 * (1.0 + ci_level), n - 1))
    hw = tcrit * sd / np.sqrt(n)
    return Interval(m, m - hw, m + hw, n, ci_level, sd)


def paired_difference_ci(
    a: ArrayLike, b: ArrayLike, ci_level: float = 0.95
) -> Interval:
    """Interval on the mean of ``a - b`` for paired samples.

    ``a`` and ``b`` must be the same length and aligned scenario-by-scenario.
    A negative mean with an interval entirely below zero means ``a`` is
    smaller than ``b`` (better, for a cost metric) at this sample size.
    """
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"paired samples must have equal length, got {x.shape} and {y.shape}")
    return mean_ci(x - y, ci_level=ci_level)


def format_interval(interval: Interval, unit: str = "", digits: int = 3) -> str:
    """Human-readable ``mean [low, high] unit`` string."""
    u = f" {unit}" if unit else ""
    return (
        f"{interval.mean:.{digits}f} "
        f"[{interval.ci_low:.{digits}f}, {interval.ci_high:.{digits}f}]{u}"
    )
