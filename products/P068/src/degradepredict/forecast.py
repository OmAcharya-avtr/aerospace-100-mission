"""Predictive distributions over remaining life until a limit crossing.

Every forecaster in this package returns one of these objects rather than a
point estimate, because the question the product exists to answer -- "when does
this cross, and is the interval around that date calibrated" -- cannot be scored
without a distribution. Two representations are provided and both expose the
same three operations: a quantile function, a central interval, and a CDF (which
is what the probability integral transform needs).

All times are remaining life in days, measured forward from the forecast origin.
Remaining life is strictly positive by construction: a unit that has already
crossed is not a forecasting case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["CrossingForecast", "QuantileForecast", "TransformedNormalForecast"]

_EPS = 1e-12


class CrossingForecast:
    """Base class: a predictive distribution over remaining life in days."""

    model: str = "unnamed"

    def quantile(self, p: np.ndarray | float) -> np.ndarray | float:
        """Quantile function of remaining life, days. ``p`` in (0, 1).

        Returns a Python float for a scalar argument and an array otherwise.
        """
        raise NotImplementedError

    def cdf(self, remaining_days: np.ndarray | float) -> np.ndarray | float:
        """Predictive CDF at a remaining life in days. Scalar in, scalar out."""
        raise NotImplementedError

    def median(self) -> float:
        """Median remaining life, days."""
        return float(np.asarray(self.quantile(0.5)).reshape(-1)[0])

    def interval(self, level: float = 0.90) -> tuple[float, float]:
        """Equal-tailed central interval on remaining life, days.

        ``level`` is the nominal coverage, so ``level=0.90`` returns the 5th and
        95th percentiles. Equal-tailed, not shortest.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level!r}")
        alpha = 0.5 * (1.0 - level)
        lo, hi = self.quantile(np.array([alpha, 1.0 - alpha]))
        return float(lo), float(hi)

    @staticmethod
    def _shape_out(result: np.ndarray, source) -> np.ndarray | float:
        """Return a float when the caller passed a scalar, else the array."""
        if np.isscalar(source) or np.asarray(source).ndim == 0:
            return float(np.asarray(result).reshape(-1)[0])
        return np.asarray(result, dtype=float)

    @staticmethod
    def _check_p(p: np.ndarray | float) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(p, dtype=float))
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0) or np.any(arr >= 1.0):
            raise ValueError("quantile levels must be finite and strictly inside (0, 1)")
        return arr


@dataclass
class TransformedNormalForecast(CrossingForecast):
    """Normal uncertainty on a monotone transform of the crossing time.

    The three analytic baselines all reduce to ordinary least squares of a
    transformed metric on a transformed time, so the crossing time solves a
    linear equation in ``w = T**z`` for a declared exponent ``z``. The delta
    method gives ``w`` a normal distribution; this class carries it, truncates it
    below at ``w_floor = origin_days**z`` (the unit has not crossed yet), and
    maps it through ``T = w**(1/z)``, which is monotone increasing for ``z > 0``
    so quantiles map directly.

    Reference for the delta method applied to a ratio of regression estimates:
    Fieller, E. C. (1954), "Some problems in interval estimation", Journal of the
    Royal Statistical Society Series B 16(2):175-185; and Draper, N. R. and
    Smith, H. (1998), *Applied Regression Analysis*, 3rd ed., Wiley, section 3.2
    on inverse regression. The delta-method normal is the small-``g`` limit of
    the Fieller set; :mod:`degradepredict.baselines` computes both and
    ``validation/validate_baselines.py`` measures how far apart they are.

    Assumption: the sampling distribution of ``w`` is well approximated by a
    normal. This fails when the fitted slope is not significantly different from
    zero (Fieller's ``g`` near or above 1), in which case the true confidence set
    is unbounded or disjoint and this object understates the uncertainty. The
    ``g`` value is carried on the forecast so the caller can see it.

    Attributes
    ----------
    mu_w, sd_w:
        Mean and standard deviation of the transformed crossing time, units of
        ``day**z``.
    z:
        Transform exponent, dimensionless, strictly positive. ``z = 1`` for the
        linear and exponential baselines.
    origin_days:
        Forecast origin, days of service age.
    fieller_g:
        Fieller's ``g = t_crit**2 * var(slope) / slope**2`` at the level the
        baseline was fitted with. ``g >= 1`` means the exact confidence set is
        not a bounded interval.
    model:
        Label used in reports.
    """

    mu_w: float
    sd_w: float
    z: float
    origin_days: float
    fieller_g: float = float("nan")
    model: str = "transformed-normal"

    def __post_init__(self) -> None:
        if not np.isfinite(self.z) or self.z <= 0:
            raise ValueError(f"z must be finite and positive, got {self.z!r}")
        if not np.isfinite(self.sd_w) or self.sd_w <= 0:
            raise ValueError(f"sd_w must be finite and positive, got {self.sd_w!r}")
        if not np.isfinite(self.origin_days) or self.origin_days < 0:
            raise ValueError(f"origin_days must be finite and non-negative, got "
                             f"{self.origin_days!r}")

    @property
    def _floor(self) -> float:
        return float(self.origin_days**self.z)

    def _dist(self) -> stats.rv_continuous:
        a = (self._floor - self.mu_w) / self.sd_w
        return stats.truncnorm(a, np.inf, loc=self.mu_w, scale=self.sd_w)

    def quantile(self, p: np.ndarray | float) -> np.ndarray | float:
        arr = self._check_p(p)
        w = self._dist().ppf(arr)
        w = np.maximum(w, self._floor)
        out = np.maximum(np.power(w, 1.0 / self.z) - self.origin_days, _EPS)
        return self._shape_out(out, p)

    def cdf(self, remaining_days: np.ndarray | float) -> np.ndarray | float:
        r = np.atleast_1d(np.asarray(remaining_days, dtype=float))
        w = np.power(np.maximum(r + self.origin_days, 0.0), self.z)
        return self._shape_out(np.asarray(self._dist().cdf(w), dtype=float), remaining_days)


@dataclass
class QuantileForecast(CrossingForecast):
    """Predictive distribution carried as a finite set of quantiles.

    This is what the learned forecaster produces: one gradient-boosted quantile
    regression per level, so the distribution is known only at those levels.
    Between them the quantile function is linearly interpolated in probability;
    outside the outermost levels it is held constant.

    That flat extrapolation is a real approximation and it biases scores in a
    known direction: a realisation beyond the outermost fitted quantile is
    charged a pinball loss against a tail that is shorter than the model's true
    tail, so CRPS computed this way is **optimistic for tail misses**. It is
    applied identically to every model scored through
    :func:`degradepredict.metrics.crps_from_quantiles`, including the analytic
    baselines, so the comparison is like for like even though the absolute value
    is not the exact CRPS of the baselines' normal law.

    Quantile crossing (a fitted 0.75 quantile below the fitted 0.5) is possible
    because the levels are fitted independently; it is removed by a cumulative
    maximum over levels, following the standard rearrangement argument of
    Chernozhukov, V., Fernandez-Val, I. and Galichon, A. (2010), "Quantile and
    probability curves without crossing", Econometrica 78(3):1093-1125, which
    shows the rearranged curve is no further from the true quantile curve than
    the unsorted one.

    Attributes
    ----------
    levels:
        Strictly increasing probabilities in (0, 1).
    values_days:
        Predicted remaining life in days at each level, after rearrangement.
    """

    levels: np.ndarray
    values_days: np.ndarray
    model: str = "quantile-forecast"

    def __post_init__(self) -> None:
        self.levels = np.asarray(self.levels, dtype=float)
        self.values_days = np.asarray(self.values_days, dtype=float)
        if self.levels.shape != self.values_days.shape:
            raise ValueError("levels and values_days must have the same shape")
        if self.levels.ndim != 1 or self.levels.size < 2:
            raise ValueError("need at least two quantile levels")
        if np.any(self.levels <= 0) or np.any(self.levels >= 1):
            raise ValueError("levels must be strictly inside (0, 1)")
        if not np.all(np.diff(self.levels) > 0):
            raise ValueError("levels must be strictly increasing")
        if not np.all(np.isfinite(self.values_days)):
            raise ValueError("values_days must all be finite")
        self.values_days = np.maximum.accumulate(np.maximum(self.values_days, _EPS))

    def quantile(self, p: np.ndarray | float) -> np.ndarray | float:
        arr = self._check_p(p)
        return self._shape_out(np.interp(arr, self.levels, self.values_days), p)

    def cdf(self, remaining_days: np.ndarray | float) -> np.ndarray | float:
        r = np.atleast_1d(np.asarray(remaining_days, dtype=float))
        v = self.values_days
        # Strictly increasing abscissa for np.interp; ties would make it pick an
        # arbitrary level, so nudge duplicates apart by one ulp-scale step.
        vv = np.maximum.accumulate(v + np.arange(v.size) * _EPS)
        out = np.clip(np.interp(r, vv, self.levels), 0.0, 1.0)
        return self._shape_out(out, remaining_days)
