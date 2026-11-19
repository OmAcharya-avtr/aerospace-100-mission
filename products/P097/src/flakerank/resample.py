"""Percentile bootstrap intervals with the resampling unit declared on the result.

The resampling unit is the whole content of a bootstrap interval. Resample the
wrong unit and the interval is a number with the right shape and no meaning:
resampling *attempts* when the dependence lives between attempts in the same
session gives an interval that is too narrow by whatever the within-session
correlation is, and nothing about the returned float says so.

So :class:`BootstrapInterval` carries ``resampling_unit`` as a field, and
:meth:`BootstrapInterval.require_resampling_unit` raises when a caller's
expectation and the producer's declaration disagree. The ground-truth
accumulation in :mod:`flakerank.baselines` calls it on its own result before
returning, the rank-disagreement statistic in :mod:`flakerank.disagreement`
does the same, and ``tests/test_resample.py`` asserts both.

Method: the percentile bootstrap (Efron, B., 1979, "Bootstrap methods: another
look at the jackknife", *Annals of Statistics* 7(1), 1-26; Efron, B. &
Tibshirani, R. J., 1993, *An Introduction to the Bootstrap*, Chapman & Hall).
The interval is ``[Q(alpha/2), Q(1 - alpha/2)]`` of the resampled statistic,
with ``alpha = 1 - confidence``.

What the percentile interval does NOT do, stated because it matters here:

* it does not correct for bias or skew, and the wasted-time statistic is
  strongly right-skewed, so the interval is approximate. BCa would be better
  and needs an acceleration estimate the product does not compute;
* it is a *resampling* interval and inherits the independence assumption of the
  resampling unit. Sessions are taken as exchangeable. A CI outage that makes
  three consecutive sessions fail together violates that, and the interval is
  then too narrow;
* for a test with zero observed failures every resample gives zero, so the
  interval is ``[0, 0]``. That is a correct statement about the resampling
  distribution and a misleading statement about the test, whose true rate is
  only bounded above. ``README.md`` records it under Limitations.

Every interval in this product comes from ``numpy.random.default_rng`` (PCG64)
at a declared seed, so two runs at the same seed give the same interval.
``tests/test_determinism.py`` asserts that.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BootstrapInterval",
    "bootstrap_mean_interval",
    "bootstrap_statistic_interval",
]


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile bootstrap interval with its resampling unit declared.

    Attributes:
        point: the statistic on the observed sample, in the statistic's own
            units.
        low: lower percentile of the resampled statistic.
        high: upper percentile of the resampled statistic.
        confidence: nominal coverage, e.g. ``0.95``.
        resamples: number of bootstrap resamples drawn.
        resampling_unit: what was resampled, e.g. ``"session"`` or ``"test"``.
            This is not decoration: see the module docstring.
        n_units: number of units in the observed sample.
        seed: the seed the resampler was constructed with.
        units: the units of ``point``, ``low`` and ``high``, e.g. ``"second"``
            or ``"dimensionless"``.
    """

    point: float
    low: float
    high: float
    confidence: float
    resamples: int
    resampling_unit: str
    n_units: int
    seed: int
    units: str = "dimensionless"

    @property
    def width(self) -> float:
        """``high - low``, in the statistic's units."""
        return self.high - self.low

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies in the closed interval ``[low, high]``.

        A closed test is used deliberately. The interval ends are empirical
        quantiles of a resampled statistic, so a value equal to an end is an
        observed resample value and excluding it would make the coverage
        statement depend on a floating-point tie.
        """
        return self.low <= value <= self.high

    def require_resampling_unit(self, expected: str) -> BootstrapInterval:
        """Return ``self`` if the resampling unit matches, else raise.

        Args:
            expected: the unit the caller believes was resampled.

        Returns:
            ``self``, so the call can wrap a return value.

        Raises:
            ValueError: when the declared and expected units differ. The
                message says why that is not a cosmetic disagreement.
        """
        if self.resampling_unit != expected:
            raise ValueError(
                f"this interval was built by resampling {self.resampling_unit!r}, "
                f"but {expected!r} was expected. The resampling unit is the whole "
                "content of a bootstrap interval: resampling attempts when the "
                "dependence lives between attempts in the same session gives an "
                "interval that is too narrow by the within-session correlation, and "
                "nothing about the number itself says so."
            )
        return self


def _check_confidence(confidence: float) -> float:
    """Validate a nominal coverage in ``(0, 1)``."""
    value = float(confidence)
    if not 0.0 < value < 1.0:
        raise ValueError(
            f"confidence must lie strictly in (0, 1), got {value!r}. It is a nominal "
            "coverage, not a percentage."
        )
    return value


def _check_resamples(resamples: int) -> int:
    """Validate a resample count."""
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 2:
        raise ValueError(
            f"resamples must be an integer >= 2, got {resamples!r}. Two is the smallest "
            "number for which a percentile has any meaning, and a reported interval "
            "should use far more; this product declares its count in every result."
        )
    return resamples


def bootstrap_statistic_interval(
    units: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    resampling_unit: str,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 97044,
    value_units: str = "dimensionless",
) -> BootstrapInterval:
    """Percentile bootstrap interval for ``statistic`` over rows of ``units``.

    Rows of ``units`` are the resampling unit. For a one-dimensional array each
    element is a unit; for a two-dimensional array each ROW is a unit, which is
    how a whole session's per-test wasted time is resampled together.

    Args:
        units: observed sample, shape ``(n,)`` or ``(n, k)``.
        statistic: maps a resampled sample of the same shape to one float.
        resampling_unit: name of the unit, recorded on the result.
        resamples: number of resamples.
        confidence: nominal coverage in ``(0, 1)``.
        seed: seed for ``numpy.random.default_rng``.
        value_units: units of the statistic, recorded on the result.

    Returns:
        A :class:`BootstrapInterval`.

    Raises:
        ValueError: for an empty sample, a sample with more than two
            dimensions, an invalid confidence or an invalid resample count.
    """
    sample = np.asarray(units, dtype=np.float64)
    if sample.ndim not in (1, 2):
        raise ValueError(
            f"units must be one- or two-dimensional, got {sample.ndim} dimensions; "
            "rows are the resampling unit"
        )
    if sample.shape[0] == 0:
        raise ValueError("units must contain at least one resampling unit")
    if not resampling_unit:
        raise ValueError(
            "resampling_unit must be a non-empty name: an interval whose unit is "
            "unstated cannot be checked by the caller"
        )
    confidence = _check_confidence(confidence)
    resamples = _check_resamples(resamples)
    count = int(sample.shape[0])
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        rows = generator.integers(0, count, size=count)
        draws[index] = statistic(sample[rows])
    alpha = 1.0 - confidence
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapInterval(
        point=float(statistic(sample)),
        low=float(low),
        high=float(high),
        confidence=confidence,
        resamples=resamples,
        resampling_unit=resampling_unit,
        n_units=count,
        seed=int(seed),
        units=value_units,
    )


def bootstrap_mean_interval(
    units: np.ndarray,
    *,
    resampling_unit: str,
    total: bool = False,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 97044,
    value_units: str = "second",
) -> BootstrapInterval:
    """Interval for the mean, or for the sum, over the resampling unit.

    The ground-truth accumulation needs an interval on a TOTAL -- wasted
    seconds over the whole history -- and the bootstrap gives one by resampling
    ``n`` units and scaling the resampled mean by ``n``. Resampling the sum
    directly would be the same thing written less clearly.

    Args:
        units: observed per-unit values, shape ``(n,)``.
        resampling_unit: name of the unit, recorded on the result.
        total: if true the statistic is ``n * mean``, the total over the
            observed number of units; if false it is the mean per unit.
        resamples: number of resamples.
        confidence: nominal coverage.
        seed: seed for ``numpy.random.default_rng``.
        value_units: units of the statistic.

    Returns:
        A :class:`BootstrapInterval`.
    """
    sample = np.asarray(units, dtype=np.float64)
    if sample.ndim != 1:
        raise ValueError(f"units must be one-dimensional for a mean, got shape {sample.shape}")
    scale = float(sample.shape[0]) if total else 1.0

    def statistic(resampled: np.ndarray) -> float:
        return float(resampled.mean()) * scale

    return bootstrap_statistic_interval(
        sample,
        statistic,
        resampling_unit=resampling_unit,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
        value_units=value_units,
    )
