"""Bootstrap intervals on the aggregate stale fraction, with a declared unit.

THE RESAMPLING UNIT IS DECLARED AND ASSERTED, NOT DEFAULTED
-----------------------------------------------------------
:func:`bootstrap_stale_fraction` has **no default** for ``unit``. A bootstrap
whose resampling unit is not stated is not a reported interval, because the
unit decides the answer: evidence items that share a subject artifact are not
independent, and resampling items pretends they are.

Two units are implemented and both are reported side by side:

- ``"cluster"`` -- resample SUBJECT ARTIFACTS with replacement and take every
  evidence item attached to each drawn artifact. This is the cluster (or
  "block") bootstrap of the dependence structure the corpus actually has: all
  the evidence about one artifact goes stale together when that artifact's
  closure changes. This is the unit this product reports as primary.
- ``"item"`` -- resample evidence items with replacement. Narrower, and
  narrower for the wrong reason: it treats correlated items as independent
  draws. It is reported so the reader can see how much the dependence costs.

The percentile method is used, not the bias-corrected and accelerated variant:
B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman and
Hall, 1993, chapters 13 and 14 for the percentile interval, and chapter 8 for
the cluster form in which the resampled unit is the independent group rather
than the observation. The percentile interval is chosen because the statistic
is a bounded proportion and the corpus has enough clusters for the percentile
interval to be the simpler honest choice; BCa is not implemented and that is
recorded in README.md under Limitations.

The interval is a statement about resampling variability of the measured
corpus. It is **not** a statement about a population of real assurance cases,
because the corpus is synthetic. That distinction is restated wherever a number
from here appears.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: The resampling units this module implements. ``unit`` has no default.
UNITS: tuple[str, ...] = ("cluster", "item")


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile bootstrap interval with its unit and settings recorded.

    Attributes
    ----------
    point:
        The statistic on the observed data, dimensionless fraction.
    low, high:
        Percentile interval endpoints, dimensionless fractions.
    unit:
        The declared resampling unit, one of :data:`UNITS`.
    n_units:
        Number of resampling units in the observed data.
    resamples:
        Number of bootstrap resamples drawn.
    confidence:
        Nominal coverage, e.g. ``0.95``.
    seed:
        Seed of the ``numpy.random.default_rng`` used.
    """

    point: float
    low: float
    high: float
    unit: str
    n_units: int
    resamples: int
    confidence: float
    seed: int

    @property
    def half_width(self) -> float:
        """Half the interval width, dimensionless."""
        return 0.5 * (self.high - self.low)

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies in the closed interval."""
        return self.low <= value <= self.high


def bootstrap_stale_fraction(
    stale: Sequence[int],
    groups: Sequence[str],
    *,
    unit: str,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 980098,
) -> BootstrapInterval:
    """Percentile bootstrap interval on the aggregate stale fraction.

    Parameters
    ----------
    stale:
        0/1 staleness labels, one per evidence item.
    groups:
        The subject artifact of each evidence item, aligned with ``stale``.
        Used as the cluster label for ``unit="cluster"``.
    unit:
        ``"cluster"`` or ``"item"``. No default: see the module docstring.
    resamples:
        Bootstrap resamples. 2000 is the value every reported number uses.
    confidence:
        Nominal coverage of the percentile interval.
    seed:
        Seed for ``numpy.random.default_rng`` (PCG64).

    Returns
    -------
    BootstrapInterval
    """
    if unit not in UNITS:
        raise ValueError(
            f"unit={unit!r} is not a resampling unit this module implements. Pass one "
            f"of {UNITS}: 'cluster' resamples subject artifacts and is the unit this "
            f"product reports, 'item' resamples evidence items and is reported beside "
            f"it. There is no default, because an unstated unit is not a reported "
            f"interval."
        )
    if len(stale) != len(groups):
        raise ValueError(
            f"stale has {len(stale)} labels and groups has {len(groups)}; the cluster "
            f"label of each evidence item must be aligned with its label."
        )
    if not stale:
        raise ValueError("bootstrap_stale_fraction() needs at least one evidence item")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 2:
        raise ValueError(f"resamples must be an int of at least 2, got {resamples!r}")

    labels = np.asarray([int(bool(value)) for value in stale], dtype=np.int64)
    point = float(labels.mean())
    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence

    if unit == "item":
        n = labels.size
        draws = rng.integers(0, n, size=(resamples, n))
        statistics = labels[draws].mean(axis=1)
        n_units = n
    else:
        unique = sorted(set(groups))
        index = {name: position for position, name in enumerate(unique)}
        buckets: list[np.ndarray] = [
            labels[np.asarray([index[name] == position for name in groups])]
            for position in range(len(unique))
        ]
        sizes = np.asarray([bucket.size for bucket in buckets], dtype=np.int64)
        sums = np.asarray([int(bucket.sum()) for bucket in buckets], dtype=np.int64)
        draws = rng.integers(0, len(unique), size=(resamples, len(unique)))
        statistics = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
        n_units = len(unique)

    low = float(np.quantile(statistics, alpha / 2.0))
    high = float(np.quantile(statistics, 1.0 - alpha / 2.0))
    return BootstrapInterval(
        point=point,
        low=low,
        high=high,
        unit=unit,
        n_units=n_units,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
