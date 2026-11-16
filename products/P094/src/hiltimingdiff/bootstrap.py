"""Bootstrap whose resampling unit is the PASS, declared by SPEC §4.

The unit matters more than the method here. A pass is 1500 consecutive timed
frames; adjacent frames share cache state, branch-predictor state and whatever
the ``schedutil`` governor happened to be doing, so they are serially
correlated and a frame-unit bootstrap would resample 1500 near-copies and
return an interval far narrower than the data supports. SPEC §4 therefore
declares the pass as the unit, and that declaration is not negotiable: it is
asserted by ``tests/test_bootstrap.py`` and measured against a frame-unit
control in ``validation/validate_bootstrap_unit.py``.

Resampling is of pass IDENTITIES, jointly across all six stages. A pass is one
draw and it carries its whole 6-vector of stage medians with it, so the
stage-to-stage correlation that the divergence spread and the rank statistic
both depend on survives the resample. Resampling each stage independently would
produce an interval on the spread that is too narrow for the same reason a
frame-unit bootstrap is: it would throw away the dependence.

The percentile method is used. It is the honest choice for a ratio of medians
on 540 units: the statistic is not pivotal, a normal-theory interval on a ratio
is wrong in the tail, and BCa's acceleration estimate would need a jackknife
over the same correlated units. The interval is a percentile interval and is
labelled as one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: The declared resampling unit. SPEC §4. Asserted, not assumed.
RESAMPLING_UNIT = "pass"

#: Confidence level of every interval this package reports.
CONFIDENCE = 0.95

#: Bootstrap resamples. 4000 is enough for a 95 per cent percentile interval
#: whose endpoints are stable to about a part in a thousand on 540 units, and
#: it runs in well under a second on two cores.
RESAMPLES = 4000

#: Seed for every bootstrap in this package, so an interval is reproducible.
BOOTSTRAP_SEED = 94094


@dataclass(frozen=True)
class Interval:
    """A percentile bootstrap interval over the declared resampling unit."""

    point: float
    low: float
    high: float
    confidence: float
    resamples: int
    units: int
    unit: str = RESAMPLING_UNIT

    @property
    def width(self) -> float:
        """``high - low``, in the units of the statistic."""
        return self.high - self.low

    @property
    def relative_width(self) -> float:
        """Interval width as a fraction of the point estimate.

        ``nan`` when the point estimate is zero, rather than an infinity that
        would propagate silently into a report.
        """
        if self.point == 0.0:
            return float("nan")
        return self.width / abs(self.point)

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies within the closed interval."""
        return bool(self.low <= value <= self.high)

    def format(self, digits: int = 4) -> str:
        """``point [low, high]`` to ``digits`` decimal places."""
        return (
            f"{self.point:.{digits}f} "
            f"[{self.low:.{digits}f}, {self.high:.{digits}f}]"
        )


def unit_resamples(
    n_units: int,
    *,
    resamples: int = RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Index matrix of shape ``(resamples, n_units)``, drawn with replacement.

    One row is one resample of the declared unit. The same matrix is reused for
    every statistic in a report so that two intervals in the same table rest on
    the same draws.
    """
    if n_units < 2:
        raise ValueError(
            f"a bootstrap over {RESAMPLING_UNIT} units needs at least 2 units, "
            f"got {n_units}. A single unit has no sampling distribution."
        )
    if resamples < 1:
        raise ValueError(f"resamples must be positive, got {resamples}")
    generator = np.random.default_rng(seed)
    return generator.integers(0, n_units, size=(int(resamples), int(n_units)))


def percentile_interval(
    point: float,
    replicates: np.ndarray,
    *,
    confidence: float = CONFIDENCE,
    units: int,
    unit: str = RESAMPLING_UNIT,
) -> Interval:
    """Percentile interval from bootstrap replicates.

    Parameters
    ----------
    point:
        The statistic on the observed data, not the mean of the replicates.
    replicates:
        One statistic value per resample. ``nan`` replicates are dropped and
        the count of survivors is what the interval is taken over.
    confidence:
        Two-sided coverage, e.g. ``0.95``.
    units:
        Number of resampling units, recorded on the interval.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    values = np.asarray(replicates, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(
            "every bootstrap replicate is non-finite; no interval is reported "
            "rather than an interval of nan"
        )
    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(finite, [100.0 * tail, 100.0 * (1.0 - tail)])
    return Interval(
        point=float(point),
        low=float(low),
        high=float(high),
        confidence=float(confidence),
        resamples=int(finite.size),
        units=int(units),
        unit=unit,
    )


def bootstrap_statistic(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    confidence: float = CONFIDENCE,
    resamples: int = RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    unit: str = RESAMPLING_UNIT,
) -> Interval:
    """Interval for ``statistic`` over a one-dimensional array of unit values.

    ``values[i]`` is the contribution of unit ``i``. Each resample draws
    ``len(values)`` units with replacement and applies ``statistic`` to the
    drawn values.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    indices = unit_resamples(array.size, resamples=resamples, seed=seed)
    replicates = np.array([statistic(array[row]) for row in indices], dtype=np.float64)
    return percentile_interval(
        float(statistic(array)),
        replicates,
        confidence=confidence,
        units=array.size,
        unit=unit,
    )


def bootstrap_ratio_of_medians(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    confidence: float = CONFIDENCE,
    resamples: int = RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Interval:
    """Interval for ``median(numerator) / median(denominator)``.

    Both arrays are indexed by the same resampling unit and are resampled
    together, because the numerator and the denominator of a divergence ratio
    are two measurements of the same pass.
    """
    top = np.asarray(numerator, dtype=np.float64).ravel()
    bottom = np.asarray(denominator, dtype=np.float64).ravel()
    if top.shape != bottom.shape:
        raise ValueError(
            "numerator and denominator must be indexed by the same units: "
            f"got {top.shape} and {bottom.shape}"
        )
    indices = unit_resamples(top.size, resamples=resamples, seed=seed)

    def ratio(sample_top: np.ndarray, sample_bottom: np.ndarray) -> float:
        low = float(np.median(sample_bottom))
        if low == 0.0:
            return float("nan")
        return float(np.median(sample_top)) / low

    replicates = np.array(
        [ratio(top[row], bottom[row]) for row in indices], dtype=np.float64
    )
    return percentile_interval(
        ratio(top, bottom),
        replicates,
        confidence=confidence,
        units=top.size,
    )
