"""Bootstrap intervals whose resampling unit is the HANDOVER EVENT, declared and asserted.

WHY THE UNIT IS THE EVENT AND NOT THE SLOT
------------------------------------------

The ground-truth decomposition is accumulated slot by slot over a simulated
handover. Slots within one handover are not independent: the occupancy at the
start of a slot is the occupancy at the end of the previous one, and once the
buffer is full it tends to stay full, so consecutive slots drop together. A
bootstrap that resampled slots would treat those dependent observations as
independent and would report an interval narrower than the data support — the
classic consequence of resampling below the unit of independence (Davison, A. C.
and Hinkley, D. V., *Bootstrap Methods and their Application*, Cambridge
University Press, 1997, section 8.2; Field, C. A. and Welsh, A. H.,
"Bootstrapping clustered data", *Journal of the Royal Statistical Society,
Series B*, volume 69, number 3, pages 369-390, 2007).

Handover EVENTS are independent by construction of the corpus: each draws its
own acquisition time, restart latency, burst sequence and jitter sequence from
independent streams. The event is therefore the unit, and this module refuses to
produce an interval without being told so:
:class:`BootstrapInterval` carries ``resampling_unit`` and asserts it equals
:data:`RESAMPLING_UNIT` at construction, and :func:`event_bootstrap` is the only
function here that can build one.

The same statement appears in ``validation/VALIDATION.md`` under "Resampling
unit", because a reader of an interval cannot infer the unit from the number.

METHOD
------

Non-parametric percentile bootstrap over events (Efron, B. and Tibshirani, R. J.,
*An Introduction to the Bootstrap*, Chapman and Hall, 1993, chapter 13). For
``R`` resamples, draw ``n`` event indices with replacement, recompute the
statistic on the resampled events, and take the ``alpha/2`` and ``1-alpha/2``
empirical percentiles of the ``R`` values. ``numpy.percentile``'s default linear
interpolation is used and is part of what the numpy ceiling in
``pyproject.toml`` pins.

The percentile bootstrap is used rather than BCa because the statistics here are
sums and means of non-negative per-event volumes, for which the percentile
interval is adequate and the acceleration estimate would add a jackknife pass
over the corpus for no change a reader of this product would act on. That is a
declared choice and a limitation, recorded in ``README.md``.

VALIDITY RANGE
--------------

Percentile intervals on a heavily skewed statistic from a small number of events
undercover. The corpus here has of the order of a thousand events per split and
the coverage actually achieved is measured by
``validation/validate_negative_controls.py`` rather than assumed. Below about 30
events in a group this module still returns an interval and
:attr:`BootstrapInterval.n_events` is how a caller sees that it is thin;
:func:`event_bootstrap` raises below 5.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: The one resampling unit this module supports. Asserted, not documented only.
RESAMPLING_UNIT = "handover event"


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile bootstrap interval with its resampling unit on the object.

    Attributes
    ----------
    point:
        The statistic on the observed events.
    low, high:
        Lower and upper percentile bounds at :attr:`confidence`.
    confidence:
        Nominal coverage, e.g. 0.95.
    resampling_unit:
        Must equal :data:`RESAMPLING_UNIT`. Asserted at construction.
    n_events:
        Number of independent units resampled.
    n_resamples:
        Number of bootstrap replicates.
    """

    point: float
    low: float
    high: float
    confidence: float
    resampling_unit: str
    n_events: int
    n_resamples: int

    def __post_init__(self) -> None:
        if self.resampling_unit != RESAMPLING_UNIT:
            raise ValueError(
                "BootstrapInterval.resampling_unit must be "
                f"{RESAMPLING_UNIT!r}, got {self.resampling_unit!r}. The unit of "
                "independence for a handover trace is the handover event, not the "
                "accounting slot: slots within one handover share an occupancy state "
                "and resampling them would report an interval narrower than the data "
                "support."
            )
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(
                f"confidence must lie in (0, 1), got {self.confidence!r}"
            )
        if self.n_events < 1:
            raise ValueError(f"n_events must be at least 1, got {self.n_events!r}")
        if self.n_resamples < 1:
            raise ValueError(
                f"n_resamples must be at least 1, got {self.n_resamples!r}"
            )
        if self.low > self.high:
            raise ValueError(
                f"low ({self.low!r}) exceeds high ({self.high!r}); the percentiles "
                "were taken in the wrong order"
            )

    @property
    def width(self) -> float:
        """``high - low``, in the statistic's own units."""
        return self.high - self.low

    def excludes(self, value: float) -> bool:
        """Whether ``value`` lies strictly outside ``[low, high]``.

        A strict comparison on a floating-point bound. Callers that need a
        decision near a bound must look at :attr:`low` and :attr:`high`
        themselves; this package never asserts the BOOLEAN returned here as an
        invariant, only the bounds to a tolerance.
        """
        return value < self.low or value > self.high

    def disjoint_from(self, other: BootstrapInterval) -> bool:
        """Whether this interval and ``other`` do not overlap. Same caveat as above."""
        return self.high < other.low or other.high < self.low


def event_bootstrap(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    resamples: int = 4000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BootstrapInterval:
    """Percentile bootstrap of ``statistic`` over the EVENT axis of ``values``.

    Parameters
    ----------
    values:
        Per-event observations. The FIRST axis is the event axis and is the only
        axis resampled; further axes (for example the three mechanism columns)
        are carried along so that a statistic may be a function of the whole
        per-event row.
    statistic:
        Maps a resampled ``values``-shaped array to one float.
    resamples:
        Number of replicates, at least 1. 4000 is this package's default and is
        what every number in ``validation/`` was produced with unless its own
        output file says otherwise.
    confidence:
        Nominal coverage in ``(0, 1)``.
    rng:
        Generator for the resample indices. ``None`` uses a fixed seed so a
        caller who does not supply one still gets a reproducible interval.

    Returns
    -------
    BootstrapInterval
        With :data:`RESAMPLING_UNIT` recorded on it.

    Raises
    ------
    ValueError
        Below 5 events, on a non-positive ``resamples``, or on a ``confidence``
        outside ``(0, 1)``. Five is the point below which a percentile interval
        over events is a statement about at most five numbers and this package
        declines to present it as an interval.
    """
    observations = np.asarray(values, dtype=float)
    if observations.ndim == 0:
        raise ValueError("values must have at least one axis, the event axis")
    count = int(observations.shape[0])
    if count < 5:
        raise ValueError(
            f"event_bootstrap needs at least 5 events, got {count}. Below that a "
            "percentile interval is a statement about at most five numbers."
        )
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")

    generator = np.random.default_rng(96_096) if rng is None else rng
    replicates = np.empty(resamples, dtype=float)
    for index in range(resamples):
        draw = generator.integers(0, count, size=count)
        replicates[index] = float(statistic(observations[draw]))

    alpha = 1.0 - confidence
    low, high = np.percentile(replicates, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return BootstrapInterval(
        point=float(statistic(observations)),
        low=float(low),
        high=float(high),
        confidence=float(confidence),
        resampling_unit=RESAMPLING_UNIT,
        n_events=count,
        n_resamples=int(resamples),
    )
