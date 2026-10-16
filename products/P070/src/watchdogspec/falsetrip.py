"""False-trip rate from the declared jitter statistics.

A **false trip** is the supervisor declaring a task faulty while the task is
in fact alive and emitting heartbeats. It costs whatever the first rung of the
escalation ladder costs, and on a vehicle a spurious safe-mode entry is not
free. The rate is computable from two numbers the specification already has to
declare: the jitter bound ``J`` and the jitter standard deviation ``sigma``.

The model
---------
Heartbeat ``n`` is emitted at ``n * P + d_n`` with ``d_n`` independent and
identically distributed ``N(0, sigma^2)`` ms. The gap between consecutive
receipts is then::

    gap_n = P + (d_n - d_{n-1}) = P + Delta_n,   Delta_n ~ N(0, 2 sigma^2)   (5)

The supervisor (see :mod:`watchdogspec.latency`) arms deadlines every
``W = P + J`` after the last receipt and declares on the ``k + 1``-th expiry.
A healthy beat therefore causes a declaration exactly when::

    gap_n > (k + 1) * W
    <=> Delta_n > k * P + (k + 1) * J                                        (6)

so the per-gap false-trip probability is the upper tail of a centred normal
with standard deviation ``sigma * sqrt(2)``::

    p = Phi_bar( (k P + (k+1) J) / (sigma sqrt(2)) )
      = 0.5 * erfc( (k P + (k+1) J) / (2 sigma) )                            (7)

``Phi_bar`` is the standard normal upper tail; the ``erfc`` form is the
standard identity ``Phi_bar(x) = 0.5 erfc(x / sqrt(2))`` (Abramowitz, M. and
Stegun, I. A., *Handbook of Mathematical Functions*, National Bureau of
Standards Applied Mathematics Series 55, 1964, section 26.2). It is evaluated
with :func:`math.erfc`, which is accurate in the far tail where
``1 - Phi(x)`` computed by subtraction is not.

Rates follow from the beat rate ``3600000 / P`` beats per hour::

    false_trips_per_hour = p * 3600000 / P                                   (8)
    mean_time_between_false_trips_hours = 1 / false_trips_per_hour           (9)

Assumptions
-----------
1. Emission deviations are independent, identically distributed and Gaussian.
   Real heartbeat jitter is usually neither Gaussian nor independent: a
   scheduler overload delays several beats together, which makes the true rate
   **higher** than equation (7) at the same ``sigma``. Equation (7) is not a
   bound; it is the rate implied by the declared statistics.
2. Each gap is treated as an independent Bernoulli trial. The deadline chain
   resets on every receipt, so this is exact for the first false trip and a
   rare-event approximation for the long-run rate; it is accurate to within a
   relative ``O(p)`` and ``p`` is normally well below ``1e-3``.
3. ``sigma = 0`` means the author declares the jitter bound to be hard. The
   rate is then exactly zero, and that is a statement about the declaration,
   not about the system.
4. The bounded model of :mod:`watchdogspec.latency` (``|deviation| <= J``) and
   the Gaussian model here are **mutually inconsistent on purpose**. The first
   gives the worst case if the bound holds; the second gives the rate at which
   it does not. A specification that declares both is declaring a bound and a
   violation rate for it, and this package reports both numbers side by side
   rather than pretending one supersedes the other.

Validity range
--------------
``P > 0``, ``J >= 0``, ``sigma >= 0``, ``k >= 0`` integer. Equation (7)
underflows to ``0.0`` in IEEE double for arguments beyond about ``26.6``
standard deviations, which corresponds to ``p`` below roughly ``1e-308``; the
functions report such a value as ``0.0`` and callers should read it as
"smaller than this arithmetic can represent", not as "impossible".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model import Spec, Task

#: Milliseconds in an hour. Used by equation (8).
MS_PER_HOUR = 3_600_000.0

#: Below this probability the double-precision tail is not informative and the
#: reported mean time between false trips is ``inf``.
UNDERFLOW_P = 1e-300


@dataclass(frozen=True, slots=True)
class FalseTripRate:
    """Result of equations (7) to (9) for one task.

    Attributes
    ----------
    per_gap:
        ``p`` of equation (7), dimensionless probability per heartbeat gap.
    per_hour:
        Equation (8), expected false trips per hour of healthy operation.
    mtbft_hours:
        Equation (9), ``inf`` when ``per_hour`` is zero or underflows.
    z:
        ``(k P + (k+1) J) / (sigma sqrt(2))``, the tail argument in standard
        deviations of the gap. ``inf`` when ``sigma == 0``.
    threshold_ms:
        ``k P + (k+1) J``, the gap excess over ``P`` that trips the supervisor.
    exact:
        ``True`` when ``sigma == 0`` and the rate is exactly zero by
        declaration rather than by evaluation of a tail.
    """

    per_gap: float
    per_hour: float
    mtbft_hours: float
    z: float
    threshold_ms: float
    exact: bool


def normal_upper_tail(z: float) -> float:
    """``Phi_bar(z)``, the standard normal upper tail, via ``erfc``.

    Accurate in the far tail; see the module docstring for the identity and
    its source. Agrees with the exact value to within the accuracy of
    :func:`math.erfc`, which is of order one unit in the last place over the
    representable range.
    """
    if z != z:
        raise ValueError("z must not be NaN")
    if z == float("inf"):
        return 0.0
    if z == float("-inf"):
        return 1.0
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def false_trip_probability(
    period_ms: float,
    jitter_ms: float,
    jitter_sigma_ms: float,
    missed_beats_tolerated: int,
) -> float:
    """Equation (7): false-trip probability per heartbeat gap, dimensionless.

    Returns exactly ``0.0`` when ``jitter_sigma_ms == 0``, which is the
    declaration "the bound is hard", not a computed tail.

    Raises
    ------
    ValueError
        On a non-positive period, a negative jitter or sigma, a negative
        tolerance, or a non-finite input.
    """
    if not period_ms > 0.0 or period_ms != period_ms or period_ms == float("inf"):
        raise ValueError(f"period_ms must be finite and strictly positive, got {period_ms!r}")
    for label, value in (("jitter_ms", jitter_ms), ("jitter_sigma_ms", jitter_sigma_ms)):
        if not value >= 0.0 or value != value or value == float("inf"):
            raise ValueError(f"{label} must be finite and non-negative, got {value!r}")
    if isinstance(missed_beats_tolerated, bool) or not isinstance(missed_beats_tolerated, int):
        raise TypeError("missed_beats_tolerated must be an int")
    if missed_beats_tolerated < 0:
        raise ValueError("missed_beats_tolerated must be >= 0")
    if jitter_sigma_ms == 0.0:
        return 0.0
    k = missed_beats_tolerated
    threshold = k * period_ms + (k + 1) * jitter_ms
    return 0.5 * math.erfc(threshold / (2.0 * jitter_sigma_ms))


def false_trip_rate(task: Task) -> FalseTripRate:
    """Equations (7) to (9) for one declared task.

    The rate is per hour of **healthy** operation: it counts declarations the
    supervisor would make about a task that never actually failed.
    """
    k = task.missed_beats_tolerated
    threshold = k * task.heartbeat_period_ms + (k + 1) * task.heartbeat_jitter_ms
    sigma = task.jitter_sigma_ms
    if sigma == 0.0:
        return FalseTripRate(
            per_gap=0.0,
            per_hour=0.0,
            mtbft_hours=float("inf"),
            z=float("inf"),
            threshold_ms=threshold,
            exact=True,
        )
    z = threshold / (sigma * math.sqrt(2.0))
    p = 0.5 * math.erfc(threshold / (2.0 * sigma))
    per_hour = p * MS_PER_HOUR / task.heartbeat_period_ms
    mtbft = float("inf") if per_hour <= 0.0 or p < UNDERFLOW_P else 1.0 / per_hour
    return FalseTripRate(
        per_gap=p,
        per_hour=per_hour,
        mtbft_hours=mtbft,
        z=z,
        threshold_ms=threshold,
        exact=False,
    )


def spec_false_trip_rates(spec: Spec) -> dict[str, FalseTripRate]:
    """Equation (7) to (9) for every task in the specification."""
    return {task.id: false_trip_rate(task) for task in spec.tasks}
