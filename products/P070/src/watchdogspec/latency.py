"""Closed-form worst-case detection and escalation arithmetic.

All times in milliseconds. Nothing here is measured; everything is computed
from the numbers a specification declares, which is the whole point of the
package: it checks a declaration against itself.

The model
---------
A supervised task emits a heartbeat nominally every ``P`` ms. The
specification asserts a bound ``J`` on how far the receipt-to-receipt gap may
deviate from ``P`` while the task is healthy, so the **beat window** is::

    W = P + J                                                        (1)

The supervisor arms a deadline ``W`` after each receipt. On each expiry with
no intervening receipt it arms the next deadline ``W`` later and increments a
miss counter; it declares the task faulty on the ``k + 1``-th consecutive
expiry, where ``k`` is the declared missed-beat tolerance. The last internal
expiry is therefore at::

    t_expire = t_ref + (k + 1) * W                                   (2)

where ``t_ref`` is the receipt of the last accepted heartbeat. The supervisor
only acts at its own poll instants, released every ``Q`` ms with release
jitter bounded by ``Jq`` (Audsley et al. 1993; Tindell and Clark 1994), and
then needs up to ``O`` ms to declare. Hence the **worst-case detection
latency**::

    WCDL = (k + 1) * (P + J) + Q + Jq + O                            (3)

and the **worst-case response latency** to the completion of rung ``r`` of
the escalation ladder::

    WCRL(r) = WCDL + sum_{i=1..r} latency_i                          (4)

Equations (1) to (4) are elementary composition of declared bounds. They are
not taken from a paper and they are not novel; the bounded-release model they
compose is the standard one cited above, and the general discipline of
deadline monitoring and escalation is covered in Burns, A. and Wellings, A.,
*Real-Time Systems and Programming Languages*, 4th edition, Addison-Wesley,
2009, ISBN 978-0-321-41745-9, and in Kopetz, H., *Real-Time Systems*, 2nd
edition, Springer, 2011, ISBN 978-1-4419-8236-0. The schedulability arithmetic
that decides whether the supervised tasks can meet their own periods in the
first place is a different question and is not addressed here: see Liu, C. L.
and Layland, J. W., "Scheduling Algorithms for Multiprogramming in a Hard-Real
-Time Environment", *Journal of the ACM*, vol. 20, no. 1, 1973, pp. 46-61,
doi:10.1145/321738.321743, and P036 RtClock in this portfolio.

Assumptions, all of which the arithmetic depends on
---------------------------------------------------
1. One supervisor, which does not itself fail and is not starved for longer
   than ``Jq``.
2. The declared jitter bound ``J`` holds. When the stochastic model in
   :mod:`watchdogspec.falsetrip` has ``sigma > 0`` it does **not** hold with
   probability one, and that module computes the rate at which it is violated.
   The two models are deliberately inconsistent and both are reported.
3. Heartbeats are not lost in transport beyond what ``J`` absorbs, and are not
   reordered.
4. The ladder is executed strictly in order, one rung at a time, and a rung's
   declared latency already includes any confirmation timeout before the next
   rung starts.
5. Every latency is measured from the same origin: the receipt of the last
   accepted heartbeat.

Validity range
--------------
``P > 0``, ``J >= 0``, ``k >= 0`` integer, ``D > 0``, ``Q > 0``, ``Jq >= 0``,
``O >= 0``, every rung latency ``>= 0``. Outside that range the functions
raise :class:`ValueError`. The arithmetic says nothing useful when ``J >= P``
(check ``WS008``) because the beat window then admits a reordering the model
does not describe.

Equation (3) is an **upper bound everywhere in that range, and tight only
while ``Jq <= Q``.** With a poll release jitter larger than the poll period,
the poll released one period later can execute earlier than the delayed next
poll, so the worst observation delay is ``2 Q`` rather than ``Q + Jq`` and the
closed form over-estimates. This was found by a property-based run against
the simulation, not assumed; the counterexample ``P = 1, J = 0, k = 0,
Q = 1, Jq = 2, O = 0`` (closed form 4.0 ms, attainable 3.0 ms) is pinned in
``tests/test_regression.py``. A supervisor whose poll is released later than
its own next release is outside the model in any case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .model import Ladder, Spec, Supervisor, Task

#: Default absolute tolerance, in ms, used when comparing a computed latency
#: against a declared deadline. A latency and a deadline that differ by less
#: than this are reported as :attr:`Margin.MARGINAL`, not as met or missed.
#: One nanosecond is far below any real supervision timescale and far above
#: the rounding of a sum of a dozen millisecond-scale doubles.
DEFAULT_TOL_MS = 1e-9

#: Relative part of the comparison band, in units in the last place of the
#: larger quantity being compared. An ABSOLUTE tolerance alone is not enough:
#: at a latency of 8.4e6 ms one ulp is already 1.9e-9 ms, so ``DEFAULT_TOL_MS``
#: falls below the rounding it exists to absorb and the three-valued verdict
#: stops being stable under a re-association of the same sum. That was found
#: by a property-based run at the dependency floor on 2026-10-16 and is pinned
#: in ``tests/test_regression.py``. Sixteen ulp comfortably covers the error of
#: a six-term sum, whose measured worst case on the validation grid is 1 ulp.
REL_GUARD_ULP = 16.0

#: One unit in the last place of 1.0 in IEEE 754 binary64.
_EPS = 2.220446049250313e-16


class Margin(Enum):
    """Verdict of comparing a latency against a deadline.

    Three values, not two, on purpose. ``MET`` and ``MISSED`` are produced by
    a strict comparison of a rounded floating-point sum against a declared
    threshold; at the threshold that comparison is not a property of the
    specification but of the last bit of the sum. ``MARGINAL`` is the band
    within ``tol_ms`` of the threshold where this package declines to decide,
    and callers are told so rather than handed a boolean that a different
    summation order would flip.
    """

    MET = "met"
    MARGINAL = "marginal"
    MISSED = "missed"


def comparison_band_ms(tol_ms: float = DEFAULT_TOL_MS, scale_ms: float = 0.0) -> float:
    """Half-width of the undecided band, in ms.

    ``max(tol_ms, REL_GUARD_ULP * eps * abs(scale_ms))``. ``scale_ms`` is the
    magnitude of the quantities being compared -- the latency or the deadline,
    whichever is larger. Passing ``0.0`` gives the absolute tolerance alone,
    which is correct only while that tolerance is above one unit in the last
    place of the quantities involved.
    """
    if not (tol_ms >= 0.0) or tol_ms == float("inf"):
        raise ValueError(f"tol_ms must be a finite non-negative number of ms, got {tol_ms!r}")
    if scale_ms != scale_ms or abs(scale_ms) == float("inf"):
        raise ValueError(f"scale_ms must be finite, got {scale_ms!r}")
    return max(tol_ms, REL_GUARD_ULP * _EPS * abs(scale_ms))


def verdict(
    slack_ms: float, tol_ms: float = DEFAULT_TOL_MS, scale_ms: float = 0.0
) -> Margin:
    """Classify a slack, in ms, as met, marginal or missed.

    ``slack_ms`` is ``deadline - latency``. Positive means time to spare.

    The result is :attr:`Margin.MARGINAL` whenever ``abs(slack_ms)`` does not
    exceed :func:`comparison_band_ms`. The band is exactly that expression in
    exact arithmetic; the slack itself is a floating-point sum and agrees with
    the exact value only to within rounding, which is the whole reason the band
    exists.

    Raises
    ------
    ValueError
        If ``tol_ms`` is negative or not finite, or ``scale_ms`` is not finite.
    """
    band = comparison_band_ms(tol_ms, scale_ms)
    if abs(slack_ms) <= band:
        return Margin.MARGINAL
    return Margin.MET if slack_ms > 0.0 else Margin.MISSED


def _require_positive(value: float, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{label} must be a real number of ms, got {type(value).__name__}")
    value = float(value)
    if not value > 0.0 or value != value or value == float("inf"):
        raise ValueError(f"{label} must be finite and strictly positive (ms), got {value!r}")
    return value


def _require_non_negative(value: float, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{label} must be a real number of ms, got {type(value).__name__}")
    value = float(value)
    if not value >= 0.0 or value != value or value == float("inf"):
        raise ValueError(f"{label} must be finite and non-negative (ms), got {value!r}")
    return value


def beat_window_ms(period_ms: float, jitter_ms: float) -> float:
    """Equation (1): ``W = P + J``, in ms.

    Parameters
    ----------
    period_ms:
        ``P``, strictly positive.
    jitter_ms:
        ``J``, non-negative.
    """
    return _require_positive(period_ms, "period_ms") + _require_non_negative(jitter_ms, "jitter_ms")


def worst_case_detection_latency_ms(
    period_ms: float,
    jitter_ms: float,
    missed_beats_tolerated: int,
    poll_period_ms: float,
    poll_jitter_ms: float = 0.0,
    declaration_overhead_ms: float = 0.0,
) -> float:
    """Equation (3): worst-case detection latency, in ms.

    Measured from the receipt of the last accepted heartbeat to the instant
    the supervisor declares the task faulty.

    The result is exact in exact arithmetic and agrees with the exact value to
    within floating-point rounding of a sum of six terms. Compare it against a
    deadline with :func:`verdict`, never with ``<=``.

    Raises
    ------
    ValueError
        On a non-positive period or poll period, a negative jitter, overhead or
        tolerance, or a non-finite input.
    TypeError
        If ``missed_beats_tolerated`` is not an integer.
    """
    if isinstance(missed_beats_tolerated, bool) or not isinstance(missed_beats_tolerated, int):
        raise TypeError(
            "missed_beats_tolerated must be an int, got "
            f"{type(missed_beats_tolerated).__name__}"
        )
    if missed_beats_tolerated < 0:
        raise ValueError(
            f"missed_beats_tolerated must be >= 0, got {missed_beats_tolerated}"
        )
    window = beat_window_ms(period_ms, jitter_ms)
    poll = _require_positive(poll_period_ms, "poll_period_ms")
    poll_jit = _require_non_negative(poll_jitter_ms, "poll_jitter_ms")
    overhead = _require_non_negative(declaration_overhead_ms, "declaration_overhead_ms")
    return (missed_beats_tolerated + 1) * window + poll + poll_jit + overhead


@dataclass(frozen=True, slots=True)
class RungTiming:
    """Timing of one rung, all in ms, measured from the last accepted heartbeat."""

    index: int
    action: str
    start_ms: float
    finish_ms: float
    slack_ms: float
    margin: Margin
    cumulative_cost: float


@dataclass(frozen=True, slots=True)
class TaskTiming:
    """Everything the closed form says about one task. Times in ms.

    Attributes
    ----------
    detection_latency_ms:
        ``WCDL`` of equation (3).
    tolerated_silence_ms:
        ``k * W``: the longest silence the supervisor treats as healthy.
    full_ladder_latency_ms:
        ``WCRL`` to the completion of the last rung, equation (4).
    detection_slack_ms:
        ``D - WCDL``. Positive means detection alone fits inside the deadline.
    ladder_slack_ms:
        ``D - WCRL(last)``. Positive means the whole ladder fits.
    deepest_rung_reachable:
        Largest ``r`` whose completion is not :attr:`Margin.MISSED` against the
        deadline; ``0`` when even detection misses it. ``len(rungs)`` when the
        whole ladder fits.
    """

    task_id: str
    beat_window_ms: float
    tolerated_silence_ms: float
    detection_latency_ms: float
    deadline_ms: float
    detection_slack_ms: float
    detection_margin: Margin
    rungs: tuple[RungTiming, ...]
    full_ladder_latency_ms: float
    ladder_slack_ms: float
    ladder_margin: Margin
    deepest_rung_reachable: int
    total_cost: float
    tolerance_slack_ms: float
    tolerance_margin: Margin


def task_timing(
    task: Task,
    supervisor: Supervisor,
    ladder: Ladder,
    tol_ms: float = DEFAULT_TOL_MS,
) -> TaskTiming:
    """Apply equations (1) to (4) to one task and its ladder.

    Parameters
    ----------
    task, supervisor, ladder:
        Declared objects. ``ladder`` must be the one ``task.escalation`` names;
        this function does not look it up.
    tol_ms:
        Absolute tolerance in ms for every deadline comparison. See
        :func:`verdict`.

    Returns
    -------
    TaskTiming
        Latencies, slacks and a three-valued margin for each comparison.

    Raises
    ------
    ValueError
        On any declared quantity outside the validity range documented in the
        module docstring.
    """
    detection = worst_case_detection_latency_ms(
        task.heartbeat_period_ms,
        task.heartbeat_jitter_ms,
        task.missed_beats_tolerated,
        supervisor.poll_period_ms,
        supervisor.poll_jitter_ms,
        supervisor.declaration_overhead_ms,
    )
    deadline = _require_positive(task.liveness_deadline_ms, "liveness_deadline_ms")
    window = task.beat_window_ms
    silence = task.tolerated_silence_ms

    rungs: list[RungTiming] = []
    running = detection
    cost = 0.0
    for index, rung in enumerate(ladder.rungs, start=1):
        latency = _require_non_negative(rung.latency_ms, f"ladder[{ladder.id}].rungs[{index}]")
        start = running
        running = running + latency
        cost += rung.recovery_cost
        slack = deadline - running
        rungs.append(
            RungTiming(
                index=index,
                action=rung.action.value,
                start_ms=start,
                finish_ms=running,
                slack_ms=slack,
                margin=verdict(slack, tol_ms, max(abs(deadline), abs(running))),
                cumulative_cost=cost,
            )
        )

    deepest = 0
    for timing in rungs:
        if timing.margin is Margin.MISSED:
            break
        deepest = timing.index

    detection_slack = deadline - detection
    ladder_slack = deadline - running
    # Check WS002 compares the tolerated silence against the deadline. A
    # tolerance of zero gives silence 0 and slack = D > 0, which is right:
    # tolerating no missed beat cannot be the thing that breaks the deadline.
    tolerance_slack = deadline - silence

    return TaskTiming(
        task_id=task.id,
        beat_window_ms=window,
        tolerated_silence_ms=silence,
        detection_latency_ms=detection,
        deadline_ms=deadline,
        detection_slack_ms=detection_slack,
        detection_margin=verdict(detection_slack, tol_ms, max(abs(deadline), abs(detection))),
        rungs=tuple(rungs),
        full_ladder_latency_ms=running,
        ladder_slack_ms=ladder_slack,
        ladder_margin=verdict(ladder_slack, tol_ms, max(abs(deadline), abs(running))),
        deepest_rung_reachable=deepest,
        total_cost=cost,
        tolerance_slack_ms=tolerance_slack,
        tolerance_margin=verdict(tolerance_slack, tol_ms, max(abs(deadline), abs(silence))),
    )


def spec_timing(spec: Spec, tol_ms: float = DEFAULT_TOL_MS) -> dict[str, TaskTiming]:
    """Timing for every task whose escalation ladder is declared.

    Tasks naming an undeclared ladder are omitted; check ``WS012`` reports them.
    """
    out: dict[str, TaskTiming] = {}
    for task in spec.tasks:
        ladder = spec.ladder(task.escalation)
        if ladder is None:
            continue
        out[task.id] = task_timing(task, spec.supervisor, ladder, tol_ms=tol_ms)
    return out


def max_tolerance_for_deadline(
    period_ms: float,
    jitter_ms: float,
    deadline_ms: float,
    poll_period_ms: float,
    poll_jitter_ms: float = 0.0,
    declaration_overhead_ms: float = 0.0,
    ladder_latency_ms: float = 0.0,
) -> int:
    """Largest ``k`` for which the whole ladder still completes within ``D``.

    Solves equation (4) for ``k``::

        (k + 1) * W + Q + Jq + O + sum(latency_i) <= D
        k <= (D - Q - Jq - O - sum(latency_i)) / W  -  1

    Returns ``-1`` when no non-negative ``k`` works, which means the deadline
    cannot be met even with the supervisor declaring on the first missed beat.
    The floor is taken after subtracting a relative guard of
    ``4 * eps * D`` so that a ``k`` which only fits by one bit of rounding is
    not reported as fitting.
    """
    window = beat_window_ms(period_ms, jitter_ms)
    deadline = _require_positive(deadline_ms, "deadline_ms")
    budget = (
        deadline
        - _require_positive(poll_period_ms, "poll_period_ms")
        - _require_non_negative(poll_jitter_ms, "poll_jitter_ms")
        - _require_non_negative(declaration_overhead_ms, "declaration_overhead_ms")
        - _require_non_negative(ladder_latency_ms, "ladder_latency_ms")
    )
    guard = 4.0 * 2.220446049250313e-16 * abs(deadline)
    value = math.floor((budget - guard) / window) - 1
    return max(value, -1)
