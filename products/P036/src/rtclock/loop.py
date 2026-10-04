"""A fixed-rate loop driver that reports drift instead of absorbing it.

The two ways to wait
--------------------
A loop that must run at period ``T`` can compute its wait in two ways, and
the choice decides whether timing error accumulates:

``mode="absolute"``
    Wait until the *absolute* release instant ``t0 + k*T``, recomputed from
    the clock each iteration. Error in one wait is corrected by the next, so
    drift is bounded and does not grow with ``k``.

``mode="relative"``
    Wait a *fixed* ``T`` after the previous release, the pattern a bare
    ``time.sleep(T)`` inside a ``while`` loop produces. Every wait's error is
    added to the total, so drift grows linearly in ``k``.

Drift is defined identically in both modes, against the ideal schedule::

    scheduled_k = t0 + k * T          (ideal release instant, units s)
    drift_k     = start_k - scheduled_k   (units s, positive = late)

With a sleep primitive of fractional rate error ``eps`` and constant wake-up
delay ``d`` (see :class:`rtclock.timebase.SkewedSimulatedTimebase`) and a
zero-cost body, both modes have closed-form drift. Writing ``a = T*eps + d``:

    relative:  drift_k = k * a
    absolute:  drift_k = a * (1 - (-eps)**k) / (1 + eps),  k >= 1

The absolute-mode expression is bounded by ``|a| / (1 - |eps|)`` for
``|eps| < 1`` and tends to ``a / (1 + eps)``; the relative-mode expression is
unbounded. Both are derived by unrolling the recurrence
``wait_k = T - drift_(k-1)`` and are checked against the implementation in
``validation/validate_loop_drift.py``. Validity: ``|eps| < 1`` and every wait
positive, i.e. ``drift_(k-1) < T``.

Nothing here makes CPython on a general-purpose OS a real-time platform. The
loop driver *measures and reports* timing error; it does not reduce it. On
the host clock the reported numbers carry the clock's measured resolution as
an error term -- see :func:`rtclock.timebase.measure_clock_resolution`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .histogram import LatencyHistogram, OverrunReport, overrun_report
from .timebase import Timebase

ScheduleMode = Literal["absolute", "relative"]

__all__ = [
    "FixedRateLoop",
    "IterationRecord",
    "LoopReport",
    "ScheduleMode",
    "absolute_mode_drift",
    "relative_mode_drift",
]


def relative_mode_drift(index: int, period_s: float, skew_ppm: float, wake_delay_s: float) -> float:
    """Closed-form drift of relative-mode scheduling: ``k * (T*eps + d)``, units s.

    Args:
        index: iteration index ``k``, >= 0.
        period_s: nominal period ``T``, units s, > 0.
        skew_ppm: sleep-primitive rate error, units ppm.
        wake_delay_s: constant per-wait latency ``d``, units s.

    Returns:
        Expected drift, units s.

    Raises:
        ValueError: if ``index < 0`` or ``period_s <= 0``.
    """
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")
    if period_s <= 0.0:
        raise ValueError(f"period_s must be > 0 s, got {period_s}")
    return index * (period_s * skew_ppm * 1.0e-6 + wake_delay_s)


def absolute_mode_drift(index: int, period_s: float, skew_ppm: float, wake_delay_s: float) -> float:
    """Closed-form drift of absolute-mode scheduling, units s.

    ``drift_0 = 0`` and ``drift_k = a (1 - (-eps)^k) / (1 + eps)`` for
    ``k >= 1``, with ``a = T*eps + d`` and ``eps = skew_ppm * 1e-6``.

    Args:
        index: iteration index ``k``, >= 0.
        period_s: nominal period ``T``, units s, > 0.
        skew_ppm: sleep-primitive rate error, units ppm. Requires
            ``|eps| < 1``, i.e. ``|skew_ppm| < 1e6``.
        wake_delay_s: constant per-wait latency ``d``, units s.

    Returns:
        Expected drift, units s.

    Raises:
        ValueError: if ``index < 0``, ``period_s <= 0`` or ``|skew_ppm| >= 1e6``.
    """
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")
    if period_s <= 0.0:
        raise ValueError(f"period_s must be > 0 s, got {period_s}")
    eps = skew_ppm * 1.0e-6
    if abs(eps) >= 1.0:
        raise ValueError(f"|skew_ppm| must be < 1e6 for the closed form, got {skew_ppm}")
    if index == 0:
        return 0.0
    a = period_s * eps + wake_delay_s
    return a * (1.0 - (-eps) ** index) / (1.0 + eps)


@dataclass(frozen=True)
class IterationRecord:
    """One loop iteration, all times in seconds on the driver's timebase.

    Attributes:
        index: zero-based iteration index ``k``.
        scheduled_s: ideal release instant ``t0 + k*T``.
        start_s: clock reading taken immediately before the body ran.
        end_s: clock reading taken immediately after the body returned.
        drift_s: ``start_s - scheduled_s``; positive means late.
        duration_s: ``end_s - start_s``; the body's measured execution time.
        wait_s: the wait requested before this iteration. Negative means the
            loop was already past its release instant and did not sleep.
    """

    index: int
    scheduled_s: float
    start_s: float
    end_s: float
    drift_s: float
    duration_s: float
    wait_s: float

    @property
    def completion_s(self) -> float:
        """Alias of ``end_s``, for reading against a deadline."""
        return self.end_s

    @property
    def late_release(self) -> bool:
        """True when the requested wait was negative, i.e. the release slipped."""
        return self.wait_s < 0.0


@dataclass(frozen=True)
class LoopReport:
    """Result of a :meth:`FixedRateLoop.run`.

    Attributes:
        records: one :class:`IterationRecord` per iteration, in order.
        period_s: nominal period, units s.
        mode: the scheduling mode used.
        deadline_s: relative deadline each iteration was judged against,
            units s. Defaults to ``period_s``.
        start_s: the loop's ``t0``, units s.
        clock_quantum_s: the timebase's measured resolution if one was
            supplied, units s, else ``None``. Carried so that every derived
            number can be quoted with its error term.
    """

    records: tuple[IterationRecord, ...]
    period_s: float
    mode: ScheduleMode
    deadline_s: float
    start_s: float
    clock_quantum_s: float | None = None

    @property
    def drifts_s(self) -> tuple[float, ...]:
        """Per-iteration drift, units s."""
        return tuple(r.drift_s for r in self.records)

    @property
    def durations_s(self) -> tuple[float, ...]:
        """Per-iteration body execution time, units s."""
        return tuple(r.duration_s for r in self.records)

    @property
    def max_abs_drift_s(self) -> float:
        """Largest ``|drift|`` over the run, units s. 0.0 for an empty run."""
        return max((abs(r.drift_s) for r in self.records), default=0.0)

    @property
    def final_drift_s(self) -> float:
        """Drift of the last iteration, units s. 0.0 for an empty run."""
        return self.records[-1].drift_s if self.records else 0.0

    @property
    def late_releases(self) -> int:
        """Number of iterations whose wait was negative."""
        return sum(1 for r in self.records if r.late_release)

    def drift_slope_s_per_iteration(self) -> float:
        """Least-squares slope of drift against iteration index, units s/iteration.

        A slope indistinguishable from zero is the signature of absolute-mode
        scheduling; a non-zero slope is the accumulated rate error of
        relative-mode scheduling. Computed with ``math.fsum`` over the exact
        index values, no NumPy, so the number does not depend on a BLAS.

        Raises:
            ValueError: if fewer than two iterations were run.
        """
        n = len(self.records)
        if n < 2:
            raise ValueError("drift slope needs at least 2 iterations")
        xs = [float(r.index) for r in self.records]
        ys = [r.drift_s for r in self.records]
        mx = math.fsum(xs) / n
        my = math.fsum(ys) / n
        sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
        sxx = math.fsum((x - mx) ** 2 for x in xs)
        return sxy / sxx

    def duration_histogram(self) -> LatencyHistogram:
        """Body execution times as a :class:`~rtclock.histogram.LatencyHistogram`."""
        h = LatencyHistogram(label="body duration")
        h.extend(self.durations_s)
        return h

    def overruns(self, budget_s: float | None = None) -> OverrunReport:
        """Overrun accounting on body duration against ``budget_s``.

        Defaults to ``deadline_s``. Strict ``>``; see
        :func:`rtclock.histogram.overrun_report`.
        """
        return overrun_report(self.durations_s, self.deadline_s if budget_s is None else budget_s)


class FixedRateLoop:
    """Drive a callable at a fixed period and record what the clock says.

    Args:
        period_s: nominal period ``T``, units s, must be > 0.
        timebase: any :class:`~rtclock.timebase.Timebase`.
        mode: ``"absolute"`` (default, self-correcting) or ``"relative"``.
        deadline_s: relative deadline for overrun accounting, units s.
            Defaults to ``period_s``; must be > 0 and <= ``period_s``.

    Raises:
        ValueError: on a non-positive period, an unknown mode, or a deadline
            outside ``(0, period_s]``.
        TypeError: if ``timebase`` does not provide ``now`` and ``sleep``.
    """

    def __init__(
        self,
        period_s: float,
        timebase: Timebase,
        mode: ScheduleMode = "absolute",
        deadline_s: float | None = None,
    ) -> None:
        if not math.isfinite(float(period_s)) or float(period_s) <= 0.0:
            raise ValueError(f"period_s must be a finite value > 0 s, got {period_s}")
        if mode not in ("absolute", "relative"):
            raise ValueError(f"mode must be 'absolute' or 'relative', got {mode!r}")
        has_now = callable(getattr(timebase, "now", None))
        has_sleep = callable(getattr(timebase, "sleep", None))
        if not (has_now and has_sleep):
            raise TypeError("timebase must provide now() and sleep()")
        self.period_s = float(period_s)
        self.timebase = timebase
        self.mode: ScheduleMode = mode
        if deadline_s is None:
            self.deadline_s = self.period_s
        else:
            d = float(deadline_s)
            if not math.isfinite(d) or d <= 0.0 or d > self.period_s:
                raise ValueError(
                    f"deadline_s must be in (0, period_s = {self.period_s}] s, got {deadline_s}"
                )
            self.deadline_s = d

    def run(
        self,
        body: Callable[[int], None] | None = None,
        iterations: int = 100,
        clock_quantum_s: float | None = None,
    ) -> LoopReport:
        """Run ``iterations`` iterations and return the timing record.

        Iteration ``k`` is released at the ideal instant ``t0 + k*T``, where
        ``t0`` is the clock reading at entry. The first iteration is released
        immediately, so ``drift_0`` is exactly 0 by construction.

        Args:
            body: called as ``body(k)``. ``None`` means a zero-cost body,
                which is the configuration used to isolate clock behaviour.
            iterations: number of iterations, >= 0.
            clock_quantum_s: measured clock resolution to record in the
                report, units s. Pass
                ``MonotonicTimebase().resolution.measured_tick_s`` when
                running on the host clock so the report carries its own error
                term.

        Returns:
            A :class:`LoopReport`.

        Raises:
            ValueError: if ``iterations < 0``.
        """
        if iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {iterations}")
        t0 = self.timebase.now()
        records: list[IterationRecord] = []
        for k in range(iterations):
            scheduled = t0 + k * self.period_s
            if k == 0:
                wait = 0.0
            elif self.mode == "absolute":
                wait = scheduled - self.timebase.now()
            else:
                wait = self.period_s
            if wait > 0.0:
                self.timebase.sleep(wait)
            start = self.timebase.now()
            if body is not None:
                body(k)
            end = self.timebase.now()
            records.append(
                IterationRecord(
                    index=k,
                    scheduled_s=scheduled,
                    start_s=start,
                    end_s=end,
                    drift_s=start - scheduled,
                    duration_s=end - start,
                    wait_s=wait,
                )
            )
        return LoopReport(
            records=tuple(records),
            period_s=self.period_s,
            mode=self.mode,
            deadline_s=self.deadline_s,
            start_s=t0,
            clock_quantum_s=clock_quantum_s,
        )
