"""Timebases: a monotonic wall-clock one and a deterministic virtual one.

Units
-----
Every timestamp and duration in :mod:`hilforge` is a ``float`` in **seconds**.
Nanosecond integers are used internally where exactness matters
(:class:`MonotonicTimebase` reads ``time.perf_counter_ns``) and converted once
at the boundary.

Clock resolution as an error term
---------------------------------
A duration is a difference of two quantised timestamps. If the clock reports
in steps of ``delta`` seconds, each timestamp carries a quantisation error
that is modelled as uniform on ``[-delta/2, +delta/2]``, whose variance is
``delta**2 / 12`` (Bennett 1948, "Spectra of Quantized Signals", *Bell System
Technical Journal* 27(3):446-472, §2 — the standard uniform-quantisation
result). Two independent timestamps give a duration variance of
``2 * delta**2 / 12 = delta**2 / 6``, i.e. a standard uncertainty of

    u_res = delta / sqrt(6)                                           [s]

This is a *lower bound* on duration uncertainty, not the whole budget: the
cost of the timing calls themselves is a bias, measured separately by
:func:`measure_call_overhead`. Combination of the two follows JCGM 100:2008
(*Evaluation of measurement data — Guide to the expression of uncertainty in
measurement*), §5.1.2: independent terms add in quadrature.

Validity range
--------------
The quantisation model assumes ``delta`` is small compared with the measured
duration and that successive quantisation errors are independent. On Linux
with ``CLOCK_MONOTONIC`` at nanosecond resolution the term is negligible
(``u_res ~ 4e-10 s``); on a platform whose monotonic clock ticks at 15.6 ms it
dominates everything else. :func:`clock_report` measures rather than assumes.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .errors import ConfigurationError, TimebaseRegressionError

__all__ = [
    "ClockReport",
    "MonotonicTimebase",
    "Timebase",
    "VirtualTimebase",
    "clock_report",
    "duration_resolution_uncertainty",
    "measure_call_overhead",
    "measure_resolution",
]

_NS_PER_S = 1_000_000_000


@runtime_checkable
class Timebase(Protocol):
    """The HAL timebase contract.

    Implementations must be monotonic non-decreasing: two successive calls to
    :meth:`now` never return a decreasing value. A regression is a fault, not
    a rounding artefact, and must raise
    :class:`~hilforge.errors.TimebaseRegressionError`.
    """

    @property
    def resolution_s(self) -> float:
        """Smallest distinguishable step of this timebase [s]."""

    @property
    def is_virtual(self) -> bool:
        """``True`` if time advances only when the caller advances it."""

    def now(self) -> float:
        """Current time [s] on an arbitrary but fixed epoch."""

    def sleep_until(self, t_s: float) -> None:
        """Block (or, for a virtual timebase, jump) until ``t_s`` [s]."""


class MonotonicTimebase:
    """Wall-clock timebase over :func:`time.perf_counter_ns`.

    ``perf_counter`` is chosen over ``monotonic`` because CPython documents it
    as the highest-resolution clock available with a defined monotonic
    behaviour; its resolution is reported by
    :func:`time.get_clock_info`. The first :meth:`now` call defines the epoch,
    so timestamps start near zero and float64 keeps ~1e-16 s relative
    precision for a run of any realistic length.

    Parameters
    ----------
    guard_monotonic:
        If ``True`` (default) every :meth:`now` compares against the previous
        value and raises :class:`~hilforge.errors.TimebaseRegressionError` on
        a decrease. The guard costs one comparison per call.
    """

    def __init__(self, *, guard_monotonic: bool = True) -> None:
        self._epoch_ns: int | None = None
        self._last_ns: int = 0
        self._guard = bool(guard_monotonic)
        info = time.get_clock_info("perf_counter")
        self._resolution_s = float(info.resolution)

    @property
    def resolution_s(self) -> float:
        """Reported clock resolution [s] from :func:`time.get_clock_info`."""
        return self._resolution_s

    @property
    def is_virtual(self) -> bool:
        return False

    def now(self) -> float:
        raw = time.perf_counter_ns()
        if self._epoch_ns is None:
            self._epoch_ns = raw
        t_ns = raw - self._epoch_ns
        if self._guard and t_ns < self._last_ns:
            raise TimebaseRegressionError(
                f"perf_counter_ns went backwards: {t_ns} ns after {self._last_ns} ns "
                f"(delta {t_ns - self._last_ns} ns)"
            )
        self._last_ns = max(self._last_ns, t_ns)
        return t_ns / _NS_PER_S

    def sleep_until(self, t_s: float) -> None:
        """Sleep until ``t_s``; returns immediately if that time has passed."""
        remaining = t_s - self.now()
        if remaining > 0.0:
            time.sleep(remaining)


class VirtualTimebase:
    """Deterministic timebase whose clock only moves when told to.

    This is what makes a seeded HIL run reproducible to the bit: no wall-clock
    value enters the data path. Time is held as an integer number of ticks of
    ``resolution_s``, so advancing by a multiple of the resolution is exact and
    repeated advances do not accumulate float error.

    Parameters
    ----------
    resolution_s:
        Tick size [s]. Must be positive. Durations handed to :meth:`advance`
        are rounded to the nearest tick, which is the virtual analogue of a
        real clock's quantisation.
    start_s:
        Initial time [s], default 0.0.
    """

    def __init__(self, *, resolution_s: float = 1e-9, start_s: float = 0.0) -> None:
        if not (resolution_s > 0.0):
            raise ConfigurationError(f"resolution_s must be > 0, got {resolution_s!r}")
        if start_s < 0.0:
            raise ConfigurationError(f"start_s must be >= 0, got {start_s!r}")
        self._res = float(resolution_s)
        self._ticks = round(start_s / self._res)
        self.sleep_calls = 0

    @property
    def resolution_s(self) -> float:
        return self._res

    @property
    def is_virtual(self) -> bool:
        return True

    def now(self) -> float:
        return self._ticks * self._res

    def advance(self, dt_s: float) -> float:
        """Advance by ``dt_s`` [s], rounded to the tick. Returns the new time."""
        if dt_s < 0.0:
            raise TimebaseRegressionError(
                f"virtual timebase cannot advance by a negative duration: {dt_s!r} s"
            )
        self._ticks += round(dt_s / self._res)
        return self.now()

    def set_ticks(self, ticks: int) -> None:
        """Force the tick counter, allowing a deliberate backwards step.

        Used only by the fault-injection tests for the "timebase steps
        backwards" failure mode; a backwards step here is detected downstream
        by :class:`hilforge.timing.MonotonicGuard`, not by this setter.
        """
        self._ticks = int(ticks)

    def sleep_until(self, t_s: float) -> None:
        """Jump forward to ``t_s`` if it is in the future; never goes back."""
        self.sleep_calls += 1
        target = round(t_s / self._res)
        self._ticks = max(self._ticks, target)


@dataclass(frozen=True)
class ClockReport:
    """Measured properties of a timebase, with the resolution error term.

    Attributes
    ----------
    reported_resolution_s:
        What the platform claims, from :func:`time.get_clock_info`.
    measured_resolution_s:
        Smallest non-zero difference seen between successive reads [s].
    call_overhead_s:
        Median cost of one ``now()`` call pair measured over an empty
        interval [s]; a bias on every duration, not a random error.
    duration_uncertainty_s:
        ``measured_resolution_s / sqrt(6)`` [s] — the standard uncertainty a
        single duration inherits from quantisation alone (see module
        docstring).
    n_samples:
        Number of read pairs used.
    """

    reported_resolution_s: float
    measured_resolution_s: float
    call_overhead_s: float
    duration_uncertainty_s: float
    n_samples: int

    def as_text(self) -> str:
        """Human-readable block for a benchmark record."""
        return (
            f"reported resolution      : {self.reported_resolution_s:.3e} s\n"
            f"measured resolution      : {self.measured_resolution_s:.3e} s\n"
            f"call-pair overhead (med) : {self.call_overhead_s:.3e} s\n"
            f"u(duration) from res.    : {self.duration_uncertainty_s:.3e} s "
            f"(= delta/sqrt(6), Bennett 1948)\n"
            f"samples                  : {self.n_samples}"
        )


def duration_resolution_uncertainty(resolution_s: float) -> float:
    """Standard uncertainty [s] a duration inherits from clock quantisation.

    ``u = resolution_s / sqrt(6)``, from two independent timestamps each with
    uniform quantisation variance ``resolution_s**2 / 12`` (Bennett 1948).

    Parameters
    ----------
    resolution_s:
        Clock step [s], must be positive.
    """
    if not (resolution_s > 0.0):
        raise ConfigurationError(f"resolution_s must be > 0, got {resolution_s!r}")
    return resolution_s / 6.0**0.5


def measure_resolution(n: int = 2000) -> float:
    """Smallest non-zero gap [s] between successive ``perf_counter_ns`` reads.

    Returns the reported resolution if every observed gap is zero, which
    happens on a clock coarser than the loop that samples it.
    """
    if n < 2:
        raise ConfigurationError(f"n must be >= 2, got {n!r}")
    gaps = []
    prev = time.perf_counter_ns()
    for _ in range(n):
        cur = time.perf_counter_ns()
        if cur > prev:
            gaps.append(cur - prev)
        prev = cur
    if not gaps:
        return float(time.get_clock_info("perf_counter").resolution)
    return min(gaps) / _NS_PER_S


def measure_call_overhead(n: int = 2000) -> float:
    """Median duration [s] reported for a zero-length interval.

    This is the bias that one ``now(); now()`` pair adds to every measured
    duration. It is reported, not subtracted: subtracting a median from
    individual samples would make short durations negative.
    """
    if n < 1:
        raise ConfigurationError(f"n must be >= 1, got {n!r}")
    samples = []
    for _ in range(n):
        a = time.perf_counter_ns()
        b = time.perf_counter_ns()
        samples.append(b - a)
    return statistics.median(samples) / _NS_PER_S


def clock_report(n: int = 2000) -> ClockReport:
    """Measure the wall-clock timebase and return its uncertainty terms."""
    reported = float(time.get_clock_info("perf_counter").resolution)
    measured = measure_resolution(n)
    overhead = measure_call_overhead(n)
    return ClockReport(
        reported_resolution_s=reported,
        measured_resolution_s=measured,
        call_overhead_s=overhead,
        duration_uncertainty_s=duration_resolution_uncertainty(measured),
        n_samples=n,
    )
