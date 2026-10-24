"""Timebases, in integer nanoseconds, with the quantisation error term named.

Units
-----
Every timestamp crossing the hardware abstraction layer is an **integer number
of nanoseconds**. Integers are used deliberately: a timestamp exchange is
differenced four ways by the estimators in :mod:`timetransfer.twoway`, and an
integer representation makes those differences exact, which in turn makes
simulation/device parity checkable byte for byte rather than to a tolerance.
Seconds appear only at the estimator boundary, as ``float``.

Quantisation as an error term
-----------------------------
A timestamp reported in steps of ``delta`` seconds carries a quantisation
error modelled as uniform on ``[-delta/2, +delta/2]``, variance
``delta**2 / 12`` (Bennett 1948, "Spectra of Quantized Signals", *Bell System
Technical Journal* 27(3):446-472, the standard uniform-quantisation result).

The two-way offset estimator differences four such timestamps with coefficients
``(-1, +1, +1, -1)`` and a factor of one half, so the quantisation variance it
inherits is ``4 * delta**2 / 12 / 4 = delta**2 / 12``, i.e. a standard
uncertainty of

    u_q(theta_hat) = delta / sqrt(12)                                      [s]

Combination with the other terms of the offset uncertainty budget follows
JCGM 100:2008 (*Evaluation of measurement data -- Guide to the expression of
uncertainty in measurement*) section 5.1.2: independent contributions add in
quadrature. See :mod:`timetransfer.uncertainty`.

Validity range
--------------
The uniform model assumes ``delta`` is small compared with the quantity being
measured and that the four quantisation errors are independent. Neither holds
when two timestamps fall in the same clock tick; :func:`clock_report` measures
the resolution instead of assuming it.
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
    "NS_PER_S",
    "Timebase",
    "VirtualTimebase",
    "clock_report",
    "measure_call_overhead",
    "measure_resolution_ns",
    "offset_quantisation_uncertainty",
]

NS_PER_S = 1_000_000_000
"""Nanoseconds per second. The one magic number in this package."""


@runtime_checkable
class Timebase(Protocol):
    """The HAL timebase contract.

    Implementations are monotone non-decreasing: two successive :meth:`now_ns`
    calls never return a decreasing value, and a regression raises
    :class:`~timetransfer.errors.TimebaseRegressionError`.
    """

    @property
    def resolution_ns(self) -> int:
        """Smallest distinguishable step of this timebase [ns]."""

    @property
    def is_virtual(self) -> bool:
        """``True`` if time advances only when the caller advances it."""

    def now_ns(self) -> int:
        """Current reading [ns] on an arbitrary but fixed epoch."""


class MonotonicTimebase:
    """Wall-clock timebase over :func:`time.perf_counter_ns`.

    ``perf_counter_ns`` is used rather than ``monotonic_ns`` because CPython
    documents it as the highest-resolution clock with defined monotone
    behaviour. The first :meth:`now_ns` call fixes the epoch so readings start
    near zero.

    Parameters
    ----------
    guard_monotonic:
        When ``True`` (default) every read is compared against the previous one
        and a decrease raises
        :class:`~timetransfer.errors.TimebaseRegressionError`.
    """

    def __init__(self, *, guard_monotonic: bool = True) -> None:
        self._epoch_ns: int | None = None
        self._last_ns = 0
        self._guard = bool(guard_monotonic)
        reported = time.get_clock_info("perf_counter").resolution
        self._resolution_ns = max(1, round(reported * NS_PER_S))

    @property
    def resolution_ns(self) -> int:
        """Reported resolution [ns] from :func:`time.get_clock_info`."""
        return self._resolution_ns

    @property
    def is_virtual(self) -> bool:
        return False

    def now_ns(self) -> int:
        raw = time.perf_counter_ns()
        if self._epoch_ns is None:
            self._epoch_ns = raw
        value = raw - self._epoch_ns
        if self._guard and value < self._last_ns:
            raise TimebaseRegressionError(
                f"perf_counter_ns went backwards: {value} ns after {self._last_ns} ns"
            )
        self._last_ns = max(self._last_ns, value)
        return value


class VirtualTimebase:
    """Deterministic timebase whose reading only moves when told to.

    This is what makes a seeded acquisition reproducible exactly: no wall-clock
    value enters the data path. Time is held as an integer nanosecond count, so
    repeated advances do not accumulate float error.

    Parameters
    ----------
    resolution_ns:
        Tick size [ns], must be >= 1. Advances are rounded to this tick, which
        is the virtual analogue of a real clock's quantisation.
    start_ns:
        Initial reading [ns], default 0.
    """

    def __init__(self, *, resolution_ns: int = 1, start_ns: int = 0) -> None:
        if int(resolution_ns) < 1:
            raise ConfigurationError(f"resolution_ns must be >= 1, got {resolution_ns!r}")
        if int(start_ns) < 0:
            raise ConfigurationError(f"start_ns must be >= 0, got {start_ns!r}")
        self._res = int(resolution_ns)
        self._ticks = int(start_ns) // self._res
        self.advance_calls = 0

    @property
    def resolution_ns(self) -> int:
        return self._res

    @property
    def is_virtual(self) -> bool:
        return True

    def now_ns(self) -> int:
        return self._ticks * self._res

    def advance_ns(self, dt_ns: int) -> int:
        """Advance by ``dt_ns`` [ns], rounded to the tick. Returns the new reading."""
        if int(dt_ns) < 0:
            raise TimebaseRegressionError(
                f"virtual timebase cannot advance by a negative duration: {dt_ns!r} ns"
            )
        self.advance_calls += 1
        self._ticks += round(int(dt_ns) / self._res)
        return self.now_ns()

    def set_ns(self, value_ns: int) -> None:
        """Force the reading, allowing a deliberate backwards step.

        Used only by the fault-injection tests. A backwards step here is
        detected downstream, not refused here.
        """
        self._ticks = int(value_ns) // self._res


@dataclass(frozen=True)
class ClockReport:
    """Measured properties of the host timebase, with the derived error terms.

    Attributes
    ----------
    reported_resolution_ns:
        What the platform claims, from :func:`time.get_clock_info`.
    measured_resolution_ns:
        Smallest non-zero difference observed between successive reads [ns].
    call_overhead_ns:
        Median reading reported for a zero-length interval [ns]; a bias on
        every duration, not a random error, and reported rather than
        subtracted because subtracting a median makes short durations negative.
    offset_quantisation_uncertainty_s:
        ``measured_resolution_ns * 1e-9 / sqrt(12)`` [s] -- the standard
        uncertainty a single two-way offset estimate inherits from timestamp
        quantisation alone (see the module docstring).
    n_samples:
        Number of read pairs used.
    """

    reported_resolution_ns: int
    measured_resolution_ns: int
    call_overhead_ns: float
    offset_quantisation_uncertainty_s: float
    n_samples: int

    def as_text(self) -> str:
        """Human-readable block for a validation or benchmark record."""
        return (
            f"reported resolution        : {self.reported_resolution_ns} ns\n"
            f"measured resolution        : {self.measured_resolution_ns} ns\n"
            f"call-pair overhead (median): {self.call_overhead_ns:.1f} ns\n"
            f"u_q(offset) from resolution: "
            f"{self.offset_quantisation_uncertainty_s:.6e} s "
            f"(= delta/sqrt(12), Bennett 1948)\n"
            f"samples                    : {self.n_samples}"
        )


def offset_quantisation_uncertainty(resolution_ns: float) -> float:
    """Standard uncertainty [s] a two-way offset estimate inherits from quantisation.

    ``u = resolution_s / sqrt(12)``. Four timestamps each with uniform
    quantisation variance ``delta**2 / 12`` (Bennett 1948) enter the estimator
    with coefficients ``(-1, +1, +1, -1)`` and a factor of one half, giving
    ``4 * (delta**2/12) * (1/2)**2 = delta**2 / 12``.

    Parameters
    ----------
    resolution_ns:
        Timestamp resolution [ns], must be positive.

    Returns
    -------
    float
        Standard uncertainty in seconds.
    """
    if not (float(resolution_ns) > 0.0):
        raise ConfigurationError(f"resolution_ns must be > 0, got {resolution_ns!r}")
    return (float(resolution_ns) / NS_PER_S) / 12.0**0.5


def measure_resolution_ns(n: int = 2000) -> int:
    """Smallest non-zero gap [ns] between successive ``perf_counter_ns`` reads.

    Falls back to the reported resolution when every observed gap is zero,
    which happens on a clock coarser than the loop sampling it.
    """
    if int(n) < 2:
        raise ConfigurationError(f"n must be >= 2, got {n!r}")
    gaps: list[int] = []
    prev = time.perf_counter_ns()
    for _ in range(int(n)):
        cur = time.perf_counter_ns()
        if cur > prev:
            gaps.append(cur - prev)
        prev = cur
    if not gaps:
        return max(1, round(time.get_clock_info("perf_counter").resolution * NS_PER_S))
    return min(gaps)


def measure_call_overhead(n: int = 2000) -> float:
    """Median duration [ns] reported for a zero-length interval."""
    if int(n) < 1:
        raise ConfigurationError(f"n must be >= 1, got {n!r}")
    samples = []
    for _ in range(int(n)):
        a = time.perf_counter_ns()
        b = time.perf_counter_ns()
        samples.append(b - a)
    return float(statistics.median(samples))


def clock_report(n: int = 2000) -> ClockReport:
    """Measure the host timebase and return it with its derived error terms."""
    reported = max(1, round(time.get_clock_info("perf_counter").resolution * NS_PER_S))
    measured = measure_resolution_ns(n)
    return ClockReport(
        reported_resolution_ns=reported,
        measured_resolution_ns=measured,
        call_overhead_ns=measure_call_overhead(n),
        offset_quantisation_uncertainty_s=offset_quantisation_uncertainty(measured),
        n_samples=int(n),
    )
