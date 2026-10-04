"""Monotonic timebase abstraction with a measured, not assumed, resolution.

Three implementations share one protocol:

* :class:`MonotonicTimebase` — the host's ``time.monotonic_ns`` clock. Its
  resolution is **measured** by :func:`measure_clock_resolution`, never taken
  from a constant.
* :class:`SimulatedTimebase` — a deterministic virtual clock. Advances only
  when asked. Used for every test that would otherwise assert on wall-clock
  behaviour of a shared machine.
* :class:`SkewedSimulatedTimebase` — a deterministic virtual clock whose
  *sleep* primitive has a fractional rate error (skew) and a constant
  wake-up delay relative to the clock used for timestamping. This is the
  injection mechanism for the drift validation in :mod:`rtclock.loop`.

Measurement uncertainty from clock quantization
-----------------------------------------------
A clock that advances in steps of ``q`` seconds reports a timestamp that is
the true instant quantized to a multiple of ``q``. Treating each endpoint's
quantization error as independent and uniform on ``[-q/2, +q/2]`` (the
standard treatment of a rounding error in JCGM 100:2008 / GUM, Sec. 4.3.7 and
F.2.2.1), a single endpoint has standard uncertainty ``q / sqrt(12)``. A
duration ``t2 - t1`` combines two such endpoints in quadrature:

    u(dt) = q / sqrt(6)        (standard uncertainty, units s)
    |dt - dt_true| <= q        (worst case, units s)

Assumption: the two endpoints quantize independently. That is optimistic when
``dt`` is an exact multiple of ``q``; the worst-case bound ``±q`` holds
regardless and is the bound reported alongside every measured duration in
this package.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .units import NS_PER_S

__all__ = [
    "ClockResolution",
    "MonotonicTimebase",
    "SimulatedTimebase",
    "SkewedSimulatedTimebase",
    "Timebase",
    "duration_uncertainty_s",
    "measure_clock_resolution",
]


@runtime_checkable
class Timebase(Protocol):
    """A monotonic non-decreasing clock with a sleep primitive.

    ``now()`` returns seconds since an unspecified epoch (differences are
    meaningful, absolute values are not). ``sleep(duration_s)`` blocks for
    approximately ``duration_s`` seconds; a non-positive duration returns
    immediately.
    """

    def now(self) -> float:
        """Current time, units s, monotonic non-decreasing."""
        ...

    def sleep(self, duration_s: float) -> None:
        """Advance time by approximately ``duration_s`` seconds."""
        ...


@dataclass(frozen=True)
class ClockResolution:
    """Result of a clock-resolution measurement.

    Attributes:
        advertised_s: resolution the platform reports via
            ``time.get_clock_info(name).resolution``, units s. This is a claim,
            not a measurement.
        measured_tick_s: smallest non-zero difference observed between
            successive readings, units s. This is the observable granularity:
            no duration shorter than this can be distinguished from zero.
        median_call_delta_s: median difference between successive readings,
            units s. Dominated by the cost of reading the clock, so it is an
            upper bound on the shortest interval this clock can *time*, as
            opposed to merely represent.
        samples: number of clock readings taken.
        zero_delta_fraction: fraction of successive readings that were
            identical. A non-zero value means the clock is coarser than the
            read loop.
        clock_name: the ``time`` module clock that was measured.
        method: one-line description of how the numbers were obtained.
    """

    advertised_s: float
    measured_tick_s: float
    median_call_delta_s: float
    samples: int
    zero_delta_fraction: float
    clock_name: str
    method: str

    @property
    def worst_case_duration_error_s(self) -> float:
        """Worst-case error on a duration from two quantized endpoints, units s."""
        return self.measured_tick_s

    @property
    def standard_duration_uncertainty_s(self) -> float:
        """Standard uncertainty ``q / sqrt(6)`` on a duration, units s."""
        return duration_uncertainty_s(self.measured_tick_s)


def duration_uncertainty_s(quantum_s: float) -> float:
    """Standard uncertainty of a duration measured on a clock of step ``quantum_s``.

    Two independent uniform rounding errors of width ``q`` combine in
    quadrature: ``u = sqrt(2) * q / sqrt(12) = q / sqrt(6)``.
    Reference: JCGM 100:2008 (GUM) Sec. 4.3.7 (uniform distribution) and
    Sec. 5.1.2 (combination in quadrature).

    Args:
        quantum_s: clock step, units s, must be >= 0.

    Returns:
        Standard uncertainty, units s.

    Raises:
        ValueError: if ``quantum_s`` is negative.
    """
    if quantum_s < 0.0:
        raise ValueError(f"quantum_s must be >= 0 s, got {quantum_s}")
    return quantum_s / 6.0**0.5


def measure_clock_resolution(
    samples: int = 20_000, clock_name: str = "monotonic"
) -> ClockResolution:
    """Measure the observable resolution of a platform clock.

    Method: read ``time.monotonic_ns`` (or the named clock's ``_ns`` variant)
    ``samples`` times back to back into a preallocated list, then take the
    smallest non-zero forward difference as the observable tick and the median
    forward difference as the per-read cost. No sleeping, no allocation inside
    the loop. The advertised resolution from
    ``time.get_clock_info(clock_name).resolution`` is recorded alongside for
    comparison but is never substituted for the measurement.

    The measured tick is an *upper* bound on the true hardware granularity: if
    every successive read differs, the loop is slower than the clock and the
    true tick could be finer. ``zero_delta_fraction > 0`` is the signal that
    the clock, not the loop, is the limiting factor.

    Args:
        samples: number of clock readings, must be >= 2.
        clock_name: ``"monotonic"`` or ``"perf_counter"``.

    Returns:
        A :class:`ClockResolution`.

    Raises:
        ValueError: if ``samples < 2`` or ``clock_name`` is unsupported.
    """
    if samples < 2:
        raise ValueError(f"samples must be >= 2, got {samples}")
    readers = {
        "monotonic": time.monotonic_ns,
        "perf_counter": time.perf_counter_ns,
    }
    if clock_name not in readers:
        raise ValueError(f"clock_name must be one of {sorted(readers)}, got {clock_name!r}")
    read = readers[clock_name]

    buf = [0] * samples
    for i in range(samples):
        buf[i] = read()

    deltas = [b - a for a, b in zip(buf[:-1], buf[1:], strict=True)]
    nonzero = [d for d in deltas if d > 0]
    if not nonzero:
        raise RuntimeError(
            f"clock {clock_name!r} did not advance over {samples} reads; cannot measure"
        )
    deltas_sorted = sorted(deltas)
    mid = len(deltas_sorted) // 2
    median_ns = (
        deltas_sorted[mid]
        if len(deltas_sorted) % 2 == 1
        else 0.5 * (deltas_sorted[mid - 1] + deltas_sorted[mid])
    )
    zero_fraction = sum(1 for d in deltas if d == 0) / len(deltas)

    return ClockResolution(
        advertised_s=float(time.get_clock_info(clock_name).resolution),
        measured_tick_s=min(nonzero) / NS_PER_S,
        median_call_delta_s=median_ns / NS_PER_S,
        samples=samples,
        zero_delta_fraction=zero_fraction,
        clock_name=clock_name,
        method=(
            f"{samples} back-to-back time.{clock_name}_ns() reads; observable tick = "
            "min non-zero forward difference; per-read cost = median forward difference"
        ),
    )


class MonotonicTimebase:
    """The host monotonic clock, in seconds, with a measured resolution.

    ``time.monotonic_ns`` is used rather than ``time.monotonic`` so that the
    integer nanosecond value is not passed through a float before differencing.
    The float returned by :meth:`now` is still a float64, so for a process
    uptime of 1e6 s the representable step is about 2e-10 s -- below the
    measured tick on any platform this package has been run on, but stated
    here because it is the one place the abstraction loses precision.
    """

    def __init__(self, resolution: ClockResolution | None = None) -> None:
        """Args: resolution: a prior measurement; measured on first use if omitted."""
        self._resolution = resolution

    def now(self) -> float:
        """Current monotonic time, units s."""
        return time.monotonic_ns() / NS_PER_S

    def sleep(self, duration_s: float) -> None:
        """Sleep for ``duration_s`` seconds; non-positive returns immediately.

        ``time.sleep`` guarantees *at least* the requested duration (CPython
        docs, ``time.sleep``); it does not bound the overshoot. The overshoot
        is exactly what :class:`rtclock.loop.FixedRateLoop` reports as drift.
        """
        if duration_s > 0.0:
            time.sleep(duration_s)

    @property
    def resolution(self) -> ClockResolution:
        """Measured resolution of this clock, measured once and cached."""
        if self._resolution is None:
            self._resolution = measure_clock_resolution(clock_name="monotonic")
        return self._resolution


class SimulatedTimebase:
    """A deterministic virtual clock. Ideal: no skew, no jitter, no overshoot.

    Every ``sleep(d)`` advances :meth:`now` by exactly ``max(d, 0)``. Used for
    tests and examples that must be reproducible on a loaded machine.
    """

    def __init__(self, start_s: float = 0.0) -> None:
        """Args: start_s: initial reported time, units s."""
        self._t = float(start_s)

    def now(self) -> float:
        """Current virtual time, units s."""
        return self._t

    def sleep(self, duration_s: float) -> None:
        """Advance virtual time by ``max(duration_s, 0)`` seconds."""
        if duration_s > 0.0:
            self._t += float(duration_s)


class SkewedSimulatedTimebase:
    """A virtual clock whose sleep primitive is miscalibrated by a known amount.

    Models the common real configuration in which the thing that *waits* (an
    OS timer, a hardware tick) and the thing that *timestamps* (the monotonic
    clock) are not the same oscillator. A request to sleep ``d`` seconds
    advances the timestamping clock by::

        d * (1 + skew_ppm * 1e-6) + wake_delay_s

    so a positive ``skew_ppm`` means each wait is proportionally too long, and
    ``wake_delay_s`` is a fixed per-wait wake-up latency. Both are exact and
    noise-free by construction: this class exists so that the drift a loop
    driver reports can be compared against a closed form rather than against a
    measurement taken on a shared CPU.

    Validity: ``skew_ppm > -1e6`` (a skew of -1e6 ppm would stop the clock).
    """

    def __init__(
        self, skew_ppm: float = 0.0, wake_delay_s: float = 0.0, start_s: float = 0.0
    ) -> None:
        """Args:
        skew_ppm: fractional rate error of the sleep primitive, units ppm.
        wake_delay_s: constant additive latency per wait, units s, >= 0.
        start_s: initial reported time, units s.

        Raises:
            ValueError: if ``skew_ppm <= -1e6`` or ``wake_delay_s < 0``.
        """
        if skew_ppm <= -1.0e6:
            raise ValueError(f"skew_ppm must be > -1e6 ppm, got {skew_ppm}")
        if wake_delay_s < 0.0:
            raise ValueError(f"wake_delay_s must be >= 0 s, got {wake_delay_s}")
        self.skew_ppm = float(skew_ppm)
        self.wake_delay_s = float(wake_delay_s)
        self._t = float(start_s)

    @property
    def skew(self) -> float:
        """Skew as a dimensionless fraction (``skew_ppm * 1e-6``)."""
        return self.skew_ppm * 1.0e-6

    def now(self) -> float:
        """Current virtual time on the timestamping clock, units s."""
        return self._t

    def sleep(self, duration_s: float) -> None:
        """Advance by ``duration_s * (1 + skew) + wake_delay_s`` if positive."""
        if duration_s > 0.0:
            self._t += float(duration_s) * (1.0 + self.skew) + self.wake_delay_s
