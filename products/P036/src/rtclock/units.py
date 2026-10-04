"""Time and rate unit conversions with a single canonical internal unit.

Canonical internal unit: **seconds (s)**, SI base unit of time (BIPM, *The
International System of Units (SI)*, 9th ed., 2019). Every public function in
``rtclock`` takes and returns seconds unless its name ends in ``_ms``,
``_us`` or ``_ns``. The conversions here are exact decimal scalings, not
models, so they carry no assumptions beyond IEEE-754 double rounding.

Validity range: any finite non-negative float that does not overflow when
multiplied by 1e9. For a float64, 1 ns resolution is exactly representable in
the sense that ``n * 1e-9`` for integer ``n < 2**53`` round-trips through
``round(x * 1e9)``; see :func:`ns_to_s` and :func:`s_to_ns`.
"""

from __future__ import annotations

from typing import Final

NS_PER_S: Final[int] = 1_000_000_000
US_PER_S: Final[int] = 1_000_000
MS_PER_S: Final[int] = 1_000

__all__ = [
    "MS_PER_S",
    "NS_PER_S",
    "US_PER_S",
    "frequency_to_period",
    "ms_to_s",
    "ns_to_s",
    "period_to_frequency",
    "s_to_ms",
    "s_to_ns",
    "s_to_us",
    "us_to_s",
]


def _check_finite(value: float, name: str) -> float:
    """Return ``value`` as a float, raising on non-finite or non-numeric input.

    Raises:
        TypeError: if ``value`` is not a real number.
        ValueError: if ``value`` is NaN or infinite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if out != out:
        raise ValueError(f"{name} must not be NaN")
    if out in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite, got {out}")
    return out


def s_to_ns(seconds: float) -> float:
    """Convert seconds to nanoseconds. Units: s -> ns."""
    return _check_finite(seconds, "seconds") * NS_PER_S


def ns_to_s(nanoseconds: float) -> float:
    """Convert nanoseconds to seconds. Units: ns -> s."""
    return _check_finite(nanoseconds, "nanoseconds") / NS_PER_S


def s_to_us(seconds: float) -> float:
    """Convert seconds to microseconds. Units: s -> us."""
    return _check_finite(seconds, "seconds") * US_PER_S


def us_to_s(microseconds: float) -> float:
    """Convert microseconds to seconds. Units: us -> s."""
    return _check_finite(microseconds, "microseconds") / US_PER_S


def s_to_ms(seconds: float) -> float:
    """Convert seconds to milliseconds. Units: s -> ms."""
    return _check_finite(seconds, "seconds") * MS_PER_S


def ms_to_s(milliseconds: float) -> float:
    """Convert milliseconds to seconds. Units: ms -> s."""
    return _check_finite(milliseconds, "milliseconds") / MS_PER_S


def frequency_to_period(frequency_hz: float) -> float:
    """Period in seconds for a rate in hertz: ``T = 1 / f``.

    Args:
        frequency_hz: strictly positive rate, units Hz (1/s).

    Returns:
        Period, units s.

    Raises:
        ValueError: if ``frequency_hz`` is not strictly positive.
    """
    f = _check_finite(frequency_hz, "frequency_hz")
    if f <= 0.0:
        raise ValueError(f"frequency_hz must be > 0 Hz, got {f}")
    return 1.0 / f


def period_to_frequency(period_s: float) -> float:
    """Rate in hertz for a period in seconds: ``f = 1 / T``.

    Args:
        period_s: strictly positive period, units s.

    Returns:
        Rate, units Hz (1/s).

    Raises:
        ValueError: if ``period_s`` is not strictly positive.
    """
    t = _check_finite(period_s, "period_s")
    if t <= 0.0:
        raise ValueError(f"period_s must be > 0 s, got {t}")
    return 1.0 / t
