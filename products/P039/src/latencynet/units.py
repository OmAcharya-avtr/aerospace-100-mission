"""Time-unit conversions. Seconds are the single canonical unit in this package.

Every public function in :mod:`latencynet` takes and returns seconds (s).
These helpers exist so that a caller working in microseconds never has to
write a bare ``1e-6`` at a call site, which is where unit errors come from.

Conversions are exact decimal scalings; no physical model is involved, so
there is no reference, no assumption and no validity range beyond the range of
binary64.
"""

from __future__ import annotations

_MS = 1.0e-3
_US = 1.0e-6
_NS = 1.0e-9


def ms_to_s(value_ms: float) -> float:
    """Milliseconds to seconds."""
    return float(value_ms) * _MS


def us_to_s(value_us: float) -> float:
    """Microseconds to seconds."""
    return float(value_us) * _US


def ns_to_s(value_ns: float) -> float:
    """Nanoseconds to seconds."""
    return float(value_ns) * _NS


def s_to_ms(value_s: float) -> float:
    """Seconds to milliseconds."""
    return float(value_s) / _MS


def s_to_us(value_s: float) -> float:
    """Seconds to microseconds."""
    return float(value_s) / _US


def s_to_ns(value_s: float) -> float:
    """Seconds to nanoseconds."""
    return float(value_s) / _NS
