"""Timebases, clock characterisation and the resolution error term."""

from __future__ import annotations

import math

import pytest

from hilforge.errors import ConfigurationError, TimebaseRegressionError
from hilforge.timebase import (
    MonotonicTimebase,
    VirtualTimebase,
    clock_report,
    duration_resolution_uncertainty,
    measure_call_overhead,
    measure_resolution,
)


def test_virtual_timebase_starts_at_zero_and_advances_exactly():
    tb = VirtualTimebase(resolution_s=1e-9)
    assert tb.now() == 0.0
    tb.advance(0.01)
    assert tb.now() == pytest.approx(0.01, abs=1e-15)
    # 1000 advances of 1 ms must land exactly on 1 s, because the clock counts
    # integer ticks rather than accumulating a float.
    tb2 = VirtualTimebase(resolution_s=1e-9)
    for _ in range(1000):
        tb2.advance(0.001)
    assert tb2.now() == 1.0


def test_virtual_timebase_rejects_negative_advance():
    tb = VirtualTimebase()
    with pytest.raises(TimebaseRegressionError):
        tb.advance(-1e-6)


def test_virtual_timebase_rejects_bad_construction():
    with pytest.raises(ConfigurationError):
        VirtualTimebase(resolution_s=0.0)
    with pytest.raises(ConfigurationError):
        VirtualTimebase(resolution_s=-1e-9)
    with pytest.raises(ConfigurationError):
        VirtualTimebase(start_s=-1.0)


def test_virtual_sleep_until_never_goes_back():
    tb = VirtualTimebase()
    tb.advance(1.0)
    tb.sleep_until(0.5)
    assert tb.now() == 1.0
    tb.sleep_until(2.0)
    assert tb.now() == pytest.approx(2.0, abs=1e-12)
    assert tb.sleep_calls == 2


def test_virtual_is_virtual_and_real_is_not():
    assert VirtualTimebase().is_virtual is True
    assert MonotonicTimebase().is_virtual is False


def test_monotonic_timebase_is_non_decreasing():
    tb = MonotonicTimebase()
    last = tb.now()
    for _ in range(2000):
        now = tb.now()
        assert now >= last
        last = now
    assert tb.resolution_s > 0.0


def test_resolution_uncertainty_is_delta_over_sqrt_six():
    # Hand check: delta = 1e-6 s -> u = 1e-6 / sqrt(6) = 4.0824829e-07 s.
    assert duration_resolution_uncertainty(1e-6) == pytest.approx(4.0824829046e-07, rel=1e-9)
    assert duration_resolution_uncertainty(1.0) == pytest.approx(1.0 / math.sqrt(6.0))
    with pytest.raises(ConfigurationError):
        duration_resolution_uncertainty(0.0)


def test_measure_resolution_and_overhead_are_positive_and_finite():
    res = measure_resolution(500)
    over = measure_call_overhead(500)
    assert res > 0.0 and math.isfinite(res)
    assert over >= 0.0 and math.isfinite(over)


def test_measure_resolution_rejects_small_n():
    with pytest.raises(ConfigurationError):
        measure_resolution(1)
    with pytest.raises(ConfigurationError):
        measure_call_overhead(0)


def test_clock_report_is_self_consistent():
    report = clock_report(500)
    assert report.n_samples == 500
    assert report.measured_resolution_s > 0.0
    assert report.duration_uncertainty_s == pytest.approx(
        report.measured_resolution_s / math.sqrt(6.0)
    )
    text = report.as_text()
    assert "Bennett 1948" in text
    assert "measured resolution" in text
