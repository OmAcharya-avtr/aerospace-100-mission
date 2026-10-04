"""Timebase tests. Nothing here asserts a wall-clock bound.

Every timing assertion is either on a deterministic simulated clock or on
arithmetic. The host clock is exercised only for monotonicity and for the
*existence* and *consistency* of a resolution measurement -- never for a
particular achieved latency, because the build machine is shared.
"""

from __future__ import annotations

import math

import pytest

from rtclock.timebase import (
    MonotonicTimebase,
    SimulatedTimebase,
    SkewedSimulatedTimebase,
    Timebase,
    duration_uncertainty_s,
    measure_clock_resolution,
)


def test_simulated_timebase_is_exact():
    tb = SimulatedTimebase(start_s=5.0)
    assert tb.now() == 5.0
    tb.sleep(0.25)
    assert tb.now() == 5.25
    tb.sleep(-1.0)  # non-positive sleeps do nothing
    assert tb.now() == 5.25
    tb.sleep(0.0)
    assert tb.now() == 5.25


def test_simulated_timebase_is_monotonic_over_many_sleeps():
    tb = SimulatedTimebase()
    last = tb.now()
    for _ in range(1000):
        tb.sleep(1e-3)
        assert tb.now() >= last
        last = tb.now()
    # 1000 sleeps of 1e-3 s: the sum of 1000 binary64 additions of 1e-3 is not
    # exactly 1.0, so bound it by 1000 ulps of 1.0 rather than asserting equality.
    assert abs(tb.now() - 1.0) <= 1000 * math.ulp(1.0)


# Hand computation: skew 1e6 ppm = factor 2. A 0.1 s sleep advances
# 0.1 * (1 + 1.0) + 0.01 = 0.21 s.
def test_skewed_timebase_known_answer():
    tb = SkewedSimulatedTimebase(skew_ppm=1.0e6 - 1.0, wake_delay_s=0.01)
    assert tb.skew == pytest.approx(1.0 - 1e-6, rel=0, abs=1e-15)
    tb2 = SkewedSimulatedTimebase(skew_ppm=1.0e6, wake_delay_s=0.01)
    tb2.sleep(0.1)
    assert tb2.now() == pytest.approx(0.21, rel=0, abs=1e-15)


def test_skewed_timebase_zero_skew_matches_ideal():
    a = SimulatedTimebase()
    b = SkewedSimulatedTimebase(skew_ppm=0.0, wake_delay_s=0.0)
    for d in (1e-3, 1e-2, 0.5):
        a.sleep(d)
        b.sleep(d)
    assert a.now() == b.now()


def test_skewed_timebase_rejects_stopped_and_negative_delay():
    with pytest.raises(ValueError, match="skew_ppm must be > -1e6"):
        SkewedSimulatedTimebase(skew_ppm=-1.0e6)
    with pytest.raises(ValueError, match="wake_delay_s must be >= 0"):
        SkewedSimulatedTimebase(wake_delay_s=-1e-9)


def test_all_three_satisfy_the_protocol():
    for tb in (MonotonicTimebase(), SimulatedTimebase(), SkewedSimulatedTimebase()):
        assert isinstance(tb, Timebase)


def test_monotonic_timebase_is_non_decreasing():
    tb = MonotonicTimebase()
    readings = [tb.now() for _ in range(2000)]
    assert all(b >= a for a, b in zip(readings[:-1], readings[1:], strict=True))


def test_measured_resolution_is_internally_consistent():
    res = measure_clock_resolution(samples=5000)
    assert res.samples == 5000
    assert res.measured_tick_s > 0.0
    assert res.advertised_s > 0.0
    # The observable tick cannot be finer than what the platform advertises.
    assert res.measured_tick_s >= res.advertised_s
    # The median forward difference is a forward difference, so it is either 0
    # (clock coarser than the loop) or at least one observable tick.
    assert res.median_call_delta_s == 0.0 or res.median_call_delta_s >= res.measured_tick_s
    assert 0.0 <= res.zero_delta_fraction <= 1.0
    assert res.worst_case_duration_error_s == res.measured_tick_s
    assert res.standard_duration_uncertainty_s == pytest.approx(
        res.measured_tick_s / math.sqrt(6.0), rel=1e-15
    )
    assert "back-to-back" in res.method


def test_measure_clock_resolution_validates_arguments():
    with pytest.raises(ValueError, match="samples must be >= 2"):
        measure_clock_resolution(samples=1)
    with pytest.raises(ValueError, match="clock_name must be one of"):
        measure_clock_resolution(clock_name="wall")


def test_perf_counter_is_also_measurable():
    res = measure_clock_resolution(samples=2000, clock_name="perf_counter")
    assert res.clock_name == "perf_counter"
    assert res.measured_tick_s > 0.0


# Hand computation: a uniform rounding error of width q has standard deviation
# q/sqrt(12); two independent endpoints combine in quadrature to
# sqrt(2) * q/sqrt(12) = q/sqrt(6). For q = 1e-9 s this is 4.0824829046e-10 s.
def test_duration_uncertainty_known_answer():
    assert duration_uncertainty_s(1e-9) == pytest.approx(4.0824829046386e-10, rel=1e-12)
    assert duration_uncertainty_s(0.0) == 0.0
    with pytest.raises(ValueError, match="must be >= 0"):
        duration_uncertainty_s(-1.0)


def test_monotonic_timebase_caches_resolution():
    tb = MonotonicTimebase()
    first = tb.resolution
    assert tb.resolution is first


def test_monotonic_timebase_accepts_a_prior_measurement():
    res = measure_clock_resolution(samples=2000)
    tb = MonotonicTimebase(resolution=res)
    assert tb.resolution is res


def test_monotonic_sleep_ignores_non_positive():
    tb = MonotonicTimebase()
    before = tb.now()
    tb.sleep(0.0)
    tb.sleep(-5.0)
    # Only monotonicity is asserted; no bound on elapsed time on a shared host.
    assert tb.now() >= before
