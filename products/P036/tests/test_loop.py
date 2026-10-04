"""Loop-driver tests against closed-form drift on a deterministic clock.

No test here asserts a wall-clock bound. Drift is compared against the
closed-form expressions in :mod:`rtclock.loop`, which are exact for the
deterministic :class:`~rtclock.timebase.SkewedSimulatedTimebase`.
"""

from __future__ import annotations

import pytest

from rtclock.loop import (
    FixedRateLoop,
    absolute_mode_drift,
    relative_mode_drift,
)
from rtclock.timebase import (
    MonotonicTimebase,
    SimulatedTimebase,
    SkewedSimulatedTimebase,
)


def test_ideal_clock_gives_exactly_zero_drift():
    loop = FixedRateLoop(period_s=1e-3, timebase=SimulatedTimebase())
    report = loop.run(iterations=500)
    assert report.max_abs_drift_s == 0.0
    assert report.late_releases == 0
    assert len(report.records) == 500
    assert report.records[0].drift_s == 0.0


# Hand computation, relative mode, T = 10 ms, skew = +100 ppm (eps = 1e-4),
# no wake delay. Each wait of 10 ms advances the clock by
# 10 ms * 1.0001 = 10.001 ms, an error of 1 us per iteration, and relative
# scheduling never corrects it:
#   drift_k = k * 1 us.  drift_1 = 1e-6 s, drift_100 = 1e-4 s.
def test_relative_mode_drift_hand_values():
    tb = SkewedSimulatedTimebase(skew_ppm=100.0)
    report = FixedRateLoop(period_s=10e-3, timebase=tb, mode="relative").run(iterations=101)
    assert report.records[1].drift_s == pytest.approx(1e-6, rel=1e-12)
    assert report.records[100].drift_s == pytest.approx(1e-4, rel=1e-12)
    assert report.drift_slope_s_per_iteration() == pytest.approx(1e-6, rel=1e-9)


# Hand computation, absolute mode, same clock. a = T*eps + d = 1 us.
#   drift_1 = a = 1.0e-6 s
#   drift_2 = a(1 - eps) = 1e-6 * 0.9999 = 9.999e-7 s
#   drift_inf -> a/(1+eps) = 1e-6/1.0001 = 9.99900009999e-7 s
# Absolute scheduling corrects, so drift is bounded by a and does not grow.
def test_absolute_mode_drift_hand_values():
    tb = SkewedSimulatedTimebase(skew_ppm=100.0)
    report = FixedRateLoop(period_s=10e-3, timebase=tb, mode="absolute").run(iterations=1001)
    assert report.records[1].drift_s == pytest.approx(1.0e-6, rel=1e-12)
    assert report.records[2].drift_s == pytest.approx(9.999e-7, rel=1e-12)
    assert report.records[1000].drift_s == pytest.approx(1e-6 / 1.0001, rel=1e-12)
    assert report.max_abs_drift_s <= 1.0e-6 * (1.0 + 1e-12)


@pytest.mark.parametrize("skew_ppm", [-500.0, -1.0, 0.0, 1.0, 250.0, 10_000.0])
@pytest.mark.parametrize("wake_delay_s", [0.0, 1e-7, 5e-6])
@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_drift_matches_closed_form_for_every_iteration(skew_ppm, wake_delay_s, mode):
    period = 10e-3
    tb = SkewedSimulatedTimebase(skew_ppm=skew_ppm, wake_delay_s=wake_delay_s)
    report = FixedRateLoop(period_s=period, timebase=tb, mode=mode).run(iterations=200)
    closed = relative_mode_drift if mode == "relative" else absolute_mode_drift
    # Tolerance: drift is a difference of two accumulated sums, so it carries
    # binary64 roundoff of order k * ulp(k*T). For k = 200 and T = 10 ms the
    # accumulated clock value is 2 s, ulp(2.0) = 4.44e-16, and 200 additions
    # bound the error at ~1e-13 s. The 1 ps absolute floor used here is 1e-10
    # of the period and is roundoff, not model error.
    for rec in report.records:
        expected = closed(rec.index, period, skew_ppm, wake_delay_s)
        assert rec.drift_s == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_relative_mode_accumulates_and_absolute_mode_does_not():
    period, skew = 1e-3, 200.0
    rel = FixedRateLoop(
        period_s=period, timebase=SkewedSimulatedTimebase(skew_ppm=skew), mode="relative"
    ).run(iterations=5000)
    absol = FixedRateLoop(
        period_s=period, timebase=SkewedSimulatedTimebase(skew_ppm=skew), mode="absolute"
    ).run(iterations=5000)
    a = period * skew * 1e-6
    assert rel.final_drift_s == pytest.approx(4999 * a, rel=1e-9)
    assert absol.final_drift_s == pytest.approx(a / (1.0 + skew * 1e-6), rel=1e-9)
    assert rel.final_drift_s / absol.final_drift_s > 4000.0
    # Absolute mode: slope is roundoff, 2.4e-7 of the per-iteration error a.
    assert absol.drift_slope_s_per_iteration() == pytest.approx(0.0, abs=1e-12)


def test_zero_skew_zero_delay_gives_zero_drift_in_both_modes():
    for mode in ("absolute", "relative"):
        tb = SkewedSimulatedTimebase(skew_ppm=0.0, wake_delay_s=0.0)
        report = FixedRateLoop(period_s=1e-3, timebase=tb, mode=mode).run(iterations=100)
        # Exactly zero up to accumulation roundoff over 100 periods of 1 ms.
        assert report.max_abs_drift_s == pytest.approx(0.0, abs=1e-14)


def test_body_duration_is_recorded_and_fed_to_the_histogram():
    tb = SimulatedTimebase()
    costs = [1e-4, 2e-4, 3e-4, 4e-4]

    def body(k: int) -> None:
        tb.sleep(costs[k])

    report = FixedRateLoop(period_s=1e-3, timebase=tb).run(body=body, iterations=4)
    assert report.durations_s == pytest.approx(tuple(costs), rel=1e-12)
    hist = report.duration_histogram()
    assert hist.count == 4
    assert hist.maximum() == pytest.approx(4e-4, rel=1e-12)
    assert report.overruns().count == 0
    assert report.overruns(budget_s=2.5e-4).count == 2


def test_an_overrunning_body_produces_late_releases():
    tb = SimulatedTimebase()
    # A body costing 2 ms in a 1 ms loop: every release after the first is late.
    report = FixedRateLoop(period_s=1e-3, timebase=tb).run(
        body=lambda _k: tb.sleep(2e-3), iterations=10
    )
    assert report.late_releases == 9
    assert report.overruns().count == 10
    assert report.overruns().longest_consecutive_run == 10
    # Drift grows by 1 ms per iteration because the loop cannot keep up.
    assert report.final_drift_s == pytest.approx(9 * 1e-3, rel=1e-9)


def test_loop_validates_construction():
    tb = SimulatedTimebase()
    with pytest.raises(ValueError, match="period_s must be a finite value > 0"):
        FixedRateLoop(period_s=0.0, timebase=tb)
    with pytest.raises(ValueError, match="period_s must be a finite value > 0"):
        FixedRateLoop(period_s=float("inf"), timebase=tb)
    with pytest.raises(ValueError, match="mode must be"):
        FixedRateLoop(period_s=1e-3, timebase=tb, mode="fixed")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="now\\(\\) and sleep\\(\\)"):
        FixedRateLoop(period_s=1e-3, timebase=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"deadline_s must be in \(0, period_s"):
        FixedRateLoop(period_s=1e-3, timebase=tb, deadline_s=2e-3)
    with pytest.raises(ValueError, match=r"deadline_s must be in \(0, period_s"):
        FixedRateLoop(period_s=1e-3, timebase=tb, deadline_s=0.0)


def test_loop_validates_run_arguments():
    loop = FixedRateLoop(period_s=1e-3, timebase=SimulatedTimebase())
    with pytest.raises(ValueError, match="iterations must be >= 0"):
        loop.run(iterations=-1)
    empty = loop.run(iterations=0)
    assert empty.records == ()
    assert empty.max_abs_drift_s == 0.0
    assert empty.final_drift_s == 0.0
    assert empty.late_releases == 0
    with pytest.raises(ValueError, match="at least 2 iterations"):
        empty.drift_slope_s_per_iteration()


def test_deadline_defaults_to_period_and_can_be_tightened():
    tb = SimulatedTimebase()
    loop = FixedRateLoop(period_s=1e-3, timebase=tb, deadline_s=4e-4)
    report = loop.run(body=lambda _k: tb.sleep(5e-4), iterations=3)
    assert report.deadline_s == 4e-4
    assert report.overruns().count == 3


def test_closed_form_helpers_validate_input():
    with pytest.raises(ValueError, match="index must be >= 0"):
        relative_mode_drift(-1, 1e-3, 0.0, 0.0)
    with pytest.raises(ValueError, match="period_s must be > 0"):
        relative_mode_drift(1, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="index must be >= 0"):
        absolute_mode_drift(-1, 1e-3, 0.0, 0.0)
    with pytest.raises(ValueError, match="period_s must be > 0"):
        absolute_mode_drift(1, -1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match=r"\|skew_ppm\| must be < 1e6"):
        absolute_mode_drift(1, 1e-3, 1.0e6, 0.0)
    assert absolute_mode_drift(0, 1e-3, 100.0, 0.0) == 0.0
    assert relative_mode_drift(0, 1e-3, 100.0, 1.0) == 0.0


def test_report_carries_the_clock_quantum_when_given_one():
    res = MonotonicTimebase().resolution
    report = FixedRateLoop(period_s=1e-6, timebase=SimulatedTimebase()).run(
        iterations=3, clock_quantum_s=res.measured_tick_s
    )
    assert report.clock_quantum_s == res.measured_tick_s


def test_record_properties():
    tb = SimulatedTimebase()
    report = FixedRateLoop(period_s=1e-3, timebase=tb).run(
        body=lambda _k: tb.sleep(1e-4), iterations=3
    )
    rec = report.records[1]
    assert rec.completion_s == rec.end_s
    assert rec.late_release is False
    assert rec.wait_s == pytest.approx(9e-4, rel=1e-9)
