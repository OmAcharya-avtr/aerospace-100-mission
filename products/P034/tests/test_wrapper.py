"""Wrapper: transparency, stage ordering, monitor verdicts, restrictions."""

from __future__ import annotations

import math

import pytest

from faultinject.faults import SUBNORMAL_MIN, Injection
from faultinject.harness import fault_rng
from faultinject.target import GncController, SanitisingGncController
from faultinject.taxonomy import FaultKind
from faultinject.wrapper import (
    LARGE_MAGNITUDE,
    SMALLEST_NORMAL,
    InjectionWrapper,
    NumericalMonitor,
    classify_value,
)


def frames(n: int):
    return [{"pos": 0.1 * k, "vel": 0.02, "valid": 1.0} for k in range(n)]


def test_empty_wrapper_is_bit_transparent():
    bare = GncController()
    bare.reset()
    wrapped = InjectionWrapper(GncController(), [], fault_rng(1))
    wrapped.reset()
    for k, f in enumerate(frames(40)):
        assert wrapped.step(k, f)["u"] == bare.step(k, f)["u"]


def test_wrapper_calls_the_target_once_per_step():
    class Counter:
        def __init__(self):
            self.n = 0

        def reset(self):
            self.n = 0

        def step(self, k, meas):
            self.n += 1
            return {"u": 0.0}

    c = Counter()
    w = InjectionWrapper(c, [], fault_rng(1))
    w.reset()
    for k, f in enumerate(frames(13)):
        w.step(k, f)
    assert c.n == 13


def test_sensor_fault_only_inside_the_window():
    inj = Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 5.0}, 5, 3)
    bare = GncController()
    bare.reset()
    w = InjectionWrapper(GncController(), [inj], fault_rng(1))
    w.reset()
    diffs = []
    for k, f in enumerate(frames(12)):
        diffs.append(w.step(k, f)["u"] - bare.step(k, f)["u"])
    assert all(d == 0.0 for d in diffs[:5])
    assert any(d != 0.0 for d in diffs[5:8])


def test_overrun_holds_the_previous_command():
    inj = Injection.create(FaultKind.TIMING_OVERRUN, "bus", {"overrun_prob": 1.0}, 3, 4)
    w = InjectionWrapper(GncController(), [inj], fault_rng(1))
    w.reset()
    cmds = [w.step(k, f)["u"] for k, f in enumerate(frames(10))]
    assert cmds[3] == cmds[2]
    assert cmds[4] == cmds[2]
    assert w.event_counts()["overrun"] == 4


def test_dropout_marks_the_frame_invalid():
    inj = Injection.create(
        FaultKind.SENSOR_DROPOUT, "pos", {"dropout_prob": 1.0}, 0, 10
    )
    ctrl = GncController()
    w = InjectionWrapper(ctrl, [inj], fault_rng(1))
    w.reset()
    for k, f in enumerate(frames(10)):
        w.step(k, f)
    assert ctrl.skipped == 10
    assert w.event_counts()["dropout"] == 10


def test_at_most_one_transport_injection():
    a = Injection.create(FaultKind.BUS_DELAY, "bus", {"delay_steps": 2.0}, 0, 5)
    b = Injection.create(FaultKind.BUS_LOSS, "bus", {"loss_prob": 0.5}, 0, 5)
    with pytest.raises(ValueError, match="one TRANSPORT-stage"):
        InjectionWrapper(GncController(), [a, b], fault_rng(1))


def test_at_most_one_target_stage_injection():
    a = Injection.create(FaultKind.TIMING_OVERRUN, "bus", {"overrun_prob": 0.5}, 0, 5)
    with pytest.raises(ValueError, match="one TARGET-stage"):
        InjectionWrapper(GncController(), [a, a], fault_rng(1))


def test_two_sensor_injections_compose():
    a = Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 1.0}, 0, 10)
    b = Injection.create(FaultKind.SENSOR_BIAS, "vel", {"offset": 2.0}, 0, 10)
    w = InjectionWrapper(GncController(), [a, b], fault_rng(1))
    w.reset()
    w.step(0, {"pos": 0.0, "vel": 0.0, "valid": 1.0})
    assert w.history[0]["pos"] == 1.0
    assert w.history[0]["vel"] == 2.0


def test_classify_value_table():
    assert classify_value(float("nan")) == "nan"
    assert classify_value(float("inf")) == "inf"
    assert classify_value(-float("inf")) == "inf"
    assert classify_value(0.0) is None
    assert classify_value(1.0) is None
    assert classify_value(SMALLEST_NORMAL) is None
    assert classify_value(SUBNORMAL_MIN) == "subnormal"
    assert classify_value(LARGE_MAGNITUDE) is None
    assert classify_value(2.0 * LARGE_MAGNITUDE) == "large"


def test_monitor_verdicts():
    m = NumericalMonitor()
    assert m.verdict("nan") == "absent"
    m.check_in(4, {"pos": float("nan"), "vel": 0.0})
    assert m.verdict("nan") == "absorbed"
    m.check_out(4, {"u": float("nan")})
    assert m.verdict("nan") == "propagated"
    m2 = NumericalMonitor()
    m2.check_out(1, {"u": float("inf")})
    assert m2.verdict("inf") == "emitted"


def test_monitor_records_first_occurrence_only():
    m = NumericalMonitor()
    m.check_in(5, {"pos": float("nan"), "vel": 0.0})
    m.check_in(6, {"pos": float("nan"), "vel": 0.0})
    assert m.first_in["nan"] == (5, "pos")
    assert m.counts_in["nan"] == 2


def test_monitor_report_is_serialisable():
    m = NumericalMonitor()
    m.check_in(2, {"pos": SUBNORMAL_MIN, "vel": 0.0})
    rep = m.report()
    assert rep["detected"] is True
    assert rep["classes"] == ["subnormal"]
    assert rep["verdicts"]["subnormal"] == "absorbed"


def test_nan_injection_propagates_and_is_detected():
    inj = Injection.create(FaultKind.NUMERICAL_NAN, "pos", {}, 2, 3)
    w = InjectionWrapper(GncController(), [inj], fault_rng(1))
    w.reset()
    out = [w.step(k, f)["u"] for k, f in enumerate(frames(6))]
    assert math.isnan(out[2])
    assert w.monitor.verdict("nan") == "propagated"
    assert w.monitor.first_in["nan"] == (2, "pos")


def test_nan_injection_absorbed_by_a_sanitising_target():
    inj = Injection.create(FaultKind.NUMERICAL_NAN, "pos", {}, 2, 3)
    w = InjectionWrapper(SanitisingGncController(), [inj], fault_rng(1))
    w.reset()
    out = [w.step(k, f)["u"] for k, f in enumerate(frames(6))]
    assert all(math.isfinite(x) for x in out)
    assert w.monitor.detected is True
    assert w.monitor.verdict("nan") == "absorbed"


def test_reset_clears_everything():
    inj = Injection.create(FaultKind.NUMERICAL_NAN, "pos", {}, 0, 3)
    w = InjectionWrapper(GncController(), [inj], fault_rng(1))
    w.reset()
    for k, f in enumerate(frames(4)):
        w.step(k, f)
    assert w.events
    w.reset()
    assert w.events == []
    assert w.history == []
    assert w.monitor.detected is False
    assert w.delivered_source == []
