"""Fault handlers: known answers, state behaviour and validation."""

from __future__ import annotations

import math

import pytest

from faultinject.faults import (
    SUBNORMAL_MIN,
    Injection,
    Stage,
    handler_coverage,
    make_handler,
    stage_of,
)
from faultinject.taxonomy import FaultKind, kinds


def h(kind, channel, params, start=10, duration=20):
    inj = Injection.create(kind, channel, params, start, duration)
    handler = make_handler(inj)
    handler.reset()
    return inj, handler


def test_every_kind_has_a_handler():
    assert set(handler_coverage()) == set(kinds())


def test_stage_assignment():
    assert stage_of(FaultKind.SENSOR_BIAS, "pos") is Stage.SENSOR
    assert stage_of(FaultKind.ACTUATOR_STUCK, "u") is Stage.ACTUATOR
    assert stage_of(FaultKind.BUS_LOSS, "bus") is Stage.TRANSPORT
    assert stage_of(FaultKind.TIMING_OVERRUN, "bus") is Stage.TARGET
    assert stage_of(FaultKind.NUMERICAL_NAN, "pos") is Stage.SENSOR
    assert stage_of(FaultKind.NUMERICAL_NAN, "u") is Stage.ACTUATOR


def test_bias_known_answer(rng):
    # 1.0 + 0.25 = 1.25 exactly in binary64
    _, handler = h(FaultKind.SENSOR_BIAS, "pos", {"offset": 0.25})
    assert handler.apply_signal(10, 1.0, rng) == 1.25


def test_drift_known_answer(rng):
    # rate 0.01 per step, start 10; at step 13 n = 13 - 10 + 1 = 4 -> +0.04
    _, handler = h(FaultKind.SENSOR_DRIFT, "pos", {"rate": 0.01})
    assert handler.apply_signal(13, 2.0, rng) == pytest.approx(2.04, abs=1e-15)


def test_stuck_holds_the_last_observed_value(rng):
    _, handler = h(FaultKind.SENSOR_STUCK, "pos", {})
    handler.observe_signal(3.5)
    assert handler.apply_signal(10, 99.0, rng) == 3.5
    assert handler.apply_signal(11, -4.0, rng) == 3.5


def test_stuck_without_prior_observation_uses_first_value(rng):
    _, handler = h(FaultKind.SENSOR_STUCK, "pos", {})
    assert handler.apply_signal(10, 7.0, rng) == 7.0
    assert handler.apply_signal(11, 8.0, rng) == 7.0


def test_quant_collapse_known_answer(rng):
    # lsb 0.5, value 1.3: floor(1.3/0.5 + 0.5) = floor(3.1) = 3 -> 1.5
    _, handler = h(FaultKind.SENSOR_QUANT_COLLAPSE, "pos", {"lsb": 0.5})
    assert handler.apply_signal(10, 1.3, rng) == pytest.approx(1.5, abs=1e-15)
    # value -1.3: floor(-2.6 + 0.5) = floor(-2.1) = -3 -> -1.5
    assert handler.apply_signal(10, -1.3, rng) == pytest.approx(-1.5, abs=1e-15)


def test_dropout_probability_one_always_drops(rng):
    _, handler = h(FaultKind.SENSOR_DROPOUT, "pos", {"dropout_prob": 1.0})
    for k in range(10):
        handler.apply_signal(k, 1.0, rng)
        assert handler.dropped_this_step is True
    assert handler.drops == 10


def test_dropout_low_probability_mostly_keeps(rng):
    _, handler = h(FaultKind.SENSOR_DROPOUT, "pos", {"dropout_prob": 0.05})
    for k in range(200):
        handler.apply_signal(k, 1.0, rng)
    assert 0 <= handler.drops <= 30


def test_nan_denormal_overflow_values(rng):
    _, nan_h = h(FaultKind.NUMERICAL_NAN, "pos", {})
    assert math.isnan(nan_h.apply_signal(10, 1.0, rng))
    _, den_h = h(FaultKind.NUMERICAL_DENORMAL, "pos", {})
    assert den_h.apply_signal(10, 1.0, rng) == SUBNORMAL_MIN
    _, ovf_h = h(FaultKind.NUMERICAL_OVERFLOW, "pos", {"magnitude": 1e200})
    assert ovf_h.apply_signal(10, 1.0, rng) == 1e200
    assert ovf_h.apply_signal(10, -1.0, rng) == -1e200


def test_loss_of_effectiveness_known_answer(rng):
    _, handler = h(FaultKind.ACTUATOR_LOSS_EFFECTIVENESS, "u", {"retained": 0.3})
    assert handler.apply_signal(10, 2.0, rng) == pytest.approx(0.6, abs=1e-15)


def test_runaway_known_answer(rng):
    # base 2.0 observed before the window, slew 1.0, start 10;
    # step 12 -> n = 3 -> 2.0 + 3.0 = 5.0
    _, handler = h(FaultKind.ACTUATOR_RUNAWAY, "u", {"slew": 1.0})
    handler.observe_signal(2.0)
    assert handler.apply_signal(12, 0.0, rng) == pytest.approx(5.0, abs=1e-15)


def test_runaway_sign_follows_the_base_command(rng):
    _, handler = h(FaultKind.ACTUATOR_RUNAWAY, "u", {"slew": 2.0})
    handler.observe_signal(-1.0)
    assert handler.apply_signal(10, 0.0, rng) == pytest.approx(-3.0, abs=1e-15)


def test_delay_queue_is_fifo_and_persistent(rng):
    inj, handler = h(FaultKind.BUS_DELAY, "bus", {"delay_steps": 2.0}, 0, 5)
    history = []
    sources = []
    for k in range(10):
        frame = {"pos": float(k), "vel": 0.0, "valid": 1.0}
        history.append(frame)
        _, src = handler.deliver(k, frame, history, rng, inj.active(k))
        sources.append(src)
    # Steps 0 and 1 deliver nothing; from step 2 on the frame is 2 steps stale,
    # and the latency persists after the window closes at step 5.
    assert sources[:3] == [None, None, 0]
    assert sources[5:] == [3, 4, 5, 6, 7]


def test_loss_probability_one_drops_every_frame(rng):
    inj, handler = h(FaultKind.BUS_LOSS, "bus", {"loss_prob": 1.0}, 0, 5)
    for k in range(5):
        frame = {"pos": float(k), "vel": 0.0, "valid": 1.0}
        _, src = handler.deliver(k, frame, [frame], rng, inj.active(k))
        assert src is None
    assert handler.losses == 5


def test_late_sample_returns_fresh_after_window(rng):
    inj, handler = h(FaultKind.TIMING_LATE_SAMPLE, "bus", {"late_steps": 3.0}, 2, 4)
    history = []
    sources = []
    for k in range(10):
        frame = {"pos": float(k), "vel": 0.0, "valid": 1.0}
        history.append(frame)
        _, src = handler.deliver(k, frame, history, rng, inj.active(k))
        sources.append(src)
    assert sources[0:2] == [0, 1]
    assert sources[3:6] == [0, 1, 2]
    assert sources[6:] == [6, 7, 8, 9]


def test_overrun_probability_one_always_suppresses(rng):
    inj, handler = h(FaultKind.TIMING_OVERRUN, "bus", {"overrun_prob": 1.0}, 0, 5)
    assert all(handler.suppress(k, rng) for k in range(5))
    assert handler.overruns == 5


def test_injection_validation():
    with pytest.raises(ValueError, match="start_step"):
        Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 1.0}, -1, 10)
    with pytest.raises(ValueError, match="duration_steps"):
        Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 1.0}, 0, 0)


def test_injection_active_window():
    inj = Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 1.0}, 10, 5)
    assert inj.end_step == 15
    assert not inj.active(9)
    assert inj.active(10)
    assert inj.active(14)
    assert not inj.active(15)


def test_injection_round_trip():
    inj = Injection.create(FaultKind.BUS_DELAY, "bus", {"delay_steps": 4.0}, 3, 7)
    assert Injection.from_dict(inj.to_dict()) == inj


def test_injection_is_hashable():
    inj = Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 1.0}, 0, 1)
    assert len({inj, inj}) == 1


def test_make_handler_covers_the_taxonomy():
    for kind in kinds():
        from faultinject.taxonomy import spec

        sp = spec(kind)
        params = {p.name: p.representative(0) for p in sp.params}
        inj = Injection.create(kind, sp.channels[0], params, 0, 5)
        assert make_handler(inj) is not None


def test_unimplemented_interfaces_raise(rng):
    _, bias = h(FaultKind.SENSOR_BIAS, "pos", {"offset": 1.0})
    with pytest.raises(NotImplementedError):
        bias.deliver(0, {}, [], rng, True)
    with pytest.raises(NotImplementedError):
        bias.suppress(0, rng)
    _, delay = h(FaultKind.BUS_DELAY, "bus", {"delay_steps": 1.0})
    with pytest.raises(NotImplementedError):
        delay.apply_signal(0, 1.0, rng)
