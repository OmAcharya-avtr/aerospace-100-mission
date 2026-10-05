"""Harness: determinism, stream separation, aborts, trace bookkeeping."""

from __future__ import annotations

import math

import pytest

from faultinject.faults import Injection
from faultinject.harness import (
    fault_rng,
    noise_stream,
    nominal_trace,
    run_case,
)
from faultinject.taxonomy import FaultKind


def test_noise_stream_shape_and_determinism():
    a = noise_stream(5, 20)
    b = noise_stream(5, 20)
    assert a.shape == (20, 4)
    assert a.tobytes() == b.tobytes()
    assert noise_stream(6, 20).tobytes() != a.tobytes()


def test_fault_stream_is_independent_of_the_noise_stream():
    # Separate substreams: [seed, 1] vs [seed, 2].
    g = fault_rng(5)
    first = [float(g.random()) for _ in range(4)]
    assert first != list(noise_stream(5, 4)[:, 0])


def test_nominal_run_is_finite_and_tracks(nominal):
    assert nominal.finite()
    assert nominal.updates == 150
    assert nominal.skipped == 0
    assert max(abs(e) for e in nominal.tracking_error) < 1.0


def test_nominal_cache_returns_equal_traces():
    a = nominal_trace(21, 50)
    b = nominal_trace(21, 50)
    assert a.float_bytes() == b.float_bytes()


def test_trace_lengths(bias_trace):
    for seq in (
        bias_trace.p_true,
        bias_trace.v_true,
        bias_trace.p_hat,
        bias_trace.v_hat,
        bias_trace.u_applied,
        bias_trace.ref,
    ):
        assert len(seq) == 150


def test_bias_moves_the_trace(bias_trace, nominal):
    assert bias_trace.float_bytes() != nominal.float_bytes()
    dev = max(
        abs(a - b) for a, b in zip(bias_trace.p_true, nominal.p_true, strict=True)
    )
    assert dev > 0.1


def test_replay_is_bit_identical(bias_injection):
    a = run_case([bias_injection], 7, 150)
    b = run_case([bias_injection], 7, 150)
    assert a.float_bytes() == b.float_bytes()


def test_nan_injection_aborts_and_pads():
    inj = Injection.create(FaultKind.NUMERICAL_NAN, "pos", {}, 50, 10)
    tr = run_case([inj], 3, 150)
    assert tr.aborted_step == 50
    assert len(tr.p_true) == 150
    assert not tr.finite()
    assert math.isnan(tr.p_true[-1])


def test_abort_can_be_disabled():
    inj = Injection.create(FaultKind.NUMERICAL_NAN, "pos", {}, 50, 10)
    tr = run_case([inj], 3, 150, abort_on_nonfinite_state=False)
    assert tr.aborted_step is None
    assert len(tr.p_true) == 150


def test_dropout_reduces_updates():
    inj = Injection.create(
        FaultKind.SENSOR_DROPOUT, "pos", {"dropout_prob": 1.0}, 0, 150
    )
    tr = run_case([inj], 3, 150)
    assert tr.updates == 0
    assert tr.skipped == 150


def test_run_case_rejects_zero_steps():
    with pytest.raises(ValueError, match="n_steps"):
        run_case((), 1, 0)


def test_float_bytes_length_is_six_arrays_plus_innovations(nominal):
    expected = 8 * (6 * 150 + 3 * len(nominal.innovations))
    assert len(nominal.float_bytes()) == expected
