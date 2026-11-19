"""The trace simulator: conservation, phase partition, determinism, validation."""

from __future__ import annotations

import numpy as np
import pytest

from handoverloss.baselines import fluid_decomposition
from handoverloss.sequence import HandoverSequence
from handoverloss.trace import EventParameters, simulate


def _parameters(sequences: list[HandoverSequence]) -> EventParameters:
    return EventParameters.from_sequences(sequences)


@pytest.fixture
def one_event(worked_sequence) -> EventParameters:
    return _parameters([worked_sequence])


def test_conservation_identity_holds(one_event, rng):
    # produced = delivered + (q_end - q_0) + dropped, exactly, in exact
    # arithmetic. Asserted against the accumulation error of the slot sums
    # rather than bit-exactly.
    result = simulate(one_event, rng)
    left = float(result.produced_bits[0])
    right = float(
        result.delivered_bits[0]
        + (result.final_occupancy_bits[0] - result.initial_occupancy_bits[0])
        + result.total_dropped_bits[0]
    )
    slack = (
        4.0
        * result.slot_count
        * (2.0**-53)
        * (left + float(result.delivered_bits[0]))
    )
    assert abs(left - right) <= slack


def test_conservation_total_matches_the_bucket_sum(one_event, rng):
    result = simulate(one_event, rng)
    bucketed = float(result.total_dropped_bits[0])
    conserved = float(result.conservation_total_bits[0])
    slack = 4.0 * result.slot_count * (2.0**-53) * (
        float(result.produced_bits[0]) + float(result.delivered_bits[0]) + bucketed
    )
    assert abs(conserved - bucketed) <= slack


def test_deterministic_for_a_fixed_seed(one_event):
    first = simulate(one_event, np.random.default_rng(5))
    second = simulate(one_event, np.random.default_rng(5))
    # The generator stream is deterministic, so the arrays are the same objects
    # of arithmetic performed in the same order; bit-exact equality is correct
    # here and is NOT a claim about a floating-point pipeline's stability.
    assert np.array_equal(first.dropped_bits, second.dropped_bits)
    assert np.array_equal(first.produced_bits, second.produced_bits)


def test_constant_source_and_service_reproduces_the_fluid_model(worked_sequence):
    # With burst and jitter shapes zero the simulator IS a discretisation of the
    # fluid model, so the two must agree to within the discretisation error. The
    # tolerance is the fluid model's own bound on that error: one slot's worth of
    # source output per phase boundary, of which there are three.
    parameters = _parameters([worked_sequence])
    result = simulate(
        parameters,
        np.random.default_rng(1),
        slot_seconds=0.05,
        burst_log_sigma=0.0,
        service_log_sigma=0.0,
    )
    closed = fluid_decomposition(worked_sequence)
    tolerance = 3.0 * worked_sequence.source_rate_bps * 0.05
    for column, expected in enumerate(
        (closed.acquisition_bits, closed.restart_bits, closed.drain_bits)
    ):
        assert abs(float(result.dropped_bits[0, column]) - expected) <= tolerance


def test_refusing_an_overlong_window(one_event):
    with pytest.raises(ValueError, match="compute budget"):
        simulate(one_event, np.random.default_rng(1), slot_seconds=1.0e-5)


def test_no_loss_when_the_buffer_is_ample():
    sequence = HandoverSequence(
        source_rate_bps=1.0e7,
        downlink_rate_bps=2.0e9,
        buffer_capacity_bits=1.0e12,
        initial_occupancy_bits=0.0,
        acquisition_s=20.0,
        restart_s=10.0,
        ramp_s=10.0,
        horizon_s=200.0,
    )
    result = simulate(_parameters([sequence]), np.random.default_rng(3))
    assert float(result.total_dropped_bits[0]) == 0.0
    assert int(result.slots_full[0]) == 0


def test_dominant_names_match_the_argmax(rng):
    sequences = [
        HandoverSequence(
            source_rate_bps=2.0e8,
            downlink_rate_bps=1.0e9,
            buffer_capacity_bits=3.2e10,
            initial_occupancy_bits=2.9e10,
            acquisition_s=acquisition,
            restart_s=12.0,
            ramp_s=60.0,
            horizon_s=700.0,
        )
        for acquisition in (5.0, 40.0, 100.0)
    ]
    result = simulate(_parameters(sequences), rng)
    for row, name in enumerate(result.dominant_names()):
        assert name == ("acquisition", "restart", "drain")[int(result.dominant_index[row])]


def test_slot_count_is_recorded(one_event, rng):
    result = simulate(one_event, rng, slot_seconds=0.5)
    # 700 s window at 0.5 s slots.
    assert result.slot_count == 1400
    assert result.slot_seconds == 0.5


def test_mismatched_array_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        EventParameters(
            source_rate_bps=np.array([1e8, 2e8]),
            downlink_rate_bps=np.array([1e9]),
            buffer_capacity_bits=np.array([1e10]),
            initial_occupancy_bits=np.array([0.0]),
            acquisition_s=np.array([10.0]),
            restart_s=np.array([5.0]),
            ramp_s=np.array([5.0]),
            horizon_s=100.0,
        )


def test_event_parameters_reject_overfull_buffer():
    with pytest.raises(ValueError, match="exceeds"):
        EventParameters(
            source_rate_bps=np.array([1e8]),
            downlink_rate_bps=np.array([1e9]),
            buffer_capacity_bits=np.array([1e10]),
            initial_occupancy_bits=np.array([2e10]),
            acquisition_s=np.array([10.0]),
            restart_s=np.array([5.0]),
            ramp_s=np.array([5.0]),
            horizon_s=100.0,
        )


def test_event_parameters_reject_short_window():
    with pytest.raises(ValueError, match="shorter than"):
        EventParameters(
            source_rate_bps=np.array([1e8]),
            downlink_rate_bps=np.array([1e9]),
            buffer_capacity_bits=np.array([1e10]),
            initial_occupancy_bits=np.array([0.0]),
            acquisition_s=np.array([40.0]),
            restart_s=np.array([20.0]),
            ramp_s=np.array([30.0]),
            horizon_s=50.0,
        )


def test_event_parameters_reject_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        EventParameters(
            source_rate_bps=np.array([np.nan]),
            downlink_rate_bps=np.array([1e9]),
            buffer_capacity_bits=np.array([1e10]),
            initial_occupancy_bits=np.array([0.0]),
            acquisition_s=np.array([10.0]),
            restart_s=np.array([5.0]),
            ramp_s=np.array([5.0]),
            horizon_s=100.0,
        )


def test_event_parameters_reject_non_positive_horizon():
    with pytest.raises(ValueError, match="horizon_s must be positive"):
        EventParameters(
            source_rate_bps=np.array([1e8]),
            downlink_rate_bps=np.array([1e9]),
            buffer_capacity_bits=np.array([1e10]),
            initial_occupancy_bits=np.array([0.0]),
            acquisition_s=np.array([0.0]),
            restart_s=np.array([0.0]),
            ramp_s=np.array([0.0]),
            horizon_s=0.0,
        )


def test_from_sequences_requires_a_common_horizon(worked_sequence):
    other = HandoverSequence(
        source_rate_bps=2.0e8,
        downlink_rate_bps=1.0e9,
        buffer_capacity_bits=3.2e10,
        initial_occupancy_bits=2.9e10,
        acquisition_s=40.0,
        restart_s=12.0,
        ramp_s=60.0,
        horizon_s=800.0,
    )
    with pytest.raises(ValueError, match="share one horizon_s"):
        EventParameters.from_sequences([worked_sequence, other])


def test_from_sequences_requires_at_least_one():
    with pytest.raises(ValueError, match="at least one"):
        EventParameters.from_sequences([])


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_non_positive_slot_width_raises(one_event, value):
    with pytest.raises(ValueError, match="strictly positive"):
        simulate(one_event, np.random.default_rng(1), slot_seconds=value)


def test_negative_shape_parameters_raise(one_event):
    with pytest.raises(ValueError, match="non-negative"):
        simulate(one_event, np.random.default_rng(1), burst_log_sigma=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        simulate(one_event, np.random.default_rng(1), service_log_sigma=-0.1)
