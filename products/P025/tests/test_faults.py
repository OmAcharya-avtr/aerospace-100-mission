"""Fault models: hand-checked injection, latching behaviour and validation."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fdiscope.faults import (
    ACTUATOR_FAULTS,
    FAULT_CLASSES,
    SENSOR_FAULTS,
    FaultSpec,
    FaultType,
    apply_actuator_fault,
    apply_sensor_fault,
    class_index,
)


class TestTaxonomy:
    def test_eight_classes_with_none_first(self):
        assert len(FAULT_CLASSES) == 8
        assert FAULT_CLASSES[0] is FaultType.NONE

    def test_sensor_and_actuator_sets_partition_the_faults(self):
        assert SENSOR_FAULTS.isdisjoint(ACTUATOR_FAULTS)
        assert SENSOR_FAULTS | ACTUATOR_FAULTS | {FaultType.NONE} == set(FAULT_CLASSES)

    def test_class_index_round_trips(self):
        for i, fault in enumerate(FAULT_CLASSES):
            assert class_index(fault) == i


class TestFaultSpecValidation:
    def test_rejects_negative_onset(self):
        with pytest.raises(ValueError, match="onset_step"):
            FaultSpec(FaultType.SENSOR_STUCK, -1)

    @pytest.mark.parametrize("bad", [-1, 2, 5])
    def test_rejects_bad_channel(self, bad):
        with pytest.raises(ValueError, match="channel"):
            FaultSpec(FaultType.SENSOR_STUCK, 0, 0.0, bad)

    def test_rejects_non_finite_magnitude(self):
        with pytest.raises(ValueError, match="finite"):
            FaultSpec(FaultType.SENSOR_BIAS, 0, float("nan"), 0)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_rejects_out_of_range_loss_of_effectiveness(self, bad):
        with pytest.raises(ValueError, match="loss-of-effectiveness|magnitude"):
            FaultSpec(FaultType.ACTUATOR_LOSS_OF_EFFECT, 0, bad, 0)

    def test_accepts_total_loss_of_effectiveness(self):
        assert FaultSpec(FaultType.ACTUATOR_LOSS_OF_EFFECT, 0, 1.0, 0).magnitude == 1.0

    @pytest.mark.parametrize("kind", [FaultType.SENSOR_BIAS, FaultType.SENSOR_DRIFT])
    def test_rejects_zero_magnitude_for_additive_faults(self, kind):
        with pytest.raises(ValueError, match="non-zero magnitude"):
            FaultSpec(kind, 0, 0.0, 0)

    def test_rejects_wrong_kind_type(self):
        with pytest.raises(TypeError, match="FaultType"):
            FaultSpec("sensor_bias", 0)

    def test_is_active_class(self):
        assert not FaultSpec().is_active_class
        assert FaultSpec(FaultType.SENSOR_STUCK, 0).is_active_class


class TestSensorFaults:
    def test_bias_known_answer(self):
        # z = (0.1, -0.2), bias +0.05 on channel 0 from step 10.
        spec = FaultSpec(FaultType.SENSOR_BIAS, 10, 0.05, 0)
        before, _ = apply_sensor_fault(np.array([0.1, -0.2]), spec, 9, 0.1, None)
        after, _ = apply_sensor_fault(np.array([0.1, -0.2]), spec, 10, 0.1, None)
        assert np.allclose(before, [0.1, -0.2])
        assert np.allclose(after, [0.15, -0.2])

    def test_drift_known_answer(self):
        # drift rate 0.02 per second on channel 1, onset 10, dt = 0.1:
        # at step 15 the elapsed time is (15 - 10) * 0.1 = 0.5 s
        # so the added error is 0.02 * 0.5 = 0.01
        spec = FaultSpec(FaultType.SENSOR_DRIFT, 10, 0.02, 1)
        out, _ = apply_sensor_fault(np.array([0.0, 1.0]), spec, 15, 0.1, None)
        assert np.allclose(out, [0.0, 1.01])

    def test_drift_is_zero_on_the_onset_sample(self):
        spec = FaultSpec(FaultType.SENSOR_DRIFT, 10, 0.02, 1)
        out, _ = apply_sensor_fault(np.array([0.0, 1.0]), spec, 10, 0.1, None)
        assert np.allclose(out, [0.0, 1.0])

    def test_stuck_latches_the_first_faulted_sample(self):
        spec = FaultSpec(FaultType.SENSOR_STUCK, 5, 0.0, 0)
        first, latch = apply_sensor_fault(np.array([7.0, 1.0]), spec, 5, 0.1, None)
        second, latch = apply_sensor_fault(np.array([9.0, 2.0]), spec, 6, 0.1, latch)
        assert first[0] == 7.0
        assert second[0] == 7.0
        assert second[1] == 2.0
        assert latch == 7.0

    def test_dropout_zeroes_the_channel(self):
        spec = FaultSpec(FaultType.SENSOR_DROPOUT, 0, 0.0, 1)
        out, _ = apply_sensor_fault(np.array([3.0, 4.0]), spec, 0, 0.1, None)
        assert np.allclose(out, [3.0, 0.0])

    def test_actuator_fault_leaves_the_measurement_alone(self):
        spec = FaultSpec(FaultType.ACTUATOR_STUCK, 0, 0.0, 0)
        out, _ = apply_sensor_fault(np.array([3.0, 4.0]), spec, 5, 0.1, None)
        assert np.allclose(out, [3.0, 4.0])

    def test_input_is_not_mutated(self):
        z = np.array([1.0, 2.0])
        spec = FaultSpec(FaultType.SENSOR_DROPOUT, 0, 0.0, 0)
        apply_sensor_fault(z, spec, 0, 0.1, None)
        assert np.allclose(z, [1.0, 2.0])


class TestActuatorFaults:
    def test_loss_of_effectiveness_known_answer(self):
        # u = 0.02 N m with 60 % loss -> 0.4 * 0.02 = 0.008 N m
        spec = FaultSpec(FaultType.ACTUATOR_LOSS_OF_EFFECT, 0, 0.6, 0)
        out, _ = apply_actuator_fault(0.02, spec, 0, 0.1, None)
        assert np.isclose(out, 0.008)

    def test_total_loss_of_effectiveness_gives_zero_torque(self):
        spec = FaultSpec(FaultType.ACTUATOR_LOSS_OF_EFFECT, 0, 1.0, 0)
        out, _ = apply_actuator_fault(0.02, spec, 0, 0.1, None)
        assert out == 0.0

    def test_loss_of_effectiveness_is_invisible_at_zero_command(self):
        # (1 - l) * 0 == 0 for every l: the documented blind spot.
        spec = FaultSpec(FaultType.ACTUATOR_LOSS_OF_EFFECT, 0, 0.9, 0)
        out, _ = apply_actuator_fault(0.0, spec, 0, 0.1, None)
        assert out == 0.0

    def test_stuck_latches_the_first_faulted_command(self):
        spec = FaultSpec(FaultType.ACTUATOR_STUCK, 3, 0.0, 0)
        first, latch = apply_actuator_fault(0.01, spec, 3, 0.1, None)
        second, latch = apply_actuator_fault(-0.02, spec, 4, 0.1, latch)
        assert first == 0.01
        assert second == 0.01

    def test_runaway_known_answer(self):
        # ramp 1e-3 N m/s from onset 100, dt = 0.1, evaluated at step 130:
        # elapsed = 3.0 s -> added torque 3e-3 N m
        spec = FaultSpec(FaultType.ACTUATOR_RUNAWAY, 100, 1e-3, 0)
        out, _ = apply_actuator_fault(0.005, spec, 130, 0.1, None)
        assert np.isclose(out, 0.008)

    def test_sensor_fault_leaves_the_command_alone(self):
        spec = FaultSpec(FaultType.SENSOR_BIAS, 0, 1.0, 0)
        out, _ = apply_actuator_fault(0.02, spec, 5, 0.1, None)
        assert out == 0.02

    def test_before_onset_nothing_happens(self):
        spec = FaultSpec(FaultType.ACTUATOR_RUNAWAY, 100, 1e-3, 0)
        out, latch = apply_actuator_fault(0.005, spec, 99, 0.1, None)
        assert out == 0.005
        assert latch is None


class TestProperties:
    @settings(max_examples=60, deadline=None)
    @given(
        u=st.floats(-0.1, 0.1, allow_nan=False),
        loss=st.floats(0.01, 1.0, allow_nan=False),
    )
    def test_loss_of_effectiveness_never_increases_the_magnitude(self, u, loss):
        spec = FaultSpec(FaultType.ACTUATOR_LOSS_OF_EFFECT, 0, loss, 0)
        out, _ = apply_actuator_fault(u, spec, 0, 0.1, None)
        assert abs(out) <= abs(u) + 1e-15

    @settings(max_examples=60, deadline=None)
    @given(
        bias=st.floats(-1.0, 1.0, allow_nan=False).filter(lambda v: abs(v) > 1e-9),
        step=st.integers(0, 50),
    )
    def test_bias_shifts_exactly_one_channel(self, bias, step):
        spec = FaultSpec(FaultType.SENSOR_BIAS, 0, bias, 0)
        z = np.array([0.3, -0.4])
        out, _ = apply_sensor_fault(z, spec, step, 0.1, None)
        assert np.isclose(out[0] - z[0], bias)
        assert out[1] == z[1]
