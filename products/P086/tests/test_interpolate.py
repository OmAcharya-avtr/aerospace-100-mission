"""Reconstruction: the grid, the fills, the refusals and the error measurement."""

from __future__ import annotations

import numpy as np
import pytest

from telemgap import classify, dataset, policy
from telemgap.interpolate import (
    grid_for,
    measured_error,
    nearest_indices,
    reconstruct,
)
from telemgap.record import RecordSchema, TelemetryRecord


def test_grid_known_answer(small_schema: RecordSchema, record_builder) -> None:
    # Frames at t = 0, 1, 6: the grid is 0 .. 6 at cadence 1, so seven slots.
    record = record_builder(
        small_schema, [40] * 3, timestamps=[0.0, 1.0, 6.0], counters=[0, 1, 6]
    )
    times = grid_for(record)
    assert times.size == 7
    assert times[0] == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert times[-1] == pytest.approx(6.0, rel=1e-12, abs=1e-15)


def test_nearest_indices_known_answers() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0])
    query = np.array([-5.0, 0.4, 0.6, 1.5, 2.9, 99.0])
    # 1.5 is equidistant from 1.0 and 2.0; ties go to the lower index.
    assert list(nearest_indices(times, query)) == [0, 0, 1, 1, 3, 3]
    with pytest.raises(ValueError, match="must not be empty"):
        nearest_indices(np.empty(0), query)


def test_reconstruct_fills_a_permitted_gap(small_schema: RecordSchema, record_builder) -> None:
    # Frames at t = 0, 1, 4, 5 with values 30.0: a 3 s gap with two interior
    # slots. With a large tolerance the policy permits linear interpolation.
    record = record_builder(
        small_schema, [40, 42, 46, 48], timestamps=[0.0, 1.0, 4.0, 5.0], counters=[0, 1, 4, 5]
    )
    report = classify.classify(record)
    plan = policy.plan(record, report, {"probe": 100.0})
    assert len(plan.decisions) == 1
    assert not plan.decisions[0].refused
    built = reconstruct(record, report.findings.gaps, plan)
    assert built.n_slots == 6
    assert int(built.observed.sum()) == 4
    assert int(built.synthesised.sum()) == 2
    assert int(built.refused.sum()) == 0
    assert np.all(np.isfinite(built.values))
    assert built.fill_fraction == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert built.refused_fraction == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_reconstruct_leaves_a_refused_gap_empty(
    small_schema: RecordSchema, record_builder
) -> None:
    record = record_builder(
        small_schema, [40, 42, 46, 48], timestamps=[0.0, 1.0, 4.0, 5.0], counters=[0, 1, 4, 5]
    )
    report = classify.classify(record)
    plan = policy.plan(record, report, {"probe": 1e-9})
    assert plan.decisions[0].refused
    built = reconstruct(record, report.findings.gaps, plan)
    assert int(built.synthesised.sum()) == 0
    assert int(built.refused.sum()) == 2
    assert np.isnan(built.values[2, 0])
    assert np.isnan(built.values[3, 0])
    assert built.method_used[(0, 0)] == "none"
    assert built.refused_fraction == pytest.approx(2.0 / 6.0, rel=1e-12, abs=0.0)


def test_linear_fill_is_the_chord(small_schema: RecordSchema, record_builder) -> None:
    # Raw 40, 42, 46, 48 decode to 30, 31, 33, 34 K, a gentle ramp that trips
    # no screen, so the endpoint gate passes. Frames sit at t = 0, 1, 4, 5, so
    # the gap runs from t = 1 (31.0) to t = 4 (33.0) and the interior slots at
    # t = 2 and t = 3 must read 31 + 2/3 and 31 + 4/3.
    record = record_builder(
        small_schema, [40, 42, 46, 48], timestamps=[0.0, 1.0, 4.0, 5.0], counters=[0, 1, 4, 5]
    )
    report = classify.classify(record)
    assert report.anomalies == tuple(
        anomaly for anomaly in report.anomalies if anomaly.observation == "gap"
    )
    plan = policy.plan(record, report, {"probe": 100.0})
    built = reconstruct(record, report.findings.gaps, plan)
    assert built.method_used[(0, 0)] == "linear"
    assert built.values[2, 0] == pytest.approx(31.0 + 2.0 / 3.0, rel=1e-12, abs=0.0)
    assert built.values[3, 0] == pytest.approx(31.0 + 4.0 / 3.0, rel=1e-12, abs=0.0)


def test_hold_fill_repeats_the_left_endpoint(
    small_schema: RecordSchema, record_builder
) -> None:
    import dataclasses

    # A channel with no declared curvature and a tiny slew, so the policy has
    # no measurable curvature and falls through to the hold method.
    channel = dataclasses.replace(
        small_schema.channels[0], max_curvature=0.0, max_slew=1e-9, noise_sigma=1e-9
    )
    schema = dataclasses.replace(small_schema, channels=(channel,))
    record = record_builder(
        schema, [40, 42, 46, 48], timestamps=[0.0, 1.0, 4.0, 5.0], counters=[0, 1, 4, 5]
    )
    report = classify.classify(record)
    plan = policy.plan(record, report, {"probe": 1.0})
    assert plan.decisions[0].method == "hold"
    built = reconstruct(record, report.findings.gaps, plan)
    assert built.values[2, 0] == pytest.approx(31.0, rel=1e-12, abs=0.0)
    assert built.values[3, 0] == pytest.approx(31.0, rel=1e-12, abs=0.0)


def test_reconstruct_rejects_a_mismatched_plan(
    small_schema: RecordSchema, record_builder
) -> None:
    record = record_builder(
        small_schema, [40, 42, 46, 48], timestamps=[0.0, 1.0, 4.0, 5.0], counters=[0, 1, 4, 5]
    )
    report = classify.classify(record)
    plan = policy.plan(record, report, {"probe": 100.0})
    with pytest.raises(ValueError, match="disagree"):
        reconstruct(record, (), plan)


def test_slot_collisions_are_counted(small_schema: RecordSchema, record_builder) -> None:
    # Two frames inside one cadence slot with different counters: the later one
    # is discarded from the grid and the discard is reported, never silent.
    record = record_builder(
        small_schema, [40] * 3, timestamps=[0.0, 0.05, 1.0], counters=[0, 1, 2]
    )
    report = classify.classify(record)
    plan = policy.plan(record, report, {"probe": 100.0})
    built = reconstruct(record, report.findings.gaps, plan)
    assert built.slot_collisions == 1


def test_measured_error_is_nan_without_synthesised_slots(
    small_schema: RecordSchema, record_builder
) -> None:
    record = record_builder(small_schema, [40] * 5)
    report = classify.classify(record)
    plan = policy.plan(record, report, {"probe": 100.0})
    built = reconstruct(record, report.findings.gaps, plan)
    error, count = measured_error(built, np.arange(5.0), np.full(5, 30.0), 0)
    assert np.isnan(error)
    assert count == 0


def test_measured_error_known_answer(small_schema: RecordSchema, record_builder) -> None:
    # Endpoints 31.0 and 33.0 with a truth that is flat at 30.0 inside the
    # gap: the chord reads 31.667 and 32.333, so the maximum error is 7/3.
    record = record_builder(
        small_schema, [40, 42, 46, 48], timestamps=[0.0, 1.0, 4.0, 5.0], counters=[0, 1, 4, 5]
    )
    report = classify.classify(record)
    plan = policy.plan(record, report, {"probe": 100.0})
    built = reconstruct(record, report.findings.gaps, plan)
    truth_times = np.arange(6.0)
    truth_values = np.full(6, 30.0)
    error, count = measured_error(built, truth_times, truth_values, 0)
    assert count == 2
    assert error == pytest.approx(7.0 / 3.0, rel=1e-12, abs=0.0)


def test_measured_error_validates_shapes(small_schema: RecordSchema, record_builder) -> None:
    record = record_builder(small_schema, [40] * 5)
    report = classify.classify(record)
    plan = policy.plan(record, report, {"probe": 100.0})
    built = reconstruct(record, report.findings.gaps, plan)
    with pytest.raises(ValueError, match="same shape"):
        measured_error(built, np.arange(5.0), np.zeros(4), 0)


def test_declared_bound_holds_on_a_generated_record() -> None:
    # The policy's realised error must stay inside what it predicted, on the
    # two channels whose curvature is declared. A record with no clock reset,
    # because a clock reset moves the record off the generation time base.
    generated = dataset.generate(samples=2500, seed=515, n_clock_resets=0)
    record = generated.record
    report = classify.classify(record)
    tolerances = {
        "battery_voltage": 0.05,
        "valve_current": 0.05,
        "gyro_rate": 0.02,
        "tank_pressure": 0.02,
    }
    plan = policy.plan(record, report, tolerances)
    built = reconstruct(record, report.findings.gaps, plan)
    for column, channel in enumerate(record.schema.channels):
        if channel.max_curvature <= 0.0:
            continue
        error, count = measured_error(
            built, generated.truth_times, generated.truth_values[:, column], column
        )
        if count == 0:
            continue
        permitted = [
            decision.predicted_error
            for decision in plan.decisions
            if decision.channel == column and not decision.refused
        ]
        assert error <= max(permitted) * (1.0 + 1e-9), channel.name


def test_reconstruction_of_a_record_with_no_gaps(nominal_record: TelemetryRecord) -> None:
    report = classify.classify(nominal_record)
    plan = policy.plan(nominal_record, report, {"probe": 1.0})
    assert plan.decisions == ()
    built = reconstruct(nominal_record, report.findings.gaps, plan)
    assert built.n_slots == 40
    assert int(built.refused.sum()) == 0
    assert built.method_used == {}
