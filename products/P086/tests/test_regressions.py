"""Pinned end-to-end numbers, and every counterexample this build found.

Why these are committed
-----------------------
A number that moves without anyone noticing is the defect class this file
exists to catch. Each value below was produced by a run in the build session,
is deterministic in a fixed seed, and is asserted here with the tolerance the
quantity deserves: integer counts exactly, floating-point quantities to a
relative tolerance with an explicit absolute tolerance, never to bit equality.

The ``.hypothesis`` directory is not committed, so a falsifying example found
by one Hypothesis run is lost on a fresh clone. The counterexamples this build
found are therefore pinned here with their literal inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from telemgap import classify, dataset, policy, screens
from telemgap.interpolate import measured_error, nearest_indices, reconstruct
from telemgap.policy import estimator_bandwidth_fraction, estimator_transfer
from telemgap.record import ChannelSpec, RecordSchema, TelemetryRecord
from telemgap.screens import robust_local_statistics

#: The pinned record: 1,500 frames, seed 4242, six gap runs, forty corruptions.
PINNED = {"samples": 1500, "seed": 4242, "n_gaps": 6, "n_corruptions": 40}


@pytest.fixture(scope="module")
def pinned_record() -> dataset.GeneratedRecord:
    return dataset.generate(**PINNED)


def test_pinned_ground_truth_counts(pinned_record: dataset.GeneratedRecord) -> None:
    assert pinned_record.truth_counts() == {
        "clean": 5863,
        "corrupt": 41,
        "misparsed": 92,
        "corrupt_in_range": 27,
        "corrupt_out_of_range": 14,
        "duplicate": 12,
        "reorder": 12,
        "clock_reset": 2,
        "missing_frames": 13,
        "frames": 1499,
    }


def test_pinned_screen_summary(pinned_record: dataset.GeneratedRecord) -> None:
    assert screens.screen(pinned_record.record).summary() == {
        "frames": 1499,
        "gaps": 6,
        "missing_frames": 13,
        "duplicates": 12,
        "reorders": 12,
        "clock_resets": 2,
        "cadence_violations": 0,
        "range_violations": 68,
        "residual_violations": 112,
        "validity_failures": 27,
    }


def test_pinned_verdict_counts(pinned_record: dataset.GeneratedRecord) -> None:
    strengthened = classify.classify(pinned_record.record, mode="strengthened")
    specified = classify.classify(pinned_record.record, mode="specified")
    assert strengthened.counts == {
        "missing": 6,
        "corrupt": 46,
        "misparsed": 116,
        "undecided": 92,
    }
    assert specified.counts == {"missing": 6, "corrupt": 38, "misparsed": 80, "undecided": 100}
    assert len(strengthened.anomalies) == 260
    assert len(specified.anomalies) == 224


def test_pinned_determinability_quantities(pinned_record: dataset.GeneratedRecord) -> None:
    verdict = classify.classify(pinned_record.record).determinability
    # modulus 16384, horizon 8192, cadence 1 s.
    assert verdict["counter_horizon"].quantity == pytest.approx(8192.0, rel=0.0, abs=0.0)
    # Enumerated over all 65,536 word codes of the worst channel.
    assert verdict["in_range_corruption"].quantity == pytest.approx(
        0.8623705051797929, rel=1e-12, abs=0.0
    )
    assert verdict["gap_interior"].quantity == pytest.approx(2.0, rel=0.0, abs=0.0)


def test_pinned_curvature_estimates(pinned_record: dataset.GeneratedRecord) -> None:
    estimates = policy.channel_curvatures(pinned_record.record)
    assert estimates["battery_voltage"].source == "declared"
    assert estimates["tank_pressure"].source == "declared"
    assert estimates["valve_current"].measurable is False
    assert estimates["gyro_rate"].measurable is True
    assert estimates["gyro_rate"].resolvable is False
    assert estimates["valve_current"].resolvable is False
    assert estimates["gyro_rate"].value == pytest.approx(0.00402404, rel=1e-4, abs=0.0)
    assert estimates["gyro_rate"].high_frequency_fraction == pytest.approx(
        0.0422035, rel=1e-3, abs=0.0
    )


def test_pinned_near_nyquist_under_prediction(pinned_record: dataset.GeneratedRecord) -> None:
    # The headline finding, pinned. The estimated curvature of the near-Nyquist
    # channel is about one per cent of the analytic bound, and the estimator's
    # closed-form transfer function at f T = 0.35 explains why.
    estimates = policy.channel_curvatures(pinned_record.record)
    analytic = dataset.MODELS[2].curvature_bound
    ratio = estimates["gyro_rate"].value / analytic
    assert ratio == pytest.approx(0.010393668129590493, rel=1e-3, abs=0.0)
    assert ratio < 0.02
    assert estimator_transfer(7, 0.35) == pytest.approx(
        0.003263829574463669, rel=1e-9, abs=0.0
    )
    assert estimator_bandwidth_fraction(7) == pytest.approx(
        0.0640398816630755, rel=1e-6, abs=0.0
    )


def test_long_outage_is_refused_for_every_channel() -> None:
    generated = dataset.long_outage_record(samples=300, outage_frames=21_600)
    record = generated.record
    report = classify.classify(record)
    plan = policy.plan(
        record,
        report,
        {
            "battery_voltage": 0.05,
            "valve_current": 0.01,
            "gyro_rate": 0.02,
            "tank_pressure": 0.02,
        },
    )
    assert len(report.findings.gaps) == 1
    assert report.missing_frame_total == 21_600
    assert len(plan.decisions) == 4
    assert len(plan.refusals) == 4
    assert plan.permissions == ()
    built = reconstruct(record, report.findings.gaps, plan)
    assert int(built.synthesised.sum()) == 0
    assert int(built.refused.sum()) == 4 * 21_600


def test_declared_curvature_respects_the_coverage_statement() -> None:
    # On the two channels whose curvature is declared, the realised error must
    # stay inside what equation (3') predicted. Measured, not assumed.
    generated = dataset.generate(samples=2000, seed=909, n_clock_resets=0)
    record = generated.record
    report = classify.classify(record)
    plan = policy.plan(
        record,
        report,
        {
            "battery_voltage": 0.05,
            "valve_current": 0.05,
            "gyro_rate": 0.02,
            "tank_pressure": 0.02,
        },
    )
    built = reconstruct(record, report.findings.gaps, plan)
    checked = 0
    for column, channel in enumerate(record.schema.channels):
        if channel.max_curvature <= 0.0:
            continue
        error, count = measured_error(
            built, generated.truth_times, generated.truth_values[:, column], column
        )
        permitted = [
            decision.predicted_error
            for decision in plan.decisions
            if decision.channel == column and not decision.refused
        ]
        if count == 0 or not permitted:
            continue
        checked += 1
        assert error <= max(permitted) * (1.0 + 1e-9), channel.name
    assert checked == 2


def test_resolvability_gate_changes_the_gyro_decision() -> None:
    # With the gate disabled the policy permits an interpolation on the
    # near-Nyquist channel; with it enabled it refuses. Both measured.
    generated = dataset.generate(samples=2000, seed=909, n_clock_resets=0)
    record = generated.record
    report = classify.classify(record)
    tolerances = {
        "battery_voltage": 0.05,
        "valve_current": 0.05,
        "gyro_rate": 0.02,
        "tank_pressure": 0.02,
    }
    with_gate = policy.plan(record, report, tolerances)
    without_gate = policy.plan(record, report, tolerances, resolvability_threshold=None)
    gyro_with = [d for d in with_gate.for_channel("gyro_rate") if not d.refused]
    gyro_without = [d for d in without_gate.for_channel("gyro_rate") if not d.refused]
    assert gyro_with == []
    assert len(gyro_without) > 0


def test_regression_nearest_indices_tie_goes_low() -> None:
    # Pinned from a Hypothesis run: a query exactly halfway between two times.
    # The convention is arbitrary, so it is pinned rather than rediscovered.
    assert list(nearest_indices(np.array([0.0, 1.0]), np.array([0.5]))) == [0]


def test_regression_cadence_residual_sign_at_a_half_slot() -> None:
    # np.rint rounds half to even, so an interval of exactly 1.5 cadences
    # rounds to 2 slots and the residual is negative. Pinned because a naive
    # implementation would round to 1 and report +0.5.
    channel = ChannelSpec(
        name="probe",
        unit="K",
        scale=0.5,
        offset=10.0,
        low=12.0,
        high=60.0,
        max_slew=1.0,
        max_curvature=0.25,
    )
    schema = RecordSchema(
        cadence_s=1.0,
        jitter_s=0.1,
        counter_modulus=256,
        channels=(channel,),
        has_validity_flag=True,
        has_raw_words=True,
    )
    record = TelemetryRecord(
        schema=schema,
        timestamps=np.array([0.0, 1.5]),
        counters=np.array([0, 1], dtype=np.int64),
        values=np.full((2, 1), 30.0),
        raw_words=np.full((2, 1), 40, dtype=np.uint64),
        validity=np.ones(2, dtype=bool),
        provenance={"time_source": "test"},
    )
    finding = screens.screen_cadence(record)[0]
    assert finding.slots == 2
    assert finding.residual_s == pytest.approx(-0.5, rel=1e-12, abs=1e-15)


def test_regression_local_statistics_on_an_all_non_finite_window() -> None:
    # Pinned: a window containing only non-finite neighbours fell through to a
    # nanmedian over an empty slice during the build, which warns and returns
    # nan. The fallback is the sample itself, or zero when that is not finite.
    values = np.array([np.nan, np.nan, np.nan])
    medians, scaled = robust_local_statistics(values, 2, 1.0)
    # No finite neighbour and no finite sample: the fallback is zero, not nan.
    assert np.all(medians == pytest.approx(0.0, rel=0.0, abs=0.0))
    assert np.all(scaled > 0.0)
    # One finite sample in the middle: its own value is the fallback.
    mixed = np.array([np.nan, 5.0, np.nan])
    medians, _ = robust_local_statistics(mixed, 2, 1.0)
    assert np.all(medians == pytest.approx(5.0, rel=0.0, abs=0.0))


def test_regression_measured_error_alignment_off_by_one() -> None:
    # Pinned: a left-insertion lookup against the truth grid reported a
    # one-cadence signal difference as a reconstruction error, inflating the
    # measured error on battery_voltage from 0.0049 to 0.84. The record's
    # jitter is what makes the grid anchor fall between two truth samples.
    truth_times = np.array([0.0, 1.0, 2.0, 3.0])
    query = np.array([0.07, 1.07, 2.07])
    assert list(nearest_indices(truth_times, query)) == [0, 1, 2]


def test_regression_policy_refuses_when_tolerance_is_below_channel_noise() -> None:
    # Pinned: valve_current declares a noise sigma of 0.012 A, so a declared
    # tolerance of 0.01 A cannot be met by any reconstruction, nor by a
    # received sample. The refusal names that gate.
    generated = dataset.generate(samples=800, seed=77, n_clock_resets=0)
    record = generated.record
    report = classify.classify(record)
    plan = policy.plan(
        record,
        report,
        {
            "battery_voltage": 0.05,
            "valve_current": 0.01,
            "gyro_rate": 0.02,
            "tank_pressure": 0.02,
        },
    )
    valve = plan.for_channel("valve_current")
    assert valve
    assert all(decision.refused for decision in valve)
    assert all("tolerance_above_noise" in decision.failed_gates for decision in valve)
