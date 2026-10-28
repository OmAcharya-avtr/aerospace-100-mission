"""The rule screens, including the vectorised-versus-loop agreement check."""

from __future__ import annotations

import numpy as np
import pytest

from telemgap import screens
from telemgap.record import RecordSchema, TelemetryRecord
from telemgap.screens import (
    MAD_TO_SIGMA,
    _robust_local_statistics_loop,
    consolidate_gaps,
    counter_delta,
    robust_local_statistics,
    screen,
    screen_cadence,
    screen_clock,
    screen_counters,
    screen_local_residual,
    screen_ranges,
    screen_validity,
    timeline_order,
)


@pytest.mark.parametrize(
    ("previous", "current", "modulus", "expected"),
    [
        # Hand-calculated from equation (2), d = (c - c_prev) mod M.
        (0, 1, 256, 1),
        (255, 0, 256, 1),
        (255, 2, 256, 3),
        (5, 5, 256, 0),
        (5, 4, 256, 255),
        (0, 128, 256, 128),
        (0, 129, 256, 129),
        (16383, 0, 16384, 1),
    ],
)
def test_counter_delta_known_answers(
    previous: int, current: int, modulus: int, expected: int
) -> None:
    assert counter_delta(previous, current, modulus) == expected


def test_counter_delta_rejects_small_modulus() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        counter_delta(0, 1, 1)


def test_counter_screen_kinds(small_schema: RecordSchema, record_builder) -> None:
    # counters 0,1,1,2,6,5 -> nominal, repeated, nominal, forward_gap(3), reorder(1)
    record = record_builder(
        small_schema, [40] * 6, timestamps=list(range(6)), counters=[0, 1, 1, 2, 6, 5]
    )
    findings = screen_counters(record)
    assert [finding.kind for finding in findings] == [
        "nominal",
        "repeated_counter",
        "nominal",
        "forward_gap",
        "reorder",
    ]
    assert findings[3].frames_implied == 3
    assert findings[4].frames_implied == 1


def test_counter_screen_reorder_boundary(small_schema: RecordSchema, record_builder) -> None:
    # modulus 256, horizon 128 -> delta 128 is a forward gap, 129 is a reorder.
    forward = record_builder(small_schema, [40, 40], counters=[0, 128])
    backward = record_builder(small_schema, [40, 40], counters=[0, 129])
    assert screen_counters(forward)[0].kind == "forward_gap"
    assert screen_counters(forward)[0].frames_implied == 127
    assert screen_counters(backward)[0].kind == "reorder"
    assert screen_counters(backward)[0].frames_implied == 127


def test_clock_screen_separates_reset_from_reorder(
    small_schema: RecordSchema, record_builder
) -> None:
    # Timestamp goes back 5 s while the counter moves forward -> clock reset.
    reset = record_builder(
        small_schema, [40] * 4, timestamps=[0.0, 1.0, -4.0, -3.0], counters=[0, 1, 2, 3]
    )
    findings = screen_clock(reset, screen_counters(reset))
    assert [finding.kind for finding in findings] == ["clock_reset"]
    assert findings[0].delta_s == pytest.approx(-5.0, rel=0.0, abs=0.0)
    # Timestamp and counter both go back -> consistent reorder, not a reset.
    reorder = record_builder(
        small_schema, [40] * 4, timestamps=[0.0, 1.0, 0.4, 2.0], counters=[0, 2, 1, 3]
    )
    findings = screen_clock(reorder, screen_counters(reorder))
    assert [finding.kind for finding in findings] == ["reorder_consistent"]


def test_clock_screen_ignores_jitter(small_schema: RecordSchema, record_builder) -> None:
    # jitter_s is 0.1; a 0.05 s backward step must not be reported.
    record = record_builder(small_schema, [40] * 3, timestamps=[0.0, 1.0, 0.95])
    assert screen_clock(record, screen_counters(record)) == ()


def test_cadence_screen_kinds(small_schema: RecordSchema, record_builder) -> None:
    # Intervals 1.0 (nominal), 3.0 (time_gap, 2 missing), 0.0 (same_slot),
    # 1.5 (jitter_exceeded). np.rint rounds half to even, so 1.5 / 1.0 rounds
    # to 2 slots and the residual is -0.5, not +0.5. The sign is recorded here
    # because it is a property of the rounding rule and not of the data.
    record = record_builder(
        small_schema, [40] * 5, timestamps=[0.0, 1.0, 4.0, 4.0, 5.5]
    )
    findings = screen_cadence(record)
    assert [finding.kind for finding in findings] == [
        "nominal",
        "time_gap",
        "same_slot",
        "jitter_exceeded",
    ]
    assert findings[1].slots == 3
    assert findings[3].slots == 2
    assert findings[3].residual_s == pytest.approx(-0.5, rel=1e-12, abs=1e-15)


def test_range_screen_known_answers(small_schema: RecordSchema, record_builder) -> None:
    # scale 0.5 offset 10.0, range [12, 60]. raw 0 -> 10.0 (below),
    # raw 40 -> 30.0 (inside), raw 200 -> 110.0 (above by 50).
    record = record_builder(small_schema, [0, 40, 200])
    findings = screen_ranges(record)
    assert [finding.index for finding in findings] == [0, 2]
    assert findings[0].excess == pytest.approx(2.0, rel=1e-12, abs=1e-15)
    assert findings[1].excess == pytest.approx(50.0, rel=1e-12, abs=1e-15)
    assert not findings[0].non_finite


def test_range_screen_flags_non_finite(small_schema: RecordSchema) -> None:
    record = TelemetryRecord(
        schema=small_schema,
        timestamps=np.arange(3, dtype=float),
        counters=np.arange(3, dtype=np.int64),
        values=np.array([[30.0], [float("nan")], [30.0]]),
        raw_words=np.array([[40], [40], [40]], dtype=np.uint64),
        validity=np.ones(3, dtype=bool),
    )
    findings = screen_ranges(record)
    assert len(findings) == 1
    assert findings[0].non_finite
    assert np.isinf(findings[0].excess)


def test_mad_constant_value() -> None:
    # Rousseeuw & Croux 1993 consistency constant.
    assert MAD_TO_SIGMA == pytest.approx(1.4826, rel=0.0, abs=0.0)


@pytest.mark.parametrize("trial", range(12))
def test_vectorised_local_statistics_matches_loop(trial: int) -> None:
    rng = np.random.default_rng(1000 + trial)
    n = int(rng.integers(1, 80))
    values = rng.normal(size=n)
    if trial % 3 == 0 and n > 4:
        values[rng.integers(0, n, size=2)] = np.nan
    if trial % 4 == 0 and n > 4:
        values[int(rng.integers(0, n))] = np.inf
    window = int(rng.integers(2, 9))
    fast = robust_local_statistics(values, window, 1.0)
    slow = _robust_local_statistics_loop(values, window, 1.0)
    assert np.allclose(fast[0], slow[0], rtol=0.0, atol=1e-12)
    assert np.allclose(fast[1], slow[1], rtol=0.0, atol=1e-12)


def test_local_statistics_excludes_self() -> None:
    # A single spike must not pull the median it is compared against.
    values = np.zeros(21)
    values[10] = 100.0
    medians, scaled = robust_local_statistics(values, 5, 1.0)
    assert medians[10] == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert scaled[10] > 0.0


def test_local_statistics_rejects_tiny_window() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        robust_local_statistics(np.zeros(10), 1, 1.0)


def test_residual_screen_finds_in_range_outlier(
    small_schema: RecordSchema, record_builder
) -> None:
    # All raw 40 (30.0 K) except one raw 60 (40.0 K): inside [12, 60] and so
    # invisible to the range screen, but far outside its neighbourhood.
    raws = [40] * 21
    raws[10] = 60
    record = record_builder(small_schema, raws)
    assert screen_ranges(record) == ()
    findings = screen_local_residual(record, window=5, z_threshold=8.0)
    assert [finding.index for finding in findings] == [10]
    assert findings[0].value == pytest.approx(40.0, rel=1e-12, abs=1e-15)


def test_residual_screen_rejects_bad_threshold(nominal_record: TelemetryRecord) -> None:
    with pytest.raises(ValueError, match="z_threshold"):
        screen_local_residual(nominal_record, z_threshold=0.0)


def test_timeline_order_sorts_and_deduplicates(
    small_schema: RecordSchema, record_builder
) -> None:
    record = record_builder(
        small_schema, [40] * 4, timestamps=[0.0, 2.0, 1.0, 2.0], counters=[0, 2, 1, 2]
    )
    assert timeline_order(record) == (0, 2, 1)


def test_gap_consolidation_known_answer(small_schema: RecordSchema, record_builder) -> None:
    # Frames at t = 0, 1, 6, 7 -> one gap of 5 s, 4 missing frames.
    record = record_builder(
        small_schema, [40] * 4, timestamps=[0.0, 1.0, 6.0, 7.0], counters=[0, 1, 6, 7]
    )
    gaps = consolidate_gaps(record, screen_counters(record))
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.missing_frames_from_time == 4
    assert gap.missing_frames_from_counter == 4
    assert gap.agrees
    assert gap.duration_s == pytest.approx(5.0, rel=0.0, abs=0.0)
    assert gap.unobserved_s == pytest.approx(4.0, rel=1e-12, abs=1e-15)


def test_gap_deficits_can_disagree(small_schema: RecordSchema, record_builder) -> None:
    # Five seconds of timeline but only one frame of counter: the clock and the
    # counter disagree, which is evidence about the clock.
    record = record_builder(
        small_schema, [40] * 3, timestamps=[0.0, 1.0, 6.0], counters=[0, 1, 2]
    )
    gaps = consolidate_gaps(record, screen_counters(record))
    assert len(gaps) == 1
    assert gaps[0].missing_frames_from_time == 4
    assert gaps[0].missing_frames_from_counter == 0
    assert not gaps[0].agrees


def test_screen_rejects_non_record() -> None:
    with pytest.raises(TypeError, match="TelemetryRecord"):
        screen([1, 2, 3])


def test_screen_summary_keys(nominal_record: TelemetryRecord) -> None:
    findings = screen(nominal_record)
    summary = findings.summary()
    assert set(summary) == {
        "frames",
        "gaps",
        "missing_frames",
        "duplicates",
        "reorders",
        "clock_resets",
        "cadence_violations",
        "range_violations",
        "residual_violations",
        "validity_failures",
    }
    assert summary["frames"] == 40
    assert summary["gaps"] == 0
    assert summary["range_violations"] == 0
    assert summary["validity_failures"] == 0


def test_screen_notes_disagreement(small_schema: RecordSchema, record_builder) -> None:
    record = record_builder(
        small_schema, [40] * 3, timestamps=[0.0, 1.0, 6.0], counters=[0, 1, 2]
    )
    findings = screen(record)
    assert any("disagrees with the counter-domain deficit" in note for note in findings.notes)


def test_screen_finds_every_injected_class(generated_small) -> None:
    findings = screen(generated_small.record)
    summary = findings.summary()
    assert summary["gaps"] >= 1
    assert summary["duplicates"] >= 1
    assert summary["reorders"] >= 1
    assert summary["range_violations"] >= 1
    assert summary["residual_violations"] >= 1


def test_violation_index_properties(generated_small) -> None:
    findings = screen(generated_small.record)
    assert findings.range_violation_indices == tuple(sorted(set(findings.range_violation_indices)))
    assert findings.residual_violation_indices == tuple(
        sorted(set(findings.residual_violation_indices))
    )


def test_module_exports_mad_floor_fraction() -> None:
    assert 0.0 < screens.MAD_FLOOR_FRACTION < 1e-6


def test_validity_screen_lists_failed_frames(small_schema: RecordSchema, record_builder) -> None:
    validity = [True] * 5
    validity[2] = False
    record = record_builder(small_schema, [40] * 5, validity=validity)
    findings = screen_validity(record)
    assert [finding.index for finding in findings] == [2]
    assert findings[0].counter == 2
    assert screen(record).validity_failed_indices == (2,)


def test_validity_screen_is_empty_without_the_field(nominal_record: TelemetryRecord) -> None:
    assert screen_validity(nominal_record.without_validity()) == ()
