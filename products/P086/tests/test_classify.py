"""The three-way classification: every row of the decision table, on constructed records."""

from __future__ import annotations

import numpy as np
import pytest

from telemgap import classify as classify_module
from telemgap.classify import MODES, OBSERVATIONS, AnomalyClass, classify
from telemgap.record import RecordSchema, TelemetryRecord


def verdicts(record: TelemetryRecord, mode: str = "strengthened") -> dict[str, list[str]]:
    """Observation name to the verdicts assigned to it, for table-row assertions."""
    report = classify(record, mode=mode)
    out: dict[str, list[str]] = {}
    for anomaly in report.anomalies:
        out.setdefault(anomaly.observation, []).append(anomaly.verdict.value)
    return out


def test_classes_and_modes() -> None:
    assert [member.value for member in AnomalyClass] == [
        "missing",
        "corrupt",
        "misparsed",
        "undecided",
    ]
    assert MODES == ("specified", "strengthened")


def test_gap_is_missing_with_confidence_one(small_schema: RecordSchema, record_builder) -> None:
    record = record_builder(
        small_schema, [40] * 4, timestamps=[0.0, 1.0, 6.0, 7.0], counters=[0, 1, 6, 7]
    )
    report = classify(record)
    gaps = report.by_observation("gap")
    assert len(gaps) == 1
    assert gaps[0].verdict is AnomalyClass.MISSING
    assert gaps[0].confidence == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert report.missing_frame_total == 4


def test_out_of_range_with_failed_validity_is_corrupt(
    small_schema: RecordSchema, record_builder
) -> None:
    raws = [40] * 21
    raws[10] = 50000
    validity = [True] * 21
    validity[10] = False
    record = record_builder(small_schema, raws, validity=validity)
    report = classify(record)
    anomaly = next(
        item for item in report.anomalies if item.observation == "value_out_of_range"
    )
    assert anomaly.verdict is AnomalyClass.CORRUPT
    assert anomaly.confidence == pytest.approx(0.90, rel=0.0, abs=0.0)
    assert any("error-control verdict failed" in text for text in anomaly.evidence)


def test_out_of_range_with_rescue_is_misparsed(
    small_schema: RecordSchema, record_builder
) -> None:
    # 0x2800 reads 5130.0 and byteswaps to 0x0028 = 40 -> 30.0, which is the
    # neighbours' value. Validity passes, so the verdict is misparsed at 0.80.
    raws = [40] * 21
    raws[10] = 0x2800
    record = record_builder(small_schema, raws)
    report = classify(record)
    anomaly = next(
        item for item in report.anomalies if item.observation == "value_out_of_range"
    )
    assert anomaly.verdict is AnomalyClass.MISPARSED
    assert anomaly.confidence == pytest.approx(0.80, rel=0.0, abs=0.0)
    assert any("byteswap" in text for text in anomaly.evidence)


def test_out_of_range_without_validity_lowers_the_tier(
    small_schema: RecordSchema, record_builder
) -> None:
    raws = [40] * 21
    raws[10] = 0x2800
    record = record_builder(small_schema, raws).without_validity()
    report = classify(record)
    anomaly = next(
        item for item in report.anomalies if item.observation == "value_out_of_range"
    )
    assert anomaly.verdict is AnomalyClass.MISPARSED
    assert anomaly.confidence == pytest.approx(0.60, rel=0.0, abs=0.0)


def test_out_of_range_without_rescue_is_undecided(
    small_schema: RecordSchema, record_builder
) -> None:
    # 0xFFFF reads 32777.5; no declared alternative lands in [12, 60] near 30.
    raws = [40] * 21
    raws[10] = 0xFFFF
    record = record_builder(small_schema, raws)
    report = classify(record)
    anomaly = next(
        item for item in report.anomalies if item.observation == "value_out_of_range"
    )
    assert anomaly.verdict is AnomalyClass.UNDECIDED
    assert anomaly.confidence == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert anomaly.reasons_undecided
    assert anomaly.is_undecided


def test_undecided_reason_names_the_missing_field(
    small_schema: RecordSchema, record_builder
) -> None:
    raws = [40] * 21
    raws[10] = 0xFFFF
    record = record_builder(small_schema, raws).without_raw_words()
    report = classify(record)
    anomaly = next(
        item for item in report.anomalies if item.observation == "value_out_of_range"
    )
    assert anomaly.verdict is AnomalyClass.UNDECIDED
    assert any("raw words" in reason for reason in anomaly.reasons_undecided)


def test_local_outlier_only_in_strengthened_mode(
    small_schema: RecordSchema, record_builder
) -> None:
    raws = [40] * 21
    raws[10] = 60  # 40.0 K: inside [12, 60] but far from its neighbours.
    record = record_builder(small_schema, raws)
    assert "value_local_outlier" not in verdicts(record, mode="specified")
    assert "value_local_outlier" in verdicts(record, mode="strengthened")


def test_local_outlier_with_failed_validity_is_corrupt(
    small_schema: RecordSchema, record_builder
) -> None:
    raws = [40] * 21
    raws[10] = 60
    validity = [True] * 21
    validity[10] = False
    record = record_builder(small_schema, raws, validity=validity)
    report = classify(record)
    anomaly = next(
        item for item in report.anomalies if item.observation == "value_local_outlier"
    )
    assert anomaly.verdict is AnomalyClass.CORRUPT
    assert anomaly.confidence == pytest.approx(0.80, rel=0.0, abs=0.0)


def test_identical_duplicate_is_misparsed(small_schema: RecordSchema, record_builder) -> None:
    record = record_builder(
        small_schema, [40] * 4, timestamps=[0.0, 1.0, 1.0, 2.0], counters=[0, 1, 1, 2]
    )
    report = classify(record)
    anomaly = next(
        item for item in report.anomalies if item.observation == "repeated_counter"
    )
    assert anomaly.verdict is AnomalyClass.MISPARSED
    assert anomaly.confidence == pytest.approx(0.90, rel=0.0, abs=0.0)
    assert anomaly.detail["payload_identical"] == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_differing_duplicate_with_failed_validity_is_corrupt(
    small_schema: RecordSchema, record_builder
) -> None:
    record = record_builder(
        small_schema,
        [40, 40, 44, 40],
        timestamps=[0.0, 1.0, 1.0, 2.0],
        counters=[0, 1, 1, 2],
        validity=[True, True, False, True],
    )
    report = classify(record)
    anomaly = next(
        item for item in report.anomalies if item.observation == "repeated_counter"
    )
    assert anomaly.verdict is AnomalyClass.CORRUPT


def test_differing_duplicate_with_passing_validity_is_undecided(
    small_schema: RecordSchema, record_builder
) -> None:
    record = record_builder(
        small_schema,
        [40, 40, 44, 40],
        timestamps=[0.0, 1.0, 1.0, 2.0],
        counters=[0, 1, 1, 2],
    )
    report = classify(record)
    anomaly = next(
        item for item in report.anomalies if item.observation == "repeated_counter"
    )
    assert anomaly.verdict is AnomalyClass.UNDECIDED
    assert any("same counter" in reason for reason in anomaly.reasons_undecided)


def test_reorder_is_misparsed(small_schema: RecordSchema, record_builder) -> None:
    record = record_builder(
        small_schema, [40] * 4, timestamps=[0.0, 1.0, 1.9, 3.0], counters=[0, 2, 1, 3]
    )
    report = classify(record)
    anomaly = next(item for item in report.anomalies if item.observation == "reorder")
    assert anomaly.verdict is AnomalyClass.MISPARSED
    assert anomaly.confidence == pytest.approx(0.70, rel=0.0, abs=0.0)
    assert anomaly.detail["frames_backward"] == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_clock_reset_is_misparsed(small_schema: RecordSchema, record_builder) -> None:
    record = record_builder(
        small_schema, [40] * 4, timestamps=[0.0, 1.0, -4.0, -3.0], counters=[0, 1, 2, 3]
    )
    report = classify(record)
    anomaly = next(item for item in report.anomalies if item.observation == "clock_reset")
    assert anomaly.verdict is AnomalyClass.MISPARSED
    assert anomaly.detail["delta_s"] == pytest.approx(-5.0, rel=0.0, abs=0.0)


def test_clean_record_produces_no_anomalies(nominal_record: TelemetryRecord) -> None:
    # The negative direction: a control that must NOT fire.
    report = classify(nominal_record)
    assert report.anomalies == ()
    assert report.counts == {"missing": 0, "corrupt": 0, "misparsed": 0, "undecided": 0}
    assert report.undecided == ()


def test_determinability_is_the_first_field() -> None:
    # Binding rule 17: the verdict is a computed output, and it comes first.
    fields = [field.name for field in classify_module.IntegrityReport.__dataclass_fields__.values()]
    assert fields[0] == "determinability"


def test_report_rejects_unknown_observation(nominal_record: TelemetryRecord) -> None:
    report = classify(nominal_record)
    with pytest.raises(ValueError, match="unknown observation"):
        report.by_observation("nonsense")
    for name in OBSERVATIONS:
        assert report.by_observation(name) == ()


def test_classify_rejects_bad_arguments(nominal_record: TelemetryRecord) -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        classify(nominal_record, mode="nonsense")
    with pytest.raises(TypeError, match="TelemetryRecord"):
        classify(np.zeros(3))


def test_anomalies_are_sorted_by_position(generated_small) -> None:
    report = classify(generated_small.record)
    keys = [
        (anomaly.index, anomaly.channel, anomaly.observation) for anomaly in report.anomalies
    ]
    assert keys == sorted(keys)


def test_withholding_raw_words_empties_the_misparsed_class(generated_small) -> None:
    with_raw = classify(generated_small.record)
    without = classify(generated_small.record.without_raw_words())
    value_level = [
        anomaly
        for anomaly in without.anomalies
        if anomaly.observation in ("value_out_of_range", "value_local_outlier")
    ]
    assert all(anomaly.verdict is not AnomalyClass.MISPARSED for anomaly in value_level)
    assert len(without.undecided) > len(with_raw.undecided)


def test_strengthened_mode_finds_more_than_specified(generated_small) -> None:
    specified = classify(generated_small.record, mode="specified")
    strengthened = classify(generated_small.record, mode="strengthened")
    assert len(strengthened.anomalies) > len(specified.anomalies)
    assert specified.mode == "specified"
    assert strengthened.mode == "strengthened"
