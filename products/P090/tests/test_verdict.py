"""The decision table, the integrity/consequence separation and the ordering."""

from __future__ import annotations

import math

import pytest

from tlecheck.consequence import BENIGN, SILENTLY_WRONG, WILL_FAIL_LOUDLY
from tlecheck.elements import apply_checksum, format_tle, parse_tle
from tlecheck.epoch import epoch_julian_date
from tlecheck.screens import run
from tlecheck.verdict import UNREADABLE_FORMAT_NAMES, classify, screen

from conftest import REFERENCE_FIELDS, record_of


def _set(**overrides):
    fields = dict(REFERENCE_FIELDS)
    fields.update(overrides)
    return parse_tle(*format_tle(**fields))


def test_determinability_is_the_first_field(reference_set):
    verdict = screen(reference_set)
    fields = list(verdict.__dataclass_fields__)
    assert fields[0] == "determinability"


def test_determinability_is_populated_before_any_screen(reference_set):
    verdict = screen(reference_set)
    assert len(verdict.determinability) == 11
    assert verdict.determinability[0].name == "propagated_ephemeris_accuracy"


def test_clean_element_set_is_benign(clean_record):
    verdict = screen(
        clean_record.candidate,
        history=list(clean_record.history),
        reference_jd=clean_record.use_jd,
    )
    assert verdict.predicted_class == BENIGN
    assert verdict.integrity_ok is True
    assert verdict.problems == ()


def test_physical_bound_violation_predicts_a_loud_failure():
    candidate = parse_tle(
        _set().line1,
        apply_checksum(_set().line2[:26] + "9999999" + _set().line2[33:68]),
    )
    verdict = screen(candidate)
    assert verdict.predicted_class == WILL_FAIL_LOUDLY
    assert verdict.silent_risk_score == 0.0


def test_checksum_mismatch_breaks_integrity_without_predicting_consequence(clean_record):
    broken_line1 = (
        clean_record.candidate.line1[:68]
        + str((int(clean_record.candidate.line1[68]) + 4) % 10)
    )
    candidate = parse_tle(broken_line1, clean_record.candidate.line2)
    verdict = screen(
        candidate,
        history=list(clean_record.history),
        reference_jd=clean_record.use_jd,
    )
    assert verdict.integrity_ok is False
    assert verdict.predicted_class == BENIGN
    names = {problem.name for problem in verdict.problems}
    assert "checksum_mismatch" in names
    problem = next(p for p in verdict.problems if p.name == "checksum_mismatch")
    assert problem.predicted_class == BENIGN
    assert "Which character changed is not determinable" in problem.detail


def test_unreadable_field_sets_the_score_to_infinity(clean_record):
    broken = parse_tle(
        clean_record.candidate.line1,
        clean_record.candidate.line2[:26] + "ABCDEFG" + clean_record.candidate.line2[33:],
    )
    verdict = screen(
        broken, history=list(clean_record.history), reference_jd=clean_record.use_jd
    )
    assert verdict.integrity_ok is False
    assert math.isinf(verdict.silent_risk_score)
    assert verdict.predicted_class == SILENTLY_WRONG


def test_unreadable_names_are_the_two_documented_ones():
    assert UNREADABLE_FORMAT_NAMES == frozenset(
        {"parse_fault", "checksum_column_not_a_digit"}
    )


def test_unreadable_does_not_predict_a_loud_failure(clean_record):
    """The measurement that settled this is in the module docstring."""
    broken = parse_tle(
        clean_record.candidate.line1,
        clean_record.candidate.line2[:26] + "ABCDEFG" + clean_record.candidate.line2[33:],
    )
    verdict = screen(broken, history=list(clean_record.history))
    assert verdict.predicted_class != WILL_FAIL_LOUDLY


def test_stale_epoch_predicts_silently_wrong(clean_record):
    verdict = screen(
        clean_record.candidate,
        history=list(clean_record.history),
        reference_jd=epoch_julian_date(clean_record.candidate) + 30.0,
        epoch_requirement_days=7.0,
    )
    names = {problem.name for problem in verdict.problems}
    assert "epoch_stale" in names
    assert verdict.predicted_class == SILENTLY_WRONG
    assert verdict.silent_risk_score > 1.0


def test_future_epoch_is_a_distinct_problem(clean_record):
    verdict = screen(
        clean_record.candidate,
        reference_jd=epoch_julian_date(clean_record.candidate) - 5.0,
    )
    names = {problem.name for problem in verdict.problems}
    assert "epoch_in_future" in names


def test_cross_source_divergence_predicts_silently_wrong(small_records):
    record = record_of(small_records, "cross_source_divergence")
    verdict = screen(
        record.candidate,
        history=list(record.history),
        other_sources=list(record.other_sources),
        reference_jd=record.use_jd,
    )
    names = {problem.name for problem in verdict.problems}
    assert "cross_source_divergence" in names
    problem = next(p for p in verdict.problems if p.name == "cross_source_divergence")
    assert "not determinable" in problem.detail


def test_thresholds_are_carried_with_the_verdict(reference_set):
    verdict = screen(
        reference_set,
        epoch_requirement_days=3.0,
        history_threshold=5.0,
        divergence_threshold_km=2.0,
        continuity_threshold=6.0,
        silent_threshold_km=0.5,
    )
    assert verdict.thresholds["epoch_requirement_days"] == 3.0
    assert verdict.thresholds["history_threshold"] == 5.0
    assert verdict.thresholds["divergence_threshold_km"] == 2.0
    assert verdict.thresholds["continuity_threshold"] == 6.0
    assert verdict.thresholds["silent_threshold_km"] == 0.5


def test_score_is_zero_when_a_physical_bound_fails(clean_record):
    candidate = parse_tle(
        clean_record.candidate.line1,
        apply_checksum(
            clean_record.candidate.line2[:26]
            + "9999999"
            + clean_record.candidate.line2[33:68]
        ),
    )
    evidence = run(candidate, history=list(clean_record.history))
    predicted, score, _problems, _intact = classify(evidence)
    assert predicted == WILL_FAIL_LOUDLY
    assert score == 0.0


def test_classify_returns_four_values(reference_set):
    result = classify(run(reference_set))
    assert len(result) == 4
    assert isinstance(result[3], bool)


def test_threshold_of_one_is_the_rules_own_operating_point(clean_record):
    verdict = screen(
        clean_record.candidate,
        history=list(clean_record.history),
        reference_jd=clean_record.use_jd,
    )
    assert (verdict.predicted_class == SILENTLY_WRONG) == (
        verdict.silent_risk_score > 1.0
    )


def test_integrity_is_intact_for_a_well_formed_set(reference_set):
    assert screen(reference_set).integrity_ok is True


def test_a_benign_verdict_does_not_claim_accuracy(reference_set):
    from tlecheck.report import verdict_text

    text = " ".join(verdict_text(screen(reference_set)).split())
    assert "not a statement that an ephemeris propagated from this element set" in text


@pytest.mark.parametrize("corruption", ["sign_flip", "field_shift", "data_digit"])
def test_corrupted_records_are_screened_without_raising(small_records, corruption):
    record = record_of(small_records, corruption)
    verdict = screen(
        record.candidate,
        history=list(record.history),
        other_sources=list(record.other_sources),
        reference_jd=record.use_jd,
    )
    assert verdict.predicted_class in {BENIGN, SILENTLY_WRONG, WILL_FAIL_LOUDLY}
