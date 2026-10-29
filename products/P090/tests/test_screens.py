"""The six screens, and the difference between no history and an empty history."""

from __future__ import annotations

import pytest

from tlecheck.bounds import BoundsPolicy
from tlecheck.elements import format_tle, parse_tle
from tlecheck.screens import ScreenEvidence, format_screen, run

from conftest import REFERENCE_FIELDS, REFERENCE_LINE1, REFERENCE_LINE2, record_of


def _set(**overrides):
    fields = dict(REFERENCE_FIELDS)
    fields.update(overrides)
    return parse_tle(*format_tle(**fields))


def test_screen_names_are_the_six(clean_record):
    evidence = run(clean_record.candidate, history=list(clean_record.history))
    assert evidence.names == (
        "format",
        "bounds",
        "epoch",
        "history",
        "continuity",
        "cross-source",
    )


def test_format_screen_clean_on_the_reference(reference_set):
    assert format_screen(reference_set) == ()


def test_format_screen_finds_a_checksum_mismatch():
    broken = parse_tle(REFERENCE_LINE1[:68] + "7", REFERENCE_LINE2)
    findings = format_screen(broken)
    names = {finding.name for finding in findings}
    assert "checksum_mismatch" in names
    finding = next(f for f in findings if f.name == "checksum_mismatch")
    assert finding.line == 1
    assert "states checksum 7" in finding.detail


def test_format_screen_finds_a_non_digit_checksum_column():
    broken = parse_tle(REFERENCE_LINE1[:68] + "X", REFERENCE_LINE2)
    names = {finding.name for finding in format_screen(broken)}
    assert "checksum_column_not_a_digit" in names


def test_format_screen_reports_parse_faults_on_the_right_line():
    broken = parse_tle(
        REFERENCE_LINE1, REFERENCE_LINE2[:26] + "ABCDEFG" + REFERENCE_LINE2[33:]
    )
    findings = [f for f in format_screen(broken) if f.name == "parse_fault"]
    assert findings
    assert all(finding.line == 2 for finding in findings)


def test_no_history_argument_leaves_the_history_screen_none(reference_set):
    evidence = run(reference_set)
    assert evidence.history is None
    assert evidence.continuity is None


def test_empty_history_is_not_the_same_as_no_history(reference_set):
    evidence = run(reference_set, history=[])
    assert evidence.history is not None
    assert evidence.history.usable is False
    assert evidence.continuity is not None
    assert evidence.continuity.usable is False


def test_no_reference_time_leaves_the_epoch_screen_none(reference_set):
    evidence = run(reference_set)
    assert evidence.epoch is None


def test_reference_time_runs_the_epoch_screen(clean_record):
    evidence = run(clean_record.candidate, reference_jd=clean_record.use_jd)
    assert evidence.epoch is not None
    assert evidence.epoch.requirement_days == 7.0


def test_other_sources_run_the_divergence_screen(small_records):
    record = record_of(small_records, "cross_source_divergence")
    evidence = run(
        record.candidate,
        other_sources=list(record.other_sources),
        reference_jd=record.use_jd,
    )
    assert len(evidence.divergences) == 1


def test_derived_geometry_is_always_present(reference_set):
    evidence = run(reference_set)
    assert evidence.derived is not None
    assert evidence.derived.perigee_altitude_km == pytest.approx(411.39, rel=0.0, abs=0.02)


def test_policy_is_honoured():
    evidence = run(_set(), policy=BoundsPolicy(max_eccentricity=1.0e-6))
    names = {finding.name for finding in evidence.bound_findings}
    assert "eccentricity_above_policy" in names


def test_evidence_is_an_evidence_object(reference_set):
    assert isinstance(run(reference_set), ScreenEvidence)


def test_history_of_another_object_is_a_caller_error(clean_record):
    stranger = _set(satnum=11111)
    with pytest.raises(ValueError):
        run(clean_record.candidate, history=[*clean_record.history, stranger])


def test_silent_threshold_reaches_the_continuity_screen(clean_record):
    evidence = run(
        clean_record.candidate,
        history=list(clean_record.history),
        reference_jd=clean_record.use_jd,
        silent_threshold_km=1.0e-9,
    )
    assert evidence.continuity is not None
    assert evidence.continuity.resolves_threshold is False


def test_divergence_without_a_reference_time_uses_the_candidate_epoch(small_records):
    record = record_of(small_records, "cross_source_divergence")
    evidence = run(record.candidate, other_sources=list(record.other_sources))
    assert evidence.divergences
    assert evidence.divergences[0].usable is True
