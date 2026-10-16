"""Rendering. Text, JSON and markdown must carry the same numbers."""

from __future__ import annotations

import json

import pytest

from watchdogspec import (
    SCOPE_NOTE,
    format_codes,
    format_false_trip_table,
    format_findings,
    format_json,
    format_markdown,
    format_report,
    format_timing_table,
    report_to_dict,
    run_checks,
)


def test_text_report_names_every_task_and_the_verdict(worked_report):
    text = format_report(worked_report)
    for task_id in ("TASK-GNC", "TASK-IMU", "TASK-TLM", "TASK-FDIR"):
        assert task_id in text
    assert "INCONSISTENT" in text
    assert "Exit code 1" in text
    assert SCOPE_NOTE in text


def test_text_report_of_a_consistent_spec_says_so(minimal_spec):
    text = format_report(run_checks(minimal_spec))
    assert "CONSISTENT" in text
    assert "INCONSISTENT" not in text
    assert "none" in text


def test_report_has_no_marketing_language_or_exclamation(worked_report):
    text = format_report(worked_report)
    assert "!" not in text
    for banned in ("flight-safe", "certified", "production-ready", "mission-ready"):
        assert banned not in text.lower()


def test_timing_table_rows_match_the_timings(worked_report):
    table = format_timing_table(worked_report)
    assert table.count("TASK-") == 4
    assert "WCDL" in table


def test_false_trip_table_mentions_the_two_models(worked_report):
    table = format_false_trip_table(worked_report)
    assert "sigma = 0" in table
    assert "different models" in table


def test_findings_section_lists_every_code(planted_report):
    text = format_findings(planted_report)
    for index in range(1, 13):
        assert f"WS{index:03d}" in text


def test_json_round_trips_and_carries_the_same_numbers(worked_report):
    payload = json.loads(format_json(worked_report))
    assert payload["exit_code"] == 1
    assert payload["consistent"] is False
    assert len(payload["findings"]) == 2
    gnc = payload["timings"]["TASK-GNC"]
    assert gnc["detection_latency_ms"] == pytest.approx(75.5, abs=1e-9)
    assert gnc["full_ladder_latency_ms"] == pytest.approx(687.5, abs=1e-9)
    assert len(gnc["rungs"]) == 4
    assert payload["false_trips"]["TASK-TLM"]["per_hour"] == pytest.approx(89.1155, rel=1e-5)


def test_json_renders_infinities_as_null(planted_report):
    # T-CLEAN declares no jitter sigma, so the rate is exactly zero and both
    # the tail argument and the mean time between false trips are infinite.
    payload = json.loads(format_json(planted_report))
    assert payload["false_trips"]["T-CLEAN"]["mtbft_hours"] is None
    assert payload["false_trips"]["T-CLEAN"]["z"] is None
    assert payload["false_trips"]["T-CLEAN"]["exact_zero"] is True


def test_report_to_dict_is_json_serialisable(planted_report):
    json.dumps(report_to_dict(planted_report))


def test_markdown_has_a_table_per_section(worked_report):
    text = format_markdown(worked_report)
    assert sum(1 for line in text.splitlines() if line.startswith("|---")) == 3
    assert "## Findings" in text
    assert SCOPE_NOTE in text


def test_markdown_of_a_clean_report_says_none(minimal_spec):
    text = format_markdown(run_checks(minimal_spec))
    assert "None." in text


def test_format_codes_lists_twelve_checks():
    text = format_codes()
    assert text.count("WS0") == 12


def test_renderers_are_deterministic(worked_report):
    assert format_report(worked_report) == format_report(worked_report)
    assert format_json(worked_report) == format_json(worked_report)
    assert format_markdown(worked_report) == format_markdown(worked_report)
