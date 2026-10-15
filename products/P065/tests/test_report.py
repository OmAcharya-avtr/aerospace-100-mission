"""Renderers: text, JSON, markdown and the check catalogue."""

from __future__ import annotations

import json

from hazardtrace import (
    SCOPE_NOTE,
    format_codes,
    format_coverage_table,
    format_findings,
    format_grid,
    format_json,
    format_markdown,
    format_report,
    load_analysis,
    load_junit,
    load_requirements,
    report_to_dict,
    run_checks,
)
from hazardtrace.checks import CHECKS


def _worked(cases_dir, fixtures_dir):
    return run_checks(
        load_analysis(cases_dir / "p024_bdot_detumble.yaml"),
        requirements=load_requirements(
            fixtures_dir / "p024_bdot" / "docs" / "REQUIREMENTS.md"
        ),
        tests=load_junit(fixtures_dir / "p024_bdot" / "junit.xml"),
    )


def test_text_report_carries_the_scope_note(cases_dir, fixtures_dir):
    text = format_report(_worked(cases_dir, fixtures_dir))
    assert SCOPE_NOTE in text
    assert "INCOMPLETE" in text
    assert "Exit code 1" in text


def test_text_report_is_byte_identical_between_runs(cases_dir, fixtures_dir):
    first = format_report(_worked(cases_dir, fixtures_dir))
    second = format_report(_worked(cases_dir, fixtures_dir))
    assert first == second


def test_text_report_names_every_finding(cases_dir, fixtures_dir):
    report = _worked(cases_dir, fixtures_dir)
    text = format_report(report)
    for finding in report.findings:
        assert finding.code in text
        assert finding.message in text


def test_coverage_table_prints_every_denominator(cases_dir, fixtures_dir):
    table = format_coverage_table(_worked(cases_dir, fixtures_dir).coverage)
    for label, ratio in _worked(cases_dir, fixtures_dir).coverage.as_items():
        assert label in table
        assert ratio.definition in table
    assert "/8 = 75.0000 %" in table


def test_undefined_ratio_is_printed_as_undefined(cases_dir):
    report = run_checks(load_analysis(cases_dir / "complete_minimal.yaml"))
    table = format_coverage_table(report.coverage)
    assert "undefined" in table
    assert report.coverage.test_resolution.defined is False
    assert report.coverage.test_resolution.fraction != report.coverage.test_resolution.fraction


def test_grid_renders_one_row_per_control_action(cases_dir):
    report = run_checks(load_analysis(cases_dir / "p024_bdot_detumble.yaml"))
    text = format_grid(report)
    assert text.count("CA-DIPOLE") == 1
    assert text.count("CA-EXIT") == 1
    assert "cells covered: 6/8" in text
    assert text == text.rstrip("\n") + "\n"


def test_findings_renderer_says_none_when_clean(cases_dir):
    report = run_checks(load_analysis(cases_dir / "complete_minimal.yaml"))
    assert format_findings(report).strip().endswith("none")


def test_json_is_valid_and_round_trips(cases_dir, fixtures_dir):
    report = _worked(cases_dir, fixtures_dir)
    payload = json.loads(format_json(report))
    assert payload == json.loads(json.dumps(report_to_dict(report)))
    assert payload["verdict"]["exit_code"] == 1
    assert payload["verdict"]["scope_note"] == SCOPE_NOTE
    assert payload["grid"]["total_cells"] == 8
    assert payload["grid"]["covered_cells"] == 6
    assert len(payload["findings"]) == 11
    assert len(payload["grid"]["cells"]) == 8


def test_json_counts_match_the_analysis(cases_dir, fixtures_dir):
    payload = json.loads(format_json(_worked(cases_dir, fixtures_dir)))
    assert payload["counts"] == {
        "losses": 4,
        "hazards": 6,
        "constraints": 5,
        "control_actions": 2,
        "feedback_paths": 1,
        "ucas": 7,
        "exclusions": 1,
        "loss_scenarios": 6,
        "mitigations": 6,
        "evidence": 7,
    }


def test_markdown_has_a_table_per_section(cases_dir, fixtures_dir):
    text = format_markdown(_worked(cases_dir, fixtures_dir))
    assert "## Coverage" in text
    assert "## Enumeration grid" in text
    assert "## Findings" in text
    assert "**gap**" in text
    assert SCOPE_NOTE in text


def test_markdown_says_none_when_clean(cases_dir):
    report = run_checks(load_analysis(cases_dir / "complete_minimal.yaml"))
    text = format_markdown(report)
    assert "## Findings\n\nNone." in text
    assert "**COMPLETE**" in text


def test_codes_table_lists_every_check():
    text = format_codes()
    for spec in CHECKS:
        assert spec.code in text
        assert spec.title in text
