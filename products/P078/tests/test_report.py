"""The renderer. The clean verdict always carries the sentence that bounds it."""

from __future__ import annotations

import pytest
from conftest import tiny_case

from cmdguard.lint import LintStrength, static_lint
from cmdguard.report import (
    CLEAN_DISCLAIMER,
    render_check,
    render_comparison,
    render_dictionary_summary,
    render_state_at,
)
from cmdguard.trace import forward_execute

FORBIDDEN = ("flight-safe", "flight safe", "certified", "mission-ready", "production-ready")


def test_clean_render_carries_the_disclaimer(cases: dict) -> None:
    case = cases["nominal_science_pass"]
    text = render_check(forward_execute(case), case)
    assert CLEAN_DISCLAIMER in text
    assert "not a statement that the" in CLEAN_DISCLAIMER


@pytest.mark.parametrize("name", ["nominal_science_pass", "resource_and_exclusion"])
def test_no_render_makes_a_safety_claim(cases: dict, name: str) -> None:
    case = cases[name]
    report = forward_execute(case)
    blob = " ".join(
        [
            render_check(report, case),
            render_dictionary_summary(case),
            render_comparison(report, static_lint(case, LintStrength.WITH_TIMING)),
        ]
    ).lower()
    for phrase in FORBIDDEN:
        assert phrase not in blob
    assert (
        "safe to uplink" not in blob
        or "not a statement that the sequence is safe to uplink" in blob
    )


def test_violation_render_includes_every_trace_entry(cases: dict) -> None:
    report = forward_execute(cases["resource_and_exclusion"])
    text = render_check(report, cases["resource_and_exclusion"])
    for violation in report.violations:
        for entry in violation.trace:
            assert entry.render() in text


def test_render_check_lists_counts_by_kind(cases: dict) -> None:
    report = forward_execute(cases["resource_and_exclusion"])
    text = render_check(report)
    assert "resource_limit" in text
    assert "ordering-only defects : 6" in text
    assert "peak held draw" in text


def test_render_comparison_rejects_mismatched_cases(cases: dict) -> None:
    report = forward_execute(cases["nominal_science_pass"])
    lint = static_lint(cases["late_payload_enable"])
    with pytest.raises(ValueError, match="cannot compare"):
        render_comparison(report, lint)


def test_render_comparison_names_the_missed_and_the_extra(cases: dict) -> None:
    case = cases["resource_and_exclusion"]
    text = render_comparison(forward_execute(case), static_lint(case))
    assert "missed by the lint:" in text
    assert "lint miss rate on this case" in text


def test_render_comparison_handles_a_case_with_nothing_to_miss(cases: dict) -> None:
    case = cases["nominal_science_pass"]
    text = render_comparison(forward_execute(case), static_lint(case))
    assert "undefined (no violation to miss)" in text


def test_render_comparison_reports_agreement() -> None:
    case = tiny_case([(0.0, "SET_P")])
    text = render_comparison(forward_execute(case), static_lint(case))
    assert "agreed with the forward walk" in text


def test_render_state_at_prints_predicates_and_mode(cases: dict) -> None:
    report = forward_execute(cases["late_payload_enable"])
    text = render_state_at(report, 5)
    assert "command          : PAYLOAD_SAFE" in text
    assert "payload_enabled" in text
    assert "mode             : SCIENCE" in text


@pytest.mark.parametrize("step", [-1, 99])
def test_render_state_at_rejects_an_out_of_range_step(cases: dict, step: int) -> None:
    report = forward_execute(cases["late_payload_enable"])
    with pytest.raises(IndexError, match="outside the sequence"):
        render_state_at(report, step)


def test_dictionary_summary_reports_declarations(cases: dict) -> None:
    text = render_dictionary_summary(cases["irreversible_before_verify"])
    assert "irreversible commands    : 1" in text
    assert "ANT_DEPLOY" in text
    assert "unverified at t=0      : 1" in text
    assert "attitude_verified" in text


def test_dictionary_summary_lists_undeclared_commands_used() -> None:
    case = tiny_case([(0.0, "SET_P"), (1.0, "GHOST")])
    text = render_dictionary_summary(case)
    assert "undeclared commands used : 1" in text
    assert "GHOST" in text
