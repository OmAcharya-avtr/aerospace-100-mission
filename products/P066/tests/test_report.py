"""The three renderings, and the scope statement that travels with them."""

from __future__ import annotations

import json

import pytest

from budgetroll import (
    json_report,
    markdown_report,
    rollup,
    run_checks,
    sensitivity,
    text_report,
    to_dict,
)
from budgetroll.report import SCOPE, findings_text, rollup_table, sensitivity_table


def test_text_report_carries_the_scope_statement(clean_mass):
    text = text_report(clean_mass)
    assert SCOPE in text
    assert "not flight-qualified" in text
    assert "not certified" in text


def test_text_report_carries_the_policy_source(clean_mass):
    assert clean_mass.policy.source.split(".")[0] in text_report(clean_mass)


def test_text_report_states_the_verdict(clean_mass, planted):
    assert "Verdict: no blocking findings" in text_report(clean_mass)
    assert "Verdict: BLOCKING FINDINGS" in text_report(planted)


def test_text_report_lists_all_four_combination_values(cubesat_power):
    text = text_report(cubesat_power)
    for label in (
        "linear (fully correlated upper bound)",
        "root-sum-square (independent)",
        "declared hierarchy at the root",
        "exact form with declared correlations",
    ):
        assert label in text


def test_markdown_report_is_a_table(planted):
    md = markdown_report(planted)
    assert md.startswith("# Budget rollup:")
    assert "| code | severity | subject | message |" in md
    assert "BR008" in md


def test_markdown_report_handles_a_clean_budget(clean_mass):
    md = markdown_report(clean_mass)
    assert "| - | - | - | no findings |" in md


def test_json_report_round_trips(cubesat_power):
    payload = json.loads(json_report(cubesat_power))
    assert payload["budget"]["unit"] == "W"
    assert payload["blocking"] is True
    assert len(payload["nodes"]) == 15  # 6 internal nodes plus 9 leaves
    assert len(payload["sensitivity"]) == 9
    assert payload["severity_counts"] == {"error": 4, "warning": 2, "info": 3}


def test_json_report_keys_are_sorted(cubesat_power):
    text = json_report(cubesat_power)
    payload = json.loads(text)
    assert list(payload) == sorted(payload)


def test_to_dict_combination_comparison(cubesat_link):
    comparison = to_dict(cubesat_link)["combination_comparison"]
    assert comparison["linear_all_leaves"] == pytest.approx(2.10, rel=0, abs=1e-12)
    assert comparison["rss_all_leaves"] < comparison["linear_all_leaves"]


def test_findings_text_on_an_empty_list():
    assert findings_text([]) == "no findings"


def test_rollup_table_indents_by_depth(clean_mass):
    lines = rollup_table(rollup(clean_mass)).splitlines()
    assert lines[3].startswith("Dry mass")
    assert lines[4].startswith("  Structure")
    assert lines[5].startswith("    Side panels")


def test_sensitivity_table_names_the_unit(clean_mass):
    table = sensitivity_table(sensitivity(clean_mass), "kg")
    assert "(magnitudes in kg; shares are dimensionless)" in table


def test_sensitivity_table_on_an_empty_ranking():
    assert "leaf" in sensitivity_table([], "kg")


def test_reports_are_deterministic(cubesat_power):
    assert text_report(cubesat_power) == text_report(cubesat_power)
    assert json_report(cubesat_power) == json_report(cubesat_power)
    assert markdown_report(cubesat_power) == markdown_report(cubesat_power)


def test_report_findings_match_run_checks(planted):
    payload = to_dict(planted)
    assert [f["code"] for f in payload["findings"]] == [f.code for f in run_checks(planted)]
