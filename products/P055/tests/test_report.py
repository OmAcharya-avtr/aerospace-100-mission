"""Tests for report formatting, including that denominators are always printed."""

from __future__ import annotations

import json

from conftest import document, edge, node

from assuregraph import (
    SCOPE_STATEMENT,
    format_coverage_table,
    format_findings,
    format_report,
    parse_case,
    report_as_dict,
    run_checks,
)


def _incomplete(tmp_path):
    doc = document(
        [
            node("G1", "goal"),
            node("G5", "goal"),
            node("Sn1", "solution", evidence={"path": "nope.txt"}),
            node("A1", "assumption"),
        ],
        [edge("G1", "Sn1"), edge("G1", "A1", "in_context_of")],
        top_goals=["G1"],
    )
    return run_checks(parse_case(doc, base_dir=str(tmp_path)))


def test_scope_statement_mentions_the_three_things_it_must(tmp_path) -> None:
    assert "not a safe system" in SCOPE_STATEMENT
    assert "no argument's soundness" in SCOPE_STATEMENT
    assert "not a DO-178C or ARP4754A compliance tool" in SCOPE_STATEMENT


def test_report_carries_the_scope_statement(tmp_path) -> None:
    assert SCOPE_STATEMENT in format_report(_incomplete(tmp_path))


def test_report_states_the_exit_code(tmp_path) -> None:
    text = format_report(_incomplete(tmp_path))
    assert "Exit code 1" in text
    assert "INCOMPLETE" in text


def test_complete_report_states_what_complete_does_not_mean(tmp_path, artifact) -> None:
    _, digest = artifact("a.txt")
    doc = document(
        [
            node("G1", "goal"),
            node("Sn1", "solution", evidence={"path": "a.txt", "sha256": digest}),
        ],
        [edge("G1", "Sn1")],
        top_goals=["G1"],
    )
    text = format_report(run_checks(parse_case(doc, base_dir=str(tmp_path))))
    assert "Exit code 0" in text
    assert "not a statement about any system" in text


def test_every_coverage_fraction_prints_its_denominator(tmp_path) -> None:
    table = format_coverage_table(_incomplete(tmp_path).coverage)
    assert table.count("in denominator") >= 3


def test_undefined_fraction_prints_as_not_available() -> None:
    cov = run_checks(parse_case(document([node("G1", "goal", undeveloped=True)]))).coverage
    table = format_coverage_table(cov)
    assert "n/a (denominator 0)" in table
    assert "100.0 % (0 in denominator)" not in table


def test_unverifiable_count_is_stated_as_not_fresh(tmp_path, artifact) -> None:
    artifact("a.txt")
    doc = document(
        [node("G1", "goal"), node("Sn1", "solution", evidence={"path": "a.txt"})],
        [edge("G1", "Sn1")],
        top_goals=["G1"],
    )
    table = format_coverage_table(run_checks(parse_case(doc, base_dir=str(tmp_path))).coverage)
    assert "they are not fresh" in table


def test_findings_text_names_every_check_even_with_zero_findings(tmp_path) -> None:
    text = format_findings(_incomplete(tmp_path))
    for check in ("unsupported_claims", "missing_evidence", "cycles", "orphan_nodes"):
        assert check in text


def test_findings_markdown_is_a_table(tmp_path) -> None:
    text = format_findings(_incomplete(tmp_path), markdown=True)
    assert text.lstrip().startswith("| check")
    assert "|---" in text


def test_findings_text_wraps_to_the_requested_width(tmp_path) -> None:
    # Every line fits the width unless it holds a single token longer than the
    # width -- a digest or a path, which must not be split because the reader
    # copies it. Checked by asserting that any over-long line is a one-token
    # line.
    text = format_findings(_incomplete(tmp_path), width=70)
    for line in text.splitlines():
        if len(line) > 70:
            assert len(line.split()) == 1, line


def test_no_findings_says_so(tmp_path, artifact) -> None:
    _, digest = artifact("a.txt")
    doc = document(
        [
            node("G1", "goal"),
            node("Sn1", "solution", evidence={"path": "a.txt", "sha256": digest}),
        ],
        [edge("G1", "Sn1")],
        top_goals=["G1"],
    )
    text = format_findings(run_checks(parse_case(doc, base_dir=str(tmp_path))))
    assert "No findings." in text


def test_markdown_report_has_headings(tmp_path) -> None:
    text = format_report(_incomplete(tmp_path), markdown=True)
    assert text.startswith("# assuregraph report")
    assert "## Findings" in text
    assert "## Coverage summary" in text


def test_report_as_dict_is_json_serialisable(tmp_path) -> None:
    payload = report_as_dict(_incomplete(tmp_path))
    text = json.dumps(payload)
    assert json.loads(text)["exit_code"] == 1


def test_report_as_dict_states_denominators_next_to_fractions(tmp_path) -> None:
    payload = report_as_dict(_incomplete(tmp_path))
    assert set(payload["coverage"]["fractions"]) == set(payload["coverage"]["denominators"])


def test_report_as_dict_carries_every_node_id(tmp_path) -> None:
    payload = report_as_dict(_incomplete(tmp_path))
    flagged = {i for f in payload["findings"] for i in f["node_ids"]}
    assert {"G5", "Sn1", "A1"} <= flagged


def test_report_as_dict_carries_the_scope_statement(tmp_path) -> None:
    assert report_as_dict(_incomplete(tmp_path))["scope_statement"] == SCOPE_STATEMENT


def test_format_report_width_is_passed_through(tmp_path) -> None:
    # Only the findings blocks wrap; the scope statement and the coverage table
    # are not reflowed. Compare the wrapped region rather than the whole report.
    def finding_lines(text: str) -> list[str]:
        body = text.split("Findings", 1)[1].split("Coverage summary", 1)[0]
        return [line for line in body.splitlines() if line.startswith("    ")]

    narrow = finding_lines(format_report(_incomplete(tmp_path), width=60))
    wide = finding_lines(format_report(_incomplete(tmp_path), width=400))
    assert len(narrow) > len(wide)
    assert max(len(line) for line in wide) > 60


def test_format_report_width_does_not_change_the_content(tmp_path) -> None:
    # Wrapping must only move line breaks; no word may be lost, added or split.
    narrow = format_report(_incomplete(tmp_path), width=60).split()
    wide = format_report(_incomplete(tmp_path), width=400).split()
    assert narrow == wide


def test_a_digest_is_never_split_across_a_line_break(tmp_path, artifact) -> None:
    # A 64-character SHA-256 digest is longer than a narrow wrap column. It must
    # still appear whole on one line, because the reader copies it.
    _, digest = artifact("a.txt", "v1\n")
    doc = document(
        [
            node("G1", "goal"),
            node("Sn1", "solution", evidence={"path": "a.txt", "sha256": digest}),
        ],
        [edge("G1", "Sn1")],
        top_goals=["G1"],
    )
    case = parse_case(doc, base_dir=str(tmp_path))
    (tmp_path / "a.txt").write_text("v2\n", encoding="utf-8")
    text = format_report(run_checks(case), width=40)
    assert digest in text
