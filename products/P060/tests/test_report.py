"""Rendering: text, JSON and the markdown traceability table."""

from __future__ import annotations

import json

import pytest

from traceaudit import (
    BOOKKEEPING_NOTICE,
    Outcome,
    TestCase,
    TestReport,
    TraceConfig,
    audit,
    parse_junit_xml,
    parse_requirements_file,
    parse_requirements_text,
    render_json,
    render_markdown_matrix,
    render_text,
    to_dict,
)

DOC = "## REQ-001 Traced\n## REQ-002 Untraced\n"


def _result(kind="junit-xml"):
    doc = parse_requirements_text(DOC, source="s.md")
    report = TestReport(
        cases=(
            TestCase(
                node_id="tests/test_x.py::test_a",
                classname="tests.test_x",
                name="test_a",
                outcome=Outcome.PASSED if kind == "junit-xml" else Outcome.UNKNOWN,
                file="tests/test_x.py",
                claims=("REQ-001",),
            ),
        ),
        source="j.xml" if kind == "junit-xml" else "c.txt",
        kind=kind,
        declared_totals={"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
        if kind == "junit-xml" else {},
    )
    return audit(doc, report, config=TraceConfig(heuristic_assertions=False))


@pytest.mark.verifies("REQ-026")
def test_text_report_carries_the_bookkeeping_notice():
    text = render_text(_result())
    assert BOOKKEEPING_NOTICE in text
    assert "not evidence of adequacy" in text


@pytest.mark.verifies("REQ-026")
def test_json_report_carries_the_bookkeeping_notice():
    data = json.loads(render_json(_result()))
    assert data["bookkeeping_notice"] == BOOKKEEPING_NOTICE


@pytest.mark.verifies("REQ-011")
def test_text_report_prints_each_coverage_figure_with_its_denominator():
    text = render_text(_result())
    assert "nominal coverage: 1/2 = 50.0000 %" in text
    assert "executed coverage: 1/2 = 50.0000 %" in text
    assert "passing coverage: 1/2 = 50.0000 %" in text
    assert "definition: requirements with >=1 claiming test of any outcome" in text


@pytest.mark.verifies("REQ-011")
def test_text_report_states_that_outcome_coverage_is_undefined_for_a_collection_report():
    text = render_text(_result(kind="collect-only"))
    assert "executed coverage: undefined" in text
    assert "passing coverage : undefined" in text


@pytest.mark.verifies("REQ-010")
def test_show_matrix_prints_both_traced_and_untraced_rows():
    text = render_text(_result(), show_matrix=True)
    assert "requirement -> tests" in text
    assert "tests/test_x.py::test_a  [passed]" in text
    assert "REQ-002    -" in text


@pytest.mark.verifies("REQ-013")
def test_text_report_ends_with_the_finding_and_exit_summary():
    text = render_text(_result())
    assert "1 finding(s), 1 blocking; exit status 1" in text
    assert "TA001    1  requirement has no test" in text


@pytest.mark.verifies("REQ-013")
def test_text_report_says_none_when_there_are_no_findings():
    doc = parse_requirements_text("## REQ-001 A\n", source="s.md")
    report = TestReport(
        cases=(
            TestCase(node_id="t::a", classname="t", name="a", outcome=Outcome.PASSED,
                     claims=("REQ-001",)),
        ),
        source="j.xml", kind="junit-xml",
    )
    text = render_text(audit(doc, report, config=TraceConfig(heuristic_assertions=False)))
    assert "findings\n  none" in text
    assert "0 finding(s), 0 blocking; exit status 0" in text


@pytest.mark.verifies("REQ-023")
def test_json_report_contains_every_required_section():
    data = json.loads(render_json(_result()))
    for key in (
        "requirements_sources", "test_report", "declarations", "unique_requirement_ids",
        "coverage", "requirement_to_tests", "test_to_requirements", "unknown_claims",
        "findings", "counts_by_code", "notes", "exit_code",
    ):
        assert key in data, key


@pytest.mark.verifies("REQ-023")
def test_json_coverage_entries_carry_numerator_denominator_and_definition():
    data = json.loads(render_json(_result()))
    for figure in data["coverage"]:
        assert set(figure) == {"label", "numerator", "denominator", "percent", "definition"}
    nominal = next(f for f in data["coverage"] if f["label"] == "nominal coverage")
    assert (nominal["numerator"], nominal["denominator"]) == (1, 2)
    assert nominal["percent"] == pytest.approx(50.0)


@pytest.mark.verifies("REQ-023")
def test_json_findings_carry_code_subject_message_and_flags():
    data = json.loads(render_json(_result()))
    finding = data["findings"][0]
    assert finding["code"] == "TA001"
    assert finding["subject"] == "REQ-002"
    assert finding["heuristic"] is False
    assert finding["ignored"] is False


@pytest.mark.verifies("REQ-023")
def test_to_dict_is_json_serialisable_and_matches_render_json():
    result = _result()
    assert json.loads(render_json(result)) == json.loads(json.dumps(to_dict(result)))


@pytest.mark.verifies("REQ-023")
def test_json_test_to_requirements_omits_tests_with_no_claims():
    doc = parse_requirements_text("## REQ-001 A\n", source="s.md")
    report = TestReport(
        cases=(
            TestCase(node_id="t::a", classname="t", name="a", outcome=Outcome.PASSED,
                     claims=("REQ-001",)),
            TestCase(node_id="t::b", classname="t", name="b", outcome=Outcome.PASSED),
        ),
        source="j.xml", kind="junit-xml",
    )
    data = to_dict(audit(doc, report, config=TraceConfig(heuristic_assertions=False)))
    assert list(data["test_to_requirements"]) == ["t::a"]


@pytest.mark.verifies("REQ-024")
def test_markdown_matrix_has_a_row_per_requirement():
    table = render_markdown_matrix(_result())
    lines = table.splitlines()
    assert lines[0] == "| requirement | title | tests | outcomes |"
    assert len(lines) == 4
    assert "| REQ-001 | Traced | tests/test_x.py::test_a | passed |" in table
    assert "| REQ-002 | Untraced | none | - |" in table


@pytest.mark.verifies("REQ-024")
def test_markdown_matrix_joins_several_tests_with_a_line_break():
    doc = parse_requirements_text("## REQ-001 A\n", source="s.md")
    report = TestReport(
        cases=(
            TestCase(node_id="t::a", classname="t", name="a", outcome=Outcome.PASSED,
                     claims=("REQ-001",)),
            TestCase(node_id="t::b", classname="t", name="b", outcome=Outcome.SKIPPED,
                     claims=("REQ-001",)),
        ),
        source="j.xml", kind="junit-xml",
    )
    table = render_markdown_matrix(audit(doc, report,
                                         config=TraceConfig(heuristic_assertions=False)))
    assert "t::a<br>t::b" in table
    assert "passed<br>skipped" in table


@pytest.mark.verifies("REQ-024")
def test_markdown_matrix_escapes_a_pipe_in_a_title():
    doc = parse_requirements_text("## REQ-001 A | B\n", source="s.md")
    report = TestReport(cases=(), source="j.xml", kind="junit-xml")
    table = render_markdown_matrix(audit(doc, report,
                                         config=TraceConfig(heuristic_assertions=False)))
    assert "A \\| B" in table


@pytest.mark.verifies("REQ-005")
def test_text_report_prints_the_outcome_breakdown(sample_project):
    doc = parse_requirements_file(sample_project / "docs" / "REQUIREMENTS.md")
    report = parse_junit_xml(sample_project / "junit.xml")
    text = render_text(audit(doc, report, config=TraceConfig(heuristic_assertions=False)))
    assert "passed 5, failed 1, errored 0, skipped 1, xfailed 1, unknown 0" in text


@pytest.mark.verifies("REQ-023")
def test_notes_are_rendered_when_present(sample_project):
    doc = parse_requirements_file(sample_project / "docs" / "REQUIREMENTS.md")
    report = parse_junit_xml(sample_project / "junit.xml")
    text = render_text(audit(doc, report))
    assert "notes" in text
    assert "no test source root" in text
