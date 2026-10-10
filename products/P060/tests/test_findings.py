"""The checks themselves, each against a purpose-built input."""

from __future__ import annotations

import pytest

from traceaudit import (
    CODE_DESCRIPTIONS,
    Outcome,
    TestCase,
    TestReport,
    TraceConfig,
    audit,
    outcome_breakdown,
    parse_junit_xml,
    parse_requirements_file,
    parse_requirements_text,
)

DOC = (
    "## REQ-001 Traced and passing\n"
    "## REQ-002 Untraced\n"
    "## REQ-003 Skipped only\n"
    "## REQ-004 Failing only\n"
    "## REQ-005 Declared twice\n"
    "## REQ-005 Declared twice again\n"
)


def _case(name, outcome, claims, file="tests/test_x.py"):
    return TestCase(
        node_id=f"{file}::{name}",
        classname="tests.test_x",
        name=name,
        outcome=outcome,
        file=file,
        claims=tuple(claims),
    )


def _report():
    return TestReport(
        cases=(
            _case("test_one", Outcome.PASSED, ["REQ-001"]),
            _case("test_three", Outcome.SKIPPED, ["REQ-003"]),
            _case("test_four", Outcome.FAILED, ["REQ-004"]),
            _case("test_five", Outcome.PASSED, ["REQ-005"]),
            _case("test_ghost", Outcome.PASSED, ["REQ-099"]),
        ),
        source="j.xml",
        kind="junit-xml",
        declared_totals={"tests": 5, "failures": 1, "errors": 0, "skipped": 1},
    )


@pytest.fixture
def result():
    doc = parse_requirements_text(DOC, source="s.md")
    return audit(doc, _report(), config=TraceConfig(heuristic_assertions=False))


def _codes(result):
    return [f.code for f in result.findings]


@pytest.mark.verifies("REQ-013")
def test_ta001_is_reported_for_an_untraced_requirement(result):
    found = [f for f in result.findings if f.code == "TA001"]
    assert [f.subject for f in found] == ["REQ-002"]
    assert found[0].location == "s.md:2"
    assert "no test claims" in found[0].message


@pytest.mark.verifies("REQ-014")
def test_ta002_is_reported_for_an_undeclared_claim(result):
    found = [f for f in result.findings if f.code == "TA002"]
    assert [f.subject for f in found] == ["REQ-099"]
    assert "tests/test_x.py::test_ghost" in found[0].message


@pytest.mark.verifies("REQ-015")
def test_ta003_is_reported_once_per_duplicated_id_with_every_location(result):
    found = [f for f in result.findings if f.code == "TA003"]
    assert [f.subject for f in found] == ["REQ-005"]
    assert "s.md:5" in found[0].message and "s.md:6" in found[0].message
    assert "declared 2 times" in found[0].message


@pytest.mark.verifies("REQ-016")
def test_ta004_is_reported_for_a_skip_only_trace(result):
    found = [f for f in result.findings if f.code == "TA004"]
    assert [f.subject for f in found] == ["REQ-003"]
    assert "[skipped]" in found[0].message


@pytest.mark.verifies("REQ-016")
def test_ta004_names_xfail_separately(tmp_path):
    doc = parse_requirements_text("## REQ-001 A\n", source="s.md")
    report = TestReport(
        cases=(_case("test_a", Outcome.XFAILED, ["REQ-001"]),),
        source="j.xml", kind="junit-xml",
    )
    result = audit(doc, report, config=TraceConfig(heuristic_assertions=False))
    assert [f.code for f in result.findings] == ["TA004"]
    assert "[xfailed]" in result.findings[0].message


@pytest.mark.verifies("REQ-017")
def test_ta005_is_reported_for_a_failing_only_trace(result):
    found = [f for f in result.findings if f.code == "TA005"]
    assert [f.subject for f in found] == ["REQ-004"]
    assert "[failed]" in found[0].message


@pytest.mark.verifies("REQ-013")
def test_the_whole_finding_set_is_exactly_the_five_expected(result):
    assert _codes(result) == ["TA001", "TA002", "TA003", "TA004", "TA005"]
    assert result.exit_code == 1


@pytest.mark.verifies("REQ-020")
def test_a_clean_input_has_no_findings_and_exit_zero():
    doc = parse_requirements_text("## REQ-001 A\n", source="s.md")
    report = TestReport(
        cases=(_case("test_a", Outcome.PASSED, ["REQ-001"]),),
        source="j.xml", kind="junit-xml",
    )
    result = audit(doc, report, config=TraceConfig(heuristic_assertions=False))
    assert result.findings == ()
    assert result.exit_code == 0


@pytest.mark.verifies("REQ-019")
def test_an_ignored_code_is_reported_but_not_blocking():
    doc = parse_requirements_text(DOC, source="s.md")
    cfg = TraceConfig(heuristic_assertions=False, ignored_codes=("TA003", "TA005"))
    result = audit(doc, _report(), config=cfg)
    assert len(result.findings) == 5
    assert {f.code for f in result.blocking} == {"TA001", "TA002", "TA004"}
    ignored = [f for f in result.findings if f.ignored]
    assert {f.code for f in ignored} == {"TA003", "TA005"}
    assert result.exit_code == 1


@pytest.mark.verifies("REQ-019")
def test_ignoring_every_triggered_code_gives_exit_zero():
    doc = parse_requirements_text(DOC, source="s.md")
    cfg = TraceConfig(
        heuristic_assertions=False,
        ignored_codes=("TA001", "TA002", "TA003", "TA004", "TA005", "TA006"),
    )
    result = audit(doc, _report(), config=cfg)
    assert len(result.findings) == 5
    assert result.blocking == ()
    assert result.exit_code == 0


@pytest.mark.verifies("REQ-018")
def test_ta006_is_reported_and_marked_heuristic(sample_project):
    doc = parse_requirements_file(sample_project / "docs" / "REQUIREMENTS.md")
    report = parse_junit_xml(sample_project / "junit.xml")
    result = audit(doc, report, test_root=sample_project / "tests")
    found = [f for f in result.findings if f.code == "TA006"]
    assert len(found) == 1
    assert found[0].subject == "tests/test_monitor.py::test_request_counter"
    assert found[0].heuristic is True
    assert "no assertion detected" in found[0].message


@pytest.mark.verifies("REQ-018")
def test_ta006_is_absent_without_a_test_root(sample_project):
    doc = parse_requirements_file(sample_project / "docs" / "REQUIREMENTS.md")
    report = parse_junit_xml(sample_project / "junit.xml")
    result = audit(doc, report)
    assert [f.code for f in result.findings if f.code == "TA006"] == []
    assert any("no test source root" in note for note in result.notes)


@pytest.mark.verifies("REQ-018")
def test_the_heuristic_can_be_switched_off_entirely(sample_project):
    doc = parse_requirements_file(sample_project / "docs" / "REQUIREMENTS.md")
    report = parse_junit_xml(sample_project / "junit.xml")
    cfg = TraceConfig(heuristic_assertions=False)
    result = audit(doc, report, config=cfg, test_root=sample_project / "tests")
    assert [f.code for f in result.findings if f.code == "TA006"] == []
    assert any("disabled by configuration" in note for note in result.notes)


@pytest.mark.verifies("REQ-008")
def test_outcome_dependent_checks_are_skipped_on_a_collection_report():
    doc = parse_requirements_text(DOC, source="s.md")
    report = TestReport(
        cases=(_case("test_three", Outcome.UNKNOWN, ["REQ-003"]),),
        source="c.txt", kind="collect-only",
    )
    result = audit(doc, report, config=TraceConfig(heuristic_assertions=False))
    assert "TA004" not in _codes(result)
    assert "TA005" not in _codes(result)
    assert any("TA004 and TA005 were not computed" in note for note in result.notes)


@pytest.mark.verifies("REQ-009")
def test_a_header_total_mismatch_is_noted():
    doc = parse_requirements_text("## REQ-001 A\n", source="s.md")
    report = TestReport(
        cases=(_case("test_a", Outcome.PASSED, ["REQ-001"]),),
        source="j.xml", kind="junit-xml",
        declared_totals={"tests": 7, "failures": 0, "errors": 0, "skipped": 0},
    )
    result = audit(doc, report, config=TraceConfig(heuristic_assertions=False))
    assert any("tests=7" in note and "1 <testcase>" in note for note in result.notes)


@pytest.mark.verifies("REQ-009")
def test_a_skipped_total_mismatch_is_noted():
    doc = parse_requirements_text("## REQ-001 A\n", source="s.md")
    report = TestReport(
        cases=(_case("test_a", Outcome.SKIPPED, ["REQ-001"]),),
        source="j.xml", kind="junit-xml",
        declared_totals={"tests": 1, "failures": 0, "errors": 0, "skipped": 4},
    )
    result = audit(doc, report, config=TraceConfig(heuristic_assertions=False))
    assert any("skipped=4" in note for note in result.notes)


@pytest.mark.verifies("REQ-009")
def test_the_junit_skipped_total_counts_xfails_too(sample_project):
    # pytest writes skipped="2" for one skip plus one xfail; counting the cases
    # separates them, and no note is raised because the sum agrees.
    report = parse_junit_xml(sample_project / "junit.xml")
    assert report.declared_totals["skipped"] == 2
    assert report.outcome_counts["skipped"] == 1
    assert report.outcome_counts["xfailed"] == 1
    doc = parse_requirements_file(sample_project / "docs" / "REQUIREMENTS.md")
    result = audit(doc, report, config=TraceConfig(heuristic_assertions=False))
    assert not [n for n in result.notes if "skipped=" in n]


@pytest.mark.verifies("REQ-013")
def test_counts_by_code_is_zero_filled_over_every_code(result):
    counts = result.counts_by_code()
    assert set(counts) == set(CODE_DESCRIPTIONS)
    assert counts["TA006"] == 0
    assert sum(counts.values()) == len(result.findings)


@pytest.mark.verifies("REQ-013")
def test_findings_are_sorted_by_code_then_subject():
    doc = parse_requirements_text("## REQ-003 C\n## REQ-001 A\n## REQ-002 B\n", source="s.md")
    report = TestReport(cases=(), source="j.xml", kind="junit-xml")
    result = audit(doc, report, config=TraceConfig(heuristic_assertions=False))
    assert [f.subject for f in result.findings] == ["REQ-001", "REQ-002", "REQ-003"]


@pytest.mark.verifies("REQ-005")
def test_outcome_breakdown_drops_an_empty_unknown_bucket():
    report = TestReport(
        cases=(_case("test_a", Outcome.PASSED, []),), source="j.xml", kind="junit-xml"
    )
    assert "unknown" not in outcome_breakdown(report)
    report2 = TestReport(
        cases=(_case("test_a", Outcome.UNKNOWN, []),), source="c.txt", kind="collect-only"
    )
    assert outcome_breakdown(report2)["unknown"] == 1


@pytest.mark.verifies("REQ-018")
def test_finding_render_marks_heuristic_and_ignored():
    doc = parse_requirements_text(DOC, source="s.md")
    cfg = TraceConfig(heuristic_assertions=False, ignored_codes=("TA001",))
    result = audit(doc, _report(), config=cfg)
    rendered = next(f.render() for f in result.findings if f.code == "TA001")
    assert "ignored" in rendered
    assert "TA001 REQ-002" in rendered
