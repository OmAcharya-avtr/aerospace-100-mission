"""Edge cases and degenerate inputs."""

from __future__ import annotations

import pytest

from traceaudit import (
    Outcome,
    TestCase,
    TestReport,
    TraceConfig,
    audit,
    build_matrix,
    parse_junit_xml,
    parse_requirements_text,
    render_text,
)

NO_HEURISTIC = TraceConfig(heuristic_assertions=False)


@pytest.mark.verifies("REQ-011")
def test_no_requirements_and_no_tests_is_clean_with_a_zero_denominator():
    doc = parse_requirements_text("", source="s.md")
    report = TestReport(cases=(), source="j.xml", kind="junit-xml")
    result = audit(doc, report, config=NO_HEURISTIC)
    assert result.findings == ()
    assert result.exit_code == 0
    assert result.coverage[0].denominator == 0
    assert "undefined" in render_text(result)


@pytest.mark.verifies("REQ-013")
def test_requirements_but_no_tests_flags_every_requirement():
    doc = parse_requirements_text("## REQ-001 A\n## REQ-002 B\n", source="s.md")
    report = TestReport(cases=(), source="j.xml", kind="junit-xml")
    result = audit(doc, report, config=NO_HEURISTIC)
    assert [f.code for f in result.findings] == ["TA001", "TA001"]
    assert result.coverage[0].numerator == 0


@pytest.mark.verifies("REQ-014")
def test_tests_but_no_requirements_flags_every_claim():
    doc = parse_requirements_text("", source="s.md")
    report = TestReport(
        cases=(
            TestCase(node_id="t::a", classname="t", name="a", outcome=Outcome.PASSED,
                     claims=("REQ-001",)),
        ),
        source="j.xml", kind="junit-xml",
    )
    result = audit(doc, report, config=NO_HEURISTIC)
    assert [f.code for f in result.findings] == ["TA002"]


@pytest.mark.verifies("REQ-005")
def test_an_empty_junit_report_parses_to_zero_cases(tmp_path):
    path = tmp_path / "j.xml"
    path.write_text('<testsuite name="pytest" tests="0"></testsuite>', encoding="utf-8")
    report = parse_junit_xml(path)
    assert report.cases == ()
    assert report.claimed_ids == ()
    assert report.outcome_counts["passed"] == 0


@pytest.mark.verifies("REQ-010")
def test_one_test_claiming_many_requirements_traces_all_of_them():
    doc = parse_requirements_text("## REQ-001 A\n## REQ-002 B\n## REQ-003 C\n", source="s.md")
    report = TestReport(
        cases=(
            TestCase(node_id="t::a", classname="t", name="a", outcome=Outcome.PASSED,
                     claims=("REQ-001", "REQ-002", "REQ-003")),
        ),
        source="j.xml", kind="junit-xml",
    )
    result = audit(doc, report, config=NO_HEURISTIC)
    assert result.findings == ()
    assert result.coverage[0].numerator == 3


@pytest.mark.verifies("REQ-010")
def test_many_tests_claiming_one_requirement_list_in_report_order():
    doc = parse_requirements_text("## REQ-001 A\n", source="s.md")
    cases = tuple(
        TestCase(node_id=f"t::{i}", classname="t", name=f"t{i}", outcome=Outcome.PASSED,
                 claims=("REQ-001",))
        for i in range(5)
    )
    matrix = build_matrix(doc, TestReport(cases=cases, source="j.xml", kind="junit-xml"))
    assert matrix.requirement_to_tests["REQ-001"] == tuple(f"t::{i}" for i in range(5))


@pytest.mark.verifies("REQ-005")
def test_a_testcase_with_both_failure_and_skipped_children_reads_as_skipped(tmp_path):
    # Not emitted by pytest, but a hand-written or merged report can contain it.
    # The skipped element is checked first, which is stated in testreports.py.
    path = tmp_path / "j.xml"
    path.write_text(
        '<testsuite tests="1"><testcase classname="t" name="test_x">'
        '<skipped type="pytest.skip" message="s"/><failure message="f"/>'
        "</testcase></testsuite>",
        encoding="utf-8",
    )
    assert parse_junit_xml(path).cases[0].outcome is Outcome.SKIPPED


@pytest.mark.verifies("REQ-005")
def test_an_unknown_skipped_type_reads_as_skipped_not_xfailed(tmp_path):
    path = tmp_path / "j.xml"
    path.write_text(
        '<testsuite tests="1"><testcase classname="t" name="test_x">'
        '<skipped type="something.else" message="m"/></testcase></testsuite>',
        encoding="utf-8",
    )
    assert parse_junit_xml(path).cases[0].outcome is Outcome.SKIPPED


@pytest.mark.verifies("REQ-005")
def test_a_skipped_element_with_no_type_reads_as_skipped(tmp_path):
    path = tmp_path / "j.xml"
    path.write_text(
        '<testsuite tests="1"><testcase classname="t" name="test_x">'
        "<skipped/></testcase></testsuite>",
        encoding="utf-8",
    )
    case = parse_junit_xml(path).cases[0]
    assert case.outcome is Outcome.SKIPPED
    assert case.message == ""


@pytest.mark.verifies("REQ-006")
def test_separators_split_on_runs_and_drop_empties(tmp_path):
    path = tmp_path / "j.xml"
    path.write_text(
        '<testsuite tests="1"><testcase classname="t" name="test_x"><properties>'
        '<property name="requirements" value=" REQ-001 ,, ; REQ-002  "/>'
        "</properties></testcase></testsuite>",
        encoding="utf-8",
    )
    assert parse_junit_xml(path).cases[0].claims == ("REQ-001", "REQ-002")


@pytest.mark.verifies("REQ-006")
def test_a_custom_separator_is_honoured(tmp_path):
    path = tmp_path / "j.xml"
    path.write_text(
        '<testsuite tests="1"><testcase classname="t" name="test_x"><properties>'
        '<property name="requirements" value="REQ-001|REQ-002"/>'
        "</properties></testcase></testsuite>",
        encoding="utf-8",
    )
    cfg = TraceConfig(claim_value_separators="|")
    assert parse_junit_xml(path, config=cfg).cases[0].claims == ("REQ-001", "REQ-002")


@pytest.mark.verifies("REQ-001")
def test_a_declaration_with_a_trailing_colon_and_em_dash_is_parsed():
    doc = parse_requirements_text("## REQ-001: Colon\n## REQ-002 — Dash\n", source="s.md")
    assert [(r.id, r.title) for r in doc.declarations] == [
        ("REQ-001", "Colon"), ("REQ-002", "Dash")
    ]


@pytest.mark.verifies("REQ-001")
def test_crlf_line_endings_are_handled():
    doc = parse_requirements_text("## REQ-001 A\r\n## REQ-002 B\r\n", source="s.md")
    assert [r.id for r in doc.declarations] == ["REQ-001", "REQ-002"]
    assert doc.declarations[0].title == "A"


@pytest.mark.verifies("REQ-001")
def test_an_unclosed_code_fence_swallows_the_rest_of_the_document():
    # Documented behaviour: the fence state is a toggle, so an unclosed fence
    # hides everything after it. A silent loss of requirements would be worse.
    doc = parse_requirements_text("## REQ-001 A\n```\n## REQ-002 B\n", source="s.md")
    assert [r.id for r in doc.declarations] == ["REQ-001"]


@pytest.mark.verifies("REQ-018")
def test_a_claiming_test_with_no_matching_source_function_is_noted(tmp_path):
    (tmp_path / "test_other.py").write_text("def test_y():\n    assert 1 == 1\n",
                                            encoding="utf-8")
    doc = parse_requirements_text("## REQ-001 A\n", source="s.md")
    report = TestReport(
        cases=(
            TestCase(node_id="tests/test_gone.py::test_absent", classname="tests.test_gone",
                     name="test_absent", outcome=Outcome.PASSED, file="tests/test_gone.py",
                     claims=("REQ-001",)),
        ),
        source="j.xml", kind="junit-xml",
    )
    result = audit(doc, report, test_root=tmp_path)
    assert result.counts_by_code()["TA006"] == 0
    assert any("could not be matched to a source function" in n for n in result.notes)


@pytest.mark.verifies("REQ-018")
def test_a_parametrised_case_matches_its_source_function(tmp_path):
    (tmp_path / "test_p.py").write_text("def test_each():\n    pass\n", encoding="utf-8")
    doc = parse_requirements_text("## REQ-001 A\n", source="s.md")
    report = TestReport(
        cases=(
            TestCase(node_id="tests/test_p.py::test_each[3]", classname="tests.test_p",
                     name="test_each[3]", outcome=Outcome.PASSED, file="tests/test_p.py",
                     claims=("REQ-001",)),
        ),
        source="j.xml", kind="junit-xml",
    )
    result = audit(doc, report, test_root=tmp_path)
    assert result.counts_by_code()["TA006"] == 1


@pytest.mark.verifies("REQ-018")
def test_a_non_claiming_test_is_never_flagged_by_the_heuristic(tmp_path):
    (tmp_path / "test_q.py").write_text("def test_empty():\n    pass\n", encoding="utf-8")
    doc = parse_requirements_text("", source="s.md")
    report = TestReport(
        cases=(
            TestCase(node_id="tests/test_q.py::test_empty", classname="tests.test_q",
                     name="test_empty", outcome=Outcome.PASSED, file="tests/test_q.py"),
        ),
        source="j.xml", kind="junit-xml",
    )
    result = audit(doc, report, test_root=tmp_path)
    assert result.findings == ()


@pytest.mark.verifies("REQ-018")
def test_a_nonexistent_test_root_is_noted_not_raised(tmp_path):
    doc = parse_requirements_text("## REQ-001 A\n", source="s.md")
    report = TestReport(
        cases=(
            TestCase(node_id="t::a", classname="t", name="a", outcome=Outcome.PASSED,
                     claims=("REQ-001",)),
        ),
        source="j.xml", kind="junit-xml",
    )
    result = audit(doc, report, test_root=tmp_path / "absent")
    assert any("assertion heuristic not run" in n for n in result.notes)
    assert result.findings == ()


@pytest.mark.verifies("REQ-016")
def test_an_xpassed_test_is_indistinguishable_from_a_pass(tmp_path):
    # pytest writes an xpassed case as a bare <testcase>, so TA004 cannot see
    # it. This is a limitation of junit XML and is recorded in README.md.
    path = tmp_path / "j.xml"
    path.write_text(
        '<testsuite tests="1"><testcase classname="t" name="test_x"><properties>'
        '<property name="requirement_id" value="REQ-001"/>'
        "</properties></testcase></testsuite>",
        encoding="utf-8",
    )
    doc = parse_requirements_text("## REQ-001 A\n", source="s.md")
    result = audit(doc, parse_junit_xml(path), config=NO_HEURISTIC)
    assert result.findings == ()
    assert result.coverage[2].numerator == 1


@pytest.mark.verifies("REQ-018")
def test_index_records_paths_relative_to_the_test_root(tmp_path):
    from traceaudit import index_test_files

    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "test_nested.py").write_text("def test_n():\n    pass\n",
                                                     encoding="utf-8")
    index, _ = index_test_files(tmp_path)
    assert index[("test_nested", "test_n")].file == "sub/test_nested.py"
    assert str(tmp_path) not in index[("test_nested", "test_n")].file


@pytest.mark.verifies("REQ-022")
def test_ta006_location_uses_the_test_root_name_by_default(sample_project):
    from traceaudit import parse_requirements_file

    doc = parse_requirements_file(sample_project / "docs" / "REQUIREMENTS.md")
    report = parse_junit_xml(sample_project / "junit.xml")
    result = audit(doc, report, test_root=sample_project / "tests")
    finding = next(f for f in result.findings if f.code == "TA006")
    assert finding.location == "tests/test_monitor.py:42"


@pytest.mark.verifies("REQ-022")
def test_ta006_location_honours_relative_to(sample_project):
    from traceaudit import parse_requirements_file

    doc = parse_requirements_file(sample_project / "docs" / "REQUIREMENTS.md")
    report = parse_junit_xml(sample_project / "junit.xml")
    result = audit(doc, report, test_root=sample_project / "tests",
                   relative_to=sample_project.parent)
    finding = next(f for f in result.findings if f.code == "TA006")
    assert finding.location == "sample_project/tests/test_monitor.py:42"
    assert str(sample_project) not in finding.location
