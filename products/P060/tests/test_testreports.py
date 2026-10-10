"""Test-report parsing: junit XML and pytest collection output."""

from __future__ import annotations

import pytest

from traceaudit import (
    Outcome,
    TraceConfig,
    load_report,
    parse_collection_report,
    parse_junit_xml,
)

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="1" failures="1" skipped="2" tests="6">
<testcase classname="tests.test_a" name="test_pass" file="tests/test_a.py" line="4">
  <properties><property name="requirement_id" value="REQ-001"/></properties></testcase>
<testcase classname="tests.test_a" name="test_fail" file="tests/test_a.py" line="9">
  <failure message="assert 1 == 2">trace</failure></testcase>
<testcase classname="tests.test_a" name="test_error" file="tests/test_a.py" line="14">
  <error message="fixture blew up">trace</error></testcase>
<testcase classname="tests.test_a" name="test_skip" file="tests/test_a.py" line="19">
  <skipped type="pytest.skip" message="no simulator">tests/test_a.py:20: no simulator</skipped></testcase>
<testcase classname="tests.test_a" name="test_xfail" file="tests/test_a.py" line="24">
  <skipped type="pytest.xfail" message="known defect"/></testcase>
<testcase classname="tests.test_b.TestGroup" name="test_inner[3-4]" file="tests/test_b.py" line="7">
  <properties><property name="requirements" value="REQ-002, REQ-003"/></properties></testcase>
</testsuite></testsuites>
"""
# Hand count: 6 cases; one each passed/failed/errored/skipped/xfailed, one more passed.


@pytest.fixture
def junit_file(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(JUNIT, encoding="utf-8")
    return path


@pytest.mark.verifies("REQ-005")
def test_outcomes_are_classified_from_the_child_elements(junit_file):
    report = parse_junit_xml(junit_file)
    by_name = {c.name: c.outcome for c in report.cases}
    assert by_name["test_pass"] is Outcome.PASSED
    assert by_name["test_fail"] is Outcome.FAILED
    assert by_name["test_error"] is Outcome.ERRORED
    assert by_name["test_skip"] is Outcome.SKIPPED
    assert by_name["test_xfail"] is Outcome.XFAILED


@pytest.mark.verifies("REQ-005")
def test_outcome_counts_are_counted_from_the_cases(junit_file):
    report = parse_junit_xml(junit_file)
    counts = report.outcome_counts
    assert counts == {
        "passed": 2, "failed": 1, "errored": 1,
        "skipped": 1, "xfailed": 1, "unknown": 0,
    }


@pytest.mark.verifies("REQ-005")
def test_executed_and_illusory_classify_the_outcomes():
    assert Outcome.PASSED.executed and Outcome.FAILED.executed and Outcome.ERRORED.executed
    assert not Outcome.SKIPPED.executed and not Outcome.XFAILED.executed
    assert Outcome.SKIPPED.illusory and Outcome.XFAILED.illusory
    assert not Outcome.PASSED.illusory and not Outcome.UNKNOWN.illusory


@pytest.mark.verifies("REQ-005")
def test_skip_message_and_line_are_recorded(junit_file):
    report = parse_junit_xml(junit_file)
    case = next(c for c in report.cases if c.name == "test_skip")
    assert case.message == "no simulator"
    assert case.line == 20  # junit writes a zero-based line; 19 + 1
    assert case.file == "tests/test_a.py"


@pytest.mark.verifies("REQ-006")
def test_single_property_claim_is_read(junit_file):
    report = parse_junit_xml(junit_file)
    case = next(c for c in report.cases if c.name == "test_pass")
    assert case.claims == ("REQ-001",)
    assert case.properties == (("requirement_id", "REQ-001"),)


@pytest.mark.verifies("REQ-006")
def test_multiple_ids_in_one_property_value_are_split(junit_file):
    report = parse_junit_xml(junit_file)
    case = next(c for c in report.cases if c.name.startswith("test_inner"))
    assert case.claims == ("REQ-002", "REQ-003")


@pytest.mark.verifies("REQ-006")
def test_property_names_outside_the_configured_set_are_ignored(tmp_path):
    xml = ('<testsuite tests="1"><testcase classname="t" name="test_x">'
           '<properties><property name="ticket" value="REQ-005"/></properties>'
           "</testcase></testsuite>")
    path = tmp_path / "j.xml"
    path.write_text(xml, encoding="utf-8")
    report = parse_junit_xml(path, config=TraceConfig(claim_from_node_id=False))
    assert report.cases[0].claims == ()


@pytest.mark.verifies("REQ-006")
def test_a_value_that_is_not_an_id_is_not_claimed(tmp_path):
    xml = ('<testsuite tests="1"><testcase classname="t" name="test_x">'
           '<properties><property name="requirement_id" value="not-an-id"/></properties>'
           "</testcase></testsuite>")
    path = tmp_path / "j.xml"
    path.write_text(xml, encoding="utf-8")
    report = parse_junit_xml(path)
    assert report.cases[0].claims == ()


@pytest.mark.verifies("REQ-007")
def test_node_id_claims_are_read_with_the_default_pattern(tmp_path):
    # The default node-id pattern is case sensitive: REQ_014 matches, req_014 does not.
    xml = ('<testsuite tests="2">'
           '<testcase classname="tests.test_REQ_014" name="test_margin" '
           'file="tests/test_REQ_014.py"/>'
           '<testcase classname="tests.test_req_015" name="test_other" '
           'file="tests/test_req_015.py"/></testsuite>')
    path = tmp_path / "j.xml"
    path.write_text(xml, encoding="utf-8")
    report = parse_junit_xml(path)
    assert report.cases[0].claims == ("REQ-014",)
    assert report.cases[1].claims == ()


@pytest.mark.verifies("REQ-007")
def test_node_id_claims_can_be_disabled(tmp_path):
    xml = ('<testsuite tests="1"><testcase classname="tests.test_REQ_014" '
           'name="test_margin" file="tests/test_REQ_014.py"/></testsuite>')
    path = tmp_path / "j.xml"
    path.write_text(xml, encoding="utf-8")
    report = parse_junit_xml(path, config=TraceConfig(claim_from_node_id=False))
    assert report.cases[0].claims == ()


@pytest.mark.verifies("REQ-007")
def test_node_id_claim_pattern_and_template_are_configurable(tmp_path):
    xml = '<testsuite tests="1"><testcase classname="t" name="test_SRS_GN_07"/></testsuite>'
    path = tmp_path / "j.xml"
    path.write_text(xml, encoding="utf-8")
    cfg = TraceConfig(
        id_pattern=r"SRS-[A-Z]{2}-\d{2}",
        node_id_claim_pattern=r"SRS_([A-Z]{2})_(\d{2})",
        node_id_claim_template="SRS-{0}-{1}",
    )
    report = parse_junit_xml(path, config=cfg)
    assert report.cases[0].claims == ("SRS-GN-07",)


@pytest.mark.verifies("REQ-007")
def test_a_built_claim_that_does_not_match_the_id_pattern_is_dropped(tmp_path):
    # The template must produce something the id pattern accepts; lower case does not.
    xml = '<testsuite tests="1"><testcase classname="t" name="test_srs_gn_07"/></testsuite>'
    path = tmp_path / "j.xml"
    path.write_text(xml, encoding="utf-8")
    cfg = TraceConfig(
        id_pattern=r"SRS-[A-Z]{2}-\d{2}",
        node_id_claim_pattern=r"srs_([a-z]{2})_(\d{2})",
        node_id_claim_template="SRS-{0}-{1}",
    )
    assert parse_junit_xml(path, config=cfg).cases[0].claims == ()


@pytest.mark.verifies("REQ-005")
def test_node_id_is_built_from_file_class_and_name(junit_file):
    report = parse_junit_xml(junit_file)
    ids = [c.node_id for c in report.cases]
    assert "tests/test_a.py::test_pass" in ids
    assert "tests/test_b.py::TestGroup::test_inner[3-4]" in ids


@pytest.mark.verifies("REQ-005")
def test_node_id_falls_back_to_classname_when_no_file_attribute(tmp_path):
    xml = '<testsuite tests="1"><testcase classname="tests.test_a" name="test_x"/></testsuite>'
    path = tmp_path / "j.xml"
    path.write_text(xml, encoding="utf-8")
    report = parse_junit_xml(path)
    assert report.cases[0].node_id == "tests.test_a::test_x"
    assert report.cases[0].file is None


@pytest.mark.verifies("REQ-009")
def test_declared_totals_are_read_from_the_testsuite_element(junit_file):
    report = parse_junit_xml(junit_file)
    assert report.declared_totals == {
        "tests": 6, "failures": 1, "errors": 1, "skipped": 2
    }


@pytest.mark.verifies("REQ-005")
def test_function_name_strips_the_parametrisation_suffix(junit_file):
    report = parse_junit_xml(junit_file)
    case = next(c for c in report.cases if c.name.startswith("test_inner"))
    assert case.name == "test_inner[3-4]"
    assert case.function_name == "test_inner"


@pytest.mark.verifies("REQ-005")
def test_claimed_ids_are_unique_and_in_first_seen_order(junit_file):
    report = parse_junit_xml(junit_file)
    assert report.claimed_ids == ("REQ-001", "REQ-002", "REQ-003")


@pytest.mark.verifies("REQ-008")
def test_collection_report_gives_unknown_outcomes(product_root):
    report = parse_collection_report(
        product_root / "fixtures" / "sample_project" / "collect_only.txt"
    )
    assert len(report.cases) == 8
    assert {c.outcome for c in report.cases} == {Outcome.UNKNOWN}
    assert report.kind == "collect-only"


@pytest.mark.verifies("REQ-008")
def test_collection_report_recovers_file_class_and_name(tmp_path):
    path = tmp_path / "collect.txt"
    path.write_text(
        "tests/test_a.py::test_one\n"
        "tests/sub/test_b.py::TestGroup::test_two[1-2]\n"
        "\n8 tests collected in 0.19s\n",
        encoding="utf-8",
    )
    report = parse_collection_report(path)
    assert [c.node_id for c in report.cases] == [
        "tests/test_a.py::test_one",
        "tests/sub/test_b.py::TestGroup::test_two[1-2]",
    ]
    assert report.cases[1].classname == "tests.sub.test_b.TestGroup"
    assert report.cases[1].name == "test_two[1-2]"
    assert report.cases[1].file == "tests/sub/test_b.py"


@pytest.mark.verifies("REQ-008")
def test_collection_report_ignores_the_summary_lines(tmp_path):
    path = tmp_path / "collect.txt"
    path.write_text("tests/test_a.py::test_one\n\n1 test collected in 0.11s\n",
                    encoding="utf-8")
    report = parse_collection_report(path)
    assert len(report.cases) == 1


@pytest.mark.verifies("REQ-008")
def test_load_report_dispatches_on_the_suffix(junit_file, tmp_path):
    assert load_report(junit_file).kind == "junit-xml"
    path = tmp_path / "collect.txt"
    path.write_text("tests/test_a.py::test_one\n", encoding="utf-8")
    assert load_report(path).kind == "collect-only"


@pytest.mark.verifies("REQ-022")
def test_missing_junit_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="junit XML report not found"):
        parse_junit_xml(tmp_path / "absent.xml")


@pytest.mark.verifies("REQ-022")
def test_unparseable_xml_raises_value_error(tmp_path):
    path = tmp_path / "broken.xml"
    path.write_text("<testsuite><testcase>", encoding="utf-8")
    with pytest.raises(ValueError, match="not parseable XML"):
        parse_junit_xml(path)


@pytest.mark.verifies("REQ-022")
def test_xml_without_a_testsuite_raises_value_error(tmp_path):
    path = tmp_path / "other.xml"
    path.write_text("<coverage><file name='a.py'/></coverage>", encoding="utf-8")
    with pytest.raises(ValueError, match="no <testsuite> element"):
        parse_junit_xml(path)


@pytest.mark.verifies("REQ-022")
def test_missing_collection_report_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="collection report not found"):
        parse_collection_report(tmp_path / "absent.txt")


@pytest.mark.verifies("REQ-005")
def test_bare_testsuite_root_is_accepted(tmp_path):
    path = tmp_path / "j.xml"
    path.write_text(
        '<testsuite tests="1"><testcase classname="t" name="test_x"/></testsuite>',
        encoding="utf-8",
    )
    assert len(parse_junit_xml(path).cases) == 1


@pytest.mark.verifies("REQ-005")
def test_several_testsuites_are_concatenated(tmp_path):
    path = tmp_path / "j.xml"
    path.write_text(
        '<testsuites><testsuite tests="1"><testcase classname="a" name="test_x"/></testsuite>'
        '<testsuite tests="1"><testcase classname="b" name="test_y"/></testsuite></testsuites>',
        encoding="utf-8",
    )
    report = parse_junit_xml(path)
    assert len(report.cases) == 2
    assert report.declared_totals["tests"] == 2


@pytest.mark.verifies("REQ-022")
def test_relative_to_is_applied_to_the_report_label(tmp_path):
    (tmp_path / "out").mkdir()
    path = tmp_path / "out" / "junit.xml"
    path.write_text('<testsuite tests="0"></testsuite>', encoding="utf-8")
    report = parse_junit_xml(path, relative_to=tmp_path)
    assert report.source == "out/junit.xml"
