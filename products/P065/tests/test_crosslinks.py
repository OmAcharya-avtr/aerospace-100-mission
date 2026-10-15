"""The two repository parsers: requirements markdown and junit XML."""

from __future__ import annotations

import pytest

from hazardtrace import (
    AnalysisFormatError,
    TestOutcome,
    load_junit,
    load_requirements,
    parse_junit,
    parse_requirements,
)

DOC = """
# Requirements

### REQ-001 Heading form

Prose that mentions REQ-004 mid-sentence, which is not a declaration.

- REQ-002 Bullet form

**REQ-003** Bold form

```
### REQ-900 inside a fence
```

### REQ-001 Declared a second time
"""


def test_declaration_forms():
    # Hand count: heading, bullet and bold forms are declarations; the
    # mid-sentence mention and the fenced line are not; REQ-001 is declared
    # twice, so `declared` has 4 entries over 3 unique ids.
    index = parse_requirements(DOC, source="doc.md")
    assert index.unique == {"REQ-001", "REQ-002", "REQ-003"}
    assert index.declared == ("REQ-001", "REQ-002", "REQ-003", "REQ-001")
    assert "REQ-004" not in index
    assert "REQ-900" not in index


def test_first_declaration_line_wins():
    index = parse_requirements(DOC)
    assert index.lines["REQ-001"] == 4
    assert index.source == "<string>"


def test_contains_accepts_a_non_string():
    index = parse_requirements(DOC)
    assert 3 not in index


def test_custom_pattern():
    index = parse_requirements("## SR-12 something\n", pattern=r"SR-\d+")
    assert index.unique == {"SR-12"}


def test_invalid_pattern_is_rejected():
    with pytest.raises(AnalysisFormatError):
        parse_requirements("x", pattern="REQ-(")


def test_unreadable_requirements_file(tmp_path):
    with pytest.raises(AnalysisFormatError):
        load_requirements(tmp_path / "absent.md")


def test_shipped_requirements_fixture(fixtures_dir):
    index = load_requirements(fixtures_dir / "p024_bdot" / "docs" / "REQUIREMENTS.md")
    assert len(index.unique) == 13
    assert "REQ-021" not in index
    assert "REQ-022" not in index


JUNIT_ALL_OUTCOMES = (
    '<testsuites><testsuite name="pytest" tests="5">'
    '<testcase classname="tests.test_x" name="t_pass" file="tests/test_x.py"/>'
    '<testcase classname="tests.test_x" name="t_fail"><failure message="m"/></testcase>'
    '<testcase classname="tests.test_x" name="t_err"><error message="m"/></testcase>'
    '<testcase classname="tests.test_x" name="t_skip">'
    '<skipped type="pytest.skip" message="m"/></testcase>'
    '<testcase classname="tests.test_x" name="t_xfail">'
    '<skipped type="pytest.xfail" message="m"/></testcase>'
    "</testsuite></testsuites>"
)


def test_all_five_outcomes_are_classified():
    index = parse_junit(JUNIT_ALL_OUTCOMES, source="junit.xml")
    assert index.counts == {
        TestOutcome.PASSED: 1,
        TestOutcome.FAILED: 1,
        TestOutcome.ERRORED: 1,
        TestOutcome.SKIPPED: 1,
        TestOutcome.XFAILED: 1,
    }
    assert TestOutcome.PASSED.is_pass is True
    for outcome in (
        TestOutcome.FAILED,
        TestOutcome.ERRORED,
        TestOutcome.SKIPPED,
        TestOutcome.XFAILED,
    ):
        assert outcome.is_pass is False


def test_identifier_aliases():
    index = parse_junit(JUNIT_ALL_OUTCOMES)
    for identifier in (
        "t_pass",
        "tests.test_x::t_pass",
        "tests.test_x.t_pass",
        "tests/test_x.py::t_pass",
    ):
        assert len(index.lookup(identifier)) == 1, identifier
    assert index.lookup("tests/test_y.py::t_pass") == ()


def test_nested_class_aliases():
    xml = (
        '<testsuite name="pytest" tests="1"><testcase '
        'classname="tests.test_x.TestGroup" name="t_one"/></testsuite>'
    )
    index = parse_junit(xml)
    assert len(index.lookup("tests/test_x.py::TestGroup::t_one")) == 1
    assert len(index.lookup("tests.test_x.TestGroup::t_one")) == 1


def test_single_testsuite_root_is_accepted():
    index = parse_junit('<testsuite name="pytest" tests="1"><testcase name="a"/></testsuite>')
    assert len(index.cases) == 1


def test_unexpected_root_is_rejected():
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_junit("<report/>")
    assert "<report>" in str(excinfo.value)


def test_malformed_xml_is_rejected():
    with pytest.raises(AnalysisFormatError):
        parse_junit("<testsuite>")


def test_unreadable_junit_file(tmp_path):
    with pytest.raises(AnalysisFormatError):
        load_junit(tmp_path / "absent.xml")


def test_shipped_p024_junit_fixture(fixtures_dir):
    # Produced by a real pytest run of P024 DetumbleSim's suite, normalised by
    # fixtures/regenerate_fixtures.py. 305 tests, all passed.
    index = load_junit(fixtures_dir / "p024_bdot" / "junit.xml")
    assert len(index.cases) == 305
    assert index.counts[TestOutcome.PASSED] == 305
    assert sum(index.counts.values()) == 305
    assert (
        len(index.lookup("tests/test_simulate.py::TestSimulation::test_first_step_commands_no_dipole"))
        == 1
    )


def test_shipped_planted_junit_fixture(fixtures_dir):
    index = load_junit(fixtures_dir / "planted" / "junit.xml")
    assert len(index.cases) == 5
    assert index.counts[TestOutcome.PASSED] == 1
    assert index.counts[TestOutcome.FAILED] == 1
    assert index.counts[TestOutcome.ERRORED] == 1
    assert index.counts[TestOutcome.SKIPPED] == 1
    assert index.counts[TestOutcome.XFAILED] == 1


def test_empty_report_has_no_cases():
    index = parse_junit('<testsuite name="pytest" tests="0"/>')
    assert index.cases == ()
    assert sum(index.counts.values()) == 0
