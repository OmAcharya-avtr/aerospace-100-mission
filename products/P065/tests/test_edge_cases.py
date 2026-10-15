"""Edge cases and input validation at the boundaries."""

from __future__ import annotations

import pytest

from hazardtrace import (
    Analysis,
    Ratio,
    Severity,
    format_report,
    parse_analysis,
    parse_junit,
    parse_requirements,
    run_checks,
)


def test_a_document_with_only_a_title_is_valid_and_clean():
    report = run_checks(parse_analysis("meta:\n  title: nothing else\n"))
    assert report.findings == ()
    assert report.exit_code == 0
    # Every ratio is undefined, because every denominator is zero.
    for _label, ratio in report.coverage.as_items():
        assert ratio.defined is False


def test_report_renders_for_an_empty_analysis():
    assert "nothing else" in format_report(
        run_checks(parse_analysis("meta:\n  title: nothing else\n"))
    )


def test_ratio_with_a_zero_denominator_is_undefined_not_zero():
    ratio = Ratio(0, 0, "nothing")
    assert ratio.defined is False
    assert ratio.fraction != ratio.fraction  # NaN
    assert ratio.percent != ratio.percent
    assert str(ratio) == "0/0 = undefined"


def test_ratio_formats_with_four_decimals():
    assert str(Ratio(1, 3, "d")) == "1/3 = 33.3333 %"
    assert str(Ratio(3, 3, "d")) == "3/3 = 100.0000 %"


def test_ratio_fraction_is_within_the_unit_interval():
    ratio = Ratio(7, 8, "d")
    assert ratio.fraction == pytest.approx(0.875, abs=1e-12)
    assert 0.0 <= ratio.fraction <= 1.0


def test_requirements_document_with_no_declarations():
    index = parse_requirements("# Nothing here\n\nJust prose.\n")
    assert index.unique == frozenset()
    assert index.declared == ()


def test_unclosed_code_fence_hides_everything_after_it():
    # A known limitation, shared with any line-oriented markdown parser: an
    # unclosed fence swallows the rest of the document, which makes coverage
    # look WORSE here rather than better, because declarations go missing.
    index = parse_requirements("```\n### REQ-001 hidden\n")
    assert index.unique == frozenset()


def test_junit_with_no_testsuite_element():
    index = parse_junit("<testsuites/>")
    assert index.cases == ()


def test_testcase_without_a_classname():
    index = parse_junit('<testsuite name="p" tests="1"><testcase name="solo"/></testsuite>')
    assert len(index.lookup("solo")) == 1
    assert index.cases[0].classname == ""


def test_two_cases_with_the_same_identifier_must_both_pass():
    xml = (
        '<testsuite name="p" tests="2">'
        '<testcase classname="tests.test_a" name="dup"/>'
        '<testcase classname="tests.test_a" name="dup"><failure message="m"/></testcase>'
        "</testsuite>"
    )
    index = parse_junit(xml, source="junit.xml")
    assert len(index.lookup("tests/test_a.py::dup")) == 2
    case = (
        "meta:\n  title: t\nmitigations:\n  - id: M\n    description: d\n"
        "    evidence: [E]\nevidence:\n  - id: E\n    kind: test\n    description: d\n"
        "    tests: ['tests/test_a.py::dup']\n"
    )
    report = run_checks(parse_analysis(case), tests=index)
    assert [f.code for f in report.findings] == ["HT014"]


def test_analysis_with_control_actions_but_no_controller_feedback():
    text = (
        "meta:\n  title: t\ncontrol_structure:\n  controllers:\n    - id: C\n"
        "      name: c\n  actuators:\n    - id: A\n      name: a\n"
        "  controlled_processes:\n    - id: P\n      name: p\n"
    )
    report = run_checks(parse_analysis(text))
    # Hand count: no control action, so no enumeration cells; C has no feedback
    # (HT015) and all three components are referenced by nothing (3 x HT016).
    assert [f.code for f in report.findings] == ["HT015", "HT016", "HT016", "HT016"]
    assert report.grid.total == 0


def test_blocking_severities_default_to_error_only():
    report = run_checks(Analysis(title="t"))
    assert report.blocking_severities == frozenset({Severity.ERROR})


def test_very_long_identifiers_are_handled():
    long_id = "L-" + "9" * 200
    text = f"meta:\n  title: t\nlosses:\n  - id: {long_id}\n    description: d\n"
    report = run_checks(parse_analysis(text))
    assert report.findings[0].subjects == (long_id,)


def test_unicode_in_descriptions_survives_rendering():
    text = "meta:\n  title: t\nlosses:\n  - id: L-1\n    description: ± 0.1 deg/s\n"
    report = run_checks(parse_analysis(text))
    assert "±" in report.analysis.losses[0].description
    assert format_report(report)
