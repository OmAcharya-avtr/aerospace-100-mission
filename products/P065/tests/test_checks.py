"""One test per check, on the smallest document that triggers it.

Each test states the hand-computed expected finding set in a comment. The
pattern is: build a document whose only defect is the one under test, assert
that exactly one finding is raised and that it carries the expected code and
subject.
"""

from __future__ import annotations

import pytest

from hazardtrace import (
    MIN_JUSTIFICATION_CHARS,
    CheckSpec,
    Severity,
    check_spec,
    parse_analysis,
    parse_junit,
    parse_requirements,
    run_checks,
)
from hazardtrace.checks import CHECKS

# A document with one control action, all four cells excluded with adequate
# justifications, and nothing else declared. Hand count: 0 findings.
CLEAN_ACTION = """
meta:
  title: t
control_structure:
  controllers:
    - id: CTRL-1
      name: c
  actuators:
    - id: ACT-1
      name: a
  sensors:
    - id: SNS-1
      name: s
  controlled_processes:
    - id: PROC-1
      name: p
  control_actions:
    - id: CA-1
      name: one
      controller: CTRL-1
      actuator: ACT-1
      process: PROC-1
  feedback:
    - id: FB-1
      name: f
      sensor: SNS-1
      controller: CTRL-1
      process: PROC-1
exclusions:
  - control_action: CA-1
    uca_class: not_provided
    justification: a justification long enough to pass the policy
  - control_action: CA-1
    uca_class: provided
    justification: a justification long enough to pass the policy
  - control_action: CA-1
    uca_class: wrong_timing_or_order
    justification: a justification long enough to pass the policy
  - control_action: CA-1
    uca_class: stopped_too_soon_or_applied_too_long
    justification: a justification long enough to pass the policy
"""


def codes(report) -> list[str]:
    """Finding codes in report order."""
    return [f.code for f in report.findings]


def test_clean_document_raises_nothing():
    # Hand count: 1 control action x 4 classes = 4 cells, all excluded, so
    # enumeration is 4/4 and no other section is declared. 0 findings, exit 0.
    report = run_checks(parse_analysis(CLEAN_ACTION))
    assert report.findings == ()
    assert report.exit_code == 0
    assert report.is_complete is True
    assert report.coverage.enumeration.numerator == 4
    assert report.coverage.enumeration.denominator == 4


def test_ht001_enumeration_gap():
    # Remove the last exclusion: 3 of 4 cells covered, 1 gap, 1 HT001.
    text = CLEAN_ACTION.split("  - control_action: CA-1\n    uca_class: stopped")[0]
    report = run_checks(parse_analysis(text))
    assert codes(report) == ["HT001"]
    assert report.findings[0].subjects == (
        "CA-1",
        "stopped_too_soon_or_applied_too_long",
    )
    assert report.coverage.enumeration.numerator == 3


def test_ht002_short_justification():
    # Replace one justification with "n/a": 3 characters, below the policy.
    text = CLEAN_ACTION.replace(
        "    uca_class: provided\n    justification: a justification long enough to pass the policy",
        "    uca_class: provided\n    justification: n/a",
        1,
    )
    report = run_checks(parse_analysis(text))
    assert codes(report) == ["HT002"]
    assert str(MIN_JUSTIFICATION_CHARS) in report.findings[0].message


def test_justification_exactly_at_the_policy_length_is_accepted():
    # Boundary: the policy is >= MIN_JUSTIFICATION_CHARS, so a justification of
    # exactly that many characters passes.
    exact = "x" * MIN_JUSTIFICATION_CHARS
    text = CLEAN_ACTION.replace(
        "    uca_class: provided\n    justification: a justification long enough to pass the policy",
        f"    uca_class: provided\n    justification: {exact}",
        1,
    )
    assert run_checks(parse_analysis(text)).findings == ()


def test_justification_one_character_short_is_rejected():
    short = "x" * (MIN_JUSTIFICATION_CHARS - 1)
    text = CLEAN_ACTION.replace(
        "    uca_class: provided\n    justification: a justification long enough to pass the policy",
        f"    uca_class: provided\n    justification: {short}",
        1,
    )
    assert codes(run_checks(parse_analysis(text))) == ["HT002"]


def test_ht003_conflict():
    # Add a UCA in the already-excluded 'provided' cell: 1 HT003, and the UCA
    # itself then has no scenario, so HT006 fires too. Hand count: 2 findings.
    text = CLEAN_ACTION + (
        "ucas:\n  - id: U-1\n    control_action: CA-1\n    uca_class: provided\n"
        "    description: d\n    hazards: []\n"
    )
    report = run_checks(parse_analysis(text))
    assert codes(report) == ["HT003", "HT006", "HT010"]


def test_ht004_hazard_with_no_constraint():
    text = CLEAN_ACTION + (
        "losses:\n  - id: L-1\n    description: d\n"
        "hazards:\n  - id: H-1\n    description: d\n    losses: [L-1]\n"
    )
    report = run_checks(parse_analysis(text))
    assert codes(report) == ["HT004"]
    assert report.findings[0].subjects == ("H-1",)


def test_ht005_constraint_with_no_requirement():
    text = CLEAN_ACTION + (
        "losses:\n  - id: L-1\n    description: d\n"
        "hazards:\n  - id: H-1\n    description: d\n    losses: [L-1]\n"
        "constraints:\n  - id: SC-1\n    description: d\n    hazards: [H-1]\n"
    )
    assert codes(run_checks(parse_analysis(text))) == ["HT005"]


def test_ht006_uca_with_no_scenario():
    text = CLEAN_ACTION.replace(
        "  - control_action: CA-1\n    uca_class: provided\n"
        "    justification: a justification long enough to pass the policy\n",
        "",
    ) + (
        "losses:\n  - id: L-1\n    description: d\n"
        "hazards:\n  - id: H-1\n    description: d\n    losses: [L-1]\n"
        "constraints:\n  - id: SC-1\n    description: d\n    hazards: [H-1]\n"
        "    requirements: [REQ-001]\n"
        "ucas:\n  - id: U-1\n    control_action: CA-1\n    uca_class: provided\n"
        "    description: d\n    hazards: [H-1]\n"
    )
    assert codes(run_checks(parse_analysis(text))) == ["HT006"]


def test_ht007_scenario_with_no_mitigation():
    text = CLEAN_ACTION + (
        "loss_scenarios:\n  - id: LS-1\n    description: d\n"
    )
    assert codes(run_checks(parse_analysis(text))) == ["HT007"]


def test_ht008_mitigation_with_no_evidence():
    text = CLEAN_ACTION + "mitigations:\n  - id: M-1\n    description: d\n"
    assert codes(run_checks(parse_analysis(text))) == ["HT008"]


def test_ht009_hazard_with_no_loss():
    text = CLEAN_ACTION + (
        "hazards:\n  - id: H-1\n    description: d\n"
        "constraints:\n  - id: SC-1\n    description: d\n    hazards: [H-1]\n"
        "    requirements: [REQ-001]\n"
    )
    assert codes(run_checks(parse_analysis(text))) == ["HT009"]


def test_ht010_uca_with_no_hazard():
    text = CLEAN_ACTION.replace(
        "  - control_action: CA-1\n    uca_class: provided\n"
        "    justification: a justification long enough to pass the policy\n",
        "",
    ) + (
        "ucas:\n  - id: U-1\n    control_action: CA-1\n    uca_class: provided\n"
        "    description: d\n"
        "loss_scenarios:\n  - id: LS-1\n    description: d\n    ucas: [U-1]\n"
        "    mitigations: [M-1]\n"
        "mitigations:\n  - id: M-1\n    description: d\n    evidence: [EV-1]\n"
        "evidence:\n  - id: EV-1\n    kind: review\n    description: d\n"
    )
    assert codes(run_checks(parse_analysis(text))) == ["HT010"]


def test_ht011_loss_with_no_hazard_is_a_warning():
    text = CLEAN_ACTION + "losses:\n  - id: L-1\n    description: d\n"
    report = run_checks(parse_analysis(text))
    assert codes(report) == ["HT011"]
    assert report.findings[0].severity is Severity.WARNING
    # A warning does not block: exit code stays 0.
    assert report.exit_code == 0
    assert report.is_complete is True


def test_ht012_unresolved_requirement():
    text = CLEAN_ACTION + (
        "losses:\n  - id: L-1\n    description: d\n"
        "hazards:\n  - id: H-1\n    description: d\n    losses: [L-1]\n"
        "constraints:\n  - id: SC-1\n    description: d\n    hazards: [H-1]\n"
        "    requirements: [REQ-999]\n"
    )
    index = parse_requirements("### REQ-001 declared\n", source="doc.md")
    report = run_checks(parse_analysis(text), requirements=index)
    assert codes(report) == ["HT012"]
    assert report.coverage.requirement_resolution.numerator == 0
    assert report.coverage.requirement_resolution.denominator == 1


def test_requirement_checks_are_skipped_when_no_document_is_supplied():
    text = CLEAN_ACTION + (
        "losses:\n  - id: L-1\n    description: d\n"
        "hazards:\n  - id: H-1\n    description: d\n    losses: [L-1]\n"
        "constraints:\n  - id: SC-1\n    description: d\n    hazards: [H-1]\n"
        "    requirements: [REQ-999]\n"
    )
    report = run_checks(parse_analysis(text))
    assert report.findings == ()
    # The ratio is undefined rather than 100 %, so nobody reads a missing check
    # as a passing one.
    assert report.coverage.requirement_resolution.defined is False


JUNIT = (
    '<testsuite name="pytest" tests="2"><testcase classname="tests.test_a" '
    'name="test_ok" file="tests/test_a.py"/><testcase classname="tests.test_a" '
    'name="test_bad" file="tests/test_a.py"><failure message="m"/></testcase>'
    "</testsuite>"
)


def _evidence_case(test_id: str) -> str:
    return CLEAN_ACTION + (
        "mitigations:\n  - id: M-1\n    description: d\n    evidence: [EV-1]\n"
        "evidence:\n  - id: EV-1\n    kind: test\n    description: d\n"
        f"    tests: ['{test_id}']\n"
    )


def test_ht013_test_not_in_the_report():
    report = run_checks(
        parse_analysis(_evidence_case("tests/test_a.py::test_absent")),
        tests=parse_junit(JUNIT, source="junit.xml"),
    )
    assert codes(report) == ["HT013"]
    assert report.coverage.test_resolution.numerator == 0


def test_ht014_cited_test_did_not_pass():
    report = run_checks(
        parse_analysis(_evidence_case("tests/test_a.py::test_bad")),
        tests=parse_junit(JUNIT, source="junit.xml"),
    )
    assert codes(report) == ["HT014"]
    assert "failed" in report.findings[0].message
    assert report.coverage.test_resolution.numerator == 1
    assert report.coverage.test_passing.numerator == 0


def test_a_cited_test_that_passed_raises_nothing():
    report = run_checks(
        parse_analysis(_evidence_case("tests/test_a.py::test_ok")),
        tests=parse_junit(JUNIT, source="junit.xml"),
    )
    assert report.findings == ()
    assert report.coverage.test_passing.numerator == 1


def test_ht015_controller_with_no_feedback():
    text = CLEAN_ACTION.replace(
        "  feedback:\n    - id: FB-1\n      name: f\n      sensor: SNS-1\n"
        "      controller: CTRL-1\n      process: PROC-1\n",
        "",
    )
    report = run_checks(parse_analysis(text))
    # Removing the only feedback path also orphans the sensor, so HT016 fires.
    assert codes(report) == ["HT015", "HT016"]


def test_ht016_orphan_component():
    text = CLEAN_ACTION.replace(
        "  controlled_processes:",
        "    - id: ACT-SPARE\n      name: unused\n  controlled_processes:",
    )
    report = run_checks(parse_analysis(text))
    assert codes(report) == ["HT016"]
    assert report.findings[0].subjects == ("ACT-SPARE",)
    assert report.findings[0].severity is Severity.WARNING


def test_findings_are_ordered_by_check_then_subject():
    text = CLEAN_ACTION.replace(
        "  controlled_processes:",
        "    - id: ACT-Z\n      name: unused z\n    - id: ACT-A\n      name: unused a\n"
        "  controlled_processes:",
    ) + "mitigations:\n  - id: M-1\n    description: d\n"
    report = run_checks(parse_analysis(text))
    assert codes(report) == ["HT008", "HT016", "HT016"]
    assert [f.subjects[0] for f in report.findings[1:]] == ["ACT-A", "ACT-Z"]


def test_every_declared_check_is_reachable_from_the_shipped_fixtures(cases_dir, fixtures_dir):
    from hazardtrace import load_analysis, load_junit, load_requirements

    report = run_checks(
        load_analysis(cases_dir / "planted_gaps.yaml"),
        requirements=load_requirements(fixtures_dir / "planted" / "docs" / "REQUIREMENTS.md"),
        tests=load_junit(fixtures_dir / "planted" / "junit.xml"),
    )
    assert sorted(set(codes(report))) == [spec.code for spec in CHECKS]


def test_check_spec_lookup():
    assert isinstance(check_spec("HT001"), CheckSpec)
    assert check_spec("HT011").severity is Severity.WARNING
    with pytest.raises(KeyError):
        check_spec("HT999")


def test_check_codes_are_unique_and_contiguous():
    numbers = [int(spec.code[2:]) for spec in CHECKS]
    assert numbers == list(range(1, len(CHECKS) + 1))


def test_report_counts_by_severity_sum_to_the_finding_count(cases_dir):
    from hazardtrace import load_analysis

    report = run_checks(load_analysis(cases_dir / "planted_gaps.yaml"))
    assert sum(report.count_by_severity().values()) == len(report.findings)
    assert sum(report.count_by_code().values()) == len(report.findings)


def test_finding_str_names_its_subjects():
    report = run_checks(parse_analysis(CLEAN_ACTION + "mitigations:\n  - id: M-1\n    description: d\n"))
    assert str(report.findings[0]).startswith("HT008 error:")
    assert "(M-1)" in str(report.findings[0])
