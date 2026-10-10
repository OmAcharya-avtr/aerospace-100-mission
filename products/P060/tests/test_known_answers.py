"""Known answers on the bundled fixtures, hand-counted before the code was run.

The sample project was constructed so that every finding code has exactly one
input that produces it. The numbers below were derived by reading
`fixtures/sample_project/docs/REQUIREMENTS.md` and
`fixtures/sample_project/tests/test_monitor.py`:

Declared ids, counted by hand: REQ-001 .. REQ-008, plus a second declaration of
REQ-007 in the appendix. Nine declarations, eight unique ids.

Claims, by test, read off the markers:

    test_rejects_out_of_range_speed      REQ-001   passed
    test_accumulates_momentum            REQ-002   passed
    test_desaturation_threshold          REQ-003   skipped
    test_hysteresis_band                 REQ-004   xfailed
    test_request_counter                 REQ-006   passed, no assertion in body
    test_reset_clears_momentum           REQ-007   passed
    test_saturation_ceiling              REQ-008   failed
    test_claims_an_undeclared_requirement REQ-042  passed

REQ-005 is claimed by nothing.

Coverage, hand arithmetic, denominator 8:

    nominal  : 001 002 003 004 006 007 008            = 7/8 = 87.5 %
    executed : 001 002 006 007 008                    = 5/8 = 62.5 %
    passing  : 001 002 006 007                        = 4/8 = 50.0 %

Findings, hand enumeration, total 7:

    TA001 x1  REQ-005
    TA002 x1  REQ-042
    TA003 x1  REQ-007
    TA004 x2  REQ-003 (skipped), REQ-004 (xfailed)
    TA005 x1  REQ-008
    TA006 x1  test_request_counter
"""

from __future__ import annotations

import pytest

from traceaudit import audit, parse_collection_report, parse_junit_xml, parse_requirements_file


@pytest.fixture
def sample_result(sample_project):
    doc = parse_requirements_file(sample_project / "docs" / "REQUIREMENTS.md",
                                  relative_to=sample_project)
    report = parse_junit_xml(sample_project / "junit.xml", relative_to=sample_project)
    return audit(doc, report, test_root=sample_project / "tests")


@pytest.mark.verifies("REQ-011")
def test_hand_computed_coverage_numerators_and_denominators(sample_result):
    figures = {f.label: (f.numerator, f.denominator) for f in sample_result.coverage}
    assert figures["nominal coverage"] == (7, 8)
    assert figures["executed coverage"] == (5, 8)
    assert figures["passing coverage"] == (4, 8)


@pytest.mark.verifies("REQ-011")
def test_hand_computed_coverage_percentages(sample_result):
    percents = {f.label: f.percent for f in sample_result.coverage}
    assert percents["nominal coverage"] == pytest.approx(87.5)
    assert percents["executed coverage"] == pytest.approx(62.5)
    assert percents["passing coverage"] == pytest.approx(50.0)


@pytest.mark.verifies("REQ-013")
def test_hand_enumerated_finding_counts(sample_result):
    assert sample_result.counts_by_code() == {
        "TA001": 1, "TA002": 1, "TA003": 1, "TA004": 2, "TA005": 1, "TA006": 1
    }
    assert len(sample_result.findings) == 7
    assert sample_result.exit_code == 1


@pytest.mark.verifies("REQ-013")
def test_hand_enumerated_finding_subjects(sample_result):
    subjects = [(f.code, f.subject) for f in sample_result.findings]
    assert subjects == [
        ("TA001", "REQ-005"),
        ("TA002", "REQ-042"),
        ("TA003", "REQ-007"),
        ("TA004", "REQ-003"),
        ("TA004", "REQ-004"),
        ("TA005", "REQ-008"),
        ("TA006", "tests/test_monitor.py::test_request_counter"),
    ]


@pytest.mark.verifies("REQ-010")
def test_hand_read_mapping_of_every_requirement(sample_result):
    mapping = sample_result.matrix.requirement_to_tests
    assert mapping["REQ-001"] == ("tests/test_monitor.py::test_rejects_out_of_range_speed",)
    assert mapping["REQ-005"] == ()
    assert mapping["REQ-007"] == ("tests/test_monitor.py::test_reset_clears_momentum",)
    assert sample_result.matrix.unknown_claims == {
        "REQ-042": ("tests/test_monitor.py::test_claims_an_undeclared_requirement",)
    }


@pytest.mark.verifies("REQ-005")
def test_hand_counted_outcomes_of_the_fixture_report(sample_result):
    assert sample_result.report.outcome_counts == {
        "passed": 5, "failed": 1, "errored": 0,
        "skipped": 1, "xfailed": 1, "unknown": 0,
    }


@pytest.mark.verifies("REQ-008")
def test_the_same_fixture_through_the_collection_report_loses_the_outcomes(sample_project):
    # Same eight tests, no outcomes: nominal coverage is unchanged at 7/8, and
    # the two outcome-dependent findings disappear, leaving 3 of the 7.
    doc = parse_requirements_file(sample_project / "docs" / "REQUIREMENTS.md",
                                  relative_to=sample_project)
    report = parse_collection_report(sample_project / "collect_only.txt")
    result = audit(doc, report, test_root=sample_project / "tests")
    assert len(report.cases) == 8
    assert [f.label for f in result.coverage] == ["nominal coverage"]
    assert result.coverage[0].numerator == 0
    assert result.counts_by_code()["TA004"] == 0
    assert result.counts_by_code()["TA005"] == 0


@pytest.mark.verifies("REQ-007")
def test_a_collection_report_carries_no_property_claims(sample_project):
    # The collection report has no properties at all, so with node-id claiming
    # alone nothing is claimed and every requirement reads as untraced. This is
    # the honest cost of using a collection report and is why junit is preferred.
    doc = parse_requirements_file(sample_project / "docs" / "REQUIREMENTS.md")
    report = parse_collection_report(sample_project / "collect_only.txt")
    assert report.claimed_ids == ()
    result = audit(doc, report, test_root=sample_project / "tests")
    assert result.counts_by_code()["TA001"] == 8
