"""Known-answer tests over the shipped hand-constructed cases.

`examples/cases/planted_gaps.yaml` was built to contain exactly one instance of
every check. The expected answer is therefore known by construction and is
written out in full below rather than derived from a run.
"""

from __future__ import annotations

import pytest

from hazardtrace import (
    Severity,
    load_analysis,
    load_junit,
    load_requirements,
    run_checks,
)
from hazardtrace.checks import CHECKS

# Hand-constructed expected answer for planted_gaps.yaml: code -> subject tuple.
PLANTED_EXPECTED: dict[str, tuple[str, ...]] = {
    "HT001": ("CA-1", "stopped_too_soon_or_applied_too_long"),
    "HT002": ("CA-1", "wrong_timing_or_order"),
    "HT003": ("CA-2", "not_provided"),
    "HT004": ("H-2",),
    "HT005": ("SC-2",),
    "HT006": ("UCA-3",),
    "HT007": ("LS-2",),
    "HT008": ("M-2",),
    "HT009": ("H-3",),
    "HT010": ("UCA-2",),
    "HT011": ("L-2",),
    "HT012": ("REQ-299",),
    "HT013": ("tests/test_planted.py::test_absent",),
    "HT014": ("tests/test_planted.py::test_fails",),
    "HT015": ("CTRL-B",),
    "HT016": ("SNS-ORPHAN",),
}


def _planted_report(cases_dir, fixtures_dir):
    return run_checks(
        load_analysis(cases_dir / "planted_gaps.yaml"),
        requirements=load_requirements(
            fixtures_dir / "planted" / "docs" / "REQUIREMENTS.md"
        ),
        tests=load_junit(fixtures_dir / "planted" / "junit.xml"),
    )


def test_planted_case_raises_exactly_one_finding_per_check(cases_dir, fixtures_dir):
    report = _planted_report(cases_dir, fixtures_dir)
    assert len(report.findings) == len(CHECKS) == 16
    assert {f.code: f.subjects for f in report.findings} == PLANTED_EXPECTED


def test_planted_case_severity_split(cases_dir, fixtures_dir):
    # Hand count from CHECKS: 14 error codes, 2 warning codes (HT011, HT016).
    counts = _planted_report(cases_dir, fixtures_dir).count_by_severity()
    assert counts[Severity.ERROR] == 14
    assert counts[Severity.WARNING] == 2
    assert counts[Severity.INFO] == 0


def test_planted_case_exit_code_and_verdict(cases_dir, fixtures_dir):
    report = _planted_report(cases_dir, fixtures_dir)
    assert report.exit_code == 1
    assert report.is_complete is False
    assert len(report.blocking) == 14


def test_planted_case_enumeration_coverage(cases_dir, fixtures_dir):
    # Hand count: 2 control actions x 4 classes = 8 cells. CA-1 has a UCA in
    # not_provided and in provided, a short-but-present exclusion in
    # wrong_timing_or_order, and nothing in the fourth. CA-2 has all four
    # covered (one of them by a conflicting pair). 7 of 8 covered.
    coverage = _planted_report(cases_dir, fixtures_dir).coverage
    assert coverage.enumeration.numerator == 7
    assert coverage.enumeration.denominator == 8
    # A ratio is a float: compared with a tolerance, never for bit equality.
    assert coverage.enumeration.percent == pytest.approx(87.5, abs=1e-9)


def test_planted_case_cross_link_coverage(cases_dir, fixtures_dir):
    # Hand count: cited requirements are REQ-201 (declared) and REQ-299 (not):
    # 1 of 2. Cited tests are test_passes (passed), test_fails (failed) and
    # test_absent (missing): 2 of 3 resolve, 1 of 3 passes.
    coverage = _planted_report(cases_dir, fixtures_dir).coverage
    assert (coverage.requirement_resolution.numerator,
            coverage.requirement_resolution.denominator) == (1, 2)
    assert (coverage.test_resolution.numerator,
            coverage.test_resolution.denominator) == (2, 3)
    assert (coverage.test_passing.numerator,
            coverage.test_passing.denominator) == (1, 3)


def test_complete_minimal_case_is_clean(cases_dir):
    # Hand count: 1 control action, 4 UCAs one per class, every link closed.
    report = run_checks(load_analysis(cases_dir / "complete_minimal.yaml"))
    assert report.findings == ()
    assert report.exit_code == 0
    assert report.coverage.enumeration.numerator == 4
    assert report.coverage.enumeration.denominator == 4


def test_worked_example_finding_counts(cases_dir, fixtures_dir):
    # The P024 B-dot worked example. These eleven findings are the genuine gaps
    # in that analysis and are deliberately NOT fixed; see validation/VALIDATION.md.
    report = run_checks(
        load_analysis(cases_dir / "p024_bdot_detumble.yaml"),
        requirements=load_requirements(
            fixtures_dir / "p024_bdot" / "docs" / "REQUIREMENTS.md"
        ),
        tests=load_junit(fixtures_dir / "p024_bdot" / "junit.xml"),
    )
    assert report.count_by_code() == {
        "HT001": 2,
        "HT002": 0,
        "HT003": 0,
        "HT004": 1,
        "HT005": 1,
        "HT006": 1,
        "HT007": 1,
        "HT008": 1,
        "HT009": 0,
        "HT010": 0,
        "HT011": 0,
        "HT012": 2,
        "HT013": 1,
        "HT014": 0,
        "HT015": 1,
        "HT016": 0,
    }
    assert len(report.findings) == 11
    assert report.exit_code == 1
    assert report.coverage.enumeration.numerator == 6
    assert report.coverage.enumeration.denominator == 8


def test_worked_example_gap_cells_are_the_two_declared_ones(cases_dir):
    report = run_checks(load_analysis(cases_dir / "p024_bdot_detumble.yaml"))
    assert [(c.control_action, c.uca_class.value) for c in report.grid.gaps] == [
        ("CA-DIPOLE", "stopped_too_soon_or_applied_too_long"),
        ("CA-EXIT", "wrong_timing_or_order"),
    ]
