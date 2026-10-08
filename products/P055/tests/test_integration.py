"""Integration tests: the shipped example cases end to end.

These run against ``examples/cases/*.yaml`` and the artifacts under
``examples/cases/evidence/``, which are the same files the README and
``validation/`` report numbers from. If a committed example drifts, these fail.
"""

from __future__ import annotations

import io
import json

from conftest import COMPLETE_CASE, CYCLIC_CASE, INCOMPLETE_CASE

from assuregraph import (
    EvidenceStatus,
    format_report,
    inspect_case_evidence,
    load_case,
    render_mermaid,
    report_as_dict,
    run_checks,
)
from assuregraph.__main__ import main


def test_complete_case_end_to_end() -> None:
    case = load_case(COMPLETE_CASE)
    report = run_checks(case)
    assert report.findings == ()
    assert report.exit_code == 0
    cov = report.coverage
    assert cov.node_counts == {
        "goal": 7,
        "strategy": 1,
        "solution": 6,
        "context": 2,
        "assumption": 1,
        "justification": 1,
    }
    assert cov.edge_counts == {"supported_by": 13, "in_context_of": 4}
    assert cov.claims_total == 8
    assert cov.claims_supported == 8
    assert cov.evidence_total == 6
    assert cov.evidence_fresh == 6
    assert cov.assumptions_discharged == 1
    assert cov.claim_support_fraction == 1.0
    assert cov.evidence_fresh_fraction == 1.0


def test_every_complete_case_artifact_is_fresh() -> None:
    reports = inspect_case_evidence(load_case(COMPLETE_CASE))
    assert len(reports) == 6
    assert {r.status for r in reports} == {EvidenceStatus.FRESH}


def test_incomplete_case_finding_set_is_exactly_as_expected() -> None:
    # Hand-worked expectation for examples/cases/incomplete_case.yaml:
    #   unsupported_claims        error   G5    (no SupportedBy child)
    #   unsupported_claims        info    G7    (declared undeveloped)
    #   unsupported_claims        info    G8    (declared undeveloped)
    #   missing_evidence          error   Sn3   (cites a path not on disk)
    #   stale_evidence            warning Sn1   (no recorded digest)
    #   stale_evidence            error   Sn2   (recorded digest does not match)
    #   undischarged_assumptions  error   A1    (discharged flag absent)
    #   orphan_nodes              error   G8    (unreachable from G1)
    #   orphan_nodes              error   C3    (unreachable from G1)
    # Totals: 6 error, 1 warning, 2 info.
    report = run_checks(load_case(INCOMPLETE_CASE))
    got = sorted((f.check, f.severity.value, f.node_ids) for f in report.findings)
    expected = sorted(
        [
            ("unsupported_claims", "error", ("G5",)),
            ("unsupported_claims", "info", ("G7",)),
            ("unsupported_claims", "info", ("G8",)),
            ("missing_evidence", "error", ("Sn3",)),
            ("stale_evidence", "warning", ("Sn1",)),
            ("stale_evidence", "error", ("Sn2",)),
            ("undischarged_assumptions", "error", ("A1",)),
            ("orphan_nodes", "error", ("G8",)),
            ("orphan_nodes", "error", ("C3",)),
        ]
    )
    assert got == expected
    assert (report.error_count, report.warning_count, report.info_count) == (6, 1, 2)
    assert report.exit_code == 1


def test_incomplete_case_coverage_numbers() -> None:
    cov = run_checks(load_case(INCOMPLETE_CASE)).coverage
    assert (cov.claims_supported, cov.claims_undeveloped, cov.claims_total) == (5, 2, 8)
    assert (cov.evidence_fresh, cov.evidence_stale, cov.evidence_unverifiable) == (0, 1, 1)
    assert cov.evidence_missing == 1
    assert cov.orphan_count == 2
    assert cov.evidence_fresh_fraction == 0.0
    assert cov.claim_support_fraction == 7 / 8


def test_cyclic_case_reports_one_region_over_four_nodes() -> None:
    report = run_checks(load_case(CYCLIC_CASE))
    cycles = report.by_check("cycles")
    assert len(cycles) == 1
    assert set(cycles[0].node_ids) == {"G1", "S1", "G2", "S2"}
    assert report.coverage.cycle_regions == 1
    assert report.exit_code == 1


def test_strict_mode_on_the_incomplete_case_keeps_the_same_exit_code() -> None:
    # It already fails; strict only changes the verdict for a case whose only
    # findings are warnings.
    assert run_checks(load_case(INCOMPLETE_CASE), strict=True).exit_code == 1


def test_formatted_report_mentions_every_offending_node() -> None:
    text = format_report(run_checks(load_case(INCOMPLETE_CASE)))
    for node_id in ("G5", "Sn3", "Sn2", "Sn1", "A1", "G8", "C3"):
        assert node_id in text


def test_mermaid_of_the_complete_case_declares_every_node() -> None:
    case = load_case(COMPLETE_CASE)
    text = render_mermaid(case, max_label_chars=48)
    for node_id in case.nodes:
        assert f"n_{node_id}" in text


def test_cli_json_round_trip_matches_the_library() -> None:
    out = io.StringIO()
    code = main(["check", INCOMPLETE_CASE, "--json"], out=out, err=io.StringIO())
    from_cli = json.loads(out.getvalue())
    from_lib = json.loads(json.dumps(report_as_dict(run_checks(load_case(INCOMPLETE_CASE)))))
    assert from_cli == from_lib
    assert code == from_cli["exit_code"] == 1


def test_three_shipped_cases_have_the_documented_exit_codes() -> None:
    assert run_checks(load_case(COMPLETE_CASE)).exit_code == 0
    assert run_checks(load_case(INCOMPLETE_CASE)).exit_code == 1
    assert run_checks(load_case(CYCLIC_CASE)).exit_code == 1
