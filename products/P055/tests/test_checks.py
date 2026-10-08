"""Known-answer tests for the six checks.

Every test here states the expected finding set in a comment, worked out by hand
from the case built in the test body, before the assertion. A test whose comment
does not match its assertion is a defect in the test.
"""

from __future__ import annotations

from conftest import document, edge, node

from assuregraph import (
    Severity,
    check_cycles,
    check_missing_evidence,
    check_orphan_nodes,
    check_stale_evidence,
    check_undischarged_assumptions,
    check_unsupported_claims,
    parse_case,
    run_checks,
)

# ---------------------------------------------------------------- unsupported


def test_lone_goal_is_unsupported() -> None:
    # Case: G1 only. Expected: 1 error on G1.
    case = parse_case(document([node("G1", "goal")]))
    findings = check_unsupported_claims(case)
    assert [(f.severity, f.node_ids) for f in findings] == [(Severity.ERROR, ("G1",))]


def test_goal_with_a_solution_is_supported() -> None:
    # Case: G1 --supported_by--> Sn1. Expected: no findings.
    doc = document(
        [node("G1", "goal"), node("Sn1", "solution", evidence={"path": "a.txt"})],
        [edge("G1", "Sn1")],
    )
    assert check_unsupported_claims(parse_case(doc)) == []


def test_strategy_without_children_is_unsupported() -> None:
    # Case: G1 --supported_by--> S1, S1 has no children.
    # Expected: 1 error on S1 (G1 is supported by S1).
    doc = document([node("G1", "goal"), node("S1", "strategy")], [edge("G1", "S1")])
    findings = check_unsupported_claims(parse_case(doc))
    assert [f.node_ids for f in findings] == [("S1",)]
    assert findings[0].severity is Severity.ERROR


def test_undeveloped_goal_is_info_not_error() -> None:
    # Case: G1 undeveloped, nothing else. Expected: 1 info on G1, 0 errors.
    case = parse_case(document([node("G1", "goal", undeveloped=True)]))
    findings = check_unsupported_claims(case)
    assert len(findings) == 1
    assert findings[0].severity is Severity.INFO


def test_in_context_of_does_not_count_as_support() -> None:
    # Case: G1 --in_context_of--> C1. A Context is not support.
    # Expected: 1 error on G1.
    doc = document(
        [node("G1", "goal"), node("C1", "context")], [edge("G1", "C1", "in_context_of")]
    )
    findings = check_unsupported_claims(parse_case(doc))
    assert [f.node_ids for f in findings] == [("G1",)]


def test_context_and_assumption_are_never_unsupported() -> None:
    # Case: G1 --in_context_of--> C1, A1, J1, G1 --supported_by--> Sn1.
    # Expected: no unsupported findings: only Goals and Strategies can be.
    doc = document(
        [
            node("G1", "goal"),
            node("C1", "context"),
            node("A1", "assumption"),
            node("J1", "justification"),
            node("Sn1", "solution", evidence={"path": "a.txt"}),
        ],
        [
            edge("G1", "C1", "in_context_of"),
            edge("G1", "A1", "in_context_of"),
            edge("G1", "J1", "in_context_of"),
            edge("G1", "Sn1"),
        ],
    )
    assert check_unsupported_claims(parse_case(doc)) == []


# ------------------------------------------------------------------- evidence


def test_missing_evidence_is_reported(tmp_path) -> None:
    # Case: Sn1 cites nope.txt which is not on disk. Expected: 1 error on Sn1.
    doc = document([node("Sn1", "solution", evidence={"path": "nope.txt"})])
    case = parse_case(doc, base_dir=str(tmp_path))
    findings = check_missing_evidence(case)
    assert [(f.severity, f.node_ids) for f in findings] == [(Severity.ERROR, ("Sn1",))]


def test_present_evidence_is_not_reported_missing(tmp_path, artifact) -> None:
    artifact("a.txt")
    doc = document([node("Sn1", "solution", evidence={"path": "a.txt"})])
    assert check_missing_evidence(parse_case(doc, base_dir=str(tmp_path))) == []


def test_directory_evidence_is_reported_missing(tmp_path) -> None:
    # A directory at the cited path is an error, not a pass.
    (tmp_path / "adir").mkdir()
    doc = document([node("Sn1", "solution", evidence={"path": "adir"})])
    findings = check_missing_evidence(parse_case(doc, base_dir=str(tmp_path)))
    assert len(findings) == 1
    assert "regular file" in findings[0].message


def test_stale_evidence_is_an_error(tmp_path, artifact) -> None:
    # Case: digest recorded from "v1", file now holds "v2". Expected: 1 error.
    _, digest = artifact("a.txt", "v1\n")
    doc = document([node("Sn1", "solution", evidence={"path": "a.txt", "sha256": digest})])
    case = parse_case(doc, base_dir=str(tmp_path))
    (tmp_path / "a.txt").write_text("v2\n", encoding="utf-8")
    findings = check_stale_evidence(case)
    assert [(f.severity, f.node_ids) for f in findings] == [(Severity.ERROR, ("Sn1",))]


def test_unrecorded_digest_is_a_warning_not_a_pass(tmp_path, artifact) -> None:
    # Case: artifact present, no sha256 recorded. Expected: 1 warning on Sn1.
    artifact("a.txt")
    doc = document([node("Sn1", "solution", evidence={"path": "a.txt"})])
    findings = check_stale_evidence(parse_case(doc, base_dir=str(tmp_path)))
    assert [(f.severity, f.node_ids) for f in findings] == [(Severity.WARNING, ("Sn1",))]


def test_missing_file_produces_no_staleness_finding(tmp_path) -> None:
    # An absent artifact is a missing_evidence finding only; reporting it twice
    # would double-count the same defect.
    doc = document([node("Sn1", "solution", evidence={"path": "nope.txt", "sha256": "0" * 64})])
    assert check_stale_evidence(parse_case(doc, base_dir=str(tmp_path))) == []


# ----------------------------------------------------------------- assumptions


def test_undischarged_assumption_is_an_error() -> None:
    # Case: A1 with no discharged flag. Expected: 1 error on A1.
    case = parse_case(document([node("A1", "assumption")]))
    findings = check_undischarged_assumptions(case)
    assert [(f.severity, f.node_ids) for f in findings] == [(Severity.ERROR, ("A1",))]


def test_discharged_assumption_is_clean() -> None:
    doc = document(
        [node("G1", "goal"), node("A1", "assumption", discharged=True, discharged_by="G1")]
    )
    assert check_undischarged_assumptions(parse_case(doc)) == []


def test_discharged_without_a_reference_is_accepted() -> None:
    # discharged: true with no discharged_by is an author's bare assertion. The
    # tool accepts it and the README says why that is weak.
    case = parse_case(document([node("A1", "assumption", discharged=True)]))
    assert check_undischarged_assumptions(case) == []


def test_two_assumptions_give_two_findings() -> None:
    doc = document([node("A1", "assumption"), node("A2", "assumption")])
    assert [f.node_ids for f in check_undischarged_assumptions(parse_case(doc))] == [
        ("A1",),
        ("A2",),
    ]


# ---------------------------------------------------------------------- cycles


def test_cycle_finding_carries_the_closed_walk() -> None:
    # Case: G1 -> S1 -> G2 -> S2 -> G1. Expected: 1 error whose node_ids are a
    # closed walk of 5 entries over 4 distinct nodes.
    doc = document(
        [node("G1", "goal"), node("S1", "strategy"), node("G2", "goal"), node("S2", "strategy")],
        [edge("G1", "S1"), edge("S1", "G2"), edge("G2", "S2"), edge("S2", "G1")],
        top_goals=["G1"],
    )
    findings = check_cycles(parse_case(doc))
    assert len(findings) == 1
    assert findings[0].node_ids[0] == findings[0].node_ids[-1]
    assert len(set(findings[0].node_ids)) == 4


def test_acyclic_case_has_no_cycle_finding() -> None:
    doc = document([node("G1", "goal"), node("S1", "strategy")], [edge("G1", "S1")])
    assert check_cycles(parse_case(doc)) == []


# --------------------------------------------------------------------- orphans


def test_orphan_is_reported_against_a_declared_top_goal() -> None:
    # Case: top_goals [G1]; G9 and C9 hang off nothing. Expected: 2 errors.
    doc = document(
        [node("G1", "goal"), node("G9", "goal"), node("C9", "context")],
        [edge("G9", "C9", "in_context_of")],
        top_goals=["G1"],
    )
    findings = check_orphan_nodes(parse_case(doc))
    assert sorted(f.node_ids[0] for f in findings) == ["C9", "G9"]


def test_two_independent_top_goals_are_not_orphans() -> None:
    # Case: top_goals [G1, G2], both present. Expected: no orphan findings.
    doc = document([node("G1", "goal"), node("G2", "goal")], [], top_goals=["G1", "G2"])
    assert check_orphan_nodes(parse_case(doc)) == []


def test_undeclared_top_goals_are_inferred_so_nothing_is_an_orphan() -> None:
    # With no declaration, every Goal with no incoming SupportedBy is a root, so
    # a second disconnected Goal is a root rather than an orphan. This is a real
    # weakness of the check and is documented in the README.
    doc = document([node("G1", "goal"), node("G9", "goal")])
    assert check_orphan_nodes(parse_case(doc)) == []


def test_context_under_a_reached_strategy_is_not_an_orphan() -> None:
    doc = document(
        [node("G1", "goal"), node("S1", "strategy"), node("C1", "context")],
        [edge("G1", "S1"), edge("S1", "C1", "in_context_of")],
        top_goals=["G1"],
    )
    assert check_orphan_nodes(parse_case(doc)) == []


# ------------------------------------------------------------------ run_checks


def test_run_checks_on_a_clean_case_exits_zero(tmp_path, artifact) -> None:
    _, digest = artifact("a.txt")
    doc = document(
        [
            node("G1", "goal"),
            node("Sn1", "solution", evidence={"path": "a.txt", "sha256": digest}),
        ],
        [edge("G1", "Sn1")],
        top_goals=["G1"],
    )
    report = run_checks(parse_case(doc, base_dir=str(tmp_path)))
    assert report.findings == ()
    assert report.is_complete is True
    assert report.exit_code == 0


def test_run_checks_groups_findings_by_check_in_declared_order(tmp_path) -> None:
    doc = document(
        [
            node("G1", "goal"),
            node("Sn1", "solution", evidence={"path": "nope.txt"}),
            node("A1", "assumption"),
        ],
        [edge("G1", "Sn1"), edge("G1", "A1", "in_context_of")],
        top_goals=["G1"],
    )
    report = run_checks(parse_case(doc, base_dir=str(tmp_path)))
    checks = [f.check for f in report.findings]
    assert checks == ["missing_evidence", "undischarged_assumptions"]


def test_strict_promotes_warnings_to_blocking(tmp_path, artifact) -> None:
    artifact("a.txt")
    doc = document(
        [node("G1", "goal"), node("Sn1", "solution", evidence={"path": "a.txt"})],
        [edge("G1", "Sn1")],
        top_goals=["G1"],
    )
    case = parse_case(doc, base_dir=str(tmp_path))
    assert run_checks(case).exit_code == 0
    assert run_checks(case, strict=True).exit_code == 1


def test_coverage_counts_on_a_small_case(tmp_path, artifact) -> None:
    # Case: G1 -> Sn1 (fresh), G1 in_context_of A1 (undischarged).
    # Expected coverage: 1 goal, 1 solution, 1 assumption; claims 1/1 supported;
    # evidence 1/1 present, 1/1 fresh; assumptions 0/1 discharged.
    _, digest = artifact("a.txt")
    doc = document(
        [
            node("G1", "goal"),
            node("Sn1", "solution", evidence={"path": "a.txt", "sha256": digest}),
            node("A1", "assumption"),
        ],
        [edge("G1", "Sn1"), edge("G1", "A1", "in_context_of")],
        top_goals=["G1"],
    )
    cov = run_checks(parse_case(doc, base_dir=str(tmp_path))).coverage
    assert cov.node_counts["goal"] == 1
    assert cov.node_counts["solution"] == 1
    assert cov.claims_total == 1
    assert cov.claims_supported == 1
    assert cov.evidence_fresh == 1
    assert cov.assumptions_discharged == 0
    assert cov.claim_support_fraction == 1.0
    assert cov.assumption_discharged_fraction == 0.0


def test_undefined_fractions_are_none_not_zero_or_one() -> None:
    # A case with no Solutions and no Assumptions: freshness and discharge are
    # undefined, and reporting either as 100 % would be a lie.
    cov = run_checks(parse_case(document([node("G1", "goal", undeveloped=True)]))).coverage
    assert cov.evidence_present_fraction is None
    assert cov.evidence_fresh_fraction is None
    assert cov.assumption_discharged_fraction is None
    assert cov.claim_support_fraction == 1.0


def test_unverifiable_is_excluded_from_the_freshness_denominator(tmp_path, artifact) -> None:
    artifact("a.txt")
    doc = document(
        [node("G1", "goal"), node("Sn1", "solution", evidence={"path": "a.txt"})],
        [edge("G1", "Sn1")],
        top_goals=["G1"],
    )
    cov = run_checks(parse_case(doc, base_dir=str(tmp_path))).coverage
    assert cov.evidence_unverifiable == 1
    assert cov.evidence_fresh + cov.evidence_stale == 0
    assert cov.evidence_fresh_fraction is None


def test_report_accessors(tmp_path) -> None:
    doc = document([node("G1", "goal"), node("A1", "assumption")],
                   [edge("G1", "A1", "in_context_of")], top_goals=["G1"])
    report = run_checks(parse_case(doc, base_dir=str(tmp_path)))
    assert report.by_check("undischarged_assumptions")
    assert report.by_check("cycles") == ()
    assert report.error_count == 2  # G1 unsupported, A1 undischarged
    assert report.warning_count == 0
    assert report.info_count == 0
    assert report.by_severity(Severity.ERROR)


def test_evidence_is_hashed_once_per_run(tmp_path, artifact, monkeypatch) -> None:
    # run_checks shares one evidence pass between the two evidence checks, so a
    # case with a large artifact is read once. Regression guard: an earlier
    # draft called inspect_case_evidence twice.
    _, digest = artifact("a.txt")
    doc = document(
        [
            node("G1", "goal"),
            node("Sn1", "solution", evidence={"path": "a.txt", "sha256": digest}),
        ],
        [edge("G1", "Sn1")],
        top_goals=["G1"],
    )
    case = parse_case(doc, base_dir=str(tmp_path))
    calls = {"n": 0}
    import assuregraph.evidence as evidence_module

    original = evidence_module.sha256_file

    def counting(path, **kwargs):
        calls["n"] += 1
        return original(path, **kwargs)

    monkeypatch.setattr(evidence_module, "sha256_file", counting)
    run_checks(case)
    assert calls["n"] == 1
