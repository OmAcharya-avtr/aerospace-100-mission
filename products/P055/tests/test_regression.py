"""Regression tests for defects found and fixed during the build.

Each test names the defect, what it broke and the date it was found. They are
kept separate from the behavioural suite so that removing one is a visible act.
"""

from __future__ import annotations

from conftest import COMPLETE_CASE, document, edge, node

from assuregraph import load_case, parse_case, render_mermaid, run_checks


def test_regression_mermaid_label_keeps_explicit_newlines() -> None:
    """Found 2026-10-08.

    ``_label`` normalised the statement with ``" ".join(statement.split())``,
    which collapsed ``\\n`` along with every other whitespace run. The
    ``<br/>`` substitution documented in :mod:`assuregraph.mermaid` was
    therefore dead code and a multi-line GSN statement rendered as one long
    line. Fixed by normalising each line separately.
    """
    case = parse_case(document([node("G1", "goal", "first line\nsecond line")]))
    assert "first line<br/>second line" in render_mermaid(case)


def test_regression_mermaid_label_still_collapses_folded_yaml_runs() -> None:
    """Companion to the above: the fix must not reintroduce ragged labels.

    A YAML folded block scalar arrives with runs of spaces; those are still
    collapsed, only newlines survive.
    """
    case = parse_case(document([node("G1", "goal", "spaced     out\ttabs")]))
    assert "spaced out tabs" in render_mermaid(case)


def test_regression_evidence_is_hashed_once_per_run(tmp_path, artifact, monkeypatch) -> None:
    """Found 2026-10-08.

    The first draft of ``run_checks`` let ``check_missing_evidence`` and
    ``check_stale_evidence`` each call ``inspect_case_evidence``, so every
    artifact was read from disk twice per run. Fixed by hashing once in
    ``run_checks`` and passing the reports into both checks.
    """
    import assuregraph.evidence as evidence_module

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
    reads = []
    original = evidence_module.sha256_file
    monkeypatch.setattr(
        evidence_module,
        "sha256_file",
        lambda path, **kw: (reads.append(path), original(path, **kw))[1],
    )
    run_checks(case)
    assert len(reads) == 1


def test_regression_unverifiable_is_never_counted_as_fresh(tmp_path, artifact) -> None:
    """Guard against the most misleading possible regression.

    An artifact with no recorded digest must never increment
    ``evidence_fresh`` nor enter the freshness denominator. A tool that
    reported "100 % fresh" for a case that recorded no digests at all would be
    worse than no tool.
    """
    artifact("a.txt")
    doc = document(
        [node("G1", "goal"), node("Sn1", "solution", evidence={"path": "a.txt"})],
        [edge("G1", "Sn1")],
        top_goals=["G1"],
    )
    cov = run_checks(parse_case(doc, base_dir=str(tmp_path))).coverage
    assert cov.evidence_fresh == 0
    assert cov.evidence_unverifiable == 1
    assert cov.evidence_fresh_fraction is None


def test_regression_shipped_complete_case_stays_at_zero_findings() -> None:
    """Guard: the committed digests in ``complete_case.yaml`` match the
    committed artifacts. Editing an artifact without updating the case makes
    this fail, which is exactly the failure mode the product exists to catch.
    """
    assert run_checks(load_case(COMPLETE_CASE)).findings == ()


def test_regression_deep_chain_does_not_recurse(make_case) -> None:
    """Guard: Tarjan and the cycle walk are iterative.

    A recursive formulation raised ``RecursionError`` on a 1500-deep chain,
    which is a plausible depth for a modular case assembled from many modules.
    """
    doc = document(
        [node(f"G{i}", "goal") for i in range(1, 1501)],
        [edge(f"G{i}", f"G{i + 1}") for i in range(1, 1500)],
        top_goals=["G1"],
    )
    report = run_checks(make_case(doc))
    assert report.by_check("cycles") == ()
    assert report.by_check("orphan_nodes") == ()
