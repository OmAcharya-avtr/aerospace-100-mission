"""Tests for evidence hashing and the five evidence states."""

from __future__ import annotations

import hashlib
import os

import pytest
from conftest import document, node

from assuregraph import (
    EvidenceStatus,
    NodeKind,
    inspect_case_evidence,
    inspect_evidence,
    parse_case,
    resolve_evidence_path,
    sha256_file,
)


def test_sha256_file_matches_hashlib(tmp_path) -> None:
    content = b"abc\ndef\n"
    path = tmp_path / "a.bin"
    path.write_bytes(content)
    assert sha256_file(str(path)) == hashlib.sha256(content).hexdigest()


def test_sha256_of_empty_file_is_the_known_constant(tmp_path) -> None:
    # SHA-256 of the empty string, FIPS 180-4. Hand-checkable constant.
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert (
        sha256_file(str(path))
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_sha256_is_chunk_size_independent(tmp_path) -> None:
    path = tmp_path / "big.bin"
    path.write_bytes(bytes(range(256)) * 400)
    assert sha256_file(str(path), chunk_bytes=7) == sha256_file(str(path), chunk_bytes=1 << 20)


def test_sha256_file_raises_on_missing_file(tmp_path) -> None:
    with pytest.raises(OSError):
        sha256_file(str(tmp_path / "nope.bin"))


def test_relative_path_resolves_against_base_dir() -> None:
    case = parse_case(document([node("G1", "goal")]), base_dir="/tmp/cases")
    assert resolve_evidence_path(case, "evidence/a.txt") == os.path.join(
        "/tmp/cases", "evidence", "a.txt"
    )


def test_absolute_path_is_left_absolute() -> None:
    case = parse_case(document([node("G1", "goal")]), base_dir="/tmp/cases")
    assert resolve_evidence_path(case, "/etc/hostname") == "/etc/hostname"


def _solution_case(tmp_path, path: str, digest: str | None):
    evidence: dict[str, object] = {"path": path}
    if digest is not None:
        evidence["sha256"] = digest
    doc = document([node("Sn1", "solution", evidence=evidence)])
    return parse_case(doc, base_dir=str(tmp_path))


def test_fresh_when_digest_matches(tmp_path, artifact) -> None:
    _, digest = artifact("a.txt")
    case = _solution_case(tmp_path, "a.txt", digest)
    report = inspect_evidence(case, case.nodes["Sn1"])
    assert report.status is EvidenceStatus.FRESH
    assert report.detail == ""
    assert report.size_bytes == len("evidence bytes\n")


def test_stale_when_digest_differs(tmp_path, artifact) -> None:
    _, digest = artifact("a.txt", "original\n")
    case = _solution_case(tmp_path, "a.txt", digest)
    (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
    report = inspect_evidence(case, case.nodes["Sn1"])
    assert report.status is EvidenceStatus.STALE
    assert report.recorded_sha256 == digest
    assert report.actual_sha256 != digest
    assert "whether the difference matters" in report.detail


def test_whitespace_only_change_is_also_stale(tmp_path, artifact) -> None:
    # The honest limit of a content hash: a trailing newline is the same finding
    # as an inverted verdict. This test exists to pin that behaviour, not to
    # defend it.
    _, digest = artifact("a.txt", "verdict: pass\n")
    case = _solution_case(tmp_path, "a.txt", digest)
    (tmp_path / "a.txt").write_text("verdict: pass\n\n", encoding="utf-8")
    assert inspect_evidence(case, case.nodes["Sn1"]).status is EvidenceStatus.STALE


def test_unverifiable_when_no_digest_recorded(tmp_path, artifact) -> None:
    artifact("a.txt")
    case = _solution_case(tmp_path, "a.txt", None)
    report = inspect_evidence(case, case.nodes["Sn1"])
    assert report.status is EvidenceStatus.UNVERIFIABLE
    assert report.recorded_sha256 is None
    assert report.actual_sha256 is not None


def test_absent_when_file_is_missing(tmp_path) -> None:
    case = _solution_case(tmp_path, "nope.txt", None)
    report = inspect_evidence(case, case.nodes["Sn1"])
    assert report.status is EvidenceStatus.ABSENT
    assert report.actual_sha256 is None
    assert report.size_bytes is None


def test_unreadable_when_path_is_a_directory(tmp_path) -> None:
    (tmp_path / "adir").mkdir()
    case = _solution_case(tmp_path, "adir", None)
    report = inspect_evidence(case, case.nodes["Sn1"])
    assert report.status is EvidenceStatus.UNREADABLE
    assert "not a regular file" in report.detail


def test_inspect_evidence_rejects_a_goal() -> None:
    case = parse_case(document([node("G1", "goal")]))
    with pytest.raises(ValueError, match="only a Solution citing an artifact"):
        inspect_evidence(case, case.nodes["G1"])


def test_inspect_case_evidence_is_in_document_order(tmp_path, artifact) -> None:
    artifact("a.txt")
    artifact("b.txt")
    doc = document(
        [
            node("Sn2", "solution", evidence={"path": "b.txt"}),
            node("Sn1", "solution", evidence={"path": "a.txt"}),
        ]
    )
    case = parse_case(doc, base_dir=str(tmp_path))
    assert [r.node_id for r in inspect_case_evidence(case)] == ["Sn2", "Sn1"]


def test_inspect_case_evidence_is_empty_without_solutions() -> None:
    case = parse_case(document([node("G1", "goal")]))
    assert inspect_case_evidence(case) == []
    assert case.nodes_of_kind(NodeKind.SOLUTION) == []
