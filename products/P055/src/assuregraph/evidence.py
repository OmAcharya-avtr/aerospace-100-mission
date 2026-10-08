"""Evidence artifact presence and content-hash freshness.

What this module does: resolve the path a Solution node cites, hash the bytes
found there with SHA-256 (FIPS 180-4, NIST, August 2015), and compare that
digest with the digest the case author recorded when the claim was written.

**What a digest mismatch means, exactly.** The bytes of the artifact are not the
bytes that were there when the claim cited it. That is all. It does not mean the
evidence has become invalid, that the change was material, or that the claim no
longer holds; a reformatted timestamp line and a reversed test verdict produce
the same mismatch. The check is a change detector for a human to adjudicate, not
an adjudication. A case whose artifacts all match is not thereby a case whose
evidence is adequate.

**What a missing recorded digest means.** Staleness cannot be computed. The
artifact is reported as unverifiable, which is a third state distinct from fresh
and from stale, and is counted separately everywhere in this package. Reporting
it as fresh would be the single most misleading thing this tool could do.

Units: digests are 64-character lowercase hexadecimal strings; sizes are bytes.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import Enum

from .model import AssuranceCase, Node, NodeKind

__all__ = [
    "EvidenceReport",
    "EvidenceStatus",
    "inspect_case_evidence",
    "inspect_evidence",
    "resolve_evidence_path",
    "sha256_file",
]

_CHUNK_BYTES = 1 << 20


class EvidenceStatus(Enum):
    """Outcome of inspecting one cited artifact."""

    #: The file exists and its SHA-256 equals the recorded digest.
    FRESH = "fresh"
    #: The file exists and its SHA-256 differs from the recorded digest. The
    #: bytes changed after the claim cited them; whether that matters is not
    #: something this package can determine.
    STALE = "stale"
    #: The file exists but the case recorded no digest, so freshness is not
    #: computable. Not the same as fresh.
    UNVERIFIABLE = "unverifiable"
    #: No file at the cited path.
    ABSENT = "absent"
    #: Something exists at the cited path but it is not a regular file, or it
    #: could not be read.
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class EvidenceReport:
    """Result of inspecting the artifact cited by one Solution node.

    Attributes:
        node_id: The Solution node that cites the artifact.
        cited_path: The path exactly as written in the case document.
        resolved_path: ``cited_path`` resolved against the case's ``base_dir``.
        status: See :class:`EvidenceStatus`.
        recorded_sha256: The digest the case recorded, or ``None``.
        actual_sha256: The digest computed from disk, or ``None`` when the file
            was absent or unreadable.
        size_bytes: Size of the file on disk in bytes, or ``None``.
        detail: One sentence naming what was wrong, empty when nothing was.
    """

    node_id: str
    cited_path: str
    resolved_path: str
    status: EvidenceStatus
    recorded_sha256: str | None
    actual_sha256: str | None
    size_bytes: int | None
    detail: str = ""


def sha256_file(path: str | os.PathLike[str], *, chunk_bytes: int = _CHUNK_BYTES) -> str:
    """Return the SHA-256 hex digest of the file at ``path``.

    Read in ``chunk_bytes`` blocks so an artifact larger than memory still
    hashes. Raises ``OSError`` if the file cannot be read.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def resolve_evidence_path(case: AssuranceCase, cited_path: str) -> str:
    """Resolve ``cited_path`` against the case's ``base_dir``.

    An absolute cited path is returned normalised and unchanged in meaning.
    """
    if os.path.isabs(cited_path):
        return os.path.normpath(cited_path)
    return os.path.normpath(os.path.join(case.base_dir, cited_path))


def inspect_evidence(case: AssuranceCase, node: Node) -> EvidenceReport:
    """Inspect the artifact cited by one Solution ``node``.

    Raises:
        ValueError: ``node`` is not a Solution, or carries no evidence. A parsed
            case cannot contain such a node; this guards direct construction.
    """
    if node.kind is not NodeKind.SOLUTION or node.evidence is None:
        raise ValueError(
            f"node {node.node_id!r} is a {node.kind.value} with "
            f"{'no' if node.evidence is None else 'an'} evidence mapping; "
            "only a Solution citing an artifact can be inspected"
        )
    evidence = node.evidence
    resolved = resolve_evidence_path(case, evidence.path)

    if not os.path.exists(resolved):
        return EvidenceReport(
            node_id=node.node_id,
            cited_path=evidence.path,
            resolved_path=resolved,
            status=EvidenceStatus.ABSENT,
            recorded_sha256=evidence.sha256,
            actual_sha256=None,
            size_bytes=None,
            detail=f"no file at {resolved!r}",
        )
    if not os.path.isfile(resolved):
        return EvidenceReport(
            node_id=node.node_id,
            cited_path=evidence.path,
            resolved_path=resolved,
            status=EvidenceStatus.UNREADABLE,
            recorded_sha256=evidence.sha256,
            actual_sha256=None,
            size_bytes=None,
            detail=f"{resolved!r} exists but is not a regular file",
        )
    try:
        actual = sha256_file(resolved)
        size = os.path.getsize(resolved)
    except OSError as exc:
        return EvidenceReport(
            node_id=node.node_id,
            cited_path=evidence.path,
            resolved_path=resolved,
            status=EvidenceStatus.UNREADABLE,
            recorded_sha256=evidence.sha256,
            actual_sha256=None,
            size_bytes=None,
            detail=f"{resolved!r} could not be read: {exc}",
        )

    if evidence.sha256 is None:
        return EvidenceReport(
            node_id=node.node_id,
            cited_path=evidence.path,
            resolved_path=resolved,
            status=EvidenceStatus.UNVERIFIABLE,
            recorded_sha256=None,
            actual_sha256=actual,
            size_bytes=size,
            detail=(
                "the case records no sha256 for this artifact, so staleness is not "
                f"computable; the current digest is {actual}"
            ),
        )
    if actual != evidence.sha256:
        return EvidenceReport(
            node_id=node.node_id,
            cited_path=evidence.path,
            resolved_path=resolved,
            status=EvidenceStatus.STALE,
            recorded_sha256=evidence.sha256,
            actual_sha256=actual,
            size_bytes=size,
            detail=(
                f"content hash changed since the claim cited it: recorded "
                f"{evidence.sha256}, found {actual}. The bytes differ; whether the "
                "difference matters is a judgement this tool does not make"
            ),
        )
    return EvidenceReport(
        node_id=node.node_id,
        cited_path=evidence.path,
        resolved_path=resolved,
        status=EvidenceStatus.FRESH,
        recorded_sha256=evidence.sha256,
        actual_sha256=actual,
        size_bytes=size,
    )


def inspect_case_evidence(case: AssuranceCase) -> list[EvidenceReport]:
    """Inspect every Solution node's artifact, in document order."""
    return [inspect_evidence(case, node) for node in case.nodes_of_kind(NodeKind.SOLUTION)]
