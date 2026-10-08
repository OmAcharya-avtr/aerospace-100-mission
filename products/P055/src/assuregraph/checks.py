"""The computed checks. These are the product.

Six checks are defined, each named in the product scope and each reporting the
offending node ids:

===========================  ===========================================
``unsupported_claims``       a Goal or Strategy with no ``SupportedBy`` child
``missing_evidence``         a cited artifact absent from disk
``stale_evidence``           a cited artifact whose content hash changed
``undischarged_assumptions`` an Assumption not marked discharged
``cycles``                   a cyclic region of the argument graph
``orphan_nodes``             a node unreachable from any top Goal
===========================  ===========================================

**What these checks are not.** Every one of them is structural or
content-addressed. None of them reads the meaning of a statement, weighs whether
a Solution actually supports the Goal above it, or forms any judgement about an
argument. A case that passes all six is a case that is *completely drawn and
whose cited files are the files that were cited* -- nothing more. See the
package docstring for the scope statement in full.

Severity is three-valued. ``ERROR`` means the case is incomplete in a way the
author can close. ``WARNING`` means something is uncomputable or undeclared and
a reviewer should look. ``INFO`` records a declaration the author made on
purpose, such as the GSN *Undeveloped* decorator. Only ``ERROR`` makes a case
incomplete by default; :func:`run_checks` with ``strict=True`` promotes
``WARNING`` as well, and the CLI exposes that as ``--strict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .evidence import EvidenceReport, EvidenceStatus, inspect_case_evidence
from .graph import find_cycles, reachable_from
from .model import AssuranceCase, EdgeKind, NodeKind

__all__ = [
    "CHECK_NAMES",
    "CheckReport",
    "Coverage",
    "Finding",
    "Severity",
    "check_cycles",
    "check_missing_evidence",
    "check_orphan_nodes",
    "check_stale_evidence",
    "check_undischarged_assumptions",
    "check_unsupported_claims",
    "run_checks",
]

#: The six check names, in the order :func:`run_checks` runs them.
CHECK_NAMES: tuple[str, ...] = (
    "unsupported_claims",
    "missing_evidence",
    "stale_evidence",
    "undischarged_assumptions",
    "cycles",
    "orphan_nodes",
)

_SUPPORTABLE = frozenset({NodeKind.GOAL, NodeKind.STRATEGY})


class Severity(Enum):
    """How a finding bears on completeness."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Finding:
    """One check result about one or more specific nodes.

    Attributes:
        check: One of :data:`CHECK_NAMES`.
        severity: See :class:`Severity`.
        node_ids: The offending node ids. Never empty. For ``cycles`` this is
            the closed walk in traversal order, so the first and last entries
            are the same node.
        message: One sentence naming the defect and what would close it.
    """

    check: str
    severity: Severity
    node_ids: tuple[str, ...]
    message: str


def check_unsupported_claims(case: AssuranceCase) -> list[Finding]:
    """Goals and Strategies with no outgoing ``SupportedBy`` relationship.

    A Goal or Strategy carrying the GSN *Undeveloped* decorator is reported at
    ``INFO`` severity instead of ``ERROR``: in GSN Standard v3 the decorator is
    the author's declaration that the element is deliberately not elaborated,
    so treating it as a defect would punish correct use of the notation. The
    declaration is still reported, because a case that is 60 % undeveloped
    diamonds should not look clean.
    """
    has_support = {e.source for e in case.edges if e.kind is EdgeKind.SUPPORTED_BY}
    findings: list[Finding] = []
    for node in case.nodes.values():
        if node.kind not in _SUPPORTABLE or node.node_id in has_support:
            continue
        if node.undeveloped:
            findings.append(
                Finding(
                    check="unsupported_claims",
                    severity=Severity.INFO,
                    node_ids=(node.node_id,),
                    message=(
                        f"{node.kind.value} {node.node_id!r} is declared Undeveloped and has "
                        "no SupportedBy child; this is a declaration, not a defect"
                    ),
                )
            )
        else:
            findings.append(
                Finding(
                    check="unsupported_claims",
                    severity=Severity.ERROR,
                    node_ids=(node.node_id,),
                    message=(
                        f"{node.kind.value} {node.node_id!r} has no SupportedBy child: nothing "
                        "in the case argues for it. Add a SupportedBy edge to a Strategy, a "
                        "sub-Goal or a Solution, or mark it undeveloped: true"
                    ),
                )
            )
    return findings


def check_missing_evidence(
    case: AssuranceCase, reports: list[EvidenceReport] | None = None
) -> list[Finding]:
    """Cited artifacts that are absent from disk or not readable as a file."""
    reports = inspect_case_evidence(case) if reports is None else reports
    findings: list[Finding] = []
    for report in reports:
        if report.status is EvidenceStatus.ABSENT:
            findings.append(
                Finding(
                    check="missing_evidence",
                    severity=Severity.ERROR,
                    node_ids=(report.node_id,),
                    message=(
                        f"Solution {report.node_id!r} cites {report.cited_path!r} but "
                        f"{report.detail}. Either produce the artifact or remove the claim "
                        "that rests on it"
                    ),
                )
            )
        elif report.status is EvidenceStatus.UNREADABLE:
            findings.append(
                Finding(
                    check="missing_evidence",
                    severity=Severity.ERROR,
                    node_ids=(report.node_id,),
                    message=(
                        f"Solution {report.node_id!r} cites {report.cited_path!r} but "
                        f"{report.detail}. A cited artifact must be a readable regular file"
                    ),
                )
            )
    return findings


def check_stale_evidence(
    case: AssuranceCase, reports: list[EvidenceReport] | None = None
) -> list[Finding]:
    """Cited artifacts whose content hash differs from the recorded digest.

    A ``STALE`` artifact is an ``ERROR``. An artifact the case recorded no
    digest for is a ``WARNING``, not a pass: staleness is not computable for it,
    and the report says so rather than implying freshness.
    """
    reports = inspect_case_evidence(case) if reports is None else reports
    findings: list[Finding] = []
    for report in reports:
        if report.status is EvidenceStatus.STALE:
            findings.append(
                Finding(
                    check="stale_evidence",
                    severity=Severity.ERROR,
                    node_ids=(report.node_id,),
                    message=(
                        f"Solution {report.node_id!r} cites {report.cited_path!r}: "
                        f"{report.detail}. Re-examine the artifact and update the recorded "
                        "sha256, or withdraw the claim"
                    ),
                )
            )
        elif report.status is EvidenceStatus.UNVERIFIABLE:
            findings.append(
                Finding(
                    check="stale_evidence",
                    severity=Severity.WARNING,
                    node_ids=(report.node_id,),
                    message=(
                        f"Solution {report.node_id!r} cites {report.cited_path!r} with no "
                        "recorded sha256, so freshness cannot be computed; the current "
                        f"digest is {report.actual_sha256}. Record it to make the artifact "
                        "checkable"
                    ),
                )
            )
    return findings


def check_undischarged_assumptions(case: AssuranceCase) -> list[Finding]:
    """Assumptions not marked discharged.

    ``discharged: true`` is an author's assertion. This check verifies that the
    assertion was made and, when ``discharged_by`` is present, that it names a
    node of the case -- :mod:`assuregraph.parse` rejects a dangling reference.
    It does not and cannot verify that the named node discharges anything.
    """
    findings: list[Finding] = []
    for node in case.nodes_of_kind(NodeKind.ASSUMPTION):
        if not node.discharged:
            findings.append(
                Finding(
                    check="undischarged_assumptions",
                    severity=Severity.ERROR,
                    node_ids=(node.node_id,),
                    message=(
                        f"Assumption {node.node_id!r} is not discharged: {node.statement!r}. "
                        "Set discharged: true with a discharged_by node id once something in "
                        "the case addresses it"
                    ),
                )
            )
    return findings


def check_cycles(case: AssuranceCase) -> list[Finding]:
    """Cyclic regions of the argument graph.

    One witness cycle is reported per strongly connected component; see
    :func:`assuregraph.graph.find_cycles` for why not all of them.
    """
    findings: list[Finding] = []
    for walk in find_cycles(case):
        findings.append(
            Finding(
                check="cycles",
                severity=Severity.ERROR,
                node_ids=tuple(walk),
                message=(
                    "the argument contains a cycle: "
                    + " -> ".join(walk)
                    + ". A claim supported, directly or transitively, by itself argues for "
                    "nothing; break the cycle"
                ),
            )
        )
    return findings


def check_orphan_nodes(case: AssuranceCase) -> list[Finding]:
    """Nodes unreachable from any top Goal.

    The top Goals are the case's declared ``top_goals``, or, when it declares
    none, the Goals with no incoming ``SupportedBy`` edge. A node unreachable
    from all of them is in the document but not in the argument. Both
    relationship types are traversed, so a Context attached only to an orphan
    Strategy is itself an orphan.

    A case with two or more independent top Goals is normal and produces no
    finding here; a case whose declared top Goals do not reach a node does.
    """
    roots = case.resolved_top_goals()
    reached = reachable_from(case, roots)
    root_text = ", ".join(roots) if roots else "the case declares none and none could be inferred"
    findings: list[Finding] = []
    for node_id, node in case.nodes.items():
        if node_id not in reached:
            findings.append(
                Finding(
                    check="orphan_nodes",
                    severity=Severity.ERROR,
                    node_ids=(node_id,),
                    message=(
                        f"{node.kind.value} {node_id!r} is not reachable from any top goal "
                        f"({root_text}): it is in the document but not in the argument"
                    ),
                )
            )
    return findings


@dataclass(frozen=True)
class Coverage:
    """Counted structure of a case, with every denominator stated.

    Attributes:
        node_counts: Number of nodes of each :class:`~assuregraph.model.NodeKind`,
            keyed by the kind's YAML value.
        edge_counts: Number of edges of each
            :class:`~assuregraph.model.EdgeKind`, keyed by the kind's YAML value.
        claims_total: Goals plus Strategies. The denominator of
            ``claim_support_fraction``.
        claims_supported: Of those, the number with at least one ``SupportedBy``
            child.
        claims_undeveloped: Of the unsupported ones, the number carrying the GSN
            *Undeveloped* decorator.
        evidence_total: Number of Solution nodes. The denominator of
            ``evidence_present_fraction``.
        evidence_fresh: Solutions whose artifact is present and whose digest
            matches.
        evidence_stale: Solutions whose artifact is present and whose digest
            does not match.
        evidence_unverifiable: Solutions whose artifact is present and for which
            no digest was recorded. The denominator of
            ``evidence_fresh_fraction`` excludes these, because freshness is not
            defined for them.
        evidence_missing: Solutions whose artifact is absent or unreadable.
        assumptions_total: Number of Assumption nodes. The denominator of
            ``assumption_discharged_fraction``.
        assumptions_discharged: Of those, the number marked discharged.
        orphan_count: Nodes unreachable from any top goal.
        cycle_regions: Number of cyclic strongly connected components.
        top_goals: The top goal ids used for the reachability computation.
    """

    node_counts: dict[str, int]
    edge_counts: dict[str, int]
    claims_total: int
    claims_supported: int
    claims_undeveloped: int
    evidence_total: int
    evidence_fresh: int
    evidence_stale: int
    evidence_unverifiable: int
    evidence_missing: int
    assumptions_total: int
    assumptions_discharged: int
    orphan_count: int
    cycle_regions: int
    top_goals: tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def _fraction(numerator: int, denominator: int) -> float | None:
        """Return ``numerator / denominator``, or ``None`` when the denominator is 0.

        A fraction with an empty denominator is not 0 and not 1; it is
        undefined, and this package reports it as undefined rather than picking
        whichever of the two looks better.
        """
        return None if denominator == 0 else numerator / denominator

    @property
    def claim_support_fraction(self) -> float | None:
        """Claims that are supported or declared undeveloped, over all claims.

        Denominator: ``claims_total`` (Goals + Strategies). ``None`` when there
        are no claims.
        """
        return self._fraction(self.claims_supported + self.claims_undeveloped, self.claims_total)

    @property
    def evidence_present_fraction(self) -> float | None:
        """Solutions whose artifact is on disk, over all Solutions.

        Denominator: ``evidence_total``. ``None`` when there are no Solutions.
        """
        return self._fraction(self.evidence_total - self.evidence_missing, self.evidence_total)

    @property
    def evidence_fresh_fraction(self) -> float | None:
        """Fresh Solutions over Solutions for which freshness is computable.

        Denominator: ``evidence_fresh + evidence_stale``, which deliberately
        excludes ``evidence_unverifiable`` and ``evidence_missing``. ``None``
        when nothing is computable.
        """
        return self._fraction(self.evidence_fresh, self.evidence_fresh + self.evidence_stale)

    @property
    def assumption_discharged_fraction(self) -> float | None:
        """Discharged Assumptions over all Assumptions.

        Denominator: ``assumptions_total``. ``None`` when there are none.
        """
        return self._fraction(self.assumptions_discharged, self.assumptions_total)


@dataclass(frozen=True)
class CheckReport:
    """Everything :func:`run_checks` computed about one case.

    Attributes:
        case_name: The case's ``name``.
        findings: Every finding from every check, grouped by check in
            :data:`CHECK_NAMES` order and, within a check, in document order.
        evidence: One :class:`~assuregraph.evidence.EvidenceReport` per Solution.
        coverage: Counted structure; see :class:`Coverage`.
        strict: Whether ``WARNING`` findings count towards incompleteness.
    """

    case_name: str
    findings: tuple[Finding, ...]
    evidence: tuple[EvidenceReport, ...]
    coverage: Coverage
    strict: bool = False

    def by_check(self, check: str) -> tuple[Finding, ...]:
        """Findings from one named check."""
        return tuple(f for f in self.findings if f.check == check)

    def by_severity(self, severity: Severity) -> tuple[Finding, ...]:
        """Findings of one severity."""
        return tuple(f for f in self.findings if f.severity is severity)

    @property
    def error_count(self) -> int:
        """Number of ``ERROR`` findings."""
        return len(self.by_severity(Severity.ERROR))

    @property
    def warning_count(self) -> int:
        """Number of ``WARNING`` findings."""
        return len(self.by_severity(Severity.WARNING))

    @property
    def info_count(self) -> int:
        """Number of ``INFO`` findings."""
        return len(self.by_severity(Severity.INFO))

    @property
    def is_complete(self) -> bool:
        """True when the case has no blocking finding.

        "Complete" here means every claim is argued or declared undeveloped,
        every cited artifact is present and matches its recorded digest, every
        assumption is marked discharged, the graph is acyclic and every node is
        in the argument. It does **not** mean the argument is sound, and it is
        not a statement about any system. Under ``strict`` the ``WARNING``
        findings block as well.
        """
        if self.error_count:
            return False
        return not (self.strict and self.warning_count)

    @property
    def exit_code(self) -> int:
        """0 when :attr:`is_complete`, otherwise 1. The CLI returns this."""
        return 0 if self.is_complete else 1


def _coverage(case: AssuranceCase, reports: list[EvidenceReport], cycle_regions: int) -> Coverage:
    has_support = {e.source for e in case.edges if e.kind is EdgeKind.SUPPORTED_BY}
    claims = [n for n in case.nodes.values() if n.kind in _SUPPORTABLE]
    supported = [n for n in claims if n.node_id in has_support]
    undeveloped = [n for n in claims if n.node_id not in has_support and n.undeveloped]
    assumptions = case.nodes_of_kind(NodeKind.ASSUMPTION)
    roots = case.resolved_top_goals()
    reached = reachable_from(case, roots)
    statuses = [r.status for r in reports]
    return Coverage(
        node_counts={k.value: len(case.nodes_of_kind(k)) for k in NodeKind},
        edge_counts={k.value: len(case.edges_of_kind(k)) for k in EdgeKind},
        claims_total=len(claims),
        claims_supported=len(supported),
        claims_undeveloped=len(undeveloped),
        evidence_total=len(reports),
        evidence_fresh=statuses.count(EvidenceStatus.FRESH),
        evidence_stale=statuses.count(EvidenceStatus.STALE),
        evidence_unverifiable=statuses.count(EvidenceStatus.UNVERIFIABLE),
        evidence_missing=statuses.count(EvidenceStatus.ABSENT)
        + statuses.count(EvidenceStatus.UNREADABLE),
        assumptions_total=len(assumptions),
        assumptions_discharged=sum(1 for n in assumptions if n.discharged),
        orphan_count=sum(1 for node_id in case.nodes if node_id not in reached),
        cycle_regions=cycle_regions,
        top_goals=roots,
    )


def run_checks(case: AssuranceCase, *, strict: bool = False) -> CheckReport:
    """Run all six checks and return the combined report.

    Evidence is hashed exactly once and the result shared between
    :func:`check_missing_evidence` and :func:`check_stale_evidence`, so a case
    with large artifacts is read once per run, not twice.

    Args:
        case: A case from :func:`assuregraph.parse.load_case`.
        strict: Promote ``WARNING`` findings to blocking, so an artifact with no
            recorded digest makes the case incomplete.

    Returns:
        A :class:`CheckReport` whose ``exit_code`` is 0 only when nothing
        blocking was found.
    """
    reports = inspect_case_evidence(case)
    cycle_findings = check_cycles(case)
    findings: list[Finding] = [
        *check_unsupported_claims(case),
        *check_missing_evidence(case, reports),
        *check_stale_evidence(case, reports),
        *check_undischarged_assumptions(case),
        *cycle_findings,
        *check_orphan_nodes(case),
    ]
    return CheckReport(
        case_name=case.name,
        findings=tuple(findings),
        evidence=tuple(reports),
        coverage=_coverage(case, reports, len(cycle_findings)),
        strict=strict,
    )
