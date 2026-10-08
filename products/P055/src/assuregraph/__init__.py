"""assuregraph -- an assurance case as a machine-checkable evidence graph.

**Scope statement. Read this first.**

*A well-formed assurance case is not a safe system.* assuregraph checks the
**structure** of an assurance case and the **freshness** of the artifacts the
case cites. It evaluates no argument's soundness. It cannot tell a good argument
from a bad one, a relevant piece of evidence from an irrelevant one, or a test
report that passed from one that failed. It is **not** a DO-178C or ARP4754A
compliance tool, it produces nothing that supports a certification argument, and
a clean run from it is not evidence of anything about a system.

What it actually computes, and the honest limit of each:

* **unsupported claims** -- a Goal or Strategy with no ``SupportedBy`` child.
  Says a box has no line out of it. Says nothing about whether the lines that do
  exist carry any weight.
* **missing evidence** -- a cited artifact that is not on disk. An exact check,
  and the only one here that is.
* **evidence staleness by content hash** -- the artifact's SHA-256 differs from
  the digest recorded when the claim cited it. This detects that the bytes
  changed. It does **not** detect whether the change matters: a corrected
  timestamp and an inverted test verdict are the same finding.
* **undischarged assumptions** -- an Assumption not marked ``discharged``.
  Checks that a human asserted the discharge, never that the discharge is valid.
* **cycles** -- a claim supported transitively by itself. Exact, one witness
  reported per cyclic region.
* **orphan nodes** -- a node unreachable from any top Goal. Exact.

**Structural completeness says nothing about whether the evidence is adequate.**
A case can satisfy all six checks while every Solution cites an empty file whose
digest was recorded from that empty file.

**No AI, by design.** There is no model in this package and none is planned. The
judgement this tool deliberately does not make -- whether an argument is
sound -- is exactly the judgement that must not be automated, because the output
would be an authoritative-sounding opinion with no accountable author. This is a
design decision, not a missing feature.

Naming follows the GSN Community Standard Version 3 (SCSC-141C, SCSC Assurance
Case Working Group, May 2021).

This software is research-grade. It is **not flight-qualified, not certified,
and not approved for operational aerospace use.**

Public API::

    from assuregraph import load_case, run_checks, render_mermaid, format_report

    case = load_case("examples/cases/complete_case.yaml")
    report = run_checks(case)
    report.exit_code       # 0 when structurally complete, 1 otherwise
"""

from __future__ import annotations

from .checks import (
    CHECK_NAMES,
    CheckReport,
    Coverage,
    Finding,
    Severity,
    check_cycles,
    check_missing_evidence,
    check_orphan_nodes,
    check_stale_evidence,
    check_undischarged_assumptions,
    check_unsupported_claims,
    run_checks,
)
from .errors import AssureGraphError, CaseFormatError, CaseIOError
from .evidence import (
    EvidenceReport,
    EvidenceStatus,
    inspect_case_evidence,
    inspect_evidence,
    resolve_evidence_path,
    sha256_file,
)
from .graph import (
    adjacency,
    find_cycles,
    node_depths,
    reachable_from,
    strongly_connected_components,
)
from .mermaid import MERMAID_SHAPES, render_mermaid
from .model import (
    PERMITTED_EDGES,
    AssuranceCase,
    Edge,
    EdgeKind,
    Evidence,
    Node,
    NodeKind,
)
from .parse import LOADER_NAME, load_case, parse_case
from .report import (
    SCOPE_STATEMENT,
    format_coverage_table,
    format_findings,
    format_report,
    report_as_dict,
)

__version__ = "0.1.0"

__all__ = [
    "CHECK_NAMES",
    "LOADER_NAME",
    "MERMAID_SHAPES",
    "PERMITTED_EDGES",
    "SCOPE_STATEMENT",
    "AssuranceCase",
    "AssureGraphError",
    "CaseFormatError",
    "CaseIOError",
    "CheckReport",
    "Coverage",
    "Edge",
    "EdgeKind",
    "Evidence",
    "EvidenceReport",
    "EvidenceStatus",
    "Finding",
    "Node",
    "NodeKind",
    "Severity",
    "__version__",
    "adjacency",
    "check_cycles",
    "check_missing_evidence",
    "check_orphan_nodes",
    "check_stale_evidence",
    "check_undischarged_assumptions",
    "check_unsupported_claims",
    "find_cycles",
    "format_coverage_table",
    "format_findings",
    "format_report",
    "inspect_case_evidence",
    "inspect_evidence",
    "load_case",
    "node_depths",
    "parse_case",
    "reachable_from",
    "render_mermaid",
    "report_as_dict",
    "resolve_evidence_path",
    "run_checks",
    "sha256_file",
    "strongly_connected_components",
]
