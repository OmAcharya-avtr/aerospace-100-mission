"""Format a :class:`~assuregraph.checks.CheckReport` as text, Markdown or JSON.

Every table in this module states its denominator in the table itself, because
a coverage percentage without a denominator is the kind of number that ends up
in a slide. An undefined fraction -- zero Solutions, so no freshness to
report -- prints as ``n/a (denominator 0)`` and never as 0 % or 100 %.

This module contains no ``print``; it returns strings. The CLI in
:mod:`assuregraph.__main__` is the only place in the package that writes to a
stream.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .checks import CHECK_NAMES, CheckReport, Coverage, Severity
from .model import EdgeKind, NodeKind

__all__ = [
    "SCOPE_STATEMENT",
    "format_coverage_table",
    "format_findings",
    "format_report",
    "report_as_dict",
]

#: The scope statement every report carries. It is in the library, not only in
#: the README, so a report pasted into a review ticket carries it too.
SCOPE_STATEMENT = (
    "assuregraph checks the structure of an assurance case and the freshness of "
    "the artifacts it cites. A well-formed assurance case is not a safe system. "
    "This tool evaluates no argument's soundness and is not a DO-178C or "
    "ARP4754A compliance tool."
)

_SEVERITY_ORDER = (Severity.ERROR, Severity.WARNING, Severity.INFO)


def _fraction_text(value: float | None, denominator: int) -> str:
    if value is None:
        return f"n/a (denominator {denominator})"
    return f"{100.0 * value:.1f} % ({denominator} in denominator)"


def _render_table(headers: list[str], rows: list[list[str]], markdown: bool) -> list[str]:
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    if markdown:
        out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        out.extend("| " + " | ".join(row) + " |" for row in rows)
        return out
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    out.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        out.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return out


def format_coverage_table(coverage: Coverage, *, markdown: bool = False) -> str:
    """Render the coverage summary as a two-part table.

    The first part counts nodes and edges by type. The second part gives the
    four coverage fractions, each with its denominator printed beside it.
    """
    node_rows = [
        [kind.conventional_prefix, kind.value, str(coverage.node_counts.get(kind.value, 0))]
        for kind in NodeKind
    ]
    edge_rows = [
        [kind.value, str(coverage.edge_counts.get(kind.value, 0))] for kind in EdgeKind
    ]
    metric_rows = [
        [
            "claims argued or declared undeveloped",
            f"{coverage.claims_supported + coverage.claims_undeveloped}/{coverage.claims_total}",
            _fraction_text(coverage.claim_support_fraction, coverage.claims_total),
        ],
        [
            "cited artifacts present on disk",
            f"{coverage.evidence_total - coverage.evidence_missing}/{coverage.evidence_total}",
            _fraction_text(coverage.evidence_present_fraction, coverage.evidence_total),
        ],
        [
            "artifacts fresh, of those checkable",
            f"{coverage.evidence_fresh}/{coverage.evidence_fresh + coverage.evidence_stale}",
            _fraction_text(
                coverage.evidence_fresh_fraction,
                coverage.evidence_fresh + coverage.evidence_stale,
            ),
        ],
        [
            "assumptions marked discharged",
            f"{coverage.assumptions_discharged}/{coverage.assumptions_total}",
            _fraction_text(coverage.assumption_discharged_fraction, coverage.assumptions_total),
        ],
    ]

    lines: list[str] = ["Node and edge counts", ""]
    lines.extend(_render_table(["prefix", "GSN element", "count"], node_rows, markdown))
    lines.append("")
    lines.extend(_render_table(["GSN relationship", "count"], edge_rows, markdown))
    lines.append("")
    lines.append("Coverage, with denominators")
    lines.append("")
    lines.extend(_render_table(["metric", "count", "fraction"], metric_rows, markdown))
    lines.append("")
    lines.append(
        f"Unverifiable artifacts (no recorded sha256): {coverage.evidence_unverifiable}. "
        "These are excluded from the freshness denominator; they are not fresh."
    )
    lines.append(
        f"Orphan nodes: {coverage.orphan_count}. "
        f"Cyclic regions: {coverage.cycle_regions}."
    )
    top = ", ".join(coverage.top_goals) if coverage.top_goals else "none"
    lines.append(f"Top goals used for reachability: {top}.")
    return "\n".join(lines) + "\n"


def format_findings(report: CheckReport, *, markdown: bool = False, width: int = 96) -> str:
    """Render every finding, grouped by check, with the offending node ids.

    Markdown mode produces one table row per finding, which is what a README or
    a review comment wants. Text mode produces a per-check tally followed by one
    wrapped block per finding, because the ``detail`` messages are sentences and
    a sentence in a fixed-width table column is unreadable.

    Args:
        report: The report to render.
        markdown: Table form rather than wrapped blocks.
        width: Wrap column for text mode. Ignored in Markdown mode. A single
            token longer than ``width`` -- a SHA-256 digest, a filesystem
            path -- is never split, so an individual line may exceed it.
    """
    tally = [
        [
            check,
            str(len(report.by_check(check))),
            ", ".join(
                f"{sev.value}={sum(1 for f in report.by_check(check) if f.severity is sev)}"
                for sev in _SEVERITY_ORDER
                if any(f.severity is sev for f in report.by_check(check))
            )
            or "none",
        ]
        for check in CHECK_NAMES
    ]
    if markdown:
        rows = [
            [f.check, f.severity.value, "`" + "` -> `".join(f.node_ids) + "`", f.message]
            for f in report.findings
        ]
        lines = _render_table(["check", "findings", "by severity"], tally, True)
        lines.append("")
        if rows:
            lines.extend(_render_table(["check", "severity", "node ids", "detail"], rows, True))
        else:
            lines.append("No findings.")
        return "\n".join(lines) + "\n"

    lines = _render_table(["check", "findings", "by severity"], tally, False)
    lines.append("")
    if not report.findings:
        lines.append("No findings.")
        return "\n".join(lines) + "\n"
    for check in CHECK_NAMES:
        for finding in report.by_check(check):
            lines.append(
                f"[{finding.severity.value}] {finding.check}  nodes: "
                + " -> ".join(finding.node_ids)
            )
            lines.extend(
                textwrap.wrap(
                    finding.message,
                    width=width,
                    initial_indent="    ",
                    subsequent_indent="    ",
                    # A 64-character digest or a long path must never be split
                    # across a line break: the reader copies those. A line may
                    # therefore exceed ``width``, and that is the better defect.
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
            lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def format_report(report: CheckReport, *, markdown: bool = False, width: int = 96) -> str:
    """Render the whole report: header, scope statement, findings, coverage, verdict.

    ``width`` is the wrap column passed through to :func:`format_findings` in
    text mode. Raise it when a caller needs a finding message, such as one
    carrying a long filesystem path, to stay on one line.
    """
    heading = "# " if markdown else ""
    lines = [
        f"{heading}assuregraph report: {report.case_name}",
        "",
        SCOPE_STATEMENT,
        "",
        f"{'## ' if markdown else ''}Findings",
        "",
    ]
    lines.append(format_findings(report, markdown=markdown, width=width).rstrip("\n"))
    lines.extend(["", f"{'## ' if markdown else ''}Coverage summary", ""])
    lines.append(format_coverage_table(report.coverage, markdown=markdown).rstrip("\n"))
    verdict = "COMPLETE" if report.is_complete else "INCOMPLETE"
    lines.extend(
        [
            "",
            f"{'## ' if markdown else ''}Verdict",
            "",
            f"{report.error_count} error, {report.warning_count} warning, "
            f"{report.info_count} info finding(s). "
            f"strict={str(report.strict).lower()}.",
            f"Structural verdict: {verdict}. Exit code {report.exit_code}.",
            "",
            "COMPLETE means the case is fully drawn and its cited files are the files "
            "that were cited. It is not a statement about any system, and it is not "
            "evidence that the evidence is adequate.",
        ]
    )
    return "\n".join(lines) + "\n"


def report_as_dict(report: CheckReport) -> dict[str, Any]:
    """Return the report as JSON-serialisable plain data.

    Keys are stable; this is the form the CLI's ``--json`` writes and the form a
    continuous-integration job should consume rather than scraping the text.
    """
    coverage = report.coverage
    return {
        "case_name": report.case_name,
        "scope_statement": SCOPE_STATEMENT,
        "strict": report.strict,
        "is_complete": report.is_complete,
        "exit_code": report.exit_code,
        "counts": {
            "error": report.error_count,
            "warning": report.warning_count,
            "info": report.info_count,
        },
        "findings": [
            {
                "check": f.check,
                "severity": f.severity.value,
                "node_ids": list(f.node_ids),
                "message": f.message,
            }
            for f in report.findings
        ],
        "evidence": [
            {
                "node_id": e.node_id,
                "cited_path": e.cited_path,
                "resolved_path": e.resolved_path,
                "status": e.status.value,
                "recorded_sha256": e.recorded_sha256,
                "actual_sha256": e.actual_sha256,
                "size_bytes": e.size_bytes,
                "detail": e.detail,
            }
            for e in report.evidence
        ],
        "coverage": {
            "node_counts": dict(coverage.node_counts),
            "edge_counts": dict(coverage.edge_counts),
            "claims_total": coverage.claims_total,
            "claims_supported": coverage.claims_supported,
            "claims_undeveloped": coverage.claims_undeveloped,
            "evidence_total": coverage.evidence_total,
            "evidence_fresh": coverage.evidence_fresh,
            "evidence_stale": coverage.evidence_stale,
            "evidence_unverifiable": coverage.evidence_unverifiable,
            "evidence_missing": coverage.evidence_missing,
            "assumptions_total": coverage.assumptions_total,
            "assumptions_discharged": coverage.assumptions_discharged,
            "orphan_count": coverage.orphan_count,
            "cycle_regions": coverage.cycle_regions,
            "top_goals": list(coverage.top_goals),
            "fractions": {
                "claim_support": coverage.claim_support_fraction,
                "evidence_present": coverage.evidence_present_fraction,
                "evidence_fresh": coverage.evidence_fresh_fraction,
                "assumption_discharged": coverage.assumption_discharged_fraction,
            },
            "denominators": {
                "claim_support": coverage.claims_total,
                "evidence_present": coverage.evidence_total,
                "evidence_fresh": coverage.evidence_fresh + coverage.evidence_stale,
                "assumption_discharged": coverage.assumptions_total,
            },
        },
    }
