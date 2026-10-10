"""Rendering an :class:`~traceaudit.findings.AuditResult` as text or JSON.

No function here prints; they return strings, so the CLI owns all output.
"""

from __future__ import annotations

import json
from typing import Any

from .findings import CODE_DESCRIPTIONS, AuditResult

BOOKKEEPING_NOTICE = (
    "traceability is a necessary bookkeeping condition, not evidence of adequacy: "
    "a requirement traced to a test that asserts nothing is still traced"
)


def render_text(result: AuditResult, *, show_matrix: bool = False) -> str:
    """Human-readable report, ending with the finding count."""
    lines: list[str] = []
    lines.append("traceaudit report")
    lines.append(f"  requirements document : {', '.join(result.document.sources)}")
    lines.append(f"  test report           : {result.report.source} ({result.report.kind})")
    lines.append(
        f"  declarations parsed   : {len(result.document.declarations)} "
        f"({result.matrix.n_requirements} unique ids)"
    )
    counts = result.report.outcome_counts
    lines.append(
        "  test cases parsed     : "
        f"{len(result.report.cases)} "
        f"(passed {counts['passed']}, failed {counts['failed']}, "
        f"errored {counts['errored']}, skipped {counts['skipped']}, "
        f"xfailed {counts['xfailed']}, unknown {counts['unknown']})"
    )
    lines.append(f"  claiming test cases   : {sum(1 for c in result.report.cases if c.claims)}")
    lines.append("")
    lines.append("coverage, with denominators stated")
    for figure in result.coverage:
        lines.append(f"  {figure.render()}")
        lines.append(f"      definition: {figure.definition}")
    if not result.matrix.outcomes_known:
        lines.append("  executed coverage: undefined (report carries no outcomes)")
        lines.append("  passing coverage : undefined (report carries no outcomes)")
    lines.append("")

    if show_matrix:
        lines.append("requirement -> tests")
        for req, tests in result.matrix.requirement_to_tests.items():
            if tests:
                for i, node in enumerate(tests):
                    mark = req if i == 0 else ""
                    outcome = result.matrix.outcome_by_test[node].value
                    lines.append(f"  {mark:<10s} {node}  [{outcome}]")
            else:
                lines.append(f"  {req:<10s} -")
        lines.append("")

    if result.notes:
        lines.append("notes")
        for note in result.notes:
            lines.append(f"  - {note}")
        lines.append("")

    lines.append("findings")
    if not result.findings:
        lines.append("  none")
    else:
        for finding in result.findings:
            lines.append(f"  {finding.render()}")
    lines.append("")
    by_code = result.counts_by_code()
    lines.append("summary by code")
    for code, description in CODE_DESCRIPTIONS.items():
        lines.append(f"  {code} {by_code.get(code, 0):>4d}  {description}")
    lines.append("")
    lines.append(
        f"{len(result.findings)} finding(s), {len(result.blocking)} blocking; "
        f"exit status {result.exit_code}"
    )
    lines.append(f"NOTE: {BOOKKEEPING_NOTICE}.")
    return "\n".join(lines)


def to_dict(result: AuditResult) -> dict[str, Any]:
    """JSON-serialisable form of the whole result."""
    return {
        "requirements_sources": list(result.document.sources),
        "test_report": {
            "source": result.report.source,
            "kind": result.report.kind,
            "declared_totals": dict(result.report.declared_totals),
            "outcome_counts": result.report.outcome_counts,
            "cases": len(result.report.cases),
        },
        "declarations": [
            {"id": r.id, "title": r.title, "location": r.location}
            for r in result.document.declarations
        ],
        "unique_requirement_ids": list(result.matrix.requirement_to_tests),
        "coverage": [
            {
                "label": f.label,
                "numerator": f.numerator,
                "denominator": f.denominator,
                "percent": f.percent,
                "definition": f.definition,
            }
            for f in result.coverage
        ],
        "requirement_to_tests": {
            k: list(v) for k, v in result.matrix.requirement_to_tests.items()
        },
        "test_to_requirements": {
            k: list(v) for k, v in result.matrix.test_to_requirements.items() if v
        },
        "unknown_claims": {k: list(v) for k, v in result.matrix.unknown_claims.items()},
        "findings": [
            {
                "code": f.code,
                "subject": f.subject,
                "message": f.message,
                "location": f.location,
                "heuristic": f.heuristic,
                "ignored": f.ignored,
            }
            for f in result.findings
        ],
        "counts_by_code": result.counts_by_code(),
        "notes": list(result.notes),
        "exit_code": result.exit_code,
        "bookkeeping_notice": BOOKKEEPING_NOTICE,
    }


def render_json(result: AuditResult) -> str:
    """Indented JSON, stable key order."""
    return json.dumps(to_dict(result), indent=2, sort_keys=False)


def render_markdown_matrix(result: AuditResult) -> str:
    """A markdown traceability table, for pasting into a review document."""
    lines = ["| requirement | title | tests | outcomes |", "|---|---|---|---|"]
    for req, tests in result.matrix.requirement_to_tests.items():
        decl = result.document.first(req)
        title = (decl.title if decl else "").replace("|", "\\|")
        if tests:
            names = "<br>".join(t.replace("|", "\\|") for t in tests)
            outcomes = "<br>".join(
                result.matrix.outcome_by_test[t].value for t in tests
            )
        else:
            names, outcomes = "none", "-"
        lines.append(f"| {req} | {title} | {names} | {outcomes} |")
    return "\n".join(lines)
