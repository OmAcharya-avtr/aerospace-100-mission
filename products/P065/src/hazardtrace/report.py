"""Rendering. Text, JSON and markdown, all from the same Report.

Every renderer repeats the scope statement, because the one thing a hazard
analysis tool must not do is let a clean exit code travel without it.
"""

from __future__ import annotations

import json

from .checks import CHECKS, Coverage, Report, Severity
from .enumeration import CellState
from .model import UCA_CLASSES

#: Printed by every renderer, verbatim, next to the verdict.
SCOPE_NOTE = (
    "A complete verdict means the analysis is fully drawn and the identifiers it "
    "cites resolve. It is not a statement that the system is safe, not a statement "
    "that the hazard analysis is correct or adequate, and it discharges no safety "
    "obligation."
)

_STATE_MARK = {
    CellState.UCA: "U",
    CellState.EXCLUDED: "x",
    CellState.CONFLICT: "!",
    CellState.GAP: ".",
}


def format_coverage_table(coverage: Coverage) -> str:
    """Coverage figures as an aligned text table, denominators included."""
    lines = ["coverage, with denominators stated"]
    for label, ratio in coverage.as_items():
        lines.append(f"  {label:<32}{ratio}")
        lines.append(f"      definition: {ratio.definition}")
    return "\n".join(lines) + "\n"


def format_grid(report: Report) -> str:
    """The enumeration grid as a text matrix.

    Legend: ``U`` a declared UCA, ``x`` a justified exclusion, ``!`` both
    (a contradiction), ``.`` an enumeration gap.
    """
    grid = report.grid
    width = 20
    header = "  ".join(f"{c.value[:width]:<{width}}" for c in UCA_CLASSES)
    lines = [
        "UCA enumeration grid  (U declared, x excluded, ! both, . gap)",
        f"  {'control action':<16}{header}",
    ]
    for action in report.analysis.structure.control_actions:
        marks = []
        for uca_class in UCA_CLASSES:
            cell = grid.cell(action.id, uca_class)
            mark = _STATE_MARK[cell.state] if cell is not None else "?"
            detail = ",".join(cell.uca_ids) if cell is not None and cell.uca_ids else ""
            marks.append(f"{mark} {detail}".strip().ljust(width))
        lines.append((f"  {action.id:<16}" + "  ".join(marks)).rstrip())
    covered, total = grid.completeness()
    lines.append(f"  cells covered: {covered}/{total}; gaps: {len(grid.gaps)}")
    return "\n".join(lines) + "\n"


def format_findings(report: Report) -> str:
    """Findings, one per line, in check order."""
    if not report.findings:
        return "findings\n  none\n"
    lines = ["findings"]
    for finding in report.findings:
        where = f"  [{report.analysis.source}:{finding.line}]" if finding.line else ""
        lines.append(
            f"  {finding.code} {finding.severity.value:<7} {finding.message}{where}"
        )
    lines.append("")
    lines.append("summary by code")
    counts = report.count_by_code()
    for spec in CHECKS:
        if counts[spec.code]:
            lines.append(f"  {spec.code}  {counts[spec.code]:>3}  {spec.title}")
    return "\n".join(lines) + "\n"


def format_report(report: Report) -> str:
    """The whole text report: header, grid, coverage, findings, verdict."""
    analysis = report.analysis
    structure = analysis.structure
    counts = report.count_by_severity()
    head = [
        "hazardtrace report",
        f"  analysis            : {analysis.title}",
        f"  system              : {analysis.system or '(not declared)'}",
        f"  source              : {analysis.source}",
        f"  revision            : {analysis.revision or '(not declared)'}",
        f"  losses / hazards    : {len(analysis.losses)} / {len(analysis.hazards)}",
        f"  constraints         : {len(analysis.constraints)}",
        f"  control actions     : {len(structure.control_actions)}",
        f"  feedback paths      : {len(structure.feedback)}",
        f"  UCAs / exclusions   : {len(analysis.ucas)} / {len(analysis.exclusions)}",
        f"  scenarios           : {len(analysis.loss_scenarios)}",
        f"  mitigations         : {len(analysis.mitigations)}",
        f"  evidence artifacts  : {len(analysis.evidence)}",
        f"  requirements source : {report.requirements_source or '(not supplied)'}",
        f"  test report source  : {report.tests_source or '(not supplied)'}",
    ]
    verdict = [
        "verdict",
        f"  {counts[Severity.ERROR]} error, {counts[Severity.WARNING]} warning, "
        f"{counts[Severity.INFO]} info finding(s)",
        f"  structural verdict: {'COMPLETE' if report.is_complete else 'INCOMPLETE'}. "
        f"Exit code {report.exit_code}.",
        "",
        f"  {SCOPE_NOTE}",
    ]
    return (
        "\n".join(head)
        + "\n\n"
        + format_grid(report)
        + "\n"
        + format_coverage_table(report.coverage)
        + "\n"
        + format_findings(report)
        + "\n"
        + "\n".join(verdict)
        + "\n"
    )


def report_to_dict(report: Report) -> dict[str, object]:
    """The report as a plain dictionary, suitable for ``json.dumps``."""
    return {
        "analysis": {
            "title": report.analysis.title,
            "system": report.analysis.system,
            "revision": report.analysis.revision,
            "source": report.analysis.source,
        },
        "counts": {
            "losses": len(report.analysis.losses),
            "hazards": len(report.analysis.hazards),
            "constraints": len(report.analysis.constraints),
            "control_actions": len(report.analysis.structure.control_actions),
            "feedback_paths": len(report.analysis.structure.feedback),
            "ucas": len(report.analysis.ucas),
            "exclusions": len(report.analysis.exclusions),
            "loss_scenarios": len(report.analysis.loss_scenarios),
            "mitigations": len(report.analysis.mitigations),
            "evidence": len(report.analysis.evidence),
        },
        "grid": {
            "total_cells": report.grid.total,
            "covered_cells": report.grid.covered,
            "by_state": {
                state.value: report.grid.count(state) for state in CellState
            },
            "cells": [
                {
                    "control_action": cell.control_action,
                    "uca_class": cell.uca_class.value,
                    "state": cell.state.value,
                    "ucas": list(cell.uca_ids),
                }
                for cell in report.grid.cells
            ],
        },
        "coverage": {
            label: {
                "numerator": ratio.numerator,
                "denominator": ratio.denominator,
                "defined": ratio.defined,
                "definition": ratio.definition,
            }
            for label, ratio in report.coverage.as_items()
        },
        "findings": [
            {
                "code": finding.code,
                "severity": finding.severity.value,
                "message": finding.message,
                "subjects": list(finding.subjects),
                "line": finding.line,
            }
            for finding in report.findings
        ],
        "verdict": {
            "complete": report.is_complete,
            "exit_code": report.exit_code,
            "scope_note": SCOPE_NOTE,
        },
    }


def format_json(report: Report) -> str:
    """The report as indented JSON with sorted keys, newline terminated."""
    return json.dumps(report_to_dict(report), indent=2, sort_keys=True) + "\n"


def format_markdown(report: Report) -> str:
    """The report as markdown tables, for pasting into a review."""
    lines = [
        f"# hazardtrace: {report.analysis.title}",
        "",
        f"Source `{report.analysis.source}`. "
        f"Verdict **{'COMPLETE' if report.is_complete else 'INCOMPLETE'}**, "
        f"exit code {report.exit_code}.",
        "",
        f"> {SCOPE_NOTE}",
        "",
        "## Coverage",
        "",
        "| figure | value | definition |",
        "|---|---|---|",
    ]
    for label, ratio in report.coverage.as_items():
        lines.append(f"| {label} | {ratio} | {ratio.definition} |")
    lines += ["", "## Enumeration grid", "", "| control action | " + " | ".join(
        c.label for c in UCA_CLASSES
    ) + " |", "|---|" + "---|" * len(UCA_CLASSES)]
    for action in report.analysis.structure.control_actions:
        row = []
        for uca_class in UCA_CLASSES:
            cell = report.grid.cell(action.id, uca_class)
            if cell is None:
                row.append("?")
            elif cell.uca_ids:
                prefix = "conflict " if cell.excluded else ""
                row.append(prefix + ", ".join(cell.uca_ids))
            elif cell.excluded:
                row.append("excluded")
            else:
                row.append("**gap**")
        lines.append(f"| `{action.id}` | " + " | ".join(row) + " |")
    lines += ["", "## Findings", ""]
    if not report.findings:
        lines.append("None.")
    else:
        lines += ["| code | severity | subject | message |", "|---|---|---|---|"]
        for finding in report.findings:
            subjects = ", ".join(f"`{s}`" for s in finding.subjects) or "-"
            lines.append(
                f"| {finding.code} | {finding.severity.value} | {subjects} | "
                f"{finding.message} |"
            )
    return "\n".join(lines) + "\n"


def format_codes() -> str:
    """The check catalogue as a text table."""
    lines = ["hazardtrace checks", f"  {'code':<8}{'severity':<10}title"]
    for spec in CHECKS:
        lines.append(f"  {spec.code:<8}{spec.severity.value:<10}{spec.title}")
    return "\n".join(lines) + "\n"
