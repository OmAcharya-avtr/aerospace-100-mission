"""Text rendering of a diff, an impact report and a coverage verdict.

No function here prints. Every one returns a string, so the library can be
used without a terminal and so the CLI is the only place output happens.

Rendering order is the product's ranking rule made visible: the silent tier
comes first in a diff, the silently changed findings come first in an impact
report, and the coverage verdict is printed with the findings rather than
after them.
"""

from __future__ import annotations

from collections.abc import Sequence

from .changes import CLASS_NOTES, ChangeClass, Severity
from .diff import DiffResult
from .impact import IMPACT_NOTES, ImpactClass, ImpactReport


def format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A fixed-width table. Columns are as wide as their widest cell."""
    cells = [[str(cell) for cell in row] for row in rows]
    columns = len(headers)
    for row in cells:
        if len(row) != columns:
            raise ValueError(f"row {row!r} has {len(row)} cells, expected {columns}")
    widths = [len(str(header)) for header in headers]
    for row in cells:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = ["  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    lines.append("  ".join("-" * width for width in widths))
    for row in cells:
        lines.append(
            "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()
        )
    return "\n".join(lines)


def taxonomy_table() -> str:
    """The change-class taxonomy as a table, silent tier first."""
    rows = [
        [change_class.value, change_class.severity.value, CLASS_NOTES[change_class]]
        for change_class in sorted(
            ChangeClass, key=lambda c: (c.severity.rank, c.value)
        )
    ]
    return format_table(["change class", "tier", "what it means for an existing reference"], rows)


def impact_class_table() -> str:
    """The impact classes as a table, in report order."""
    rows = [
        [impact_class.value, str(impact_class.rank), IMPACT_NOTES[impact_class]]
        for impact_class in sorted(ImpactClass, key=lambda c: c.rank)
    ]
    return format_table(["impact class", "rank", "meaning"], rows)


def render_diff(result: DiffResult, *, max_rows: int = 0) -> str:
    """Render a diff with the silent tier first.

    ``max_rows`` limits the number of changes listed per tier; 0 lists all.
    """
    lines = [
        f"dictionary {result.old_version} -> {result.new_version}",
        f"items {result.old_item_count} -> {result.new_item_count}",
        "",
        "SILENT CHANGES FIRST. A silent change leaves every existing reference resolving",
        "and running, and alters what it means. Within-tier order is a declared",
        "convention and is not a measured ranking.",
        "",
    ]
    for tier, changes in (
        (Severity.SILENT, result.silent),
        (Severity.LOUD, result.loud),
        (Severity.ADVISORY, result.advisory),
    ):
        lines.append(f"-- {tier.value} ({len(changes)}) " + "-" * 40)
        shown = changes if max_rows == 0 else changes[:max_rows]
        lines.extend(f"  {change.one_line()}" for change in shown)
        if len(shown) < len(changes):
            lines.append(f"  ... {len(changes) - len(shown)} more")
        lines.append("")

    lines.append("-- what this diff could not determine " + "-" * 24)
    lines.append(f"  incomparable fields : {result.limits.incomparable_fields}")
    lines.append(f"  ambiguous renames   : {result.limits.ambiguous_renames}")
    lines.append(f"  unmatched removals  : {result.limits.unmatched_removed}")
    lines.append(f"  unmatched additions : {result.limits.unmatched_added}")
    lines.append(f"  fully determinable  : {result.limits.determinable}")
    for reason in result.limits.reasons:
        lines.append(f"  - {reason}")
    if result.incomparable:
        lines.append("")
        lines.append("-- incomparable fields, named " + "-" * 32)
        lines.extend(f"  {field.one_line()}" for field in result.incomparable)
    return "\n".join(lines)


def render_impact(report: ImpactReport, *, max_rows: int = 0) -> str:
    """Render an impact report: findings ranked, then the coverage verdict."""
    lines = [
        f"impact of {report.old_version} -> {report.new_version} on the given corpus",
        "",
    ]
    counts = report.class_counts
    lines.append(
        format_table(
            ["impact class", "references", "meaning"],
            [
                [impact_class.value, str(count), IMPACT_NOTES[impact_class]]
                for impact_class, count in counts.items()
            ],
        )
    )
    lines.append("")
    for impact_class in sorted(ImpactClass, key=lambda c: c.rank):
        findings = report.of_class(impact_class)
        if not findings:
            continue
        lines.append(f"-- {impact_class.value} ({len(findings)}) " + "-" * 36)
        shown = findings if max_rows == 0 else findings[:max_rows]
        lines.extend(f"  {finding.one_line()}" for finding in shown)
        if len(shown) < len(findings):
            lines.append(f"  ... {len(findings) - len(shown)} more")
        lines.append("")
    lines.append("-- coverage verdict: what this assessment could not determine " + "-" * 4)
    lines.extend(f"  {line}" for line in report.verdict.report_lines())
    lines.append("")
    lines.append("-- interpretation " + "-" * 44)
    lines.append(f"  {report.interpretation}")
    return "\n".join(lines)
