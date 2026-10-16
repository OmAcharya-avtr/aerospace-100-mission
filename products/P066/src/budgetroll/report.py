"""Text, markdown and JSON renderings of a rollup and its findings.

Every rendering repeats the scope statement beside the verdict, because a
report that travels without it is the one that gets quoted.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from .aggregate import Rollup, linear_rollup_uncertainty, rollup, rss_rollup_uncertainty
from .checks import Finding, has_blocking, run_checks, severity_counts
from .model import Budget
from .sensitivity import LeafSensitivity, sensitivity

SCOPE = (
    "budgetroll checks the arithmetic of a declared budget and the consistency of the "
    "assumptions declared with it. It does not know whether any estimate is good, "
    "whether any margin is adequate, or whether any system works. Educational-grade "
    "(validation level 1): not flight-qualified, not certified, not approved for "
    "operational aerospace use."
)


def _num(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.2f} %"


def rollup_table(roll: Rollup) -> str:
    """Render the rollup as an indented fixed-width table."""
    unit = roll.unit
    header = (
        f"{'node':<34s}{'rule':<11s}{'nominal':>12s}{'conting.':>11s}"
        f"{'predicted':>12s}{'1 sigma':>11s}{'alloc':>12s}{'margin':>12s}"
        f"{'margin@k':>12s}"
    )
    lines = [
        f"Budget {roll.budget_id} [{roll.quantity}] in {unit}; k = {roll.k_sigma:g} sigma",
        header,
        "-" * len(header),
    ]
    for r in roll.results:
        label = ("  " * r.depth) + r.name
        lines.append(
            f"{label[:33]:<34s}{r.combination:<11s}{r.nominal:>12.4f}{r.allowance:>11.4f}"
            f"{r.predicted:>12.4f}{r.uncertainty:>11.4f}"
            f"{_num(r.allocation):>12s}{_num(r.margin_absolute):>12s}"
            f"{_num(r.margin_at_k):>12s}"
        )
    return "\n".join(lines)


def findings_text(findings: Iterable[Finding]) -> str:
    """Render findings one per line, or a single line when there are none."""
    rows = [str(f) for f in findings]
    return "\n".join(rows) if rows else "no findings"


def sensitivity_table(rows: Iterable[LeafSensitivity], unit: str) -> str:
    """Render the leaf sensitivity ranking."""
    rows = list(rows)
    step = rows[0].step if rows else 0.0
    header = (
        f"{'leaf':<30s}{'predicted':>12s}{'value share':>14s}"
        f"{'var share':>12s}{f'd margin @ +{100 * step:g}%':>18s}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.name[:29]:<30s}{row.predicted:>12.4f}{_pct(row.value_share):>14s}"
            f"{_pct(row.variance_share):>12s}{_num(row.margin_delta):>18s}"
        )
    lines.append(f"(magnitudes in {unit}; shares are dimensionless)")
    return "\n".join(lines)


def text_report(budget: Budget, step: float = 0.10) -> str:
    """A complete text report: rollup, combination comparison, checks, ranking."""
    roll = rollup(budget)
    findings = run_checks(budget, roll)
    counts = severity_counts(findings)
    linear = linear_rollup_uncertainty(budget)
    rss = rss_rollup_uncertainty(budget)
    parts = [
        f"budgetroll report for {budget.name}",
        "=" * 72,
        SCOPE,
        "",
        f"Margin policy {budget.policy.name!r}: {budget.policy.description}",
        f"Policy source: {budget.policy.source}",
        "",
        rollup_table(roll),
        "",
        "Combination of leaf uncertainties, all leaves at once:",
        f"  linear (fully correlated upper bound)  {linear:.6f} {roll.unit}",
        f"  root-sum-square (independent)          {rss:.6f} {roll.unit}",
        f"  declared hierarchy at the root         {roll.root.uncertainty:.6f} {roll.unit}",
        f"  exact form with declared correlations  {roll.root.exact_uncertainty:.6f} {roll.unit}",
        "",
        (
            f"Findings: {counts['error']} error, {counts['warning']} warning, "
            f"{counts['info']} info"
        ),
        findings_text(findings),
        "",
        sensitivity_table(sensitivity(budget, step=step), roll.unit),
        "",
        f"Verdict: {'BLOCKING FINDINGS' if has_blocking(findings) else 'no blocking findings'}",
    ]
    return "\n".join(parts)


def markdown_report(budget: Budget, step: float = 0.10) -> str:
    """The same content as :func:`text_report` in GitHub-flavoured markdown."""
    roll = rollup(budget)
    findings = run_checks(budget, roll)
    counts = severity_counts(findings)
    lines = [
        f"# Budget rollup: {budget.name}",
        "",
        f"> {SCOPE}",
        "",
        (
            f"Unit `{budget.unit}` · quantity `{budget.quantity}` · "
            f"k = {budget.k_sigma:g} sigma · policy `{budget.policy.name}`"
        ),
        "",
        f"Policy source: {budget.policy.source}",
        "",
        "## Rollup",
        "",
        (
            "| node | rule | nominal | contingency | predicted | 1 sigma "
            "| allocation | margin | margin at k |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in roll.results:
        indent = "&nbsp;" * (4 * r.depth)
        lines.append(
            f"| {indent}{r.name} | {r.combination} | {r.nominal:.4f} | {r.allowance:.4f} "
            f"| {r.predicted:.4f} | {r.uncertainty:.4f} | {_num(r.allocation)} "
            f"| {_num(r.margin_absolute)} | {_num(r.margin_at_k)} |"
        )
    lines += [
        "",
        "## Findings",
        "",
        f"{counts['error']} error, {counts['warning']} warning, {counts['info']} info.",
        "",
        "| code | severity | subject | message |",
        "|---|---|---|---|",
    ]
    for f in findings:
        lines.append(f"| {f.code} | {f.severity} | `{f.subject}` | {f.message} |")
    if not findings:
        lines.append("| - | - | - | no findings |")
    lines += [
        "",
        "## Sensitivity",
        "",
        f"| leaf | predicted | value share | variance share | d margin at +{100 * step:g}% |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sensitivity(budget, step=step):
        lines.append(
            f"| {row.name} | {row.predicted:.4f} | {_pct(row.value_share)} "
            f"| {_pct(row.variance_share)} | {_num(row.margin_delta)} |"
        )
    return "\n".join(lines) + "\n"


def to_dict(budget: Budget, step: float = 0.10) -> dict:
    """A JSON-serialisable dictionary of the whole report."""
    roll = rollup(budget)
    findings = run_checks(budget, roll)
    return {
        "scope": SCOPE,
        "budget": {
            "id": budget.id,
            "name": budget.name,
            "unit": budget.unit,
            "quantity": budget.quantity,
            "k_sigma": budget.k_sigma,
            "policy": {
                "name": budget.policy.name,
                "description": budget.policy.description,
                "source": budget.policy.source,
                "levels": dict(budget.policy.levels),
            },
        },
        "nodes": [
            {
                "id": r.id,
                "name": r.name,
                "depth": r.depth,
                "is_leaf": r.is_leaf,
                "combination": r.combination,
                "nominal": r.nominal,
                "allowance": r.allowance,
                "predicted": r.predicted,
                "contingency": r.contingency,
                "uncertainty": r.uncertainty,
                "exact_uncertainty": r.exact_uncertainty,
                "allocation": r.allocation,
                "margin_absolute": r.margin_absolute,
                "margin_fraction": r.margin_fraction,
                "margin_at_k": r.margin_at_k,
                "consumption_of_parent_allocation": r.consumption_of_parent_allocation,
            }
            for r in roll.results
        ],
        "combination_comparison": {
            "linear_all_leaves": linear_rollup_uncertainty(budget),
            "rss_all_leaves": rss_rollup_uncertainty(budget),
            "declared_hierarchy_root": roll.root.uncertainty,
            "exact_correlated_root": roll.root.exact_uncertainty,
        },
        "findings": [
            {"code": f.code, "severity": f.severity, "subject": f.subject, "message": f.message}
            for f in findings
        ],
        "severity_counts": severity_counts(findings),
        "blocking": has_blocking(findings),
        "sensitivity": [
            {
                "id": row.id,
                "name": row.name,
                "nominal": row.nominal,
                "predicted": row.predicted,
                "uncertainty": row.uncertainty,
                "value_share": row.value_share,
                "variance_share": row.variance_share,
                "margin_delta": row.margin_delta,
                "step": row.step,
            }
            for row in sensitivity(budget, step=step)
        ],
    }


def json_report(budget: Budget, step: float = 0.10, indent: int = 2) -> str:
    """The report as a JSON string with sorted keys, for diffing."""
    return json.dumps(to_dict(budget, step=step), indent=indent, sort_keys=True) + "\n"
