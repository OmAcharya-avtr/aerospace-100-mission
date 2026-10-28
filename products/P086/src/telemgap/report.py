"""Plain-text rendering. No colour, no unicode box drawing, no library dependency.

Every renderer here puts the determinability verdict first, because that is the
order the batch's binding rule 17 requires and because it is the order a reader
needs: what the record cannot tell you, before what it appears to say.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .benchmark import ScoreTable
from .classify import IntegrityReport
from .determinability import DeterminabilityVerdict
from .learned import CLASS_NAMES
from .policy import PolicyPlan


def format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Fixed-width text table. Columns are sized to their widest cell.

    A row whose length differs from ``headers`` raises ``ValueError`` rather
    than being padded, because a silently padded row is a wrong table.
    """
    headers = [str(header) for header in headers]
    text_rows = []
    for index, row in enumerate(rows):
        cells = [str(cell) for cell in row]
        if len(cells) != len(headers):
            raise ValueError(
                f"row {index} has {len(cells)} cells but there are {len(headers)} headers"
            )
        text_rows.append(cells)
    widths = [len(header) for header in headers]
    for cells in text_rows:
        for column, cell in enumerate(cells):
            widths[column] = max(widths[column], len(cell))
    lines = ["  ".join(header.ljust(widths[column]) for column, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    for cells in text_rows:
        lines.append("  ".join(cell.ljust(widths[column]) for column, cell in enumerate(cells)))
    return "\n".join(lines)


def _number(value: float, digits: int = 6) -> str:
    """Render a float compactly, with ``nan`` and ``inf`` spelled out."""
    if value is None:
        return "-"
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}g}"


def render_determinability(verdict: DeterminabilityVerdict) -> str:
    """The seven limits, as a table followed by their statements."""
    rows = [
        [
            limit.key,
            "yes" if limit.determinable else "NO",
            _number(limit.quantity),
            limit.unit,
            limit.method,
        ]
        for limit in verdict.limits
    ]
    table = format_table(
        ["limit", "determinable", "quantity", "unit", "method"], rows
    )
    lines = ["WHAT THIS RECORD CANNOT DETERMINE", "", table, ""]
    for limit in verdict.limits:
        lines.append(f"{limit.key}:")
        lines.append(f"  {limit.statement}")
        lines.append(f"  consequence: {limit.consequence}")
    return "\n".join(lines)


def render_integrity(report: IntegrityReport, max_anomalies: int = 20) -> str:
    """Screen summary, verdict counts and the first ``max_anomalies`` anomalies."""
    summary = report.findings.summary()
    lines = [
        f"INTEGRITY REPORT (mode={report.mode})",
        "",
        format_table(
            ["quantity", "value"], [[key, str(value)] for key, value in summary.items()]
        ),
        "",
        format_table(
            ["verdict", "count"],
            [[key, str(value)] for key, value in report.counts.items()],
        ),
    ]
    for note in report.findings.notes:
        lines.extend(["", f"note: {note}"])
    if report.anomalies:
        rows = [
            [
                str(anomaly.index),
                str(anomaly.channel),
                anomaly.observation,
                anomaly.verdict.value,
                f"{anomaly.confidence:.2f}",
                anomaly.evidence[0] if anomaly.evidence else "",
            ]
            for anomaly in report.anomalies[:max_anomalies]
        ]
        lines.extend(
            [
                "",
                f"first {min(max_anomalies, len(report.anomalies))} of "
                f"{len(report.anomalies)} anomalies "
                "(confidence is a declared ordinal tier, not a probability):",
                "",
                format_table(
                    ["frame", "chan", "observation", "verdict", "tier", "evidence"], rows
                ),
            ]
        )
    return "\n".join(lines)


def render_policy(plan: PolicyPlan, max_rows: int = 24) -> str:
    """The gap policy: one row per gap and channel, with the gate that decided it."""
    rows = [
        [
            str(decision.gap_index),
            decision.channel_name,
            f"{decision.duration_s:.3f}",
            f"{decision.slots:.2f}",
            _number(decision.tolerance, 4),
            _number(decision.curvature.value, 4),
            decision.curvature.source[:4],
            "yes" if decision.curvature.measurable else "NO",
            "yes" if decision.curvature.resolvable else "NO",
            _number(decision.coverage, 4),
            _number(decision.predicted_linear, 4),
            _number(decision.predicted_hold, 4),
            decision.decision,
            decision.method,
        ]
        for decision in plan.decisions[:max_rows]
    ]
    header = [
        "gap",
        "channel",
        "dur_s",
        "slots",
        "tol",
        "M2",
        "M2src",
        "meas",
        "resl",
        "k*sig",
        "eq3prime",
        "eq4plus",
        "decision",
        "method",
    ]
    summary = plan.summary()
    lines = [
        "GAP POLICY",
        "",
        f"declared max_gap_slots = {plan.max_gap_slots:g}  "
        f"curvature_quantile = {plan.curvature_quantile:g}  "
        f"coverage_k = {plan.coverage_k:g}  half_window = {plan.half_window:d}",
        "",
        "eq3prime and eq4plus include the k-sigma coverage term and are COVERAGE",
        "statements, not bounds. See telemgap.policy equation (3').",
        "",
        format_table(
            [
                "channel",
                "M2",
                "source",
                "noise floor",
                "measurable",
                "hf fraction",
                "resolvable",
                "sigma",
                "sigma src",
                "estimator bandwidth Hz",
            ],
            [
                [
                    name,
                    _number(estimate.value, 4),
                    estimate.source,
                    _number(estimate.noise_floor, 4),
                    "yes" if estimate.measurable else "NO",
                    _number(estimate.high_frequency_fraction, 4),
                    "yes" if estimate.resolvable else "NO",
                    _number(estimate.sigma, 4),
                    estimate.sigma_source,
                    _number(estimate.bandwidth_hz, 4),
                ]
                for name, estimate in plan.curvatures.items()
            ],
        ),
        "",
        format_table(["quantity", "value"], [[key, str(value)] for key, value in summary.items()]),
        "",
        f"first {len(rows)} of {len(plan.decisions)} decisions:",
        "",
        format_table(header, rows),
    ]
    if plan.refusals:
        lines.extend(["", "refusals, with the gate that refused them:", ""])
        for decision in plan.refusals[:max_rows]:
            lines.append(
                f"  gap {decision.gap_index} {decision.channel_name}: {decision.reason}"
            )
    return "\n".join(lines)


def render_scores(table: ScoreTable) -> str:
    """Per-class precision, recall and undecided rate, equations (13) to (15)."""
    rows = [
        [
            score.name,
            str(score.support),
            str(score.predicted),
            str(score.true_positive),
            _number(score.precision, 4),
            _number(score.recall, 4),
            _number(score.f1, 4),
            str(score.undecided),
            _number(score.undecided_rate, 4),
            _number(score.surfaced_rate, 4),
        ]
        for score in table.scores
    ]
    header = [
        "class",
        "support",
        "predicted",
        "tp",
        "precision",
        "recall",
        "f1",
        "undecided",
        "undec_rate",
        "surfaced",
    ]
    confusion_rows = [
        [CLASS_NAMES[row]] + [str(int(value)) for value in table.confusion[row]]
        for row in range(len(CLASS_NAMES))
    ]
    return "\n".join(
        [
            f"{table.label}  population={table.population}  "
            f"decided={table.decided_fraction:.4f}  "
            f"accuracy_on_decided={_number(table.accuracy_decided, 4)}",
            "",
            format_table(header, rows),
            "",
            "confusion (rows true, columns predicted):",
            "",
            format_table(
                ["true \\ pred", *CLASS_NAMES, "undecided"], confusion_rows
            ),
        ]
    )
