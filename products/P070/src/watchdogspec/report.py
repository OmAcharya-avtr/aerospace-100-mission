"""Rendering a :class:`watchdogspec.checks.Report` as text, JSON or markdown.

No numbers are computed here. Every figure printed comes from
:mod:`watchdogspec.checks`, :mod:`watchdogspec.latency` or
:mod:`watchdogspec.falsetrip`, and the units are the ones those modules
document: milliseconds for time, dimensionless for probabilities and costs.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .checks import CHECKS, Report, Severity

#: Printed under every verdict. It is the one sentence a reader of a clean
#: report most needs and is least likely to supply for themselves.
SCOPE_NOTE = (
    "A consistent verdict means the declared numbers do not contradict each other. "
    "It is not a measurement of any running system, and it is not evidence that the "
    "declaration matches one."
)


def _ms(value: float) -> str:
    if value == math.inf:
        return "inf"
    return f"{value:.4f}"


def _rate(value: float) -> str:
    if value == 0.0:
        return "0"
    if value == math.inf:
        return "inf"
    return f"{value:.6g}"


def format_timing_table(report: Report) -> str:
    """One row per task: window, detection latency, ladder latency, slack, deepest rung."""
    header = (
        f"  {'task':<14} {'W/ms':>9} {'k*W/ms':>9} {'WCDL/ms':>10} {'ladder/ms':>11} "
        f"{'D/ms':>10} {'slack/ms':>11} {'rung':>5}"
    )
    lines = ["worst-case timing, all from the last accepted heartbeat", header]
    for task_id in sorted(report.timings):
        timing = report.timings[task_id]
        lines.append(
            f"  {task_id:<14} {_ms(timing.beat_window_ms):>9} "
            f"{_ms(timing.tolerated_silence_ms):>9} {_ms(timing.detection_latency_ms):>10} "
            f"{_ms(timing.full_ladder_latency_ms):>11} {_ms(timing.deadline_ms):>10} "
            f"{_ms(timing.ladder_slack_ms):>11} "
            f"{timing.deepest_rung_reachable:>3}/{len(timing.rungs):<1}"
        )
    if not report.timings:
        lines.append("  (no task has a declared escalation ladder)")
    lines.append(
        "  W = P + J beat window; k*W tolerated silence; WCDL worst-case detection "
        "latency;"
    )
    lines.append(
        "  ladder = WCDL + every rung; slack = D - ladder; rung = deepest rung reachable."
    )
    return "\n".join(lines)


def format_false_trip_table(report: Report) -> str:
    """One row per task: trip threshold, sigma distance, per-gap probability, per-hour rate."""
    header = (
        f"  {'task':<14} {'threshold/ms':>13} {'z/sigma':>9} {'p per gap':>12} "
        f"{'per hour':>12} {'MTBFT/h':>12}"
    )
    lines = ["false-trip rate from the declared jitter statistics", header]
    for task_id in sorted(report.false_trips):
        rate = report.false_trips[task_id]
        lines.append(
            f"  {task_id:<14} {_ms(rate.threshold_ms):>13} {_rate(rate.z):>9} "
            f"{_rate(rate.per_gap):>12} {_rate(rate.per_hour):>12} "
            f"{_rate(rate.mtbft_hours):>12}"
        )
    if not report.false_trips:
        lines.append("  (no task has a declared escalation ladder)")
    lines.append(
        "  sigma = 0 is a declaration that the jitter bound is hard; the rate is then "
        "exactly 0."
    )
    lines.append(
        "  The Gaussian tail and the bounded worst case are different models and both "
        "are reported."
    )
    return "\n".join(lines)


def format_findings(report: Report) -> str:
    """Findings in report order, then a count per code."""
    lines = ["findings"]
    if not report.findings:
        lines.append("  none")
    for finding in report.findings:
        lines.append(f"  {finding.code}  {finding.severity.value:<7} {finding.message}")
    counts = report.count_by_code()
    if counts:
        lines.append("")
        lines.append("findings per code")
        for code in CHECKS:
            if code in counts:
                lines.append(f"  {code}  {counts[code]:>3}  {CHECKS[code].summary}")
    return "\n".join(lines)


def format_report(report: Report) -> str:
    """The full text report: header, timing, false trips, findings, verdict."""
    severities = report.count_by_severity()
    errors = severities.get(Severity.ERROR.value, 0)
    warnings = severities.get(Severity.WARNING.value, 0)
    verdict = "CONSISTENT" if report.is_consistent else "INCONSISTENT"
    parts = [
        f"watchdogspec report for {report.system}",
        f"  source            : {report.spec_source}",
        f"  tasks timed       : {len(report.timings)}",
        f"  comparison tol    : {report.tol_ms:.3e} ms",
        "",
        format_timing_table(report),
        "",
        format_false_trip_table(report),
        "",
        format_findings(report),
        "",
        "verdict",
        f"  {errors} error, {warnings} warning finding(s)",
        f"  declaration verdict: {verdict}. Exit code {report.exit_code}.",
        "",
        f"  {SCOPE_NOTE}",
    ]
    return "\n".join(parts) + "\n"


def report_to_dict(report: Report) -> dict[str, Any]:
    """The whole report as plain Python types, suitable for :func:`json.dumps`."""
    return {
        "system": report.system,
        "source": report.spec_source,
        "tol_ms": report.tol_ms,
        "exit_code": report.exit_code,
        "consistent": report.is_consistent,
        "findings": [
            {
                "code": finding.code,
                "severity": finding.severity.value,
                "subject": finding.subject,
                "message": finding.message,
            }
            for finding in report.findings
        ],
        "timings": {
            task_id: {
                "beat_window_ms": timing.beat_window_ms,
                "tolerated_silence_ms": timing.tolerated_silence_ms,
                "detection_latency_ms": timing.detection_latency_ms,
                "deadline_ms": timing.deadline_ms,
                "detection_slack_ms": timing.detection_slack_ms,
                "detection_margin": timing.detection_margin.value,
                "full_ladder_latency_ms": timing.full_ladder_latency_ms,
                "ladder_slack_ms": timing.ladder_slack_ms,
                "ladder_margin": timing.ladder_margin.value,
                "deepest_rung_reachable": timing.deepest_rung_reachable,
                "total_cost": timing.total_cost,
                "rungs": [
                    {
                        "index": rung.index,
                        "action": rung.action,
                        "start_ms": rung.start_ms,
                        "finish_ms": rung.finish_ms,
                        "slack_ms": rung.slack_ms,
                        "margin": rung.margin.value,
                        "cumulative_cost": rung.cumulative_cost,
                    }
                    for rung in timing.rungs
                ],
            }
            for task_id, timing in report.timings.items()
        },
        "false_trips": {
            task_id: {
                "per_gap": rate.per_gap,
                "per_hour": rate.per_hour,
                "mtbft_hours": None if rate.mtbft_hours == math.inf else rate.mtbft_hours,
                "z": None if rate.z == math.inf else rate.z,
                "threshold_ms": rate.threshold_ms,
                "exact_zero": rate.exact,
            }
            for task_id, rate in report.false_trips.items()
        },
        "scope_note": SCOPE_NOTE,
    }


def format_json(report: Report) -> str:
    """Indented, sorted-key JSON of :func:`report_to_dict`."""
    return json.dumps(report_to_dict(report), indent=2, sort_keys=True) + "\n"


def format_markdown(report: Report) -> str:
    """Markdown tables, for pasting into a review comment."""
    lines = [
        f"# watchdogspec: {report.system}",
        "",
        f"Source: `{report.spec_source}`. Comparison tolerance {report.tol_ms:.3e} ms.",
        "",
        "## Worst-case timing (ms, from the last accepted heartbeat)",
        "",
        "| task | W | k*W | WCDL | ladder | D | slack | deepest rung |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for task_id in sorted(report.timings):
        timing = report.timings[task_id]
        lines.append(
            f"| {task_id} | {_ms(timing.beat_window_ms)} | "
            f"{_ms(timing.tolerated_silence_ms)} | {_ms(timing.detection_latency_ms)} | "
            f"{_ms(timing.full_ladder_latency_ms)} | {_ms(timing.deadline_ms)} | "
            f"{_ms(timing.ladder_slack_ms)} | "
            f"{timing.deepest_rung_reachable}/{len(timing.rungs)} |"
        )
    lines += [
        "",
        "## False-trip rate",
        "",
        "| task | threshold (ms) | z (sigma) | p per gap | per hour |",
        "|---|---|---|---|---|",
    ]
    for task_id in sorted(report.false_trips):
        rate = report.false_trips[task_id]
        lines.append(
            f"| {task_id} | {_ms(rate.threshold_ms)} | {_rate(rate.z)} | "
            f"{_rate(rate.per_gap)} | {_rate(rate.per_hour)} |"
        )
    lines += ["", "## Findings", ""]
    if not report.findings:
        lines.append("None.")
    else:
        lines += ["| code | severity | subject | message |", "|---|---|---|---|"]
        for finding in report.findings:
            lines.append(
                f"| {finding.code} | {finding.severity.value} | {finding.subject} | "
                f"{finding.message} |"
            )
    severities = report.count_by_severity()
    lines += [
        "",
        f"**{severities.get('error', 0)} error, {severities.get('warning', 0)} warning. "
        f"Exit code {report.exit_code}.**",
        "",
        SCOPE_NOTE,
        "",
    ]
    return "\n".join(lines)


def format_codes() -> str:
    """The check catalogue, one line per code."""
    lines = ["watchdogspec checks"]
    for code, spec in CHECKS.items():
        lines.append(f"  {code}  {spec.severity.value:<7} {spec.summary}")
    return "\n".join(lines) + "\n"
