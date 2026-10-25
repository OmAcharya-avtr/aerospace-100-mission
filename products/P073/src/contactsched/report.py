"""Text rendering of schedules, contention maps and fairness reports.

Every function here returns a string. Nothing in this package prints; the CLI in
:mod:`contactsched.__main__` is the only thing that writes to a stream, so the
same renderers are used by the tests, the validation scripts and the command
line and cannot drift apart.

Times are rendered as ``HH:MM:SS`` offsets from the scenario epoch rather than as
calendar times. An offset is unambiguous without stating a time standard, and
this package does not model the UT1 against UTC difference well enough to
justify printing an absolute timestamp to the second.
"""

from __future__ import annotations

from .assignment import Assignment
from .contention import ContentionCell
from .fairness import FairnessReport
from .problem import SchedulingProblem
from .visibility import ContactWindow


def hms(seconds: float) -> str:
    """Render a non-negative offset in seconds as ``HH:MM:SS``.

    Offsets beyond 24 hours keep counting in the hours field rather than
    wrapping, because a scheduling horizon longer than a day is ordinary and a
    wrapped hour would be read as the wrong day.
    """
    total = int(round(float(seconds)))
    sign = "-" if total < 0 else ""
    total = abs(total)
    return f"{sign}{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def format_windows(windows: list[ContactWindow], limit: int | None = None) -> str:
    """Table of visibility windows."""
    lines = [
        f"{'antenna':<14} {'spacecraft':<12} {'rise':>9} {'set':>9} "
        f"{'dur_s':>8} {'peak_el':>8}  flags",
        "-" * 78,
    ]
    shown = windows if limit is None else windows[: int(limit)]
    for window in shown:
        flags = []
        if window.clipped_start:
            flags.append("clipped-start")
        if window.clipped_end:
            flags.append("clipped-end")
        if window.mask_limited:
            flags.append("mask-limited")
        lines.append(
            f"{window.station:<14} {window.spacecraft:<12} {hms(window.rise_s):>9} "
            f"{hms(window.set_s):>9} {window.duration_s:8.1f} "
            f"{window.max_elevation_deg:8.2f}  {','.join(flags)}"
        )
    if limit is not None and len(windows) > int(limit):
        lines.append(f"... {len(windows) - int(limit)} further windows not shown")
    lines.append(f"{len(windows)} windows, {sum(w.duration_s for w in windows):.1f} s total")
    return "\n".join(lines)


def format_schedule(problem: SchedulingProblem, assignment: Assignment) -> str:
    """Table of the selected contacts, in time order."""
    lines = [
        f"method {assignment.method}, status {assignment.status}, "
        f"objective {assignment.objective:.6f} ({assignment.objective_mode})",
        f"{'start':>9} {'end':>9} {'antenna':<14} {'spacecraft':<12} {'request':<8} prio",
        "-" * 70,
    ]
    by_id = problem.request_by_id
    for start, end, station, spacecraft, request_id in assignment.timeline(problem):
        priority = by_id[request_id].priority
        lines.append(
            f"{hms(start):>9} {hms(end):>9} {station:<14} {spacecraft:<12} "
            f"{request_id:<8} {priority}"
        )
    lines.append(f"{assignment.contact_count} contacts scheduled")
    for note in assignment.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def format_fairness(report: FairnessReport) -> str:
    """The contention and fairness accounting, as a readable block."""
    lines = [
        f"fairness report for {report.method}",
        f"{'request':<8} {'mission':<13} {'prio':>4} {'need':>5} {'got':>4} "
        f"{'need_s':>8} {'got_s':>8} {'satisf':>7}  state",
        "-" * 78,
    ]
    for outcome in report.outcomes:
        state = "starved" if outcome.starved else ("no-demand" if not outcome.has_demand else "ok")
        lines.append(
            f"{outcome.request_id:<8} {outcome.mission:<13} {outcome.priority:>4} "
            f"{outcome.requested_contacts:>5} {outcome.granted_contacts:>4} "
            f"{outcome.requested_seconds:8.0f} {outcome.allocated_seconds:8.0f} "
            f"{outcome.satisfaction:7.4f}  {state}"
        )
    lines.append("")
    lines.append(f"per-request Jain index        : {report.jain:.6f}")
    lines.append(f"per-mission Jain index        : {report.jain_by_mission:.6f}")
    lines.append(f"max-min share                 : {report.max_min:.6f}")
    lines.append(f"requests starved              : {len(report.starved)}")
    lines.append(f"total seconds short           : {report.total_seconds_short:.1f}")
    lines.append(
        f"priority inversions           : {report.inversion_count} of "
        f"{report.comparable_pairs} comparable pairs (rate {report.inversion_rate:.6f})"
    )
    lines.append(f"worst inversion gap           : {report.worst_inversion_gap:.6f}")
    lines.append(
        "delivered satisfaction by priority: "
        + ", ".join(f"p{p}={v:.4f}" for p, v in sorted(report.delivered_by_priority.items()))
    )
    lines.append(f"priority promise held in aggregate: {report.priority_monotone}")
    if report.all_starved:
        lines.append(
            "every request with declared demand received nothing; the Jain index of "
            "1.0 above is the degenerate all-zero convention and the max-min share "
            "is the number to read"
        )
    if report.inversions:
        lines.append("")
        lines.append("inversions:")
        for inversion in report.inversions:
            lines.append(
                f"  {inversion.higher_request} (p{inversion.higher_priority}, "
                f"satisfaction {inversion.higher_satisfaction:.4f}) below "
                f"{inversion.lower_request} (p{inversion.lower_priority}, "
                f"satisfaction {inversion.lower_satisfaction:.4f}), gap "
                f"{inversion.gap:.4f}"
            )
    return "\n".join(lines)


def format_contention(cells: list[ContentionCell], limit: int | None = 20) -> str:
    """Table of the most contended antenna-time bins."""
    ordered = sorted(
        cells,
        key=lambda c: (-(c.ratio if c.ratio != float("inf") else 1e18), c.station, c.bin_start_s),
    )
    shown = ordered if limit is None else ordered[: int(limit)]
    lines = [
        f"{'antenna':<14} {'bin start':>9} {'bin end':>9} {'supply_s':>9} "
        f"{'demand_s':>9} {'ratio':>9}  requests",
        "-" * 86,
    ]
    for cell in shown:
        ratio = "inf" if cell.ratio == float("inf") else f"{cell.ratio:9.4f}"
        lines.append(
            f"{cell.station:<14} {hms(cell.bin_start_s):>9} {hms(cell.bin_end_s):>9} "
            f"{cell.supply_s:9.1f} {cell.demand_s:9.1f} {ratio:>9}  "
            f"{','.join(cell.requests)}"
        )
    lines.append(
        f"{len(cells)} cells, "
        f"{sum(1 for c in cells if c.ratio > 1.0)} with demand above supply"
    )
    return "\n".join(lines)


def format_backends() -> str:
    """What solver backends this interpreter can run, as a measured block.

    Printed by ``python -m contactsched backends``. It is the measurement behind
    the statement in README.md that ``pulp`` was evaluated and rejected, in a
    form a reader can reproduce on their own machine in one command.
    """
    from .exact import available_backends

    lines = ["solver backends detected in this interpreter:"]
    for name, detail in sorted(available_backends().items()):
        lines.append(f"  {name:<14} {detail}")
    lines.append("")
    lines.append(
        "HiGHS through scipy.optimize.milp is the default and the only backend any "
        "test or validation script in this repository exercises."
    )
    return "\n".join(lines)
