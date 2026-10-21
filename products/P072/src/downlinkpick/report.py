"""Text rendering of a selection, its price and its left-behind report.

Every function here returns a string. Nothing in this module prints, because
printing from a library makes it unusable inside anything else; the CLI in
:mod:`downlinkpick.__main__` is the only place in the package that writes to a
stream.
"""

from __future__ import annotations

from .marginal import LeftBehind, WindowMarginal, value_left_behind
from .selection import Selection, SelectionProblem, check_invariants


def _bits(value: float) -> str:
    return f"{value / 1.0e9:10.4f} Gbit"


def format_plan(problem: SelectionProblem) -> str:
    """One line per contact window: times, duration, capacity, reliability."""
    lines = [
        f"{'window':>10s} {'station':>8s} {'start [s]':>12s} {'end [s]':>12s} "
        f"{'dur [s]':>9s} {'capacity':>16s} {'reliab':>7s}"
    ]
    for window in problem.plan:
        lines.append(
            f"{window.name:>10s} {window.station:>8s} {window.start_s:12.1f} "
            f"{window.end_s:12.1f} {window.duration_s:9.1f} "
            f"{_bits(window.capacity_bits()):>16s} {window.reliability:7.3f}"
        )
    lines.append(
        f"{'total':>10s} {'':>8s} {'':>12s} {'':>12s} "
        f"{problem.plan.total_contact_s():9.1f} "
        f"{_bits(problem.plan.total_capacity_bits()):>16s}"
    )
    return "\n".join(lines)


def format_selection(problem: SelectionProblem, selection: Selection) -> str:
    """The selection itself: product, window, size, value, density."""
    lines = [
        f"method {selection.method}, status {selection.status}, "
        f"value source {problem.value_source}",
        f"{'product':>12s} {'window':>8s} {'size':>16s} {'value':>10s} "
        f"{'value/Gbit':>12s}",
    ]
    for assignment in selection.assignments:
        density = assignment.value / (assignment.bits / 1.0e9)
        lines.append(
            f"{assignment.product:>12s} {assignment.window:>8s} "
            f"{_bits(assignment.bits):>16s} {assignment.value:10.4f} {density:12.4f}"
        )
    lines.append(
        f"{'selected':>12s} {len(selection.assignments):8d} "
        f"{_bits(selection.total_bits()):>16s} {selection.total_value:10.4f}"
    )
    used = selection.bits_by_window()
    capacity = problem.capacity_bits()
    lines.append("")
    lines.append(f"{'window':>10s} {'committed':>16s} {'capacity':>16s} {'spare':>16s}")
    for name in problem.plan.names:
        spare = capacity[name] - used.get(name, 0.0)
        lines.append(
            f"{name:>10s} {_bits(used.get(name, 0.0)):>16s} "
            f"{_bits(capacity[name]):>16s} {_bits(spare):>16s}"
        )
    return "\n".join(lines)


def format_left_behind(rows: tuple[LeftBehind, ...]) -> str:
    """The left-behind report, most valuable first, with reasons and shortfalls."""
    if not rows:
        return "nothing left behind: every product was selected"
    lines = [
        f"{'product':>12s} {'value':>10s} {'size':>16s} {'reason':>11s} "
        f"{'window':>8s} {'short [Gbit]':>13s} {'short [s]':>11s}"
    ]
    for row in rows:
        window = row.best_window or "-"
        seconds = "inf" if row.extra_seconds_needed == float("inf") else (
            f"{row.extra_seconds_needed:11.2f}"
        )
        lines.append(
            f"{row.product:>12s} {row.value:10.4f} {_bits(row.bits):>16s} "
            f"{row.reason:>11s} {window:>8s} {row.extra_bits_needed / 1.0e9:13.5f} "
            f"{seconds:>11s}"
        )
    totals = value_left_behind(rows)
    lines.append("")
    lines.append("value left behind by reason [value units]")
    for reason, total in totals.items():
        lines.append(f"  {reason:>11s} {total:10.4f}")
    lines.append(f"  {'all':>11s} {sum(totals.values()):10.4f}")
    return "\n".join(lines)


def format_marginals(marginals: tuple[WindowMarginal, ...]) -> str:
    """Marginal value of contact time per window."""
    if not marginals:
        return "no windows priced"
    step = marginals[0].delta_s
    lines = [
        f"finite-difference step {step:.3f} s, rate held at the window's final value",
        "a window's end is also its delivery epoch, so an extension can push a product",
        "past its expiry; the expiry-crossing column explains a zero or negative price",
        f"{'window':>10s} {'+bits [Gbit]':>13s} {'+value':>10s} {'value/s':>12s} "
        f"{'value/Gbit':>12s} {'crossings':>9s} entering / leaving",
    ]
    for marginal in marginals:
        per_gbit = (
            "-" if marginal.value_per_gbit is None else f"{marginal.value_per_gbit:12.4f}"
        )
        movement = (
            ", ".join(marginal.entering) + " / " + ", ".join(marginal.leaving)
            if (marginal.entering or marginal.leaving)
            else "-"
        )
        lines.append(
            f"{marginal.window:>10s} {marginal.delta_bits / 1.0e9:13.5f} "
            f"{marginal.delta_value:10.4f} {marginal.value_per_second:12.6f} "
            f"{per_gbit:>12s} {len(marginal.expiry_crossings):9d} {movement}"
        )
    crossed = sorted({name for m in marginals for name in m.expiry_crossings})
    if crossed:
        lines.append("")
        lines.append("products whose expiry falls inside an extension: " + ", ".join(crossed))
    return "\n".join(lines)


def format_invariants(problem: SelectionProblem, selection: Selection) -> str:
    """The invariant report, including the two residual statistics."""
    report = check_invariants(problem, selection)
    lines = [
        f"invariants: {'all hold' if report.ok else str(len(report.violations)) + ' violated'}",
        f"  worst capacity overshoot : {report.worst_capacity_overshoot_bits:+.6e} bits",
        f"  value residual (relative): {report.value_residual_rel:.3e}",
    ]
    for violation in report.violations:
        lines.append(f"  VIOLATION {violation}")
    return "\n".join(lines)


def format_full_report(
    problem: SelectionProblem,
    selection: Selection,
    marginals: tuple[WindowMarginal, ...],
    left_behind: tuple[LeftBehind, ...],
) -> str:
    """Plan, selection, left-behind report, marginal prices and invariants."""
    blocks = [
        "contact plan",
        format_plan(problem),
        "",
        "selection",
        format_selection(problem, selection),
        "",
        "left behind",
        format_left_behind(left_behind),
        "",
        "marginal value of contact time",
        format_marginals(marginals),
        "",
        format_invariants(problem, selection),
    ]
    return "\n".join(blocks)
