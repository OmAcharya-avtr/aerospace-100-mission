"""Rendering. Every function returns a string; nothing in this module writes to a stream.

The command-line interface in :mod:`activityplan.__main__` is the only writer in
the package. Keeping the rendering separate means the feasibility report -- the
product's primary deliverable -- can be tested against its exact text, embedded
in a document, or diffed between runs.
"""

from __future__ import annotations

import math

from .feasibility import FeasibilityReport
from .greedy import GreedyResult
from .model import Problem
from .timeline import Plan, RasterisedProblem

__all__ = [
    "render_declaration",
    "render_feasibility",
    "render_plan",
    "render_rejections",
    "render_table",
]

_RULE = "-" * 78


def render_table(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> str:
    """A fixed-width text table. Columns are sized to their widest cell."""
    if not headers:
        raise ValueError("a table needs at least one column")
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(
                f"row {row!r} has {len(row)} cells but there are {len(headers)} headers"
            )
    widths = [len(header) for header in headers]
    for row in rows:
        for position, cell in enumerate(row):
            widths[position] = max(widths[position], len(cell))
    lines = ["  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)).rstrip()]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return "\n".join(lines)


def render_declaration(problem: Problem, step_s: float) -> str:
    """The declared problem, as a block of text. No plan involved."""
    resources = problem.resources
    generated, fixed = problem.baseline_energy_balance_wh()
    lines = [
        f"instance            : {problem.name}",
        f"horizon             : {problem.horizon_s:.1f} s "
        f"({problem.horizon_s / resources.solar.orbit_period_s:.2f} declared orbit periods)",
        f"grid step           : {step_s:.1f} s -> {int(round(problem.horizon_s / step_s))} steps",
        f"activities          : {problem.n_activities}, total priority "
        f"{problem.total_priority:.1f}",
        f"precedences         : {len(problem.precedences)}",
        f"exclusions          : {len(problem.exclusions)}",
        f"require empty rec.  : {problem.require_empty_recorder}",
        "",
        f"battery             : {resources.battery_capacity_wh:.1f} W*h nameplate, "
        f"DoD limit {resources.max_depth_of_discharge:.3f}, floor {resources.floor_wh:.2f} W*h, "
        f"usable {resources.usable_wh:.2f} W*h",
        f"initial state       : {resources.initial_state_of_charge:.3f} of nameplate "
        f"= {resources.initial_energy_wh:.2f} W*h",
        f"bus load            : {resources.bus_load_w:.1f} W over the whole horizon",
        f"array               : peak {resources.solar.peak_power_w:.1f} W, period "
        f"{resources.solar.orbit_period_s:.1f} s, eclipse fraction "
        f"{resources.solar.eclipse_fraction:.3f}, phase {resources.solar.phase_s:.1f} s",
        f"recorder            : {resources.recorder_capacity_mbit:.1f} Mbit, initially "
        f"{resources.initial_recorder_mbit:.1f} Mbit",
        f"contacts            : {len(resources.contacts)}, total capacity "
        f"{resources.total_contact_capacity_mbit:.1f} Mbit",
        "",
        f"energy generated    : {generated:.3f} W*h (closed form over the horizon)",
        f"energy fixed load   : {fixed:.3f} W*h (bus plus transmitters)",
        f"energy if all placed: {problem.total_activity_energy_wh:.3f} W*h of activity demand",
        f"data if all placed  : {problem.total_data_mbit:.1f} Mbit",
    ]
    return "\n".join(lines)


def render_plan(plan: Plan, rasterised: RasterisedProblem) -> str:
    """The timeline as a table of placed activities, earliest start first."""
    problem = rasterised.problem
    rows: list[tuple[str, ...]] = []
    for name in plan.placed:
        index = problem.index(name)
        activity = problem.activity(name)
        start_step = plan.starts[name]
        start_s = start_step * rasterised.grid.step_s
        finish_s = rasterised.finish_s(index, start_step)
        rows.append(
            (
                name,
                str(start_step),
                f"{start_s:.1f}",
                f"{finish_s:.1f}",
                f"{activity.duration_s:.1f}",
                f"{activity.energy_wh:.3f}",
                f"{activity.peak_power_w:.1f}",
                f"{activity.data_mbit:.1f}",
                f"{activity.priority:.1f}",
            )
        )
    table = render_table(
        (
            "activity",
            "step",
            "start/s",
            "finish/s",
            "dur/s",
            "energy/Wh",
            "peak/W",
            "data/Mbit",
            "priority",
        ),
        tuple(rows),
    )
    unplaced = tuple(
        activity.name for activity in problem.activities if activity.name not in plan.starts
    )
    footer = [
        "",
        f"placed   : {plan.n_placed} of {problem.n_activities}, "
        f"priority {plan.value(problem):.1f} of {problem.total_priority:.1f}",
        f"not placed: {', '.join(unplaced) if unplaced else 'none'}",
    ]
    return table + "\n" + "\n".join(footer)


def render_feasibility(report: FeasibilityReport, rasterised: RasterisedProblem) -> str:
    """The feasibility report: the binding constraint and the slack on everything else.

    This is the product's primary deliverable. The constraints are listed in
    increasing normalised slack, so the first line is the binding one, and every
    entry carries its physical slack, its unit, its normaliser and where it is
    tightest. Diagnostics that cannot be violated by construction are listed
    separately below the constraints and are never named as binding.
    """
    binding = report.binding
    verdict = "FEASIBLE" if report.feasible else "INFEASIBLE"
    lines = [
        _RULE,
        f"feasibility report  : {rasterised.problem.name}",
        _RULE,
        f"verdict             : {verdict}",
        f"objective           : {report.value:.1f} of "
        f"{rasterised.problem.total_priority:.1f} priority units placed "
        f"({report.plan.n_placed} of {rasterised.problem.n_activities} activities)",
        f"forgone             : {report.forgone_priority:.1f} priority units on "
        f"{len(report.unplaced)} activities: "
        f"{', '.join(report.unplaced) if report.unplaced else 'none'}",
        f"BINDING CONSTRAINT  : {binding.name}",
        f"                      slack {binding.slack:.6g} {binding.unit} "
        f"= {binding.normalised:+.6f} of {binding.scale:.6g} {binding.unit}",
        f"                      at {binding.where}",
    ]
    resource = report.binding_resource
    if resource is not None and resource.name != binding.name:
        lines.append(
            f"tightest resource   : {resource.name}, slack {resource.slack:.6g} "
            f"{resource.unit} = {resource.normalised:+.6f} of its normaliser"
        )
    if report.unplaceable:
        lines.append(
            f"unplaceable         : {', '.join(report.unplaceable)} -- no admissible "
            "start step exists, which is a property of the declaration, not of the plan"
        )
    if not report.feasible:
        lines.append(f"violations          : {len(report.violations)}")

    candidates = sorted(report.candidates, key=lambda entry: entry.normalised)
    rows = tuple(
        (
            entry.name,
            f"{entry.slack:.6g}",
            entry.unit,
            f"{entry.scale:.6g}",
            f"{entry.normalised:+.6f}",
            "ok" if entry.satisfied else "VIOLATED",
            entry.where,
        )
        for entry in candidates
    )
    lines += [
        "",
        "constraints, tightest normalised slack first",
        render_table(
            ("constraint", "slack", "unit", "normaliser", "normalised", "status", "where"), rows
        ),
    ]

    diagnostics = tuple(entry for entry in report.slacks if not entry.binding_candidate)
    if diagnostics:
        rows = tuple(
            (
                entry.name,
                f"{entry.slack:.6g}",
                entry.unit,
                f"{entry.normalised:+.6f}",
                entry.where,
            )
            for entry in diagnostics
        )
        lines += [
            "",
            "diagnostics, never named as binding (cannot be violated by construction)",
            render_table(("diagnostic", "slack", "unit", "normalised", "where"), rows),
        ]

    audit = report.energy_audit()
    lines += [
        "",
        "energy audit, W*h",
        render_table(
            ("quantity", "value"),
            tuple((key, f"{value:.9g}") for key, value in audit.items()),
        ),
        "",
        "the two generated entries compare the whole-horizon closed-form solar integral",
        "against the sum of the per-step closed-form integrals; residual is the closure",
        "of the battery recursion. Both are reported, not asserted.",
        _RULE,
    ]
    return "\n".join(lines)


def render_rejections(result: GreedyResult) -> str:
    """Why the greedy planner dropped each activity it could not place."""
    if not result.rejections:
        return "greedy planner placed every activity."
    rows = tuple(
        (
            record.name,
            str(record.candidates_tried),
            record.dominant_reason,
            ", ".join(f"{name} x{count}" for name, count in record.binding_counts) or "-",
        )
        for record in result.rejections
    )
    return render_table(
        ("dropped", "starts tried", "dominant binding constraint", "binding breakdown"), rows
    )


def format_gap(greedy_value: float, exact_value: float) -> str:
    """``greedy / exact`` as a percentage shortfall, or a note when the optimum is zero."""
    if exact_value <= 0.0:
        return "exact optimum is 0.0; a relative gap is undefined"
    shortfall = (exact_value - greedy_value) / exact_value
    if not math.isfinite(shortfall):
        return "not finite"
    return f"{100.0 * shortfall:.2f} % below the exact optimum"
