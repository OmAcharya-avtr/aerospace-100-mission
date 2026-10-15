"""Render the declared control structure.

Two renderers from the same declaration:

* :func:`render_mermaid` emits a Mermaid flowchart, which GitHub renders
  natively, so a control structure is reviewable in a pull request as text.
* :func:`render_control_structure_png` draws the classic STPA control-structure
  figure with matplotlib (Agg backend only, never ``show()``): controller at
  the top, actuator and sensor on the two sides, controlled process at the
  bottom, control actions down the left, feedback up the right.

Both are deterministic: declaration order in, same bytes out.
"""

from __future__ import annotations

import os
from pathlib import Path

from .checks import Report
from .enumeration import CellState
from .model import Analysis, ComponentKind

_MERMAID_SHAPE = {
    ComponentKind.CONTROLLER: ("[", "]"),
    ComponentKind.ACTUATOR: ("([", "])"),
    ComponentKind.SENSOR: ("([", "])"),
    ComponentKind.PROCESS: ("[[", "]]"),
}


def _escape(text: str, max_chars: int) -> str:
    """Trim and strip characters Mermaid treats as syntax."""
    clean = text.replace('"', "'").replace("\n", " ").strip()
    if len(clean) > max_chars:
        clean = clean[: max_chars - 1].rstrip() + "…"
    return clean


def render_mermaid(
    analysis: Analysis, report: Report | None = None, max_label_chars: int = 44
) -> str:
    """Mermaid ``flowchart TD`` source for the declared control structure.

    When ``report`` is supplied, a control action carrying an enumeration gap
    is given the ``gap`` class, so the picture and the finding list cannot
    disagree.
    """
    lines = ["flowchart TD"]
    for component in analysis.structure.components:
        open_shape, close_shape = _MERMAID_SHAPE[component.kind]
        label = _escape(f"{component.id}: {component.name}", max_label_chars)
        lines.append(f'    {component.id}{open_shape}"{label}"{close_shape}')

    gap_actions: set[str] = set()
    if report is not None:
        gap_actions = {cell.control_action for cell in report.grid.gaps}

    for action in analysis.structure.control_actions:
        label = _escape(f"{action.id}: {action.name}", max_label_chars)
        suffix = "  %% enumeration gap" if action.id in gap_actions else ""
        lines.append(f'    {action.controller} -->|"{label}"| {action.actuator}{suffix}')
        lines.append(f"    {action.actuator} --> {action.process}")
    for path in analysis.structure.feedback:
        label = _escape(f"{path.id}: {path.name}", max_label_chars)
        lines.append(f"    {path.process} --> {path.sensor}")
        lines.append(f'    {path.sensor} -.->|"{label}"| {path.controller}')
    return "\n".join(lines) + "\n"


def render_control_structure_png(
    report: Report, path: str | os.PathLike[str], dpi: int = 150
) -> Path:
    """Draw the control structure and the enumeration grid into one PNG.

    Parameters
    ----------
    report : Report
        A report, so that enumeration gaps can be marked on the figure.
    path : path-like
        Output PNG path. Parent directories are created.
    dpi : int
        Output resolution in dots per inch.

    Returns
    -------
    Path
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    analysis = report.analysis
    structure = analysis.structure
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.0, 6.0), width_ratios=[1.1, 1.0])

    def box(axis, x, y, w, h, title, body, colour):
        axis.add_patch(
            Rectangle(
                (x, y), w, h, facecolor=colour, edgecolor="#333333", linewidth=1.2
            )
        )
        axis.text(
            x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=9,
            fontweight="bold",
        )
        axis.text(
            x + w / 2, y + h * 0.3, body, ha="center", va="center", fontsize=7.5,
            wrap=True,
        )

    def arrow(axis, start, end, text, style, colour):
        axis.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.2,
                linestyle=style, color=colour, shrinkA=0, shrinkB=0,
            )
        )
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        axis.text(
            mid[0], mid[1], text, fontsize=7, ha="center", va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        )

    controllers = structure.by_kind(ComponentKind.CONTROLLER)
    actuators = structure.by_kind(ComponentKind.ACTUATOR)
    sensors = structure.by_kind(ComponentKind.SENSOR)
    processes = structure.by_kind(ComponentKind.PROCESS)

    left.set_xlim(0, 10)
    left.set_ylim(0, 10)
    left.axis("off")
    left.set_title(
        f"declared control structure\n{analysis.system or analysis.title}", fontsize=10
    )

    def names(items):
        return "\n".join(f"{c.id}" for c in items) or "(none declared)"

    box(left, 3.0, 8.0, 4.0, 1.6, "controller(s)", names(controllers), "#dce7f5")
    box(left, 0.3, 4.6, 3.2, 1.6, "actuator(s)", names(actuators), "#e6f0dc")
    box(left, 6.5, 4.6, 3.2, 1.6, "sensor(s)", names(sensors), "#f7ead2")
    box(left, 3.0, 1.0, 4.0, 1.6, "controlled process", names(processes), "#eeeeee")

    n_actions = len(structure.control_actions)
    n_feedback = len(structure.feedback)
    gap_actions = {cell.control_action for cell in report.grid.gaps}
    arrow(
        left, (4.0, 8.0), (1.9, 6.2),
        f"{n_actions} control action(s)\n{len(gap_actions)} with gaps",
        "solid", "#1f4e79",
    )
    arrow(left, (1.9, 4.6), (4.0, 2.6), "applied to process", "solid", "#1f4e79")
    if n_feedback:
        arrow(left, (6.0, 2.6), (8.1, 4.6), "measured", "dashed", "#8a5a00")
        arrow(left, (8.1, 6.2), (6.0, 8.0), f"{n_feedback} feedback path(s)", "dashed",
              "#8a5a00")
    else:
        left.text(
            7.5, 5.6, "no feedback path declared\n(open loop)", fontsize=8,
            color="#a00000", ha="center",
        )

    classes = ["not\nprovided", "provided", "wrong timing\nor order",
               "stopped soon /\napplied long"]
    marks = {
        CellState.UCA: ("U", "#2d6a2d"),
        CellState.EXCLUDED: ("x", "#666666"),
        CellState.CONFLICT: ("!", "#a00000"),
        CellState.GAP: ("GAP", "#a00000"),
    }
    actions = list(structure.control_actions)
    right.set_xlim(-0.5, 4.0)
    right.set_ylim(-0.6, max(1, len(actions)) - 0.4)
    right.set_xticks(range(4))
    right.set_xticklabels(classes, fontsize=7)
    right.set_yticks(range(len(actions)))
    right.set_yticklabels([a.id for a in actions], fontsize=8)
    right.invert_yaxis()
    covered, total = report.grid.completeness()
    right.set_title(
        f"UCA enumeration grid: {covered}/{total} cells covered", fontsize=10
    )
    for row in range(len(actions)):
        for col in range(4):
            grid_cell = report.grid.cells[row * 4 + col]
            text, colour = marks[grid_cell.state]
            right.add_patch(
                Rectangle(
                    (col - 0.45, row - 0.45), 0.9, 0.9,
                    facecolor="#f6dede" if grid_cell.state is CellState.GAP else "#eef3ee",
                    edgecolor="#999999", linewidth=0.8,
                )
            )
            label = text if grid_cell.state is not CellState.UCA else ",".join(
                grid_cell.uca_ids
            )
            right.text(
                col, row, label, ha="center", va="center", fontsize=6.5, color=colour,
            )
    right.grid(False)
    for spine in right.spines.values():
        spine.set_visible(False)
    right.tick_params(length=0)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(
        0.5, 0.02,
        "hazardtrace: declared structure and computed enumeration coverage. "
        "Research-grade; discharges no safety obligation.",
        fontsize=8, ha="center", color="#555555",
    )
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
