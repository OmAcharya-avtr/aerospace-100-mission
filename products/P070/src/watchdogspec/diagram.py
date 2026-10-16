"""Mermaid and matplotlib renderings of a declared specification.

Matplotlib is imported lazily and the Agg backend is selected before pyplot,
so importing :mod:`watchdogspec` never requires a display and never needs
matplotlib at all unless a PNG is asked for.
"""

from __future__ import annotations

from pathlib import Path

from .checks import Report
from .model import Spec

_ACTION_STYLE = {
    "log": "#8d99ae",
    "notify": "#6a8caf",
    "restart_task": "#3f7d3f",
    "safe_mode": "#c08a00",
    "reset": "#a00000",
}


def _safe(identifier: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in identifier)


def render_mermaid(spec: Spec, report: Report | None = None) -> str:
    """A ``flowchart LR`` of supervisor, tasks and ladders.

    When ``report`` is given, a task with an error finding is marked. The
    output is deterministic for a given specification.
    """
    flagged: set[str] = set()
    if report is not None:
        flagged = {
            finding.subject for finding in report.findings if finding.severity.value == "error"
        }
    lines = ["flowchart LR", f'    SUP["{spec.supervisor.name}<br/>poll '
             f'{spec.supervisor.poll_period_ms:g} ms"]']
    for task in spec.tasks:
        mark = " (error)" if task.id in flagged else ""
        label = (
            f"{task.id}{mark}<br/>P={task.heartbeat_period_ms:g} J={task.heartbeat_jitter_ms:g} "
            f"k={task.missed_beats_tolerated} D={task.liveness_deadline_ms:g}"
        )
        lines.append(f'    {_safe(task.id)}["{label}"]')
        lines.append(f"    {_safe(task.id)} -.->|heartbeat| SUP")
    for ladder in spec.ladders:
        rungs = " -> ".join(rung.action.value for rung in ladder.rungs)
        lines.append(f'    {_safe(ladder.id)}[["{ladder.id}<br/>{rungs}"]]')
    for task in spec.tasks:
        lines.append(f"    SUP -->|escalate| {_safe(task.escalation)}")
        break
    seen = {spec.tasks[0].escalation} if spec.tasks else set()
    for task in spec.tasks[1:]:
        if task.escalation in seen:
            continue
        seen.add(task.escalation)
        lines.append(f"    SUP -->|escalate| {_safe(task.escalation)}")
    return "\n".join(lines) + "\n"


def render_escalation_png(
    spec: Spec,
    report: Report,
    path: str | Path,
    dpi: int = 150,
) -> Path:
    """Draw every task's escalation timeline against its deadline.

    One horizontal band per timed task: detection, then one bar per rung,
    with the liveness deadline as a vertical line. Agg backend only; the
    figure is never shown. Parent directories are created.

    Raises
    ------
    ValueError
        If the report contains no timed task.
    """
    if not report.timings:
        raise ValueError("report has no timed task to draw")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ladders = {ladder.id: ladder for ladder in spec.ladders}
    task_ids = [task.id for task in spec.tasks if task.id in report.timings]
    height = max(2.2, 0.72 * len(task_ids) + 1.6)
    fig, axis = plt.subplots(figsize=(11.0, height))

    for row, task_id in enumerate(task_ids):
        timing = report.timings[task_id]
        task = spec.task(task_id)
        assert task is not None
        ladder = ladders[task.escalation]
        axis.barh(
            row,
            timing.detection_latency_ms,
            left=0.0,
            height=0.52,
            color="#2b3a55",
            edgecolor="white",
        )
        for rung, timed in zip(ladder.rungs, timing.rungs, strict=True):
            axis.barh(
                row,
                timed.finish_ms - timed.start_ms,
                left=timed.start_ms,
                height=0.52,
                color=_ACTION_STYLE[rung.action.value],
                edgecolor="white",
            )
        axis.plot(
            [timing.deadline_ms, timing.deadline_ms],
            [row - 0.36, row + 0.36],
            color="#a00000",
            linewidth=2.0,
            linestyle="--",
        )
        axis.text(
            timing.deadline_ms,
            row + 0.40,
            f"D={timing.deadline_ms:g}",
            fontsize=7,
            color="#a00000",
            ha="center",
            va="bottom",
        )

    axis.set_yticks(range(len(task_ids)))
    axis.set_yticklabels(
        [
            f"{task_id}  ({report.timings[task_id].deepest_rung_reachable}"
            f"/{len(report.timings[task_id].rungs)} rungs)"
            for task_id in task_ids
        ],
        fontsize=8,
    )
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("worst-case time from the last accepted heartbeat (ms, log scale)")
    axis.set_title(
        f"Escalation timelines against declared liveness deadlines: {spec.system}",
        fontsize=10,
    )
    axis.grid(axis="x", alpha=0.3, linestyle=":")
    handles = [Patch(facecolor="#2b3a55", label="detection (WCDL)")] + [
        Patch(facecolor=colour, label=action) for action, colour in _ACTION_STYLE.items()
    ]
    axis.legend(handles=handles, fontsize=7, ncol=3, loc="upper right", framealpha=0.9)
    fig.text(
        0.5,
        0.012,
        "watchdogspec, research-grade. Arithmetic over a declared specification; "
        "no running system is measured.",
        ha="center",
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target
