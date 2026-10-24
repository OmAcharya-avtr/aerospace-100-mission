"""Timeline of a command sequence with every violation and its cause marked.

The figure the product exists for: each violation is drawn at the step that
tripped it, with an arrow back to the step the provenance trace names. A
violation without that arrow is a task; with it, it is a diff.
"""

from __future__ import annotations

import _common
from _common import FORWARD, KIND_COLOURS, REFUSE, save, style

_common.bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from cmdguard.loader import load_case  # noqa: E402
from cmdguard.trace import forward_execute  # noqa: E402


def panel(axis, case, report, title: str) -> None:
    steps = case.sequence.steps
    times = [step.t_s for step in steps]
    axis.set_title(title)
    axis.set_xlabel("time after sequence epoch (s)")
    axis.set_yticks([])
    axis.set_ylim(-2.6, len(report.kind_counts()) + 1.6)
    axis.hlines(0.0, min(times) - 10.0, max(times) + 20.0, color=FORWARD, linewidth=1.2)
    # Labels are staggered over three rows. A first version put them all on one
    # row and the mnemonics overlapped illegibly wherever two commands were
    # close in time, which is most of a real sequence.
    offsets = (-20, -32, -44, -56)
    for index, step in enumerate(steps):
        axis.plot([step.t_s], [0.0], marker="o", markersize=4, color=FORWARD)
        axis.annotate(
            f"{index} {step.command}",
            (step.t_s, 0.0),
            textcoords="offset points",
            xytext=(0, offsets[index % len(offsets)]),
            ha="center",
            fontsize=6.5,
            color="#333333",
        )
    rows: dict[str, float] = {}
    for violation in report.violations:
        row = rows.setdefault(violation.kind.value, 0.6 + 0.75 * len(rows))
        colour = KIND_COLOURS.get(violation.kind.value, REFUSE)
        axis.plot([violation.t_s], [row], marker="v", markersize=7, color=colour)
        axis.annotate(
            violation.kind.value,
            (violation.t_s, row),
            textcoords="offset points",
            xytext=(6, 2),
            fontsize=6.5,
            color=colour,
        )
        cause = next((e for e in violation.trace if e.step is not None
                      and e.step != violation.step), None)
        if cause is not None and cause.t_s is not None:
            axis.annotate(
                "",
                xy=(cause.t_s, 0.12),
                xytext=(violation.t_s, row - 0.08),
                arrowprops={
                    "arrowstyle": "->",
                    "color": colour,
                    "linewidth": 0.9,
                    "connectionstyle": "arc3,rad=0.25",
                },
            )


def main() -> None:
    style()
    figure = plt.figure(figsize=(11.0, 7.4))
    grid = figure.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.45)

    late = load_case("examples/cases/late_payload_enable.yaml")
    late_report = forward_execute(late)
    panel(
        figure.add_subplot(grid[0]),
        late,
        late_report,
        "late_payload_enable: one violation, and the arrow is the finding\n"
        "step 6 PAYLOAD_SCAN is illegal because step 5 PAYLOAD_SAFE left the payload disabled",
    )

    multi = load_case("examples/cases/resource_and_exclusion.yaml")
    multi_report = forward_execute(multi)
    panel(
        figure.add_subplot(grid[1]),
        multi,
        multi_report,
        f"resource_and_exclusion: {multi_report.n_violations} violations across four kinds,\n"
        "each arrow pointing at the earlier step that created the state it tripped over",
    )

    figure.suptitle(
        "cmdguard forward execution: violations with the state trace that reaches them",
        fontsize=11,
    )
    print(f"wrote {save(figure, 'violation_timeline')}")
    print(f"late_payload_enable    : {late_report.n_violations} violation(s), "
          f"steps to review {late_report.involved_steps()}")
    print(f"resource_and_exclusion : {multi_report.n_violations} violation(s), "
          f"steps to review {multi_report.involved_steps()}")


if __name__ == "__main__":
    main()
