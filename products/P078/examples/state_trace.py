"""The predicate and mode trace through a sequence, with the cause highlighted.

A violation report is only actionable if the reader can see what the state was
when the command ran. This figure is that state, step by step: a grid of
predicate values with the mode along the top, the violating step marked, and
the step the provenance trace names marked too.
"""

from __future__ import annotations

import _common
from _common import FORWARD, LINT, OK, REFUSE, save, style

_common.bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cmdguard.loader import load_case  # noqa: E402
from cmdguard.trace import forward_execute  # noqa: E402

VALUE_TO_CODE = {"true": 2, "false": 1, "unknown": 0}
CODE_COLOURS = [LINT, REFUSE, OK]
CODE_LABELS = ["unverified", "false", "true"]


def main() -> None:
    style()
    case = load_case("examples/cases/late_payload_enable.yaml")
    report = forward_execute(case)
    names = sorted({name for snap in report.snapshots for name in snap.predicates})
    grid = np.zeros((len(names), len(report.snapshots)), dtype=int)
    for column, snapshot in enumerate(report.snapshots):
        for row, name in enumerate(names):
            value = snapshot.predicates.get(name)
            grid[row, column] = VALUE_TO_CODE[value.value] if value is not None else 0

    figure = plt.figure(figsize=(10.5, 5.2))
    axis = figure.add_subplot(111)
    axis.grid(False)
    from matplotlib.colors import ListedColormap

    axis.imshow(
        grid,
        aspect="auto",
        cmap=ListedColormap(CODE_COLOURS),
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names, fontsize=8)
    axis.set_xticks(range(len(report.snapshots)))
    axis.set_xticklabels(
        [f"{s.step}\n{s.command}\n{s.mode}" for s in report.snapshots],
        fontsize=6.5,
    )
    axis.set_xlabel("step, command, and the mode it left behind")

    # imshow puts row 0 at the top and y increases downward. The two call-outs
    # go ABOVE the grid, at a y smaller than -0.5, with annotation_clip off:
    # placed below the grid they landed on top of the x tick labels, and placed
    # above without annotation_clip they were silently clipped away.
    above = -0.62
    axis.set_xlim(-0.6, len(report.snapshots) - 0.4)
    violation = report.violations[0]
    cause = violation.trace[0]
    axis.add_patch(
        plt.Rectangle(
            (violation.step - 0.5, -0.5),
            1.0,
            len(names),
            fill=False,
            edgecolor=REFUSE,
            linewidth=2.2,
            clip_on=False,
        )
    )
    axis.annotate(
        f"step {violation.step} {violation.command}\n{violation.kind.value}",
        (violation.step + 0.45, above),
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=REFUSE,
        annotation_clip=False,
    )
    if cause.step is not None:
        axis.add_patch(
            plt.Rectangle(
                (cause.step - 0.5, -0.5),
                1.0,
                len(names),
                fill=False,
                edgecolor=FORWARD,
                linewidth=2.2,
                linestyle="--",
                clip_on=False,
            )
        )
        axis.annotate(
            f"step {cause.step} {cause.command}\n{cause.note}",
            (cause.step - 0.45, above),
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=FORWARD,
            annotation_clip=False,
        )

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=CODE_COLOURS[code],
                   label=CODE_LABELS[code], markersize=9)
        for code in (2, 1, 0)
    ]
    axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.42),
                ncol=3, fontsize=8)
    axis.set_title(
        "late_payload_enable: the predicate state at every step\n"
        "the dashed box is where payload_enabled became false; the solid box is the "
        "command that then could not run",
        fontsize=10,
        pad=34,
    )
    print(f"wrote {save(figure, 'state_trace')}")
    print(f"predicates tracked : {len(names)} -- {', '.join(names)}")
    print(f"violating step     : {violation.step} {violation.command} "
          f"({violation.kind.value})")
    print(f"cause named by the trace: step {cause.step} {cause.command} at "
          f"t={cause.t_s:.1f} s")


if __name__ == "__main__":
    main()
