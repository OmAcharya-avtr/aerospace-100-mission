"""The lattice, the label rules, and the one fact the product exists for.

Three panels. The meet table as a grid; the chain as a Hasse diagram with the
label each element admits; and the dilution curve -- what happens to an
aggregate's label as a single simulated reading is added to an otherwise live
set of growing size. The third panel is the figure to look at: the label does
not degrade gradually, it changes at the first simulated reading and stays
changed however much live data is added.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    LABEL_COLOURS,
    NEUTRAL,
    PROVENANCE_COLOURS,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

from boardprobe import (  # noqa: E402
    ORDER,
    ORIN_NANO,
    LiveSource,
    Provenance,
    SimulatedSource,
    attribute,
    label_for,
    meet,
)


def main() -> int:
    style()
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))

    left, middle, right = axes

    grid = np.array([[ORDER.index(meet(a, b)) for b in ORDER] for a in ORDER])
    left.imshow(grid, cmap="cividis", vmin=0, vmax=2)
    left.set_xticks(range(3), [p.value for p in ORDER], rotation=20)
    left.set_yticks(range(3), [p.value for p in ORDER])
    for row in range(3):
        for column in range(3):
            left.text(
                column, row, meet(ORDER[row], ORDER[column]).value,
                ha="center", va="center", fontsize=8,
                color="white" if grid[row, column] < 2 else "black",
            )
    left.set_title("meet: the weaker of two provenances")
    left.grid(False)

    middle.set_xlim(-0.6, 1.9)
    middle.set_ylim(-0.5, 2.5)
    for index, provenance in enumerate(ORDER):
        colour = PROVENANCE_COLOURS[provenance.value]
        middle.scatter([0.0], [index], s=220, color=colour, zorder=3)
        middle.text(0.18, index, provenance.value, va="center", fontsize=9)
        label = label_for(provenance).value
        middle.text(
            1.05, index, label, va="center", fontsize=9, weight="bold",
            color=LABEL_COLOURS[label],
        )
    middle.plot([0.0, 0.0], [0.0, 2.0], color=NEUTRAL, linewidth=1.2, zorder=1)
    middle.text(1.05, 2.42, "label admitted", fontsize=8, color=NEUTRAL)
    middle.text(0.18, 2.42, "provenance", fontsize=8, color=NEUTRAL)
    middle.set_title("the chain, and the one label each element admits")
    middle.set_xticks([])
    middle.set_yticks([])
    middle.grid(False)

    clock = LiveSource(ORIN_NANO)
    simulated = SimulatedSource(ORIN_NANO, seed=11)
    sizes = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    pure = []
    diluted = []
    for size in sizes:
        live = [clock.read("monotonic_clock") for _ in range(size)]
        pure.append(ORDER.index(attribute(live).provenance))
        diluted.append(
            ORDER.index(
                attribute([*live, simulated.read("die_temp_cpu")]).provenance
            )
        )
    right.step(sizes, pure, where="post", color=PROVENANCE_COLOURS["live"],
               linewidth=1.8, label="live readings only")
    right.step(sizes, diluted, where="post",
               color=PROVENANCE_COLOURS["simulated"], linewidth=1.8,
               linestyle="--", label="the same, plus ONE simulated reading")
    right.set_xscale("log")
    right.set_yticks(range(3), [p.value for p in ORDER])
    right.set_ylim(-0.4, 2.4)
    right.set_xlabel("live readings in the aggregate")
    right.set_title("one simulated reading is not diluted")
    right.legend(loc="center left", fontsize=8)

    figure.suptitle(
        "boardprobe: provenance propagates by the lattice meet "
        "(Level 3, hardware-pending; no figure measured on the board)",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = save(figure, "provenance_lattice")
    print(f"written: {path}")
    print(f"meet table cells plotted: {grid.size}")
    print(f"dilution sizes: {sizes}")
    print(
        "label with 500 live readings and one simulated: "
        f"{label_for(Provenance(ORDER[diluted[-1]].value)).value}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
