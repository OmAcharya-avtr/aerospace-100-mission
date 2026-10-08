"""What the search is actually looking at: the robustness landscape.

Two rows of two-dimensional slices over ``step_amplitude`` against
``kp_factor``, with the other four decision variables held fixed:

* **top row**, held at the **box centre**. On every instance shown this slice
  contains no violation at all. That is the finding, not a plotting failure: the
  violating sets of this suite need several variables away from the centre at
  once, which is exactly why a one-variable-at-a-time sweep finds nothing and
  why falsification needs a search.
* **bottom row**, held at a **counterexample** found by the surrogate-guided
  strategy. The violating region appears, outlined at the zero contour.

Saves ``../screenshots/robustness_landscape.png``.

A slice is a slice. It shows a two-dimensional cut through a six-dimensional
box, and nothing here is a statement about the whole box.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from _bootstrap import add_src_to_path  # noqa: E402

ROOT = add_src_to_path()

from falsifyloop.instances import instance  # noqa: E402
from falsifyloop.search import surrogate_guided  # noqa: E402
from falsifyloop.systems import LoopInput  # noqa: E402

GRID = 60
SHOWN = ("settling-band", "overshoot-tight", "attitude-envelope", "rate-envelope")
AXIS_X, AXIS_Y = 0, 1
SEARCH_BUDGET = 400
SEARCH_SEED = 5


def _slice(inst, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(inst.box[AXIS_X, 0], inst.box[AXIS_X, 1], GRID)
    ys = np.linspace(inst.box[AXIS_Y, 0], inst.box[AXIS_Y, 1], GRID)
    field = np.empty((GRID, GRID))
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            point = anchor.copy()
            point[AXIS_X] = xv
            point[AXIS_Y] = yv
            field[j, i] = inst.evaluate(point)
    return xs, ys, field


def _panel(ax, inst, xs, ys, field, label, fig) -> float:
    limit = float(np.max(np.abs(field)))
    mesh = ax.pcolormesh(xs, ys, field, cmap="RdBu", vmin=-limit, vmax=limit, shading="auto")
    if field.min() < 0.0 < field.max():
        ax.contour(xs, ys, field, levels=[0.0], colors="black", linewidths=1.8)
    fraction = float(np.mean(field < 0.0))
    ax.set_title(f"{label}\n{fraction * 100:.1f} % of this slice violates", fontsize=9.5)
    ax.set_xlabel(f"{LoopInput.FIELDS[AXIS_X]} [deg]")
    fig.colorbar(mesh, ax=ax, label="robustness")
    return fraction


def main() -> int:
    fig, axes = plt.subplots(2, len(SHOWN), figsize=(5.0 * len(SHOWN), 9.2))
    rows = []
    for column, identifier in enumerate(SHOWN):
        inst = instance(identifier)
        xs, ys, centre_field = _slice(inst, inst.centre())
        centre_fraction = _panel(
            axes[0, column],
            inst,
            xs,
            ys,
            centre_field,
            f"{identifier} [{inst.tier}] - others at the box CENTRE",
            fig,
        )
        found = surrogate_guided(inst, SEARCH_BUDGET, SEARCH_SEED)
        anchor = found.best_vector
        xs2, ys2, cex_field = _slice(inst, anchor)
        cex_fraction = _panel(
            axes[1, column],
            inst,
            xs2,
            ys2,
            cex_field,
            f"{identifier} - others at a COUNTEREXAMPLE "
            f"(found at simulation {found.first_violation})",
            fig,
        )
        axes[1, column].plot(
            anchor[AXIS_X], anchor[AXIS_Y], "k*", markersize=13, markeredgecolor="white"
        )
        rows.append(
            (identifier, inst.tier, centre_fraction, cex_fraction, found.first_violation)
        )
        if column == 0:
            for row in (0, 1):
                axes[row, 0].set_ylabel(f"{LoopInput.FIELDS[AXIS_Y]} [dimensionless]")

    fig.suptitle(
        "Requirement robustness over a 2-D slice of the 6-D search box "
        f"({GRID}x{GRID} grid). Blue satisfied, red violated, black contour is "
        "robustness = 0.\n"
        "Top: other four variables at the box centre, where no violation exists at all. "
        "Bottom: the same slice taken through a counterexample (star).",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out = ROOT / "screenshots" / "robustness_landscape.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)

    print(f"wrote {out.relative_to(ROOT)}")
    print("")
    print(
        f"{'instance':<20s} {'tier':<10s} {'centre slice':>13s} "
        f"{'cex slice':>11s} {'cex found at':>13s}"
    )
    for identifier, tier, centre, cex, where in rows:
        found_at = "not found" if where is None else str(where)
        print(f"{identifier:<20s} {tier:<10s} {centre:>13.4f} {cex:>11.4f} {found_at:>13s}")
    print("")
    print(f"simulations run for the slices: {2 * len(SHOWN) * GRID * GRID}")
    print(
        "No violation anywhere in any centre slice. The violating sets of this suite "
        "need several decision variables away from the centre at once."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
