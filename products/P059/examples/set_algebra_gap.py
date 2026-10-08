"""Measure the gap the Pontryagin difference leaves behind.

Run: `python examples/set_algebra_gap.py`
Writes: `../screenshots/set_algebra_gap.png`

What to notice: eroding P by Q and dilating back (left panel) returns a
strictly smaller set -- the shaded corners are the 8.00 % of P that is lost.
Dilating first and eroding second (right panel) returns P exactly.  The
Pontryagin difference is not an inverse of the Minkowski sum.
"""

from __future__ import annotations

import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from invariantset import (  # noqa: E402
    Polytope,
    is_subset,
    minkowski_sum,
    pontryagin_difference,
)
from invariantset.plotting import plot_polytope  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "screenshots"


def main() -> None:
    P = Polytope(
        np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]),
        np.array([0.0, 0.0, 1.0]),
    )
    Q = Polytope.from_box([0.0, 0.0], [0.1, 0.1])

    eroded = pontryagin_difference(P, Q)
    reopened = minkowski_sum(eroded, Q)
    dilated = minkowski_sum(P, Q)
    closed = pontryagin_difference(dilated, Q)

    area_P = P.volume()
    area_reopened = reopened.volume()
    deficit = area_P - area_reopened

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8))

    plot_polytope(ax1, P, color="k", linewidth=2.0, label=f"P, area {area_P:.6f}")
    plot_polytope(ax1, eroded, color="tab:blue", linewidth=1.4,
                  label=f"P (-) Q, area {eroded.volume():.6f}")
    plot_polytope(ax1, reopened, color="crimson", linewidth=1.8,
                  label=f"(P (-) Q) (+) Q, area {area_reopened:.6f}")
    ax1.set_title(
        f"erode then dilate loses {deficit:.6f} "
        f"({100.0 * deficit / area_P:.2f} % of P)\n"
        "(P (-) Q) (+) Q is a STRICT subset of P"
    )
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.25)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    plot_polytope(ax2, dilated, color="tab:green", linewidth=1.4,
                  label=f"P (+) Q, area {dilated.volume():.6f}")
    plot_polytope(ax2, P, color="k", linewidth=2.6, label=f"P, area {area_P:.6f}")
    plot_polytope(ax2, closed, color="crimson", linewidth=1.2, linestyle="--",
                  label=f"(P (+) Q) (-) Q, area {closed.volume():.6f}")
    forward = is_subset(closed, P)[1]
    backward = is_subset(P, closed)[1]
    ax2.set_title(
        "dilate then erode recovers P exactly\n"
        f"worst containment margins {forward:.1e} and {backward:.1e}"
    )
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.25)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")

    fig.suptitle(
        "invariantset: the Pontryagin difference is not an inverse of the Minkowski sum",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "set_algebra_gap.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")
    print(f"area(P)                = {area_P:.9f}")
    print(f"area(P (-) Q)          = {eroded.volume():.9f}")
    print(f"area((P (-) Q) (+) Q)  = {area_reopened:.9f}")
    print(f"deficit                = {deficit:.9f}  ({100.0 * deficit / area_P:.4f} % of P)")
    print(f"area((P (+) Q) (-) Q)  = {closed.volume():.9f}  (equals area(P))")


if __name__ == "__main__":
    main()
