"""The closed-form re-run cost, plotted and tabulated. The mathematical core.

Writes ``screenshots/geometric_rerun_cost.png``.

What to notice in the figure. The left panel is ``E[F] = p/(1-p)``, which
diverges as ``p -> 1``; the capped curves peel away from it and stay finite,
and at ``p = 1`` each equals its own budget exactly. The right panel is the
cost surface in seconds: the iso-cost contours are hyperbolas, so a given
amount of wasted time is reached either by a frequent failure in a fast test or
by a rare failure in a slow one. Two tests on the same contour cost the same
and a flake-rate dashboard ranks them hundreds of places apart.

Runs from any working directory. Reads no clock -- the runtimes on the axes are
model inputs, not measurements.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from flakerank.geometric import (  # noqa: E402
    expected_failed_attempts,
    expected_wasted_seconds,
)

#: Retry budgets drawn, with ``None`` for the unbounded policy.
BUDGETS = (1, 2, 3, 5, 10, None)


def main() -> int:
    """Draw the closed form and print the hand-checkable values."""
    _bootstrap.style()
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    probabilities = np.linspace(0.0, 0.98, 400)
    for budget in BUDGETS:
        values = [expected_failed_attempts(float(p), budget) for p in probabilities]
        label = "unbounded" if budget is None else f"budget m = {budget}"
        left.plot(probabilities, values, linewidth=1.4, label=label)
    left.axvline(1.0 / 44.0, color="crimson", linestyle=":", linewidth=1.2)
    left.annotate(
        "p = 1/44, the rate\nthis mission shipped",
        (1.0 / 44.0, 2.2),
        textcoords="offset points",
        xytext=(10, 0),
        fontsize=8,
        color="crimson",
    )
    left.set_ylim(0.0, 5.0)
    left.set_xlabel("per-attempt failure probability p")
    left.set_ylabel("expected failed attempts per invocation E[F]")
    left.set_title(
        "E[F] = p/(1-p) unbounded, equation (2)\n"
        "capped: (p - p^m)/(1-p) + p^m, equation (5)"
    )
    left.legend(fontsize=8, loc="upper left")

    grid_p = np.logspace(-3.0, np.log10(0.6), 160)
    grid_t = np.logspace(-2.0, 3.0, 160)
    mesh_p, mesh_t = np.meshgrid(grid_p, grid_t)
    cost = mesh_t * mesh_p / (1.0 - mesh_p)
    contours = right.contour(
        mesh_p,
        mesh_t,
        cost,
        levels=np.logspace(-3, 3, 7),
        linewidths=0.9,
    )
    right.clabel(contours, inline=True, fontsize=6, fmt="%.3g s")
    right.scatter(
        [1.0 / 44.0], [600.0], s=90, facecolor="none", edgecolor="crimson", linewidth=1.6
    )
    right.annotate(
        f"1-in-44 in a 600 s test\n{expected_wasted_seconds(1 / 44, 600.0):.3f} s wasted"
        " per invocation",
        (1.0 / 44.0, 600.0),
        textcoords="offset points",
        xytext=(-158, -48),
        fontsize=8,
        color="crimson",
    )
    right.scatter([0.5], [0.1], s=90, facecolor="none", edgecolor="darkorange", linewidth=1.6)
    right.annotate(
        f"1-in-2 in a 0.1 s test\n{expected_wasted_seconds(0.5, 0.1):.3f} s wasted"
        " per invocation",
        (0.5, 0.1),
        textcoords="offset points",
        xytext=(-164, -6),
        fontsize=8,
        color="darkorange",
    )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("per-attempt failure probability p")
    right.set_ylabel("failing-attempt runtime t (s)")
    right.set_title(
        "iso-cost contours of t * p/(1-p), equation (3)\n"
        "the marked tests differ 22x in rate, 140x in cost, oppositely"
    )

    figure.tight_layout()
    destination = ROOT / _bootstrap.SCREENSHOTS / "geometric_rerun_cost.png"
    figure.savefig(destination)
    plt.close(figure)

    print("flakerank -- the closed-form geometric re-run cost")
    print("")
    print("E[N] = 1/(1-p) is the mean of a geometric variable on {1, 2, 3, ...}")
    print("  (Feller 1968, Volume I; Johnson, Kemp & Kotz 2005, chapter 5).")
    print("E[F] = E[N] - 1 = p/(1-p) is the wasted part: the final passing")
    print("  attempt is work that had to happen anyway.")
    print("E[F_m] = (p - p^m)/(1-p) + p^m under a cap of m attempts.")
    print("")
    print(f"{'p':>12} {'E[F] unbounded':>16} {'E[F_1]':>10} {'E[F_2]':>10} {'E[F_3]':>10}")
    for probability in (0.0, 1.0 / 44.0, 0.05, 0.25, 0.5, 0.75):
        print(
            f"{probability:>12.8f} {expected_failed_attempts(probability):>16.9f}"
            f" {expected_failed_attempts(probability, 1):>10.6f}"
            f" {expected_failed_attempts(probability, 2):>10.6f}"
            f" {expected_failed_attempts(probability, 3):>10.6f}"
        )
    print("")
    print("the two tests marked on the right panel:")
    slow = expected_wasted_seconds(1.0 / 44.0, 600.0)
    fast = expected_wasted_seconds(0.5, 0.1)
    print(f"  1-in-44 in a 600 s test: {slow!r} s wasted per invocation")
    print(f"  1-in-2 in a 0.1 s test : {fast!r} s wasted per invocation")
    print(f"  cost ratio             : {slow / fast!r}")
    print(f"  flake-rate ratio       : {0.5 / (1.0 / 44.0)!r}, the other way round")
    print("")
    print("the divergence at p = 1 is raised, never returned as inf:")
    try:
        expected_failed_attempts(1.0)
    except ValueError as error:
        print(f"  expected_failed_attempts(1.0) -> ValueError: {str(error)[:60]}...")
    print(f"  expected_failed_attempts(1.0, retry_budget=3) -> "
          f"{expected_failed_attempts(1.0, 3)!r}")
    print("")
    print(f"figure written to {_bootstrap.relative(destination)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
