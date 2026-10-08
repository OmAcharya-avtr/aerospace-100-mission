"""Show the recursion failing to converge, and reporting that it failed.

Run: `python examples/non_convergence.py`
Writes: `../screenshots/non_convergence.png`

What to notice: the shrink margin is flat at 5.1e-2 for every iteration, so no
iteration cap would have been enough; the volume is still falling when the cap
is reached; and the set the recursion is holding is not invariant, which is
why it must be reported as an outer bound rather than as the answer.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from invariantset import (  # noqa: E402
    get_system,
    growth_table,
    maximal_robust_invariant_set,
    verify_robust_invariance,
)
from invariantset.plotting import plot_polytope  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "screenshots"


def main() -> None:
    system = get_system("slow_pair")
    caps = (10, 20, 40)
    results = {
        cap: maximal_robust_invariant_set(
            system.A, system.X, system.W, max_iter=cap, track_geometry=True
        )
        for cap in caps
    }
    deep = results[max(caps)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    rows = growth_table(deep)
    ax1.plot([r.k for r in rows], [r.shrink_margin for r in rows], "o-", label="shrink margin")
    ax1.axhline(deep.convergence_tol, color="crimson", linestyle="--",
                label=f"convergence tol {deep.convergence_tol:.0e}")
    ax1.set_yscale("log")
    ax1.set_xlabel("iteration k")
    ax1.set_ylabel("shrink margin [b units]")
    ax1.set_title(
        "the margin does not decay: ratio 1.000000 per iteration\n"
        "no iteration cap would have converged"
    )
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    for cap, colour in zip(caps, ("tab:blue", "tab:orange", "crimson"), strict=True):
        res = results[cap]
        inv, margin = verify_robust_invariance(system.A, res.polytope, system.W)
        plot_polytope(
            ax2,
            res.polytope,
            color=colour,
            linewidth=1.6,
            label=(
                f"cap {cap}: area {res.polytope.volume():.6f}, "
                f"invariant {inv} (margin {margin:+.2e})"
            ),
        )
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_title("each cap returns a different OUTER bound, none of them S_inf")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="lower center", fontsize=7)

    fig.suptitle(
        "invariantset: non-convergence of the one-step-set recursion, reported not hidden",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "non_convergence.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")
    print(deep.report())
    for cap in caps:
        res = results[cap]
        inv, margin = verify_robust_invariance(system.A, res.polytope, system.W)
        print(
            f"cap {cap:<4d} {res.termination:<14s} facets={res.polytope.n_halfspaces:<4d} "
            f"area={res.polytope.volume():.9f} invariant={inv} margin={margin:+.6e}"
        )


if __name__ == "__main__":
    main()
