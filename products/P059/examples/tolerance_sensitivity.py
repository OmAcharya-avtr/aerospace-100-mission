"""Show the redundancy tolerance changing the computed answer.

Run: `python examples/tolerance_sensitivity.py`
Writes: `../screenshots/tolerance_sensitivity.png`

What to notice: everything at or below 3e-3 returns the same 16-facet set; at
4e-3 and above the recursion stops contracting and never converges, and the
set it is left holding is larger than S_inf and is not invariant.  The
threshold sits on the final iteration's shrink margin.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from invariantset import (  # noqa: E402
    CONVERGED,
    get_system,
    maximal_robust_invariant_set,
    tolerance_sweep,
    verify_robust_invariance,
)
from invariantset.plotting import plot_polytope  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "screenshots"
TOLERANCES = [1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 3e-3, 4e-3, 1e-2, 5e-2]


def main() -> None:
    system = get_system("attitude_loop")
    rows = tolerance_sweep(system.A, system.X, system.W, TOLERANCES, max_iter=60)

    reference = maximal_robust_invariant_set(
        system.A, system.X, system.W, max_iter=60, redundancy_tol=1e-9, track_geometry=False
    )
    loose = maximal_robust_invariant_set(
        system.A, system.X, system.W, max_iter=400, redundancy_tol=5e-2, track_geometry=False
    )
    ref_inv, ref_margin = verify_robust_invariance(system.A, reference.polytope, system.W)
    loose_inv, loose_margin = verify_robust_invariance(system.A, loose.polytope, system.W)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    tol_values = [r.redundancy_tol for r in rows]
    facets = [r.facets if r.facets is not None else 0 for r in rows]
    colours = ["tab:green" if r.termination == CONVERGED else "crimson" for r in rows]
    ax1.bar(range(len(rows)), facets, color=colours)
    ax1.set_xticks(range(len(rows)))
    ax1.set_xticklabels([f"{t:.0e}" for t in tol_values], rotation=45, ha="right", fontsize=8)
    ax1.set_xlabel("redundancy tolerance [b units]")
    ax1.set_ylabel("facets of the returned set")
    ax1.set_title(
        "green = converged, red = iteration cap\n"
        "the answer changes between 3e-3 and 4e-3"
    )
    ax1.grid(alpha=0.25, axis="y")

    plot_polytope(
        ax2,
        reference.polytope,
        color="tab:green",
        linewidth=2.0,
        label=(
            f"tol 1e-9: {reference.polytope.n_halfspaces} facets, "
            f"area {reference.polytope.volume():.6f}, invariant {ref_inv}"
        ),
    )
    plot_polytope(
        ax2,
        loose.polytope,
        color="crimson",
        linewidth=1.6,
        linestyle="--",
        label=(
            f"tol 5e-2: {loose.polytope.n_halfspaces} facets, "
            f"area {loose.polytope.volume():.6f}, invariant {loose_inv}"
        ),
    )
    ax2.set_xlabel("theta [rad]")
    ax2.set_ylabel("theta_dot [rad/s]")
    ax2.set_title(
        "a loose tolerance returns a larger set that is not invariant\n"
        f"worst facet margins {ref_margin:.2e} and {loose_margin:+.2e}"
    )
    ax2.grid(alpha=0.25)
    ax2.legend(loc="lower center", fontsize=7)

    fig.suptitle(
        "invariantset: the redundancy tolerance is not a cosmetic setting", fontsize=10
    )
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "tolerance_sensitivity.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")
    for r in rows:
        vol = "-" if r.volume is None else f"{r.volume:.9f}"
        marg = "-" if r.invariance_margin is None else f"{r.invariance_margin:+.6e}"
        print(
            f"tol {r.redundancy_tol:.0e}  {r.termination:<14s} it={r.iterations:<4d} "
            f"facets={r.facets}  area={vol}  margin={marg}"
        )


if __name__ == "__main__":
    main()
