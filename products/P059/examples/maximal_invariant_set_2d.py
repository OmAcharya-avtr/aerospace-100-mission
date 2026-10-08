"""Plot the maximal robust invariant set of the illustrative attitude loop.

Run: `python examples/maximal_invariant_set_2d.py`
Writes: `../screenshots/maximal_invariant_set_2d.png`

What to notice: the nested sequence Omega_0 ... Omega_5 contracts onto S_inf,
and S_inf is a proper subset of the declared constraint set X.  The states in
X but outside S_inf cannot be held inside X against every admissible
disturbance sequence.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from invariantset import (  # noqa: E402
    get_system,
    maximal_robust_invariant_set,
    pre_set,
    verify_robust_invariance,
)
from invariantset.plotting import plot_polytope  # noqa: E402
from invariantset.setalgebra import intersect  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "screenshots"


def main() -> None:
    system = get_system("attitude_loop")
    result = maximal_robust_invariant_set(
        system.A, system.X, system.W, max_iter=50, track_geometry=True
    )
    invariant, margin = verify_robust_invariance(system.A, result.polytope, system.W)

    # Rebuild the intermediate iterates so they can be drawn.
    iterates = [system.X.remove_redundant()]
    omega = iterates[0]
    for _ in range(result.iterations - 1):
        omega = intersect(omega, pre_set(system.A, omega, system.W)).remove_redundant()
        iterates.append(omega)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    cmap = plt.get_cmap("viridis")
    for i, poly in enumerate(iterates):
        plot_polytope(
            ax,
            poly,
            color=cmap(i / max(len(iterates) - 1, 1)),
            linewidth=1.0,
            label=f"Omega_{i}" if i in (0, len(iterates) - 1) else None,
        )
    plot_polytope(ax, result.polytope, color="crimson", linewidth=2.2, label="S_inf")
    ax.set_xlabel("theta [rad]")
    ax.set_ylabel("theta_dot [rad/s]")
    ax.set_title(
        f"maximal robust invariant set, {result.iterations} iterations\n"
        f"{result.polytope.n_halfspaces} facets, "
        f"area {result.polytope.volume():.6f} rad.rad/s"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    ks = [r.k for r in result.history]
    ax2.plot(ks, [r.halfspaces_raw for r in result.history], "o-", label="rows before removal")
    ax2.plot(ks, [r.halfspaces for r in result.history], "s-", label="facets kept")
    ax2.plot(ks, [r.vertices for r in result.history], "^--", label="vertices")
    ax2.set_xlabel("iteration k")
    ax2.set_ylabel("count")
    ax2.set_title("representation growth per iteration (measured)")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)

    fig.suptitle(
        "invariantset: attitude loop, dt = 0.05 s, disturbance 0.12 rad/s^2 for one sample"
        f"  |  independently verified invariant: {invariant}, worst facet margin {margin:.2e}",
        fontsize=9,
    )
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "maximal_invariant_set_2d.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")
    print(result.report())


if __name__ == "__main__":
    main()
