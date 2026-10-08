"""Measure how the representation grows per iteration, in 2-D and 3-D.

Run: `python examples/iteration_growth.py`
Writes: `../screenshots/iteration_growth.png`

What to notice: in two dimensions the vertex count equals the facet count and
both grow by four per iteration; in three dimensions the vertex count grows by
eight per iteration against four facets, so the V-representation outgrows the
H-representation and the brute-force vertex enumeration is the first thing to
become unusable.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from invariantset import get_system, growth_table, maximal_robust_invariant_set  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "screenshots"


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for name, ax in (("damped_rotation_2d", ax1), ("damped_rotation_3d", ax2)):
        system = get_system(name)
        result = maximal_robust_invariant_set(
            system.A,
            system.X,
            system.W,
            max_iter=30,
            track_geometry=True,
            vertex_budget=500_000,
        )
        rows = growth_table(result)
        ks = [r.k for r in rows]
        ax.plot(ks, [r.halfspaces_raw for r in rows], "o-", label="rows before removal")
        ax.plot(ks, [r.halfspaces for r in rows], "s-", label="facets kept")
        ax.plot(ks, [r.vertices for r in rows], "^--", label="vertices")
        ax.plot(ks, [r.removed for r in rows], "v:", label="rows removed")
        ax.set_xlabel("iteration k")
        ax.set_ylabel("count")
        ax.set_title(
            f"{name} (dim {system.dim})\n"
            f"{result.termination} at k={result.iterations}, "
            f"{result.polytope.n_halfspaces} facets, "
            f"{rows[-1].vertices} vertices"
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        print(f"== {name}: {result.termination} at k={result.iterations}")
        print("  k  raw  facets  removed  vertices  volume        margin_ratio")
        for r in rows:
            ratio = "-" if r.margin_ratio is None else f"{r.margin_ratio:.6f}"
            print(
                f"  {r.k:<3d}{r.halfspaces_raw:<5d}{r.halfspaces:<8d}{r.removed:<9d}"
                f"{r.vertices:<10d}{r.volume:<14.9f}{ratio}"
            )

    fig.suptitle(
        "invariantset: measured representation growth of the one-step-set recursion",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "iteration_growth.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
