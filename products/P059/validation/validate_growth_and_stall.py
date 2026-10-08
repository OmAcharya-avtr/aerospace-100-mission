"""Measure representation growth, iteration stalling, and non-convergence.

Facet counts, vertex counts and iteration counts are deterministic and are
reported as the primary numbers.  Wall-clock timings are included only where
they are needed to justify the compute budget, and they move 10-20 % between
runs on these two shared cores.
"""

from __future__ import annotations

import time

import numpy as np

from invariantset import (
    Polytope,
    get_system,
    growth_table,
    is_subset,
    maximal_robust_invariant_set,
    verify_robust_invariance,
)


def growth_block(name: str, max_iter: int = 30) -> None:
    system = get_system(name)
    t0 = time.perf_counter()
    res = maximal_robust_invariant_set(
        system.A, system.X, system.W, max_iter=max_iter,
        track_geometry=True, vertex_budget=500_000,
    )
    elapsed = time.perf_counter() - t0
    rows = growth_table(res)
    print(f"\n== {name} (dim {system.dim}): {system.description}")
    print(f"   {res.termination} at k={res.iterations}, elapsed {elapsed:.2f} s")
    print("   k   raw  facets  removed  vertices  vtx/facet  volume          "
          "shrink_margin   margin_ratio")
    for r in rows:
        ratio = "   -    " if r.margin_ratio is None else f"{r.margin_ratio:8.6f}"
        print(
            f"   {r.k:<3d} {r.halfspaces_raw:<4d} {r.halfspaces:<7d} {r.removed:<8d} "
            f"{r.vertices:<9d} {r.vertices / r.halfspaces:<10.4f} {r.volume:<15.9f} "
            f"{r.shrink_margin:<15.6e} {ratio}"
        )
    first, last = rows[0], rows[-1]
    print(
        f"   measured growth: raw rows {first.halfspaces_raw} -> {last.halfspaces_raw} "
        f"(+{(last.halfspaces_raw - first.halfspaces_raw) / max(len(rows) - 1, 1):.1f}/iter), "
        f"facets {first.halfspaces} -> {last.halfspaces} "
        f"(+{(last.halfspaces - first.halfspaces) / max(len(rows) - 1, 1):.1f}/iter), "
        f"vertices {first.vertices} -> {last.vertices} "
        f"(+{(last.vertices - first.vertices) / max(len(rows) - 1, 1):.1f}/iter)"
    )
    if res.polytope is not None:
        strict, margin = verify_robust_invariance(system.A, res.polytope, system.W)
        engineering = verify_robust_invariance(system.A, res.polytope, system.W, tol=1e-9)[0]
        print(
            f"   independent invariance check: strict(tol=0) {strict}, "
            f"tol=1e-9 {engineering}, worst facet margin {margin:+.6e}"
        )


def main() -> None:
    print("A. Measured representation growth")
    growth_block("damped_rotation_2d")
    growth_block("damped_rotation_3d")
    growth_block("attitude_loop", max_iter=50)

    print("\n\nB. Where the brute-force vertex enumeration gives up")
    print("   C(m, n) bases are solved per enumeration; the budget is max_bases.")
    for dim, m in ((2, 32), (3, 30), (3, 60), (4, 40), (5, 40)):
        bases = 1
        for k in range(dim):
            bases = bases * (m - k) // (k + 1)
        print(f"   dim {dim}, m = {m:<3d} rows -> C(m, n) = {bases:,} bases")
    print("   default max_bases = 200000, so dim 4 with 40 rows (91,390) still runs")
    print("   and dim 5 with 40 rows (658,008) is refused rather than left to hang.")
    t0 = time.perf_counter()
    P = Polytope.unit_box(4, 1.0)
    extra = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 0.5, 0.0], [0.0, 0.3, -1.0, 0.8]])
    P = Polytope(np.vstack([P.A, extra]), np.concatenate([P.b, [3.0, 1.5, 1.2]]))
    V = P.vertices()
    print(f"   4-D example, {P.n_halfspaces} rows: {V.shape[0]} vertices in "
          f"{time.perf_counter() - t0:.2f} s")

    print("\n\nC. Stalling and non-convergence")
    system = get_system("slow_pair")
    print(f"   system: {system.description}")
    print(f"   eigenvalues of A: {np.linalg.eigvals(system.A)}")
    for cap in (10, 20, 40, 80):
        t0 = time.perf_counter()
        res = maximal_robust_invariant_set(
            system.A, system.X, system.W, max_iter=cap, track_geometry=False
        )
        elapsed = time.perf_counter() - t0
        inv, margin = verify_robust_invariance(system.A, res.polytope, system.W)
        print(
            f"   cap {cap:<4d} {res.termination:<14s} facets={res.polytope.n_halfspaces:<4d} "
            f"area={res.polytope.volume():.9f} invariant={inv} margin={margin:+.6e} "
            f"({elapsed:.2f} s)"
        )
    deep = maximal_robust_invariant_set(
        system.A, system.X, system.W, max_iter=80, track_geometry=False
    )
    margins = [r.shrink_margin for r in deep.history]
    ratios = [b / a for a, b in zip(margins[:-1], margins[1:], strict=True)]
    print(
        f"   shrink margin over 80 iterations: min {min(margins):.6e}, "
        f"max {max(margins):.6e}, ratio min {min(ratios):.9f} max {max(ratios):.9f}"
    )
    print("   the margin does not decay at all, so the recursion cannot terminate;")
    print("   this is reported as termination='iteration_cap', not as an answer.")
    shallow = maximal_robust_invariant_set(
        system.A, system.X, system.W, max_iter=10, track_geometry=False
    )
    print(
        f"   nesting check: cap-80 result subset of cap-10 result = "
        f"{is_subset(deep.polytope, shallow.polytope, tol=1e-9)[0]}"
    )

    print("\n   full report emitted for the non-converged case:")
    short = maximal_robust_invariant_set(
        system.A, system.X, system.W, max_iter=5, track_geometry=False
    )
    for row in short.report().splitlines():
        print(f"     {row}")

    print("\n\nD. Marginally stable system: eigenvalues exactly on the unit circle")
    theta = 0.3
    A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X = Polytope.from_box([0.0, 0.0], [1.0, 1.0])
    W = Polytope.from_box([0.0, 0.0], [0.01, 0.01])
    for cap in (10, 30, 60):
        res = maximal_robust_invariant_set(A, X, W, max_iter=cap, track_geometry=False)
        inv, margin = verify_robust_invariance(A, res.polytope, W)
        print(
            f"   cap {cap:<4d} {res.termination:<14s} facets={res.polytope.n_halfspaces:<4d} "
            f"area={res.polytope.volume():.9f} invariant={inv} margin={margin:+.6e}"
        )
    print("   |lambda| = 1 means nothing contracts; the recursion shaves a sliver")
    print("   per iteration forever. No cap value fixes this.")


if __name__ == "__main__":
    main()
