"""The documented case where a tolerance change alters the computed answer.

Two separate tolerances are swept, because they fail in different ways:

* `redundancy_tol`, the slack in the redundant-row test (2).  Once it exceeds
  the per-iteration shrink margin, the rows the pre-set contributes look
  redundant and are discarded, the recursion stops contracting, and it never
  converges.
* `convergence_tol`, the slack in the invariance test (9).  It buys an
  iteration and costs exactly that much of the invariance guarantee, which is
  then measured independently.
"""

from __future__ import annotations

from invariantset import (
    get_system,
    maximal_robust_invariant_set,
    tolerance_sweep,
    verify_robust_invariance,
)

REFERENCE_AREA = 0.538347284806255  # redundancy_tol = 1e-9, measured below


def main() -> None:
    system = get_system("attitude_loop")
    print("system: attitude_loop")
    print(f"        {system.description}")
    print("        units: theta [rad], theta_dot [rad/s]; areas in rad.rad/s")

    print("\nA. Redundancy-tolerance sweep, max_iter = 60, convergence_tol = 1e-9")
    tols = [0.0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 2e-3, 3e-3, 4e-3, 1e-2, 5e-2]
    rows = tolerance_sweep(system.A, system.X, system.W, tols, max_iter=60)
    print("   redundancy_tol  termination     iters facets  area            "
          "inv_margin      area/reference")
    for r in rows:
        area = "      -        " if r.volume is None else f"{r.volume:<15.9f}"
        marg = "      -        " if r.invariance_margin is None else f"{r.invariance_margin:<15.6e}"
        ratio = "   -" if r.volume_vs_reference is None else f"{r.volume_vs_reference:.9f}"
        facets = "-" if r.facets is None else str(r.facets)
        print(
            f"   {r.redundancy_tol:<15.1e} {r.termination:<15s} {r.iterations:<5d} "
            f"{facets:<7s} {area} {marg} {ratio}"
        )
    distinct = {None if r.volume is None else round(r.volume, 9) for r in rows}
    print(f"   distinct answers across the sweep: {len(distinct)}")
    print("   THE TOLERANCE CHANGED THE ANSWER." if len(distinct) > 1 else "   no change")

    print("\nB. Why 3e-3 works and 4e-3 does not: the final shrink margin")
    res = maximal_robust_invariant_set(
        system.A, system.X, system.W, max_iter=60, track_geometry=False
    )
    positives = [r.shrink_margin for r in res.history if r.shrink_margin > 0.0]
    final_margin = positives[-1]
    print(f"   per-iteration shrink margins: {[f'{m:.6e}' for m in positives]}")
    print(f"   smallest positive shrink margin = {final_margin:.12e}")
    lo, hi = 1e-3, 1e-1
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        probe = maximal_robust_invariant_set(
            system.A, system.X, system.W, max_iter=25,
            redundancy_tol=mid, track_geometry=False,
        )
        if probe.termination == "converged":
            lo = mid
        else:
            hi = mid
    print(f"   bisected threshold: converges up to {lo:.12e}")
    print(f"                       fails from      {hi:.12e}")
    print(f"   the threshold brackets the final shrink margin {final_margin:.12e}: "
          f"{lo <= final_margin <= hi}")
    print("   mechanism: a redundancy tolerance at or above the shrink margin")
    print("   discards exactly the rows the pre-set contributed, so Omega_{k+1}")
    print("   equals Omega_k in representation but not in intent, and the")
    print("   recursion neither contracts nor terminates.")

    print("\nC. What a loose tolerance actually returns")
    print("   (the cap is part of the answer when the recursion does not converge)")
    for tol, cap in ((4e-3, 150), (1e-2, 150), (5e-2, 150), (5e-2, 400)):
        loose = maximal_robust_invariant_set(
            system.A, system.X, system.W, max_iter=cap,
            redundancy_tol=tol, track_geometry=False,
        )
        strict_ok, margin = verify_robust_invariance(system.A, loose.polytope, system.W)
        eng_ok = verify_robust_invariance(system.A, loose.polytope, system.W, tol=1e-9)[0]
        area = loose.polytope.volume()
        print(
            f"   tol {tol:.0e} cap {cap:<4d}: {loose.termination}, "
            f"facets {loose.polytope.n_halfspaces}, "
            f"area {area:.9f} ({100.0 * (area / REFERENCE_AREA - 1.0):+.4f} % vs reference), "
            f"invariant strict {strict_ok} / 1e-9 {eng_ok}, margin {margin:+.6e}"
        )
    print("   each of these is a WRONG answer, not a coarse one: the returned set")
    print("   is larger than S_inf and fails the independent invariance test.")
    print("   note also that tol 5e-2 gives a different set at cap 150 and at cap 400:")
    print("   a non-converged outer bound is a function of the cap, not of the system.")

    print("\nD. Convergence-tolerance sweep, redundancy_tol = 1e-9, max_iter = 60")
    print("   convergence_tol  iters facets  area            inv_margin      invariant(1e-9)")
    for ctol in (0.0, 1e-9, 1e-6, 1e-4, 1e-3, 3e-3, 5e-3, 1e-2, 5e-2):
        res = maximal_robust_invariant_set(
            system.A, system.X, system.W, max_iter=60,
            convergence_tol=ctol, track_geometry=False,
        )
        if res.polytope is None:
            print(f"   {ctol:<16.1e} {res.termination}")
            continue
        _, margin = verify_robust_invariance(system.A, res.polytope, system.W)
        eng_ok = verify_robust_invariance(system.A, res.polytope, system.W, tol=1e-9)[0]
        print(
            f"   {ctol:<16.1e} {res.iterations:<5d} {res.polytope.n_halfspaces:<7d} "
            f"{res.polytope.volume():<15.9f} {margin:<15.6e} {eng_ok}"
        )
    print("   the convergence tolerance is a slack on the guarantee, not on the")
    print("   arithmetic: at 5e-3 the recursion stops an iteration early and the")
    print("   set it returns violates the invariance condition by 3.63e-3 in b units,")
    print("   which on the theta facet is 3.63e-3 rad of constraint the set does not hold.")

    print("\nE. The same sweep on a system where no tolerance changes the answer")
    decoupled = get_system("decoupled_2d")
    rows = tolerance_sweep(
        decoupled.A, decoupled.X, decoupled.W, [1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1],
        max_iter=20,
    )
    for r in rows:
        print(
            f"   tol {r.redundancy_tol:.0e}: {r.termination}, facets {r.facets}, "
            f"area {r.volume:.9f}"
        )
    print("   X is already invariant with margin -0.4, far larger than any tolerance")
    print("   swept, so nothing moves. Sensitivity is a property of the system and")
    print("   the margin, not of the implementation alone.")


if __name__ == "__main__":
    main()
