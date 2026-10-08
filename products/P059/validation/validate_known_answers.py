"""Known-answer checks against sets computed by hand.

Every expected value below was derived on paper before the code was run; the
derivations are reproduced in VALIDATION.md section 2 and in the docstrings of
tests/test_known_answers.py.  All of them are the MAXIMAL robust invariant
set; the minimal robust positively invariant set of Rakovic et al. 2005 is a
different object and check 1 shows the two numbers side by side.
"""

from __future__ import annotations

import numpy as np

from invariantset import (
    Polytope,
    get_system,
    maximal_robust_invariant_set,
    verify_robust_invariance,
)


def line(label: str, got, expected, ok: bool) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label:<52s} got {got}   expected {expected}")


def main() -> None:
    failures = 0

    print("1. 1-D, lambda = 0.8, |w| <= 0.1, |x| <= 1")
    print("   threshold w/(1-|lambda|) = 0.1/0.2 = 0.5 <= 1, so S_inf = X")
    s = get_system("scalar_invariant_equals_x")
    res = maximal_robust_invariant_set(s.A, s.X, s.W)
    inv, margin = verify_robust_invariance(s.A, res.polytope, s.W)
    checks = [
        ("termination", res.termination, "converged", res.termination == "converged"),
        ("iterations", res.iterations, 1, res.iterations == 1),
        ("S_inf length", f"{res.polytope.volume():.12f}", "2.0",
         abs(res.polytope.volume() - 2.0) < 1e-12),
        ("worst facet margin", f"{margin:.12e}", "-0.1 exactly",
         abs(margin + 0.1) < 1e-12),
        ("MAXIMAL set half-width", f"{res.polytope.support([1.0]):.12f}", "1.0",
         abs(res.polytope.support([1.0]) - 1.0) < 1e-12),
        ("MINIMAL RPI half-width (different object)", f"{0.1 / (1 - 0.8):.12f}", "0.5",
         abs(0.1 / (1 - 0.8) - 0.5) < 1e-12),
    ]
    for label, got, exp, ok in checks:
        line(label, got, exp, ok)
        failures += not ok
    print(f"   the two differ by a factor of {1.0 / 0.5:.1f}; "
          "using 0.5 here would be the classic error")

    print("\n2. 1-D, lambda = 0.9, |w| <= 0.2, |x| <= 1: S_inf is EMPTY")
    print("   r_k = 2 - (10/9)^k, empty at the first k with r_k < 0")
    s = get_system("scalar_empty")
    res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=50, track_geometry=True)
    line("termination", res.termination, "empty", res.termination == "empty")
    failures += res.termination != "empty"
    line("iterations", res.iterations, 7, res.iterations == 7)
    failures += res.iterations != 7
    print("   k   measured half-width      closed form 2 - (10/9)^k    abs diff")
    for rec in res.history[:-1]:
        measured = rec.volume / 2.0
        closed = 2.0 - (10.0 / 9.0) ** rec.k
        ok = abs(measured - closed) < 1e-12
        failures += not ok
        print(
            f"   {rec.k}   {measured:.12f}            {closed:.12f}            "
            f"{abs(measured - closed):.3e}  {'PASS' if ok else 'FAIL'}"
        )
    print(f"   k=7 closed form would be {2.0 - (10.0 / 9.0) ** 7:.12f} < 0, hence empty")

    print("\n3. 1-D threshold: lambda = 0.5, |w| <= 0.25, |x| <= 0.5")
    print("   b = w/(1-lambda) = 0.5 exactly, so X is invariant with zero margin")
    A = np.array([[0.5]])
    res = maximal_robust_invariant_set(
        A, Polytope.from_box([0.0], [0.5]), Polytope.from_box([0.0], [0.25])
    )
    inv, margin = verify_robust_invariance(A, res.polytope, Polytope.from_box([0.0], [0.25]))
    line("termination", res.termination, "converged", res.termination == "converged")
    failures += res.termination != "converged"
    line("worst facet margin", f"{margin:.3e}", "0 exactly", margin == 0.0)
    failures += margin != 0.0
    res2 = maximal_robust_invariant_set(
        A, Polytope.from_box([0.0], [0.49]), Polytope.from_box([0.0], [0.25]), max_iter=200
    )
    line("b = 0.49 (just below threshold)", res2.termination, "empty",
         res2.termination == "empty")
    failures += res2.termination != "empty"

    print("\n4. 2-D singular A = [[0,1],[0,0]], box X half-width 1, box W half-width 0.1")
    print("   hand trace: Omega_1 = {|x1|<=1, |x2|<=0.9}, Omega_2 = Omega_1")
    s = get_system("nilpotent_2d")
    res = maximal_robust_invariant_set(s.A, s.X, s.W, track_geometry=True)
    inv, margin = verify_robust_invariance(s.A, res.polytope, s.W)
    for label, got, exp, ok in [
        ("termination", res.termination, "converged", res.termination == "converged"),
        ("iterations", res.iterations, 2, res.iterations == 2),
        ("facets", res.polytope.n_halfspaces, 4, res.polytope.n_halfspaces == 4),
        ("vertices", res.polytope.vertices().shape[0], 4,
         res.polytope.vertices().shape[0] == 4),
        ("h(e1)", f"{res.polytope.support([1.0, 0.0]):.12f}", "1.0",
         abs(res.polytope.support([1.0, 0.0]) - 1.0) < 1e-12),
        ("h(e2)", f"{res.polytope.support([0.0, 1.0]):.12f}", "0.9",
         abs(res.polytope.support([0.0, 1.0]) - 0.9) < 1e-12),
        ("area", f"{res.polytope.volume():.12f}", "3.6",
         abs(res.polytope.volume() - 3.6) < 1e-9),
        ("worst facet margin", f"{margin:.3e}", "0 exactly", margin == 0.0),
    ]:
        line(label, got, exp, ok)
        failures += not ok

    print("\n5. 2-D decoupled A = 0.5 I, box X half-width 1, box W half-width 0.1")
    print("   per-axis threshold 0.1/0.5 = 0.2 <= 1, so S_inf = X = [-1,1]^2")
    s = get_system("decoupled_2d")
    res = maximal_robust_invariant_set(s.A, s.X, s.W)
    inv, margin = verify_robust_invariance(s.A, res.polytope, s.W)
    for label, got, exp, ok in [
        ("iterations", res.iterations, 1, res.iterations == 1),
        ("area", f"{res.polytope.volume():.12f}", "4.0",
         abs(res.polytope.volume() - 4.0) < 1e-9),
        ("worst facet margin", f"{margin:.12f}", "-0.4 exactly", abs(margin + 0.4) < 1e-12),
    ]:
        line(label, got, exp, ok)
        failures += not ok

    print("\n6. attitude loop: gain derivation and measured outcome")
    s = get_system("attitude_loop")
    poles = np.sort_complex(np.linalg.eigvals(s.A))
    print(f"   trace(A_cl)                  = {np.trace(s.A):.12f}   [1.8]")
    print(f"   det(A_cl)                    = {np.linalg.det(s.A):.12f}   [0.82]")
    print(f"   closed-loop poles            = {poles[0]:.9f}, {poles[1]:.9f}   [0.9 +/- 0.1j]")
    print(f"   W half-widths [rad, rad/s]   = {s.W.support([1.0, 0.0]):.9e}, "
          f"{s.W.support([0.0, 1.0]):.9e}   [1.5e-4, 6.0e-3]")
    res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=50, track_geometry=True)
    inv, margin = verify_robust_invariance(s.A, res.polytope, s.W)
    x_inv, x_margin = verify_robust_invariance(s.A, s.X.remove_redundant(), s.W)
    area_X = s.X.volume()
    area_S = res.polytope.volume()
    print(f"   iterations                   = {res.iterations}")
    print(f"   facets of S_inf              = {res.polytope.n_halfspaces}")
    print(f"   vertices of S_inf            = {res.polytope.vertices().shape[0]}")
    print(f"   area of X    [rad.rad/s]     = {area_X:.12f}")
    print(f"   area of S_inf [rad.rad/s]    = {area_S:.12f}")
    print(f"   S_inf covers                 = {100.0 * area_S / area_X:.6f} % of X")
    print(f"   S_inf independently invariant= {inv}, worst facet margin {margin:.6e}")
    print(f"   X itself invariant           = {x_inv}, worst facet margin {x_margin:+.6e}")
    failures += not inv
    failures += x_inv

    print(f"\nFAILED CHECKS: {failures}")


if __name__ == "__main__":
    main()
