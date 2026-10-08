"""Check the support function and the set-algebra identities.

Every check compares an implementation against an independent computation:
the closed-form box support against the generic linear programme, the support
function against explicit vertex enumeration, and the Minkowski sum against
the support-additivity identity rather than against itself.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from invariantset import (
    Polytope,
    get_system,
    intersect,
    is_subset,
    maximal_robust_invariant_set,
    minkowski_sum,
    pontryagin_difference,
)

SEED = 59001


def _lp_support(P: Polytope, c: np.ndarray) -> float:
    res = linprog(-c, A_ub=P.A, b_ub=P.b, bounds=(None, None), method="highs")
    assert res.status == 0, res.message
    return float(-res.fun)


def _random_bounded(rng: np.random.Generator, dim: int) -> Polytope:
    half = rng.uniform(0.5, 2.0, size=dim)
    box = Polytope.from_box(np.zeros(dim), half)
    extra = rng.normal(size=(rng.integers(0, 4), dim))
    if extra.shape[0] == 0:
        return Polytope(box.A, box.b)
    extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
    rhs = rng.uniform(0.5, 3.0, size=extra.shape[0])
    return Polytope(np.vstack([box.A, extra]), np.concatenate([box.b, rhs]))


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"seed = {SEED}")

    # 1. Box closed form against the generic LP.
    worst = 0.0
    n = 0
    for dim in (1, 2, 3, 4):
        for _ in range(250):
            half = rng.uniform(0.1, 3.0, size=dim)
            centre = rng.uniform(-2.0, 2.0, size=dim)
            box = Polytope.from_box(centre, half)
            generic = Polytope(box.A, box.b)
            c = rng.normal(size=dim)
            worst = max(worst, abs(box.support(c) - _lp_support(generic, c)))
            n += 1
    print(f"1. box closed form vs generic LP      : {n} cases, worst abs diff {worst:.6e}")

    # 2. Support function against explicit vertex enumeration.
    worst = 0.0
    n = 0
    for dim in (1, 2, 3):
        for _ in range(60):
            P = _random_bounded(rng, dim)
            V = P.vertices()
            for _ in range(10):
                c = rng.normal(size=dim)
                worst = max(worst, abs(P.support(c) - float(np.max(V @ c))))
                n += 1
    print(f"2. support vs vertex enumeration      : {n} cases, worst abs diff {worst:.6e}")

    # 3. Positive homogeneity h(t c) = t h(c), Rockafellar 1970.
    worst = 0.0
    n = 0
    for _ in range(400):
        P = _random_bounded(rng, 2)
        c = rng.normal(size=2)
        t = float(rng.uniform(0.0, 5.0))
        lhs, rhs = P.support(t * c), t * P.support(c)
        worst = max(worst, abs(lhs - rhs) / max(abs(rhs), 1.0))
        n += 1
    print(f"3. positive homogeneity               : {n} cases, worst rel diff {worst:.6e}")

    # 4. Subadditivity h(c1 + c2) <= h(c1) + h(c2).
    worst = -np.inf
    n = 0
    for _ in range(400):
        P = _random_bounded(rng, 2)
        c1, c2 = rng.normal(size=2), rng.normal(size=2)
        worst = max(worst, P.support(c1 + c2) - P.support(c1) - P.support(c2))
        n += 1
    print(f"4. subadditivity slack (<=0 required) : {n} cases, worst {worst:.6e}")

    # 5. Minkowski additivity h_{P+Q} = h_P + h_Q, Schneider 1993.
    worst = 0.0
    n = 0
    for _ in range(120):
        P = _random_bounded(rng, 2)
        Q = Polytope.from_box(np.zeros(2), rng.uniform(0.02, 0.3, size=2))
        S = minkowski_sum(P, Q)
        for _ in range(8):
            c = rng.normal(size=2)
            worst = max(worst, abs(S.support(c) - P.support(c) - Q.support(c)))
            n += 1
    print(f"5. Minkowski support additivity       : {n} cases, worst abs diff {worst:.6e}")

    # 6. Identity (I3): (P - Q) + Q subset P.  Worst escape margin must be <=0.
    worst = -np.inf
    n = 0
    deficits = []
    for _ in range(120):
        P = _random_bounded(rng, 2)
        Q = Polytope.from_box(np.zeros(2), rng.uniform(0.02, 0.15, size=2))
        D = pontryagin_difference(P, Q)
        if D.is_empty():
            continue
        R = minkowski_sum(D, Q)
        worst = max(worst, is_subset(R, P)[1])
        deficits.append(1.0 - R.volume() / P.volume())
        n += 1
    print(f"6. (P-Q)+Q subset P, worst escape     : {n} cases, worst margin {worst:.6e}")
    print(
        f"   area deficit of (P-Q)+Q vs P       : median {np.median(deficits) * 100:.4f} %, "
        f"max {np.max(deficits) * 100:.4f} %, min {np.min(deficits) * 100:.4f} %"
    )

    # 7. Identity (I4): (P + Q) - Q = P exactly, both directions.
    worst = -np.inf
    n = 0
    for _ in range(120):
        P = _random_bounded(rng, 2)
        Q = Polytope.from_box(np.zeros(2), rng.uniform(0.02, 0.15, size=2))
        C = pontryagin_difference(minkowski_sum(P, Q), Q)
        worst = max(worst, is_subset(C, P)[1], is_subset(P, C)[1])
        n += 1
    print(f"7. (P+Q)-Q == P, worst either way     : {n} cases, worst margin {worst:.6e}")

    # 8. The deterministic witness: a triangle, hand-computed.
    P = Polytope(
        np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]), np.array([0.0, 0.0, 1.0])
    )
    Q = Polytope.from_box([0.0, 0.0], [0.1, 0.1])
    D = pontryagin_difference(P, Q)
    R = minkowski_sum(D, Q)
    C = pontryagin_difference(minkowski_sum(P, Q), Q)
    print("8. worked triangle case (hand-computed targets in brackets)")
    print(f"   area(P)                            = {P.volume():.9f}   [0.5]")
    print(f"   area(P (-) Q)                      = {D.volume():.9f}   [0.18]")
    print(f"   area((P (-) Q) (+) Q)              = {R.volume():.9f}   [0.46]")
    print(f"   area deficit                       = {P.volume() - R.volume():.9f}   [0.04]")
    print(
        f"   deficit as a fraction of P         = "
        f"{100.0 * (1.0 - R.volume() / P.volume()):.6f} %   [8 %]"
    )
    print(f"   area((P (+) Q) (-) Q)              = {C.volume():.9f}   [0.5]")
    print(f"   P subset (P (-) Q) (+) Q           = {is_subset(P, R)[0]}   [False]")
    print(f"   (P (-) Q) (+) Q subset P           = {is_subset(R, P)[0]}   [True]")

    # 9. Intersection support bound.
    worst = -np.inf
    n = 0
    for _ in range(300):
        P = _random_bounded(rng, 2)
        Q = _random_bounded(rng, 2)
        R = intersect(P, Q)
        c = rng.normal(size=2)
        worst = max(worst, R.support(c) - min(P.support(c), Q.support(c)))
        n += 1
    print(f"9. h_(P&Q) <= min(h_P, h_Q), worst    : {n} cases, worst {worst:.6e}")

    # 10. Redundancy removal preserves membership.
    mismatches = 0
    removed = 0
    n = 0
    for _ in range(200):
        P = _random_bounded(rng, 2)
        R = P.remove_redundant(tol=1e-10)
        removed += P.n_halfspaces - R.n_halfspaces
        for x in rng.uniform(-3.0, 3.0, size=(20, 2)):
            slack = float(np.min(P.b - P.A @ x))
            if abs(slack) < 1e-7:
                continue
            n += 1
            if P.contains(x, tol=0.0) != R.contains(x, tol=0.0):
                mismatches += 1
    print(
        f"10. redundancy removal membership     : {n} points over 200 polytopes, "
        f"{mismatches} mismatches, {removed} rows removed"
    )

    # 11. When DOES (P (-) Q) (+) Q equal P?  The textbook answer is "when Q is
    # a summand of P", i.e. P = R (+) Q for some convex R (Schneider 1993,
    # summands and decomposition).  For a box Q in the plane that shows up as
    # a concrete test: the opening returns P when the four axis directions are
    # already facet normals of P and no facet is lost in the erosion.
    print("11. when the assumed identity happens to hold")
    box = Polytope.from_box([0.0, 0.0], [0.1, 0.1])
    triangle = Polytope(
        np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]), np.array([0.0, 0.0, 1.0])
    )
    pentagon = Polytope(
        np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([0.0, 0.0, 1.0, 0.8, 0.8]),
    )
    for label, poly in (
        ("triangle (e1, e2 are NOT facet normals)", triangle),
        ("pentagon (e1, e2 ARE facet normals)", pentagon),
    ):
        opened = minkowski_sum(pontryagin_difference(poly, box), box)
        print(
            f"    {label:<42s} area {poly.volume():.9f} -> {opened.volume():.9f}, "
            f"deficit {poly.volume() - opened.volume():+.3e} "
            f"({100.0 * (1.0 - opened.volume() / poly.volume()):+.4f} %)"
        )
    print("    the opened triangle IS the pentagon: 0.46 either way, and the")
    print("    pentagon is a fixed point of the opening while the triangle is not.")
    print("    the same effect on the shipped S_inf, as W is scaled up:")
    attitude = get_system("attitude_loop")
    S = maximal_robust_invariant_set(attitude.A, attitude.X, attitude.W).polytope
    for scale in (1, 20, 40, 60):
        Q = Polytope.from_box([0.0, 0.0], [1.5e-4 * scale, 6.0e-3 * scale])
        eroded = pontryagin_difference(S, Q)
        if eroded.is_empty():
            print(f"    W x {scale:<3d}: erosion is empty")
            continue
        kept = eroded.remove_redundant().n_halfspaces
        opened = minkowski_sum(eroded, Q)
        print(
            f"    W x {scale:<3d}: S facets {S.n_halfspaces} -> eroded facets {kept:<3d} "
            f"area {S.volume():.9f} -> {opened.volume():.9f}, deficit "
            f"{100.0 * (1.0 - opened.volume() / S.volume()):+.4f} %"
        )


if __name__ == "__main__":
    main()
