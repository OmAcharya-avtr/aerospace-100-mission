"""Validation 1: the support-function and set algebra the switching condition uses.

Everything else in this package is a consequence of these identities being
correct, so each is checked against an independent computation rather than
asserted.

Checks
  1  Box closed form against the generic linear programme, 4000 random directions
     in dimensions 1 to 5.
  2  Box support against explicit vertex enumeration, which is the definition.
  3  Positive homogeneity and subadditivity over random directions.
  4  Minkowski additivity of the support function on boxes.
  5  Pontryagin difference against vertex-shift enumeration, 20000 random points.
  6  Monotonicity of erosion in the disturbance set.
  7  Preimage membership identity under random linear maps.
  8  Redundancy removal preserves the set, checked by membership on random points.

Runtime: about 25 s on one contended core.
"""

from __future__ import annotations

import itertools
import time

from _bootstrap import add_src_to_path

add_src_to_path()

import numpy as np  # noqa: E402

from simplexguard.polytope import Box, Polytope, box  # noqa: E402

SEED = 51051
rng = np.random.default_rng(SEED)
start = time.perf_counter()
failures = 0


def report(name: str, ok: bool, detail: str) -> None:
    global failures
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures += 1
    print(f"[{status}] {name}: {detail}")


print(f"validate_support_function.py   seed = {SEED}")
print("=" * 78)

# --- check 1: box closed form against the generic LP ------------------------
worst = 0.0
count = 0
for n in (1, 2, 3, 4, 5):
    for _ in range(800):
        centre = rng.uniform(-2.0, 2.0, n)
        radius = rng.uniform(0.0, 3.0, n)
        b = Box(centre - radius, centre + radius)
        generic = Polytope(b.A, b.b)
        c = rng.normal(size=n)
        closed, lp = b.support(c), generic.support(c)
        worst = max(worst, abs(closed - lp) / (1.0 + abs(lp)))
        count += 1
report(
    "check 1  box closed form vs generic LP",
    worst <= 1e-9,
    f"{count} directions in dim 1-5, worst relative difference {worst:.3e}, tolerance 1e-9",
)

# --- check 2: box support against vertex enumeration ------------------------
worst = 0.0
count = 0
for n in (1, 2, 3, 4):
    signs = np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
    for _ in range(500):
        centre = rng.uniform(-1.0, 1.0, n)
        radius = rng.uniform(0.0, 2.0, n)
        b = Box(centre - radius, centre + radius)
        verts = b.lower + signs * (b.upper - b.lower)
        c = rng.normal(size=n)
        worst = max(worst, abs(b.support(c) - float(np.max(verts @ c))))
        count += 1
report(
    "check 2  box support vs vertex enumeration",
    worst <= 1e-12,
    f"{count} directions in dim 1-4, worst absolute difference {worst:.3e}, tolerance 1e-12",
)

# --- check 3: homogeneity and subadditivity --------------------------------
worst_hom = worst_sub = 0.0
for _ in range(3000):
    b = box(rng.uniform(0.01, 3.0, 3))
    c1, c2 = rng.normal(size=3), rng.normal(size=3)
    t = float(rng.uniform(0.0, 5.0))
    expected = t * b.support(c1)
    worst_hom = max(worst_hom, abs(b.support(t * c1) - expected) / (1.0 + abs(expected)))
    gap = b.support(c1) + b.support(c2) - b.support(c1 + c2)
    worst_sub = min(worst_sub, gap) if gap < worst_sub else worst_sub
report(
    "check 3a positive homogeneity h(t c) = t h(c)",
    worst_hom <= 1e-12,
    f"3000 cases, worst relative difference {worst_hom:.3e}, tolerance 1e-12",
)
report(
    "check 3b subadditivity h(c1+c2) <= h(c1)+h(c2)",
    worst_sub >= -1e-12,
    f"3000 cases, most negative slack {worst_sub:.3e}, tolerance -1e-12",
)

# --- check 4: Minkowski additivity on boxes --------------------------------
worst = 0.0
for _ in range(2000):
    r1, r2 = rng.uniform(0.0, 2.0, 2), rng.uniform(0.0, 2.0, 2)
    p, w, total = box(r1), box(r2), box(r1 + r2)
    c = rng.normal(size=2)
    expected = p.support(c) + w.support(c)
    worst = max(worst, abs(total.support(c) - expected) / (1.0 + abs(expected)))
report(
    "check 4  h_{P+W}(c) = h_P(c) + h_W(c)",
    worst <= 1e-12,
    f"2000 cases, worst relative difference {worst:.3e}, tolerance 1e-12",
)

# --- check 5: Pontryagin difference against vertex shifts -------------------
mismatches = 0
tested = 0
boundary_skipped = 0
for _ in range(100):
    rows = np.vstack([np.eye(2), -np.eye(2), rng.normal(size=(2, 2))])
    offs = np.concatenate([rng.uniform(0.5, 2.0, 4), rng.uniform(0.5, 2.0, 2)])
    p = Polytope(rows, offs)
    w = box(rng.uniform(0.01, 0.2, 2))
    eroded = p.erode(w)
    verts = w.vertices()
    pts = rng.uniform(-3.0, 3.0, size=(200, 2))
    for x in pts:
        if abs(eroded.slack(x)) <= 1e-9:
            boundary_skipped += 1
            continue
        tested += 1
        if eroded.contains(x, tol=0.0) != all(p.contains(x + v, tol=1e-12) for v in verts):
            mismatches += 1
report(
    "check 5  P (-) W by support function vs vertex shifts",
    mismatches == 0,
    f"{tested} points over 100 random polytopes, {mismatches} mismatches "
    f"({boundary_skipped} boundary points skipped)",
)

# --- check 6: erosion monotonicity -----------------------------------------
violations = 0
for _ in range(400):
    p = box(rng.uniform(0.5, 3.0, 2))
    r = rng.uniform(0.01, 0.3, 2)
    small, large = box(r), box(r * float(rng.uniform(1.1, 4.0)))
    if not p.erode(small).contains_polytope(p.erode(large), tol=1e-9):
        violations += 1
report(
    "check 6  W1 subset W2 implies P(-)W2 subset P(-)W1",
    violations == 0,
    f"400 nested pairs, {violations} violations",
)

# --- check 7: preimage identity --------------------------------------------
mismatches = 0
tested = 0
for _ in range(200):
    p = box(rng.uniform(0.5, 2.0, 2))
    m = rng.normal(size=(2, 2))
    pre = p.preimage(m)
    for x in rng.uniform(-4.0, 4.0, size=(50, 2)):
        if abs(p.slack(m @ x)) <= 1e-9:
            continue
        tested += 1
        if pre.contains(x, tol=1e-12) != p.contains(m @ x, tol=1e-12):
            mismatches += 1
report(
    "check 7  x in M^-1 P  iff  M x in P",
    mismatches == 0,
    f"{tested} points over 200 random maps, {mismatches} mismatches",
)

# --- check 8: redundancy removal preserves the set -------------------------
mismatches = 0
tested = 0
removed_total = 0
for _ in range(200):
    core = np.vstack([np.eye(2), -np.eye(2)])
    radius = rng.uniform(0.5, 2.0, 2)
    extra_rows = rng.normal(size=(6, 2))
    extra_offs = np.array(
        [Polytope(core, np.concatenate([radius, radius])).support(r) + rng.uniform(0.0, 1.0)
         for r in extra_rows]
    )
    p = Polytope(np.vstack([core, extra_rows]), np.concatenate([radius, radius, extra_offs]))
    reduced = p.minimal()
    removed_total += p.n_halfspaces - reduced.n_halfspaces
    for x in rng.uniform(-3.0, 3.0, size=(40, 2)):
        if abs(p.slack(x)) <= 1e-9:
            continue
        tested += 1
        if p.contains(x, tol=1e-12) != reduced.contains(x, tol=1e-12):
            mismatches += 1
report(
    "check 8  minimal() preserves membership",
    mismatches == 0,
    f"{tested} points over 200 polytopes, {mismatches} mismatches, "
    f"{removed_total} redundant rows removed in total",
)

print("=" * 78)
print(f"failed checks: {failures}")
print(f"elapsed: {time.perf_counter() - start:.1f} s")
raise SystemExit(1 if failures else 0)
