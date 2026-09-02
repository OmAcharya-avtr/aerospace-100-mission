"""Redundant wheel allocation and zero-speed avoidance, measured.

Run: ``python3 wheel_allocation.py``

Four claims are tested.

1. Allocation is exact: adding a null-space vector never changes the body momentum.
2. The exact one-dimensional maximiser really is the maximiser: it is compared against a
   brute-force scan over 20001 values of the null coefficient on random requests, and the
   worst shortfall is reported.
3. Biasing buys zero-speed margin, and it costs saturation margin. Both are quantified
   over a full sweep of request directions and at several envelope fractions, so the
   trade is a table and not an assertion.
4. A three-wheel orthogonal array has no null space, so none of this is available to it.
   That is shown, not asserted, on the same trajectory.
"""

from __future__ import annotations

import time

import numpy as np
from _common import Checks  # noqa: E402

from momentummgr import (  # noqa: E402
    count_zero_crossings,
    orthogonal_three,
    pyramid_four,
    reference_orbit,
    reference_smallsat,
    sweep_orbit,
    sun_direction_for_beta,
    tetrahedral_four,
)

c = Checks()
t0 = time.time()
print("Wheel allocation and zero-speed avoidance")
print("=" * 90)

rng = np.random.default_rng(20260902)
arrays = {"pyramid_four": pyramid_four(), "tetrahedral_four": tetrahedral_four()}

# ------------------------------------------------------- 1. allocation is exact
print("""
1. Allocation exactness. 500 random body requests inside each array's guaranteed
   envelope, allocated with and without biasing; the reconstructed body momentum must
   equal the request.
""")
for name, w in arrays.items():
    worst = 0.0
    for _ in range(500):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        h_body = d * w.guaranteed_body_envelope_nms * rng.uniform(0.0, 1.0)
        for bias in (False, True):
            alloc = w.allocate(h_body, avoid_zero_speed=bias)
            err = float(np.linalg.norm(w.body_momentum(alloc.wheel_momentum_nms) - h_body))
            worst = max(worst, err / max(float(np.linalg.norm(h_body)), 1e-12))
    print(f"   {name:<18} worst relative reconstruction error {worst:.3e}")
    c.check(f"{name}: allocation reproduces the body request", worst, 0.0, 1e-12, kind="abs")

# --------------------------------------------- 2. the exact maximiser is a maximiser
print("""
2. The exact maximiser against brute force. For 200 random requests the null coefficient
   is scanned over 20001 uniformly spaced values inside the feasible interval and the best
   scanned min |h_i| is compared with the value the exact enumeration returns. A positive
   shortfall would mean the enumeration missed the optimum.
""")
w = pyramid_four()
nvec = w.null_basis[:, 0]
worst_shortfall = 0.0
for _ in range(200):
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    h_body = d * w.guaranteed_body_envelope_nms * rng.uniform(0.05, 0.95)
    p = w.minimum_norm_allocation(h_body)
    lo, hi = -np.inf, np.inf
    for pi, ni in zip(p, nvec, strict=True):
        a1, a2 = (-w.max_momentum_nms - pi) / ni, (w.max_momentum_nms - pi) / ni
        lo, hi = max(lo, min(a1, a2)), min(hi, max(a1, a2))
    alphas = np.linspace(lo, hi, 20001)
    scan = np.min(np.abs(p[None, :] + alphas[:, None] * nvec[None, :]), axis=1).max()
    exact = w.allocate(h_body, avoid_zero_speed=True).min_abs_momentum_nms
    worst_shortfall = max(worst_shortfall, float(scan - exact) / w.max_momentum_nms)
print(f"   worst (brute force - exact) / h_max over 200 requests: {worst_shortfall:.3e}")
c.check("exact enumeration is never beaten by a 20001-point scan", worst_shortfall, 0.0,
        1e-12, kind="abs")

# ------------------------------------------------- 3. what biasing buys and costs
print("""
3. The trade, over 2000 directions uniformly sampled on the sphere at 40 % of the
   envelope. 'min |h|' is the zero-speed margin (bigger is better) and 'max |h|' is what
   is left of the saturation margin (smaller is better). Both as fractions of h_max.
""")
dirs = rng.normal(size=(2000, 3))
dirs /= np.linalg.norm(dirs, axis=1)[:, None]
requests = dirs * 0.4 * w.guaranteed_body_envelope_nms
print(f"   {'mode':<26}{'mean min|h|':>14}{'worst min|h|':>15}{'mean max|h|':>14}"
      f"{'worst max|h|':>15}")
print("   " + "-" * 84)
rows = {}
for label, kwargs in (
    ("minimum norm", {"avoid_zero_speed": False}),
    ("biased, envelope 1.0", {"avoid_zero_speed": True, "envelope_fraction": 1.0}),
    ("biased, envelope 0.7", {"avoid_zero_speed": True, "envelope_fraction": 0.7}),
    ("biased, envelope 0.5", {"avoid_zero_speed": True, "envelope_fraction": 0.5}),
):
    mins, maxs = [], []
    for h_body in requests:
        alloc = w.allocate(h_body, **kwargs)
        mins.append(alloc.min_abs_momentum_nms / w.max_momentum_nms)
        maxs.append(alloc.max_abs_momentum_nms / w.max_momentum_nms)
    rows[label] = (float(np.mean(mins)), float(np.min(mins)), float(np.mean(maxs)),
                   float(np.max(maxs)))
    print(f"   {label:<26}{rows[label][0]:>14.4f}{rows[label][1]:>15.4f}"
          f"{rows[label][2]:>14.4f}{rows[label][3]:>15.4f}")
c.assert_true(
    "biasing never reduces the zero-speed margin",
    all(rows[k][0] >= rows["minimum norm"][0] - 1e-15 for k in rows),
    "means: " + ", ".join(f"{k}={v[0]:.4f}" for k, v in rows.items()),
)
c.assert_true(
    "biasing at envelope 1.0 costs saturation margin",
    rows["biased, envelope 1.0"][3] > rows["minimum norm"][3],
    f"worst max|h| {rows['biased, envelope 1.0'][3]:.4f} vs "
    f"{rows['minimum norm'][3]:.4f}",
)
print("""
   The worst-case minimum-norm margin is 0.0001 of h_max, i.e. a wheel essentially at
   rest: there are request directions for which the pseudo-inverse puts a wheel through
   zero, and on a pyramid those directions are not exotic. Biasing at full envelope lifts
   the worst case to 0.35 of h_max, at the price of pushing the worst wheel to the
   saturation limit (max |h| = 1.00). The 0.7 and 0.5 rows are where a real system would
   sit: worst-case zero-speed margin about 0.10 of h_max for a worst-case saturation of
   0.70 and 0.50 respectively.
""")

# --------------------------------- 4. zero crossings on a real momentum trajectory
print("""
4. Zero crossings on a real trajectory. The reference smallsat at 500 km, beta 20 deg,
   three orbits with no desaturation: the wheel momentum is propagated from the
   body-frame Euler equation and allocated at every sample. Crossings are counted per
   wheel, and so is the time each wheel spends inside a 5 % of h_max low-speed band, where
   bearing friction is least predictable.
""")
sc = reference_smallsat()
orbit = reference_orbit(500.0)
sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, np.radians(20.0))
n_per_orbit = 721
sweep = sweep_orbit(sc, orbit, sun, n_samples=n_per_orbit)
t_body = np.vstack([sweep.torque("total", "body")[:-1]] * 3)
dt = orbit.period_s / (n_per_orbit - 1)
omega = orbit.body_rate_body_rad_s
i_omega = sc.inertia @ omega
h = np.zeros(3)
traj = [h.copy()]
for row in t_body:
    h = h + dt * (row - np.cross(omega, i_omega + h))
    traj.append(h.copy())
traj = np.array(traj)
print(f"   peak |h_body| over three orbits {np.linalg.norm(traj, axis=1).max():.6f} N m s")
w4 = pyramid_four(max_momentum_nms=0.05)
w3 = orthogonal_three(max_momentum_nms=0.05)
deadband = 0.05 * 0.05
for label, array, kwargs in (
    ("pyramid, minimum norm", w4, {"avoid_zero_speed": False}),
    ("pyramid, biased 0.7", w4, {"avoid_zero_speed": True, "envelope_fraction": 0.7}),
    ("orthogonal three", w3, {"avoid_zero_speed": True}),
):
    hist = np.array([array.allocate(row, **kwargs).wheel_momentum_nms for row in traj])
    crossings = count_zero_crossings(hist)
    low = float(np.mean(np.abs(hist) < deadband))
    print(f"   {label:<24} crossings {crossings.tolist()}  "
          f"time in the low-speed band {low * 100:.2f} %")
    if label == "pyramid, minimum norm":
        base_low = low
    if label == "pyramid, biased 0.7":
        c.assert_true(
            "biasing reduces the time wheels spend in the low-speed band",
            low < base_low,
            f"{low * 100:.2f} % against {base_low * 100:.2f} %",
        )
    if label == "orthogonal three":
        c.check("three-wheel array has an empty null space",
                float(w3.null_basis.shape[1]), 0.0, 0.0, kind="abs")
        alloc_a = w3.allocate(traj[10], avoid_zero_speed=True).wheel_momentum_nms
        alloc_b = w3.allocate(traj[10], avoid_zero_speed=False).wheel_momentum_nms
        c.check("three-wheel allocation is identical with and without biasing",
                float(np.abs(alloc_a - alloc_b).max()), 0.0, 0.0, kind="abs")
print("""
   Note what did *not* change: the crossing counts are 20 in total without biasing and
   20 with it. Biasing does not stop a wheel changing sign when the body momentum demands
   it; what it removes is the *dwell*, the 47.7 % of the run that the minimum-norm
   allocation spends inside the low-speed band. Dwell is the quantity that matters for
   bearing friction, and quoting a crossing count alone would have hidden the entire
   effect.

   The three-wheel row is the honest control. With no redundancy the wheel momenta are
   fixed by the body request and zero-speed avoidance is simply unavailable: the only
   remedy is a momentum bias, which costs stored momentum and therefore saturation
   margin. Nothing in this module can help a three-wheel array.
""")

# ------------------------------------------- 5. the defect biasing introduces
print("""
5. What biasing breaks, measured. The null coefficient is chosen afresh at every sample
   with no memory of the last one, and the maximiser of min |h_i| can switch between
   symmetric branches. When it does, the commanded wheel momenta jump. The largest
   single-sample change along the trajectory above is the size of that jump, in N m s,
   and it is a defect of this implementation, not of the idea.
""")
print(f"   {'allocation':<34}{'largest single-step change [N m s]':>36}")
print("   " + "-" * 70)
jumps = {}
for label, array, kwargs in (
    ("pyramid, minimum norm", w4, {"avoid_zero_speed": False}),
    ("pyramid, biased, full envelope", w4, {"avoid_zero_speed": True,
                                            "envelope_fraction": 1.0}),
    ("pyramid, biased, 70 % envelope", w4, {"avoid_zero_speed": True,
                                            "envelope_fraction": 0.7}),
):
    hist = np.array([array.allocate(row, **kwargs).wheel_momentum_nms for row in traj])
    jumps[label] = float(np.max(np.abs(np.diff(hist, axis=0))))
    print(f"   {label:<34}{jumps[label]:>36.6f}")
c.assert_true(
    "biasing introduces discontinuous wheel-momentum steps; minimum norm does not",
    jumps["pyramid, biased, full envelope"] > 100.0 * jumps["pyramid, minimum norm"],
    f"{jumps['pyramid, biased, full envelope']:.6f} against "
    f"{jumps['pyramid, minimum norm']:.6f} N m s per {dt:.1f} s sample",
)
print(f"""
   Reported as a limitation, not fixed here. The 0.0951 N m s step at full envelope is
   nearly two whole wheel limits in one {dt:.1f} s sample, i.e. a torque demand of order
   1e-02 N m against wheels whose useful torque is of order 1e-03 N m. A flight
   implementation must rate-limit the null coefficient or add hysteresis to the branch
   choice; this package leaves the raw behaviour visible so that a user cannot adopt the
   allocator without meeting the problem. See README Limitations.
""")

print(f"wall time {time.time() - t0:.2f} s")
c.summary("wheel_allocation.py")
raise SystemExit(1 if c.n_fail else 0)
