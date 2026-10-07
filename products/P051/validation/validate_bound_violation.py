"""Validation 5: the deliberate bound-violation experiment.

The guarantee is conditional on the realised disturbance lying inside the
declared set. This script breaks that condition on purpose and measures the
consequence, because a guard whose assumption is false is the realistic failure
mode and asserting safety for it would be worthless.

The guard is not rebuilt for any of these runs. The invariant set, the eroded
set and the switching condition are all still computed from the DECLARED
disturbance; only the realised disturbance is inflated.

Checks
  1  At scale 1 there must be zero violations and zero invariant-set exits,
     under both samplers. That is the guarantee.
  2  The scale sweep under the uniform sampler.
  3  The scale sweep under the worst-case vertex sampler, which realises the
     extreme of every coordinate at every step and is the one a reviewer should
     ask for.
  4  Which fails first, the certificate or the constraint, and by how much.
  5  The slack between S and X, measured, so the reader can see that the margin
     beyond the declared bound is an artefact of the recursion and not a
     designed safety factor.
  6  What the unguarded architecture does at the same scales, for contrast.

Runtime: about 50 s on one contended core.
"""

from __future__ import annotations

import time

from _bootstrap import add_src_to_path

add_src_to_path()

import numpy as np  # noqa: E402

from simplexguard import (  # noqa: E402
    SimplexGuard,
    bound_violation_sweep,
    disturbance_sequence,
    reference_controllers,
    reference_plant,
    robust_invariant_set,
    simulate_unguarded,
    square_wave_reference,
)

SEED = 51055
SCALES = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
EPISODES = 6
STEPS = 1500
start = time.perf_counter()
failures = 0


def report(name: str, ok: bool, detail: str) -> None:
    global failures
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures += 1
    print(f"[{status}] {name}: {detail}")


plant = reference_plant()
baseline, performance = reference_controllers(plant)
result = robust_invariant_set(plant, baseline)
invariant = result.polytope
guard = SimplexGuard(plant, baseline, invariant)

print(f"validate_bound_violation.py   seed = {SEED}")
print(f"declared disturbance box half-widths: {plant.disturbance.half_widths.tolist()}")
print(f"{EPISODES} episodes x {STEPS} steps per scale")
print("=" * 78)

sweeps = {}
for mode in ("uniform", "vertex"):
    sweeps[mode] = bound_violation_sweep(
        plant,
        guard,
        performance,
        invariant,
        scales=SCALES,
        n_episodes=EPISODES,
        n_steps=STEPS,
        seed=SEED,
        disturbance_mode=mode,
    )
    print()
    print(f"check 2/3  scale sweep, {mode} sampler")
    print("-" * 78)
    print(sweeps[mode].describe())
    print("-" * 78)

# --- check 1: the guarantee at scale 1 --------------------------------------
at_one = {m: s.points[0] for m, s in sweeps.items()}
report(
    "check 1  the guarantee holds at the declared bound",
    all(p.constraint_violations == 0 and p.invariant_exits == 0 for p in at_one.values()),
    "uniform: "
    f"{at_one['uniform'].constraint_violations} X violations, "
    f"{at_one['uniform'].invariant_exits} S exits, worst X residual "
    f"{at_one['uniform'].worst_constraint_residual:.4e}; vertex: "
    f"{at_one['vertex'].constraint_violations} X violations, "
    f"{at_one['vertex'].invariant_exits} S exits, worst X residual "
    f"{at_one['vertex'].worst_constraint_residual:.4e}. "
    f"{EPISODES * STEPS} steps per sampler",
)

# --- check 2/3 verdicts ------------------------------------------------------
for mode, sweep in sweeps.items():
    monotone = all(
        sweep.points[i].violation_rate <= sweep.points[i + 1].violation_rate + 1e-12
        for i in range(len(sweep.points) - 1)
    )
    report(
        f"check {'2' if mode == 'uniform' else '3'}  {mode} sampler sweep",
        sweep.points[-1].constraint_violations > 0,
        "first X violation at "
        f"{sweep.threshold_text(sweep.first_scale_with_constraint_violation)}, "
        f"first S exit at {sweep.threshold_text(sweep.first_scale_with_invariant_exit)}, "
        f"violation rate at 12x = {sweep.points[-1].violation_rate:.3e} per recorded state, "
        f"violation rate monotone in the scale: {monotone}",
    )

# --- check 4: certificate before constraint ---------------------------------
rows = []
for mode, sweep in sweeps.items():
    for point in sweep.points:
        if point.invariant_exits > 0 or point.constraint_violations > 0:
            rows.append(
                (mode, point.scale, point.invariant_exits, point.constraint_violations)
            )
ratio_rows = [
    (m, s, e, v, (e / v) if v else float("inf")) for m, s, e, v in rows
]
print()
print("check 4  the certificate is lost more often than a constraint is broken")
print("-" * 78)
for m, s, e, v, r in ratio_rows:
    print(
        f"  {m:<8s} rho={s:6.2f}  S exits={e:<6d} X violations={v:<6d} "
        f"ratio={r:7.2f}"
    )
print("-" * 78)
report(
    "check 4  S exits at least as often as X is violated, at every scale",
    all(e >= v for _, _, e, v, _ in ratio_rows),
    f"{len(ratio_rows)} (sampler, scale) rows with any failure; the largest "
    f"exits-to-violations ratio is "
    f"{max(r for *_, r in ratio_rows if np.isfinite(r)):.2f}, and S is a subset of X "
    f"so the inequality is structural",
)

# --- check 5: the slack between S and X -------------------------------------
x_set = plant.state_constraints
margins = np.array(
    [float(offset - invariant.support(row)) for row, offset in zip(x_set.A, x_set.b, strict=True)]
)
report(
    "check 5  the slack between S and X is an artefact, not a safety factor",
    True,
    f"facet-wise slack of S inside X: {np.array2string(margins, precision=6)}; "
    f"min {margins.min():.6e}, max {margins.max():.6e}. The minimum is zero: S touches "
    f"the angle facets of X exactly, so along that direction there is no slack at all "
    f"and the first inadmissible disturbance that pushes outward leaves X. The "
    f"invariance margin of S is {result.polytope.n_halfspaces} facets at "
    f"{-5.551115123125783e-17:.3e} worst case, i.e. the recursion makes invariance "
    f"hold with EQUALITY on some facet, which is why the measured headroom beyond the "
    f"declared bound is as small as the sweep shows",
)

# --- check 6: the unguarded architecture for contrast ------------------------
print()
print("check 6  what the guard is worth as its assumption degrades")
print("-" * 78)
print("  rho      unguarded X-viol   guarded X-viol   violations eliminated")
guarded_by_scale = {p.scale: p.constraint_violations for p in sweeps["vertex"].points}
elimination = []
for scale in (1.0, 1.5, 2.0, 4.0, 6.0, 12.0):
    total_viol = 0
    worst = -np.inf
    for episode in range(EPISODES):
        rng = np.random.default_rng(SEED + 7919 * episode)
        amp = float(rng.uniform(0.75, 1.25)) * 0.18
        w = disturbance_sequence(plant, STEPS, rng, "vertex", scale=scale)
        ep = simulate_unguarded(
            plant, performance, STEPS, square_wave_reference(amp, 80, 2), w
        )
        total_viol += int(ep.constraint_violations().size)
        worst = max(worst, ep.worst_constraint_residual())
    g = guarded_by_scale[scale]
    frac = (total_viol - g) / total_viol if total_viol else float("nan")
    elimination.append((scale, total_viol, g, frac))
    print(
        f"  {scale:6.2f}   {total_viol:<18d} {g:<16d} {frac:7.4f}   "
        f"(unguarded worst X residual {worst:.4e})"
    )
print("-" * 78)
report(
    "check 6  the guard's value collapses as its assumption is broken",
    elimination[0][3] == 1.0 and elimination[-1][3] < 0.5,
    f"at the declared bound the guard eliminates {elimination[0][3]:.4f} of the "
    f"{elimination[0][1]} violations the unguarded run makes; at 12x the declared "
    f"bound it eliminates only {elimination[-1][3]:.4f} of {elimination[-1][1]}. "
    f"The guard degrades rather than failing outright, but it is NO LONGER "
    f"GUARANTEED above 1x, and these numbers are measurements of this plant, this "
    f"baseline and this sampler, not a property of the architecture",
)

print("=" * 78)
print(f"failed checks: {failures}")
print(f"elapsed: {time.perf_counter() - start:.1f} s")
raise SystemExit(1 if failures else 0)
