"""Validation 3: the switching condition is exact, not approximate.

The claim under test is that inequality (2) of ``simplexguard/guard.py``,

    c_j^T (A x + B u) + h_W(c_j) <= d_j   for every facet j of S,

is equivalent to "``A x + B u + w`` lies in ``S`` for every ``w`` in ``W``", with
no sampling and no tuned margin. It is checked against two independent
computations and in both directions.

Checks
  1  Against explicit enumeration of the 4 vertices of the box W, over 40000
     random (state, input) pairs including pairs deliberately placed on the
     boundary of the eroded set.
  2  Against a dense sampling of W (not just its vertices), which cannot prove
     the condition but can falsify it: an admitted input whose successor leaves S
     for some sampled w would be a counterexample.
  3  The margin's sign agrees with the boolean verdict, and the margin is a
     distance: moving the nominal successor state by the margin along the binding
     facet normal must put it exactly on the facet.
  4  Horizon-0 of the exact multi-step predictor reproduces the one-step
     condition on the unsaturated performance input, over 20000 random states.
  5  The guard never admits an input that leaves S under a declared disturbance,
     over complete simulated episodes with the worst-case vertex sampler.
  6  The two exact additional guards: an input outside U and a state outside S
     are both refused and both flagged.

Runtime: about 35 s on one contended core.
"""

from __future__ import annotations

import time

from _bootstrap import add_src_to_path

add_src_to_path()

import numpy as np  # noqa: E402

from simplexguard import (  # noqa: E402
    ExactLeadPredictor,
    Mode,
    SimplexGuard,
    disturbance_sequence,
    reference_controllers,
    reference_plant,
    robust_invariant_set,
    simulate_guarded,
    square_wave_reference,
)

SEED = 51053
rng = np.random.default_rng(SEED)
start = time.perf_counter()
failures = 0


def report(name: str, ok: bool, detail: str) -> None:
    global failures
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures += 1
    print(f"[{status}] {name}: {detail}")


print(f"validate_guard_exactness.py   seed = {SEED}")
print("=" * 78)

plant = reference_plant()
baseline, performance = reference_controllers(plant)
invariant = robust_invariant_set(plant, baseline).polytope
guard = SimplexGuard(plant, baseline, invariant)
w_verts = plant.disturbance.vertices()

# --- check 1: against vertex enumeration -----------------------------------
n_random = 30000
states = rng.uniform(-1.0, 1.0, (n_random, 2)) * np.array([0.32, 0.55])
inputs = rng.uniform(-3.2, 3.2, (n_random, 1))
# Plus 10000 pairs placed deliberately near the boundary of the eroded set, by
# bisecting the proposed input until the margin is within 1e-6 of zero.
boundary_states, boundary_inputs = [], []
while len(boundary_states) < 10000:
    x = rng.uniform(-1.0, 1.0, 2) * np.array([0.28, 0.48])
    lo, hi = 0.0, 3.0
    if guard.condition_margin(x, np.array([hi]))[0] >= 0.0:
        continue
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if guard.condition_margin(x, np.array([mid]))[0] >= 0.0:
            lo = mid
        else:
            hi = mid
    boundary_states.append(x)
    boundary_inputs.append(np.array([0.5 * (lo + hi)]))
all_states = np.vstack([states, np.array(boundary_states)])
all_inputs = np.vstack([inputs, np.array(boundary_inputs)])
mismatches = 0
rounding_only = 0
worst_mismatch_margin = 0.0
for x, u in zip(all_states, all_inputs, strict=True):
    if guard.allows(x, u) == guard.brute_force_allows(x, u):
        continue
    margin = abs(guard.condition_margin(x, u)[0])
    worst_mismatch_margin = max(worst_mismatch_margin, margin)
    if margin <= 1e-12:
        rounding_only += 1
    else:
        mismatches += 1
report(
    "check 1  support-function form vs vertex enumeration of W",
    mismatches == 0,
    f"{all_states.shape[0]} (state, input) pairs, {len(boundary_states)} of them "
    f"bisected to within 1e-9 of the switching boundary. {mismatches} mismatches with "
    f"a margin above 1e-12, and {rounding_only} disagreements AT the boundary where "
    f"|margin| <= 1e-12 (worst |margin| among all disagreements "
    f"{worst_mismatch_margin:.3e}). The two forms are algebraically identical; where "
    f"they differ it is because the eroded offset d_j - h_W(c_j) is rounded once, "
    f"while the vertex form rounds c_j^T(A x + B u + w) per vertex, and a state "
    f"bisected onto the facet can fall either side of the resulting 1-ulp gap. "
    f"Reported rather than tuned away",
)

# --- check 2: against a dense interior sampling of W ------------------------
dense_w = plant.disturbance.half_widths * rng.uniform(
    -1.0, 1.0, size=(200, 2)
)
falsifications = 0
admitted = 0
for x, u in zip(all_states[:8000], all_inputs[:8000], strict=True):
    if not guard.allows(x, u):
        continue
    admitted += 1
    nxt = plant.step(x, u)
    if not np.all(invariant.contains_many(nxt + dense_w, tol=1e-12)):
        falsifications += 1
report(
    "check 2  admitted inputs survive a dense interior sampling of W",
    falsifications == 0,
    f"{admitted} admitted pairs x 200 interior disturbance samples = "
    f"{admitted * 200} successor states, {falsifications} outside S",
)

# --- check 3: the margin is a signed distance along the binding facet -------
worst = 0.0
for x, u in zip(all_states[:5000], all_inputs[:5000], strict=True):
    margin, facet = guard.condition_margin(x, u)
    nxt = plant.step(x, u)
    row = guard.eroded_set.A[facet]
    offset = guard.eroded_set.b[facet]
    # The facet normals are unit length after reduction, so the residual along
    # the binding facet is exactly -margin.
    worst = max(worst, abs((row @ nxt - offset) + margin))
report(
    "check 3  the margin equals the binding facet residual",
    worst <= 1e-12,
    f"5000 pairs, worst absolute difference {worst:.3e}, tolerance 1e-12; facet "
    f"normals are unit length, so the margin is in state units",
)

# --- check 4: horizon-0 of the exact multi-step predictor -------------------
predictor = ExactLeadPredictor(guard, performance, 0)
xs = rng.uniform(-1.0, 1.0, (20000, 2)) * np.array([0.32, 0.55])
refs = np.column_stack([rng.uniform(-0.3, 0.3, 20000), np.zeros(20000)])
batch = predictor.predict_many(xs, refs)
direct = np.array(
    [
        guard.condition_margin(x, performance.unsaturated(x, r))[0] < 0.0
        for x, r in zip(xs, refs, strict=True)
    ]
)
report(
    "check 4  horizon-0 of the exact lead predictor equals the one-step condition",
    int(np.sum(batch != direct)) == 0,
    f"20000 random (state, reference) pairs, {int(np.sum(batch != direct))} disagreements",
)

# --- check 5: whole episodes under the worst-case sampler -------------------
exits = violations = 0
steps_total = 0
for episode in range(8):
    g = SimplexGuard(plant, baseline, invariant)
    seed_rng = np.random.default_rng(SEED + 31 * episode)
    w = disturbance_sequence(plant, 1500, seed_rng, "vertex")
    amp = float(seed_rng.uniform(0.12, 0.26))
    ep = simulate_guarded(
        plant, g, performance, 1500, square_wave_reference(amp, 80, 2), w
    )
    exits += int(ep.invariant_exits(invariant).size)
    violations += int(ep.constraint_violations().size)
    steps_total += ep.n_steps
report(
    "check 5  guarded episodes never leave S under the declared bound",
    exits == 0 and violations == 0,
    f"8 episodes x 1500 steps = {steps_total} steps with the worst-case vertex "
    f"sampler, {exits} exits from S, {violations} violations of X",
)

# --- check 6: the two exact additional guards ------------------------------
g = SimplexGuard(plant, baseline, invariant)
bad_input = g.decide(np.zeros(2), np.array([1e3]))
outside = np.array([0.15, 0.45])
lost = g.decide(outside, np.zeros(1))
report(
    "check 6a an input outside U is refused and flagged",
    bad_input.mode is Mode.BASELINE and not bad_input.input_admissible,
    f"mode {bad_input.mode}, input_admissible {bad_input.input_admissible}",
)
report(
    "check 6b a state outside S is flagged as certificate lost",
    lost.certificate_lost and lost.mode is Mode.BASELINE,
    f"state {outside.tolist()} has slack {lost.state_margin:.6e} in S, "
    f"mode {lost.mode}, certificate_lost {lost.certificate_lost}",
)

print("=" * 78)
print(f"failed checks: {failures}")
print(f"elapsed: {time.perf_counter() - start:.1f} s")
raise SystemExit(1 if failures else 0)
