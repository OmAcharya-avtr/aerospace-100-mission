"""Validation 4: the assurance accounting, over seeds, samplers and dwell settings.

This is the product. Everything here is a measurement on simulated episodes of
one plant under one disturbance model, and none of it is a safety argument.

Checks
  1  The reference scenario, reported in full: the six quantities with the
     dwell-time histogram.
  2  The same scenario over 10 seeds, with the mean and the spread of each
     quantity, so a single-seed number is never the only number.
  3  All three disturbance samplers, including the worst-case vertex sampler.
     Zero violations must hold for every one of them.
  4  The internal algebra of the report: dwell totals, switch counts and the
     authority fraction must agree with each other exactly.
  5  The minimum-baseline-dwell trade: switch rate against conservatism cost.
  6  The reference-amplitude sweep, which is the dial that controls how hard the
     performance controller pushes against the certified envelope.

Runtime: about 55 s on one contended core.
"""

from __future__ import annotations

import time

from _bootstrap import add_src_to_path

add_src_to_path()

import numpy as np  # noqa: E402

from simplexguard import (  # noqa: E402
    SimplexGuard,
    account,
    disturbance_sequence,
    reference_controllers,
    reference_plant,
    robust_invariant_set,
    simulate_baseline,
    simulate_guarded,
    simulate_unguarded,
    square_wave_reference,
)

SEED = 51054
STEPS = 2000
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
invariant = robust_invariant_set(plant, baseline).polytope
reference = square_wave_reference(0.18, 80, plant.n_states)


def run(seed: int, mode: str = "uniform", dwell: int = 1, steps: int = STEPS,
        amplitude: float = 0.18):
    guard = SimplexGuard(plant, baseline, invariant, min_baseline_dwell=dwell)
    rng = np.random.default_rng(seed)
    w = disturbance_sequence(plant, steps, rng, mode)
    ref = square_wave_reference(amplitude, 80, plant.n_states)
    guarded = simulate_guarded(plant, guard, performance, steps, ref, w)
    unguarded = simulate_unguarded(plant, performance, steps, ref, w)
    base_only = simulate_baseline(plant, baseline, steps, ref, w)
    return account(guarded, unguarded, base_only, invariant)


print(f"validate_accounting.py   seed = {SEED}, steps = {STEPS}")
print("=" * 78)

# --- check 1: the reference scenario in full --------------------------------
main = run(51)
print("check 1  the reference scenario (seed 51, uniform sampler, no hysteresis)")
print("-" * 78)
print(main.describe())
print("-" * 78)
report(
    "check 1  reference scenario",
    main.guarded_violations == 0 and main.unguarded_violations > 0,
    f"guarded violations {main.guarded_violations}, unguarded "
    f"{main.unguarded_violations}, conservatism cost ratio "
    f"{main.conservatism_cost_ratio:.6f}, baseline fraction "
    f"{main.baseline_fraction:.6f}",
)

# --- check 2: across seeds ---------------------------------------------------
seeds = list(range(51, 61))
across = [run(s) for s in seeds]
def spread(values):
    arr = np.asarray(values, dtype=float)
    return (
        f"mean {arr.mean():.6f}  min {arr.min():.6f}  "
        f"max {arr.max():.6f}  sd {arr.std(ddof=1):.6f}"
    )


print()
print(f"check 2  the same scenario over {len(seeds)} seeds")
print("-" * 78)
print(f"  switches per 1000 steps   {spread([a.switches_per_1000_steps for a in across])}")
print(f"  baseline fraction         {spread([a.baseline_fraction for a in across])}")
print(f"  baseline dwell mean       {spread([a.baseline_dwell.mean for a in across])}")
print(f"  conservatism cost ratio   {spread([a.conservatism_cost_ratio for a in across])}")
print(f"  gap recovered             {spread([a.performance_gap_recovered for a in across])}")
print(f"  unguarded violations      {spread([a.unguarded_violations for a in across])}")
print(f"  unguarded S exits         {spread([a.unguarded_invariant_exits for a in across])}")
print("-" * 78)
report(
    "check 2  zero guarded violations on every seed",
    all(a.guarded_violations == 0 for a in across)
    and all(a.guarded_invariant_exits == 0 for a in across),
    f"{len(seeds)} seeds, "
    f"{sum(a.guarded_violations for a in across)} guarded violations, "
    f"{sum(a.guarded_invariant_exits for a in across)} guarded S exits, "
    f"{sum(a.unguarded_violations for a in across)} unguarded violations over "
    f"{len(seeds) * STEPS} steps",
)
report(
    "check 2b the conservatism cost is positive on every seed",
    all(a.conservatism_cost_ratio > 1.0 for a in across),
    f"ratio range [{min(a.conservatism_cost_ratio for a in across):.6f}, "
    f"{max(a.conservatism_cost_ratio for a in across):.6f}]",
)

# --- check 3: all three samplers --------------------------------------------
print()
print("check 3  all three disturbance samplers, seed 51")
print("-" * 78)
sampler_rows = []
for mode in ("zero", "uniform", "vertex"):
    acc = run(51, mode=mode)
    sampler_rows.append((mode, acc))
    print(
        f"  {mode:<8s} switches/1k={acc.switches_per_1000_steps:7.2f} "
        f"base frac={acc.baseline_fraction:8.6f} "
        f"cost ratio={acc.conservatism_cost_ratio:8.6f} "
        f"guarded X-viol={acc.guarded_violations} "
        f"guarded S-exit={acc.guarded_invariant_exits} "
        f"unguarded X-viol={acc.unguarded_violations}"
    )
print("-" * 78)
report(
    "check 3  zero guarded violations under every sampler",
    all(a.guarded_violations == 0 and a.guarded_invariant_exits == 0 for _, a in sampler_rows),
    "zero for the zero, uniform and worst-case vertex samplers",
)

# --- check 4: internal algebra ----------------------------------------------
algebra_ok = True
details = []
for seed, acc in zip(seeds, across, strict=True):
    total = acc.baseline_dwell.total_steps + acc.performance_dwell.total_steps
    intervals = acc.baseline_dwell.n_intervals + acc.performance_dwell.n_intervals
    if total != acc.n_steps or acc.n_switches != intervals - 1:
        algebra_ok = False
        details.append(f"seed {seed}: total {total}, intervals {intervals}")
    if abs(acc.baseline_fraction - acc.baseline_dwell.total_steps / acc.n_steps) > 1e-12:
        algebra_ok = False
        details.append(f"seed {seed}: fraction mismatch")
report(
    "check 4  dwell totals, interval counts and the authority fraction agree",
    algebra_ok,
    f"{len(seeds)} seeds checked exactly" + ("" if algebra_ok else "; " + "; ".join(details)),
)

# --- check 5: the hysteresis trade ------------------------------------------
print()
print("check 5  minimum baseline dwell: switch rate against conservatism cost")
print("-" * 78)
dwell_rows = []
for dwell in (1, 2, 4, 8, 16):
    acc = run(51, dwell=dwell)
    dwell_rows.append((dwell, acc))
    print(
        f"  dwell={dwell:<3d} switches/1k={acc.switches_per_1000_steps:7.2f} "
        f"base frac={acc.baseline_fraction:8.6f} "
        f"cost ratio={acc.conservatism_cost_ratio:8.6f} "
        f"held by dwell={acc.forced_by_dwell_steps:<5d} "
        f"min baseline dwell={acc.baseline_dwell.minimum:<3d} "
        f"X-viol={acc.guarded_violations}"
    )
print("-" * 78)
monotone_switches = all(
    dwell_rows[i][1].switches_per_1000_steps >= dwell_rows[i + 1][1].switches_per_1000_steps
    for i in range(len(dwell_rows) - 1)
)
plateau = [
    dwell_rows[i + 1][0]
    for i in range(len(dwell_rows) - 1)
    if dwell_rows[i][1].switches_per_1000_steps
    == dwell_rows[i + 1][1].switches_per_1000_steps
]
monotone_cost = all(
    dwell_rows[i][1].conservatism_cost_ratio < dwell_rows[i + 1][1].conservatism_cost_ratio
    for i in range(len(dwell_rows) - 1)
)
report(
    "check 5a hysteresis does not increase the switch rate",
    monotone_switches,
    f"{dwell_rows[0][1].switches_per_1000_steps:.2f} per 1000 steps at dwell 1 falling "
    f"to {dwell_rows[-1][1].switches_per_1000_steps:.2f} at dwell 16, monotonically "
    f"non-increasing. It is NOT strictly decreasing: at dwell "
    f"{plateau if plateau else 'none'} the switch rate is unchanged, because a "
    f"two-step minimum merely extends baseline intervals that were already going to "
    f"end in a switch rather than merging two of them. The authority fraction rises "
    f"from {dwell_rows[0][1].baseline_fraction:.6f} to "
    f"{dwell_rows[1][1].baseline_fraction:.6f} for no reduction at all, which is the "
    f"worst setting in the table",
)
report(
    "check 5b hysteresis costs tracking performance",
    monotone_cost,
    f"cost ratio {dwell_rows[0][1].conservatism_cost_ratio:.6f} at dwell 1 rising to "
    f"{dwell_rows[-1][1].conservatism_cost_ratio:.6f} at dwell 16",
)
report(
    "check 5c hysteresis never costs safety",
    all(a.guarded_violations == 0 and a.guarded_invariant_exits == 0 for _, a in dwell_rows),
    "zero violations and zero S exits at every dwell setting, as the invariance of S "
    "under the baseline requires",
)

# --- check 6: the reference-amplitude sweep ---------------------------------
print()
print("check 6  reference amplitude: how hard the performance controller pushes")
print("-" * 78)
amp_rows = []
for amp in (0.06, 0.10, 0.14, 0.18, 0.22, 0.26):
    acc = run(51, amplitude=amp)
    amp_rows.append((amp, acc))
    print(
        f"  amp={amp:.2f} rad  switches/1k={acc.switches_per_1000_steps:7.2f} "
        f"base frac={acc.baseline_fraction:8.6f} "
        f"cost ratio={acc.conservatism_cost_ratio:8.6f} "
        f"guarded X-viol={acc.guarded_violations:<4d} "
        f"unguarded X-viol={acc.unguarded_violations:<5d} "
        f"unguarded S-exit={acc.unguarded_invariant_exits}"
    )
print("-" * 78)
report(
    "check 6  the guard holds at every amplitude the unguarded run breaks",
    all(a.guarded_violations == 0 for _, a in amp_rows),
    f"zero guarded violations at all {len(amp_rows)} amplitudes; the unguarded run "
    f"accumulates up to {max(a.unguarded_violations for _, a in amp_rows)} violations",
)

print("=" * 78)
print(f"failed checks: {failures}")
print(f"elapsed: {time.perf_counter() - start:.1f} s")
raise SystemExit(1 if failures else 0)
