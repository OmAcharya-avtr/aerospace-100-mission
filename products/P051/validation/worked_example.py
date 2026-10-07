"""The worked example printed in README.md, run so that its output is real.

Fifteen lines of public API from the plant to the assurance report, then the
numbers the README quotes.
"""

from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

import numpy as np  # noqa: E402

from simplexguard import (  # noqa: E402
    ExactLeadPredictor,
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

plant = reference_plant()                      # single-axis attitude, dt = 0.05 s
baseline, performance = reference_controllers(plant)

result = robust_invariant_set(plant, baseline)  # the certificate
guard = SimplexGuard(plant, baseline, result.polytope, verify=True)

rng = np.random.default_rng(51)
w = disturbance_sequence(plant, 2000, rng, "uniform")   # inside the declared bound
reference = square_wave_reference(0.18, 80, plant.n_states)

guarded = simulate_guarded(plant, guard, performance, 2000, reference, w)
unguarded = simulate_unguarded(plant, performance, 2000, reference, w)
base_only = simulate_baseline(plant, baseline, 2000, reference, w)

report = account(guarded, unguarded, base_only, result.polytope)

print("the certificate")
print(result.describe())
print()
print("the switching condition")
print(guard.describe())
print()
print("the assurance accounting")
print(report.describe())
print()
print("one decision, at the origin with the reference at +0.18 rad")
decision = guard.decide(np.zeros(2), performance(np.zeros(2), np.array([0.18, 0.0])))
print(f"  mode                     {decision.mode}")
print(f"  proposed input           {decision.proposed_input[0]:.6f} rad/s^2")
print(f"  baseline input           {decision.baseline_input[0]:.6f} rad/s^2")
print(f"  switching margin         {decision.margin:.9f}")
print(f"  binding facet of S       {decision.binding_facet}")
print(f"  state margin inside S    {decision.state_margin:.9f}")
print()
print("the first step of the episode at which the guard actually fired")
fired = int(np.flatnonzero(guarded.modes == 1)[0])
x = guarded.states[fired]
r = guarded.references[fired]
guard.reset()
decision = guard.decide(x, performance(x, r))
print(f"  step                     {fired}  (t = {fired * plant.dt:.2f} s)")
print(f"  state                    [{x[0]:.6f} rad, {x[1]:.6f} rad/s]")
print(f"  reference                {r[0]:.6f} rad")
print(f"  mode                     {decision.mode}")
print(f"  proposed input           {decision.proposed_input[0]:.6f} rad/s^2")
print(f"  applied input            {decision.applied_input[0]:.6f} rad/s^2")
print(f"  switching margin         {decision.margin:.9f}")
print(f"  binding facet of S       {decision.binding_facet}")
print(f"  state margin inside S    {decision.state_margin:.9f}")
print(f"  condition holds          {decision.condition_holds}")
print()
print("exact anticipation of the switch, 5 steps ahead")
predictor = ExactLeadPredictor(guard, performance, 5, "worst_case")
for state in (
    np.array([0.18, 0.0]),
    np.array([0.00, 0.00]),
    np.array([0.05, 0.25]),
    np.array([0.12, 0.40]),
):
    first = predictor.first_step(state, np.array([0.18, 0.0]))
    print(
        f"  x = [{state[0]:5.2f}, {state[1]:5.2f}]  earliest step the guard may fire: "
        + ("never within 5" if first < 0 else f"{first}")
    )
