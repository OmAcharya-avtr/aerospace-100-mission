# Dataset card — momentummgr synthetic desaturation episodes v0.1.0

## The data is entirely simulated

**No flight telemetry, no measured magnetometer record, no wheel tachometer log, no
on-orbit desaturation record and no real spacecraft parameters are used anywhere in this
package.** Every episode is drawn from the distributions below by a seeded
`numpy.random.Generator`, and every state is produced by the simulator in
`momentummgr/episodes.py`. Nothing was measured; nothing was observed; nothing came from a
mission.

The externally sourced numbers in the whole repository are physical constants and two
published tables, all cited in `momentummgr/constants.py` and
`momentummgr/environment.py`:

* WGS-84 Earth radius, gravitational parameter and rotation rate;
* the solar constant 1361 W m⁻² and the speed of light;
* the reduced geomagnetic dipole moment 7.96e15 T m³ (Wertz; Larson & Wertz);
* the 28-band exponential atmosphere table reproduced in Vallado, *Fundamentals of
  Astrodynamics and Applications*;
* the reference-vehicle and reference-orbit definition used for the P027 cross-check,
  which is P027 `disturbtorque`'s published reference case, reproduced as an **input
  definition** so that two independent implementations can be compared.

## Summary

| | |
|---|---|
| Generator | `momentummgr.episodes.sample_episode(seed)` (committed, deterministic) |
| Labels | `momentummgr.learned.search_best_mask(episode, seed=0)` |
| Regeneration | `python3 data/generate_dataset.py` |
| Fitting episodes | 60, seeds 1000–1059 |
| Knob-tuning episodes | 25, seeds 2000–2024 (disjoint) |
| Held-out episodes | 80, seeds 5000–5079 (disjoint) |
| Windows per episode | 56 to 59, of 600 s, i.e. about 9.3 h or six orbits |
| Feature rows | 3413 × 11 |
| Positive label rate | 0.084676 |
| Searched mean cost | 0.045487 |
| Committed files | `data/training_features.csv` (448 kB), `data/dataset_manifest.txt` |
| Integrity | SHA-256 of the CSV recorded in `data/dataset_manifest.txt` |

The CSV is well under 1 MB and is committed. It is redundant with the generator: deleting
it and rerunning `data/generate_dataset.py` reproduces it exactly.

## Sampling ranges

Drawn independently and uniformly unless stated. These are a **sampling envelope**, chosen
so that a smallsat-class wheel set fills within a few orbits; they are not a survey of
flown spacecraft and no distribution here is claimed to be representative of anything.

| Parameter | Range | Note |
|---|---|---|
| Altitude | 400–650 km | inside the exponential table's useful band |
| Inclination | 30–98° | equatorial orbits are excluded; they have a magnetically blocked axis (see `validation/magnetic_controllability.py`) |
| RAAN | 0–360° | |
| Yaw offset from LVLH | ±10° | |
| Pitch, roll offsets | ±12° | non-zero so gravity gradient is non-zero |
| Principal moments of inertia | 2–14 kg m², sorted and forced to satisfy the rigid-body triangle inequality, then permuted | |
| Drag area | 0.3–1.5 m², Cd fixed at 2.2 | |
| Aerodynamic cp offset | ±0.06 m per axis | a badly balanced vehicle |
| Sunlit area | 0.5–2.5 m² | |
| SRP reflectance q | 0.2–0.9 | |
| SRP cp offset | ±0.06 m per axis | |
| Residual magnetic dipole | ±0.35 A m² per axis | |
| Mass | 50–180 kg | reported only; no torque depends on it |
| Wheel rotor inertia | 5e−4 to 2e−3 kg m² | four-wheel isotropic pyramid |
| Per-wheel momentum limit | 0.02–0.08 N m s | smallsat class |
| Magnetorquer dipole limit | 0.5–3.0 A m² | |
| Solar beta angle | ±70°, in-plane phase 0–360° | |
| Initial wheel momentum | random direction, 0–0.35 of the array envelope | |

Environment models inside an episode: the exponential atmosphere above, and a centred
geomagnetic dipole **tilted by 9.4° and rotating with the Earth**
(`momentummgr.episodes.DEFAULT_DIPOLE_TILT_RAD`), so the field geometry does not repeat
exactly from orbit to orbit. The tilt is a configurable parameter with a representative
default, not an epoch-specific claim; this package carries no IGRF coefficients.

## Rejection sampling, and what it hides

A draw is **rejected** if running the magnetorquers in every single window still fails to
keep the wheels inside their envelope. About 8 % of raw draws are rejected (5 of 60 in a
pre-filter survey recorded during development). Such vehicles exist and are a real design
outcome, but they are a *sizing* failure and not a *scheduling* problem: including them
would compare two schedulers on episodes neither can win. Set
`sample_episode(..., require_feasible=False)` to sample the unfiltered distribution.

The consequence a user must know: **this dataset contains no under-actuated vehicles**, so
nothing in the benchmark says how either scheduler behaves on one.

## Labels

The label for a window is the action taken there by the best schedule the offline search
found for that episode. The search is described in `MODEL_CARD.md`; it is non-causal by
construction. Windows in which the safety override had already fired are dropped from the
training rows, since the override is applied identically to every policy at run time and a
classifier should not spend capacity learning it.

Label balance: 8.47 % positive. Good schedules are sparse — a typical episode dumps in
three to six of its 56 windows.

## Splits

Fitting, knob-tuning and held-out seeds are disjoint by construction and are never mixed.
The classical baseline is tuned on the union of the fitting and knob-tuning sets, so it
sees exactly the same 85 episodes the learned pipeline does, and neither sees the held-out
80.

## Known biases and limitations

* **Synthetic throughout.** Nothing here has been compared against a real spacecraft.
* **Dipole field.** The geometry the model learns to exploit is that of a centred tilted
  dipole. Against IGRF that field's direction errs by tens of degrees, and direction is the
  quantity the whole problem turns on.
* **No solar-activity variation** in the atmosphere, so aerodynamic torque above 400 km
  carries at least a factor-of-several uncertainty from its input model.
* **Circular orbits only, fixed pointing offsets only.** No eccentricity, no J2, no slews,
  no payload-driven attitude changes, no wheel friction, no magnetorquer dynamics or
  dipole-magnetometer coupling.
* **No under-actuated vehicles**, by construction — see rejection sampling above.
* **One window length and one horizon.** 600 s windows over six orbits. Nothing says how
  either scheduler behaves at 60 s or over a week.
* **Uniform, independent parameter draws.** Real spacecraft parameters are correlated
  (a large vehicle has large inertia *and* large area); this sampler produces combinations
  no engineer would build.
