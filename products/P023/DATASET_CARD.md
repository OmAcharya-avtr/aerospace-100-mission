# Dataset card — AllocLab synthetic allocation dataset

**Fully synthetic. No flight data, no test-stand data, no hardware measurement
of any kind appears anywhere in this dataset or in this repository.**

Produced by [`src/alloclab/dataset.py`](src/alloclab/dataset.py)
(`generate_dataset`). Nothing is committed as data: the generator is
committed, the arrays are not.

---

## 1. What a sample is

| Field | Shape | Units | Meaning |
|---|---|---|---|
| `torques` | (n, 3) | N·m | Commanded body torque |
| `health` | (n, m) | — | 1.0 where the effector can still move, 0.0 where it has failed |
| `commands` | (n, m) | N (thrust) | The exact QP allocation, the training label |
| `residual_norm` | (n,) | N·m | `‖tau − B u‖` of that label; non-zero exactly where the command is unattainable |
| `attainable` | (n,) | bool | Whether the degraded effector set could meet the command |

## 2. Effector configuration

`alloclab.dataset.reference_thruster_cluster(max_thrust=1.0, arm=0.5)`:
eight one-sided thrusters, `u ∈ [0, 1] N`, moment arm 0.5 m, control
effectiveness matrix `B[:, i] = r_i × F̂_i` [m].

| id | position r [m] | thrust direction on the body | torque column [m] |
|---|---|---|---|
| t1 | (0, 0.5, 0) | (0, 0, +1) | (+0.5, 0, 0) |
| t2 | (0, 0.5, 0) | (0, 0, −1) | (−0.5, 0, 0) |
| t3 | (0.5, 0, 0) | (0, 0, −1) | (0, +0.5, 0) |
| t4 | (0.5, 0, 0) | (0, 0, +1) | (0, −0.5, 0) |
| t5 | (0.5, 0, 0) | (0, +1, 0) | (0, 0, +0.5) |
| t6 | (0.5, 0, 0) | (0, −1, 0) | (0, 0, −0.5) |
| t7 | (0.5, 0.5, 0.5) | (+1, −1, 0)/√2 | +0.5(1, 1, −2)/√2 |
| t8 | (0.5, 0.5, 0.5) | (−1, +1, 0)/√2 | −0.5(1, 1, −2)/√2 |

Three antiparallel couples give ±torque about each body axis; t7 and t8 add a
skew direction with all three components non-zero, so the four distinct
generator lines are in general position and the attainable moment set has
exactly 14 vertices and volume 3.828427124746 (N·m)³ (validation §2b, §2c).

This is a plausible arrangement, **not a flight configuration from any real
vehicle.** It was chosen to be over-actuated, one-sided, and geometrically
non-degenerate.

## 3. How a sample is drawn

1. **Failures.** With probability `1 − failure_prob` (default 0.5) no effector
   fails; otherwise the number of failures is uniform on `1..max_failures`
   (default 2) and the set is drawn uniformly without replacement. Failed
   effectors are pinned to 0 N — **failed off only**; stuck-open failures are
   supported by the library (`EffectorSet.with_failures(stuck_at=...)`) but do
   not appear in this dataset.
2. **Direction.** Uniform on the unit sphere, by normalising a standard
   3-vector Gaussian.
3. **Magnitude.** `tau = rho · boundary_scale(d) · d`, where `boundary_scale`
   is the distance from the origin to the boundary of the **degraded**
   attainable moment set along `d`, and `rho ~ U(0, 1.05)`. So `rho ≤ 1` is
   attainable and `rho > 1` is not; the measured attainable fraction is 0.9550
   (train) and 0.9575 (test), against the 1/1.05 = 0.952 the sampling implies.
4. **Label.** `qp_allocate(degraded_set, tau, u_pref=lower, gamma=1e12)` —
   minimum control effort about zero thrust, which for a one-sided thruster set
   is a minimum-total-thrust preference.

The attainable moment set of each distinct failure combination is computed once
and cached (37 combinations for `m = 8`, `max_failures = 2`), so the cost is
37 convex hulls plus `n` bounded least-squares solves.

## 4. Size, splits, and cost

| Split | Samples | Seed | Attainable fraction | Generation time |
|---|---:|---:|---:|---:|
| Train | 4000 | 1234 | 0.9550 | 4.10 s |
| Test | 2000 | 5678 | 0.9575 | 1.98 s |

Same generator, different seeds; every sample is an independent draw, so there
is no grouping structure to split on. The test set is never touched during
fitting. Scikit-learn's 10% early-stopping holdout comes from inside the
training split.

Nothing is stored on disk. Regeneration is bit-for-bit deterministic from the
integer seed through `numpy.random.default_rng`, so the dataset is reproduced
by rerunning the generator rather than by downloading anything.

```python
from alloclab.dataset import generate_dataset, reference_thruster_cluster
eset = reference_thruster_cluster(max_thrust=1.0, arm=0.5)
train = generate_dataset(eset, 4000, seed=1234)
test = generate_dataset(eset, 2000, seed=5678)
```

## 5. Known biases and limitations

- **It characterises a solver, not a spacecraft.** The labels are the output of
  `qp_allocate`. Any model trained on this learns that solver's preferences —
  including its `u_pref = 0` effort bias and its `gamma = 1e12` torque-error
  weighting — and nothing about real thruster behaviour.
- **One configuration only.** Everything is the eight-thruster cluster above.
  Nothing in the dataset generalises to another geometry, another `m`, or
  another set of bounds.
- **Failed-off only.** No stuck-open thrusters, no partial degradation, no
  drifting effectiveness, no misalignment.
- **At most two simultaneous failures**, and failures are drawn independently
  of each other and of the command. Correlated failures — a lost valve driver
  taking a whole bank — are not represented.
- **Directions are isotropic**, which no real command sequence is. A real
  attitude controller commands torques correlated in time and biased toward
  particular axes.
- **No actuator physics.** Instantaneous, continuously throttleable thrust with
  no minimum impulse bit, no pulse-width modulation, no rise or fall
  transients, no plume impingement, no centre-of-mass migration, no thermal
  effects.
- **No measurement noise anywhere.** Commands are exact and the health mask is
  perfectly known. A real allocator sees an estimated `B` and a fault detector's
  best guess at which effector failed.
- **Magnitudes stop at 1.05× the attainable boundary.** Only about 4.6% of
  samples are outside the attainable set and none is far outside, so a model
  trained here sees very little of the deeply-saturated regime.
- **Class imbalance in `attainable`**: 95.5% attainable against 4.5% not.

## 6. Provenance and licence

Generated by this repository. No third-party data. Apache-2.0, the same as the
code. Copyright © 2026 OPTIMA Organisation.

## 7. Intended use

Training and benchmarking the surrogate allocator in
[`MODEL_CARD.md`](MODEL_CARD.md), and as a reproducible fixture for the
validation scripts. It is not a benchmark for control allocation research in
general and should not be cited as one.
