# Dataset card — CMGSteer synthetic manoeuvre and null-motion-oracle datasets

**Fully synthetic. No flight data, no test-stand data, no hardware measurement
of any kind appears anywhere in this dataset or in this repository.**

Produced by [`src/cmgsteer/dataset.py`](src/cmgsteer/dataset.py)
(`manoeuvre_suite`, `generate_policy_dataset`). Nothing is committed as data:
the generators are committed, the arrays are not.

---

## 1. The two datasets

### 1a. Manoeuvre suite (`manoeuvre_suite`)

A reproducible set of commanded body-torque histories, used both as the states
the policy dataset is sampled from and as the held-out benchmark.

| Field | Shape | Units | Meaning |
|---|---|---|---|
| `profiles[i].torques` | (n_steps, 3) | N·m | Commanded body torque, one row per step |
| `profiles[i].dt` | scalar | s | Step length |
| `initial_deltas` | (n_manoeuvres, n) | rad | Starting gimbal angles |

### 1b. Policy dataset (`generate_policy_dataset`)

| Field | Shape | Units | Meaning |
|---|---|---|---|
| `features` | (n, 20) | — | `cmgsteer.ml.policy_features` at a visited state |
| `coefficients` | (n,) | — | The oracle-optimal null coefficient, the training label |
| `candidate_scores` | (n, 9) | N·m·s | Horizon momentum error for each candidate coefficient |
| `candidates` | (9,) | — | The grid the oracle searched, `linspace(-1, 1, 9)` |
| `gradient_scores` | (n,) | N·m·s | Horizon score of the classical gradient policy at the same state |

## 2. Array configuration

`cmgsteer.arrays.pyramid_array()`: the standard four-CMG pyramid, skew angle
`arctan(4/3)` = 53.13010235 deg (so `sin b = 0.8` and `cos b = 0.6` exactly),
rotor momentum 1 N·m·s each, total capacity 4 N·m·s.

| CMG | Gimbal axis `g` | Reference axis `c` (momentum at `delta = 0`) |
|---|---|---|
| cmg1 | (+0.8, 0, +0.6) | (0, +1, 0) |
| cmg2 | (0, +0.8, +0.6) | (−1, 0, 0) |
| cmg3 | (−0.8, 0, +0.6) | (0, −1, 0) |
| cmg4 | (0, −0.8, +0.6) | (+1, 0, 0) |

This is the textbook configuration used throughout the SGCMG literature (Wie
2008; Kurokawa 2007). It is **not a flight configuration from any real
vehicle**, and the rotor momentum of 1 N·m·s is a normalisation, not a
specification.

## 3. How a manoeuvre is drawn

Each manoeuvre is `n_segments` rest-to-rest torque pulses concatenated.
Per segment:

1. **Axis.** Uniform on the unit sphere, by normalising a standard 3-vector
   Gaussian.
2. **Peak stored momentum.** `rho * sum(h0)` with `rho ~ U(0.35, 0.65)`, so
   1.4 to 2.6 N·m·s of the array's 4 N·m·s capacity.
3. **Shape.** `tau(t) = tau_max sin(2 pi t / T)` with
   `tau_max = pi * dh / T` and `T = 6 s`, sampled at `dt = 0.02 s`
   (300 steps per segment). Each pulse integrates to zero net momentum change.
4. **Starting gimbal angles.** Uniform on `[-0.15, 0.15]` rad per gimbal.

A single pulse returns the array's momentum to where it started but not its
gimbal angles, so a sequence of pulses is what drives the array into awkward
corners of the configuration space. That is deliberate: it is the regime in
which reconfiguring the gimbals is supposed to pay, and it is therefore the
regime that gives the learned policy its best chance.

## 4. How a policy label is produced

1. Plain SR-inverse steering is run over the manoeuvre suite with a 2 rad/s
   gimbal-rate limit, and one state in every `stride = 17` steps is recorded.
2. At each recorded state, every candidate coefficient
   `k in linspace(-1, 1, 9)` is held constant for `horizon = 25` steps of
   simulated SR steering, and scored by the path momentum error it accumulates,
   `sum |(-tau dt) - (h_next - h)|` [N·m·s].
3. The label is the argmin over the grid, refined by a three-point parabola
   through the best candidate and its neighbours (clamped to one grid spacing,
   pinned in `tests/test_dataset.py::TestPolicyDataset::test_labels_are_near_the_best_candidate`).
4. The classical gradient policy is scored over the same window for reference.

**The oracle sees the future commanded torque; the trained policy does not.**
That gap is the point of the exercise and it is also the reason the label
regression fails (`MODEL_CARD.md` §7b).

## 5. Size, splits and cost

| Split | States | Seed | Manoeuvres | Generation time |
|---|---:|---:|---:|---:|
| Train | 900 | 1234 | 20 | 30.31 s |
| Test | 300 | 5678 | 10 | 8.93 s |
| Closed-loop benchmark | — | 9012 | 16 × 900 steps | — |

The three suites come from three different seeds, so no test state and no
benchmark manoeuvre shares a trajectory with a training state. Scikit-learn's
10% early-stopping holdout comes from inside the training split.

Label distribution on the training split: mean +0.0363, standard deviation
0.5551; percentiles 0/10/25/50/75/90/100 = −1.000, −0.750, −0.290, +0.025,
+0.406, +1.000, +1.000. **18.3% of labels sit at a grid edge (`|k| > 0.99`)**,
so the target is partly a saturated bang-bang decision.

Nothing is stored on disk. Regeneration is bit-for-bit deterministic from the
integer seed through `numpy.random.default_rng`:

```python
from cmgsteer import pyramid_array
from cmgsteer.dataset import generate_policy_dataset
array = pyramid_array()
train = generate_policy_dataset(array, 900, seed=1234, horizon=25,
                               n_candidates=9, stride=17, n_manoeuvres=20)
test = generate_policy_dataset(array, 300, seed=5678, horizon=25,
                               n_candidates=9, stride=17, n_manoeuvres=10)
```

## 6. Known biases and limitations

- **It characterises a lookahead oracle, not a spacecraft.** The labels are the
  output of a 25-step grid search over one steering law. A model trained here
  learns that oracle's preferences — its horizon, its objective, its candidate
  grid, its `lam0 = 0.01` and `mu = 10` — and nothing about real CMG hardware.
- **The horizon is arbitrary.** 25 steps is 0.5 s at `dt = 0.02`, one twelfth
  of a segment. A longer horizon would give different labels and was not tried;
  it would also cost proportionally more to generate.
- **The candidate grid quantises the label.** Nine values plus a parabolic
  refinement; a finer grid was not tried.
- **One array geometry only.** Everything is the four-CMG unit-momentum pyramid
  above. Nothing generalises to another skew angle, another number of CMGs,
  another rotor momentum, or a roof array.
- **No failures in the dataset.** Every state has four healthy gimbals. A locked
  gimbal changes the feature-vector length, so the trained model cannot be
  applied after a CMG failure at all.
- **Isotropic, uncorrelated manoeuvre axes**, which no real command sequence is.
  A real attitude profile commands torques correlated in time and biased toward
  particular axes, and an anticipating policy would have much more to exploit
  there.
- **One rate limit.** 2 rad/s throughout, in the state-visiting run, the
  rollouts and the benchmark.
- **One pulse shape and duration.** Sinusoidal, 6 s. No bang-bang manoeuvres, no
  sustained holds at high momentum, no disturbance rejection.
- **Explicit Euler integration.** The scores contain the integration error of
  the gimbal update as well as the steering error; it is first order in `dt` and
  is the same for every candidate at a given state, so it biases the absolute
  scores upward but not the comparison between candidates.
- **No measurement noise.** Gimbal angles and rotor momenta are exact
  everywhere in the dataset. `validation/validate_uncertainty.py` studies what
  noise does, separately, and the policy was never trained with it.
- **Sampling is along an SR-inverse trajectory**, so the state distribution is
  the one plain SR steering produces. A policy that changed the trajectory would
  see states this dataset does not contain — the standard distribution-shift
  problem of behaviour cloning, unaddressed here.

## 7. Provenance and licence

Generated by this repository. No third-party data. AGPL-3.0-or-later, the same
as the code. Copyright © 2026 OPTIMA Organisation.

## 8. Intended use

Training and benchmarking the null-motion policy in
[`MODEL_CARD.md`](MODEL_CARD.md), and as a reproducible fixture for the
validation scripts and the regression suite. It is not a benchmark for CMG
steering research in general and should not be cited as one.
