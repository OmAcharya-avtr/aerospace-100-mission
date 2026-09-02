# Validation evidence — CMGSteer 0.1.0

Validation Level 3. Every number below was produced by running the script named
beside it, in this repository, on 2 x86-64 cores under Python 3.11.15,
numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.9, on
2026-09-02. Raw stdout is saved next to each script.

| Script | Raw output | What it establishes | Wall clock |
|---|---|---|---:|
| `validate_jacobian.py` | `jacobian_output.txt` | The Jacobian is the derivative of the momentum map | 10 s |
| `validate_singularity.py` | `singularity_output.txt` | The singularity measure vanishes on the analytic singular set, and the classification matches the geometric theory | 9 s |
| `validate_steering.py` | `steering_output.txt` | The pseudo-inverse is exact away from a singularity; the SR-inverse error equals its closed form; the momentum error is first order in `dt` | 12 s |
| `validate_uncertainty.py` | `uncertainty_output.txt` | Monte Carlo error propagation against first-order predictions | 19 s |
| `validate_nullmotion_ml.py` | `nullmotion_ml_output.txt` | The learned null-motion policy against the classical policies and against plain SR | 125 s |
| `validate_performance.py` | `performance_output.txt` | Wall-clock cost of every operation that goes in a loop | 37 s |

Reproduce all six:

```bash
python validation/validate_jacobian.py
python validation/validate_singularity.py
python validation/validate_steering.py
python validation/validate_uncertainty.py
python validation/validate_nullmotion_ml.py
python validation/validate_performance.py
```

Two array geometries recur throughout:

| Name | n | Geometry | Notes |
|---|---:|---|---|
| Standard pyramid | 4 | Skew angle `arctan(4/3)` = 53.13010235 deg, so `sin b = 0.8` and `cos b = 0.6` exactly | The textbook configuration; rotor momentum 1 N·m·s each, capacity 4 N·m·s |
| Roof | 4 | Two pairs of parallel gimbal axes, ridges tilted 45 deg | Rank-deficient at `delta = 0` by construction |

---

## 1. The Jacobian against numerical differentiation of the momentum map

`validate_jacobian.py`, seed 20260902.

The analytic `A(delta)` is compared with a central difference of `h(delta)`
over 400 configurations drawn uniformly from `[-pi, pi]^4`, at six step sizes,
so the optimum of the truncation/round-off trade-off is visible rather than
assumed.

| Step [rad] | Pyramid worst abs. deviation | Roof worst abs. deviation |
|---:|---:|---:|
| 1e-3 | 1.666667e-07 | 1.666666e-07 |
| 1e-4 | 1.667076e-09 | 1.668054e-09 |
| **1e-5** | **4.061274e-11** | **4.238054e-11** |
| 1e-6 | 3.651611e-10 | 3.662140e-10 |
| 1e-7 | 3.787304e-09 | 3.585553e-09 |
| 1e-8 | 3.191703e-08 | 3.537727e-08 |

**Worst deviation reported: 4.061274e-11 N·m·s/rad for the pyramid and
4.238054e-11 for the roof array, both at the optimal step of 1e-5 rad.** The
quadratic fall and linear rise on either side of that step are the expected
signature of a correct central difference, so the residual is dominated by the
difference scheme rather than by the analytic Jacobian.

| Check | Result | Tolerance | Outcome |
|---|---:|---:|---|
| Pyramid Jacobian vs central differences | 4.061274e-11 | 1e-8 | PASS |
| Roof Jacobian vs central differences | 4.238054e-11 | 1e-8 | PASS |
| Pyramid manipulability gradient `dm/ddelta` vs central differences | 1.058439e-09 | 1e-7 | PASS |
| Roof manipulability gradient vs central differences | 1.041955e-09 | 1e-7 | PASS |
| Pyramid momentum map vs the published closed form, 2000 configurations | 8.881784e-16 N·m·s | 1e-13 | PASS |
| Torque convention `tau = -A ddelta`, 2000 pairs | 0.000000e+00 N·m | 1e-13 | PASS |

The closed-form check is against the four-CMG pyramid momentum map as it is
written throughout the SGCMG literature (Wie 2008; Kurokawa 2007); the package
never uses that form internally, so the agreement to 8.9e-16 N·m·s is a genuine
independent check of the general construction.

## 2. The singularity measure on analytically known singular configurations

`validate_singularity.py`.

For a unit direction `u` and a sign vector `eps`, the configuration in which
each rotor momentum is `h_hat_i = eps_i * normalise(u - (u . g_i) g_i)` makes
every Jacobian column perpendicular to `u`, so `m = 0` exactly. Sweeping 500
Fibonacci-sphere directions over all 16 sign vectors gives 8000 analytically
singular configurations per array.

| Array | Configurations | Worst `m` | Worst `sigma_min` [N·m·s/rad] |
|---|---:|---:|---:|
| Pyramid | 8000 | **1.009474e-15** | 5.671167e-16 |
| Roof | 8000 | **9.936793e-16** | 5.671167e-16 |

For scale, `m` over 5000 uniformly random pyramid configurations has mean
9.581813e-01, median 1.009224e+00 and 1st percentile 1.648394e-01, so the
singular-set values sit about fifteen orders of magnitude below the regular
scale.

### 2b. Hand-computable configurations of the standard pyramid

| Configuration | `m` | `h` computed [N·m·s] | `h` by hand | Deviation |
|---|---:|---|---|---:|
| All gimbals at 0 deg | 1.152000000000001 | (0, 0, 0) | `A A^T = diag(0.72, 0.72, 2.56)`, so `m = 0.72 × 1.6 = 1.152` | 1e-15 |
| All gimbals at +90 deg | 1.959435e-16 | (0, 0, 3.200000) | `4 h0 sin b = 4 × 0.8 = 3.2` along +z | 4.440892e-16 |
| (90, 90, 90, −90) deg | 3.989069e-16 | (0, −1.2, 1.6) | `s1 + s2 + s3 − s4`, `|h| = 2` exactly | 2.220446e-16 |

### 2c. Classification of the singular set

3200 classified points on the pyramid's singular set (200 directions × 16 sign
vectors):

| Kind | Passability | Count | Fraction |
|---|---|---:|---:|
| external | elliptic | 400 | 0.1250 |
| internal | elliptic | 738 | 0.2306 |
| internal | hyperbolic | 2062 | 0.6444 |

**Every one of the 400 external (saturation) singularities is elliptic**, which
is what the geometric theory requires: with all `eps_i` equal the second-order
form is definite, so no momentum-preserving motion can leave the surface. 64%
of the internal singularities are hyperbolic, i.e. escapable by null motion.

A subtlety this validation surfaced and that the code documents: at an external
singularity the manipulability gradient is **not** zero — its projection onto
`null(A)` has magnitude 1.6 at the +z saturation point — so a gradient null
motion does move away from the singular *configuration*. What it cannot do is
keep the momentum: every direction in `null(A)` reduces `h . u` from 3.199999999
to 3.199000, so the array leaves the envelope inward. "Elliptic" constrains the
momentum, not the measure. Pinned in
`tests/test_failure_modes.py::TestExternalSingularity::test_leaving_an_external_singularity_costs_momentum`.

### 2d. Singular surfaces in momentum space

Outer (saturation) envelope, 4000 directions: minimum radius 2.975341090,
maximum radius 3.298439824 N·m·s, against a capacity `sum(h0)` of 4. Sphericity
(min/max) 0.902045. The 14 internal surfaces span radii from 0.033599 to
3.015232 N·m·s; the `++--` and `--++` surfaces reach closest to the origin
(minimum radius 0.033599 N·m·s), and the `++++`/`----` pair is the envelope
itself (2.980002 to 3.298425 N·m·s over the 800-direction sweep).

## 3. Steering-law exactness and the SR-inverse closed form

`validate_steering.py`, seed 20260902.

### 3a. The pseudo-inverse reproduces the commanded torque

4000 states per array — 2000 uniformly random plus 2000 perturbations of
analytically constructed singular configurations, so the near-singular bands are
actually populated — binned by singularity measure.

| `m` band | Pyramid `n` | Worst \|err\| [N·m] | Worst rate [rad/s] |
|---|---:|---:|---:|
| [0.5, inf) | 1818 | 1.665614e-14 | 2.363123e+00 |
| [0.1, 0.5) | 571 | 3.204519e-14 | 9.172053e+00 |
| [0.01, 0.1) | 561 | 6.241461e-13 | 8.797790e+01 |
| [1e-4, 0.01) | 1031 | 8.060199e-12 | 9.360398e+03 |
| [0, 1e-4) | 19 | 7.471545e-11 | 9.889665e+03 |

**Away from a singularity (`m >= 0.1`) the worst torque error is 3.204519e-14 N·m
for the pyramid and 3.332901e-14 N·m for the roof array**, against a tolerance
of 1e-12. The error grows to 7.5e-11 N·m only inside the last band, where the
gimbal rate has already reached 1e4 rad/s and the inverse is useless for other
reasons.

### 3b. SR-inverse torque error against its closed form

With `A = U S V^T`, the SR inverse `ddelta = A^T (A A^T + lam I)^{-1} b` leaves

    tau_err = sum_k [lam / (sigma_k^2 + lam)] (u_k . tau) u_k

Measured against that expression over 200 random states per value of `lam`
(pyramid):

| `lam` [(N·m·s/rad)²] | Mean \|err\| measured [N·m] | Closed form [N·m] | Worst deviation [N·m] | Mean err / \|tau\| |
|---:|---:|---:|---:|---:|
| 1e-10 | 1.738473840e-10 | 1.738473817e-10 | 1.313280e-15 | 4.030e-10 |
| 1e-08 | 1.738472357e-08 | 1.738472358e-08 | 1.747485e-15 | 4.030e-08 |
| 1e-06 | 1.738326471e-06 | 1.738326471e-06 | 2.123911e-15 | 4.029e-06 |
| 1e-05 | 1.737003064e-05 | 1.737003064e-05 | 3.128357e-15 | 4.026e-05 |
| 1e-04 | 1.724043709e-04 | 1.724043709e-04 | 2.331468e-15 | 3.995e-04 |
| 1e-03 | 1.616628445e-03 | 1.616628445e-03 | 1.502271e-15 | 3.739e-03 |
| 1e-02 | 1.216936786e-02 | 1.216936786e-02 | 5.342948e-16 | 2.765e-02 |
| 1e-01 | 6.870827669e-02 | 6.870827669e-02 | 4.163336e-16 | 1.502e-01 |
| 1e+00 | 2.331821083e-01 | 2.331821083e-01 | 3.885781e-16 | 5.062e-01 |
| 1e+01 | 4.073907215e-01 | 4.073907215e-01 | 6.661338e-16 | 8.858e-01 |

**Worst componentwise deviation across all ten values of `lam` and 200 states:
1.240235e-14 N·m for the pyramid, 4.430958e-14 N·m for the roof array.** The
error is linear in `lam` for `lam << sigma_min^2` and saturates at `|tau|` for
`lam >> sigma_max^2`, exactly as the expression requires.

At an exact singularity with the command along the singular direction the error
is `|tau|` for every `lam > 0` — measured 1.000000000e-01 N·m at all ten values
of `lam`, with the gimbal rate falling as `1/lam` from 4.898587e-08 to
4.898587e-19 rad/s. A command in the plane the array can still act in is met to
5.0e-10 N·m at `lam = 1e-8`.

### 3c. Gimbal-rate growth approaching a singularity

Approaching an internal singularity along a straight line (pyramid, commanded
torque (0.1, −0.05, 0.2) N·m):

| Offset [rad] | `m` | pinv peak rate [rad/s] | SR peak rate [rad/s] | SR \|err\| [N·m] |
|---:|---:|---:|---:|---:|
| 1e-1 | 2.868062e-01 | 9.619349e-01 | 9.370982e-01 | 5.947419e-03 |
| 1e-2 | 2.892523e-02 | 9.620343e+00 | 2.697782e-01 | 2.207104e-01 |
| 1e-3 | 2.892939e-03 | 9.635462e+01 | 3.541091e-02 | 2.268709e-01 |
| 1e-4 | 2.892960e-04 | 9.636962e+02 | 1.725728e-02 | 2.269130e-01 |
| 1e-5 | 2.892962e-05 | 9.637112e+03 | 1.568492e-02 | 2.269128e-01 |
| 1e-6 | 2.892962e-06 | 9.637127e+04 | 1.579710e-02 | 2.269127e-01 |

The pseudo-inverse rate grows exactly as `1/m`; the SR inverse holds the rate
below 1 rad/s and pays a bounded torque error of 0.227 N·m instead. That trade
is the whole point of the SR inverse and it is what the table shows.

### 3d. GSR inverse and integration convergence

With `eps0 = 0` the GSR inverse reduces to the SR inverse to **0.000000e+00 rad/s**
over 500 random states — an exact reduction, because the dither matrix becomes
the identity. With `eps0 = 0.05` the off-diagonal terms cycle with the
prescribed period and the torque error varies by about 2e-11 N·m across the
cycle at a near-singular state, which is the mechanism the law relies on to
avoid getting stuck.

Momentum-error convergence with the integration step (`validate_steering.py`
§6, pseudo-inverse on a rest-to-rest profile whose instantaneous torque error
stays at round-off):

| `dt` [s] | Net momentum error [N·m·s] | Ratio to previous | Max \|tau err\| [N·m] |
|---:|---:|---:|---:|
| 0.10000 | 1.006766e-02 | — | 5.711852e-16 |
| 0.05000 | 5.018896e-03 | 2.0060 | 6.472541e-16 |
| 0.02500 | 2.505703e-03 | 2.0030 | 6.575861e-16 |
| 0.01250 | 1.251914e-03 | 2.0015 | 6.026860e-16 |
| 0.00625 | 6.257226e-04 | 2.0007 | 6.748944e-16 |

## 4. Uncertainty analysis

`validate_uncertainty.py`, 30 states × 200 Monte Carlo trials per sigma.
Ratios are formed per state and then averaged; pooling the Monte Carlo across
states and comparing an rms with a mean prediction gives a spurious factor of
about 2, which an earlier draft of this script did and this one does not.

### 4a. Gimbal-angle measurement error

First order: `rms|e| = sigma sqrt(sum_j h0_j^2 ddelta_j^2)`.

| sigma [deg] | Mean MC rms [N·m] | Mean first order [N·m] | Mean ratio | p5 | p95 |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 9.298374e-06 | 9.195019e-06 | 0.9923 | 0.9297 | 1.0390 |
| 0.010 | 9.249416e-05 | 9.195019e-05 | 1.0088 | 0.9649 | 1.0556 |
| 0.100 | 8.893509e-04 | 9.195019e-04 | 0.9994 | 0.9476 | 1.0631 |
| 1.000 | 9.656653e-03 | 9.195019e-03 | 1.0185 | 0.9502 | 1.0689 |

### 4b. Rotor momentum error

| sigma [relative] | Mean MC rms [N·m] | Mean first order [N·m] | Mean ratio |
|---:|---:|---:|---:|
| 1e-4 | 2.700990e-05 | 2.703781e-05 | 0.9982 |
| 1e-3 | 2.710551e-04 | 2.703781e-04 | 0.9998 |
| 1e-2 | 2.699231e-03 | 2.703781e-03 | 0.9973 |
| 5e-2 | 1.353661e-02 | 1.351891e-02 | 1.0000 |

### 4c. Singularity-measure sensitivity

`sigma_m = |grad m| sigma` to first order:

| sigma [deg] | MC std(`m`) | `|grad m| sigma` | Ratio |
|---:|---:|---:|---:|
| 0.001 | 1.688692e-05 | 1.727797e-05 | 0.9774 |
| 0.010 | 1.744023e-04 | 1.727797e-04 | 1.0094 |
| 0.100 | 1.751027e-03 | 1.727797e-03 | 1.0134 |
| 1.000 | 1.719061e-02 | 1.727797e-02 | 0.9949 |

### 4d. Manoeuvre-level effect

Six manoeuvres, SR inverse, 2 rad/s rate limit, angle noise injected at every
step:

| sigma [deg] | Mean path error [N·m·s] | Mean net error [N·m·s] | Min `m` | vs noise-free |
|---:|---:|---:|---:|---:|
| 0.000 | 2.549712e-01 | 6.846035e-02 | 1.463964e-01 | 1.0000 |
| 0.001 | 2.550009e-01 | 6.848938e-02 | 1.463806e-01 | 1.0001 |
| 0.010 | 2.552861e-01 | 6.875814e-02 | 1.462402e-01 | 1.0012 |
| 0.100 | 2.591470e-01 | 7.187941e-02 | 1.449180e-01 | 1.0164 |
| 1.000 | 3.537793e-01 | 7.573297e-02 | 8.562432e-02 | 1.3875 |

**Gimbal-angle knowledge better than about 0.1 deg makes no practical difference
to this steering error budget; at 1 deg the noise adds 39% to the path
momentum error and halves the minimum singularity measure reached.**

## 5. The learned null-motion policy against the classical policies

`validate_nullmotion_ml.py`. Seeds: training data 1234, test data 5678,
benchmark suite 9012, bootstrap 4242, model `random_state = 0`. Trained once,
no hyperparameter search, reported as it came out. Full discussion in
[`../MODEL_CARD.md`](../MODEL_CARD.md).

### 5a. Oracle headroom on the horizon objective

The oracle sees 25 steps of future commanded torque and picks the null-motion
coefficient minimising the momentum error accumulated over them.

| Split | `k = 0` (plain SR) | Best candidate | Classical gradient | Oracle gain |
|---|---:|---:|---:|---:|
| train (900) | 1.301202e-02 | 1.006399e-02 | 1.193082e-02 | 22.66% |
| test (300) | 2.502476e-02 | 1.922236e-02 | 2.279211e-02 | 23.19% |

Even with perfect foresight over 25 steps, the best available null motion is
worth only about 23%. The classical gradient policy captures 8.92% of the plain
SR score on the same objective.

### 5b. Label-level accuracy on the held-out set

| Quantity | Value |
|---|---:|
| Mean absolute error | 0.492465 |
| RMS error | 0.623504 |
| Label standard deviation | 0.596145 |
| **R²** | **−0.093894** |
| Sign agreement with the oracle label | 0.5833 |

**The regression is worse than predicting the training mean.** The oracle label
depends on the next 25 steps of commanded torque, which is not a function of the
state the policy observes, so a large part of the label is unpredictable in
principle from these features.

### 5c. Oracle gap at the predicted coefficient

| Quantity | Value |
|---|---:|
| Mean horizon score at the predicted `k` | 2.241476e-02 N·m·s |
| Mean horizon score at `k = 0` | 2.502476e-02 N·m·s |
| Mean horizon score at the oracle `k` | 1.922236e-02 N·m·s |
| **Fraction of the oracle gain captured** | **44.98%** |
| States where the policy is worse than `k = 0` | 52.33% |

The policy captures 45% of the available gain in the mean while being worse than
doing nothing on 52% of individual states: the objective is flat near its
optimum and the gains are concentrated in a minority of states.

### 5d. Closed-loop benchmark, 16 held-out manoeuvres × 900 steps

Gimbal-rate limit 2 rad/s, `dt` 0.02 s, benchmark seed 9012.

| Configuration | Path err [N·m·s] | Net err [N·m·s] | Min `m` | RMS tau err [N·m] | Saturated steps | Wall [s] |
|---|---:|---:|---:|---:|---:|---:|
| pinv | 7.371954e-01 | 2.900413e-01 | 0.256271 | 1.818783e-01 | 30.94 | 3.90 |
| **sr** | **4.921487e-01** | **2.181053e-01** | 0.260340 | 1.141659e-01 | 22.94 | 3.72 |
| gsr | 4.920096e-01 | 2.192314e-01 | 0.260342 | 1.140798e-01 | 22.94 | 3.82 |
| sr+grad | 4.571102e-01 | 3.025358e-01 | 0.234056 | 1.168396e-01 | 18.00 | 10.19 |
| sr+learned | 4.793887e-01 | 2.926328e-01 | 0.230814 | 1.142026e-01 | 20.06 | 60.96 |

Paired differences against plain SR, percentile bootstrap 95% CI over 10 000
resamples of the 16 manoeuvres:

| Configuration | Metric | Mean difference | 95% CI | Wins | Verdict |
|---|---|---:|---|---:|---|
| pinv | path | +2.450466e-01 | [+7.7848e-02, +4.4834e-01] | 1/16 | **worse than SR** |
| pinv | net | +7.193604e-02 | [+1.7733e-02, +1.4087e-01] | 0/16 | **worse than SR** |
| pinv | min `m` | −4.068397e-03 | [−9.3699e-03, −3.5470e-04] | 4/16 | **worse than SR** |
| gsr | path | −1.391912e-04 | [−3.8615e-03, +3.3893e-03] | 9/16 | indistinguishable |
| gsr | net | +1.126072e-03 | [−4.4682e-04, +3.8641e-03] | 8/16 | indistinguishable |
| gsr | min `m` | +2.077834e-06 | [−4.8725e-05, +5.1948e-05] | 9/16 | indistinguishable |
| sr+grad | path | −3.503851e-02 | [−2.0949e-01, +1.6292e-01] | 10/16 | indistinguishable |
| sr+grad | net | +8.443053e-02 | [−9.5878e-03, +2.1718e-01] | 7/16 | indistinguishable |
| sr+grad | min `m` | −2.628385e-02 | [−1.2153e-01, +6.1568e-02] | 8/16 | indistinguishable |
| sr+learned | path | −1.276004e-02 | [−2.2962e-01, +2.3460e-01] | 9/16 | indistinguishable |
| sr+learned | net | +7.452750e-02 | [−4.2746e-02, +2.2013e-01] | 9/16 | indistinguishable |
| sr+learned | min `m` | −2.952585e-02 | [−1.3701e-01, +6.8000e-02] | 7/16 | indistinguishable |

**The finding.** Against the plain singularity-robust inverse, neither the
classical gradient null motion nor the learned policy produces a difference
distinguishable from zero on any of the three metrics, on 16 manoeuvres. The
only configuration that *is* distinguishable is the plain pseudo-inverse, and it
is distinguishably **worse** — 49.8% more path momentum error, 33.0% more net
momentum error, and a lower minimum singularity measure, all with intervals
excluding zero. The generalised SR inverse is indistinguishable from the SR
inverse on this suite, which is expected: its dither only matters when the array
is actually held on a singular surface, and these manoeuvres pass through
singular regions rather than stalling in them.

### 5e. Confidence output

| Quantity | Value |
|---|---:|
| Pearson r(confidence, \|label error\|) | **−0.1661** |
| Pearson r(ensemble spread, \|label error\|) | +0.1599 |
| Mean ensemble spread / rms label error | 0.2870 |

Mean absolute label error by confidence decile (lowest confidence first):
0.6332, 0.5122, 0.4799, 0.5894, 0.4500, 0.5152, 0.5294, 0.4405, 0.4140, 0.3609.
The trend is downward but non-monotone, and the correlation is weak. The
normalised oracle gap by the same deciles is 0.2185, 0.5237, 0.4761, 0.3853,
0.3692, 0.2517, 0.2067, 0.3207, 0.4120, 0.2710 — no usable ordering at all.
**The confidence output is not calibrated and is not usable as a gate.** The
ensemble spread is 0.287 of the rms error, so it understates the error by about
71% and must never be used as a variance.

### 5f. Runtime

| Configuration | µs per step | vs SR |
|---|---:|---:|
| pinv | 271.2 | 1.05× |
| sr | 258.3 | 1.00× |
| gsr | 265.1 | 1.03× |
| sr+grad | 707.6 | 2.74× |
| sr+learned | 4233.1 | **16.39×** |

The learned policy is 16× the cost of the law it fails to distinguishably
improve on. Five scikit-learn `predict` calls on one sample dominate, and a
sequential control loop cannot batch them.

## 6. Performance

`validate_performance.py`. Full table in `performance_output.txt`.

| Operation | µs per call |
|---|---:|
| `array.momentum` | 92.71 |
| `array.jacobian` | 112.37 |
| `singularity_measure` | 36.76 |
| `manipulability_gradient` | 456.91 |
| `unit_null_vector` | 705.61 |
| `classify_singularity` | 519.11 |
| `pseudo_inverse_steer` | 554.96 |
| `sr_inverse_steer` | 517.83 |
| `gsr_inverse_steer` | 539.30 |
| `GradientNullMotion.rates` | 533.49 |
| `LearnedNullMotion.rates` | 3234.94 |

Cost is flat in the number of CMGs from 4 to 16 (`sr_inverse_steer` 536 µs at
n = 4, 526 µs at n = 16): the work is dominated by Python and NumPy call
overhead on 3×n matrices, not by arithmetic. A 1000-step run takes 0.54 s
(pinv) to 1.12 s (SR with gradient null motion). A recorded run costs 249 bytes
per step. The fused single-SVD rollout path used by dataset generation is
**5.35× faster** than the public path and agrees with it to 1e-15
(`tests/test_dataset.py::TestFastStepperAgreement`).

## What failed, and what the classical method won

- **The learned policy's label regression has a negative R² (−0.0939).** It is
  worse than predicting the training mean. Reported in §5b, in the model card,
  and in the README.
- **No null-motion policy, classical or learned, beats plain SR-inverse
  distinguishably** on any metric over 16 manoeuvres (§5d). Both increase the
  net momentum error by 34–39% in the mean and both reduce the minimum
  singularity measure reached; neither difference has an interval excluding
  zero. The honest summary is that the SR inverse is competitive and 16× cheaper
  than the learned alternative.
- **The confidence output does not rank trustworthiness usefully** (§5e,
  r = −0.166, non-monotone deciles). It should not be used to gate anything.
- **The momentum error of any run is first order in `dt`** because the gimbal
  angles are integrated by explicit Euler. Halving `dt` halves the error
  (1.006766e-02 → 5.018896e-03 → 2.505703e-03 → 1.251914e-03 → 6.257226e-04
  N·m·s over dt = 0.1 to 0.00625 s, ratios 2.0060 / 2.0030 / 2.0015 / 2.0007,
  `validate_steering.py` §6) while the instantaneous torque error stays at
  6.7e-16 N·m. Pinned in
  `tests/test_simulate.py::TestRunSteering::test_momentum_error_is_first_order_in_dt`.
  Any absolute momentum-error number in this document is therefore a property of
  the integration step as much as of the steering law, which is why the
  instantaneous torque error is reported alongside it everywhere.
- **An earlier draft of the uncertainty script compared a pooled Monte Carlo rms
  with a mean first-order prediction** and reported a spurious ratio of 2.15.
  The aggregation is now per state. The defect is recorded here rather than
  silently fixed.
