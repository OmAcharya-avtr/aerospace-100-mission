# Validation — navbench 0.1.0 (Level 3, research-grade)

**Status: TESTING.** Every number in this document was produced by running the
committed scripts in `validation/` during the build session of **2026-08-29**.
The raw stdout of each run is committed beside the script as
`<script>_output.txt`. Nothing here is quoted from literature as if it were a
measurement, and nothing is estimated.

Reproduce any of it from the product root with

```bash
PYTHONPATH=src python3 validation/<script>.py
```

Every script exits non-zero on failure, except `v6` which is a **reporting**
script: an honest negative result for the learned model is a valid outcome,
not a test failure, so it always exits 0 and prints what it measured.

| # | Script | Raw output | Result |
|---|---|---|---|
| v1 | `v1_riccati_steady_state.py` | `v1_riccati_steady_state_output.txt` | **PASS** |
| v2 | `v2_nees_nis_consistency.py` | `v2_nees_nis_consistency_output.txt` | **PASS** |
| v3 | `v3_ukf_vs_ekf.py` | `v3_ukf_vs_ekf_output.txt` | **PASS** |
| v4 | `v4_mekf_quaternion.py` | `v4_mekf_quaternion_output.txt` | **PASS** |
| v5 | `v5_sensor_and_truth_models.py` | `v5_sensor_and_truth_models_output.txt` | **PASS** |
| v6 | `v6_adaptive_q_benchmark.py` | `v6_adaptive_q_benchmark_output.txt` | reports (see §6) |
| v7 | `v7_performance.py` | `v7_performance_output.txt` | **PASS** |

Total runtime of all seven scripts on the 2-core build machine: about **130 s**,
the longest single script being v5 at roughly 45 s. No script exceeds the 2-minute
per-script budget.

Automated tests: **715 passed** (`python -m pytest tests/ -q`, 39 s).
Style: `ruff check src/ tests/ examples/ validation/` → `All checks passed!`

---

## 1. The filter reproduces the analytic steady-state Riccati solution

`validation/v1_riccati_steady_state.py`

### 1a. Scalar random walk, hand-solved

Model `x_k = x_{k-1} + w`, `Var(w) = q`; `z_k = x_k + v`, `Var(v) = r`
(`F = H = 1`). At the fixed point

```
update : p⁺ = (1 − K) p,   K = p/(p + r)
predict: p  = p⁺ + q
```

so `p = p·r/(p+r) + q`, i.e. `p² − q p − q r = 0`, and the admissible root is

```
P⁻_∞ = ½( q + √(q² + 4 q r) )      K_∞ = P⁻_∞/(P⁻_∞ + r)      P⁺_∞ = P⁻_∞ − q
```

With `q = r = 1` the answer is the golden ratio φ = 1.618033988749895.

| q | r | hand `P⁻_∞` | filter `P⁻_∞` | \|diff\| | iterations |
|---|---|---|---|---|---|
| 1.0 | 1.0 | 1.618033988749895e+00 | 1.618033988749895e+00 | **2.220e−16** | 18 |
| 0.25 | 4.0 | 1.132782218537319e+00 | 1.132782218537317e+00 | 1.776e−15 | 68 |
| 2.0 | 0.5 | 2.414213562373095e+00 | 2.414213562373095e+00 | 4.441e−16 | 11 |
| 1e−3 | 1e3 | 1.000500124999992e+00 | 1.000500124994897e+00 | 5.095e−12 | 13351 |

Worst relative deviation over the first three cases: **4.441e−16**, tolerance
1e−12 → **PASS**.

The fourth row is the low-SNR case and is reported as measured: the fixed-point
iteration converges **linearly with a rate approaching 1** as `tr(Q)/tr(R)` → 0,
so the residual exceeds the last increment by ~50×. Tightening `tol` to 1e−16
reaches the floating-point floor of 1.1e−13 relative after 15447 iterations.
This is documented in `steady_state_riccati`'s docstring and pinned by
`tests/test_kf.py::test_scalar_closed_form_low_snr_is_looser`. It is a property
of the solver, not an error in the closed form.

### 1b. Two-state constant velocity — Kalata's published closed form

Discrete white-noise acceleration model, position-only measurement. Kalata's
tracking index gives the exact steady-state α–β gains (Kalata, P. R. (1984),
"The Tracking Index: A Generalized Parameter for α-β and α-β-γ Target
Trackers", *IEEE Trans. Aerospace and Electronic Systems* AES-20(2), 174-182):

```
Λ = σ_a T²/σ_v    ρ = (4 + Λ − √(8Λ + Λ²))/4    α = 1 − ρ²
β = 2(2 − α) − 4√(1 − α)                        K_∞ = [α, β/T]ᵀ
```

| T [s] | σ_a [m/s²] | σ_v [m] | Λ | Kalata `K_∞` | filter `K_∞` | max \|diff\| |
|---|---|---|---|---|---|---|
| 1.0 | 0.10 | 2.0 | 0.050 | [0.270867118992629, 0.042694639037219] | [0.270867118992628, 0.042694639037219] | **3.331e−16** |
| 0.5 | 1.00 | 0.5 | 0.500 | [0.628373457204967, 0.609611796797792] | [0.628373457204967, 0.609611796797792] | 4.441e−16 |
| 2.0 | 0.02 | 10.0 | 0.008 | [0.118799444828764, 0.003754891327687] | [0.118799444828764, 0.003754891327687] | 1.505e−16 |

Tolerance 1e−12 → **PASS**.

### 1c. Independent solver cross-check (SciPy DARE)

Same problem solved by `scipy.linalg.solve_discrete_are(Fᵀ, Hᵀ, Q, R)`. SciPy is
used only in this validation script, never inside the library.

| case | max \|P⁻(iterated) − P⁻(SciPy)\| |
|---|---|
| T=1.0, σ_a=0.10, σ_v=2.0 | 1.088e−14 |
| T=0.5, σ_a=1.00, σ_v=0.5 | 2.554e−15 |
| T=2.0, σ_a=0.02, σ_v=10 | 1.377e−12 |

Tolerance 1e−10 → **PASS**.

### 1d. A *running* filter converges to that solution

Not the algebra but the recursion the code actually executes: a
`KalmanFilter` stepped 600 times with random measurements.

| case | max \|P⁺(600 steps) − P⁺_∞\| (relative) | max \|K − K_∞\| (relative) |
|---|---|---|
| dt = 1.0 s, q̃ = 0.05 m²/s³, σ_z = 3 m | 1.332e−15 (**4.622e−16**) | 1.110e−16 (3.466e−16) |
| dt = 0.1 s, q̃ = 2.0 m²/s³, σ_z = 0.5 m | 1.887e−15 (2.213e−15) | 9.992e−16 (1.380e−15) |

Tolerance 1e−12 relative → **PASS**.

---

## 2. NEES and NIS inside their chi-squared bounds — and provably outside when mis-specified

`validation/v2_nees_nis_consistency.py`

60 independent Monte Carlo runs (seeds 90210 + i) × 200 steps of a 1-D CWNA
constant-velocity truth, `dt = 1 s`, `q̃ = 0.05 m²/s³`, `σ_z = 3 m`, burn-in 30.
Bounds from Bar-Shalom, Li & Kirubarajan (2001) Eq. (5.4.2-3).

Acceptance regions used throughout:

| statistic | dof | single sample | ensemble of M = 60 |
|---|---|---|---|
| NEES | 2 | [0.05064, 7.37776] | **[1.5262, 2.5369]** |
| NIS | 1 | — | **[0.6747, 1.3883]** |

### 2a. Correctly specified filter — must be INSIDE

| statistic | measured mean | bound | % of steps inside | verdict |
|---|---|---|---|---|
| ANEES (dof 2, expectation 2.0) | **2.0223** | [1.5262, 2.5369] | 98.2 % | **inside** |
| ANIS (dof 1, expectation 1.0) | **1.0241** | [0.6747, 1.3883] | 92.4 % | **inside** |

**PASS.**

### 2b. Deliberately mis-specified filters — must LEAVE the bounds

| mis-specification | ANEES | ANIS | % NEES steps inside | direction | verdict |
|---|---|---|---|---|---|
| `Q` too small by 25× | **26.6001** | 2.0903 | 0.0 % | above (optimistic) | **PASS** |
| `Q` too large by 25× | **1.1354** | 0.8150 | 2.4 % | below (pessimistic) | **PASS** |
| `R` too small by 9× | 10.5989 | **7.8740** | 0.0 % | above (optimistic) | **PASS** |
| `R` too large by 9× | 1.1287 | **0.1649** | 0.6 % | below (pessimistic) | **PASS** |

The under-modelled-`Q` case is 13× its own upper bound. This is the headline
claim of the product: a filter can be badly wrong about its own uncertainty and
the only thing that says so is this test.

### 2c. Innovation whiteness (Bar-Shalom et al. Eq. (5.4.3-2))

Band ±1.96/√N at 95 %, N = 170.

| filter | max \|ρ(l)\|, l ≥ 1 | band | ρ(1…5) | verdict |
|---|---|---|---|---|
| correct | **0.0984** at lag 10 | ±0.1503 | −0.0303, −0.0696, +0.0460, −0.0730, +0.0144 | white → **PASS** |
| `Q` too small by 25× | **0.4113** at lag 3 | ±0.1503 | +0.3922, +0.3588, +0.4113, +0.3211, +0.3489 | correlated → **PASS** |

---

## 3. UKF matches EKF when nearly linear, degrades more gracefully when not

`validation/v3_ukf_vs_ekf.py`

### 3a. Near-linear regime

2-D constant-velocity target at a mean range of 11 107 m, `σ_range = 20 m`,
`σ_bearing = 5 mrad` (cross-range resolution `r σ_θ` = 55.5 m), `q̃ = 0.05 m²/s³`,
200 steps, seed 31415. Criteria fixed before the run.

| quantity | measured | tolerance |
|---|---|---|
| EKF position RMSE | 14.606708082 m | — |
| UKF position RMSE | 14.607395092 m | — |
| relative RMSE difference | **4.703e−05** | 1e−2 |
| max \|x_EKF − x_UKF\| / σ_pos | **1.780e−03** | 1e−2 |
| max \|P_EKF − P_UKF\| / max\|P_EKF\| | 8.159e−06 | — |

**PASS.**

### 3b. Strongly nonlinear regime

Target passing 206 m from the sensor, `σ_range = 60 m`, `σ_bearing = 0.35 rad`
(20°, so the cross-range error at closest approach is ~72 m — comparable with
the range itself), `q̃ = 5 m²/s³`, 40 independent runs × 60 steps.
Divergence threshold: terminal NEES above `χ²₄(0.9999) = 23.51`.

| filter | mean RMSE [m] | median | p90 | mean NEES | diverged / 40 |
|---|---|---|---|---|---|
| EKF | **72.358** | 43.170 | 106.044 | **33.657** | **3** |
| UKF | **50.047** | 40.409 | 57.271 | **12.504** | **1** |

RMSE ratio EKF/UKF = **1.446**; NEES ratio = **2.692**. **PASS** (UKF lower on
both).

**Stated plainly:** *both* filters are inconsistent in this regime (mean NEES
33.7 and 12.5 against dof 4). The claim under test is comparative degradation,
not consistency. Neither filter should be trusted here.

### 3c. Control — the difference is the nonlinearity, not the implementation

With a linear measurement matrix and identical noise, both filters must reduce
to the linear KF:

| comparison | max relative \|Δx\| | max relative \|ΔP\| |
|---|---|---|
| EKF vs KF | **0.000e+00** | **0.000e+00** |
| UKF vs KF | 3.558e−16 | 4.824e−15 |

Tolerance 1e−11 relative → **PASS**.

---

## 4. Quaternion normalization and MEKF reset behaviour

`validation/v4_mekf_quaternion.py`

**Related prior art.** Product **P007 (QuatKit)** in this portfolio is a
dedicated quaternion toolbox using the same scalar-first Hamilton convention.
NavBench imports nothing from it — every product in this portfolio is
self-contained — so `navbench.attitude` is an independent implementation,
validated here against `scipy.spatial.transform.Rotation` as an outside
reference. P007 is cited as related work, not reused.

### 4a. Quaternion algebra (2000 random unit quaternions)

| check | measured | tolerance |
|---|---|---|
| DCM vs `scipy` `Rotation` | **6.661e−16** | 1e−14 |
| quat → DCM → quat round trip | 2.220e−16 | 1e−13 |
| DCM orthogonality `max\|RRᵀ − I\|` | 8.882e−16 | 1e−14 |
| `\|det R − 1\|` | 1.110e−15 | 1e−14 |
| `R(a⊗b) = R(a)R(b)` | 7.772e−16 | 1e−14 |
| rotation-vector exp/log **relative** round trip, 1e−14 … 3 rad | **2.168e−16** | 1e−14 |
| axis-angle round trip | 4.441e−16 | 1e−14 |

**PASS.**

> **A defect this check found.** The first run of this script failed the
> rotation-vector row at 1.98e−14 absolute (≈200 % relative) for `\|a\| = 1e−14`.
> The cause was a `1e-12` cut-off in `axis_angle_from_quat` that discarded the
> axis direction for very small rotations and returned `[1, 0, 0]` instead. The
> threshold is now the smallest normal double, and the tolerance was changed to
> **relative** because the absolute error legitimately scales with `\|a\|` across
> 14 decades. The regression is pinned by
> `tests/test_attitude.py::test_round_trip_relative`.

### 4b. Rigid-body integrator (RK4)

| check | measured | tolerance |
|---|---|---|
| principal-axis spin vs analytic, 200 s | `max\|1 − \|q·q_exact\|\|` = **2.220e−16** | 1e−14 |
| torque-free 300 s, dt = 0.1 s: relative energy drift | **5.948e−13** | 1e−12 |
| torque-free 300 s: inertial `\|H\|` drift | **2.923e−11** | 1e−9 |

Convergence order (10 s span, dt chosen to divide it exactly, `\|ω\| ≈ 1 rad/s`):

| dt [s] | final attitude error [rad] | ratio to previous |
|---|---|---|
| 0.10000 | 5.956265e−07 | — |
| 0.05000 | 3.711760e−08 | **16.05** |
| 0.02500 | 2.315924e−09 | **16.03** |
| 0.01250 | 1.446139e−10 | **16.01** |

RK4's ideal ratio is 16. **PASS.**

### 4c. Quaternion normalization

| check | measured |
|---|---|
| `quat_propagate`, 200 000 steps: `max\|\|q\| − 1\|` | **2.220e−16** (tolerance 1e−14) |
| contrast: 20 000 steps of **un-normalised** first-order Euler integration | `\|\|q\| − 1\|` = **1.184e−01** |
| MEKF, 600 steps with resets: `max\|\|q\| − 1\|` | **2.220e−16** |

**PASS.** The contrast row is why every propagation path renormalises.

### 4d. MEKF multiplicative reset

| check | measured | verdict |
|---|---|---|
| reference after reset equals `q_before ⊗ δq(â)` | max \|diff\| = **0.000e+00** (tolerance 1e−15) | **PASS** |
| `\|q\|` after reset | 1.00000000000000000 | — |
| innovation norm before / after reset | 2.291288e−03 → **9.165115e−09** rad | **PASS** |
| max reset angle over a 600-step run | 4.296e−02 rad (initial acquisition) | — |
| median reset angle after step 100 | 0.000e+00 rad (updates every 4th step) | — |
| **neglected** covariance-reset Jacobian `G = I − ½[â×]`: worst `\|GPGᵀ − P\|/max\|P\|` over the 20 largest resets | **4.613e−04** | reported |

The covariance reset Jacobian (Markley 2003, "Attitude Error Representations
for Kalman Filtering", *JGCD* 26(2), 311-317, §V) is **not** applied by this
implementation. Its neglected effect is second order in the reset angle and
measured at 4.6e−04 relative at the largest reset in the run — that is the
initial acquisition transient, not steady-state operation. **It would not be
negligible after a large attitude-acquisition manoeuvre**, and that limit is
repeated in README Limitations.

### 4e. MEKF Monte Carlo consistency

30 independent runs × 300 steps of 0.5 s, burn-in 50. Star tracker
σ = 3e−5 rad (6.2 arcsec) every 4th step; gyro ARW 0.05 deg/√hr, RRW
0.5 deg/hr^1.5. Initial attitude and bias errors drawn **from** `P₀`
(σ_a0 = 0.05 rad, σ_b0 = 2e−6 rad/s), which is what makes NEES meaningful.

| statistic | measured | bound | % of steps inside |
|---|---|---|---|
| ANEES (dof 6) | **6.1472** | [4.8247, 7.3015] | **99.2 %** |
| NIS (dof 3), 1890 pooled updates | **2.9768** | [2.8906, 3.1114] | — |

**PASS.**

### 4f. A discretisation trap this validation exposed

The truth generator's `interval_rate()` returns the *effective constant rate*
over each interval, `rotvec(q_k* ⊗ q_{k+1})/Δt` — which is what a
rate-integrating gyro actually reports (Farrenkopf 1978; Markley & Crassidis
2014 §4.7.2). Feeding the filter an endpoint sample of `ω` instead injects a
deterministic error:

| quantity | measured |
|---|---|
| `max\|ω(end of interval) − ω_effective\|` | 6.469e−05 rad/s |
| deterministic attitude error per step from an endpoint sample | **3.234e−05 rad** |
| per-step gyro angle noise `σ_v√Δt` for comparison | 1.028e−05 rad |

The bias is 3× the noise, so it is nearly invisible in RMSE but accumulates
coherently between star-tracker updates. **Measured before the fix: attitude
RMS 1.26e−04 rad, mean NEES 1925 against dof 6. After: 4.2e−05 rad, ANEES
6.15.** This is exactly the class of defect NEES exists to catch, and it is
recorded here rather than quietly corrected.

---

## 5. Sensor models and truth-trajectory conservation laws

`validation/v5_sensor_and_truth_models.py`

### 5a. Gyro Allan deviation vs IEEE Std 952-2020

400 000 samples at 0.01 s (4000 s), `σ_v = 1.000e−03 rad/s^{1/2}`,
`σ_u = 1.000e−03 rad/s^{3/2}`. Overlapping Allan variance from the integrated
angle (IEEE Std 952-2020, Annex C, Eq. (C.9)); theory
`σ_A(τ) = sqrt(σ_v²/τ + σ_u² τ/3)`; predicted minimum at
`τ* = √3 σ_v/σ_u = 1.7321 s`.

| τ [s] | measured [rad/s] | theory [rad/s] | ratio |
|---|---|---|---|
| 0.02 | 7.062799e−03 | 7.071539e−03 | **0.9988** |
| 0.10 | 3.167281e−03 | 3.167544e−03 | **0.9999** |
| 1.00 | 1.152806e−03 | 1.154701e−03 | 0.9984 |
| 2.00 | 1.069784e−03 | 1.080123e−03 | 0.9904 |
| 10.00 | 1.841294e−03 | 1.852926e−03 | 0.9937 |
| 50.00 | 3.754053e−03 | 4.084932e−03 | 0.9190 |

Tolerance is `max(3 × 1/√(2(N/m − 1)), 0.02)` — the Allan estimator's own
relative uncertainty, so long clusters are judged loosely because they are
intrinsically noisy. All rows pass.

| log-log slope | measured | theory |
|---|---|---|
| τ = 0.02 … 0.10 s (ARW-dominated) | **−0.4983** | −0.5 |
| τ = 10 … 50 s (RRW-dominated) | **+0.4417** | +0.5 |

**PASS.** The long-τ slope falls short of +0.5 because the longest cluster has
only 8 independent samples; that is estimator variance, not a model error.
**There is deliberately no flicker (slope 0) bias-instability plateau** in this
model — see README Limitations.

### 5b. Sensor noise statistics vs specification

| sensor | check | measured | spec | verdict |
|---|---|---|---|---|
| star tracker | `E\|r̂_b × b\|²` (2 observable components) | 5.006432e−09 | 2σ² = 5.000000e−09 (ratio **1.0013**) | PASS |
| star tracker | quaternion-output per-axis σ | [4.98193e−05, 5.03659e−05, 4.97892e−05] | 5.000e−05 | PASS |
| sun sensor | valid fraction with `eclipse_prob = 0.3` | **0.7006** | 0.700 ± 0.0097 | PASS |
| GPS | dropout fraction | **0.0487** | 0.05 | PASS |
| GPS | position σ per axis | [2.4981, 2.4972, 2.4879] m | 2.5 m | PASS |
| GPS | velocity σ per axis | [0.10028, 0.10017, 0.10114] m/s | 0.1 m/s | PASS |
| accelerometer | `max\|mean − (Rᵀa + b)\|` | 0.00018 m/s² | 3σ/√N = 0.00030 | PASS |
| accelerometer | relative σ error | 0.0059 | — | PASS |

### 5c. Two-body orbit propagator

μ = 3.986004418e14 m³/s² (IERS Conventions 2010, Table 1.1);
R_E = 6378137.0 m (WGS-84). 1000 RK4 steps per revolution.

| altitude | a [km] | period [s] | specific energy vs −μ/2a | energy drift / rev | \|h\| drift / rev | closure \|r(T) − r(0)\|/a |
|---|---|---|---|---|---|---|
| 500 km | 6878.1 | 5676.98 | rel. **0.000e+00** | **1.708e−12** | 8.539e−13 | **2.325e−10** |
| 800 km | 7178.1 | 6052.41 | rel. 1.342e−16 | 1.713e−12 | 8.562e−13 | 2.325e−10 |
| 35786 km (GEO) | 42164.1 | 86163.99 | rel. 0.000e+00 | 1.710e−12 | 8.553e−13 | 2.325e−10 |

Tolerances 1e−9 / 1e−9 / 1e−7 → **PASS**.

### 5d. Airborne coordinated turn

200 m/s at Ω = 0.02 rad/s over one full circle; analytic radius `|v|/Ω` = 10 000 m.

| check | measured | tolerance |
|---|---|---|
| max relative radius error | **7.094e−15** | 1e−12 |
| max relative ground-speed error | 7.674e−15 | 1e−12 |

**PASS.**

---

## 6. AI element — learned adaptive Q vs fixed hand-tuned Q vs classical Mehra IAE

`validation/v6_adaptive_q_benchmark.py` · full detail in `MODEL_CARD.md`

Truth acceleration PSD drawn per run as `q̃_true = q̃_nom · 10^u`,
`u ~ U(−1.5, 1.5)` (1/32 … 32×). Training: 150 runs, seeds 20260812 + i,
2550 windows. Held-out: 60 runs, seeds 20260812 + 100000 + i — **disjoint by
construction**. All three tuners adjust the same single scalar with the same
window (40) and cadence (20 steps), strictly causally.

Compute: dataset 8.22 s + fit 1.98 s + 60 runs × 3 tuners 16.41 s = **25.8 s total**.

### Results on the held-out runs

| tuner | position RMSE [m] | velocity RMSE [m/s] | ANEES (dof 2) | ANIS (dof 1) | \|log₁₀λ − u\| |
|---|---|---|---|---|---|
| fixed (hand-tuned `Q_nom`) | 2.09982 | 0.728119 | 5.5935 | 1.3330 | n/a |
| classical Mehra IAE | 2.27394 | 1.254886 | **1.1778** | 0.7396 | 1.8164 |
| **learned** | **1.92299** | **0.671698** | 4.4935 | 1.1691 | **0.5586** |

Acceptance bands over 60 runs: ANEES [1.5262, 2.5369], ANIS [0.6747, 1.3883].

### Is either adaptive method actually adapting?

| tuner | λ min | λ median | λ max | pinned at upper clip (64) | pinned at lower clip | corr(log₁₀λ, true u) |
|---|---|---|---|---|---|---|
| Mehra IAE | 64.0000 | 64.0000 | 64.0000 | **1.0000 (60/60 runs)** | 0.0000 | **−0.0000** |
| learned | 0.1460 | 0.5439 | 19.5782 | 0.0000 | 0.0000 | **+0.5943** |

**Read this before believing the consistency column.** The Mehra scheme is
pinned at its upper clip on **60 of 60** held-out runs and its correlation with
the true scale is **exactly zero**. It is therefore *not adapting at all*: it is
applying a constant 64× inflation of `Q`. That is why its ANEES (1.1778) sits
*below* the lower bound — the filter is uniformly pessimistic — and why that
value happens to land closer to dof = 2 than the two optimistic alternatives.

This is a measured property of the estimator in this regime, not an
implementation error: `Q̂ = K Ĉ Kᵀ` over a 40-step window, on a filter whose
innovations are dominated by measurement noise, produces a trace ratio above the
clip in every run — and the clip is mandatory because the raw estimator is
unbounded. The honest conclusion is that **classical IAE, restricted to a scalar
knob at this window length, does not work on this problem**, and that its
apparent consistency win is an artefact of saturation. The learned tuner, by
contrast, correlates with the true scale at **+0.5943** and never touches either
clip.

### The verdict, stated as measured — it is a SPLIT

* **The learned tuner wins on error.** Lowest position RMSE (1.923 m vs 2.100 m
  fixed and 2.274 m Mehra), beating fixed on **44/60** runs with a paired mean
  difference of **−0.17683 ± 0.09361 m** (95 % CI, excludes zero) and Mehra on
  **47/60** (−0.35094 ± 0.14531 m). It also recovers the true scale far better:
  mean |log₁₀λ − u| = **0.5586** vs Mehra's **1.8164**.
* **The classical scheme has the ANEES closest to the ideal 2.0 (1.1778) — but
  it earns that by saturating, not by adapting** (see the table above: pinned at
  the clip on 60/60 runs, zero correlation with the true scale). The learned
  tuner reads 4.4935 and the fixed baseline 5.5935. Calling Mehra the
  consistency winner without that caveat would be misleading.
* **None of the three is actually consistent.** Mehra is *below* the lower bound
  (pessimistic), the learned tuner and the fixed baseline are *above* the upper
  bound (optimistic). Scaling a single scalar is not enough to make this filter
  consistent across a 1000× spread in true process noise. That is the honest
  headline and it is repeated in README and MODEL_CARD.
* **Reporting only RMSE would have hidden the consistency split, and reporting
  only the consistency numbers would have hidden the saturation.** Both are the
  argument for keeping error metrics, consistency metrics and estimator
  diagnostics visible together — which is what this product is for.

### Stratified by the size of the mis-specification

| stratum | n | fixed RMSE / NEES | Mehra RMSE / NEES | learned RMSE / NEES |
|---|---|---|---|---|
| \|u\| ≤ 0.5 (within ~3×) | 20 | **1.80124** / 2.5307 | 2.27280 / **1.1310** | 1.86642 / 3.4732 |
| 0.5 < \|u\| ≤ 1.0 (3–10×) | 18 | 1.77850 / 2.8652 | 2.23129 / **1.1018** | **1.69454** / 3.1482 |
| \|u\| > 1.0 (>10×) | 22 | 2.63417 / 10.6102 | 2.30986 / **1.2826** | **2.16133** / 6.5218 |

**The fixed baseline is the best choice when the mis-specification is small**
(|u| ≤ 0.5): adapting cannot help when nothing needs adapting, and the learned
tuner's noise costs it 3.6 % RMSE there. The learned tuner earns its place only
at |u| > 0.5, where it cuts RMSE by 18 % against fixed at |u| > 1.0.

### Confidence output — a measured NEGATIVE result

| quantity | measured |
|---|---|
| mean confidence `exp(−σ_ensemble)` over held-out runs | 0.9206 |
| range | [0.9001, 0.9534] |
| **correlation(confidence, \|log₁₀λ − u\|)** | **+0.2206** |
| mean \|log₁₀λ − u\| in the high-confidence half | 0.6419 |
| mean \|log₁₀λ − u\| in the low-confidence half | 0.4752 |

A useful confidence output should be **negatively** correlated with the error.
The measured correlation is **positive (+0.22)**, and the high-confidence half
is *worse* than the low-confidence half (0.642 vs 0.475). **The confidence
output as implemented does not carry usable information about the prediction
error and must not be relied on.** The ensemble spread is also very narrow
(0.900–0.953), consistent with five bootstrap-resampled gradient-boosted trees
on 2550 samples being nearly identical models — bootstrap spread is measuring
model variance, which here is small, not predictive uncertainty. This was not
retuned to make it look better. See MODEL_CARD §7 and README Limitations.

---

## 7. Performance

`validation/v7_performance.py` · single-threaded, `n_jobs = 1` throughout.
numpy 2.4.4, Python 3.11.15, 2-core build machine.

| benchmark | time | throughput |
|---|---|---|
| linear KF, 2 states, 20 000 steps | 2.4891 s | **8035 steps/s** |
| EKF, 4 states, analytic Jacobians, 5000 steps | 0.6152 s | **8127 steps/s** |
| EKF, 4 states, numerical Jacobians, 2000 steps | 0.3993 s | 5009 steps/s |
| UKF, 4 states, 9 sigma points, 5000 steps | 1.2490 s | **4003 steps/s** |
| MEKF, 6 error states, 5000 steps | 0.7338 s | **6814 steps/s** |
| rigid-body RK4 attitude, 20 000 steps | 5.7828 s | 3459 steps/s |
| two-body RK4 orbit, 20 000 steps | 0.3262 s | 61 303 steps/s |
| NEES over 50 000 samples, 4 states | 0.4732 s | 105 670 samples/s |
| steady-state Riccati, 2 states, 200 solves | 1.1322 s | 176.6 solves/s |

**ML compute budget:** dataset generation (150 runs, 2550 windows) 9.92 s +
ensemble fit 2.08 s = **11.99 s total**, against the build guide's 180 s
budget. **PASS.**

Bounds roughly 5–10× looser than these figures are asserted in
`tests/test_performance.py`, so a real regression fails the suite while a slower
host does not.

---

## 8. What was NOT validated

Read this section before trusting anything above.

1. **No real hardware data.** Every trajectory, every sensor sample and every
   measurement in this package is synthetic and generated by the committed
   scripts. No gyro, star tracker, accelerometer or GNSS receiver was involved.
   The filters are validated against *the models*, and the models are validated
   against *their own analytic forms* — not against flight data.
2. **No third-party filter cross-check** beyond the SciPy DARE solver in §1c
   and `scipy.spatial.transform.Rotation` in §4a. There is no comparison with a
   reference GNC implementation.
3. **The gyro model has no flicker (1/f) bias-instability floor.** A real Allan
   deviation curve has a flat region between the ARW and RRW asymptotes; this
   model has only the two asymptotes. Datasheet "bias instability" figures
   therefore cannot be entered directly.
4. **The orbit model is pure two-body.** No J2, drag, third body or solar
   radiation pressure. It is a filter test signal, not an ephemeris.
5. **GNSS errors are white and isotropic.** Real errors are time-correlated
   (ionosphere, multipath) and geometry-dependent (DOP). Neither is modelled.
6. **The MEKF covariance reset Jacobian is neglected** (§4d), measured at
   4.6e−04 relative at the largest reset seen. Not validated for large
   acquisition manoeuvres.
7. **The learned tuner is validated only on the generative process that
   produced its training data** — the same CWNA model with a scalar Q
   mis-specification. It has never seen a manoeuvring target, a different
   dynamic model, coloured process noise, or a mis-specified `R`. Its
   confidence output is measurably uninformative (§6).
8. **Divergence detection is a convention, not a theorem.** The threshold is
   the single-sample `χ²_n` 99.99 % quantile, chosen by this package.
9. **Single-run time-averaged NEES/NIS are indicative only.** Successive steps
   are correlated, so the chi-squared bounds do not strictly apply; every
   defensible claim in this document uses independent Monte Carlo runs. The
   library labels the two cases differently (`independent=True/False`) and the
   CLI prints the caveat.
