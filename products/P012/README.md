# NavBench

**Status:** TESTING · **Class:** flagship · **Validation level:** 3 · **AI:** yes

A controlled bench for aerospace attitude and navigation filters. One truth
trajectory, one sensor suite, four estimators — and the diagnostic almost every
other filter toolkit omits: **NEES and NIS against chi-squared bounds**, so you
can see not just how wrong a filter is, but whether it *knows* how wrong it is.

---

## Executive overview

Filter selection in GNC is largely folklore. "Use a UKF, it handles
nonlinearity better." "Bump `Q` until it stops lagging." Both statements can be
true and both can be catastrophically wrong for a given problem, and RMS error
will not tell you which.

NavBench does four things:

1. **Generates truth** — rigid-body attitude under applied torque (RK4,
   convergence order measured at **16.05 / 16.03 / 16.01** against the ideal 16),
   plus a two-body orbit (**1.7e−12** relative energy drift per revolution) or a
   flat-Earth airborne track.
2. **Models sensors honestly** — a gyro whose Allan deviation matches the IEEE
   Std 952-2020 two-term form to within **0.1 %** over the ARW-dominated decade,
   a star tracker, a sun sensor with eclipse and field-of-view handling, an
   accelerometer, and a GPS-like fix with dropouts.
3. **Runs four estimators** on identical data — linear KF, EKF, UKF and a
   multiplicative EKF with a quaternion reference state.
4. **Scores consistency, not just error.** A correctly specified filter's
   ensemble-average NEES lands at **2.0223** inside its 95 % band
   [1.5262, 2.5369]; the same filter with `Q` 25× too small reads **26.6001** —
   thirteen times its own upper bound — while its RMSE only rises by 78 %.

The AI element is a learned adaptive process-noise tuner, and **the honest
result is a split.** The learned tuner gives the lowest held-out position RMSE
(1.923 m vs 2.100 m for a fixed hand-tuned `Q`, winning 44 of 60 paired runs)
and is the only method whose scale estimate tracks the truth (correlation
**+0.59** vs the classical scheme's **0.00**). The classical Mehra scheme has
the ANEES closest to the ideal 2.0 — **but only because it saturates at its clip
on 60 of 60 runs**, applying a constant 64× inflation rather than adapting.
**None of the three is actually consistent**, and the learned tuner's confidence
output is measurably uninformative (correlation with the actual error **+0.22**,
the wrong sign). All of that is in `MODEL_CARD.md` rather than buried.

This is a v0.1 release. Nothing in it has been compared against data from a
real vehicle.

## Aerospace problem

Every spacecraft, aircraft and UAV runs an estimator, and every estimator
reports a covariance that something downstream believes: a guidance law sizing
a manoeuvre, a fault detector setting a threshold, a fusion node weighting one
sensor against another, an integrity monitor deciding whether a fix is usable.

When that covariance is wrong, nothing obviously breaks. The estimate still
looks plausible. RMS error may even be fine. What breaks is everything that
trusted the number — and it breaks later, in flight, in a way that is hard to
trace back.

The tool for catching this is nearly a century old in its statistics and forty
years old in its GNC form: normalised estimation error squared (NEES) and
normalised innovation squared (NIS), tested against chi-squared bounds over a
Monte Carlo ensemble (Bar-Shalom, Li & Kirubarajan 2001, §5.4). It is standard
practice in tracking literature and routinely absent from filter libraries.
NavBench makes it the primary output.

## Intended users

- **GNC engineers** choosing between EKF, UKF and MEKF formulations for an
  attitude or navigation problem, and needing evidence rather than folklore.
- **Estimation researchers** who need a reproducible bench with truth, sensor
  models and consistency scoring already wired together.
- **Systems engineers** validating that a supplier's filter reports an honest
  covariance.
- **Students** of estimation theory: every equation is cited, and the failure
  modes are demonstrated rather than described.

Not intended for: flight software, operational navigation, certification
evidence, or any decision affecting real hardware.

## Engineering theory

Every equation below carries its source, units, assumptions and validity range.

### Attitude kinematics and rigid-body dynamics

| Quantity | Equation | Units | Source |
|---|---|---|---|
| Quaternion kinematics | `q̇ = ½ q ⊗ [0, ω_body]` | 1/s | Markley & Crassidis 2014, Ch. 3; Shuster 1993 §2 |
| Exact propagation, constant ω | `q(t+Δt) = q ⊗ [cos(\|ω\|Δt/2), sin(\|ω\|Δt/2) ω/\|ω\|]` | — | Markley & Crassidis 2014 Eq. (3.21) |
| Attitude matrix | `R = (q₀² − \|q_v\|²)I + 2 q_v q_vᵀ + 2 q₀[q_v×]` | — | Shuster 1993 §2 (active form) |
| Euler's equation | `J ω̇ = τ − ω × (J ω)` | N·m | Wertz 1978 Eq. (16-3); Markley & Crassidis 2014 Eq. (3.81) |

**Convention (governs the whole package):** scalar-first `q = [q₀, q₁, q₂, q₃]`,
Hamilton product, `q` is body-to-inertial so `v_inertial = R(q) v_body`, `ω` in
body axes. This matches `scipy.spatial.transform.Rotation` up to scipy's
scalar-last storage, verified to **6.661e−16** over 2000 random quaternions.
**Assumptions:** rigid body, constant inertia, no reaction wheels or slosh.
**Validity:** any rate; the closed-form propagation is exact for constant ω and
first-order accurate for a sampled time-varying rate.

### Two-body orbital motion

| Quantity | Equation | Units | Source |
|---|---|---|---|
| Two-body acceleration | `r̈ = −μ r/\|r\|³` | m/s² | Vallado 2013 §1.3 Eq. (1-14) |
| Circular speed | `v = sqrt(μ/r)` | m/s | Vallado 2013 Eq. (1-32) |
| Specific energy | `ε = v²/2 − μ/r = −μ/(2a)` | J/kg | Vallado 2013 §1.3 |
| μ (Earth) | 3.986004418e14 | m³/s² | IERS Conventions (2010), Table 1.1 |
| R_E | 6378137.0 | m | WGS-84 |

**Assumptions:** point-mass central body. **Validity:** minutes to hours as a
filter test signal. **Not modelled:** J2, drag, third body, solar radiation
pressure. This is not an ephemeris propagator.

### Gyro noise (IEEE Std 952-2020, Annex C)

```
ω_meas(t) = ω_true(t) + b(t) + η_v(t)          ḃ(t) = η_u(t)
E[η_v η_vᵀ] = σ_v² I δ(t−t')   σ_v  [rad/s^{1/2}]   angle random walk
E[η_u η_uᵀ] = σ_u² I δ(t−t')   σ_u  [rad/s^{3/2}]   rate random walk
```

Sampled at Δt (Markley & Crassidis 2014 Eqs. (4.53)-(4.54)):
`ω_k = ω_true,k + b_k + (σ_v/√Δt)N(0,I)`, `b_{k+1} = b_k + σ_u√Δt N(0,I)`.

Implied Allan deviation: `σ_A(τ) = sqrt(σ_v²/τ + σ_u² τ/3)` — slope −1/2 at
short τ, +1/2 at long τ, minimum at `τ* = √3 σ_v/σ_u`. Measured against this
form in `validation/v5`. Datasheet conversions:
`σ_v = ARW[deg/√hr]·(π/180)/60`, `σ_u = RRW[deg/hr^{3/2}]·(π/180)/3600^{1.5}`.
**Validity:** Δt short compared with the bias correlation time.
**Not modelled:** the flicker (1/f) bias-instability plateau — see Limitations.

### Kalman filter family

Recursions follow Kalman (1960) in the notation of Bar-Shalom, Li &
Kirubarajan (2001) §5.2. The covariance update is **always** Joseph form,
`P⁺ = (I−KH)P⁻(I−KH)ᵀ + K R Kᵀ`, which is algebraically equal to `(I−KH)P⁻`
only at the optimal gain but stays symmetric positive semi-definite for **any**
gain (Bucy & Joseph 1968; Bierman 1977 §3.2). Demonstrated in
`tests/test_failure_modes.py`: over-relaxing the gain by 1.5× drives the short
form to a negative eigenvalue while the Joseph form stays positive definite.

* **EKF** — first-order linearisation (Bar-Shalom et al. §10.3; Jazwinski 1970
  §8.3). **Validity:** the linearisation error is O(\|x−x̂\|²) weighted by the
  model curvature.
* **UKF** — scaled unscented transform, `λ = α²(n+κ) − n`, lower-Cholesky
  sigma points taken column-wise (Julier & Uhlmann 1997, 2004; Wan & van der
  Merwe 2000). Captures mean and covariance to second order for any
  nonlinearity, exactly for affine maps. **Validity note:** small α amplifies
  round-off by roughly `1/α²`.

### Multiplicative EKF (MEKF)

State `x = [a (3); Δb (3)]` — attitude error rotation vector [rad] and gyro
bias error [rad/s] — with the multiplicative error definition
`q_true = q̂ ⊗ δq(a)`. Error dynamics (Lefferts, Markley & Shuster 1982 §III;
Markley & Crassidis 2014 §6.2.4):

```
ȧ  = −[ω̂×] a − Δb − η_v          ω̂ = ω_gyro − b̂
Δḃ = η_u
```

Discrete transition `Φ = exp(F_c Δt)` in closed form; process noise
(Farrenkopf 1978; Markley & Crassidis 2014 Eq. (6.93)):

```
Q_d = [[ (σ_v²Δt + σ_u²Δt³/3) I ,  −(σ_u²Δt²/2) I ],
       [ −(σ_u²Δt²/2) I         ,   σ_u²Δt I       ]]
```

Unit-vector measurements use `H = [[r̂_b×], 0]`; the rank-2 QUEST covariance
`σ²(I − r̂_b r̂_bᵀ)` (Shuster & Oh 1981) is regularised to `σ² I`, the standard
treatment. Reset: `q̂⁺ = normalize(q̂ ⊗ δq(â))`, `b̂⁺ = b̂ + Δb̂`, `x⁺ = 0`.
The covariance reset Jacobian `I − ½[â×]` (Markley 2003 §V) is **neglected**;
its magnitude is measured, not assumed — see Limitations.

**Related prior art.** Product **P007 (QuatKit)** in this portfolio is a
dedicated quaternion toolbox with the same convention, and **P017 (EstimKit)**
is a compact KF/EKF/UKF/RTS family. NavBench **imports neither** — every
product in this portfolio is self-contained — and reimplements the algebra
independently, validating it here against `scipy.spatial.transform.Rotation`
and against published closed forms. Both are cited as related work, not reused.

### Consistency diagnostics — the reason this product exists

Bar-Shalom, Li & Kirubarajan (2001) §5.4:

```
NEES  ε_k   = x̃_kᵀ P_k⁻¹ x̃_k      ~ χ²_n    E[ε] = n     (needs truth)
NIS   ε^ν_k = ν_kᵀ S_k⁻¹ ν_k       ~ χ²_m    E[ε^ν] = m   (no truth needed)
```

Over M **independent** runs, `M ε̄ ~ χ²_{M·d}`, giving the acceptance region
`[χ²_{M d}(α/2)/M, χ²_{M d}(1−α/2)/M]` (Eq. 5.4.2-3). Above the band → the
filter is **optimistic** (covariance too small, the classic under-modelled `Q`);
below → **pessimistic**. Whiteness is tested separately with the ±1.96/√N band
(Eq. 5.4.3-2).

**Validity caveat, enforced in the API.** Averaging over *time* within one run
assumes independence that does not hold. `ConsistencyResult` carries an
`independent` flag, single-run results are labelled *indicative*, and every
defensible claim in `validation/` uses independent Monte Carlo runs.

## Architecture

```
src/navbench/
├── attitude.py     quaternion algebra, DCM, Euler angles, rigid-body dynamics
├── truth.py        AttitudeTruth / OrbitTruth / AirborneTruth generators
├── sensors.py      GyroModel, StarTrackerModel, SunSensorModel,
│                   AccelerometerModel, GpsModel
├── models.py       CWNA/DWNA constant-velocity models, radar measurement
├── kf.py           KalmanFilter, Joseph update, steady-state Riccati solver
├── ekf.py          ExtendedKalmanFilter, numerical Jacobian
├── ukf.py          UnscentedKalmanFilter, MerweSigmaPoints, unscented transform
├── mekf.py         MultiplicativeEKF, Phi(omega, dt), Farrenkopf Q_d
├── consistency.py  nees, nis, chi2_bounds, consistency_test,
│                   ensemble_consistency, innovation_whiteness
├── adaptive.py     MehraAdaptiveQ (classical), LearnedAdaptiveQ (ML),
│                   innovation_features, run_adaptive_kf, dataset generator
├── bench.py        score_run, compare_scores, divergence convention
└── cli.py          python -m navbench {riccati,bench,attitude,consistency,adaptive}
```

Dependencies: numpy, scipy, scikit-learn. Matplotlib only in `examples/`
(Agg backend). No PyTorch. No cross-product imports.

## Installation

```bash
cd products/P012
pip install -e ".[dev]"            # or just run with PYTHONPATH=src
```

Python 3.11+. Everything below works without installing, using `PYTHONPATH=src`.

## Quick start

```python
import numpy as np
from navbench import (KalmanFilter, constant_velocity_cwna, simulate_linear_system,
                      nees, ensemble_consistency)

f, q = constant_velocity_cwna(dt=1.0, q_psd=0.05)   # m^2/s^3
h, r = np.array([[1.0, 0.0]]), np.array([[9.0]])    # position-only, sigma = 3 m

runs = np.zeros((60, 200))
for i in range(60):                                  # independent Monte Carlo
    rng = np.random.default_rng(90210 + i)
    truth, meas = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 200, rng)
    res = KalmanFilter(f, h, q, r, np.zeros(2), np.diag([100.0, 10.0])).run(meas)
    runs[i] = nees(truth - res.x_post, res.p_post)

avg, lo, hi = ensemble_consistency(runs[:, 30:], dof=2)
print(f"ANEES {avg.mean():.4f} in [{lo:.4f}, {hi:.4f}]")   # -> 2.0223 in [1.5262, 2.5369]
```

Now break it on purpose — pass the filter a `Q` 25× too small and the same line
prints `ANEES 26.6001`, while position RMSE only rises from 1.68 m to 2.98 m.
That gap is the product.

## Configuration

CLI (`python -m navbench <cmd> [--json]`, exit code 2 with a one-line message
on invalid input):

| subcommand | what it does | key options |
|---|---|---|
| `riccati` | steady-state Riccati solution for a named model | `--model {random-walk,cv-cwna,cv-dwna} --q --r --dt` |
| `bench` | KF/EKF/UKF on the radar scenario, scored | `--seed --steps --q --sigma-range --sigma-bearing --alpha --burn-in` |
| `attitude` | MEKF on a rigid-body attitude scenario | `--steps --dt --arw --rrw --sigma-st --meas-every --sigma-a0 --sigma-b0` |
| `consistency` | Monte Carlo NEES/NIS with chi-squared bounds | `--runs --steps --q-mismatch --alpha-level` |
| `adaptive` | train and benchmark the three Q tuners | `--train-runs --test-runs --steps --members` |

```console
$ python -m navbench consistency --runs 50 --q-mismatch 0.04
navbench consistency — 50 independent runs x 150 steps, Q mismatch factor 0.04
  ANEES (dof 2): mean 25.9..., bounds [1.4844, 2.5912], 0.0 % of steps inside
```

## Examples

Each writes a PNG into `screenshots/`:

| script | figure |
|---|---|
| `examples/estimator_bench.py` | EKF / UKF / converted-KF on one radar truth: geometry, error, NEES with bounds, error-vs-consistency bars |
| `examples/nees_nis_consistency.py` | ensemble NEES and NIS against their bands for a correct filter and for `Q` 25× too small / too large |
| `examples/mekf_attitude.py` | attitude and gyro-bias error with 3σ envelopes, attitude NEES, and the multiplicative reset angle |
| `examples/adaptive_q_tuning.py` | held-out RMSE and NEES for the three tuners, scale recovery, and λ over one run |
| `examples/gyro_allan_deviation.py` | measured vs analytic Allan deviation for three gyro grades, plus a bias random-walk realisation |

```bash
PYTHONPATH=src python3 examples/estimator_bench.py
```

## Validation

Full evidence with raw script output in **`validation/VALIDATION.md`**; the
requirement-to-evidence matrix is in **`docs/REQUIREMENTS.md`**. All seven
scripts were executed in the build session; each saves its stdout beside it.

| check | result |
|---|---|
| Scalar Riccati vs hand-solved closed form (`q = r = 1` → golden ratio 1.618033988749895) | \|diff\| **2.220e−16** |
| 2-state gains vs Kalata's published α–β closed form | max \|diff\| **3.331e−16** (tolerance 1e−12) |
| Cross-check vs `scipy.linalg.solve_discrete_are` | **1.088e−14** |
| A *running* filter converges to `P⁺_∞` in 600 steps | **4.622e−16** relative |
| Correctly specified filter, 60 runs: ANEES (dof 2) | **2.0223** in [1.5262, 2.5369]; ANIS **1.0241** in [0.6747, 1.3883] |
| `Q` 25× too small → optimistic | ANEES **26.6001**, 0.0 % of steps inside |
| `Q` 25× too large → pessimistic | ANEES **1.1354** |
| `R` 9× too small / too large → NIS | **7.8740** / **0.1649** |
| Innovation whiteness, correct vs `Q`/25 | max \|ρ\| **0.0984** vs **0.4113**, band ±0.1503 |
| UKF vs EKF, near-linear (11 km range) | relative RMSE difference **4.703e−05** |
| UKF vs EKF, strongly nonlinear (206 m closest approach, 20° bearing σ) | mean RMSE **50.0** vs **72.4** m; mean NEES **12.5** vs **33.7**; 1 vs 3 divergences in 40 runs |
| Both reduce to the linear KF on a linear measurement | **0.0** (EKF) / **3.6e−16** (UKF) relative |
| Quaternion algebra vs `scipy` Rotation, 2000 quaternions | **6.661e−16** |
| RK4 convergence order (ideal 16) | **16.05 / 16.03 / 16.01** |
| `quat_propagate` over 200 000 steps | max \|\|q\|−1\| **2.220e−16** (un-normalised Euler: 1.184e−01) |
| MEKF reset equals `q_before ⊗ δq(â)` | **0.000e+00** |
| MEKF Monte Carlo, 30 runs: ANEES (dof 6) | **6.1472** in [4.8247, 7.3015], 99.2 % of steps inside; NIS **2.9768** in [2.8906, 3.1114] |
| Gyro Allan deviation vs IEEE 952 form | ratio **0.9988–0.9190**; slopes **−0.4983** / **+0.4417** vs −0.5 / +0.5 |
| Two-body orbit, energy drift per revolution | **1.708e−12**; closure **2.325e−10** |
| Coordinated-turn radius vs `\|v\|/Ω` | **7.094e−15** relative |

**Three defects that this validation caught during the build** are recorded in
`docs/REQUIREMENTS.md` §4 rather than quietly fixed: a small-rotation axis
defect in `axis_angle_from_quat`, a cancellation range in the MEKF transition
matrix, and a rate-discretisation bias that produced a mean NEES of **1925**
against dof 6 before it was found.

## Benchmark results

Single-threaded on the 2-core build machine (`validation/v7_performance.py`):

| benchmark | throughput |
|---|---|
| linear KF, 2 states | **8035 steps/s** |
| EKF, 4 states, analytic Jacobians | **8127 steps/s** |
| EKF, 4 states, numerical Jacobians | 5009 steps/s |
| UKF, 4 states, 9 sigma points | **4003 steps/s** |
| MEKF, 6 error states | **6814 steps/s** |
| rigid-body RK4 attitude truth | 3459 steps/s |
| two-body RK4 orbit truth | 61 303 steps/s |
| NEES, 4-state error | 105 670 samples/s |
| steady-state Riccati solve | 176.6 solves/s |

ML pipeline: dataset 9.92 s + fit 2.08 s = **11.99 s** against a 180 s budget.

## AI model details

Full detail in **`MODEL_CARD.md`**; data provenance in **`DATASET_CARD.md`**.

**Problem.** Predict `log₁₀ λ` with `Q = λ Q_nominal` from a sliding window of
the filter's own innovation statistics.

**Baselines, implemented and validated first.** (a) a fixed hand-tuned `Q` set
at the geometric centre of the test distribution — the best possible single
fixed choice; (b) classical innovation-based adaptive estimation
`Q̂ = K Ĉ Kᵀ` (Mehra 1970/1972; Mohamed & Schwarz 1999 Eq. 12), projected onto
the same scalar knob so the comparison is like for like.

**Architecture.** Five bootstrap-resampled `GradientBoostingRegressor`s
(150 trees, depth 3) over six scale-free innovation features. No model artifact
is committed; it regenerates deterministically in 12 s.

**Test split.** Train seeds `20260812 + i`, `i ∈ [0,150)` (2550 windows);
held-out seeds `20260812 + 100000 + i`, `i ∈ [0,60)` — **disjoint by
construction**. Evaluation is strictly causal.

**Metrics on the held-out runs:**

| tuner | position RMSE [m] | ANEES (dof 2) | ANIS (dof 1) | \|log₁₀λ − u\| |
|---|---|---|---|---|
| fixed (hand-tuned) | 2.09982 | 5.5935 | 1.3330 | n/a |
| classical Mehra IAE | 2.27394 | **1.1778** | 0.7396 | 1.8164 |
| **learned** | **1.92299** | 4.4935 | 1.1691 | **0.5586** |

95 % bands: ANEES [1.5262, 2.5369], ANIS [0.6747, 1.3883].

**The result stated plainly — it is a split, not a win.**

* The **learned tuner wins on error**: lowest RMSE, 44/60 paired wins over the
  fixed baseline, paired mean difference **−0.17683 ± 0.09361 m** (95 % CI
  excludes zero), and it recovers the true scale 3.3× better than Mehra.
* The **classical scheme has the ANEES closest to 2.0 (1.1778) — but it gets
  there by saturating, not by adapting.** It is pinned at its upper clip on
  **60 of 60** held-out runs with **zero** correlation to the true scale, so it
  is applying a constant 64× inflation of `Q`. The learned tuner correlates with
  the true scale at **+0.5943** and never touches either clip. Reporting the
  ANEES column without this sentence would be misleading.
* **None of the three is consistent.** Mehra is below its lower bound, the
  other two above their upper bound. A single scalar on `Q` is not enough
  across a 1000× spread in true process noise. The fix is a richer `Q`
  parameterisation, not a better regressor on the same knob.
* At small mis-specification (\|u\| ≤ 0.5) the **fixed baseline is best**
  (1.801 vs 1.866 m): adapting cannot help when nothing needs adapting.
* **The confidence output is uninformative.** Correlation with the actual error
  is **+0.22** — the wrong sign — and the high-confidence half of the runs is
  *worse* than the low-confidence half (0.642 vs 0.475). The bootstrap spread
  measures model variance, which is small here, not predictive uncertainty.
  **It must not be used as a health signal.** No retuning was done to improve
  the appearance of this result.

**Uncertainty output.** `AdaptiveQPrediction(log10_scale, log10_std, scale,
confidence, extrapolating)`. The `extrapolating` flag is meaningful and tested;
`confidence` is not (above).

**This model is not certified for operational flight use.**

## Hardware requirements

CPU only. Runs comfortably on 2 cores with under 200 MB of memory. No GPU, no
PyTorch, no network access. The full test suite takes 39 s; all seven validation
scripts together about 130 s.

## Limitations

1. **No real hardware data anywhere.** Every trajectory and every sensor sample
   is synthetic and generated by committed scripts. The filters are validated
   against *the models*; the models are validated against *their own analytic
   forms*. Nothing here has been compared with a flown gyro, star tracker or
   GNSS receiver.
2. **The gyro has no flicker (1/f) bias-instability floor.** A real Allan
   deviation curve has a flat plateau between the ARW and RRW asymptotes; this
   model has only the two asymptotes. A datasheet "bias instability" figure
   cannot be entered directly.
3. **The orbit model is pure two-body** — no J2, drag, third body or SRP. It is
   a filter test signal, not an ephemeris.
4. **GNSS errors are white and isotropic.** Real errors are time-correlated
   (ionosphere, multipath) and geometry-dependent (DOP). Neither is modelled.
5. **The MEKF neglects the covariance reset Jacobian** `I − ½[â×]`
   (Markley 2003 §V). Measured effect at the largest reset in a 600-step run:
   **4.613e−04** relative on `P`. That reset was the initial acquisition
   transient; the steady-state resets are sub-milliradian. **It would not be
   negligible after a large attitude-acquisition manoeuvre.**
6. **Single-run time-averaged NEES/NIS are indicative only.** Successive steps
   are correlated, so the chi-squared bounds do not strictly apply. The library
   flags this (`independent=False`), the CLI prints the caveat, and every
   defensible claim uses independent Monte Carlo runs.
7. **Divergence detection is a convention**, not a theorem: terminal NEES above
   the single-sample `χ²_n` 99.99 % quantile.
8. **The learned tuner generalises only over what it saw**: a 1-D CWNA truth
   with a correctly specified `R` and a *scalar* `Q` error. It has never seen a
   manoeuvring target, a different dynamic model, coloured noise, or a
   mis-specified `R` — and it will mis-attribute an `R` error to `Q`.
9. **The learned tuner's confidence output does not work** (see AI model
   details). This is measured, not suspected.
10. **The UKF's `W₀ᶜ` may be negative** for small α, so the reconstructed
    covariance is not guaranteed positive definite for pathological inputs.
    The failure surfaces as `CovarianceCollapseError` rather than silently.
11. **The numerical-Jacobian fallback is a convenience**, accurate to about
    `eps^{2/3} ≈ 4e−11` relative. Do not use it for tight consistency work.
12. **The bench covers only the filters it implements.** No particle filter, no
    square-root or UD-factorised forms, no smoother, no IMM, no multi-hypothesis
    association.
13. **Only one classical adaptive scheme is implemented** (Mehra-style IAE), and
    on this benchmark it saturates at its clip on every held-out run rather than
    adapting. That is a measured property of `Q̂ = K Ĉ Kᵀ` at a 40-step window on
    this problem, but it means the classical arm of the comparison is weaker than
    a fuller study would make it. Sage-Husa, multiple-model adaptive estimation
    and variational Bayes methods are not compared.

## Safety statement

This software is **research-grade**. It is **not flight-qualified, not
certified, and not approved for operational aerospace use.** It must not be
placed in any control, navigation or safety loop whose failure could cause
harm, loss of vehicle, or loss of mission. Its outputs are simulation results
about simulated systems.

## Roadmap

- Square-root / UD-factorised filter forms for ill-conditioned problems.
- RTS smoother with NEES scoring over the smoothed estimate.
- Richer `Q` adaptation: per-axis and full-matrix scaling, and joint `Q`/`R`
  identification, since §6 of the validation shows a single scalar is not
  sufficient.
- Replace the bootstrap-ensemble confidence with conformal prediction intervals
  calibrated on a held-out split, since the current spread is measurably
  uninformative.
- Flicker (1/f) bias-instability term in the gyro model so datasheet figures
  can be entered directly.
- Correlated GNSS error model (first-order Gauss-Markov) and a J2 orbit term.
- An IMM estimator, which is the natural answer to manoeuvring targets that
  scalar `Q` adaptation handles badly.

## License

AGPL-3.0-only. Copyright © 2026 OPTIMA Organisation. See `LICENSE` for the
full text.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

## Citation

```bibtex
@software{navbench2026,
  title        = {NavBench: an attitude and navigation filter bench with
                  NEES/NIS consistency diagnostics},
  author       = {{OPTIMA Organisation}},
  year         = {2026},
  version      = {0.1.0},
  license      = {AGPL-3.0-only},
  note         = {Research-grade software; not flight-qualified.}
}
```

### Key references

- Bar-Shalom, Y., Li, X.-R. & Kirubarajan, T. (2001). *Estimation with
  Applications to Tracking and Navigation.* Wiley. (§5.2 filter recursions,
  §5.4 NEES/NIS consistency, §6.2-6.3 CWNA/DWNA models, §10.3 EKF, §11.7
  coordinated turn.)
- Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction
  Problems." *Trans. ASME — J. Basic Engineering* 82(D), 35-45.
- Julier, S. J. & Uhlmann, J. K. (1997). "A New Extension of the Kalman Filter
  to Nonlinear Systems." *Proc. SPIE 3068*, 182-193; and (2004) "Unscented
  Filtering and Nonlinear Estimation." *Proc. IEEE* 92(3), 401-422.
- Wan, E. A. & van der Merwe, R. (2000). "The Unscented Kalman Filter for
  Nonlinear Estimation." *IEEE AS-SPCC*, 153-158.
- Rauch, H. E., Tung, F. & Striebel, C. T. (1965). "Maximum Likelihood
  Estimates of Linear Dynamic Systems." *AIAA Journal* 3(8), 1445-1450.
  (Cited for the smoother named in the roadmap; not implemented in 0.1.0.)
- Lefferts, E. J., Markley, F. L. & Shuster, M. D. (1982). "Kalman Filtering
  for Spacecraft Attitude Estimation." *J. Guidance, Control, and Dynamics*
  5(5), 417-429.
- Markley, F. L. & Crassidis, J. L. (2014). *Fundamentals of Spacecraft
  Attitude Determination and Control.* Springer.
- Markley, F. L. (2003). "Attitude Error Representations for Kalman Filtering."
  *J. Guidance, Control, and Dynamics* 26(2), 311-317.
- Farrenkopf, R. L. (1978). "Analytic Steady-State Accuracy Solutions for Two
  Common Spacecraft Attitude Estimators." *J. Guidance and Control* 1(4),
  282-284.
- Shuster, M. D. & Oh, S. D. (1981). "Three-Axis Attitude Determination from
  Vector Observations." *J. Guidance and Control* 4(1), 70-77.
- Shuster, M. D. (1993). "A Survey of Attitude Representations."
  *J. Astronautical Sciences* 41(4), 439-517.
- Shepperd, S. W. (1978). "Quaternion from Rotation Matrix." *J. Guidance and
  Control* 1(3), 223-224.
- Wertz, J. R. (ed.) (1978). *Spacecraft Attitude Determination and Control.*
  Reidel.
- Mehra, R. K. (1970). "On the identification of variances and adaptive Kalman
  filtering." *IEEE Trans. Automatic Control* 15(2), 175-184; and (1972)
  "Approaches to adaptive filtering." *IEEE Trans. Automatic Control* 17(5),
  693-698.
- Mohamed, A. H. & Schwarz, K. P. (1999). "Adaptive Kalman filtering for
  INS/GPS." *Journal of Geodesy* 73, 193-203.
- Kalata, P. R. (1984). "The Tracking Index: A Generalized Parameter for α-β
  and α-β-γ Target Trackers." *IEEE Trans. Aerospace and Electronic Systems*
  AES-20(2), 174-182.
- Bierman, G. J. (1977). *Factorization Methods for Discrete Sequential
  Estimation.* Academic Press.
- Jazwinski, A. H. (1970). *Stochastic Processes and Filtering Theory.*
  Academic Press.
- Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*,
  4th ed. Microcosm.
- Petit, G. & Luzum, B. (eds.) (2010). *IERS Conventions (2010)*, IERS
  Technical Note 36.
- IEEE Std 952-2020. *IEEE Standard Specification Format Guide and Test
  Procedure for Single-Axis Interferometric Fiber Optic Gyros.*
