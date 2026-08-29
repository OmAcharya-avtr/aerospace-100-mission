# NavBench — Requirements and Verification Matrix

Version 0.1.0 · Validation Level 3 (research-grade, at v0.1 depth) · Status: **TESTING**

Requirement IDs are stable. `F` = functional, `N` = non-functional.
Verification methods: **T** = automated test, **A** = analysis / hand
calculation, **D** = demonstration (runnable script), **I** = inspection.

Evidence paths are relative to the product root. Every validation figure quoted
below was produced by running the named script in the build session of
2026-08-29; the raw stdout is committed beside it.

---

## 1. Functional requirements

| ID | Requirement | Rationale | Method | Evidence |
|----|-------------|-----------|--------|----------|
| **R01** | The system shall generate a rigid-body attitude truth trajectory by integrating Euler's equation `J ω̇ = τ − ω × Jω` together with the quaternion kinematics `q̇ = ½ q ⊗ [0, ω]` under a user-supplied torque, using RK4 with per-step renormalisation. | Core capability; every filter test needs a truth. | T, A | `tests/test_truth.py::TestAttitudeTrajectory` (14 tests); `validation/v4_mekf_quaternion.py` PART B — RK4 order ratios **16.05 / 16.03 / 16.01** against the ideal 16; torque-free relative energy drift **5.948e−13** over 300 s |
| **R02** | The system shall generate a position truth track: a two-body Keplerian orbit in an inertial frame (μ = 3.986004418e14 m³/s², IERS 2010) and a flat-Earth coordinated-turn airborne track. | Navigation half of the bench. | T, A | `tests/test_truth.py::TestOrbitTrajectory`, `::TestAirborneTrajectory` (18 tests); `validation/v5_sensor_and_truth_models.py` PART C — energy drift **1.708e−12** and closure **2.325e−10** per revolution at three altitudes; PART D — turn radius error **7.094e−15** relative |
| **R03** | The system shall model a rate gyro with angle random walk `σ_v` and rate random walk `σ_u` in the IEEE Std 952-2020 sense, with datasheet-unit converters, plus optional scale-factor and misalignment errors. | The dominant error source in attitude estimation. | T, A, D | `tests/test_sensors.py::TestGyroModel` (16 tests), `::TestUnitConversions` (6); `validation/v5` PART A — measured Allan deviation within **0.9988–0.9190** of the analytic form over τ = 0.02–50 s, log-log slopes **−0.4983** (ARW) and **+0.4417** (RRW); `examples/gyro_allan_deviation.py` |
| **R04** | The system shall model a star tracker (unit-vector and full-quaternion output), a sun sensor with field-of-view and eclipse handling, a triad accelerometer measuring specific force, and a GPS-like position/velocity fix — each with documented noise and a dropout mechanism. | A bench needs a sensor suite, not one sensor. | T, A | `tests/test_sensors.py::TestStarTracker`, `::TestSunSensor`, `::TestAccelerometer`, `::TestGps` (33 tests); `validation/v5` PART B — star-tracker variance ratio **1.0013**, sun-sensor valid fraction **0.7006** vs 0.700, GPS σ [2.4981, 2.4972, 2.4879] m vs 2.5 m spec |
| **R05** | The system shall provide a discrete linear Kalman filter with a Joseph-form covariance update, optional control input, per-step `F`/`Q`/`H`/`R` overrides, and per-step innovation/gain/NIS bookkeeping. | The reference estimator. | T, A | `tests/test_kf.py` (65 tests); `validation/v1_riccati_steady_state.py` §1d — a running filter reaches `P⁺_∞` to **4.622e−16** relative in 600 steps |
| **R06** | The system shall provide an EKF taking analytic Jacobians, with a documented central-difference numerical fallback, and shall reduce exactly to the linear KF on a linear-Gaussian system. | Standard nonlinear estimator; exact reduction is the correctness anchor. | T, A | `tests/test_ekf.py::TestEkfReducesToKf` — max relative deviation **0.000e+00** vs the KF; `validation/v3_ukf_vs_ekf.py` PART C |
| **R07** | The system shall provide a UKF with scaled symmetric sigma points (configurable α, β, κ), and shall reduce to the linear KF on a linear-Gaussian system to a tolerance that accounts for the `1/α²` round-off amplification. | Second-order alternative without Jacobians. | T, A | `tests/test_ukf.py::TestUkfReducesToKf` (α ∈ {1, 0.5, 0.1}); `validation/v3` PART C — UKF vs KF max relative `Δx` **3.558e−16**, `ΔP` **4.824e−15** |
| **R08** | The system shall provide a multiplicative EKF for attitude carrying a unit-quaternion reference plus a 6-element error state (attitude error + gyro bias error), with the exact discrete transition `Φ = exp(F_c Δt)` and Farrenkopf's `Q_d`, and shall fold the estimated error into the reference multiplicatively at every update. | Quaternion-state filters need the multiplicative formulation; a 4-vector state has a singular covariance. | T, A | `tests/test_mekf.py` (68 tests); `Φ` matches `scipy.linalg.expm` to **2.9e−15** across θ ∈ [1e−12, 3]; `validation/v4` PART D — reference after reset equals `q_before ⊗ δq(â)` to **0.000e+00** |
| **R09** | The system shall compute NEES and NIS and test them against two-sided chi-squared acceptance regions, for a single sample and for an ensemble of independent runs, and shall report which side of the region a failure falls on. | The reason this product exists; most tools omit it. | T, A | `tests/test_consistency.py` (58 tests); `validation/v2_nees_nis_consistency.py` — correct filter ANEES **2.0223** in [1.5262, 2.5369], `Q`/25 gives **26.6001**, `Q`×25 gives **1.1354** |
| **R10** | The system shall provide an innovation whiteness test with the ±1.96/√N acceptance band. | Consistency has a temporal dimension the mean test misses. | T, A | `tests/test_consistency.py::TestWhiteness` (12 tests); `validation/v2` PART C — correct filter max \|ρ\| **0.0984** vs band 0.1503; `Q`/25 gives **0.4113** |
| **R11** | The system shall provide a classical innovation-based adaptive `Q` estimator (`Q̂ = K Ĉ Kᵀ`, Mehra 1970/1972; Mohamed & Schwarz 1999) with a documented scalar projection and mandatory clipping. | The AI element must be benchmarked against a real classical method, not a strawman. | T, A | `tests/test_adaptive.py::TestMehraAdaptiveQ` (13 tests); `validation/v6_adaptive_q_benchmark.py` — Mehra achieves the best held-out ANEES, **1.1778** |
| **R12** | The system shall provide a learned adaptive `Q`-scale tuner trained on innovation statistics, exposing an ensemble-spread confidence output and an extrapolation flag, benchmarked on held-out runs against both R11 and a fixed hand-tuned `Q`. | AI product requirement (mission §11); uncertainty output mandatory. | T, D | `tests/test_adaptive.py::TestLearnedAdaptiveQ`, `::TestRunAdaptiveKf` (33 tests); `MODEL_CARD.md`; `validation/v6` — learned RMSE **1.92299 m** vs fixed 2.09982 m, winning 44/60 paired runs |
| **R13** | The system shall report error metrics and consistency metrics side by side and shall never collapse them into a single figure of merit; a filter that wins on RMSE and fails NEES shall be reported as failing NEES. | Engineering honesty; the split result in `validation/v6` is the case in point. | T, I, D | `tests/test_bench.py` (18 tests); `src/navbench/bench.py::compare_scores`; `validation/v6` verdict block names two different winners |
| **R14** | The system shall provide a CLI `python -m navbench` with `riccati`, `bench`, `attitude`, `consistency` and `adaptive` subcommands, `--json` output on each, exit code 2 and a one-line message (no traceback) on invalid input. | Usability and integration. | T, D | `tests/test_cli.py` (23 tests) including `::TestModuleInvocation::test_python_dash_m_bad_input_exit_2_no_traceback` |
| **R15** | Each runnable example shall produce a PNG in `screenshots/` using the Agg backend. | Reviewable evidence that the package runs end to end. | D, I | `examples/estimator_bench.py`, `nees_nis_consistency.py`, `mekf_attitude.py`, `adaptive_q_tuning.py`, `gyro_allan_deviation.py` → 5 PNGs in `screenshots/` |

## 2. Non-functional requirements

| ID | Requirement | Rationale | Method | Evidence |
|----|-------------|-----------|--------|----------|
| **N16** | Filter throughput shall exceed 1000 steps/s for the linear KF and the EKF and 500 steps/s for the UKF on a 2-core machine. | Makes Monte Carlo studies practical. | T, D | `validation/v7_performance.py` — KF **8035**, EKF **8127**, UKF **4003**, MEKF **6814** steps/s; bounds enforced by `tests/test_performance.py` |
| **N17** | Dataset generation plus model training shall complete within the 180 s compute budget of the build guide, single-threaded, deterministically from committed seeds. | Guide compute budget; reproducibility. | T, D | `validation/v7` — 9.92 s dataset + 2.08 s fit = **11.99 s**; `tests/test_performance.py::test_ml_pipeline_within_compute_budget` |
| **N18** | Numerical behaviour shall be pinned by regression tests against recorded seeded outputs so that silent drift fails the suite. | Level 3 requirement. | T | `tests/test_regression.py` — 18 tests pinning the Riccati solution, a 100-step KF run, EKF/UKF radar states, the attitude truth, the MEKF final state, gyro samples and the ML dataset |
| **N19** | Every invalid input shall raise `ValueError`/`TypeError` naming the offending quantity; loss of covariance positive definiteness shall raise `CovarianceCollapseError` rather than being silently regularised. | Error-handling policy (§3). | T | `tests/test_failure_modes.py` (30 tests); ~150 validation-error assertions across the suite |
| **N20** | Failure modes shall be tested explicitly: covariance collapse, sensor dropout, gross model mis-specification and divergence. | Level 3 requirement. | T | `tests/test_failure_modes.py::TestCovarianceCollapse`, `::TestSensorDropout`, `::TestGrossMisspecification`, `::TestDivergence`, `::TestNumericalGuards` |
| **N21** | Algebraic identities shall be checked by property-based tests over generated inputs, not only by fixed examples. | Catches defects fixed examples miss — it caught two in this build (§4). | T | `tests/test_properties.py` — 21 properties over Hypothesis-generated quaternions, covariances, sigma-point parameters and rates |
| **N22** | The source shall be ruff-clean at line length 100 under `E,F,W,I,UP,B,SIM`, with type hints and unit-carrying docstrings on every public API. | Maintainability. | I | `ruff check src/ tests/ examples/ validation/` → **All checks passed!** |
| **N23** | Every equation in code and documentation shall carry source, units, assumptions and validity range; every reported number shall come from an executed script. | Binding engineering-honesty rule. | I | Module docstrings in `src/navbench/*.py`; `validation/VALIDATION.md` with committed raw output for all seven scripts |
| **N24** | Uncertainty analysis shall accompany every claim: chi-squared acceptance regions on consistency statistics, confidence intervals on paired comparisons, and a stated estimator uncertainty where a statistic is intrinsically noisy. | Level 3 requirement. | A, T | `validation/v2` (chi-squared bands throughout), `validation/v6` (paired 95 % CI, e.g. **−0.17683 ± 0.09361 m**), `validation/v5` PART A (Allan-estimator relative uncertainty used as the tolerance) |

## 3. Error-handling policy

1. **Validate at construction.** Every model and filter validates shapes,
   finiteness, symmetry and definiteness in `__init__`/`__post_init__`. Invalid
   values raise `ValueError`; wrong callables raise `TypeError`. Messages name
   the parameter and quote the offending value.
2. **Covariance definiteness is a finding, not a nuisance.** A failed Cholesky
   factorisation of `P` or `S` raises `CovarianceCollapseError` (a
   `RuntimeError`) carrying the minimum eigenvalue. Nothing is silently jittered
   or clipped back to positive definite.
3. **Joseph form unconditionally.** The short `(I − KH)P` update is never used
   in the library; it is only exercised inside tests as the *contrast* case.
4. **Missing measurements are data, not errors.** A NaN row means "no
   measurement at this step": the filter predicts and skips the update, records
   `updated = False` and `nis = NaN`. A NaN passed to a single `update()` call
   is an error and raises.
5. **Singular geometry raises.** `radar_measurement`/`radar_jacobian` raise
   below 1e−9 m range rather than dividing by a vanishing number.
6. **The CLI never shows a traceback for user error.** `main()` catches
   `ValueError`, `TypeError` and `RuntimeError`, writes `error: <message>` to
   stderr and returns exit code 2. A traceback indicates a genuine defect.
7. **Non-convergence is reported.** `steady_state_riccati` raises when the
   fixed-point iteration does not converge, naming the last increment and the
   likely cause (an undetectable pair).
8. **The learned model refuses to guess.** `LearnedAdaptiveQ.predict` raises
   `RuntimeError` when unfitted and flags `extrapolating=True` outside the
   training feature box rather than returning a silent extrapolation.

## 4. Defects found by this requirement set during the build

Recorded because a verification matrix that never caught anything is not
evidence of anything.

| # | Found by | Defect | Resolution |
|---|---|---|---|
| 1 | `validation/v4` PART A | `axis_angle_from_quat` used a `1e-12` cut-off on `\|q_v\|` and returned the axis `[1, 0, 0]` for any rotation below ~2e−12 rad, discarding the direction (≈200 % relative error). | Threshold lowered to the smallest normal double; tolerance restated as relative. Pinned by `tests/test_attitude.py::test_round_trip_relative`. |
| 2 | `tests/test_properties.py::test_transition_composes_over_two_half_steps` | `attitude_state_transition` switched from series to closed form at θ = 1e−8, so `(1 − cos θ)/\|ω\|²` lost ~14 digits to cancellation in the range 1e−8 < θ < 1e−2; error 5e−10 absolute at θ = 4.3e−7. | Rewritten around three scalar coefficients with a series/closed-form crossover at θ = 1e−2. Now matches `scipy.linalg.expm` to **2.9e−15** across θ ∈ [1e−12, 3]. |
| 3 | `validation/v4` PART E (initial run) | The CLI and examples fed the MEKF an *endpoint* sample of the true rate, injecting a deterministic 3.2e−05 rad/step attitude error — three times the gyro noise. Attitude RMS 1.26e−04 rad, mean NEES **1925** against dof 6. | `AttitudeTruth.interval_rate()` added, returning the effective constant rate over each interval, which is what a rate-integrating gyro reports. After: 4.2e−05 rad, ANEES **6.147**. |

## 5. Requirement coverage summary

| ID | R01 | R02 | R03 | R04 | R05 | R06 | R07 | R08 | R09 | R10 | R11 | R12 | R13 | R14 | R15 | N16 | N17 | N18 | N19 | N20 | N21 | N22 | N23 | N24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Verified | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |

All 24 requirements are verified by the evidence above: **715 automated tests**,
**7 validation scripts** with committed raw output, and **5 runnable examples**.
No requirement is waived or deferred.

**What "verified" means here.** Internal consistency: agreement with cited
closed-form results, self-consistency of the Monte Carlo, and reproducibility
from committed seeds. **No requirement has been verified against measured
hardware data.** Every trajectory and every sensor sample in this package is
synthetic. See `validation/VALIDATION.md` §8 and README Limitations.
