# Changelog

All notable changes to NavBench are recorded here.

## 0.1.0 — 2026-08-29

Initial release. Validation level 3, status TESTING.

### Truth generation
- `attitude_trajectory`: rigid-body attitude under an arbitrary applied torque,
  integrating Euler's equation together with the quaternion kinematics by
  classical RK4 with per-step renormalisation. Measured convergence ratios
  16.05 / 16.03 / 16.01 against RK4's ideal 16; torque-free relative energy
  drift 5.9e−13 over 300 s.
- `AttitudeTruth.interval_rate()`: the effective constant body rate over each
  sampling interval, which is what a rate-integrating gyro actually reports.
  Added after the endpoint-sample alternative was found to inject a
  deterministic 3.2e−05 rad/step attitude error and drive the MEKF's mean NEES
  to 1925 against dof 6.
- `orbit_trajectory` / `circular_orbit_state`: two-body Keplerian propagation
  (μ from IERS Conventions 2010). Energy drift 1.7e−12 and closure 2.3e−10 per
  revolution at 500 km, 800 km and GEO.
- `airborne_trajectory`: flat-Earth constant-velocity and coordinated-turn
  track, integrated analytically in the horizontal plane (radius error 7.1e−15
  relative over a full circle).

### Sensor models
- `GyroModel` with angle random walk and rate random walk in the IEEE Std
  952-2020 sense, plus optional scale-factor and misalignment errors, a
  vectorised whole-run sampler, and datasheet-unit converters
  (`arw_deg_per_sqrt_hour_to_si`, `rrw_deg_per_hour_1p5_to_si`). Measured Allan
  deviation matches the analytic two-term form to within 0.1 % over the
  ARW-dominated decade.
- `StarTrackerModel` (unit-vector and full-quaternion output, dropouts),
  `SunSensorModel` (field of view + eclipse), `AccelerometerModel` (specific
  force in body axes), `GpsModel` (position and optional velocity, dropouts).

### Estimators
- `KalmanFilter`: discrete linear KF with Joseph-form covariance update,
  optional control input, per-step `F`/`Q`/`H`/`R` overrides, NaN-tolerant
  batch `run()` for sensor dropouts, and per-step innovation / gain / NIS
  bookkeeping.
- `steady_state_riccati`: fixed-point solution of the filtering DARE, returning
  `P⁻_∞`, `P⁺_∞`, `K_∞` and the iteration count, with the increment-vs-error
  distinction documented for low-SNR problems.
- `ExtendedKalmanFilter` with analytic or central-difference Jacobians; reduces
  exactly to the linear KF on a linear-Gaussian system.
- `UnscentedKalmanFilter`, `MerweSigmaPoints`, `unscented_transform`: scaled
  unscented transform with configurable α/β/κ and Cholesky failure surfaced as
  `CovarianceCollapseError`.
- `MultiplicativeEKF`: 6-state attitude-error + gyro-bias-error filter over a
  unit-quaternion reference, with the exact discrete transition (matching
  `scipy.linalg.expm` to 2.9e−15 across θ ∈ [1e−12, 3]), Farrenkopf's `Q_d`,
  unit-vector and full-quaternion measurement modes, and a multiplicative reset
  that folds the estimated error into the reference exactly.

### Consistency diagnostics (the product's reason to exist)
- `nees`, `nis`, `chi2_bounds`, `consistency_test`, `ensemble_consistency`,
  `innovation_whiteness`, with an explicit `independent` flag so that
  single-run time averages are labelled indicative rather than presented as a
  valid chi-squared test.
- `score_run` / `compare_scores`: error and consistency reported side by side,
  never collapsed into one figure of merit; a divergence convention stated
  rather than assumed.

### AI element
- `MehraAdaptiveQ`: classical innovation-based adaptive estimation
  `Q̂ = K Ĉ Kᵀ` (Mehra 1970/1972; Mohamed & Schwarz 1999), with a documented
  scalar projection and mandatory clipping.
- `LearnedAdaptiveQ`: bootstrap ensemble of gradient-boosted trees over six
  scale-free innovation features, with an extrapolation flag and an
  ensemble-spread confidence output.
- `generate_adaptive_dataset`, `innovation_features`, `run_adaptive_kf`.
- **Benchmark result, reported as measured:** the learned tuner has the lowest
  held-out position RMSE (1.923 m vs 2.100 m fixed, 2.274 m Mehra; 44/60 paired
  wins, CI excluding zero) and is the only method whose scale estimate
  correlates with the truth (+0.59 vs Mehra's 0.00). The classical scheme has
  the ANEES closest to 2.0 (1.178 vs 4.494) but reaches it by saturating at its
  clip on 60/60 runs — a constant 64x inflation rather than adaptation. None of
  the three is actually consistent, and the learned confidence output is
  uninformative (correlation with error +0.22, wrong sign). Nothing was retuned
  to change this.

### CLI, examples, documentation
- `python -m navbench {riccati,bench,attitude,consistency,adaptive}` with
  `--json` on every subcommand and exit code 2 (no traceback) on invalid input.
- Five runnable examples, each saving a PNG into `screenshots/` with the Agg
  backend.
- `docs/REQUIREMENTS.md`: 24 numbered requirements with a verification matrix,
  an error-handling policy, and a record of the three defects the requirement
  set caught during the build.
- `validation/VALIDATION.md` plus seven executable validation scripts with
  committed raw output.
- `MODEL_CARD.md`, `DATASET_CARD.md`.

### Testing
- 715 tests: unit, input-validation, known-answer (hand-calculated, shown in
  test comments), edge case, integration, 21 Hypothesis property tests, 18
  pinned-seed regression tests, 9 performance guards and 30 failure-mode tests
  (covariance collapse, sensor dropout, gross mis-specification, divergence).
- Ruff-clean at line length 100 under `E,F,W,I,UP,B,SIM`.

### Defects found and fixed during this build
- `axis_angle_from_quat` discarded the rotation axis below ~2e−12 rad because
  of a `1e-12` cut-off; threshold lowered to the smallest normal double.
- `attitude_state_transition` lost up to 14 significant digits to cancellation
  for 1e−8 < θ < 1e−2; rewritten with a series/closed-form crossover at 1e−2.
- The MEKF was being fed an endpoint rate sample rather than the interval
  average, injecting a deterministic bias three times the gyro noise; fixed by
  `AttitudeTruth.interval_rate()`.
