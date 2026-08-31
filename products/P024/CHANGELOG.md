# Changelog

All notable changes to DetumbleSim are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-31

Initial release. Status: TESTING.

### Added

- **Magnetic field model** (`detumblesim.magfield`): tilted centred dipole
  from the IGRF-14 degree-1 Gauss coefficients at main-field epoch 2025.0,
  with ECEF and ECI evaluation, a uniform Earth-rotation model, geomagnetic
  pole and dipole-tilt derivation, and vectorised evaluation over position
  arrays.
- **Orbit** (`detumblesim.orbit`): unperturbed circular Keplerian orbit with
  closed-form inertial position, velocity and orbit normal (Vallado 2013).
- **Attitude dynamics** (`detumblesim.attitude`): scalar-first quaternion
  kinematics, attitude matrix and Shepperd inverse, and Euler's rigid-body
  equation (Markley & Crassidis 2014; Wertz 1978).
- **Hardware models** (`detumblesim.spacecraft`): inertia construction with a
  triangle-inequality check, and a three-axis magnetorquer with a per-axis
  dipole box.
- **Control laws** (`detumblesim.control`), implemented and validated first:
  `BDotController` (Stickler & Alfriend 1976), a field-normalised B-dot
  variant, `CrossProductController`, and the closed-form ideal B-dot torque
  used by the property tests.
- **Simulator** (`detumblesim.simulate`): RK4 on `(q, omega)` with a
  zero-order-hold dipole, a flight-realistic backward-difference `dB/dt` from
  simulated magnetometer samples with optional Gaussian noise, and a
  precomputed vectorised inertial field history.
- **First-order analytic model** (`detumblesim.analytic`): orbit-averaged
  field moments, the damping matrix `D = k(<|B|²>I − <BBᵀ>)`, modal time
  constants, the `t = tau ln(w0/wf)` detumble time in isotropic / slowest /
  fastest modes, the box-vertex maximum torque, and a dipole-limit lower bound
  on detumble time.
- **Controllability analysis** (`detumblesim.controllability`): the
  instantaneous rank-2 projector, orbit-averaged geometry factors summing to
  exactly 2, the weakest inertial direction, the anisotropy ratio, the
  time-averaged uncontrollable fraction, and residual-rate resolution along a
  fixed inertial direction.
- **Gain policies** (`detumblesim.policies`): `FixedGainPolicy`,
  `SizedGainPolicy` (observable rate-proxy sizing rule),
  `PowerLawGainPolicy` (three-coefficient log-linear fit), and
  `ScheduledGainPolicy` (learned).
- **Learned gain scheduler** (`detumblesim.scheduler`, `detumblesim.features`):
  a `RandomForestRegressor` over eight observable features predicting
  `log10(k_oracle / k_fixed)`, with an ensemble-spread confidence output, a
  confidence-weighted shrinkage toward the classical baseline, and a hard
  ±1 dex safety clamp.
- **Scenario generation and scoring** (`detumblesim.scenarios`,
  `detumblesim.evaluate`): seeded synthetic scenarios, the combined
  time-plus-coil-energy cost, the per-scenario gain oracle, the training-row
  harvester, and the power-law fit.
- **Confidence intervals** (`detumblesim.metrics`): Student-t intervals on a
  mean and on paired per-scenario differences.
- **CLI**: `python -m detumblesim {field,detumble,sweep,controllability}`.
- **Validation campaign**: `validation/field_model_check.py`,
  `momentum_monotonicity.py`, `gain_scaling.py`, `controllability_gap.py` and
  `learned_vs_fixed_ci.py`, with `validation/VALIDATION.md` and committed raw
  stdout captures.
- **Examples**: `detumble_curve.py`, `gain_sweep.py`,
  `controllability_gap.py` and `learned_vs_fixed.py`, each writing a PNG to
  `screenshots/`.
- **Dataset generator**: `data/generate_dataset.py`, writing the training
  scenario table, the feature/label matrix and a SHA-256 manifest.
- **Documentation**: `README.md`, `MODEL_CARD.md`, `DATASET_CARD.md`.
- **Tests**: 305 tests — unit, input-validation, hand-calculated known-answer,
  edge-case, Hypothesis property tests, an end-to-end integration test, CLI
  subprocess tests, and pinned-value benchmark/regression tests with a
  wall-clock budget check.

### Known issues

- **The field-model check V1-A2 FAILED.** The centred dipole is wrong by up to
  71.86 % against IGRF-14 in the South Atlantic Anomaly, against a
  pre-registered 25 % tolerance. Reported as failed; no coefficient was
  adjusted and no tolerance widened. Median error 8.967 % passes V1-A1.
- **The product specification's angular-momentum monotonicity claim is false
  for asymmetric inertia**, and is reported as falsified with a closed-form
  counterexample (`validation/VALIDATION.md`, V2-B3). Kinetic energy *is*
  monotone for any inertia, and that is what this package states and tests.
- **The learned gain scheduler does not beat the three-coefficient power-law
  regression** at any energy weight tested, and loses to the plain fixed gain
  when coil energy is weighted heavily. 100 % of its feature importance is on
  the two static vehicle parameters, so it is a per-vehicle lookup rather than
  a schedule.
- The scheduler's confidence output is an uncalibrated ensemble-spread
  heuristic with no coverage measurement.
- The `1/k` detumble-time scaling degrades to a fitted slope of −0.577 once
  detumbles take less than about one orbit.
- The sized-gain rule failed to detumble 2 of 40 held-out scenarios inside the
  simulated span.
- No environmental disturbance torque (gravity gradient, aerodynamic, solar
  pressure, residual dipole) and no orbit perturbation (J2, drag) is modelled.
- B-dot has a rate floor near the orbital rate (measured 0.156 deg/s against
  0.063 deg/s at 500 km); rates below roughly 3× the orbital rate need a rate
  estimate and the cross-product law instead.
