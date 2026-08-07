# Changelog

All notable changes to TrackBench are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-06

Initial MVP release of product P002 (flagship class, validation level 3).
Status: **TESTING**.

### Added

**`trackbench.scan` — acquisition**
- `GaussianUncertainty`: isotropic 2-D Gaussian uncertainty region with
  Rayleigh containment radius and quantile round-trip.
- `spiral_scan`: Archimedean spiral with radial pitch tied to beam overlap
  (`s = 2 R_beam (1 − overlap)`), constant arc-length dwell placement.
- `raster_scan`: serpentine raster over the bounding square.
- `coverage_fraction`: Monte Carlo covered probability mass via k-d tree.
- `simulate_acquisition`: dwell-by-dwell acquisition with per-dwell Bernoulli
  detection and a multi-pass budget.
- `expected_acquisition_time_spiral`: analytic uniform-coverage expectation
  (internal derivation, validated against Monte Carlo to −0.65 %).

**`trackbench.dynamics` — plant, disturbance, sensor**
- `GimbalAxis`: second-order axis `J θ̈ + b θ̇ = τ` with torque, rate and
  optional acceleration limits; RK4 with zero-order-hold torque.
- `TwoAxisGimbal`: two decoupled axes with a config constructor.
- `JitterPSD`: parametric flat-then-roll-off PSD with a quadrature variance.
- `synthesize_jitter`: random-phase spectral factorisation of a target PSD.
- `welch_psd`: shared Welch estimator for validation and tests.
- `AngleSensor`: NEA noise, optional quantisation, optional dropout with a
  validity flag.

**`trackbench.control` — controllers and benchmark harness**
- `PIDController`: derivative-on-measurement, conditional-integration
  anti-windup, output clipping.
- `pid_gains_from_bandwidth`, `lqr_weights_from_bandwidth`: documented tuning
  rules from target natural frequency and damping.
- `LQRController`: continuous (CARE) or discrete (DARE) infinite-horizon LQR
  via SciPy, with exposed Riccati solution and closed-loop poles.
- `zoh_discretize`: matrix-exponential zero-order-hold discretisation.
- `step_response`, `disturbance_rejection_rms`, `bandwidth_estimate`,
  `benchmark_controllers`: measured metrics, no asserted numbers.

**`trackbench.reacq` — reacquisition (AI)**
- `ReacqConfig`, `ReacqEnv`: episodic MDP with the specified state
  (time since loss, last-known offset, uncertainty growth) plus a searched-radius
  feature for Markov consistency; three actions (LOCAL / FULL / RING).
- `AlwaysFullPolicy`, `AlwaysLocalPolicy`: scripted baselines, implemented and
  benchmarked first.
- `train_q_learning`: seeded tabular Q-learning in numpy.
- `QLearningPolicy`: greedy policy with a margin × support confidence output
  and a baseline fallback on unvisited states.
- `evaluate_policy`, `compare_policies`: Monte Carlo evaluation with common
  random numbers and 95 % confidence intervals.

**`trackbench.sim` — end-to-end simulator**
- `Scenario`: validated configuration dataclass; `load_scenario` rejects
  unknown YAML keys.
- `run_episode`: acquire → track → lose lock → reacquire, with metrics and
  optional time series.
- `run_monte_carlo`, `sim_steps_per_second`.

**CLI** — `python -m trackbench run | benchmark | reacq` (argparse, `--json`,
`--version`).

**Examples and screenshots**
- `examples/ex01_scan_patterns.py`, `ex02_tracking_error.py`,
  `ex03_reacq_comparison.py`, each producing a committed PNG in
  `screenshots/`.
- `examples/scenario_leo_downlink.yaml`, `examples/scenario_high_jitter.yaml`.

**Validation** — seven rerunnable scripts with committed raw output and
`validation/VALIDATION.md`, covering spiral coverage, acquisition time,
control step response, jitter PSD, the reacquisition benchmark, performance
and the regression baseline; includes an uncertainty analysis.

**Documentation** — `README.md` (with an ASCII architecture diagram),
`docs/REQUIREMENTS.md` (23 requirements plus a verification matrix),
`MODEL_CARD.md`, `DATASET_CARD.md`, `LICENSE` (AGPL-3.0-only, © 2026 OPTIMA
Organisation).

**Tests** — 295 tests across 10 modules: unit, input validation,
known-answer (hand-calculated), property-based (Hypothesis), integration,
configuration, regression (28 pinned seeded values), performance,
failure-mode and reproducibility.

### Findings recorded during validation

- Spiral coverage at `overlap = 0` (tangent tracks) falls **1.04 % below** the
  design containment because of curvature gaps between adjacent turns. The
  one-dimensional "no radial gap" argument is insufficient for a curved
  spiral. Recorded rather than tuned away; `overlap ≥ 0.10` removes the
  shortfall.
- The `p_pass` argument of `expected_acquisition_time_spiral` is a
  **per-crossing**, not per-dwell, probability. Supplying the per-dwell value
  overestimates acquisition time by **38 %** for the reference scenario. The
  docstring was corrected to state this and to point at
  `validation/v2_acquisition_time.py`.

### Known limitations at 0.1.0

Coarse pointing stage only (no fine-steering mirror); decoupled axes; no
structural modes; no optical/atmospheric physics; illustrative parameter
values not tied to any mission; assumed loss-of-lock statistics; tabular RL
only (PyTorch unavailable in the build environment); uncalibrated confidence
output; no policy serialisation; single-threaded. See README "Limitations"
for the complete list.

[0.1.0]: https://github.com/OPTIMA-Organisation/trackbench/releases/tag/v0.1.0
