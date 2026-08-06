# TrackForge — Requirements and Verification Matrix

**Product:** P002 TrackForge · **Version:** 0.1.0 · **Validation level:** 3 (Professional, at v0.1 MVP depth)
**Status:** TESTING

Requirements are written for the v0.1 MVP scope: a research-grade PAT
simulation suite. They are *simulation* requirements — they constrain what
the software must compute and how it must be verified, not the performance of
any physical pointing system. Nothing here is a flight requirement.

Verification methods: **T** = automated test, **A** = analysis/analytic
comparison in `validation/`, **D** = demonstration (runnable example/CLI),
**I** = inspection (docstring/document review).

---

## 1. Functional requirements

| ID | Requirement | Rationale | Method | Evidence |
|----|-------------|-----------|--------|----------|
| **R-01** | The suite shall generate Archimedean-spiral acquisition scan patterns whose radial pitch equals `s = 2·R_beam·(1 − overlap)` for any `overlap ∈ [0, 1)` and whose outer radius equals the Rayleigh containment radius of the configured 2-D Gaussian uncertainty region. | Track spacing tied to beam-overlap is the standard spiral-scan design rule. | T, A | `tests/test_scan.py::test_spiral_radial_pitch_equals_track_spacing`, `::test_spiral_reaches_containment_radius`; `validation/v1_spiral_coverage.txt` |
| **R-02** | The suite shall generate serpentine raster scan patterns over the square bounding the containment disc, with row spacing equal to the same track spacing. | Raster is the reference alternative pattern; the cost/coverage trade must be measurable. | T, D | `tests/test_scan.py::test_raster_rows_spaced_by_track_spacing`, `::test_raster_is_serpentine`; `screenshots/ex01_scan_patterns.png` |
| **R-03** | For `overlap ≥ 0.10`, the covered probability mass of a spiral designed for containment *C* shall be ≥ *C* − 0.001 (Monte Carlo, ≥ 2·10⁵ samples). | Coverage is the primary correctness property of an acquisition scan. | A, T | `validation/v1_spiral_coverage.txt` (min 0.99485 vs 0.9940 criterion); `tests/test_scan.py::test_spiral_coverage_meets_containment` |
| **R-04** | The suite shall report single-pass scan time, dwell count and mean along-track scan speed for any generated pattern. | Acquisition-time budgeting. | T | `tests/test_scan.py::test_scan_time_and_speed_properties` |
| **R-05** | The suite shall simulate dwell-by-dwell acquisition with a per-dwell Bernoulli detection probability and shall return `None` when the target is never detected within the configured pass budget. | Detection is probabilistic; failure must be reported, not hidden. | T | `tests/test_scan.py::test_acquisition_returns_none_for_unreachable_target`, `tests/test_failure_modes.py::test_very_low_dwell_detection_probability_can_exhaust_passes` |
| **R-06** | The suite shall provide an analytic uniform-coverage estimate of expected acquisition time, and that estimate shall agree with Monte Carlo to better than 5 % relative for the reference scenario when the per-crossing detection probability is used. | Analytic model must be usable for first-order sizing. | A | `validation/v2_acquisition_time.txt` (−0.65 % and −0.56 %) |
| **R-07** | The suite shall model each gimbal axis as `J·θ̈ + b·θ̇ = τ` with documented torque, rate and (optional) acceleration limits, integrated by fixed-step RK4 with zero-order-hold torque. | The pointing plant must be a stated, checkable model. | T, I | `tests/test_dynamics.py::test_undamped_constant_torque_matches_kinematics`, `::test_damped_step_matches_analytic_first_order_rate`; `src/trackforge/dynamics.py` module docstring |
| **R-08** | The suite shall synthesise platform-jitter time series from a target one-sided PSD such that the Welch-estimated PSD matches the target to within 10 % (band medians) and the sample variance matches ∫S(f)df to within 10 %. | Disturbance realism is the basis of every tracking-error number. | A, T | `validation/v4_jitter_psd.txt` (worst band median 1.66 %, worst variance 0.081 %); `tests/test_dynamics.py::test_synthesized_jitter_psd_matches_target` |
| **R-09** | The suite shall model the angle sensor with a configurable noise-equivalent angle, optional quantisation and optional dropout, and shall flag dropped measurements as invalid. | Sensor limits dominate fine-tracking error. | T | `tests/test_dynamics.py::test_sensor_noise_std_matches_nea`, `::test_sensor_dropout_holds_last_value_and_flags_invalid` |
| **R-10** | The suite shall provide a PID controller with derivative-on-measurement and conditional-integration anti-windup, whose integrator does not grow while the output is saturated in the direction of the error. | Anti-windup is mandatory for a rate/torque-limited gimbal. | T | `tests/test_control.py::test_pid_anti_windup_freezes_integral_in_saturation`, `tests/test_failure_modes.py::test_anti_windup_limits_overshoot_under_saturation` |
| **R-11** | The suite shall provide an infinite-horizon LQR controller solving the continuous- or discrete-time algebraic Riccati equation via SciPy, with a documented linearisation statement. | Optimal-control reference against PID. | T, A | `tests/test_control.py::test_lqr_gain_solves_riccati_equation`, `::test_lqr_poles_match_butterworth_design`; `validation/v3_control_step_response.txt` section C |
| **R-12** | Closed-loop PD step responses shall reproduce the analytic canonical second-order overshoot, peak time and 10–90 % rise time to within 3 % relative (and 0.005 absolute in overshoot). | Hand-checkable correctness of the control/plant integration. | A, T | `validation/v3_control_step_response.txt` section A (worst 1.43 %, worst ΔMp 0.0006); `tests/test_control.py::test_pd_step_response_matches_analytic_overshoot_and_peak_time` |
| **R-13** | The suite shall provide a controller benchmark harness reporting rise time, overshoot, settling time, disturbance-rejection RMS and −3 dB bandwidth from actual simulation runs. | Comparability of controllers must be reproducible, not asserted. | T, D | `tests/test_control.py::test_benchmark_controllers_returns_expected_rows`; `python -m trackforge benchmark` (`validation/cli_benchmark.txt`) |
| **R-14** | The suite shall provide a reacquisition environment with the state (time since loss, last-known offset, uncertainty growth, searched radius) and three discrete actions (local restart, full-cone restart, expanding ring). | Reacquisition strategy selection is the AI problem being posed. | T, I | `tests/test_reacq.py::test_env_action_plan_durations_follow_area_model`, `::test_env_ring_starts_at_searched_radius` |
| **R-15** | Two scripted baseline policies (always-full, always-local) shall be implemented and benchmarked **before** and against any learned policy, on identical seeded episodes (common random numbers). | Mission rule: classical baseline first. | T, A | `validation/v5_reacq_benchmark.txt` ("BASELINES FIRST" section); `tests/test_reacq.py::test_evaluation_uses_common_random_numbers` |
| **R-16** | The learned reacquisition policy shall expose a confidence output alongside its action, and shall fall back to a baseline action on states never visited during training. | Mission rule: no bare point estimates from an ML component. | T, A | `tests/test_reacq.py::test_policy_confidence_bounds_and_fallback`, `::test_policy_confidence_rises_with_support`; `validation/v5_reacq_benchmark.txt` confidence section |
| **R-17** | The suite shall run an end-to-end episode chaining acquisition, tracking, loss of lock and reacquisition, and shall log metrics for each phase. | This is the product's headline capability. | T, D | `tests/test_integration.py::test_episode_completes_all_four_phases`; `python -m trackforge run` (`validation/cli_run.txt`) |
| **R-18** | Scenarios shall be loadable from YAML; unknown keys shall raise an error rather than being silently ignored, and out-of-range values shall raise `ValueError`/`TypeError` with an actionable message. | Silent config typos are a classic source of wrong results. | T | `tests/test_sim_config.py::test_load_scenario_rejects_unknown_keys`, `::test_scenario_field_validation` |

## 2. Non-functional requirements

| ID | Requirement | Rationale | Method | Evidence |
|----|-------------|-----------|--------|----------|
| **R-19** | Every seeded computation (scan sampling, jitter synthesis, episode, training, policy evaluation) shall be bitwise reproducible from its seed. | Level-3 evidence must be re-derivable. | T, A | `tests/test_reproducibility.py` (12 tests); `validation/v5_reacq_benchmark.txt` reproducibility section |
| **R-20** | Q-learning training and any single Monte Carlo benchmark shall complete in under 180 s on a 2-core machine; closed-loop simulation throughput shall exceed 5 000 steps/s. | Mission compute budget. | A, T | `validation/v6_performance.txt` (training 2.97 s; MC 0.73 s; 36 284 steps/s); `tests/test_performance.py` |
| **R-21** | Pinned regression values shall be committed and checked on every test run, with a rerunnable script that regenerates them. | Detect silent numerical drift. | T, A | `tests/test_regression.py` (28 pinned values); `validation/v7_regression_baseline.txt` |
| **R-22** | The suite shall degrade in a defined way — never raising an unhandled exception and never reporting nominal performance — under actuator saturation, sensor dropout, coarse quantisation, extreme noise and reacquisition timeout. | Failure modes must be observable. | T | `tests/test_failure_modes.py` (19 tests) |
| **R-23** | All public functions shall carry type hints and docstrings stating units, and the source shall pass `ruff check src/ tests/` with line length 100. | Maintainability and reviewability. | I, T | `ruff check src/ tests/` → "All checks passed!"; module docstrings |

---

## 3. Verification summary

| Method | Requirements covered |
|--------|----------------------|
| Automated test (T) | R-01…R-05, R-07…R-23 |
| Analysis (A) | R-01, R-03, R-06, R-08, R-11, R-12, R-15, R-16, R-19, R-20, R-21 |
| Demonstration (D) | R-02, R-13, R-17 |
| Inspection (I) | R-07, R-14, R-23 |

All 23 requirements are verified. Test suite: **295 tests, 295 passed,
0 failed, 0 skipped** (`python -m pytest tests/ -q`, wall time ≈ 70 s).

## 4. Requirements explicitly NOT met in v0.1 (out of scope)

These are recorded so the matrix is not read as broader than it is:

- No fine-steering-mirror (two-stage coarse/fine) loop; only the coarse gimbal stage is modelled.
- No cross-axis gimbal kinematics, gimbal lock, or structural flexible modes.
- No optical link budget, detector physics, or atmospheric propagation (the detection model is an abstract per-dwell probability).
- No hardware-in-the-loop or real-time execution guarantees.
- No validation against measured hardware or published mission telemetry — all analytic references are textbook/derived, as stated in `validation/VALIDATION.md`.
