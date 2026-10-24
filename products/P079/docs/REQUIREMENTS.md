# Requirements and their verification

Level 3 products in this mission carry a requirements document that states each
requirement, how it is verified, and where the evidence is. Every row below
points at a script that was executed and an output file that the script wrote.

Validation label: **Level 3, hardware-pending.** Nothing here has run on a
board; see `docs/DEPLOYMENT.md` section 0 for exactly what is missing.

## Functional requirements

| ID | Requirement | Verification | Evidence |
|---|---|---|---|
| F-01 | Estimate the remote clock offset from a four-timestamp exchange using the symmetric-delay estimator | Hand-calculated known answer, and exactness under a constructed symmetric delay | `tests/test_twoway.py`, `validation/validate_symmetric_bias_output.txt` section 1a |
| F-02 | Report the mean path delay, free of the clock offset | Integer identity `(t4-t1)-(t3-t2) == (t2-t1)+(t4-t3)` over generated inputs | `tests/test_properties.py::test_round_trip_is_the_exact_integer_sum_of_the_dwells` |
| F-03 | Derive the asymmetry-induced bias in closed form and verify it | Declared-asymmetry sweep against `A/2`, and a fitted slope | `validation/validate_symmetric_bias_output.txt` sections 1b, 1c |
| F-04 | Show that the bias is independent of delay, turnaround and offset | Three sweeps at fixed asymmetry | `validation/validate_symmetric_bias_output.txt` section 1d |
| F-05 | Estimate offset, skew and drift by least squares with a parameter covariance | Exact-polynomial recovery and a convergence study | `validation/validate_skew_estimators_output.txt` sections 4a, 4b |
| F-06 | Estimate offset and skew by a two-state clock-model Kalman filter | Agreement with a straight-line fit at zero process noise; covariance symmetry and positive semidefiniteness | `tests/test_skew.py`, `tests/test_properties.py` |
| F-07 | Report stability as an overlapping Allan deviation over tau | Fitted slopes and amplitudes against three closed-form processes | `validation/validate_allan_output.txt` sections 2a, 2b |
| F-08 | Report a confidence interval at each tau | Measured coverage by Monte Carlo against the closed-form value | `validation/validate_allan_output.txt` section 2c |
| F-09 | Report a **measured** uncertainty on the offset estimate | Three-component budget with a serial-correlation inflation measured from the residuals | `tests/test_uncertainty.py`, `validation/validate_uncertainty_coverage_output.txt` |
| F-10 | Measure the coverage of that uncertainty against nominal | Four treatments over disjoint train/calibrate/cover splits | `validation/validate_uncertainty_coverage_output.txt` sections 3c, 3d |
| F-11 | Correct the offset bias from exchange features | Three analytic estimators and two learned ones, scored on one held-out split | `validation/validate_asymmetry_benchmark_output.txt` |
| F-12 | Expose an uncertainty from the learned component | Quantile-pair dispersion, with its coverage measured | `validation/validate_asymmetry_benchmark_output.txt` section 7c |
| F-13 | Provide a command-line interface over the library | Every command executed, exit status checked | `validation/validate_cli_output.txt` |

## Level 4 scaffolding requirements

| ID | Requirement | Verification | Evidence |
|---|---|---|---|
| S-01 | Hardware abstraction layer over the timebase, with simulation/device parity asserted by test | Byte-identical timestamps and exactly equal estimates across four cases, plus a falsifying control | `tests/test_parity.py`, `validation/validate_parity_output.txt` |
| S-02 | Simulation mode with a seeded clock model: declared drift, jitter, asymmetry, outages; fully reproducible, no hardware | Reproducibility checked at seven levels by digest or byte comparison | `tests/test_reproducibility.py`, `validation/validate_reproducibility_output.txt` |
| S-03 | Dry-run mode exercising the full acquisition and estimation path against a recorded exchange without touching a device | Counted driver operations, zero hardware operations, and a negative control that must fire | `tests/test_dryrun.py`, `validation/validate_dryrun_output.txt` |
| S-04 | Documented deployment and recovery procedure for the Jetson Orin Nano, including loss of the exchange mid-estimate | The procedure, plus the checks it tells the operator to run, executed | `docs/DEPLOYMENT.md`, `validation/validate_deployment_output.txt` |
| S-05 | Benchmark harness producing the timing and resource table, hardware rows empty and marked unmeasured | Host table measured; three guards against filling a hardware row exercised | `tests/test_bench.py`, `validation/validate_benchmark_table_output.txt` |

## Non-functional requirements

| ID | Requirement | Verification | Evidence |
|---|---|---|---|
| N-01 | No number in any document comes from anywhere but a script executed in the build session | Every figure in README.md, VALIDATION.md and MODEL_CARD.md names its script | `validation/VALIDATION.md` |
| N-02 | No hardware number anywhere | `is_hardware` is `False` on every backend; the Level 4 table is empty | `validation/validate_benchmark_table_output.txt` section 10b |
| N-03 | Every validation script exits zero from any working directory | 13 scripts run from four directories | `validation/validate_cwd_independence_output.txt` |
| N-04 | No committed file contains an absolute path | Paths printed through `_bootstrap.relative`; scanned before hand-off | `validation/_bootstrap.py` |
| N-05 | Every dependency bounded on both sides | `pyproject.toml` | `pyproject.toml` |
| N-06 | Ruff-clean at line length 100 with E, F, W, I, UP, B | `ruff check src/ tests/ examples/ validation/` | CI workflow |
| N-07 | Seeded reproducibility of every stochastic component | Digest comparison across repeated runs | `validation/validate_reproducibility_output.txt` |
| N-08 | Property-based tests over the algebraic identities, with every counterexample pinned | Hypothesis over the estimator identities; named regression tests with literal inputs | `tests/test_properties.py`, `tests/test_regressions.py` |

## Explicitly out of scope

| ID | Not provided | Use instead |
|---|---|---|
| X-01 | Steering or disciplining a clock | `linuxptp` (`phc2sys`), `chrony` |
| X-02 | An NTP or PTP protocol implementation | `ntplib` for NTP client queries, `linuxptp` for PTP |
| X-03 | Board bring-up: power, flashing, console capture | `labgrid` |
| X-04 | A hardware timestamping driver | written per board; the five-method shim is `timetransfer.hal.TimestampDriver` |
| X-05 | Real-time guarantees | Python is not a real-time language and this package sets no scheduling policy |
| X-06 | Flicker-noise synthesis | deliberately absent: it has no finite-dimensional Markov representation, and verifying an estimator against an approximation would undermine the point |
