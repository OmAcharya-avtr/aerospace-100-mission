# Requirements — twincalib 0.1.0

Validation level 3 products carry a requirement specification with a test or a
validation script against every requirement. This is that document.

Status: TESTING. Research-grade software: **not flight-qualified, not certified,
not approved for operational aerospace use.**

Identifier scheme: `TC-<area>-<n>`. Verification method is one of **T** (a test
in `tests/`), **V** (a validation script in `validation/`), **I** (inspection of
the source), **A** (analysis, with the analysis recorded in a docstring).

Every requirement below is either **MET** with evidence, or **MET WITH A STATED
LIMITATION**, or **NOT MET** and recorded as such. There are two of the last
kind and they are at the end.

---

## Models (TC-MOD)

| ID | Requirement | Method | Evidence | Status |
|---|---|---|---|---|
| TC-MOD-1 | The two-node twin shall implement a lumped thermal network with conduction and grey-body radiation, with every equation carrying its source, units, assumptions and validity range. | I | `src/twincalib/model.py` `ThermalTwinModel` docstring: Gilmore 2002 Ch. 1, Incropera and DeWitt Ch. 5, assumptions 1-5, validity range 200-350 K | MET |
| TC-MOD-2 | The integrator shall be fourth-order accurate and its measured order shall be reported. | V | `validate_model.py` check 2: error ratios 17.765 / 16.857 / 16.423 / 16.210 per halving | MET |
| TC-MOD-3 | The transient solution shall agree with the independently derived algebraic equilibrium. | V | `validate_model.py` check 4: worst discrepancy 1.307e-12 K over three input conditions | MET |
| TC-MOD-4 | A single-node model with a closed-form solution and closed-form sensitivities shall be provided, so that the differentiation and identifiability machinery have known answers. | T, V | `tests/test_model.py`, `validate_sensitivity.py` check 1 | MET |
| TC-MOD-5 | The integrator's own discretisation error on a shipped design shall be below 1 per cent of the measurement noise. | V | `validate_model.py` check 5a: 4.761541e-06 K, 1.587e-05 of 0.3 K | MET |
| TC-MOD-6 | The sensitivity of the solution to the input discretisation shall be measured and reported. | V | `validate_model.py` check 5b: **0.626 K**, 2.09 times the noise, reported and not bounded | MET WITH A STATED LIMITATION (README Limitations 1) |
| TC-MOD-7 | A parameter vector outside the integrator's stability region shall raise with a message naming the limit, never return a finite wrong answer. | T, V | `tests/test_model.py::test_unstable_step_is_reported_not_silently_wrong`; `validate_model.py` check 7 | MET |
| TC-MOD-8 | Simulated temperatures on every shipped design shall stay inside the model's declared validity range. | V | `validate_model.py` check 9: 274.496 to 309.789 K against a declared 200-350 K | MET |
| TC-MOD-9 | Invalid inputs shall raise `ValueError` or `TypeError` with an actionable message. | V | `validate_model.py` check 10: 9 of 9 negative controls rejected | MET |

## Sensitivity and information (TC-SEN)

| ID | Requirement | Method | Evidence | Status |
|---|---|---|---|---|
| TC-SEN-1 | Sensitivities shall be computed in log-parameter coordinates, so that the spectrum is a statement about the experiment and not about the unit system. | I, A | `src/twincalib/sensitivity.py` module docstring, with the reason stated | MET |
| TC-SEN-2 | The finite-difference sensitivity shall agree with a closed form to better than 1e-6 relative at the default step. | V | `validate_sensitivity.py` check 1: 2.476e-07 relative | MET |
| TC-SEN-3 | The differentiation step shall be chosen by a measured step study, not by rule of thumb, and the study shall be committed. | V | `validate_sensitivity.py` check 1, eight decades; the default 1e-4 sits two decades inside the flat region | MET |
| TC-SEN-4 | The Fisher information shall be symmetric, positive definite on a sound design, and exactly the Gram matrix of the weighted sensitivity. | V | `validate_sensitivity.py` check 4: asymmetry 0.000e+00, eigenvalues 1.81e+02 to 2.85e+07, Gram difference 0.000e+00 | MET |
| TC-SEN-5 | A structural proportionality between two sensitivity columns that follows from the equations shall be verified numerically. | V | `validate_sensitivity.py` check 5: predicted factor 3.055556, residual 1.014e-09 on `full-sun`; 0.6539 on `baseline-eclipse` where it must fail | MET |

## Identifiability (TC-IDE)

| ID | Requirement | Method | Evidence | Status |
|---|---|---|---|---|
| TC-IDE-1 | The package shall report the numerical rank, the condition number and a collinearity index of a design before any parameter value is printed. | T, V | `tests/test_cli.py`, `validate_cli.py` check 3: verdict precedes estimate in both fits | MET |
| TC-IDE-2 | Near-null directions shall be reported as the parameter **combinations** that are unconstrained, not as a warning. | T, V | `Combination.as_text`; `validate_identifiability.py` check 4 prints them for every design | MET |
| TC-IDE-3 | A structurally non-identifiable configuration with an analytically known null direction shall be shipped and the computed direction shall match it. | T, V | `validate_identifiability.py` checks 1 and 2: 8.783e-09 and 1.341e-09 worst component error | MET |
| TC-IDE-4 | The null direction shall be verified independently of the SVD, by showing the predictions do not move along it. | V | `validate_identifiability.py` checks 1 and 2: 0.000e+00 K over scalings 0.25-10 and heat shifts 1-17 W | MET |
| TC-IDE-5 | The rank verdict shall not depend on the rank tolerance within two decades of the default, and any verdict that does move shall be reported. | V | `validate_identifiability.py` check 3: stable over 1e-6 to 1e-10; **flips at 1e-12 and 1e-5**, with the reason printed | MET |
| TC-IDE-6 | A covariance computed by pseudo-inverse shall not report a finite standard deviation for a parameter loading on a null direction. | T, V | `null_direction_loading`; `validate_profile.py` check 2: raw pinv would print 0.0187905, the package prints `inf` | MET |
| TC-IDE-7 | A design that passes the rank test and is nevertheless useless shall be shipped and identified as such. | V | `weak-excitation`: rank 6, near-null direction at 4.2156e-06, relative standard deviations to 48.76 | MET |

## Estimation (TC-EST)

| ID | Requirement | Method | Evidence | Status |
|---|---|---|---|---|
| TC-EST-1 | The baseline estimator shall be weighted nonlinear least squares on the physics alone, implemented and measured before any learned component. | I | `src/twincalib/fitting.py`; no import of `residual` or `benchmark` anywhere in it | MET |
| TC-EST-2 | The estimator shall recover known parameters on synthetic data across noise levels, and its spread shall be compared with the Cramer-Rao bound. | V | `validate_recovery.py` check 1: spread/bound 0.787 to 1.493 over 30 cells | MET |
| TC-EST-3 | Parameter covariance and correlation shall be reported, with the strongly correlated pairs called out. | T, V | `render_correlation`; `validate_identifiability.py` check 5 | MET |
| TC-EST-4 | The fit shall report a goodness-of-fit statistic that distinguishes model error from parameter error. | V | `FitResult.reduced_chi2`; `validate_recovery.py` check 2: 37.90 at sigma = 0.05 K under model error | MET |
| TC-EST-5 | The fit shall be independent of the starting point on a sound design. | V | `validate_recovery.py` check 4: worst log difference 5.375e-07 over five starts spanning a factor of four | MET |
| TC-EST-6 | A non-converged fit shall be reported as non-converged, including a fit that ends on a search bound. | T, V | `tests/test_fitting.py::test_bound_hit_is_reported_as_not_converged`; `validate_recovery.py` check 3: 0 of 6 converged on `weak-excitation` | MET |

## Profile likelihood (TC-PRO)

| ID | Requirement | Method | Evidence | Status |
|---|---|---|---|---|
| TC-PRO-1 | Profile-likelihood intervals shall be provided for every parameter, cut at the Wilks threshold. | T, V | `profile_likelihood`; threshold 3.8414588207 at 0.95 | MET |
| TC-PRO-2 | On a sound design the profile interval shall agree with the quadratic interval. | V | `validate_profile.py` check 1: width ratios 0.9975 to 1.0016 | MET |
| TC-PRO-3 | Interval coverage shall be measured over seeded repetitions, with an interval on the coverage estimate itself. | V | `validate_coverage.py`: pooled 0.9306, exact [0.8453, 0.9771], nominal 0.95 inside | MET |
| TC-PRO-4 | Coverage shall also be measured with the model-form discrepancy present, and reported however it comes out. | V | `validate_coverage.py`: pooled **0.3750**, exact [0.2636, 0.4970] | MET (the result is bad and is published) |
| TC-PRO-5 | A profile that does not cross the threshold shall be reported as open, with the searched range, never extrapolated. | T, V | `render_profiles`; `validate_profile.py` check 2: `q_sun` open below, searched [2.755, 1111.4] | MET |
| TC-PRO-6 | Where a design determines a combination but not its parts, the combination shall be reported as determined. | T, V | `validate_profile.py` check 2: `q_sun + q_par` to +0.6 per cent, worst 2.08 per cent over 8 further seeds | MET |

## Learned residual model (TC-RES)

| ID | Requirement | Method | Evidence | Status |
|---|---|---|---|---|
| TC-RES-1 | The learned component shall be implemented after the baseline and benchmarked against it on the same held-out data. | I, V | `validate_residual.py` check 1, four designs | MET |
| TC-RES-2 | The learned component shall expose an uncertainty output, not only a point estimate. | T, V | `ResidualPrediction.std`; `validate_residual.py` check 2 | MET |
| TC-RES-3 | That uncertainty shall be measured against nominal coverage and the result reported whatever it is. | V | `validate_residual.py` check 2: 0.9709 to 0.9933 against 0.9500 — **conservative, not calibrated** | MET (the result is bad and is published) |
| TC-RES-4 | The test split shall hold out whole trajectories by input phase and by noise seed, and no test sample shall appear in training. | I, V | `run_benchmark` docstring and implementation; `tests/test_benchmark.py::test_benchmark_reports_every_trajectory` | MET |
| TC-RES-5 | The feature builder shall not have access to the measurements it predicts. | I, V | `build_features` takes no measurements; `validate_residual.py` check 6: feature difference 0.000e+00 across seeds differing by 1.489 K | MET |
| TC-RES-6 | A design on which there is nothing to learn shall be shipped as a control, and the model shall be shown to find nothing on it. | V | `lumped-nodes`: discrepancy 0.1268 K against 0.3 K noise, improvement +0.0189 inside the margin, verdict **tie**, fitted amplitude 7.64e-05 | MET |
| TC-RES-7 | If the physics-only baseline wins or ties, that shall be reported and shall survive into the README, VALIDATION.md and MODEL_CARD.md. | I | README "Where the physics-only baseline wins, in full"; VALIDATION.md section 7; MODEL_CARD.md "Metrics" | MET |

## Absorption (TC-ABS)

| ID | Requirement | Method | Evidence | Status |
|---|---|---|---|---|
| TC-ABS-1 | The package shall provide a **measurement** of how much of the residual model's improvement absorbs an identifiability failure, not a caveat. | V | `decompose_residual`, `parameter_drift`; `validate_residual.py` checks 3 and 4 | MET |
| TC-ABS-2 | The decomposition shall be verified on constructed fields with known answers. | T, V | `validate_residual.py` check 7: parameter perturbation 0.000e+00 orthogonal; orthogonal field 1.000000000 | MET |
| TC-ABS-3 | The three energy fractions shall sum to one, and that shall be checked rather than assumed. | T, V | `tests/test_properties.py::test_decomposition_fractions_sum_to_one`; the `sum` column in `validate_residual.py` check 3 reads 1.000000 everywhere | MET |
| TC-ABS-4 | The sensitivity of the absorption figure to its reporting threshold shall be measured and published, including a threshold at which the conclusion changes. | V | `validate_residual.py` check 3: three cuts, and the **ordering inverts at 1e-1** with the reason printed | MET |
| TC-ABS-5 | The stability of the absorption figure against configuration shall be measured, and only the claim that survives shall be made. | V | `validate_absorption.py`: ranges [0.00161, 0.02441] and [0.08144, 0.15605] disjoint; ratio swings 3.34 to 96.74 and is **not** claimed | MET |
| TC-ABS-6 | Whether the physical parameters move towards or away from the truth under the discrepancy term shall be measured. | V | `validate_residual.py` check 4: worse on 3 of 4 designs; `baseline-eclipse` 0.132553 to 0.178772 | MET |

## Software quality (TC-SWQ)

| ID | Requirement | Method | Evidence | Status |
|---|---|---|---|---|
| TC-SWQ-1 | No library module shall write to a stream; the CLI shall be the only writer. | T | `tests/test_vocabulary.py::test_library_code_contains_no_print_statement` | MET |
| TC-SWQ-2 | Ruff shall pass at line length 100 over `src`, `tests`, `examples` and `validation`. | I | `.github/workflows/tests.yml` lint step | MET |
| TC-SWQ-3 | Public APIs shall carry type hints and docstrings with units. | I | every public function in `src/twincalib` | MET |
| TC-SWQ-4 | No floating-point pipeline shall be asserted bit-exact, and no boolean produced by a strict threshold comparison shall be asserted for equality. | I, T | `tests/test_benchmark.py::test_winner_is_consistent_with_the_improvement_and_the_margin` checks the statistic and allows the boolean to follow | MET |
| TC-SWQ-5 | Every counterexample found by Hypothesis shall be pinned as a named regression test with literal inputs. | T | `tests/test_regression.py`, three pinned cases with their measured values | MET |
| TC-SWQ-6 | Examples shall use the Agg backend and shall never call `show()`. | T | `tests/test_vocabulary.py::test_examples_use_the_agg_backend_and_never_show` | MET |
| TC-SWQ-7 | Every dependency shall be bounded on both sides at a range the suite was actually run at, and the committed validation output diffed at both ends. | V | `validation/VALIDATION.md`, "Dependency range measurement" | MET |
| TC-SWQ-8 | Scratch patterns in `.gitignore` shall be anchored, and no shipped file shall be ignored. | I | `/junit.xml` is anchored; `git check-ignore -v` run against every shipped file | MET |
| TC-SWQ-9 | Every run shall finish inside three minutes on two cores. | V | the compute-budget table in README; longest is `validate_coverage.py` at 131.2 s | MET |

## Requirements NOT met

| ID | Requirement | Why not, and what is there instead |
|---|---|---|
| TC-IDE-8 | **Global** identifiability: the package should rule out a second, distant optimum. | **NOT MET.** Everything here is local, at one parameter point. A full-rank sensitivity matrix at `theta_hat` does not exclude a distant second optimum, and nothing in this package searches for one. `validate_recovery.py` check 4 shows five starts spanning a factor of four reaching the same optimum on one sound design, which is weak evidence for one design and not a global result. The README names `emcee` as the tool whose chains would answer this, and Limitations 2 records it. |
| TC-PRO-7 | True (unconstrained) profile likelihoods on every design. | **NOT MET.** Nuisance parameters are confined to a box of plus or minus two natural log units around the unconstrained estimate, because without it a structurally non-identifiable direction walks until the fixed-step integrator becomes unstable and the profile of an unrelated parameter fails for an unrelated reason. On `weak-excitation` that box is **active**, so those profiles are box-constrained and five of six come out narrower than the quadratic interval. Both facts are printed by `validate_profile.py` check 3 and recorded in README Limitations 6. A variable-step stiff integrator would remove the need for the box and is not implemented. |
