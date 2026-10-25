# Requirements — orbitdrift (P074)

Validation Level 3. Every requirement below is stated so that it can be
passed or failed by an executed script, and every one names the script that
does it and the measured result. A requirement whose verification is
`not satisfied` is listed as such rather than removed.

The requirement identifiers are stable; the measured values are from the runs
in `validation/*_output.txt` committed alongside this file.

## 1. Scope and the claim the product is allowed to make

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-1.1 | The package shall measure the separation between two propagations of one element set, and shall not claim to measure ephemeris error against truth. | `validate_lower_bound.py`; the claim is stated in `src/orbitdrift/divergence.py` and `README.md` | satisfied |
| R-1.2 | The only error claim shall be the triangle inequality `max(\|e1\|, \|e2\|) >= D/2`, verified against a case where truth is known in closed form. | `validate_lower_bound.py` | 20 model pairs x 401 samples, 0 violations, worst margin 0.000e+00 km |
| R-1.3 | The package shall demonstrate that the bound does not apply to either model individually. | `validate_lower_bound.py` | a perfect model's error is below the bound at every one of 401 samples |
| R-1.4 | A bound written as `D` rather than `D/2` shall be shown to fail on the same data. | `validate_lower_bound.py` | 401 of 401 samples violate the `D` bound; 0 of 401 violate `D/2` |
| R-1.5 | No output shall be a covariance, and no output shall be usable as the input covariance of a conjunction assessment. | `validate_guard.py`, `tests/test_guard.py` | 23 refusals fire, 15 legitimate calls accepted |
| R-1.6 | The refusal shall cross-reference P071 ConjProb and state where a conjunction covariance has to come from. | `validate_guard.py` | every refusal message contains `P071`, `ConjProb` and `orbit determination` |

## 2. SGP4 reproduction

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-2.1 | SGP4 shall be reproduced against its own published verification vectors. | `validate_sgp4_vectors.py` | 33 cases, 666 rows compared, 1 row skipped on an error code |
| R-2.2 | The worst position disagreement shall be below 1e-6 km. | `validate_sgp4_vectors.py` | 1.154503e-07 km |
| R-2.3 | The worst velocity disagreement shall be below 1e-8 km/s. | `validate_sgp4_vectors.py` | 4.996954e-10 km/s |
| R-2.4 | The source of the vectors shall be named, including the distribution and version they were read from. | `validate_sgp4_vectors.py` | `SGP4-VER.TLE` and `tcppver.out`, package data of the installed `sgp4` 2.27, published with AIAA 2006-6753 |
| R-2.5 | Rows the propagator refuses shall be counted and named, not silently dropped. | `validate_sgp4_vectors.py`, `tests/test_vectors.py` | 1 row, SGP4 error code 3, named |
| R-2.6 | The comparison shall be shown capable of failing. | `validate_sgp4_vectors.py` | a one-character inclination edit produces a 1275.386193 km disagreement |

## 3. The independent dynamical model

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-3.1 | The force model shall declare every active term and its source. | `src/orbitdrift/forces.py`, `README.md` | two-body, `J2`-`J4` zonal, drag on a declared static atmosphere, luni-solar point masses |
| R-3.2 | The zonal acceleration shall agree with the numerical gradient of the potential it claims to be the gradient of. | `tests/test_forces.py` | agrees to 1e-7 of the acceleration norm plus a 1e-9 km/s^2 finite-difference floor |
| R-3.3 | The `J2` term shall reduce to the published closed form. | `tests/test_forces.py` | agrees to 1e-12 relative |
| R-3.4 | The atmosphere model shall be published, including its discontinuities. | `validate_atmosphere.py` | 28 bands reproduced exactly at every base altitude; largest band-join jump 0.001362 |
| R-3.5 | The luni-solar term shall be checked against an expansion with no catastrophic cancellation in it. | `validate_third_body.py` | 8 body-radius combinations, measured difference 1.4999 to 1.7309 times `\|r\|/\|r_b\|` |
| R-3.6 | The `B*` inversion shall be declared an assumption and its effect bounded. | `validate_bstar_sensitivity.py` | a factor of two in `C_D A/m` moves the median divergence by 0.0002 of itself and the worst by 0.5909; correlation of sensitivity with the drag parameter +0.9626 |
| R-3.7 | A negative `B*` shall pass through unchanged rather than be clipped. | `tests/test_forces.py`, `validate_bstar_sensitivity.py` | a negative `B*` gives a negative drag parameter and an along-track thrust |

## 4. Integrator verification (Level 3 core)

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-4.1 | Specific energy shall be conserved under conservative terms to a declared, step-dependent tolerance. | `validate_conservation.py` | tolerance `6e-9 (360/steps_per_rev)^5`; worst measured 1.7183e-09 at 360 steps/rev, 5.6265e-11 at 720, 1.9798e-12 at 1440 |
| R-4.2 | The polar component of angular momentum shall be conserved under the axially symmetric zonal field. | `validate_conservation.py` | 5.6366e-10 relative at 360 steps/rev |
| R-4.3 | The angular momentum **magnitude** shall be shown NOT to be conserved under `J2`, and the residual shown to be step-invariant. | `validate_conservation.py` | 5.125159e-04, relative spread 5.007e-05 across four step sizes |
| R-4.4 | Energy shall decrease monotonically when drag is active. | `validate_conservation.py` | every sampled difference negative, largest -6.121e-05; total change -7.072135e-03 km^2/s^2 |
| R-4.5 | The convergence order shall be measured without a reference solution. | `validate_step_refinement.py` | Richardson order 4.3701 to 4.7123 over three cases |
| R-4.6 | The convergence order shall also be measured against the closed-form two-body solution. | `validate_step_refinement.py` | error falls by 18.666 to 26.093 per halving |
| R-4.7 | The integrator shall be compared against the closed-form solution, with the closed form's own accuracy established first. | `validate_kepler_reference.py` | Kepler residual worst 2.665e-15 rad; closed-form invariants flat to 2.032e-15; relative position error 5.004e-10 at 1440 steps/rev for `e = 0.001` |
| R-4.8 | The truncation floor at the production step shall be measured and compared against the divergence it has to measure. | `validate_step_refinement.py` | floor 1.168030e-01 km at 7 days; median floor/divergence 2.798e-03, 75th percentile 7.699e-03, 90th percentile 1.593e-02, worst 1.850e-01 |
| R-4.9 | Objects for which the floor is a material fraction of the divergence shall be named. | `validate_step_refinement.py`, `README.md` Limitations | 34 of 192 above 1 per cent, 3 of 192 above 10 per cent, named |
| R-4.10 | The negative control on the conservation check shall be anchored on the true value and shall be shown to fire. | `validate_conservation.py` | an offset of 0.5 x tolerance does not fire; 3 x tolerance does |

**R-4.8 is partially satisfied.** The floor is below 1 per cent of the
divergence for three quarters of the population and reaches 18.5 per cent for
the least divergent object. That is recorded as a limitation, not as a pass.

## 5. Frames and transforms

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-5.1 | Greenwich mean sidereal time shall match the published constant at J2000.0. | `validate_frames.py` | 280.460618375 deg against 280.460618370, difference 5.000e-09 deg |
| R-5.2 | Every rotation shall be orthogonal with unit determinant. | `validate_frames.py` | worst 2.220e-16 |
| R-5.3 | Geodetic height shall be checked against an independent iterative solution, and the departure reported as a function of height. | `validate_frames.py` | 0.000 mm at the surface, 1.219 mm at 400 km, 15.488 mm at 1700 km, 35.773 mm at 3000 km |
| R-5.4 | The radial / in-track / cross-track basis shall be orthonormal, right-handed and an isometry. | `validate_frames.py` | worst 1.164e-15 |
| R-5.5 | The in-track axis shall be shown to be the one the secular growth appears on. | `validate_frames.py` | median in-track share 0.999859, largest component in 0.996466 of rows |
| R-5.6 | Permuting the basis shall break that check. | `validate_frames.py` | 0.000000 against the 0.99 threshold |
| R-5.7 | No transform shall be claimed to be cross-checked against a package that is not installed. | `validate_frames.py`, `README.md` | `astropy` recorded as not installed in every validation output |

## 6. The analytic baseline, established before anything is learned

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-6.1 | The analytic growth forms shall be implemented and validated against the simulation before the learned model is fitted. | `validate_growth_baseline.py` runs before `validate_predictors.py`; the ordering inside the benchmark is enforced in `orbitdrift.benchmark` | 11 analytic rows precede 2 learned rows in the cold task; 8 precede 4 in the extrapolation task |
| R-6.2 | The fitted growth exponent shall approach the predicted linear value where drag is negligible. | `validate_growth_baseline.py` | 0.9685 +/- 0.0628 above 1100 km over 46 objects |
| R-6.3 | The exponent shall rise toward the predicted quadratic value where drag dominates. | `validate_growth_baseline.py` | 1.2683 +/- 0.3914 below 450 km over 42 objects |
| R-6.4 | The exponent shall correlate with altitude and with the drag parameter in the predicted directions. | `validate_growth_baseline.py` | -0.2914 with log altitude, +0.5090 with the log drag parameter |
| R-6.5 | The baseline's own fit quality shall be stated before the benchmark. | `validate_growth_baseline.py` | median per-object log-residual spread 0.07469, 90th percentile 0.75145 |
| R-6.6 | Shuffling the data shall destroy the fit. | `validate_growth_baseline.py` | median log-residual spread 1.14177 shuffled against 0.07469 real |
| R-6.7 | A rank-deficient growth fit shall be refused rather than returning a least-norm answer. | `tests/test_growth.py`, `tests/test_regressions.py` | repeated ages raise `InsufficientDataError` naming rank deficiency |

## 7. The learned component and the interval

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-7.1 | The learned model shall see exactly the features the analytic baseline sees. | `src/orbitdrift/benchmark.py`, `validate_predictors.py` | identical design matrix in the cold task; in the extrapolation task the learned model additionally receives the baseline's own fitted parameters |
| R-7.2 | The split shall be by object, never by row. | `validate_predictors.py`, `tests/test_dataset.py` | 134 training and 58 test objects, disjoint |
| R-7.3 | Interval coverage shall be measured, not assumed. | `validate_predictors.py`, `validate_coverage_table.py` | every row of every table carries a measured coverage and a Clopper-Pearson interval |
| R-7.4 | Coverage shall be reported stratified, including the cells where it is miscalibrated. | `validate_coverage_table.py` | 65 cells inconsistent with the nominal level across four tables |
| R-7.5 | The product shall report where a conformally calibrated predictor is conditionally miscalibrated. | `validate_coverage_table.py` | marginal 0.8768 with altitude cells spanning 0.7802 to 0.9375, a spread of 0.1573 |
| R-7.6 | The comparison between families shall be made among predictors whose measured coverage is consistent with the level they claim. | `src/orbitdrift/benchmark.py` | three rankings reported: overall, calibrated, and not-under-covering |
| R-7.7 | A baseline win shall be reported as measured. | `validate_predictors.py` | extrapolation task: best point prediction is analytic (0.1299 against 0.1391); the only calibrated interval is analytic; no learned interval reaches its nominal coverage |
| R-7.8 | The rule of thumb the product is positioned against shall be measured. | `validate_predictors.py` | coverage 0.3017 at a claimed 0.90, interval score 1352.77 km |

## 8. Determinism and reproducibility

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-8.1 | The committed population shall be reproducible byte for byte from its seed. | `validate_reproducibility.py` | sha256 matches |
| R-8.2 | The divergence computation shall be bit-identical across runs with the same input. | `validate_reproducibility.py` | worst difference 0.000e+00 |
| R-8.3 | A subset recomputation shall reproduce the committed rows with the step pinned, and the departure shall be explained. | `validate_reproducibility.py` | 8.305428e-09 relative, attributed to the length-dependence of numpy's vectorised transcendental loops |
| R-8.4 | An unpinned step shall be shown to change the rows measurably. | `validate_reproducibility.py` | 1.141518e-02 relative |
| R-8.5 | Every fit in the benchmark shall be reproducible. | `validate_reproducibility.py` | 0 mismatched rows in both tasks |
| R-8.6 | A different split seed shall change the measured coverage. | `validate_reproducibility.py` | 13 of 13 rows change |

## 9. Suite and script hygiene

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-9.1 | No assertion shall depend on `pytest.approx`'s default absolute tolerance. | `validate_approx_audit.py`, `tests/test_regressions.py` | 0 of 67 `pytest.approx` calls pass `rel=` without an explicit `abs=` |
| R-9.2 | The audit shall be shown capable of detecting a difference. | `validate_approx_audit.py` | forcing `abs=0.0` everywhere changes 17 assertions; stripping the explicit `abs=` changes 10 (hypothesis 6.170.9) or 11 (6.140.3) |
| R-9.3 | Every validation script and every example shall exit zero from any working directory. | `validate_cwd_independence.py` | 26 scripts x 4 directories, 104 invocations, 0 non-zero exits |
| R-9.4 | No committed file shall contain an absolute path. | `tests/test_regressions.py`, `.github/workflows/tests.yml` | scan over every tracked Python file and every validation output |
| R-9.5 | The test count shall be read from junit XML and cross-checked against a literal element count. | `.github/workflows/tests.yml`, `validate_approx_audit.py` | declared equals counted in every run |
| R-9.6 | No source file shall claim flight qualification, certification or production readiness. | `tests/test_regressions.py` | phrase scan over `src/orbitdrift/*.py` with string continuations joined |

## 10. What is not verified, and why

| ID | Statement |
|----|-----------|
| R-10.1 | **Error against truth is not verified and cannot be.** There are no precision ephemerides and no element-set archive in this environment. Everything this product measures is model divergence. |
| R-10.2 | **Baseline 1 is demonstrated on exactly one real element-set pair**, catalogue 5 twenty years apart, whose magnitude is meaningless as a staleness figure. The machinery is verified on synthetic pairs with geometrically derivable separations (`validate_self_consistency.py`: chord ratio 0.99996 at every offset tested, growth-rate ratio 0.9965 to 0.9966, identical pair 7.691751e-10 km). |
| R-10.3 | **The atmosphere model has no solar-activity dependence.** Published solar-cycle variation at 400 km is about an order of magnitude; this model sits at one unspecified point in that range, and every drag-driven number inherits it. |
| R-10.4 | **Tesseral and sectoral gravity terms, solar radiation pressure, tides and relativistic corrections are not modelled.** They are part of the declared divergence rather than part of the model. |
| R-10.5 | **The population is synthetic and drawn independently in each feature.** It is a spanning set over feature space, not a representation of the real catalogue's joint distribution. |
| R-10.6 | **The predictors are only valid on the collinear feature manifold the population occupies.** `AnalyticGrowthPredictor.leverage` measures the distance from it; the training maximum is about 7.0 and an internally inconsistent hand-built vector scores 24 to 259, where the model's prediction is wrong by seven orders of magnitude. |
| R-10.7 | **The inclination recovered from a state vector has an accuracy floor near zero.** `arccos(h_z/\|h\|)` has derivative `-1/sin(i)`, so a few machine epsilons of relative error become `eps/sin(i)` radian: at `i = 6e-08` the floor is about 3e-08 radian, half the inclination itself. Found by Hypothesis, stated in `rv_to_elements`, and pinned in `tests/test_regressions.py`. |
| R-10.8 | **A fixed-step integrator is a poor fit to an eccentric orbit.** At `e = 0.59` the relative position error at 1440 steps per revolution is 1.014e-06 against 5.004e-10 at `e = 0.001`, which is why the population is capped near `e = 0.05`. |

## Licence

Apache-2.0, copyright 2026 OPTIMA Organisation. The credits line required by
the mission appears once, in README.md under Credits.
