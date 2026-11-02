# Requirements — hilverdict (P083)

Validation Level 3. Every requirement below is stated so that it can be passed
or failed by an executed script, and every one names the script that does it
and the measured result. A requirement whose verification is `not satisfied`
or `partially satisfied` is listed as such rather than removed.

The requirement identifiers are stable; the measured values are from the runs
in `validation/*_output.txt` committed alongside this file, all produced in the
session that wrote it.

## 1. Scope, and the claim the product is allowed to make

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-1.1 | The product shall decide whether a hardware-in-the-loop run was timing-faithful at a DECLARED tolerance, and shall not claim to decide whether the run was correct. | `validate_learned_model.py`; the claim is stated in `src/hilverdict/verdict.py` and `README.md` | satisfied |
| R-1.2 | No output shall be described as flight-safe, certified, mission-ready or production-ready. | `validate_cli.py` section 1 checks that the verdict output contains "not a certification"; `README.md` and `MODEL_CARD.md` carry the safety statement | satisfied |
| R-1.3 | A dismissal shall be a theorem about the declared linearisation, and the product shall say so in the same table as the number. | `validate_cli.py` section 1 requires the fragment "is a theorem about the declared linearisation" in the `certify` output | satisfied |
| R-1.4 | The verdict vocabulary shall separate "no miss mattered" from "the certificate could not settle it" from "the run is outside what any method here covers". | `tests/test_verdict.py`, `validate_negative_controls.py` section 2 | four verdicts: `TIMING_FAITHFUL`, `NOT_TIMING_FAITHFUL`, `UNDECIDED`, `UNDECIDABLE` |
| R-1.5 | Every number in `README.md`, `MODEL_CARD.md` and `validation/VALIDATION.md` shall come from a script in this repository, with its raw output committed. | 11 scripts, 11 `*_output.txt` files | satisfied |

## 2. The declared plant and its discretisation

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-2.1 | The plant's coefficients shall be declared as a fiction and attributed to no real article. | `src/hilverdict/plant.py` docstring, `README.md`, `DATASET_CARD.md` | satisfied |
| R-2.2 | The trim of equations (4)-(5) shall be an exact equilibrium of the CUBIC plant, not of its linearisation. | `validate_plant.py` section 1 | worst `max \|xdot\|` **2.220446e-16 1/s** over 7 amplitudes, tolerance 1e-14 |
| R-2.3 | Dropping the cubic term from the trim shall be shown to leave a residual far above rounding. | `validate_plant.py` section 1 | **2.232962e-01 1/s** at 10 deg |
| R-2.4 | The analytic Jacobian shall agree with central differences to the central-difference truncation error. | `validate_plant.py` section 2 | worst **2.188205e-10 1/s**, tolerance 1e-6 derived from `O(h^2)\|d3f\|`, `\|d3f\| <= 6\|m_alpha3\| = 252` |
| R-2.5 | The zero-order-hold input matrix shall agree with two constructions the implementation does not use. | `validate_plant.py` section 3 | inverse formula **2.220446e-16**, composite Simpson at 2048 panels **8.881784e-16** |
| R-2.6 | The discretisation shall not require the state Jacobian to be invertible. | `src/hilverdict/plant.py`, Van Loan block exponential; `tests/test_plant.py` | satisfied by construction; the inverse formula is used only as a cross-check |
| R-2.7 | The integrator shall converge at its advertised order against a solution that is exact. | `validate_plant.py` section 4 | orders **3.9924, 3.9958, 3.9982, 4.0074, 3.9437** against the matrix exponential on the linear plant |
| R-2.8 | The integrator's error at the declared substep count shall be below 1e-10 of the state norm. | `validate_plant.py` section 4 | **2.245295e-12** |
| R-2.9 | The closed loop shall be asymptotically stable. | `validate_plant.py` section 6 | spectral radius **0.984651507** |
| R-2.10 | The constructed monodromy shall match a step response measured through the integrator. | `validate_plant.py` section 6 | **4.085621e-13** |

## 3. The divergence metric

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-3.1 | The metric shall be weighted by the DECLARED engineering tolerances, so the verdict threshold is exactly 1 and nothing tunes it. | `validate_plant.py` section 5 | a difference of exactly one tolerance in either channel scores 1 to 1e-15; both at once score **1.41421356237309515** |
| R-3.2 | The metric shall be absolutely homogeneous and satisfy the triangle inequality. | `validate_plant.py` section 5, `tests/test_properties.py` | relative homogeneity residual **4.862279e-16** over 20000 pairs; smallest triangle slack **1.288072e-10** |
| R-3.3 | The effect of the metric's anisotropy on the certificate shall be MEASURED, not remarked on. | `validate_certificate.py` section 6 | median ratio of truth to bound **0.984466** at anisotropy 5 falling to **0.326902** at the declared 200 |

## 4. The analytic certificate (Level 3 core)

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-4.1 | Every closed form shall be checked against a hand calculation. | `validate_certificate.py` section 1 | `mu_2` on two hand matrices, difference **0.000e+00**; `phi(0,t)` exactly `t`; `gamma_0` exactly 1 |
| R-4.2 | Dahlquist's inequality, on which equations (10)-(11) rest, shall be verified numerically and shall be shown not to be vacuous. | `validate_certificate.py` section 2 | **0 violations in 3224 (matrix, time) pairs** over 124 matrices; tightest ratio **1.00000000000000022** at `t = 0` |
| R-4.3 | The bound shall be strictly increasing in the overrun and exactly proportional to the input step. | `validate_certificate.py` section 3 | smallest increment **1.820198e-02** over 500 steps; departure from proportionality **1.421085e-14** |
| R-4.4 | Equation (14) shall invert equation (13), checked on the BOUND and not on the verdict boolean. | `validate_certificate.py` section 3 | worst relative departure **1.110223e-15** over 18 cases; the bound at `1.001 tau*` exceeds the tolerance in **18 of 18** |
| R-4.5 | The certificate shall not be violated by the nonlinear re-simulation over a population disjoint from the learned study's. | `validate_certificate.py` section 4 | **0 violations of 875 misses** over 45 runs, seeds 3000-3044 |
| R-4.6 | The bound shall be shown not to be vacuous. | the same | worst ratio of truth to bound **0.710647**, so within a factor of **1.4072** of the truth at its tightest |
| R-4.7 | The looseness shall be reported by overrun band, not only in aggregate. | the same | median ratio 0.368116 below 0.1 T, 0.045942 at two periods or more |
| R-4.8 | No consequential miss shall be dismissed. | `validate_certificate.py` section 5, `validate_learned_model.py` section 1 | **0 of 50** on the soundness population; **0 of 102** on train and **0 of 47** on test |
| R-4.9 | The certificate shall dispose of the large majority of misses. | `validate_certificate.py` section 5 | dismissal rate **0.876571** |
| R-4.10 | Every factor of equation (13) shall be shown necessary by a control anchored on the re-simulated truth. | `validate_independent_bound.py` section 5 | without `c_free` **47 violations**; without `gamma_sup` **192**; without both **271**; the full bound **0** |
| R-4.11 | A factor that the population does NOT falsify shall be reported as such and shall not be offered as a control. | the same | replacing `phi(mu_S, tau)` by `tau` gives **0 violations**, worst ratio 0.717512; reported, not controlled |
| R-4.12 | The certificate shall be compared against at least one bound derived from mathematics it does not use. | `validate_independent_bound.py` sections 1-4 | four routes: the Gronwall factor of eq (16), the exact Van Loan integral, the exact linear discrete response, and adaptive quadrature |
| R-4.13 | A cross-check that is vacuous over most of its range shall be reported at the strength it has. | `validate_independent_bound.py` section 1 | the Gronwall factor exceeds 1e6 by `N = 12`; it is non-vacuous at `N = 1`, **4.920026** against `gamma_sup(1) = 1.599218`, and that is what is claimed |
| R-4.14 | The certificate shall refuse rather than extrapolate where a step of its derivation does not hold. | `validate_negative_controls.py` section 2 | three refusals: actuator saturation, incidence beyond 10 deg, mean utilisation at or above 1; all three fire, an admissible run does not |

**R-4.11 and R-4.13 are satisfied by reporting a weakness, not by passing a
test.** Both are recorded here so that a later session cannot mistake the
reported strength for a stronger claim.

## 5. The re-simulation reference

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-5.1 | The reference shall be bit-deterministic for identical inputs, and that claim shall be shown not to be vacuous. | `validate_resimulation.py` section 1 | bit-identical; a different seed changes the timing trace by up to 4.13e-02 s |
| R-5.2 | Re-simulating only the tail of a run from the reference state shall be identical to re-simulating the whole run. | `validate_resimulation.py` section 2 | **0.000000e+00** over 128 misses, bit-identical |
| R-5.3 | The VERDICT, not only the number, shall be shown stable under the integrator's step size. | `validate_resimulation.py` section 3 | **0 verdict flips** between 4 and 16 substeps; worst step-size shift **5.416102e-08** of a miss's own margin, over 47 misses |
| R-5.4 | Any shortcut that makes the reference affordable shall be off by default and verified, not assumed. | `validate_resimulation.py` section 4 | the early exit reproduces the full-horizon peak with **0.000000e+00** difference and **0** verdict flips over 228 misses, saving **0.158912** of evaluations; default `False` |
| R-5.5 | The one-at-a-time analysis's blind spot shall be measured. | `validate_resimulation.py` section 5 | the aggregate bound of eq (15) is never violated over 18 runs; the all-misses divergence exceeds the largest single miss on **6 of 18** runs, by up to **1.373314** |
| R-5.6 | The cost of the reference shall be a measurement, not an adjective. | `validate_certificate.py` section 7, `validate_learned_model.py` section 4 | **0.818047** and **0.869239** of plant evaluations saved by taking the certificate first |
| R-5.7 | The integrator's own floor shall be measured and added to the bound before any soundness comparison. | `validate_plant.py` section 7, `validate_certificate.py` section 4 | **2.396608e-09** in metric units at the declared 4 substeps, falling by 16.07, 16.03 and 16.02 per halving |

## 6. Clock drift and reset detection

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-6.1 | Drift shall be recovered against synthetic drift of KNOWN rate, over a declared sweep. | `validate_clock.py` section 1 | 9 rates from -20000 to +20000 ppm crossed with 4 jitter levels; **36 of 36** inside four of the estimator's own standard errors |
| R-6.2 | The worst absolute recovery error over the sweep shall be reported. | the same | **9.059894 ppm** against injected rates up to 20000 ppm |
| R-6.3 | The clock verdict shall agree with the declared allowance at every sweep point. | the same | **36 of 36** |
| R-6.4 | Reset detection shall be scored against the generator's own reset indices, not against the detector's output. | `validate_clock.py` section 2 | recall **1.000000**, precision **1.000000**, 24 injected resets, exact index |
| R-6.5 | The false-positive rate on reset-free series shall be reported separately. | the same | **0 detections** on 4 reset-free series, one at each jitter level |
| R-6.6 | The detector's floor shall be MEASURED and a reset below it shown to be missed. | `validate_clock.py` section 3 | largest missed **9e-05 s**, smallest detected **0.00011 s**; the control that a sub-floor reset is missed fires |
| R-6.7 | The two rate estimators shall be shown to fail differently. | `validate_clock.py` section 4 | at 40 of 201 timestamps corrupted, Theil-Sen error **0.0421 ppm** against least squares **605.9140 ppm** |
| R-6.8 | The effect of an uncorrected clock on the MISS SET shall be measured, not asserted to be negligible. | `validate_clock.py` section 5 | **28 misses** wrongly gained or lost at 20000 ppm over 12 runs; **0** after the correction |
| R-6.9 | A detected reset shall make the run undecidable, whatever the fitted rates look like. | `validate_clock.py` section 5, `validate_negative_controls.py` section 3, `tests/test_verdict.py` | satisfied; the control fires |

## 7. The learned component, and the comparison

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-7.1 | Both baselines shall be implemented and validated BEFORE the learned component, and the ordering shall be visible in the repository. | `validate_certificate.py` and `validate_resimulation.py` precede `validate_learned_model.py`; the labels in the learned study come from baseline 2 | satisfied |
| R-7.2 | The learned component shall be given every feature the certificate uses. | `src/hilverdict/features.py` | `overrun_s` and `input_step_rad` are features 2 and 4 of 14 |
| R-7.3 | The train/test split shall be by RUN, never by miss. | `validate_learned_model.py` section 1 | 72 training runs and 45 test runs, 0 shared seeds |
| R-7.4 | Every learned threshold shall be chosen on the TRAINING runs and applied unchanged to the test runs. | `validate_learned_model.py` section 2 | satisfied; the leaked oracle threshold is retained in its own row, labelled |
| R-7.5 | The scoring quantity shall be recall of consequential misses, reported with an exact interval. | the same | Clopper-Pearson on every row |
| R-7.6 | Accuracy shall not be reported. | the whole repository | satisfied |
| R-7.7 | There shall be no hyperparameter search. | `src/hilverdict/learned.py` | declared `ClassifierSpec`; no search anywhere |
| R-7.8 | The model shall expose an uncertainty or confidence output, not a point estimate. | `tests/test_learned.py`, `src/hilverdict/learned.py` | `predict_with_uncertainty` returns probability, decision, signed margin and an abstention flag; there is no point-prediction method on the class |
| R-7.9 | Calibration shall be measured. | `validate_learned_model.py` section 5 | Brier **0.010345** against a base rate of **0.052763**; worst bin gap **0.693204** |
| R-7.10 | The comparison's outcome shall be reported as measured, whichever way it came out. | `validate_learned_model.py` section 2 | certificate recall **1.0000**, 0 of 47 dismissed; learned recall **0.7234**, **13 of 47 dismissed**. The analytic bound wins outright |
| R-7.11 | The stability of the learned recall shall be measured against the certificate's. | `validate_learned_model.py` section 2b | learned recall **0.588235 to 0.812500** across three folds at one threshold; certificate **1.000000** on all three |
| R-7.12 | The residual undecided set shall be tested as the learned component's only defensible role. | `validate_learned_model.py` section 3 | on the test residual the model dismisses **11 of 47** consequential misses at the train-chosen threshold and saves **0.000000** at no cost in recall |
| R-7.13 | Two trivial baselines shall be reported, and the certificate shall beat both. | `validate_learned_model.py` section 2 | flag-everything: flag rate 1.0000; overrun threshold: 0.9574 recall at flag rate 0.6445; certificate 1.0000 at 0.1153 |

## 8. Negative controls

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-8.1 | Every control shall be anchored on the truth — the re-simulated divergence, the injected clock behaviour, the declared refusal condition or a shuffled label set — and never on an output of the thing being tested. | `validate_negative_controls.py` | satisfied for all 23 |
| R-8.2 | Every control shall be asserted to FIRE or NOT to fire as declared, and the count reported. | the same | **23 declared, 14 must-fire and all 14 fired, 9 must-not-fire and none fired** |
| R-8.3 | A control that cannot fire shall be replaced, not accepted. | `validate_negative_controls.py` sections 2 and 5 | two were: the learned-pipeline controls were saturated at a fixed 0.25 flag rate and now compare the flag rate needed for recall 1; the incidence-validity control cleared the 10 deg limit by 0.0059 deg on the cubic plant and now uses the linear variant, reaching 14.1460 deg |
| R-8.4 | A must-fire control shall corrupt an INPUT the derivation depends on, by a factor large enough that the derivation cannot survive it. | `validate_negative_controls.py` section 1 | input step /10: 142 violations; overrun /10: 159; input gain /4: 119; a 10x looser metric: 111 |
| R-8.5 | A measurement that could not be made into a control shall be reported as a measurement. | `validate_negative_controls.py` section 1 | a 10 deg operating-point error gives 0 violations and worst ratio 0.704993; reported, with the direction that IS dangerous named |

## 9. Hygiene, reproducibility and the release gate

| ID | Requirement | Verified by | Measured result |
|----|-------------|-------------|-----------------|
| R-9.1 | Every validation script shall exit zero from the filesystem root, the system temporary directory, the product root and `validation/`. | `probe_working_directories.py` | **0 non-zero exit statuses in 44 runs**; examples **0 of 14** |
| R-9.2 | No committed file shall contain an absolute host path. | `validate_cli.py` section 4 | **0 offenders in 73 text files**, with the scanner's patterns assembled from `os.sep` and its own output deleted before the scan |
| R-9.3 | Running the validation suite shall add no cache directory and no bytecode file to the tree. | `validate_cli.py` section 3a | **0 added** |
| R-9.4 | The test count shall be taken from the junit XML and cross-checked against a count of `testcase` elements. | `validate_approx_audit.py` sections 2 and 3, `.github/workflows/tests.yml` | **518 declared, 518 counted**, 0 failures, 0 errors, 0 skipped |
| R-9.5 | No assertion shall depend on the hidden 1e-12 absolute default of `pytest.approx`. | `validate_approx_audit.py` | **0 of 518**; 3 rest on an absolute bound and each is named and accounted for |
| R-9.6 | The CLI shall exit 0 on every documented invocation and 2 with an actionable message on an invalid one. | `validate_cli.py` sections 1 and 2 | 11 invocations exit 0; 5 invalid invocations exit 2 |
| R-9.7 | Lint shall be clean under both ruff installations present in the measured environment. | the reproduction commands in `README.md` | clean under 0.16.9 and 0.17.3 |
| R-9.8 | The tracked-file hygiene check shall report when it is vacuous rather than counting as evidence. | `validate_cli.py` section 3b | examined **0 tracked files** in this environment; recorded as vacuous in `VALIDATION.md` and `README.md` |

**R-9.8 is partially satisfied.** The check is correct and will mean something
once the product directory is tracked; in the environment it was run in, it
examined nothing. That is recorded rather than presented as a pass.

## 10. What is deliberately out of scope

| ID | Statement | Where it is stated |
|----|-----------|--------------------|
| R-10.1 | The product never talks to a device. It analyses a log. | `README.md`, `src/hilverdict/timeline.py` |
| R-10.2 | The miss model does not cascade; the cascade count is reported beside the direct one on every run. | `src/hilverdict/timeline.py`, every run verdict |
| R-10.3 | The analysis is one miss at a time; interactions are covered only by the superposition bound. | `src/hilverdict/timeline.py`, `validate_resimulation.py` section 5 |
| R-10.4 | Nothing here is measured against hardware, and nothing establishes behaviour on a real log. | `README.md`, `DATASET_CARD.md`, `MODEL_CARD.md` |
