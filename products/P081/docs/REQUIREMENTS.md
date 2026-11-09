# Requirements — twindiverge 0.1.0

Validation level 3. Status TESTING. Research grade: not flight-qualified, not
certified, not approved for operational aerospace use.

Every requirement below names the artefact that satisfies it and, where it is a
numerical claim, the measured value and the file it was measured in. A
requirement with no evidence is a wish.

## 1. Scope and purpose

**R1.1** The product shall attribute a digital-twin-versus-reality residual to a
subset of the twin's parameters.
*Satisfied by* `twindiverge.attribution`, `twindiverge.pipeline`.

**R1.2** The product shall report which parameters are structurally
unidentifiable from the supplied telemetry set **before** attributing anything
to the rest, and shall not return a per-parameter estimate for a parameter it
has refused.
*Satisfied by* `twindiverge.identifiability` and the order of operations in
`twindiverge.pipeline.analyse_case`. *Evidence:* on the reference set
`Attribution.parameter_estimate` is `nan` for all five parameters;
`tests/test_attribution.py::test_a_refused_parameter_gets_no_number`.

**R1.3** Refusal shall be a reported outcome, not an error path. No exception
shall be raised because a parameter is unidentifiable.
*Satisfied by* the `REFUSED` verdict and the `CaseAnalysis.gated_answer`
property. *Evidence:* `validation/validate_attribution_output.txt` part 5, which
runs one case through all five telemetry sets and tabulates the refusals.

**R1.4** The product shall not claim global structural identifiability, and
shall say so where a reader might assume it.
*Satisfied by* the module docstring of `twindiverge.identifiability`, README.md
under "Who it is not for" and under Limitations.

## 2. The plant

**R2.1** The declared plant shall be stated as an equation with its source,
units, assumptions and validity range, and its coefficient values shall be
identified as declared rather than measured.
*Satisfied by* the module docstring of `twindiverge.plant`, equation (1).

**R2.2** The plant shall raise `ValueError` with an actionable message when a
trajectory leaves its declared validity window.
*Evidence:* `tests/test_plant.py::test_a_trajectory_leaving_the_validity_window_raises`
and `::test_initial_rate_outside_the_window_raises_before_a_single_step`.

**R2.3** The integrator shall be scored against a reference derived from
mathematics it does not use.
*Satisfied by* `twindiverge.reference`, equations (20) and (21), a closed-form
solution of equation (18). *Evidence:* worst relative difference
**6.615096e-15** over three campaigns, `validation/validate_plant_output.txt`.

**R2.4** The integrator's convergence order shall be measured, not assumed.
*Evidence:* measured orders **4.0201 to 4.2425**,
`validation/validate_plant_output.txt` part 2. The declared plant's truncation
error at its declared step is already below rounding, so the order is measured
on a declared stiff parameter set named in the output as a numerical exercise.

**R2.5** The sensitivity matrix shall be computed from the forward sensitivity
equations and shall be checked against an independent route.
*Evidence:* against a complex-step derivative from an independently written
complex-arithmetic integrator, worst relative difference **4.304265e-15** over
15 columns; the central difference is reported beside it against its own
measured cancellation floor, worst disagreement **12.887** times that floor.

**R2.6** The unidentifiable direction of the rate channel shall be derived
analytically, on paper, and the derivation shall be independent of the rank
computation it is compared against.
*Satisfied by* identities A and B in the module docstring of
`twindiverge.plant` and by `identifiability.analytic_null_basis`, which is
algebra and never touches a sensitivity matrix.

## 3. Identifiability — baseline 1

**R3.1** The numerical rank shall be decided by a declared relative cutoff on
the singular values, and the distance from that cutoff to the nearest
singular-value ratio shall be reported.
*Evidence:* margins **5.6483, 5.7896, 4.4135, 4.5000 and 5.2952 decades** across
the five declared telemetry sets.

**R3.2** The Cramer-Rao bound for a parameter with a component in the null space
shall be `+inf`, and the pseudo-inverse diagonal that a naive implementation
reports instead shall be reported beside it under a name that says what it is
not.
*Satisfied by* equation (10)'s finiteness clause and
`IdentifiabilityReport.pinv_diagonal_not_a_bound`. *Evidence:*
`tests/test_regressions.py::test_the_pseudo_inverse_diagonal_is_not_the_cramer_rao_bound`,
and the coast set's claim of a relative standard error **below 1e-14** for a
parameter the excitation never excites: 1.888175e-16 in the committed output,
a value at the level of rounding that varies between runs, which is why the
test asserts a bound on it rather than the figure.

**R3.3** Near-singular directions shall be reported as identifiable parameter
combinations rather than as individual parameters.
*Satisfied by* equation (11) and `IdentifiabilityReport.combinations`.
*Evidence:* the four combinations of the reference set with bounds
**4.677542e-06**, **2.276973e-05**, **0.001224** and **0.073789**, measured to
span exactly the space of the four ratios to within **1.207418e-06 degrees**.

**R3.4** Practical identifiability shall be separated from structural, and the
difference shall be demonstrated rather than asserted.
*Satisfied by* the `REFUSED` / `WEAK` / `IDENTIFIABLE` verdicts. *Evidence:* the
record-length sweep — rank constant at 4, worst-direction bound falling by a
factor of **30979.4** — and the `rate_torque_short` set, which is structurally
full rank and practically unable to determine three of five parameters.

**R3.5** The computed null space shall agree with its derivation.
*Evidence:* worst principal angle **1.207418e-06 degrees** over all five
declared sets; dimensions agree on 5 of 5.

**R3.6** Nothing shall be computed from the normal equations, and the
consequence of forming them shall be reported.
*Evidence:* the reference set's Fisher information has a condition number of
**5.268523e+16** and a least eigenvalue of **-4.684327e-08** for a matrix that
is positive semi-definite by construction;
`tests/test_identifiability.py::test_the_fisher_matrix_is_reported_but_not_used`.

## 4. Attribution — baseline 2

**R4.1** A residual consistent with sensor noise shall be attributed to nothing.
*Satisfied by* the chi-square detection test at a declared level.
*Evidence:* detection rate **0.000000** on 40 pure-scale cases and on 40 null
cases in `validation/validate_negative_controls_output.txt` group D, and a
measured false-positive rate of **0.016667** against the declared **0.010000**
over 120 cases, 0.7340 standard errors away.

**R4.2** Estimates shall be reported per identifiable direction with confidence
intervals at a declared level, and their coverage shall be measured.
*Evidence:* marginal coverages **0.962500, 0.967500, 0.952500, 0.940000** over
400 realisations, worst deviation **1.6059** standard errors against the
declared four.

**R4.3** Where a single-parameter attribution is offered, the assumption it
rests on shall be declared and the family of perturbations it cannot be
distinguished from shall be printed.
*Satisfied by* `Attribution.equivalence_family` and the "THE ATTRIBUTION ABOVE
IS NOT UNIQUE" block of `report.render_attribution`. *Evidence:* a perturbation
and its partner give rate records agreeing to **2.340688e-16** relative over 120
test cases.

**R4.4** Where the specification names an estimator that measures weaker than
an available alternative, both shall be reported and the stronger one used.
*Evidence:* equation (14) and equation (15) tabulated on all five sets in
`validation/validate_attribution_output.txt` part 3; equation (15)'s margin runs
from **-0.025000** to **+0.000000**, and both are printed on every case.

## 5. Profile likelihood — baseline 3

**R5.1** A nonlinear profile likelihood shall be computed per parameter and
shall be the reference the linearised covariance is scored against.
*Satisfied by* `twindiverge.profile`, equation (17).

**R5.2** A flat profile shall override whatever the covariance claims, and the
override shall be a reported verdict.
*Satisfied by* `ProfileResult.overrides_covariance`. *Evidence:* **5 of 5**
parameter profiles flat at **7.385097e-10** in chi-square on the reference set while
the covariance claims finite half-widths for all five; **5 of 5** overrides.

**R5.3** Where the profile closes an interval, the covariance shall be scored
against it.
*Evidence:* ratios **0.9972, 0.9977, 1.0024, 1.1622** on the two-channel set.

**R5.4** A profile truncated by the declared inner box shall have its limits
suppressed rather than quoted.
*Satisfied by* `ProfileResult.interval_truncated` and the `box` column of
`report.render_profiles`. *Evidence:* three of five profiles on
`rate_torque_short` are truncated and quote no interval.

## 6. The learned component

**R6.1** The classical baselines shall be implemented and validated against the
simulation **before** the learned model is benchmarked.
*Evidence:* `validate_plant.py`, `validate_identifiability.py`,
`validate_attribution.py`, `validate_profile.py` and
`validate_independent_bound.py`, all of which pass with 0 failures before
`validate_learned_model.py` runs.

**R6.2** The learned model shall be benchmarked against the baselines on the
same held-out data.
*Evidence:* `validation/validate_learned_model_output.txt` part 2, five methods
on five populations of the same cases.

**R6.3** The learned model shall expose an uncertainty or confidence output and
shall not expose a bare point prediction.
*Satisfied by* `Prediction` and the absence of any label-only method.
*Evidence:* `tests/test_learned.py::test_there_is_no_bare_label_method`.

**R6.4** The learned model's behaviour inside the unidentifiable subspace shall
be measured and reported separately from its accuracy.
*Evidence:* `validation/validate_learned_model_output.txt` part 3 — mean
confidence **0.993426**, accuracy **0.000000**, confident-and-wrong
**0.991667**, ambiguity declared **0.000000**, against the analytic path's
refusal rate of **1.000000**.

**R6.5** The learned model shall have no decision threshold selectable on the
test population.
*Satisfied by* taking the label as the argmax of the probability vector.
*Evidence:*
`tests/test_regressions.py::test_the_learned_threshold_is_never_chosen_on_the_test_population`.

**R6.6** Synthetic data shall be generated by committed scripts, deterministic
from fixed seeds, and shall not be committed as data.
*Satisfied by* `twindiverge.synthetic`. *Evidence:* no data file in the
repository; `tests/test_synthetic.py::test_cases_are_a_function_of_the_seed_list_alone`.

**R6.7** The computation budget shall be measured.
*Evidence:* **0.000475 s** per case for the analytic attribution and
**0.000929 s** for the learned prediction, with the analytic setup at
**0.079955 s** once per telemetry set and the fit at **1.704589 s** once per
training set. These are wall-clock figures on two shared cores and are the only
unstable numbers in this repository; the measured learned-to-analytic ratio has
ranged from 1.96 to 2.16 across runs of the same code.

**R6.8** Any training or Monte Carlo run shall finish in under three minutes on
two cores.
*Evidence:* the whole learned study runs in **4.197 s**; the longest validation
script other than the working-directory probe is
`validate_approx_audit.py` at about 49 s.

## 7. Checks against independent mathematics

**R7.1** At least one check shall compare against a bound derived from
mathematics the implementation does not use.
*Satisfied by* `validate_independent_bound.py`: the Cramer-Rao bound against
the measured scatter of repeated nonlinear fits. *Evidence:* **0 violations in 4
and in 5 directions**; the matrix form of the inequality holds with a least
eigenvalue of **-1.171462e-09** against a sampling tolerance of
**-7.066074e-03**.

**R7.2** The slack on such a check shall follow the conditioning of the bound
expression and shall not be an absolute constant.
*Satisfied by* a sampling term `CONTROL_SIGMA / sqrt(2(n-1))` and a conditioning
term `cond(A)·eps·bound`, both printed with their sizes. *Evidence:* the
sampling term is **0.648886** of the bound and the conditioning term is at most
**3.604353e-12** of it, so a reader can see which one is doing the work.

**R7.3** The check shall be shown to be capable of failing.
*Evidence:* two controls per set, one rescaling the spread to 0.25 of the bound
(fires in **4 of 4** and **5 of 5** directions) and one to 2.00 of it (fires in
**0 of 4** and **0 of 5**).

## 8. Negative controls

**R8.1** Every check shall have at least one negative control, anchored on a
value known from the derivation and not on an output.
*Satisfied by* `validate_negative_controls.py`, groups A to J.

**R8.2** Each control shall be asserted to fire, or not to fire, as its own
reported check.
*Evidence:* **29 controls declared, 17 must fire and all 17 did, 12 must not and
none did**, 0 failures.

**R8.3** A control's criterion shall not be a nominal confidence level where
that level would make it fire by construction.
*Satisfied by* the declared four-sigma criterion in
`validation/_bootstrap.CONTROL_SIGMA`. *Evidence:* the coverage control's
realisation count was raised from 200 to 400 during the build precisely because
a saturated coverage is only 3.2444 standard errors from nominal at 200, which
is recorded in `CHANGELOG.md`.

## 9. Engineering hygiene

**R9.1** Every validation and example script shall run from any working
directory, including the release gate's.
*Evidence:* `validation/probe_working_directories_output.txt` — **0 non-zero
exit statuses in 44 + 14 runs** from four and two directories respectively.

**R9.2** No committed file shall contain an absolute host path.
*Evidence:* `validate_cli.py` part 4 — **0 offenders in 73 files scanned**, with
the patterns assembled from `os.sep` and the scanner's own output deleted before
the scan.

**R9.3** Running a validation script shall add no cache directory and no
bytecode file to the tree, and git shall track none.
*Evidence:* `validate_cli.py` part 3 — **none added**, and `git ls-files`
examines **81 tracked files with 0 offenders**. The build measured this claim
against an uncommitted tree, where it examined 0 files and was recorded as
vacuous; committing the product made it real. Release-gate check 9 enforces it
independently.

**R9.4** Test counts shall be taken from junit XML and cross-checked.
*Evidence:* **254 declared, 254 counted** in both the strict and zero runs of
`validate_approx_audit.py`, and the same cross-check in
`.github/workflows/tests.yml`.

**R9.5** Every `pytest.approx` call passing `rel=` shall pass an explicit
`abs=`, and the dependence on the hidden default shall be audited rather than
assumed.
*Evidence:* **81** calls carry `rel=`, **0** of them without an explicit `abs=`,
**0 of 254** assertions depend on the default, and **2 of 254** rest on an
absolute bound — both comparisons against an exact zero, named in the output.

**R9.6** The audit's report shall be deterministic across runs.
*Satisfied by* a Hypothesis profile with `derandomize=True` and `database=None`
in `tests/conftest.py`, and `HYPOTHESIS_STORAGE_DIRECTORY` outside the
repository for every spawned subprocess.

**R9.7** No boolean produced by a strict threshold comparison shall be asserted
for equality; the underlying statistic shall be asserted to tolerance instead.
*Satisfied throughout.* *Examples:*
`tests/test_identifiability.py::test_rank_and_its_margin` asserts the
singular-value ratios either side of the cutoff and lets the rank follow;
`tests/test_attribution.py::test_the_direction_significance_flag_follows_its_z_score`
exempts a z-score within tolerance of the critical value;
`tests/test_learned.py::test_the_abstention_flag_follows_the_confidence` does
the same for the confidence.

**R9.8** Subprocesses shall be given a temporary `HOME` and `MPLCONFIGDIR`
outside the product and `PYTHONDONTWRITEBYTECODE=1`.
*Satisfied by* `validation/_bootstrap.subprocess_environment`.

**R9.9** Every dependency shall be bounded on both sides, and what was not
measured shall be stated.
*Satisfied by* `pyproject.toml`, whose comment names the six ranges bounded at a
single measured version and the one — `ruff` — measured at both ends.

**R9.10** Ruff shall be clean under both installed versions.
*Evidence:* `ruff check src/ tests/ examples/ validation/` passes under 0.16.9
on the path and 0.17.4 as a Python module.

## 10. Known limitations at 0.1.0

**L10.1** Everything is local and first-order. No global structural
identifiability, no global sensitivity analysis.

**L10.2** One declared fictional plant, single-state, valid for one sign of
rotation inside 20 to 400 rad/s. A wheel zero crossing is outside the model.

**L10.3** The noise is independent between samples by assumption, with a
declared known standard deviation. A differentiated-encoder tachometer would
violate that and would change every interval reported here.

**L10.4** Baseline 2's linearisation error is second order and exceeds its own
statistical interval in the two best-determined directions at an 8 per cent
perturbation: **1.081920e-03** against a standard error of **4.677542e-06**.
Measured, with halving orders **1.9958 to 1.9996**.

**L10.5** The profile likelihood is a local optimiser inside a declared box. A
second minimum elsewhere would not be found.

**L10.6** The sparsity assumption behind equation (15) is not testable from the
telemetry.

**L10.7** Both ends of every dependency range were run for `ruff` only; the
others are bounded at a single measured version because the build environment is
a shared pre-pinned interpreter this product may not modify. A pin below a
future security release is a real cost, not a free safety measure.

**L10.8** The `git ls-files` hygiene check was vacuous at build time, against
an uncommitted tree, and is not any more: 81 files examined, 0 offenders.

**L10.9** The learned component's confidence is a model output and knows nothing
about the rank of the sensitivity matrix. Measuring what that costs is the
product's headline, not a defect to be fixed inside the model.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.
