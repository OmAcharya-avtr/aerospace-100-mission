# Validation evidence — twindiverge 0.1.0

**Validation level 3.** Status: TESTING. Research grade: not flight-qualified,
not certified, not approved for operational aerospace use.

Every number on this page was produced by a script in this directory, in one
session, on the environment recorded in each script's own output file. The
output files are committed beside the scripts. Nothing here is copied from a
previous run, and a check that failed is reported as failed.

## Environment

| item | value |
| --- | --- |
| Python | 3.13.18 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| pytest | 9.1.1 |
| hypothesis | 6.170.15 |
| ruff | 0.16.9 on the path and 0.17.4 as a Python module; clean under both |
| CPU cores | 2 |
| `torch` | not installed, and nothing here is compared against it |
| `sympy` | not installed; named in README.md as the symbolic route this product does not take |

## How to reproduce everything

```bash
python -m pytest tests/ -q --junit-xml=junit.xml
python validation/validate_plant.py
python validation/validate_identifiability.py
python validation/validate_attribution.py
python validation/validate_profile.py
python validation/validate_independent_bound.py
python validation/validate_learned_model.py
python validation/validate_negative_controls.py
python validation/validate_approx_audit.py
python validation/validate_examples.py
python validation/validate_cli.py
python validation/worked_example.py
python validation/probe_working_directories.py
```

Every script writes its raw output to `validation/<script-name>_output.txt`.
`probe_working_directories.py` runs all of the others from four working
directories and takes about seven minutes; the rest take between one second and
fifty seconds each, about 196 s in total.

## Test suite

Counted from the junit XML, never from pytest's stdout, because a suite that
collects zero tests exits 0 and prints success.

| quantity | value |
| --- | --- |
| `testsuite` element, `tests` | 254 |
| `testcase` elements counted | 254 |
| failures | 0 |
| errors | 0 |
| skipped | 0 |
| wall clock | 23.65 s |

The two counts are cross-checked against each other by
`.github/workflows/tests.yml` and by `validation/validate_approx_audit.py`,
which parses the same XML.

## 1. The declared plant — `validate_plant.py`

| check | reference | result | tolerance |
| --- | --- | --- | --- |
| integrator against the exact closed form of equation (18) | `twindiverge.reference`, equations (20) and (21); closed-form algebra the integrator never uses | worst relative difference **6.615096e-15** over 3 campaigns | 1e-13 |
| convergence order | Richardson extrapolation against the same closed form | measured orders **4.0201 to 4.2425** | between 3.8 and 4.6 |
| identity A, equation (6): the five rate sensitivities sum to zero | derived on paper in `twindiverge.plant` | worst relative residual **3.802603e-15** over 15 campaign-and-parameter-point combinations | 1e-11 |
| identity B, equation (7): the five torque sensitivities sum to the torque | same derivation | worst relative residual **4.471639e-16** | 1e-10 |
| forward sensitivities against a complex-step derivative | an independent complex-arithmetic integrator written out in `validate_plant.py` | worst relative difference **4.304265e-15** over 15 columns | 1e-12 |
| the same against a central difference | finite differences at a 1e-5 logarithmic step | worst disagreement is **12.887 times** the difference's own cancellation floor `eps·max|omega|/(h·|s_j|)` | 20 times |
| an unexcited parameter is exactly zero on all three routes | the corollary to identity B | `k_t` on the coast campaign: **exactly 0.0** by the variational equation, the complex step and the central difference | exact |
| the sample at the current switch | closed form | relative difference **1.079957e-15** | 1e-14 |

The last row is a defect this build found and fixed. With the zero-order-hold
current evaluated at each Runge-Kutta stage's own time rather than once per
step, the error at the end of a 20 s driven segment was **4.102564e-02 rad/s**
at a 0.1 s step and the measured convergence order was **1.0000** instead of 4.
It is pinned in `tests/test_regressions.py`.

## 2. Local identifiability — `validate_identifiability.py`

| telemetry set | channels | length | rows | rank | margin | condition | REFUSED | WEAK |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rate_full` | rate | 200 s | 2001 | **4 of 5** | 5.6483 decades | 1.577511e+04 | all five | none |
| `rate_torque_full` | rate + torque | 200 s | 4002 | **5 of 5** | 5.7896 | 1.623256e+04 | none | none |
| `rate_short` | rate | 20 s | 201 | **4 of 5** | 4.4135 | 3.859124e+05 | all five | none |
| `rate_torque_short` | rate + torque | 20 s | 402 | **5 of 5** | 4.5000 | 3.162214e+05 | none | `T_c`, `b`, `c_d` |
| `rate_coast` | rate | 60 s | 601 | **3 of 5** | 5.2952 | 5.067366e+04 | all five | none |

"Margin" is the base-10 distance from the declared rank cutoff of 1e-10 to the
nearest singular-value ratio. Every verdict has at least four decades of
margin, so none of them is an artefact of the cutoff.

**The central check.** The null space derived on paper — algebra from identities
A and B and their corollary, never touching a sensitivity matrix — against the
null space computed as the right singular vectors of equation (8):

| set | derived dimension | computed dimension | worst principal angle |
| --- | --- | --- | --- |
| `rate_full` | 1 | 1 | **0.000000e+00 deg** |
| `rate_torque_full` | 0 | 0 | both empty |
| `rate_short` | 1 | 1 | **0.000000e+00 deg** |
| `rate_torque_short` | 0 | 0 | both empty |
| `rate_coast` | 2 | 2 | **1.207418e-06 deg** |

The identifiable subspace of `rate_full` is measured to BE the space of the
four ratios `k_t/J`, `T_c/J`, `b/J`, `c_d/J`, with principal angles
**1.207418e-06**, **8.537736e-07**, **0.000000e+00** and **0.000000e+00**
degrees — which is what equation (6) predicts, since those four ratios are
exactly what equation (1) depends on after division by `J`.

Removing a perturbation's null projection changes no rate record: worst
relative difference **5.631127e-16** over 13 combinations of perturbation and
telemetry set.

**Structural against practical**, by record length, driven fraction held at one
fifth:

| length | samples | rank | least singular-value ratio | worst identifiable direction's bound |
| --- | --- | --- | --- | --- |
| 10 s | 101 | 4 | 5.176139e-17 | 2285.926965 |
| 20 s | 201 | 4 | 6.237874e-17 | 204.779634 |
| 40 s | 401 | 4 | 8.540526e-17 | 18.540781 |
| 80 s | 801 | 4 | 1.523425e-16 | 1.708846 |
| 120 s | 1201 | 4 | 1.722821e-16 | 0.427029 |
| 200 s | 2001 | 4 | 2.247702e-16 | 0.073789 |

The rank and the refusal list do not move. The bound falls by a factor of
**30979.4**. That is the operational content of the distinction between
structural and practical identifiability (Raue et al. 2009).

## 3. Attribution — `validate_attribution.py`

| check | reference | result | criterion |
| --- | --- | --- | --- |
| coverage of the identifiable directions' 95 per cent intervals | the declared level, over 400 noise realisations | worst deviation **1.6059** standard errors | declared 4 sigma |
| an equivalence pair gives the same noiseless rate record | equation (6) | worst relative difference **2.340688e-16** over 120 pairs | 1e-13 |
| a pair's observations | the same, with noise | worst difference **2.842171e-14 rad/s = 2.000 units in the last place**; 97 of 120 pairs bit-identical | 10 units |
| false-positive rate of the detection test, pure scale | declared level 0.99 | **0.016667** measured against 0.010000 expected, 0.7340 sigmas | declared 4 sigma |
| false-positive rate, nothing moved | same | **0.016667**, identical to the row above because the observations are the same | declared 4 sigma |
| detection of an ordinary single-parameter fault | — | **0.975000** over 120 cases on the 20 s record | at least 0.95 |
| every rate-only set refuses and declares its family | — | **3 of 3** refuse; 2 of 2 two-channel sets answer | all |

**Reported as found: equation (15) is not uniformly better than equation (14).**
The specification names the first-order sensitivity attribution of equation (14)
as baseline 2's attributor; the constrained maximum likelihood of equation (15)
is stronger on the reference set and weaker on two others. Both are printed on
every case by the CLI and by `render_attribution`, and neither is dropped:

| set | equation (14) | equation (15) | routes agree | difference |
| --- | --- | --- | --- | --- |
| `rate_full` | 1.000000 | 1.000000 | 1.000000 | +0.000000 |
| `rate_torque_full` | 1.000000 | 1.000000 | 1.000000 | +0.000000 |
| `rate_short` | 0.758333 | 0.741667 | 0.983333 | **-0.016667** |
| `rate_torque_short` | 0.750000 | 0.725000 | 0.975000 | **-0.025000** |
| `rate_coast` | 0.691667 | 0.691667 | 1.000000 | +0.000000 |

## 4. Profile likelihood — `validate_profile.py`

**The reference rate-only set: five of five profiles flat, five of five
overrides.**

| direction | grid half-width | max delta chi-square | shape | Wald half-width the covariance claims | verdict |
| --- | --- | --- | --- | --- | --- |
| log J | 0.071215 | 4.679350e-10 | FLAT | 0.017804 | **OVERRIDES** |
| log k_t | 0.070965 | 2.935394e-10 | FLAT | 0.017741 | **OVERRIDES** |
| log T_c | 0.028262 | 7.385097e-10 | FLAT | 0.007065 | **OVERRIDES** |
| log b | 0.100000 | 3.142304e-10 | FLAT | 0.085230 | **OVERRIDES** |
| log c_d | 0.100000 | 5.661605e-10 | FLAT | 0.113911 | **OVERRIDES** |

The 95 per cent threshold on one degree of freedom is 3.841459. The largest
value any of these five profiles reaches is 7.385097e-10, nine and a half orders
of magnitude below it. The linearised covariance nevertheless reports half-widths
between 0.7 and 11 per cent for the same five parameters. The profile is the
reference and the covariance is wrong about all five — and baseline 1's rank
computation, which shares no implementation with the profile, refuses the same
five.

**The two-channel set: the covariance is validated where the profile closes.**
Profile half-width divided by Wald half-width: **0.9972**, **0.9977**,
**1.0024**, **1.1622** for `log J`, `log k_t`, `log T_c`, `log b`. The
`log c_d` profile does not cross the threshold inside its grid and no ratio is
quoted for it.

The profile of the null direction itself is flat (max delta chi-square
**7.730705e-11**) and is **not** counted as an override, because the covariance
makes no claim about it.

On the 20 s rate-only record the chi-square valley crosses the threshold in no
identifiable direction at all (worst maximum **2.651745e-11** against
3.841459), while the linearised bound claims the best-determined combination to
5.5e-05 relative. The profile is the reference there too.

## 5. The independent bound — `validate_independent_bound.py`

The Cramer-Rao bound of Rao 1945 and Cramer 1946, computed from the Fisher
information of equation (9), against the measured scatter of estimators that
never see it. 20 independent noise realisations of the declared truth per set.
The slack is the declared four standard errors of a sample standard deviation,
`4/sqrt(2(n-1)) = 0.648886`, plus a conditioning term `cond(A)·eps` which is
reported and is at most 3.604353e-12 of the bound -- eleven orders of magnitude
smaller, so the sampling term is the one doing the work.

| set | estimator | violations | ratio of measured spread to bound |
| --- | --- | --- | --- |
| `rate_full` | linear, equation (12) | **0 of 4** | 1.002398 to 1.209669 |
| `rate_full` | nonlinear fit | **0 of 4** | 1.043121 to 57.008044 |
| `rate_torque_full` | linear, equation (12) | **0 of 5** | 0.745312 to 1.212215 |
| `rate_torque_full` | nonlinear fit | **0 of 5** | 1.070333 to 56.066860 |

The linear estimator's ratios sit on the bound, so the bound is tight as well as
true. The nonlinear global fit's are far above it in the well-determined
directions, and the reason is reported rather than smoothed over: on a problem
this ill-conditioned the global optimum wanders along a curved valley and is
biased, and the plain Cramer-Rao inequality bounds unbiased estimators only.

The matrix form of the inequality — empirical covariance minus Cramer-Rao
covariance positive semi-definite — holds in all four cases; least eigenvalues
**-1.171462e-09**, **-2.581019e-08**, **-2.968097e-07**, **-2.923864e-07**
against sampling tolerances of **-7.066074e-03** and **-7.480666e-03**.

Both controls behave: a spread rescaled to 0.25 of the bound violates the
criterion in **4 of 4** and **5 of 5** directions, and one rescaled to 2.00 of
the bound violates it in **0 of 4** and **0 of 5**.

## 6. The learned component — `validate_learned_model.py`

Telemetry set `rate_full`. 240 training cases, all of them single-parameter;
120 test cases per population from a disjoint seed set; 120
unidentifiable-subspace cases from a third disjoint set.

### Inside the unidentifiable subspace — the headline

Population `equivalent`: each case is the partner of a single-parameter fault
under equation (6), so the residual is large and easily detected and **no
single-parameter answer can be correct**.

| method | accuracy | mean confidence | confident and wrong | abstention or refusal rate | ambiguity declared |
| --- | --- | --- | --- | --- | --- |
| analytic, gated by baseline 1 | **1.000000** | n/a | 0.000000 | **1.000000** | **1.000000** |
| analytic, equation (14) | 0.000000 | 0.998334 | 1.000000 | 0.000000 | 1.000000 |
| analytic, equation (15) | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| **learned, five classes** | **0.000000** | **0.993426** | **0.991667** | 0.008333 | **0.000000** |
| learned, six classes (ablation) | 0.008333 | 0.994042 | 0.991667 | 0.008333 | 0.000000 |

The exact Clopper-Pearson interval on the learned classifier's accuracy here is
[0.000000, 0.030273] and on its confident-and-wrong rate [0.954444, 0.999789].
Its Brier score on this population is **0.993340**, against 0.006568 on
ordinary single-parameter faults.

The learned classifier returns a **confident wrong attribution exactly where
the analytic method correctly refuses to answer**. That is the expected and the
most valuable result in this product. Nothing was tuned to soften it, and the
classifier has no threshold that could have been: its label is the argmax of its
own probability vector.

The pair test makes the mechanism exact. Each `equivalent` case shares its noise
seed with its `single` partner and the two observations differ by at most two
units in the last place, so any classifier must answer both the same way:
measured, the learned classifier agrees with itself on **120 of 120** pairs and
the analytic path agrees with itself on **120 of 120** — and the analytic answer
is REFUSED. The learned confidence over both members of all 120 pairs has mean
**0.993426** and least **0.211119**. The two members of a pair have different true causes, so a
method that agrees with itself on a pair is wrong on one member of it whatever
it answers. The difference between the two paths is not the answer; it is that
one of them says so.

### Outside it — the baseline comparison

| population | method | accuracy | exact interval | mean confidence | confident and wrong |
| --- | --- | --- | --- | --- | --- |
| `single` | analytic, equation (14) | **1.000000** | [0.969727, 1.000000] | 0.998334 | 0.000000 |
| `single` | analytic, equation (15) | **1.000000** | [0.969727, 1.000000] | 1.000000 | 0.000000 |
| `single` | learned, five classes | 0.991667 | [0.954444, 0.999789] | 0.993426 | 0.000000 |
| `weak` | analytic, equation (14) | **0.941667** | [0.883502, 0.976228] | 0.931724 | 0.050000 |
| `weak` | analytic, equation (15) | **0.941667** | [0.883502, 0.976228] | 0.950976 | 0.041667 |
| `weak` | learned, five classes | 0.641667 | [0.549039, 0.727122] | 0.900976 | 0.275000 |

`weak` holds the same single-parameter faults a tenth the size, wholly outside
the learned component's training distribution. **The analytic attributor beats
the learned classifier by 0.008333 on `single` and by 0.300000 on `weak`**, and
the learned classifier's confident-error rate on `weak` is 0.275000 against the
analytic route's 0.041667. A baseline win is research evidence and is not
removed, softened or retuned away.

### On a residual that is pure noise

Populations `scale` (all five parameters multiplied by the same factor) and
`none` (nothing moved) are the same observations to within the last bit, and
every method scores identically on them, which is itself a check.

| method | accuracy | mean confidence | confident and wrong | abstention or refusal rate |
| --- | --- | --- | --- | --- |
| analytic, gated by baseline 1 | **1.000000** | n/a | 0.000000 | **1.000000** |
| analytic, equation (14) | 0.000000 | 0.000636 | 0.000000 | 0.000000 |
| analytic, equation (15) | **1.000000** | 0.246418 | 0.000000 | 1.000000 |
| learned, five classes | 0.000000 | **0.522102** | **0.991667** | 0.008333 |
| learned, six classes (ablation) | **1.000000** | 1.000000 | 0.000000 | 0.000000 |

### What the refusal class costs

| population | five-class accuracy | six-class accuracy | difference | five-class Brier | six-class Brier |
| --- | --- | --- | --- | --- | --- |
| `single` | 0.991667 | 0.991667 | +0.000000 | 0.006568 | 0.008441 |
| `weak` | 0.641667 | 0.325000 | **-0.316667** | 0.579620 | 1.197593 |
| `equivalent` | 0.000000 | 0.008333 | +0.008333 | 0.993340 | 0.992325 |
| `scale` | 0.000000 | 1.000000 | +1.000000 | 0.499721 | 1.807904e-16 |
| `none` | 0.000000 | 1.000000 | +1.000000 | 0.499721 | 1.807904e-16 |

A refusal class trained on pure-scale examples generalises to pure-scale cases
and **not** to the ambiguity case, where the residual is large and only the
attribution is impossible. Giving a classifier the option of refusing does not
give it the ability to tell when to — and here it costs a third of the accuracy
on small faults, because the refusal class absorbs them.

### Computation budget

| item | seconds |
| --- | --- |
| analytic setup, once per telemetry set | 0.079955 |
| learned fit, once per training set, both variants (240 and 480 cases) | 1.704589 |
| analytic attribution per case, mean of 40 | 0.000475 |
| learned prediction per case, mean of 40 | 0.000929 |

The learned prediction costs **1.956294** times the analytic attribution per
case, and the whole study runs in **4.197 s** on 2 cores, well inside the
declared three-minute budget. Elapsed times on two shared cores are not stable
to better than about twenty per cent and should be read as orders of
magnitude.

## 7. Negative controls — `validate_negative_controls.py`

**29 controls declared, 17 must fire and all 17 did, 12 must not fire and none
did.** Every control is anchored on a value known from the derivation, never on
an output: an estimator whose spread is already sixty times the bound survives
being halved, which is how the first version of the bound control failed to
fire.

| group | control | expectation | result |
| --- | --- | --- | --- |
| A | the derivation itself as the null space | must not fire | 0.000000e+00 deg |
| A | a coordinate axis instead | must fire | 6.343495e+01 deg |
| A | the scale direction with one sign flipped | must fire | 5.313010e+01 deg |
| A | the best identifiable direction | must fire | 9.000000e+01 deg |
| B | identity A with no offset | must not fire | residual 1.748608e-15 |
| B | identity A with a 1e-14 relative offset | must not fire | residual 1.176716e-14 |
| B | identity A with a 1e-06 relative offset | must fire | residual 1.000000e-06 |
| C | the closed form with no offset | must not fire | 6.615096e-15 relative |
| C | the closed form with `b` moved by 1e-12 | must fire | 1.995384e-13 relative |
| C | the closed form with `b` moved by 1e-06 | must fire | 1.946973e-07 relative |
| D | detection with nothing moved | must not fire | detection rate 0.000000 |
| D | detection with all five parameters multiplied by e, median chi-square 1975.0118 on 2001 rows | must not fire | detection rate 0.000000 |
| D | detection with a 0.002 identifiable change | must fire | detection rate 1.000000 |
| E | the family of equally good fits, in claimed error-bar units | must fire | 3 of 3 multiples move log J by 49.232, 492.325 and 4923.249 claimed standard errors at a chi-square unchanged to 1.855369e-10 |
| F | intervals at their declared width | must not fire | marginal coverages 0.962500, 0.967500, 0.952500, 0.940000; worst 1.6059 sigmas |
| F | intervals rescaled by 0.25 | must fire | marginal coverages 0.417500, 0.370000, 0.390000, 0.390000; worst 53.2245 sigmas |
| F | intervals rescaled by 4.00 | must fire | marginal coverages all 1.000000; worst 4.5883 sigmas |
| G | the log J profile on the rate-only set | must not fire | max delta chi-square 3.967671e-10 |
| G | the log J profile on the two-channel set | must fire | max delta chi-square 1.855831e+04 |
| H | learned accuracy with the real labels | must not fire | accuracy 0.991667 |
| H | learned accuracy with the training labels shuffled, seed 424242 | must fire | accuracy 0.216667 against a chance level of 0.200000 |
| I | an equivalence partner displaced along the scale direction | must not fire | relative rate difference 3.195020e-16 |
| I | an equivalence partner displaced along an identifiable direction | must fire | relative rate difference 1.290093e-01 |
| I | an equivalence partner displaced along a coordinate axis | must fire | relative rate difference 3.457141e-02 |
| J | the declared null-component tolerance of 1e-06 | must not fire | the measured component is 0.447214 |
| J | a null-component tolerance of 0.5 and of 1.0 | must fire | the same component, now below the tolerance |

Group E is the sharpest: the pseudo-inverse diagonal of the Fisher information
describes the minimum-norm estimator's scatter correctly (ratios 1.06 to 1.09)
and says nothing about the parameter, and the control shows the family of
equally good fits moving `log J` by arbitrary amounts at an unchanged
chi-square: multiplying the null direction by 100 moves `log J` by 44.721360 --
4923.249 of the claimed standard errors -- while the chi-square changes by
1.855369e-10. Group F's last row is why this script uses 400 realisations and
not 200: a coverage that saturates at 1.000000 is only 3.2444 standard errors
from the nominal level at 200 and could not have reached the declared four.

The pseudo-inverse diagonal's measured ratios in group E are 1.042900 to
1.080857, so it describes the minimum-norm estimator's scatter correctly and the
parameter not at all.

## 8. The `pytest.approx` audit — `validate_approx_audit.py`

| measurement | result |
| --- | --- |
| `approx` calls carrying `rel=` | 81 |
| of those, carrying `rel=` with no explicit `abs=` | **0** |
| assertions depending on the hidden 1e-12 absolute default | **0** |
| tests whose assertions rest on an absolute bound, with `abs` forced to 0.0 | **2 of 254** |
| junit cross-check, strict run | 254 declared, 254 counted |
| junit cross-check, zero run | 254 declared, 254 counted |

The two tests are
`tests.test_plant::test_explicit_sensitivities_at_the_initial_point_sum_to_zero_by_hand`
and
`tests.test_identifiability::test_the_null_projector_is_idempotent_and_symmetric`.
Both compare against an exact zero, which has no relative bound at all. Neither
is a defect.

**The list above is stable across runs.** `tests/conftest.py` registers and
loads a Hypothesis profile with `derandomize=True` and `database=None`, so the
generated examples are a function of the test identity alone, and every
subprocess started from this directory runs with
`HYPOTHESIS_STORAGE_DIRECTORY` outside the repository. A predecessor in this
mission shipped an audit whose named failing tests varied from run to run
because its examples did, which makes the report worthless as evidence.

## 9. The command line and the tree — `validate_cli.py`

16 subcommand invocations and 6 bad-argument cases, each run as a subprocess
with `HOME` and `MPLCONFIGDIR` in a temporary directory outside the repository
and `PYTHONDONTWRITEBYTECODE=1`. **25 checks, 0 failures.** Every fragment
quoted from the CLI in README.md is searched for in the actual output.

- This script's own execution adds no cache directory and no bytecode file:
  **0 added**, against **32 inherited** at entry in the committed run --
  `.ruff_cache` and the bytecode of the thirty-one modules a preceding `ruff`
  and `pytest` invocation had already cached. Inherited artifacts are reported
  as inherited and are not failed, because what a harness leaves in the tree
  before calling this script is not the product's doing; release-gate check 2
  writes `.ruff_cache` into the tree one check before check 6 re-executes this
  script, and a claim that no such directory EXISTS would be unsatisfiable.
- No text file in the tree contains an absolute host path: **0 offenders in 73
  files scanned**, with the patterns assembled from `os.sep` and the script's
  own output deleted before the scan.
- `git ls-files` examines **81 tracked files and finds 0 offenders**. This
  line is a CORRECTION: when the build measured it the product was not yet
  committed, so the check examined 0 files and was recorded here as vacuous.
  Committing the product turned a vacuous check into a real one, and it passes.
  Release-gate check 9 enforces the same claim independently.

## 10. Working directories — `probe_working_directories.py`

Every validation script run from the filesystem root, the system temporary
directory, the repository root and `validation/` — which is the directory the
release gate uses — and every example run from the repository root and
`examples/`.

| probe | runs | non-zero exit statuses |
| --- | --- | --- |
| 11 validation scripts from 4 directories | 44 | **0** |
| 7 examples from 2 directories | 14 | **0** |

Total wall clock **449.4 s** and **442.6 s** on two consecutive runs, on 2
cores with two workers, dispatching the longest scripts first.

### 10.1 The probe used to create the failure it reported

The 0 above is the figure AFTER a defect in the probe itself was fixed, and the
defect is recorded rather than quietly removed.

As first built, the probe dispatched all 44 script-directory pairs into a
two-worker pool, so two copies of the SAME script could run at once.
`validate_examples.py` unlinks each shared `screenshots/*.png` and regenerates
it, then asserts the figure exists and exceeds 10,000 bytes. Two overlapping
copies therefore race on the same files: one deletes a figure the other is
about to measure.

Measured on 2026-11-09 by the coordinating session, not by the build: **1
non-zero exit in 44 runs** — `validate_examples.py` from the repository root —
which is **not reproducible by running that script alone from that
directory**, where it exits 0. Running two copies concurrently by hand did not
reproduce it either, because the collision window is the few milliseconds
between one copy's `unlink` and the other's `stat`; the probe's four runs over
two workers simply gave it more chances than a single pair does.

Two things were wrong, and both were fixed:

1. **The probe ran a script concurrently with itself.** Each script's four
   directory runs are now grouped onto one worker and executed serially, while
   different scripts still run two at a time. The same 44 runs are spread over
   the same two workers, so the wall-clock cost is unchanged (436.1 s before,
   449.4 s and 442.6 s after — the difference is noise on two cores).
2. **The probe discarded the evidence.** It reported only the failing script's
   `stderr`, and these scripts report their own failed checks on `stdout` and
   exit 1 with an empty `stderr`. The one real failure it has ever caught
   therefore printed no detail at all. It now falls back to `stdout`.

This is the **sixth** occurrence in this mission of the class the build guide
names: *a check whose own execution can create the condition it tests is not a
check.* It is also the second occurrence of the sub-class where the harness,
not the script, is the polluter — the first being release-gate check 2 writing
`.ruff_cache` before check 6 reads the tree, which cost P084 a night.

Per the build guide, the fixed probe was run **twice** from a clean tree: both
runs returned 0 of 44 and 0 of 14, and their per-directory status tables are
byte-identical. A single clean run would have been weak evidence, because the
failure it replaced was intermittent at roughly 1 in 44.

## 11. Examples — `validate_examples.py`

All 7 examples exit 0, write their PNG, and print the fragments README.md
quotes. Figure sizes 99564 to 169249 bytes, measured per figure in
`validation/validate_examples_output.txt`. Each PNG is deleted before its
example runs, so a stale figure cannot make a broken example pass.

## 12. The worked example — `worked_example.py`

The README's worked example, executed. **5 checks, 0 failures.**

The linearisation error of equation (12) is measured, not assumed. On noiseless
telemetry, the error in the best-determined direction against the truth's
projection:

| perturbation on log J | error | halving order |
| --- | --- | --- |
| 0.080 | 1.081920e-03 | — |
| 0.040 | 2.712685e-04 | 1.9958 |
| 0.020 | 6.790768e-05 | 1.9981 |
| 0.010 | 1.698772e-05 | 1.9991 |
| 0.005 | 4.248247e-06 | 1.9996 |

Second order, as a first-order estimator of a nonlinear model must be. With
noise at the declared seed the measured orders were 2.0178, 2.0895, 2.4366 and
3.2620, because below about one per cent the statistical error of 4.677542e-06
is comparable to the second-order term and the two partly cancel; that is
recorded in the output rather than presented as a convergence result.

## Checks that failed, and checks that are weaker than they look

Nothing in the committed output files is a FAIL. Four things are nevertheless
weaker than a reader might assume, and are stated here rather than left to be
discovered:

1. **The `git ls-files` hygiene check was vacuous at build time and is not
   any more** — it examined 0 files while the product was uncommitted and now
   examines 81 with 0 offenders (section 9). Listed here because the earlier
   drafts of this document claimed the weaker thing.
2. **Three of five profiles on the `rate_torque_short` set are truncated by the
   declared inner box** of 0.5 log-parameter units, so no interval is quoted for
   them. The box exists because an unconstrained Gauss-Newton step on a problem
   whose condition number is above 1e5 left the plant's declared validity window
   in one step and sent the integrator to a rate of order 1e199. The truncation
   is reported per profile in a `box` column.
3. **Both ends of every dependency range were run for `ruff` only.** The build
   environment is a shared pre-pinned interpreter this product is forbidden to
   modify, so numpy, scipy, scikit-learn, matplotlib, pytest and hypothesis are
   bounded at the single version each was measured at. See `pyproject.toml`,
   which says so in a comment, and README.md under Limitations.
4. **The analytic path's accuracy on single-parameter faults is 0.000000** by
   construction on every rate-only telemetry set, because it refuses. That is
   the honest answer to an ill-posed question and it is tabulated at zero
   rather than omitted; the comparable number for a reader who is willing to
   assume exactly one parameter moved is equation (15)'s 1.000000.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.
