# Validation evidence

**Level 3.** Every number below was produced by a script in this directory,
executed in the session that wrote this file, and every script writes its raw
output beside itself as `<script-name>_output.txt`. Those files are committed.
If a number here and a number in an output file disagree, the output file is
right and this file is a defect.

**The record is synthetic.** See `DATASET_CARD.md`. Nothing measured here is a
statement about any real cloud field or any real ground station.

## Environment

Measured by `_bootstrap.environment_lines` and printed at the head of every
output file in this directory.

| | |
| --- | --- |
| python | 3.13.18 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| pandas | 3.0.5 |
| pytest | 9.1.1 |
| hypothesis | 6.170.15 |
| ruff | 0.16.9 on the path, 0.17.4 as a Python module; clean under **both** |
| torch | not installed; named in README.md as unavailable and compared against nothing |
| statsmodels, copulas | not installed; named in the README alternatives table and compared against nothing |
| cpu cores | 2 |

## Test suite

Counted from the **junit XML**, never from pytest's stdout. A suite that
collects zero tests prints success and exits 0; that is how a product of this
mission shipped publicly with a recorded 251 tests and a config collecting none.

| source | value |
| --- | --- |
| `testsuite` element: `tests` | **433** |
| `testsuite` element: `failures` | 0 |
| `testsuite` element: `errors` | 0 |
| `testsuite` element: `skipped` | 0 |
| independent count of `testcase` elements | **433** |

Reproduce:

```bash
python -m pytest tests/ -q --junit-xml=junit.xml
python - <<'PY'
import xml.etree.ElementTree as ET
root = ET.parse("junit.xml").getroot()
suite = root if root.tag == "testsuite" else root.find("testsuite")
print(suite.get("tests"), len(suite.findall(".//testcase")))
PY
```

## Scripts, checks and run time

| script | checks | controls | failed | seconds |
| --- | --- | --- | --- | --- |
| `validate_geometry.py` | 11 | 0 | 0 | 1.1 |
| `validate_field.py` | 10 | 2 | 0 | 1.5 |
| `validate_climatology.py` | 9 | 2 | 0 | 1.9 |
| `validate_copula.py` | 12 | 2 | 0 | 4.4 |
| `validate_bounds.py` | 11 | 1 | 0 | 1.4 |
| `validate_empirical.py` | 9 | 2 | 0 | 2.7 |
| `validate_overstatement.py` | 8 | 0 | 0 | 2.9 |
| `validate_correlation.py` | 10 | 0 | 0 | 7.9 |
| `validate_learned_model.py` | 10 | 2 | 0 | 14.6 |
| `validate_negative_controls.py` | 26 | 26 | 0 | 1.6 |
| `validate_approx_audit.py` | 6 | 0 | 0 | 119.3 |
| `validate_examples.py` | 9 | 0 | 0 | 41.7 |
| `validate_cli.py` | 23 | 0 | 0 | 35.9 |
| `worked_example.py` | 4 | 0 | 0 | 1.8 |
| `probe_working_directories.py` | 1 | 0 | 0 | 279.4 |

**159 checks across the fifteen scripts, 0 failed.** 37 negative controls are
declared across all of them — the 26 of `validate_negative_controls.py` plus 11
embedded in the scripts whose claims they guard — and 26 fired, which is exactly
the declared must-fire set.

Total for the fourteen scripts above `probe_working_directories.py`: **238.7 s**.
Elapsed times are measured on two shared cores and move by ten to twenty per
cent between runs; the counts do not.

**0 checks failed.** Every check that is weak, vacuous or non-ideal is named
below under "Where a check is weak" rather than being dropped.

## The headline

From `validate_overstatement_output.txt`, on the declared `anisotropic` record:
20 stations, 23360 epochs, seed 20260002.

| network | n | P_independent | P_measured | ratio | nines lost |
| --- | --- | --- | --- | --- | --- |
| near-pair | 2 | 0.077335 ± 0.002064 | 0.171447 ± 0.005986 | **2.2169** | +0.3458 |
| far-pair | 2 | 0.121586 ± 0.002788 | 0.116139 ± 0.003971 | 0.9552 | -0.0199 |
| mixed-trio | 3 | 0.029901 ± 0.000910 | 0.062457 ± 0.002677 | **2.0888** | +0.3199 |
| cluster | 4 | 0.007256 ± 0.000264 | 0.103125 ± 0.004592 | **14.2124** | +1.1527 |
| two-groups | 6 | 0.000866 ± 0.000037 | 0.019906 ± 0.001472 | **22.9809** | +1.3614 |
| one-per-group | 5 | 0.007408 ± 0.000253 | 0.007149 ± 0.000714 | 0.9651 | -0.0154 |

The `±` on `P_independent` is the analytic propagation of equation (27); the `±`
on `P_measured` is the moving-block bootstrap of equation (21), 400 replicates
at a block of 64 epochs.

Read as availability: the `two-groups` network is assigned **0.999134** by the
independence assumption and shows **0.980094**; `cluster` is assigned
**0.992744** and shows **0.896875**.

### The exact decomposition, equation (16)

| network | P_naive | P_conditional | P_measured | co-modulation | correlation | residual |
| --- | --- | --- | --- | --- | --- | --- |
| near-pair | 0.077335 | 0.087600 | 0.171447 | +0.010265 | +0.083847 | 0.00e+00 |
| far-pair | 0.121586 | 0.114316 | 0.116139 | **-0.007270** | +0.001823 | 0.00e+00 |
| mixed-trio | 0.029901 | 0.029270 | 0.062457 | -0.000631 | +0.033188 | 0.00e+00 |
| cluster | 0.007256 | 0.013123 | 0.103125 | +0.005867 | +0.090002 | 0.00e+00 |
| two-groups | 0.000866 | 0.002160 | 0.019906 | +0.001294 | +0.017746 | 0.00e+00 |
| one-per-group | 0.007408 | 0.007436 | 0.007149 | +0.000028 | -0.000287 | 0.00e+00 |

The identity has no cross term, so the residual is pure rounding and is exactly
0.0 on every network. **Two networks have a negative co-modulation term and a
ratio below 1**: their stations' seasonal cloud maxima fall at different times
of year, so they are jointly clear more often than their annual mean rates
suggest and the naive product *overstates* their joint unavailability. That is
reported, not hidden; it is the mechanism behind every ratio below 1 in the
table above.

### The overstatement against separation, all 190 pairs

| separation km | pairs | mean km | mean ratio | min | max | ratio > 1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 to 200 | 21 | 147.4 | **1.7389** | 1.3370 | 2.3198 | 21 of 21 |
| 200 to 400 | 9 | 246.4 | **1.6824** | 1.4245 | 2.0559 | 9 of 9 |
| 800 to 1600 | 11 | 1493.7 | 1.0884 | 1.0721 | 1.1038 | 11 of 11 |
| 1600 to 3200 | 53 | 2427.7 | 0.9824 | 0.8908 | 1.0969 | 20 of 53 |
| 3200 to 8000 | 96 | 5130.2 | 1.0130 | 0.9441 | 1.0762 | 62 of 96 |

67 of 190 pairs have a ratio below 1. The declared geometry has **no pair
between 400 km and 800 km**, which is a property of the five-cluster layout and
is stated rather than left as a silent gap in a table: the row is absent from
the output because the bin is empty, not because it was dropped.

## Checks against mathematics the implementation does not use

The distinguishing check class. "Two implementations agree" and "the answer lies
inside a bound derived from different mathematics" are not the same check, and
the second catches what the first cannot.

| check | reference | result | tolerance or criterion |
| --- | --- | --- | --- |
| equation (9) against SciPy's Genz-algorithm MVN CDF | SciPy 1.18.1, a different algorithm | worst absolute difference **1.088539e-16** over 600 cases | 1e-14 |
| haversine against the spherical law of cosines | an independently derived form | worst absolute difference **7.925394e-11 km** over 4000 coordinate pairs | 1e-05 km |
| the n = 2 Bonferroni identity | inclusion-exclusion from CLEAR-rate statistics | worst residual **2.220446e-16** over all 190 pairs | 1e-14 |
| Frechet-Hoeffding bounds, equation (17) | the marginals alone; no copula, no correlation | **0 violations in 380 comparisons** over 190 pairs by 2 routes | slack of equation (19) |
| the same, over 6 networks by 3 routes | the same | **0 violations in 18 comparisons** | the same |
| equation (10) Monte Carlo against equation (8) quadrature | a random algorithm against a deterministic one | worst deviation **1.9705 sigma** over 4 pairs | declared 4 sigma |
| the Monte Carlo error against `sqrt(p(1-p)/M)` | binomial theory | ratio **4.0124** for a 16-fold draw increase | 4.0 ± 0.5 |
| equation (3) Pythagorean identity | algebra | worst relative residual **4.111564e-16** over 190 pairs | 1e-14 |
| the quartic trapezoid against the Gauss-Legendre quadrature | a 20001-point trapezoid on the same integrand | agreement to **1e-07 relative** | the trapezoid's own truncation error |
| equation (8) at `rho = 0` against equation (15) | a product of marginals against a quadrature | worst difference **below 1e-05** | the binned nodes' own error |

## The cancellation equation (9) exists to avoid

From `validate_copula_output.txt`. The correlation excess is computed as an
integral of a positive integrand, never as `Phi_2 - Phi(h)Phi(k)`.

| rho | first-order reference | integral form | subtraction | integral rel err / rho | subtraction rel err |
| --- | --- | --- | --- | --- | --- |
| 1e-04 | 1.964128e-09 | 1.965012e-09 | 1.965012e-09 | 4.501 | 4.501067e-04 |
| 1e-08 | 1.964128e-13 | 1.964128e-13 | 1.964128e-13 | 4.500 | 4.540567e-08 |
| 1e-12 | 1.964128e-17 | 1.964128e-17 | 1.964121e-17 | 4.500 | 3.493012e-06 |
| 1e-16 | 1.964128e-21 | 1.964128e-21 | 1.905824e-21 | — | 2.968437e-02 |
| 1e-20 | 1.964128e-25 | 1.964128e-25 | **0.000000e+00** | — | **1.000000e+00** |

The integral form's apparent error at `rho >= 1e-12` is the reference
expansion's own second-order remainder, `|h k| rho / 2 = 4.5 rho`, to four
figures. Below that the remainder falls under double-precision rounding and the
integral form sits at the machine floor: worst relative error **1.915141e-16**.
The subtraction returns exactly 0.0 for a quantity of 1.964128e-25.

## The conditioning of a bound check, equation (19)

From `validate_bounds_output.txt`. The absolute slack is set by the magnitude of
the **summands** and never by the result, so the relative slack is `kappa` times
machine epsilon and diverges exactly where the expression cancels.

| lower bound f(q) | absolute slack | condition number | relative slack | lower informative |
| --- | --- | --- | --- | --- |
| 1e-01 | 1.705303e-14 | 1.100000e+01 | 1.705303e-13 | yes |
| 1e-02 | 1.449507e-14 | 1.010000e+02 | 1.449507e-12 | yes |
| 1e-04 | 1.421370e-14 | 1.000100e+04 | 1.421370e-10 | yes |
| 1e-06 | 1.421088e-14 | 1.000001e+06 | 1.421088e-08 | yes |
| 1e-08 | 1.421085e-14 | 1.000000e+08 | 1.421085e-06 | yes |
| 1e-12 | 1.421085e-14 | 1.000000e+12 | 1.421085e-02 | yes |
| 1e-17 | 1.421085e-14 | 1.000000e+17 | 1.421085e+03 | **NO** |

The absolute slack varies by a factor of **1.200000** across sixteen decades of
cancellation; the relative slack rises by a factor of **8.333e+15**, decade for
decade. At `f(q) = 1e-17` the bound falls below its own slack and is reported as
**uninformative** rather than as a pass.

**This product's first version of equation (19) was wrong.** It returned
`c eps kappa |f|`, which cancels to `c eps sum|q|`: it looked conditioned and was
independent of the bound value altogether. Measured then, a well-conditioned
bound of 0.9 got a slack of 2.700e-14 and an ill-conditioned bound of 1e-07 got
1.421e-14 — the ill-conditioned case got the **smaller** slack. The defect is
pinned in `tests/test_regressions.py` and a negative control reconstructs it.

## Negative controls

From `validate_negative_controls_output.txt`. **Every control is anchored on a
value the generator declared or on a bound derived in closed form, never on an
output**, and every offset is a multiple of the check's own criterion rather
than a chosen constant. Declared sigma is **4**, not 2: a control whose
criterion is a 95 per cent interval fires once in twenty by construction.

| | |
| --- | --- |
| controls declared | **26** |
| controls that must fire | **15** |
| of those, fired | **15** |
| controls that must NOT fire | **11** |
| of those, fired | **0** |
| control checks failed | **0** |

Representative entries:

| control | direction | result |
| --- | --- | --- |
| a value above the Frechet upper bound is detected | must fire | FIRED |
| a value below an **informative** Frechet lower bound is detected | must fire | FIRED |
| a corrupted pair clear rate breaks the n = 2 identity | must fire | FIRED, residual 2.682077e-12 against a true 1.110223e-16 |
| a Monte Carlo at a perturbed correlation misses the exact quadrature | must fire | FIRED, 6.2878 sigma |
| the Monte Carlo at the TRUE correlation does not miss it | must NOT fire | did not fire, 1.8996 sigma |
| regime 0's field does not match regime 2's declared matrix | must fire | FIRED, 5.5914 sigma |
| regime 0's field DOES match regime 0's declared matrix | must NOT fire | did not fire, 2.1433 sigma |
| an exponential fit to a GAUSSIAN correlation law leaves a visible error | must fire | FIRED, 0.155468 |
| an indefinite pair matrix is detected and projected | must fire | FIRED, least eigenvalue -9.0e-01 |
| the generator's own matrix is NOT projected | must NOT fire | did not fire, least eigenvalue 2.985671e-01 |
| **the overstatement vanishes on a record with no spatial correlation** | must NOT fire | did not fire, 0.2098 bootstrap standard errors |
| **the overstatement is present on the correlated reference record** | must fire | FIRED, 20.8149 bootstrap standard errors |
| the co-modulation term vanishes with no seasonal or diurnal modulation | must NOT fire | did not fire, -1.263258e-05 |

The last four are the headline's own controls. A record generated with **no
spatial correlation, no seasonal or diurnal modulation and no persistence** —
everything the independence assumption assumes, true by construction — produces
an overstatement of +0.000211, which is 0.21 bootstrap standard errors. On the
reference record the same measurement is 20.81.

## The `pytest.approx` audit

From `validate_approx_audit_output.txt`. `pytest.approx(expected, rel=r)` passes
if EITHER bound is met and the absolute default is `1e-12`, so a bare `rel=` on
a small quantity silently stops testing what it says.

| measurement | value |
| --- | --- |
| `approx` calls in the suite | 90 |
| calls carrying `rel=` | 90 |
| calls carrying `rel=` with no explicit `abs=` (source scan) | **0** |
| **assertions depending on the hidden 1e-12 default** (suite re-run with a bare `rel=` made to raise) | **0 of 433** |
| assertions resting on an ABSOLUTE bound (suite re-run with every `abs` forced to 0.0) | **10 of 433** |

The ten are named in the output file and each is accounted for:

| test | why its bound is absolute |
| --- | --- |
| `test_climatology::..._ar_one_series_recovers_its_coefficient` | a sampling criterion on a correlation; absolute by nature |
| `test_copula::test_the_implied_correlation_inverts_the_copula` | a residual that is exactly zero, and the root finder's own `xtol` |
| `test_correlation::test_any_declared_exponential_is_recovered_from_its_own_samples` | a comparison against an exact zero |
| `test_geometry::test_a_bearing_normal_to_the_axis_is_entirely_across_it` | the along component is exactly zero |
| `test_geometry::test_one_degree_of_latitude_is_about_111_point_2_km` | a deliberate comparison against a hand-rounded 111.1950 |
| `test_learned::test_the_interval_is_wide_enough_to_cover_most_of_the_training_target` | a coverage, a probability; absolute by nature |
| `test_regressions::..._first_order_copula_expansion_remainder_is_second_order` | a dimensionless ratio; absolute by nature |
| `test_regressions::test_regression_the_haversine_antipodal_relative_error` | a relative error compared to a measured constant |
| `test_report::test_number_round_trips_through_float` | a fixed-point rendering has an absolute precision |
| `test_uncertainty::test_the_uncorrected_share_is_biased_toward_zero_by_simulation` | a Monte Carlo mean of a quantity in [0, 1] |

**None is a defect.** The list is stable across runs: `tests/conftest.py`
registers a Hypothesis profile with `derandomize=True` and `database=None`, and
every subprocess started from this directory runs with
`HYPOTHESIS_STORAGE_DIRECTORY` outside the repository.

## Working directories

From `probe_working_directories_output.txt`. Three products in one batch of this
mission passed every check in their own build and then failed the release gate,
which runs each validation script with `cwd=validation/`.

| | |
| --- | --- |
| validation scripts probed | 14 |
| directories | 4 — filesystem root, system temporary directory, repository root, `validation/` |
| validation runs | **56** |
| example scripts probed | 7 |
| example directories | 2 — repository root, `examples/` |
| example runs | **14** |
| **total runs** | **70** |
| **non-zero exit statuses** | **0** |
| wall clock | 265.4 s, 266.0 s and 278.2 s on the three runs made during this build, all inside the gate's 600 s per-script limit |

**The probe is itself a check whose execution could create the condition it
tests**, so the grouping is explicit: every run of one script is serial on one
worker, and everything that writes into `screenshots/` — `validate_examples.py`
and all seven examples — shares a **single** serial group, because two different
scripts writing the same file is the same defect as one script overlapping
itself. A predecessor in this mission measured 1 non-zero exit in 44 runs from
exactly that cause. **The probe was run twice in a row from a clean tree and the
two runs agree on every one of the 70 exit statuses**; only the elapsed-time
columns differ, and those are labelled unstable in the output itself.

**`validate_approx_audit.py` is probed in its declared PATHS-ONLY mode**, and
that is a stated reduction rather than a silent one. It runs the whole test
suite twice; four copies is eight suite runs and about 440 seconds, which would
put the probe outside the gate's limit. Paths-only mode does everything that
depends on the caller's working directory — the bootstrap, the repository root,
the scan of `tests/`, the read of `tests/conftest.py`, the resolution of the
output path — skips only the two pytest subprocesses, whose environment
`_bootstrap.subprocess_environment` sets explicitly, and writes to a temporary
file outside the repository so a probe run cannot overwrite the committed
evidence. The committed output is always from a full run and says so.

## The learned model against the baselines

Reported in full in `MODEL_CARD.md` and in
`validate_learned_model_output.txt`. The result, in one table, with the
strongest baseline on the TRAIN cell (`per_regime` on both records):

| record | cell | baseline MAE | learned MAE | margin | 95% CI on the margin | reference half-width | inside |
| --- | --- | --- | --- | --- | --- | --- | --- |
| anisotropic | held-out SITES | 0.012509 | 0.010406 | **+0.002103** | [+0.000249, +0.002614] | 0.005952 | yes |
| anisotropic | held-out TIME | 0.008870 | 0.012329 | **-0.003459** | [-0.003248, -0.001384] | 0.005758 | yes |
| isotropic | held-out SITES | 0.008787 | 0.008023 | **+0.000764** | [-0.000604, +0.001409] | 0.004892 | yes |
| isotropic | held-out TIME | 0.008382 | 0.011550 | **-0.003168** | [-0.002945, -0.001241] | 0.006598 | yes |

**Every margin is smaller than the bootstrap half-width on the empirical
reference.** On the isotropic control, where the two-parameter exponential is
correctly specified by construction, the paired interval on the held-out-SITES
margin contains zero: the two are not distinguishable at all. On **held-out
TIME, on both records, the baseline beats the learned model** by more than the
learned model's best win.

Scored against the generator's own correlation rather than against the noisy
implied correlation, on held-out SITES of the anisotropic record:

| model | MAE against the TRUE correlation |
| --- | --- |
| independent | 0.122407 |
| exponential | 0.043280 |
| per_regime | **0.043236** |
| learned | 0.063107 |

The learned model is **closer to the target and further from the truth**. It is
fitting the estimation noise in the implied correlation, whose own rms error
against the truth is 0.073751.

## The baselines, validated before anything was benchmarked

From `validate_correlation_output.txt`.

| record | implied-rho rms error vs truth | fitted-function rms error vs truth | pooled MAE | per-regime MAE | margin |
| --- | --- | --- | --- | --- | --- |
| isotropic (correctly specified) | 0.076051 | **0.014631** | 0.055301 | 0.054554 | 0.000747 |
| anisotropic (mis-specified) | 0.073751 | **0.065972** | 0.058410 | 0.057737 | 0.000674 |

The fitted function is 4.5 times worse on the record where the isotropic form is
wrong, which is the information the learned model has to work with — and that
mis-specification error, 0.065972, is **below** the implied correlation's own
noise floor of 0.073751. That is why the learned margins are as small as they
are, and it is stated rather than left to be inferred.

**The specified baseline is not under strength.** Equation (23) pooled and
equation (24) per regime differ by 0.000674 in correlation MAE on the training
cell of the reference record. The strengthened form is used where it is stronger
and the specified form is reported beside it everywhere, as the mission build
rule requires.

## Where a check is weak, vacuous or non-ideal

The credible part of any validation section. Every item here is in a committed
output file.

- **The block bootstrap does not plateau.** `validate_empirical_output.txt`
  measures the standard error rising monotonically from 1.972045e-03 at a block
  of 1 epoch to 7.800017e-03 at 512 epochs, a factor of 3.9553, with no sign of
  settling. The record carries a seasonal cycle as well as synoptic
  persistence, so the joint-cloudy indicator is correlated at lags of months and
  no block satisfying `L / T -> 0` captures all of it. **Every interval quoted
  at the declared 64 epochs is therefore a lower bound on the sampling
  uncertainty**: a 512-epoch block gives 1.7062 times more. That direction is
  the safe one for this product's conclusion — widening the interval can only
  strengthen "the margin is inside it" — and it is named here rather than left
  for a reader to find.
- **The Bonferroni bracket is vacuous for 3 of 6 declared networks.** At six
  stations with cloud rates near 0.3 the sum of the clear rates exceeds 4 and
  the bracket leaves the unit interval entirely. It is informative only at
  n = 2, where it is an exact identity, and `validate_bounds.py` and the CLI
  both print the flag.
- **The Frechet lower bound is informative for only 8 of 190 pairs.** On cloud
  rates below 0.5 the union bound is vacuous and the lower bound clips to zero.
  The check is a real check on those 8 pairs and on the upper bound everywhere,
  and the count is reported rather than a pass being claimed on 190.
- **Equation (13) disagrees with the block bootstrap on 20 of 20 stations** by
  more than 5 per cent, ratio 0.6428 to 0.7984. The AR(1) correction understates
  the inflation because a thresholded persistent Gaussian is not an AR(1). A
  negative control asserts the disagreement is detected rather than claimed
  away, and the bootstrap figure is the one quoted for a joint rate.
- **The learned model's interval is badly under-covered out of sample**:
  0.4643 on held-out TIME against a nominal 0.80. A negative control asserts
  that the under-coverage fires.
- **Held-out TIME is not a test of temporal extrapolation.** None of the eight
  features encodes the time half, so a held-out-TIME row gets the same
  prediction as its train-cell twin and only the target differs. What it tests
  is whether a model fitted on the first half still reproduces the second
  half's joint rates. Overstating it would overstate the result.
- **One leak is declared.** The marginal climatology is estimated once over the
  whole record and used in both time halves. It is identical for every model in
  the comparison, and estimating it twice would make the two halves' targets
  incomparable.
- **The margin bootstrap resamples the record and holds the fitted models
  fixed.** It measures the uncertainty that comes from the reference being a
  finite record, which is what the specification's phrase means. It does not
  measure the uncertainty in the models' fitted parameters, which would require
  refitting inside every replicate.
- **The `git ls-files` hygiene check in `validate_cli.py` examined 0 files when
  this build measured it**, because the product was uncommitted, and it was
  recorded in the output as **VACUOUS** rather than as a pass. Release-gate
  check 9 enforces the same claim independently.
  **Corrected after commit (2026-11-10):** the coordinator committed the product
  and re-ran the script independently; the check examined **92 tracked files and
  found 0 offenders**, so it is now real rather than vacuous. The earlier,
  weaker claim is kept above rather than overwritten.
- **The coordinator's own re-measurement left 37 cache artifacts in the tree**
  — `.ruff_cache` plus 36 `__pycache__` entries and `.pyc` files, written by the
  coordinator's `ruff` and `pytest` runs before `validate_cli.py` was invoked.
  The self-check reported all 37 as **inherited**, did not fail on them, and
  separately confirmed that git tracks none. This is the third empirical
  confirmation of the 2026-11-02 narrowing of the self-check rule, and the first
  in which the polluter is the coordinating session rather than the release
  gate.
- **The baselines beat the learned model on half the hold-out cells.** That is
  research evidence. It is not removed, softened or retuned away.
- **Dependency ranges were run at both ends for `ruff` only.** See README.md
  under Limitations.

## Reproducing everything

```bash
python -m pytest tests/ -q --junit-xml=junit.xml
python validation/validate_geometry.py
python validation/validate_field.py
python validation/validate_climatology.py
python validation/validate_copula.py
python validation/validate_bounds.py
python validation/validate_empirical.py
python validation/validate_overstatement.py
python validation/validate_correlation.py
python validation/validate_learned_model.py
python validation/validate_negative_controls.py
python validation/validate_approx_audit.py
python validation/validate_examples.py
python validation/validate_cli.py
python validation/worked_example.py
python validation/probe_working_directories.py
ruff check src/ tests/ examples/ validation/
```

Every script is deterministic in its declared seeds and requires no network
connection and no data download.
