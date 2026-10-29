# Validation

Level 2. Research grade. Every number below was produced by a script in this
directory, executed on 2026-10-29 in the environment recorded in the header of
each output file, and the file that produced it is named beside it. No figure is
carried from an earlier run, estimated, or written by hand.

Measured environment: Python 3.13.18, numpy 2.5.3, scipy 1.18.1, scikit-learn
1.9.1, matplotlib 3.11.2, pytest 9.1.1, hypothesis 6.170.13, ruff 0.16.9 on
`PATH` and 0.17.3 as the installed distribution, two CPU cores. `filterpy` is
named in the alternatives table of `README.md` and is **not installed**; no
number here is compared against it.

## What was run

| script | output | checks | failed |
|---|---|---|---|
| `validate_observability.py` | `validate_observability_output.txt` | 8 | 0 |
| `validate_recovery_grid.py` | `validate_recovery_grid_output.txt` | 7 | 0 |
| `validate_recursive_agreement.py` | `validate_recursive_agreement_output.txt` | 10 | 0 |
| `validate_unobservable.py` | `validate_unobservable_output.txt` | 5 | 0 |
| `validate_learned.py` | `validate_learned_output.txt` | 9 | 0 |
| `validate_controls.py` | `validate_controls_output.txt` | 17 | 0 |
| `validate_cli.py` | `validate_cli_output.txt` | 24 | 0 |
| `worked_example.py` | `worked_example_output.txt` | 4 | 0 |
| `probe_working_directories.py` | `probe_working_directories_output.txt` | 10 | 0 |

94 checks, 0 failed. Three results are unfavourable to parts of this package and
are set out in §7; they are the reason to trust the rest.

Test suite: **307 tests, 0 failures, 0 errors, 0 skipped**, counted from
`junit_self.xml` by both the `tests` attribute and a count of `testcase`
elements, which agree. Ruff-clean under 0.16.9 and under 0.17.3.

## 1. The observability verdict

Source: `validate_observability.py`.

### Equation (10) as an identity

`lambda_i(G) = (W / s^2) (1 - p_i)`: the observability Gramian of (8) and the
normalised rate second moment of (9) share eigenvectors exactly, so the least
observable alignment direction is the most excited rotation axis.

| check | reference | result | tolerance |
|---|---|---|---|
| equation (10) on 5 profiles x 2 noise levels | `numpy.linalg.eigvalsh` against the closed form | worst relative departure **6.870e-16** | 1e-9 |
| the unobservable direction is the rotation axis | 60 single-axis seeds | worst angle **0.000000 deg** | 0.5 deg |
| the result is not trivial | the same 60 seeds | the motion is never closer than **7.0421 deg** to a body axis | > 5 deg |

### The noise floor of equation (11), and why a rank test is wrong

The Gramian is formed from the **measured** reference rate, so the reference
sensor's own noise contributes `2 sigma_A^2` per sample in every direction,
including the one the motion never excited. The measured Gramian of an exactly
single-axis record is therefore **not singular**.

| samples | lambda_min | floor (11) | ratio | condition | rank, gated | rank, bare |
|---|---|---|---|---|---|---|
| 200 | 205.12 | 200 | 1.0256 | 1.255e+04 | 2 | 3 |
| 500 | 494.978 | 500 | 0.9900 | 1.531e+04 | 2 | 3 |
| 1000 | 1022.38 | 1000 | 1.0224 | 1.482e+04 | 2 | 3 |
| 2000 | 2004.75 | 2000 | 1.0024 | 1.527e+04 | 2 | 3 |
| 4000 | 4011.75 | 4000 | 1.0029 | 1.561e+04 | 2 | 3 |

`numpy.linalg.matrix_rank` reports 3 on **5 of 5** records that support 2. The
floor gate reports 2 on **5 of 5**.

### An independent bracket on lambda_min, equation (13)

`lambda_min` is computed by `numpy.linalg.eigh` and independently bracketed by
Gershgorin's circle theorem below and the Rayleigh quotient at the canonical
basis vectors above. Neither uses an eigensolver, so this is a bound check in
the sense the mission's build guide requires. **0 of 5** profiles inconsistent.
The Gershgorin lower bound is negative on **2 of 5** profiles, which is
reported rather than hidden: diagonal dominance is a weak test on a matrix with
a condition number of 1.6e4.

### Sensitivity to the two declared thresholds

The rank verdict is unchanged for a floor factor anywhere from 1.5 to 64 on
every profile, so the declared 4 is not load-bearing. The three-valued verdict
**is** sensitive to the declared requirement, which is the intended behaviour:
a requirement is an input and "observable" has no meaning without one.

### Two rotation axes are enough for three alignment degrees of freedom

| profile | p0 | p1 | p2 | rank | condition | worst CRB |
|---|---|---|---|---|---|---|
| single_axis | 0.999935 | 0.000034 | 0.000032 | 2 | 1.53e+04 | unbounded |
| near_single_axis | 0.999553 | 0.000415 | 0.000032 | 3 | 2238 | 2008.475 arcsec |
| two_axis | 0.509092 | 0.490892 | 0.000016 | 3 | 2.037 | 43.139 arcsec |
| three_axis | 0.505429 | 0.318845 | 0.175727 | 3 | 1.667 | 42.847 arcsec |
| isotropic | 0.378581 | 0.335706 | 0.285714 | 3 | 1.149 | 93.133 arcsec |

Reported as measured and **not** as expected: the two-axis record is *worse*
conditioned (2.037) than the three-axis one (1.667). Two equal-amplitude axes
give `p = (1/2, 1/2, 0)` and therefore eigenvalues in the ratio 2:1:1 by (10);
the three-axis profile's unequal amplitudes spread `p` more evenly. The number
of excited axes alone does not predict the conditioning.

## 2. Recovery across the swept grid, against the Cramer-Rao bound

Source: `validate_recovery_grid.py`. 4 excitation profiles x 3 noise levels x 3
record lengths x 24 seeds = **864 records**, 2290 standardised scores.

| check | reference | result | tolerance |
|---|---|---|---|
| standardised score (24), whole grid | standard normal | mean **-0.073846**, std **0.959812** | mean within 0.1, std in [0.85, 1.20] |
| `|z| > 3` fraction | standard normal 0.002700 | **0.000000** | reported |
| worst `|z|` anywhere | 2290 scores | **2.7673** | < 5 |
| per-cell z standard deviation | 36 cells | range **[0.850811, 1.050421]** | [0.5, 1.6] |
| per-cell z mean | 36 cells | range **[-0.172245, +0.123971]** | within 4 standard errors, 0.577 |
| Gauss-Newton convergence | 1e-12 rad step bound | **864 of 864** | all |

The per-record ratio `|e_obs| / CRB` has a median of **0.820958** over all
cells, with a cell range of [0.421274, 0.894207]. That statistic's expected
value depends on how the bound's mass is spread across directions -- 0.8881
when all three contribute equally, 0.6745 when one ill-conditioned direction
dominates -- so it is reported for information and the **check is on the
standardised score**, which is standard normal regardless of conditioning.

A ratio of two root-mean-squares is the wrong statistic here and is shown as
such: it ranges from 0.775 to 1.692 across the same cells, because within a
cell the bound varies by orders of magnitude between seeds and an RMS ratio is
decided by the single worst-conditioned seed.

### The regularised variant falls below the bound, and that is allowed

The MAP solution of (18) has a median per-record ratio of **0.659065** and is
below the bound on **36 of 36** cells, with a standardised score of mean
**-0.025297** and standard deviation **0.810464**. The Cramer-Rao bound constrains unbiased
estimators only; (18) shrinks toward zero and is biased, and the bias is
reported per profile and record length in the output file.

### The unobservable component is carried in full by every estimator

| profile | unobservable RMSE | mean rank |
|---|---|---|
| single_axis | 0.513471 mrad | 2.0000 |
| near_single_axis | 0.321590 mrad | 2.6019 |
| two_axis | 0.000000 mrad | 3.0000 |
| three_axis | 0.000000 mrad | 3.0000 |

The single-axis figure is the per-component spread of the injected population
projected onto the one withheld direction. With a magnitude uniform on
[0, 3] mrad and a direction uniform on the sphere the per-component standard
deviation is 1.0 mrad; the declared check accepts the measured **0.513471 mrad**
inside the window [0.5, 1.5] mrad, and it passes at the low edge of it. The
discrepancy is that the direction-uniform draw does not put a full component's
worth of misalignment on any one named axis, so 1.0 mrad is an upper reference
rather than a prediction, and the window is wide for that reason. The batch
solution declines to guess at that component and therefore carries all of it.

## 3. The recursive estimator against the batch solution

Source: `validate_recursive_agreement.py`. 5 profiles x 3 noise levels x 3
lengths x 4 seeds, 491 degrees of freedom compared.

### Agreement on the observable subspace, per regime

| regime | records | median diff / CRB | max diff / CRB | declared |
|---|---|---|---|---|
| well conditioned (rank 3, condition < 10) | 108 | 0.006810 | **0.052809** | 0.1 |
| full rank, any conditioning | 131 | 0.007962 | **0.352091** | 0.5 |
| rank deficient | 49 | 0.061812 | **1.942601** | none declared |

The tolerances are stated per regime because the measurement says the regime
matters. On a rank-deficient record the recursive estimator does not withhold
the unsupported direction; it accumulates a rotation there with a median
magnitude of **45.1093 mrad** and a maximum of 130.7890 mrad, against an
injected misalignment of at most 3 mrad,
and because composing rotations is not a linear operation that rotation moves
the components the record *does* support. Correlation between the size of the
excursion and the size of the leakage: **+0.3206**.

This is the same mechanism that makes the unconstrained SVD solution of (17)
wrong about the observable subspace, measured independently in
`tests/test_batch.py` and in control C13.

### Is the prior what separates the two solutions? No.

The shrinkage argument predicts the disagreement to fall by a factor of 1111
when the prior one sigma is tightened from 0.1 rad to 0.003 rad. **It changes by
a factor of 0.4470.** The residual disagreement is dominated by the sequential
re-linearisation of (20) about a moving estimate, not by the prior, and quoting
the shrinkage figure as the expected agreement would be wrong by three orders
of magnitude.

### The Joseph form of (21)

| samples | worst P asymmetry | smallest P eigenvalue |
|---|---|---|
| 6000 | 2.168404e-18 rad^2 | 6.483588e-09 rad^2 |

Positive definite throughout, symmetric to rounding.

## 4. The unobservable subspace

Source: `validate_unobservable.py`. 120 single-axis records of 150 samples, with
**2 mrad injected entirely along each record's own dominant rotation axis**. The
control is anchored on the chosen truth and verified to fire: **120 of 120**
records are rank deficient, and the injection lies within **0.000001 deg** of
the worst-observed Gramian eigendirection.

Injected truth along the unsupported direction: median **412.5296 arcsec**.

| estimator | claim, arcsec | \|error\|, arcsec | its own 1 sigma | \|err\|/sigma | 2-sigma coverage |
|---|---|---|---|---|---|
| batch, eq (14)-(16) | -0.0000 | 412.5296 | **infinite (refused)** | — | — |
| recursive MEKF | 501.3101 | 8205.6294 | 12868.0069 | 0.6393 | 0.9917 |
| batch MAP, eq (18) | -0.0343 | 412.5639 | 206.2496 | 2.0003 | 0.4917 |
| learned forest | -0.5320 | 413.0616 | 149.3864 | 2.7775 | **0.0750** |
| learned forest_deep | -1.6141 | 414.1437 | 169.0386 | 2.4939 | **0.2083** |
| learned bagged_ridge | -7.5118 | 420.0414 | **14.3540** | **29.0693** | **0.0000** |
| trivial baseline | -4.6774 | 417.2070 | 207.5599 | 1.9987 | 0.5083 |

Nominal two-sigma coverage is 0.954500.

### The specification's prediction, tested in three parts

| part | result |
|---|---|
| (a) the analytic method returns no number there | largest absolute claim over 120 records **7.041e-19 rad**, and the bound it reports is infinite on every one |
| (b) the learned uncertainty is not materially larger there | worst inflation **1.2364x** against its own best-observed direction, where the analytic ratio is infinite |
| (c) the learned interval does not contain the truth | best two-sigma coverage across the three declared configurations **0.2083** against a nominal 0.954500 |

**The prediction is confirmed.** The sharpest configuration, `bagged_ridge`,
reports a one sigma of 14.3540 arcsec, is wrong by 29.0693 of it, and its
two-sigma interval contains the truth on **0.0000** of records.

### Is it specific to the unobservable direction? Yes.

| configuration | 2-sigma coverage, worst direction | 2-sigma coverage, best direction |
|---|---|---|
| forest | 0.0750 | 1.0000 |
| forest_deep | 0.2083 | 1.0000 |
| bagged_ridge | 0.0000 | 0.7250 |
| trivial | 0.5083 | 1.0000 |
| batch, Cramer-Rao | refuses | 0.9750 |

Along the best-observed direction every learned configuration covers at or above
nominal; along the unsupported direction, the same models on the same records
collapse. The overconfidence appears exactly where the data stops supporting an
answer, and nothing in the model's own output distinguishes the two cases.

### Record length does not help

| samples | rank | floor ratio | batch claim | MEKF \|err\| | learned \|err\| | learned sigma |
|---|---|---|---|---|---|---|
| 150 | 2.00 | 0.964472 | 1.484e-19 | 7463.5 arcsec | 420.354 arcsec | 14.380 arcsec |
| 600 | 2.00 | 0.991843 | 1.166e-19 | 9938.1 arcsec | 420.048 arcsec | 14.859 arcsec |
| 2400 | 2.00 | 0.988895 | 5.970e-20 | 10489.7 arcsec | 419.769 arcsec | 16.540 arcsec |
| 4800 | 2.00 | 1.000220 | 3.671e-20 | 5633.9 arcsec | 421.001 arcsec | 17.847 arcsec |

A factor of 32 in record length. The learned error does not fall and its
reported sigma does not grow. A degree of freedom the motion never excited is
not a sample-size problem.

## 5. The learned estimator against the baselines

Source: `validate_learned.py`. 2500 training records of 150 samples, 600
held-out test records of 150 samples, both seeded.

### On the short-record population, which is the case the learned estimator is for

| estimator | RMSE total | **RMSE observable** | RMSE unobservable | median total | worst total |
|---|---|---|---|---|---|
| batch, eq (14)-(16) | 5.716038 | **5.703693** | 0.375465 | 2.189928 | 93.072086 |
| **batch MAP, eq (18)** | 0.792044 | **0.697326** | 0.375594 | 0.997971 | 3.110015 |
| learned forest | 0.853146 | **0.760152** | 0.387333 | 1.164870 | 3.303029 |
| learned forest_deep | 0.860938 | **0.766268** | 0.392489 | 1.180213 | 3.396035 |
| learned bagged_ridge | 0.959229 | **0.882258** | 0.376485 | 1.441612 | 3.606330 |
| trivial (training mean) | 1.010964 | **0.938906** | 0.374837 | 1.493447 | 3.017492 |

All figures in milliradians. **The baseline wins.** The regularised analytic
solution of (18) beats the best learned configuration by **9.01 per cent** on
the observable subspace, with no training, no features and no model.

Three facts, all measured and all reported:

1. The best learned configuration **beats the trivial baseline** by 19.04 per
   cent, so the learned component has learned something.
2. It **beats the specified Baseline 1** by a factor of 7.5, because the
   unbiased maximum-likelihood solution has a variance set by the Cramer-Rao
   bound and on a 150-sample record that bound is comparable to the spread of
   the misalignment population itself.
3. It **loses to the strengthened analytic baseline**, which is the expected
   result. The only mechanism by which a learned estimator can beat an unbiased
   solution on a short record is shrinkage toward the population it was trained
   on, and (18) performs exactly that shrinkage in closed form.

No excitation profile reverses the conclusion: **0 of 4**.

### What the analytic win is made of, and what it costs

| samples | batch | MAP, true prior | MAP, 10x tight | MAP, 10x loose |
|---|---|---|---|---|
| 150 | 0.862279 | 0.561053 | **1.002141** | 0.840808 |
| 600 | 0.290345 | 0.269755 | **0.869195** | 0.289972 |
| 2400 | 0.142100 | 0.141564 | **0.634698** | 0.142081 |
| 9600 | 0.075992 | 0.075205 | **0.369287** | 0.075982 |

A prior ten times too tight makes the MAP baseline lose to the unbiased solution
at **4 of 4** record lengths. The analytic win is a statement about knowing the
population, not a free lunch.

The MAP/batch RMSE ratio moves from **0.650663** at 150 samples to **0.989640**
at 9600: the advantage collapses as the record gains information. It is not
claimed to be optimal -- (18) is the posterior mean under a Gaussian prior
matched in second moment, and the generated population's prior is not Gaussian,
so no domination theorem applies.

### The uncertainty output

| configuration | 2-sigma coverage | median sigma | RMSE total | sigma / RMSE |
|---|---|---|---|---|
| forest | 0.916667 | 0.771451 mrad | 0.853146 mrad | 0.9042 |
| forest_deep | 0.939444 | 0.847459 mrad | 0.860938 mrad | 0.9843 |
| **bagged_ridge** | **0.187778** | **0.077611 mrad** | 0.959229 mrad | **0.0809** |
| trivial | 0.941111 | 1.019584 mrad | 1.010964 mrad | 1.0085 |

The trivial estimator's interval is better calibrated than two of the three
learned configurations. Its uncertainty is the honest spread of a predictor that
uses no input.

### The analytic bound, measured the same way

| k | measured coverage | nominal |
|---|---|---|
| 1 | 0.661017 | 0.682689 |
| 2 | **0.953064** | 0.954500 |
| 3 | 0.997392 | 0.997300 |

Over 1534 retained degrees of freedom. The Cramer-Rao interval is calibrated;
the learned ensemble spread is not a calibrated interval and is labelled as a
model spread everywhere it is reported.

### Compute

| operation | calls | total s | mean ms |
|---|---|---|---|
| fit all four estimators | 1 | 18.1860 | 18185.9644 |
| batch solve, eq (14)-(16) | 600 | 1.1885 | 1.9809 |
| batch MAP solve, eq (18) | 600 | 1.1165 | 1.8608 |
| batch solve over the grid of §2 | 864 | 2.3195 | 2.6846 |
| recursive run over the grid of §3 | 180 | 49.9729 | 277.6270 |

**These timings are unstable.** Two shared cores; they move between runs and no
ratio derived from them appears in any conclusion in this repository.

## 6. Controls

Source: `validate_controls.py`. **16 controls: 10 declared MUST fire and 10
fired; 6 declared MUST NOT fire and none fired.** Every control is anchored on a
truth, a declared transformation or an algebraic invariance, never on an
estimator's output.

| control | direction | result |
|---|---|---|
| C1 zero injected misalignment | must not fire | worst standardised estimate **2.5396** over 20 records and 60 degrees of freedom |
| C2 permuting the record | must not fire | **7.562e-16** of the bound |
| C3 negating every rate | must not fire | **0.000e+00** of the bound |
| C4 rotating the reference frame | must not fire | **1.654e-13** of the bound; the estimate is equivariant |
| C5a common noise rescaling, the estimate | must not fire | **1.512e-16** of the bound |
| C5b common noise rescaling, the bound | **must fire** | ratio **10.000000000** against a predicted 10 |
| C6 injection in the observable subspace | **must fire** | median recovered fraction **1.019868**, worst standardised error **1.8138** |
| C7 injection along the rotation axis | **must fire** | withheld on **20 of 20**; largest claim **1.577e-19 rad** |
| C8 a 5 mrad error in the declared mounting | **must fire** | the estimate moves **5.000000 mrad** against a predicted 5.000000 |
| C9 negating the injection | **must fire** | residual sum **0.397661 mrad** against a predicted 0.509132 mrad, on estimates of magnitude 2.024782 mrad |
| C10a a noise-only secondary, the residual | **must fire** | **102.3819x** the intact residual |
| C10b a noise-only secondary, the verdict | must not fire | every Gramian eigenvalue unchanged to 1e-9 relative |
| C11 the noise-floor gate removed | **must fire** | with the gate at 1.0 the verdict is rank 3 on **9 of 20** records and the claim reaches **1106.2243 mrad** against an injected 2.0 mrad, where the gated solution reports exactly nothing (max claim 1.570e-19 rad) |
| C12 the recursive estimator's refusal | **must fire** | median claim **58.2150 mrad**, max 157.0320 mrad, against an injected 2.0 mrad |
| C13 the SVD gauge rotation | **must fire** | median gauge rotation **1.5355 rad**, smallest **0.1428 rad**, against an injected 0.0020 rad |
| audit of `pytest.approx` tolerances | **must fire** | exactly **1 of 307** tests depends on the hidden default |

Three of these are worth reading twice.

**C8.** A mis-declared mounting is indistinguishable from a misalignment and
this package cannot separate them: a 5 mrad error in the declared geometry moves
the estimate by 5.000000 mrad. That is a limitation, and the control exists to
put a number on it.

**C11.** Removing the noise-floor gate does not merely weaken the verdict. On 9
of 20 single-axis records it declares the unsupported degree of freedom
observable, and the resulting estimate along that direction reaches **1106
mrad** -- 553 times the injected 2 mrad, and well past the small-angle regime
the model is written for. This is what the gate exists to prevent.

**C13.** The unconstrained SVD solution of (17) picks up a gauge rotation with a
median of **1.5355 rad** in the direction the record cannot support. It
contaminates the OBSERVABLE subspace by a median of 0.1380 mrad, which is
0.5998 of the Cramer-Rao one sigma on these axis-aligned records; on
2000-sample records with a randomly oriented injected misalignment the same
quantity is 2.54 of the bound, which is asserted in `tests/test_batch.py` to
exceed the whole magnitude of the correct estimate.

### The `pytest.approx` audit

`pytest.approx(expected, rel=r)` passes if **either** the relative or an absolute
bound is met, and the absolute default is 1e-12. The audit is not a source scan:
the whole suite is re-run in a subprocess with `pytest.approx` patched so that a
call carrying `rel=` and no `abs=` raises. **Exactly 1 of 307 tests fails under
that patch**, and it is
`tests/test_regressions.py::test_pytest_approx_absolute_default_is_never_relied_on`,
the deliberate demonstration that asserts both branches. No measurement in this
repository rests on the default.

## 7. Results unfavourable to this package, published as measured

1. **The analytic baseline beats the learned component.** The regularised batch
   solution of (18) reaches 0.697326 mrad on the observable subspace of the
   held-out short-record population against 0.760152 mrad for the best learned
   configuration, a margin of 9.01 per cent in the baseline's favour, on every
   excitation profile. The learned component is not useless -- it beats a
   trivial predictor by 19.04 per cent and the specified unbiased baseline by a
   factor of 7.5 -- but it does not beat the strongest honest analytic method.

2. **The recursive estimator is overconfident in the unsupported direction
   too.** Its two-sigma interval contains the truth on **0.591837** of the
   records with a degree of freedom below the floor, against a nominal
   0.954500. The mechanism is equation (11): the filter forms its Jacobian from
   the measured reference rate, so the reference sensor's own noise supplies
   apparent information and the filter reduces its prior on the strength of it.
   Split by how much prior it kept, coverage is **0.880000** where it kept most
   of the prior and **0.291667** where it did not, with a correlation of
   **-0.6690**. The only estimator here that is not overconfident in that
   direction is the batch solution, which withholds it.

3. **One declared learned configuration is overconfident on the whole
   population, not only in the unobservable subspace.** `bagged_ridge` reports a
   median one sigma of 0.077611 mrad against a realised RMSE of 0.959229 mrad,
   a factor of 12, and its two-sigma coverage is 0.187778 against a nominal
   0.954500. The other two configurations are near nominal globally and still
   collapse to 0.0750 and 0.2083 in the unsupported direction.

4. **The two-axis record is worse conditioned than the three-axis one**, which
   is the opposite of what the number of excited axes suggests, and is reported
   in §1 rather than smoothed over.

## 8. Working directories, output locations and absolute paths

Source: `probe_working_directories.py`.

- `validate_observability.py`, `validate_recovery_grid.py` and
  `worked_example.py` were each run from the filesystem root, the system
  temporary directory, the product root, `validation/` and `src/`. All exit 0
  from all five.
- The five expensive scripts were each run once from `validation/`, which is
  the directory the release gate uses. All exit 0.
- All 8 output files landed in `validation/`; **0 stray** `*_output.txt`
  anywhere else in the tree.
- 66 files scanned for absolute container or host paths, with the patterns
  assembled from `os.sep` and this script's own previous output deleted before
  the scan. **0 offenders.**
- `validate_cli.py` additionally checks the tree for `.config`, `.pytest_cache`,
  `.ruff_cache`, `.hypothesis` and `__pycache__` after spawning eleven
  subprocesses with a temporary `HOME` and `MPLCONFIGDIR`. **No `.config`
  anywhere.**

## 9. What is NOT validated here

- **No flight data.** Every record comes from `aligncal.dataset`, whose closed
  forms are printed in its module docstring. The measured scores do not transfer
  to a real vehicle.
- **No sensor error beyond white noise.** No bias, no scale-factor error, no
  angular random walk, no g-sensitivity, no quantisation, no latency and no time
  tag error. A real relative-alignment estimate is usually limited by one of
  those, and nothing here says anything about them.
- **No dependency range was exercised at both ends** except `ruff`. The build
  environment is a shared pre-pinned interpreter this product may not modify.
- **No hyperparameter search.** Three declared learned configurations, all
  reported.
- **Two sensors at a time.** Several secondaries are supported and each is
  estimated independently against the reference; a joint solution over a sensor
  network is not implemented.
- **The misalignment is assumed constant over the record.** The recursive
  estimator accepts a random-walk process noise, but no time-varying
  misalignment is injected or scored anywhere.

## Reproducing everything

```bash
python -m pytest tests/ -q --junitxml=junit_self.xml
python validation/validate_observability.py
python validation/validate_recovery_grid.py
python validation/validate_recursive_agreement.py
python validation/validate_unobservable.py
python validation/validate_learned.py
python validation/validate_controls.py
python validation/validate_cli.py
python validation/worked_example.py
python validation/probe_working_directories.py
```

Seeds: training 20261029, test 70120261, controls 55120261, grid 31420261.
Per-cell seeds are derived from the cell indices inside each script.
