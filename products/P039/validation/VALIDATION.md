# Validation evidence — latencynet (P039)

Validation level 2. Research-grade. Not flight-qualified, not certified, not
approved for operational aerospace use.

Every number on this page was produced by a script in this directory, run in
this build session on Python 3.13.16 with numpy 2.5.3, scipy 1.18.1 and
scikit-learn 1.9.1. Raw stdout is committed next to each script as
`*_output.txt`, and the machine-readable results as `*.json`. No number here
was typed by hand.

| script | what it establishes | raw output | result |
|---|---|---|---|
| `validate_analytic_exactness.py` | the sum-of-stages model is exact; the Fenton-Wilkinson tail map is not | `validate_analytic_exactness_output.txt` | 19/19 PASS |
| `validate_interval_coverage.py` | measured interval coverage against nominal | `validate_interval_coverage_output.txt`, `interval_coverage.json` | 41/88 rows inside a two-sided 2-sigma band; see below |
| `validate_tail_convergence.py` | tail estimates converge at the order-statistic rate | `validate_tail_convergence_output.txt`, `tail_convergence.json` | 6/6 PASS |
| `validate_model_comparison.py` | the three-way held-out comparison, both regimes | `validate_model_comparison_output.txt`, `model_comparison.json` | reported, no gate |
| `crosscheck_edgeinfer.py` | independent cross-check against P033 EdgeInfer on p50 | `crosscheck_edgeinfer_output.txt`, `crosscheck_edgeinfer.json` | **DISAGREE by -5.8753 %**, accounted for |

Reproduce all five:

```bash
cd validation
PYTHONPATH=../src python3 validate_analytic_exactness.py
PYTHONPATH=../src python3 validate_tail_convergence.py
PYTHONPATH=../src python3 validate_interval_coverage.py
PYTHONPATH=../src python3 validate_model_comparison.py
PYTHONPATH=../src python3 crosscheck_edgeinfer.py
```

## A note on timing measurements

This product predicts latency, and it is built and validated entirely on
**injected** synthetic latency distributions with declared parameters and
fixed seeds. That is deliberate. The only host available is a shared
single-CPU-core container running five concurrent build jobs, where
wall-clock timings move by factors of 2 to 13 between runs. No correctness
assertion anywhere in this product rests on a measured time. The one measured
figure that appears at all is in part (4) of the cross-check, where it is
labelled volatile with its repeat count and measurement method.

## 1. The analytic sum-of-stages model is exact

`validate_analytic_exactness.py`. The model is two identities:

- (1) `E[S] = sum_i E[X_i]` — linearity of expectation, exact under any
  dependence;
- (2) `Var[S] = sum_i Var[X_i] + 2 sum_{i<j} Cov(X_i, X_j)` — exact given the
  covariances, reducing to `sum_i Var[X_i]` when they vanish.

Source for both: Casella & Berger (2002), *Statistical Inference*, 2nd ed.,
Thm. 4.5.6 and Sec. 4.5. The declared exactness tolerance is
`16 * eps = 3.553e-15` relative, which is a floating-point statement and not
an engineering tolerance. It was not widened.

| check | reference | measured relative error | tolerance | result |
|---|---|---|---|---|
| 2 constant stages, predicted mean == realised total | arithmetic | 0.000e+00 | 3.553e-15 | PASS |
| 3 constant stages, predicted p99 == realised total | arithmetic | 0.000e+00 | 3.553e-15 | PASS |
| 1 constant stage, predicted sd exactly zero | arithmetic | exactly 0 | — | PASS |
| independent: `fsum` sum-of-means vs `fsum` mean-of-sums, n=200000 | eq. (1) | 0.000e+00 | 8.882e-16 | PASS |
| correlated rho=0.7: same | eq. (1) | 0.000e+00 | 8.882e-16 | PASS |
| independent: total sum of the sample covariance matrix vs var of row sums | eq. (2) | 1.045e-15 | 3.553e-15 | PASS |
| correlated rho=0.7: same | eq. (2) | 1.430e-16 | 3.553e-15 | PASS |
| independent: residual from dropping covariances == 2 sum_{i<j} Cov_ij | eq. (2) | 6.431e-13 | 1.000e-09 | PASS |
| correlated rho=0.7: same | eq. (2) | 3.741e-16 | 1.000e-09 | PASS |
| declared parameters: end-to-end mean and sd, both regimes | eqs. (1), (2) | 0.000e+00 (4 rows) | 3.553e-15 | PASS |

Two details are reported rather than smoothed over.

**Summation order.** Both sides of the linearity identity are accumulated with
`math.fsum`, which is correctly rounded, so the identity holds at exactly
0.000e+00. Accumulated with numpy's default pairwise summation the two sides
differ by 1.305e-14 (independent) and 5.622e-15 (correlated) relative, against
a per-sum bound of `log2(n) * eps = 3.910e-15` (Higham 2002, *Accuracy and
Stability of Numerical Algorithms*, 2nd ed., Sec. 4.2). That is a
floating-point accumulation fact about summing 200000 terms in two different
orders, and it is printed as such, not asserted as an identity.

**The dropped-covariance residual** is a difference of two near-equal sums and
loses significant digits to cancellation, so it is checked at 1e-9 rather than
at machine precision. The identity it derives from is checked at machine
precision in the row above it.

### What is NOT exact: the Fenton-Wilkinson tail map

Turning two moments into a quantile requires a distribution, and the sum of
lognormals has none in closed form. The package uses the Fenton-Wilkinson
moment-matched lognormal (Fenton 1960, *IRE Transactions on Communications
Systems* 8(1): 57-67). Its error is measured against 2,000,000 injected draws
per pipeline and reported, never asserted away:

| pipeline | p50 | p99 | p99.9 | MC one-sigma at p99.9 |
|---|---|---|---|---|
| independent, cv about 0.2 | +0.272 % | **-1.055 %** | **-2.359 %** | 0.093 % |
| correlated rho=0.7 | +0.017 % | -0.140 % | -0.346 % | 0.131 % |
| one dominant skewed stage (cv 0.6) | +0.864 % | -1.562 % | **-4.346 %** | 0.353 % |
| five low-cv stages (cv 0.05) | -0.002 % | -0.001 % | -0.004 % | 0.019 % |

The sign is the finding: the two-moment lognormal match **under-predicts** the
tail in every case, and the error grows with the skewness of the dominant
stage and with how far into the tail you go. Under-prediction is the unsafe
direction for a deadline. This is the whole of the analytic predictor's
systematic error — equations (1) and (2) contribute none — and it is why a
fitted model can beat it even where the stages are independent.

## 2. Prediction-interval coverage against nominal

`validate_interval_coverage.py`. Splits: 200 train / 60 calibration / 150
test pipelines, probe `n = 256` passes, reference samples 25,000 passes for
train and calibration and 90,000 for test, seed 20260402. Nominal levels
0.50, 0.80, 0.90; targets p99 and p99.9; both regimes.

Measured coverage is reported with its binomial standard error
`sqrt(c(1-c)/n)`; at nominal 0.90 with n=150 that is 0.0245. The declared
acceptance band, fixed before the run, is two standard errors either side of
nominal.

Summary over 88 rows:

| interval | rows inside the two-sided 2-sigma band |
|---|---|
| native (each model's own) | 9 / 40 |
| split-conformal | 32 / 48 |
| split-conformal, one-sided guarantee `measured >= nominal - 2 SE` | **47 / 48** |

Of the 16 conformal rows outside the two-sided band, 15 are *over*-coverage.
That is the direction the split-conformal guarantee permits: its coverage lies
in `[level, level + 1/(m+1)]` in expectation (Lei et al. 2018, *JASA*
113(523), Sec. 2.2), i.e. up to 0.0164 above nominal with m=60 calibration
pipelines, and the symmetric band makes no allowance for that offset. The
one-sided row is therefore the right reading of the guarantee, and it holds on
47 of 48 rows.

Representative rows at nominal 0.90 (n=150, SE 0.0245 at nominal):

| regime | target | model | interval | measured | SE | deviation | result |
|---|---|---|---|---|---|---|---|
| independent | p99 | analytic_sum_indep | native | 0.627 | 0.039 | -6.92 SE | **FAIL** |
| independent | p99 | analytic_sum_indep | conformal | 0.933 | 0.020 | +1.64 SE | PASS |
| independent | p99 | linear_ols | native | 0.860 | 0.028 | -1.41 SE | PASS |
| independent | p99 | learned_gbt | native | 0.700 | 0.037 | -5.35 SE | **FAIL** |
| independent | p99 | learned_gbt | conformal | 0.927 | 0.021 | +1.25 SE | PASS |
| independent | p99.9 | analytic_sum_indep | native | 0.413 | 0.040 | -12.10 SE | **FAIL** |
| correlated | p99 | analytic_sum_indep | native | 0.107 | 0.025 | -31.48 SE | **FAIL** |
| correlated | p99 | analytic_sum_indep | conformal | 0.900 | 0.024 | 0.00 SE | PASS |
| correlated | p99 | linear_ols | native | 0.907 | 0.024 | +0.28 SE | PASS |
| correlated | p99.9 | analytic_sum_indep | native | 0.093 | 0.024 | -33.96 SE | **FAIL** |

**The analytic model's native interval fails at every level, in every regime,
on all 12 rows.** This was stated in `latencynet.analytic` before it was
measured and it is not a surprise: that interval propagates probe sampling
uncertainty through the GUM law of propagation (JCGM 100:2008, Sec. 5.1.2) and
contains no term for Fenton-Wilkinson model error and, in independence mode,
no term for stage dependence. An interval that claims 90 % and delivers 9.3 %
is a defect in the interval. It is reported as FAIL and the tolerance was not
loosened; the usable alternative is the conformal wrapper, which covers 0.900
against nominal 0.900 on the same pipelines.

The learned model's native quantile-loss heads also under-cover (0 of 4 rows
inside the band). The OLS Student-t interval is the only native interval that
holds up (9 of 12 rows), which is unsurprising: its assumptions — linearity in
the features and homoscedastic Gaussian error in log space — happen to be
nearly true for this population.

## 3. Tail-latency convergence at the order-statistic rate

`validate_tail_convergence.py`. Predicted: the sample quantile is
asymptotically normal with
`SE(q_hat_p) = sqrt(p(1-p)/n) / f(q_p)` (Mosteller 1946, *Annals of
Mathematical Statistics* 17(4): 377-408; David & Nagaraja 2003, *Order
Statistics*, 3rd ed., Sec. 10.2), so RMSE decays as `n^(-1/2)` and the
log-log slope is exactly -0.5.

Declared tolerances, fixed before the run: slope within ±0.04 of -0.5, and
magnitude ratio within ±0.15 of 1.0 at the largest n. The theory's own
validity condition `n(1-p) >= 5` is applied to every fitted grid.

| case | target | fitted slope | slope SE | deviation | predicted | result |
|---|---|---|---|---|---|---|
| single lognormal (mean 200 us, cv 0.45), 400 repeats | p99 | **-0.49583** | 0.00971 | +0.43 SE | -0.5 | PASS |
| single lognormal, 400 repeats | p99.9 | **-0.52648** | 0.01386 | -1.91 SE | -0.5 | PASS |
| three-stage pipeline, 300 repeats, truth from 4,000,000 passes | p99 | -0.47492 | 0.01317 | +1.90 SE | -0.5 | PASS |
| three-stage pipeline, 300 repeats | p99.9 | -0.51934 | 0.02082 | -0.93 SE | -0.5 | PASS |

Magnitude, measured RMSE divided by the closed-form order-statistic standard
error, on the single lognormal where the density at the quantile is exact:

| target | n | measured RMSE | analytic SE | ratio |
|---|---|---|---|---|
| p99 | 500 | 36.020 us | 35.508 us | 1.0144 |
| p99 | 50,000 | 3.599 us | 3.551 us | **1.0136** |
| p99.9 | 5,000 | 40.482 us | 39.195 us | 1.0328 |
| p99.9 | 150,000 | 6.646 us | 7.156 us | **0.9287** |

The slope standard error quoted above is propagated from the repeat count:
with R repeats the log-RMSE has standard deviation `1/sqrt(2R)`, so the
least-squares slope has standard error `1/(sqrt(2R) sqrt(Sxx))`. Every slope
lies within 2 of its own standard errors of -0.5.

### Where the theory stops working, measured

Section (c) deliberately evaluates p99.9 at grid points *below* the validity
condition:

| n | expected exceedances n(1-p) | measured RMSE | analytic SE | ratio |
|---|---|---|---|---|
| 200 | 0.20 | 139.18 us | 195.97 us | **0.7102** |
| 500 | 0.50 | 93.90 us | 123.94 us | 0.7576 |
| 1,500 | 1.50 | 61.80 us | 71.56 us | 0.8636 |

The fitted slope over that out-of-range grid is -0.40220. The asymptotic
expression over-states the error there because the estimator is pinned near
the sample maximum and cannot fluctuate upward. This is the validity range of
a cited result, measured rather than assumed, and it is the reason the fitted
grids start at five expected exceedances.

**Practical consequence.** To estimate p99.9 of a pipeline to within 1 % you
need roughly 150,000 passes (the measured RMSE/q at n=150,000 is 0.00967 for
the cv-0.45 lognormal). That is the expense this package exists to avoid.

## 4. The three-way comparison on held-out pipelines

`validate_model_comparison.py`. Splits: 200 train / 60 calibration / 150 test
pipelines per regime, probe `n = 256`, reference 25,000 passes for
train/calibration and 90,000 for test, seed 20260402. Winner rule, fixed in
`latencynet.compare` before the run: smallest mean absolute log error on the
test split, with significance by a paired t-test against the analytic
baseline at p < 0.05.

Mean absolute log error is, to first order, the mean relative error of the
predicted quantile.

### Independent stages — equation (2) is exact here

| target | model | mean \|dln q\| | median | bias | native cov. | conformal cov. | conformal width | p vs baseline |
|---|---|---|---|---|---|---|---|---|
| p99 | analytic_sum_indep (baseline 1) | 0.03655 | 0.02525 | -0.0227 | 0.627 | 0.933 | 0.1863 | ref |
| p99 | linear_ols (baseline 2) | **0.03300** | 0.02253 | +0.0060 | 0.860 | 0.880 | 0.1342 | 0.1306 |
| p99 | learned_gbt | 0.05167 | 0.03552 | +0.0093 | 0.700 | 0.927 | 0.2474 | 0.0009 |
| p99 | analytic_sum_cov (diagnostic) | 0.03725 | 0.02629 | -0.0221 | 0.233 | 0.940 | 0.1928 | 0.4266 |
| p99.9 | analytic_sum_indep | 0.07086 | 0.06137 | -0.0592 | 0.413 | 0.953 | 0.3429 | ref |
| p99.9 | linear_ols | **0.05005** | 0.03652 | +0.0041 | 0.873 | 0.880 | 0.1978 | 0.0000 |
| p99.9 | learned_gbt | 0.06080 | 0.04349 | +0.0076 | 0.520 | 0.887 | 0.2908 | 0.0988 |
| p99.9 | analytic_sum_cov | 0.07154 | 0.06148 | -0.0584 | 0.053 | 0.967 | 0.3515 | 0.5833 |

Reference-target Monte Carlo relative standard error on the test split: 0.304 %
at p99, 0.876 % at p99.9.

**The learned model does not beat the analytic baseline at p99 under
independence** (0.05167 against 0.03655, paired p = 0.0009 against it — a
significant *loss*). At p99.9 it is nominally ahead (0.06080 against 0.07086)
but not significantly so (p = 0.0988). That is the outcome the specification
predicted.

### Correlated stages — equation (2) is wrong here

| target | model | mean \|dln q\| | median | bias | native cov. | conformal cov. | conformal width | p vs baseline |
|---|---|---|---|---|---|---|---|---|
| p99 | analytic_sum_indep (baseline 1) | 0.15441 | 0.14950 | **-0.1530** | 0.107 | 0.900 | 0.5195 | ref |
| p99 | linear_ols (baseline 2) | **0.03155** | 0.02398 | +0.0057 | 0.907 | 0.913 | 0.1466 | 0.0000 |
| p99 | learned_gbt | 0.05094 | 0.04153 | +0.0183 | 0.680 | 0.927 | 0.2500 | 0.0000 |
| p99 | analytic_sum_cov (diagnostic) | 0.03317 | 0.02466 | -0.0145 | 0.193 | 0.953 | 0.1744 | 0.0000 |
| p99.9 | analytic_sum_indep | 0.22493 | 0.21619 | **-0.2233** | 0.093 | 0.887 | 0.7245 | ref |
| p99.9 | linear_ols | **0.04107** | 0.03100 | +0.0072 | 0.953 | 0.960 | 0.2468 | 0.0000 |
| p99.9 | learned_gbt | 0.05546 | 0.04451 | +0.0151 | 0.733 | 0.987 | 0.3641 | 0.0000 |
| p99.9 | analytic_sum_cov | 0.04916 | 0.03820 | -0.0340 | 0.067 | 0.973 | 0.2902 | 0.0000 |

Injected latent correlation across the 410 pipelines: min 0.354, mean 0.623,
max 0.900. Stages per pipeline: 2 to 6, mean 3.94. Reference-target relative
standard error: 0.374 % at p99, 1.001 % at p99.9.

**The learned model beats the analytic baseline here, at both probabilities,
with p < 0.0001.** That too is what the specification predicted. The analytic
baseline's signed bias of -0.1530 and -0.2233 means it under-predicts the tail
by 14 % and 20 % on average — again the unsafe direction.

### Which case does the data fall in

| regime | target | winner by the declared rule | significant vs baseline 1 |
|---|---|---|---|
| independent | p99 | linear_ols | no (p = 0.1306) |
| independent | p99.9 | linear_ols | yes |
| correlated | p99 | linear_ols | yes |
| correlated | p99.9 | linear_ols | yes |

Both expectations in the specification hold: the analytic baseline is
competitive where stages are independent and the learned model does not help
there; the learned model does help where they are dependent. But **neither is
the winner. The linear regression — baseline 2, thirteen features, closed-form
fit — wins all four comparisons.** The learned gradient-boosted model never
beats it.

### Capacity sweep — is that an artefact of one hyperparameter choice?

Section (c) refits the learned model at six settings spanning a factor of 16
in ensemble size, at p99, in both regimes. Reported in full:

| n_estimators | max_depth | learning_rate | independent | correlated |
|---|---|---|---|---|
| 50 | 2 | 0.05 | 0.08990 | 0.07479 |
| 100 | 3 | 0.05 | 0.05386 | 0.05257 |
| 300 | 3 | 0.05 (the shipped default) | 0.05167 | 0.05094 |
| 300 | 2 | 0.10 | 0.05838 | 0.05592 |
| 300 | 5 | 0.05 | 0.05775 | 0.05270 |
| 800 | 3 | 0.02 | 0.05196 | 0.05098 |
| *analytic_sum_indep for comparison* | | | *0.03655* | *0.15441* |
| *linear_ols for comparison* | | | *0.03300* | *0.03155* |

Across 12 settings the learned model beat the analytic baseline on 6 — exactly
the 6 correlated-regime settings, and none of the independent-regime ones —
and beat the linear baseline on **0 of 12**. The conclusion is not an artefact
of the shipped hyperparameters. The shipped setting is in fact the best of the
six in both regimes, so the learned model is not being handicapped by the
choice.

Impurity-based feature importances of the shipped learned model at p99
(biased toward high-cardinality features, Strobl et al. 2007, *BMC
Bioinformatics* 8:25 — reported as what the model leaned on, not as a causal
claim):

| feature | independent | correlated |
|---|---|---|
| `log_sum_stage_p99` | 0.8513 | 0.8439 |
| `log_sum_mean` | 0.0904 | 0.0925 |
| `log_indep_sd` | 0.0374 | 0.0034 |
| `log_dep_sd` | 0.0163 | 0.0561 |

The model does find the dependence-bearing feature in the correlated regime
(`log_dep_sd` rises from 0.0163 to 0.0561 while `log_indep_sd` collapses from
0.0374 to 0.0034), which is the right behaviour. It simply does not convert it
into an advantage over a linear fit on the same features.

## 5. Cross-check against P033 EdgeInfer — DISAGREEMENT, reported

`crosscheck_edgeinfer.py`. P033 published
`products/P033/validation/crosscheck_pipeline.json`: three lognormal stages
(60/180/30 us mean, 12/40/6 us sd), n = 2000, seed 20260401, and nominated
`p50_s` as the comparable statistic because its recorded mean was
preemption-contaminated. This implementation rebuilds the pipeline from the
declared parameters — not from P033's file — and computes p50 independently.

### Agreement on the declared pipeline and the injected cost model

| quantity | latencynet | P033 | relative difference | tolerance | result |
|---|---|---|---|---|---|
| all three stage means and sds, names, dist | matched | matched | exact | — | MATCH |
| n_samples, seed | 2000, 20260401 | 2000, 20260401 | — | — | MATCH |
| injected end-to-end mean | 270.000000000 us | 270.000000000 us | **0.000e+00** | 3.553e-15 | PASS |
| injected end-to-end sd | 42.190046219 us | 42.190046219 us | **0.000e+00** | 3.553e-15 | PASS |

### The p50 diff, as instructed

| quantity | value |
|---|---|
| latencynet sampled p50 (n=2000, seed 20260401) | **267.184676 us** |
| P033 measured p50 | **283.862500 us** |
| relative difference (ours - theirs) / theirs | **-5.8753 %** |
| declared agreement band (P033's "within a few per cent") | ±3 % |
| verdict | **DISAGREE** |

`latencynet`'s implementation was **not** changed to close this gap.

### Accounting for the disagreement

The two p50s are different quantities, and this was written down before the
diff was run. P033's `p50_s` is a *measured* wall-clock median: it drew the
stage durations, consumed each with a spin-wait, and timed the pass with
`perf_counter`, so it carries P033's spin-wait overshoot and harness cost.
`latencynet`'s p50 is the median of the *sampled* cost model, with no clock
involved.

| quantity | value |
|---|---|
| distributional median of the sum, from 4,000,000 draws | 266.074198 us |
| that median / injected mean | 0.985460 |
| latencynet n=2000 p50 / injected mean | 0.989573 |
| P033 measured p50 / injected mean (from their JSON) | 1.051343 |
| implied P033 measurement overhead on the median | **+6.6855 %** |
| observed gap, theirs over ours at n=2000 | **+6.2421 %** |
| **unexplained residual after accounting** | **+0.4174 %** |

A sum of right-skewed lognormals has its median 1.4546 % *below* its mean, so
a correct sampled p50 must sit below the injected mean; P033's measured p50
sits 5.1343 % *above* it. The whole of the 6.24 % gap is accounted for by
P033's own published `p50_over_injected_mean` ratio against the distributional
median ratio, to within 0.42 %. That residual is itself consistent with the
n=2000 sampling noise on a median (one-sigma of order 0.4 % here).

**Finding.** The two implementations agree exactly on the injected cost model
and agree to within 0.42 % on the *distribution*. They disagree by 5.88 % on
the published p50 figures because one is measured through a spin-wait on a
contended core and the other is not. The comparable statistic P033 nominated
is the right one for comparing implementations, but it must be compared
against a sampled median, not a measured one, for the two numbers to mean the
same thing. Neither implementation is wrong; the published figures are not
like-for-like.

### Volatile like-for-like measured figure

For completeness, `latencynet` also measured the same pipeline through its own
spin-wait. **WORKSTATION NUMBER. VOLATILE.**

| quantity | committed run | an earlier run of the same script |
|---|---|---|
| latencynet measured p50 | 271.880000 us | 274.433000 us |
| overhead over latencynet's sampled p50 | +1.7573 % | +2.7129 % |
| P033 measured p50 | 283.862500 us | 283.862500 us |
| measured-vs-measured relative difference | **-4.2212 %** | -3.3219 % |
| latencynet measured mean | 291.257509 us (1.08x injected) | 600.772007 us (2.23x injected) |

Measurement method: `time.perf_counter` bracket around a spin-wait that
consumes each drawn stage duration in turn; 2000 repeats, 50 warm-up
discarded; garbage collection not disabled; one shared single CPU core with
four other build jobs running concurrently; Linux 6.18.44, Python 3.13.16.

Both columns are from this session, from the same script and the same seed,
minutes apart. The measured *mean* moved from 2.23x the injected mean to
1.08x between them — a factor of two in the contamination — while the
deterministic sampled figures in parts (1) to (3) are bit-identical across
both runs. That is the entire argument for building this product on injected
distributions, demonstrated rather than asserted, and it is why **no claim in
this product rests on these figures**. They are reported because a
measured-vs-measured comparison is what a reader would otherwise ask for, and
it narrows the disagreement from -5.88 % to -4.22 %, consistent with the
accounting above.

## Checks that FAILED

Listed together so they cannot be missed.

1. **The analytic model's native prediction interval under-covers on all 12
   rows**, by up to 33.96 binomial standard errors (nominal 0.90, measured
   0.093, correlated regime, p99.9). Cause: the GUM propagation covers probe
   sampling uncertainty only. Documented in `latencynet.analytic` in advance.
   Not fixed by widening a tolerance; the conformal wrapper is the usable
   interval.
2. **The learned model's native quantile-loss interval under-covers on all 4
   rows measured** (nominal 0.90, measured 0.520 to 0.733).
3. **The covariance-aware analytic variant's native interval under-covers on
   all 12 rows** (measured 0.027 to 0.233), worse than the independence
   variant, because its point prediction is better while its uncertainty term
   is unchanged.
4. **The P033 cross-check DISAGREES on p50 by -5.8753 %**, outside the ±3 %
   band. Accounted for to within +0.4174 % by P033's own published ratios; the
   implementation was not altered.
5. **The analytic baseline loses the overall comparison in all four
   regime/probability combinations.** Its moment identities are exact to
   0.000e+00; its loss comes entirely from the Fenton-Wilkinson tail map,
   which under-predicts p99.9 by 2.4 % to 4.3 % depending on pipeline shape,
   and from ignoring stage dependence in the correlated regime.
6. **The learned model loses to the linear-regression baseline in 12 of 12
   capacity settings.** On this dataset the learned predictor is not justified
   over a thirteen-feature OLS fit.

Nothing in this section was discovered after a tolerance was adjusted. Every
tolerance quoted was fixed in the script before the run that produced the
committed output.
