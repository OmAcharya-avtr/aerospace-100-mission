# Validation evidence - `linkoutage` 0.1.0

Validation level 2 (research grade). Every number below came from running the
named script in this directory during the build session on 2026-10-06; the raw
stdout of each run is committed beside it as `*_output.txt`. Nothing here is
quoted from a paper, estimated, or carried over from another product.

Environment: Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1,
2 CPU cores and 7.8 GiB of RAM shared with four concurrent build agents.

Reproduce everything:

```bash
cd validation
python validate_series_construction.py
python validate_fade_definitions.py
python validate_analytic_lcr.py
python validate_cross_check_x1.py
python validate_distribution_fits.py
python validate_markov_gof.py
python validate_outage_classifier.py
python regenerate_outage_model.py
python worked_example.py
```

Each script runs standalone from this directory; `_bootstrap.py` puts `../src`
on `sys.path`, so no environment variable is required.

## Summary table

| # | Check | Reference | Result | Tolerance | Verdict |
|---|---|---|---|---|---|
| 1 | AR(1) filter against the direct recursion, 2e6 samples | definition, `channel.ar1_unit_variance` | max abs difference 3.11e-15 over 50 000 samples; 0.0 over all 2e6 by the independent reconstruction; `x[0] - z[0] = 0` | < 1e-12 | pass |
| 2 | Unit variance and lag-one correlation of the filtered series | AR(1) definition | variance 1.00689 (se 0.02), lag-one 0.99504 against rho = 0.99501 | 4 se; 1e-3 | pass |
| 3 | Lognormal mapping | Andrews & Phillips (2005) | mean irradiance 0.998908 against 1.0; `var(ln I)` 0.473240 against `ln 1.6` = 0.470004 | 0.01 | pass |
| 4 | Amplitude threshold maps exactly onto a Gaussian level | strict monotonicity of `x -> a` | 0 of 2 000 000 samples disagree between `a < T` and `x < u` | exactly 0 | pass |
| 5 | Hand-calculated fade statistics, 7-sample series | arithmetic shown in the script | all 8 quantities exact | 1e-12 | pass |
| 6 | Hand-calculated censored case, 5-sample series | arithmetic shown in the script | 1 down-crossing, 1 up-crossing, 2 runs, 0 complete fades, mean is `nan`, LCR 0.5 Hz, outage 0.6 | 1e-12 | pass |
| 7 | Interpolated crossing instants | linear interpolation, worked by hand | 0.666666666666666 against 2/3 | 1e-12 | pass |
| 8 | Rice partition identity | Rice (1945), discrete-record form | residual equals `1/(N-1)` exactly on three records (N = 7, 495, 19 995) | 1e-12 | pass |
| 9 | Bivariate-normal orthant probability, two independent routes | `multivariate_normal.cdf` vs 1-D `quad` | worst absolute disagreement 8.33e-17 over 24 (u, rho) pairs | < 1e-11 | pass |
| 10 | Orthant probability at rho = 0 | `(1 - Phi(u)) Phi(u)` | max abs difference 2.8e-17 over four levels | < 1e-10 | pass |
| 11 | Sample level-crossing rate vs the analytic rate | `channel.analytic_level_crossing_rate` | 9 configurations, max relative gap 7.53 %, max block-bootstrap z score 2.42, 0 of 9 cells beyond 3 sigma | all abs(z) <= 3 | pass |
| 12 | Convergence of the sample rate | sampling theory | rms relative gap 14.42 %, 9.65 %, 3.29 %, 1.94 % at N = 5e4, 2e5, 8e5, 3.2e6 (5 independent seeds each) | monotone decrease | pass |
| 13 | Sample mean fade duration vs the analytic value | Rice relation on the model | 4 configurations, relative gaps +1.02 %, -0.05 %, -1.07 %, +2.26 % against standard errors of 2.0-4.7 % | within ~1 se | pass |
| 14 | **Cross-check X1 input: mean irradiance** | Batch 05 spec, 0.998907972681559 | **0.9989079726815594**, absolute difference 4.44e-16 | < 1e-12 | pass |
| 15 | **Cross-check X1: level-crossing rate** | sample statistic of the pinned series | **8191.004095502048 Hz** | compared by the coordinating session against P041, 2 % | reported |
| 16 | **Cross-check X1: mean fade duration** | sample statistic of the pinned series | **1.5497375167867175e-05 s** | compared by the coordinating session against P041, 2 % | reported |
| 17 | Memoryless fade-duration hypothesis, positive control | geometric samples, p = 0.02, 0.06, 0.2 | chi-square p-values 0.305, 0.099, 0.842; exponential MLE recovers 4.0 as 4.0043 | p > 0.01 | pass |
| 18 | **Memoryless fade-duration hypothesis, measured channel** | chi-square on sample counts | chi-square 62510.3 on 90 dof, p = 0 at T = 0.6; rejected at all four thresholds | alpha = 0.01 | **REJECTED** |
| 19 | Fade-duration dispersion | exponential has cv = 1 | measured cv 2.571 at T = 0.6 (2.11-2.76 across thresholds) | n/a, descriptive | memoryless fails |
| 20 | Fade-duration tail | geometric survival | measured/memoryless survival ratio 2.1 at 50 samples, 27.5 at 100, 6226 at 200, 4.7e8 at 400 | n/a, descriptive | memoryless fails |
| 21 | Best fade-duration law by AIC | AIC over five candidates | lognormal (AIC -347241) ahead of Weibull (-341293) and gamma (-337757) | n/a | lognormal |
| 22 | Right-censoring bias on a short record | exponential MLE with and without censored observations | 2000-sample record: scale 1.632e-05 s with censoring, 7.947e-06 s ignoring it, -51.3 % bias | n/a, descriptive | censoring matters |
| 23 | Markov fit, positive control | simulated chain `[[0.92, 0.08], [0.015, 0.985]]` | max abs error 3.86e-04; dwell chi-square p = 0.498 and 0.214; order test p = 0.094, Cramer's V 0.0022 | recovered, not rejected | pass |
| 24 | Markov order test, negative control | explicitly second-order sequence | chi-square 270761 on 2 dof, p = 0, Cramer's V 0.520 | rejected | pass |
| 25 | **Markov dwell times, measured channel** | geometric implied by the fitted `P[i,i]` | state 0: chi-square 62525.6 on 91 dof, cv 2.571 against 0.967; state 1: chi-square 241872 on 446 dof, cv 3.154 against 0.995 | alpha = 0.01 | **REJECTED** |
| 26 | **Markov property, measured channel** | conditional-independence test | chi-square 139206 on 2 dof, p = 0, **Cramer's V = 0.2638** | effect size reported, not p | **not Markov** |
| 27 | Markov fit reproduces the mean dwell | maximum likelihood | model 15.4966 / 106.5876 samples, measured 15.4974 / 106.5660 | n/a | mean fitted, shape not |
| 28 | More states reduce the departure | same test at K = 2, 3, 4, 5, 7 | Cramer's V 0.2638, 0.2173, 0.1866, 0.1575, 0.1304 | n/a | monotone, never zero |
| 29 | Semi-Markov reproduces the dwell tail | simulation against the record | occupancy 0.1269 vs 0.1288, mean dwell 15.50 vs 15.71, P(dwell > 100) 0.0349 vs 0.0353 | n/a | reproduces, at 1553 parameters against 2 |
| 30 | **Outage classifier against three baselines** | held-out test split, 543 onsets | Brier 0.0217557 (analytic + Platt), 0.0220142 (logistic), 0.0222801 (forest + Platt), 0.0223498 (forest), 0.0264495 (constant), 0.0265626 (raw analytic) | n/a | **a baseline wins** |
| 31 | Raw analytic forecaster is worse than doing nothing on Brier | constant base-rate reference | Brier skill score **-0.0043** despite the highest AUC (0.8730) | n/a | published as a failure |
| 32 | Accuracy is uninformative at this base rate | all-negative reference | every model scores 0.970-0.973 against 0.9728 for always saying "no" | n/a | as expected |
| 33 | Baseline wins at every horizon | 50, 100, 200, 400 samples | analytic + Platt lowest Brier at all four | n/a | robust |
| 34 | Forest uncertainty output is informative | decile analysis | calibration gap 0.0029 (lowest decile) to 0.0246 (highest); Spearman 0.455, **p = 0.187** | significance at 0.05 | **not established** |
| 35 | Model regeneration is deterministic | refit from seed 4901 | all six forecasters reproduce the published test Brier to within 1.0e-7 | 5e-6 | pass |
| 36 | Quantisation changes the strict/non-strict answer | 8-, 10-, 12-bit ADC | LCR differs by +4.74 %, +1.31 %, +0.31 % between the two rules | n/a, descriptive | definitions matter |

## Cross-check X1, in full

Specified in `batch_reports/BATCH_05_SPEC.md`: the level-crossing rate and mean
fade duration of a pinned seeded series must agree between P041 CodedFade and
P049 LinkOutage to within 2 % relative, with the sample count stated.

**The series** (constructed exactly as specified, by
`validation/validate_cross_check_x1.py`):

| Parameter | Value |
|---|---|
| Model | lognormal irradiance, AR(1) Gauss-Markov log-amplitude |
| Driving noise | `numpy.random.default_rng(41).standard_normal(2000000)`, one call |
| n | 2 000 000 |
| fs | 1.0e6 Hz |
| tau | 2.0e-4 s |
| rho = exp(-1/(tau fs)) | 0.9950124791926823 |
| Correlation length | 200.0 samples |
| SI | 0.6 |
| sigma_lnI^2 = ln(1+SI) | 0.4700036292457356 |
| E[I] | 1.0 by construction |
| Threshold | amplitude 0.6 |

**Input check.** Computed mean irradiance **0.9989079726815594** against the
specified reference **0.998907972681559**; absolute difference 4.44e-16,
relative 4.45e-16. The series is reproduced. The script refuses to report any
statistic if this check fails.

**Compared quantities, as sample statistics of this series:**

| Quantity | Value |
|---|---|
| **Level-crossing rate** | **8191.004095502048 Hz** |
| **Mean fade duration** | **1.5497375167867175e-05 s** |
| Sample count N | 2 000 000 |
| Record duration (N-1)/fs | 1.999999 s |
| Down-crossings | 16 382 |
| Up-crossings | 16 383 |
| Fade runs, all | 16 383 |
| Complete fades, entering the mean | 16 382 |
| Censored runs, excluded | 1 (left-censored; 0 right-censored) |
| Single-sample fades, counted | 5198 |
| In-fade samples | 253 881 |
| Outage fraction | 0.1269405 |
| Availability | 0.8730595 |
| Median complete fade | 3e-06 s |
| Longest complete fade | 0.00069 s |

**Definitions in force** (these are the published conventions and they are
printed in the raw output beside the numbers):

* in fade: `a[n] < T`, strict;
* down-crossing: `a[n-1]` not in fade and `a[n]` in fade, for `n = 1 .. N-1`.
  A record that starts below the threshold yields no down-crossing at index 0;
* fade run: a maximal run of consecutive in-fade samples;
* duration: `L / fs` for a run of `L` below-threshold samples;
* **censoring:** a run touching sample 0 or sample `N-1` is counted, reported,
  and **excluded** from the mean;
* **single-sample rule:** a single below-threshold sample **is** a fade, of
  duration `1/fs`;
* level-crossing rate: counted down-crossings divided by `(N-1)/fs`;
* outage fraction: in-fade samples divided by `N`, over the whole record
  including censored runs.

**Sampling error of the compared quantities:** 16 382 crossings, so a Poisson
relative standard error of 0.78 % on the rate, which understates the truth
because crossings of a correlated process clump; and a standard error of
3.11e-07 s (2.01 % relative) on the mean fade duration.

**Definitional sensitivity**, so that a disagreement with P041 can be traced
rather than argued about:

| Variant | LCR [Hz] | dLCR | MFD [s] | dMFD |
|---|---|---|---|---|
| published | 8191.004096 | +0.00 % | 1.549737517e-05 | +0.00 % |
| interval-count durations | 8191.004096 | +0.00 % | 1.449737517e-05 | -6.45 % |
| interpolated crossing instants | 8191.004096 | +0.00 % | 1.546564684e-05 | -0.20 % |
| censored runs folded in as complete | 8191.004096 | +0.00 % | 1.549661234e-05 | -0.00 % |
| **single-sample excursions not fades** | **5592.002796** | **-31.73 %** | **2.223533619e-05** | **+43.48 %** |
| non-strict `a <= T` | 8191.004096 | +0.00 % | 1.549737517e-05 | +0.00 % |
| record duration `N/fs` | 8191.000000 | -0.00 % | 1.549737517e-05 | +0.00 % |

The single-sample rule is by far the dominant choice. If P041 disagrees by
roughly a third on the rate and roughly 43 % on the mean fade duration, the
cause is that rule and nothing else. Any other disagreement above about 2 % is
not explained by any definitional choice available here.

**Context, explicitly not the comparand:** the analytic discrete-time
level-crossing rate of the model is 8228.471843706175 Hz and its analytic mean
fade duration is 1.526401989602934e-05 s, so this realisation sits -0.455 % and
+1.53 % from the model. Those are model-versus-realisation gaps for one finite
sample path, not the cross-check. No wall-clock measurement appears anywhere
in this product.

## Where a baseline won, or a check did not come out clean

These are the credible rows.

* **The outage classifier is beaten by a baseline.** The analytic
  level-crossing-rate forecaster with a two-parameter Platt recalibration has a
  held-out Brier score of 0.0217557 against 0.0223498 for the 200-tree random
  forest and 0.0222801 for the recalibrated forest. Logistic regression
  (0.0220142) also beats both forest variants. The forest is fourth of six.
  It has not been retuned; the only hyperparameter search performed was over
  three values of `min_samples_leaf` on the calibration split, and all three
  are reported.
* **The baseline wins at every horizon tested** (50, 100, 200 and 400 samples),
  so this is not an artefact of the horizon chosen for the headline.
* **The raw analytic forecaster has a negative Brier skill score** (-0.0043):
  it is worse than emitting the base rate, despite having the best
  discrimination in the table (AUC 0.8730, average precision 0.3193). Its
  reliability term is 4.09e-03 against 3.19e-06 for the constant baseline. The
  cause is identified (Poisson clumping overstates `P(at least one crossing)`
  when crossings arrive in bursts) and is fixed by recalibration, but the raw
  row is published rather than dropped.
* **The forest's uncertainty output is not established.** The Spearman rank
  correlation between ensemble spread and calibration gap over ten deciles is
  0.455 with a p-value of 0.187. The trend is in the right direction and the
  top decile's gap is an order of magnitude above the bottom decile's, but ten
  points cannot establish it. Shipped with that caveat.
* **The analytic model is fitted to the family that generated the data.** The
  analytic baseline's advantage is therefore an upper bound on what it would
  achieve on a measured record. This is stated, not corrected for, because
  there is no measured record here to correct against.
* **The Poisson z scores in the analytic-rate table were wrong at first.** The
  original version of `validate_analytic_lcr.py` used `1/sqrt(n_crossings)` as
  the standard error, which assumes independent crossings and produced z scores
  up to 6.4 and a false claim in the script's own output that every cell was
  within 3 %. It was replaced by a block estimate (blocks of 20 correlation
  lengths), under which the worst z score is 2.42 and the worst relative gap is
  7.53 %. The relative gap is genuinely that large on the deepest threshold;
  the block standard error there is 5.64 %, which is why the z score and not
  the relative gap is the criterion.
* **The convergence table was initially misleading.** It used nested prefixes
  of one realisation, so the rows were a single correlated random walk rather
  than a convergence. It now uses five independent seeds per record length.
* **At two million samples every goodness-of-fit p-value is zero.** This is
  reported as such, and the Markov results are decided on Cramer's V (0.2638)
  and on the dwell coefficient-of-variation ratio (up to 3.17) rather than on a
  p-value that carries no information at that sample size.
* **Beyond about seven channel states the Markov-order test loses its
  usable middle states** to sparsity, because with rho = 0.995 a single sample
  almost never moves more than one state. The K = 12 point was computed, found
  to rest on one usable middle state, and removed from the published figure
  rather than quoted.

## Known-answer tests

The hand-calculated answers are in `validate_fade_definitions.py` with their
arithmetic written out, and again in `tests/test_fade.py` and
`tests/conftest.py` so that they run on every commit. The 7-sample series
`[1.0, 0.5, 0.5, 1.0, 1.0, 0.4, 1.0]` at threshold 0.6 and 1 Hz gives 2
down-crossings, 2 up-crossings, 2 complete fades, a mean fade duration of
1.5 s, an outage fraction of 3/7 and a level-crossing rate of 1/3 Hz, and every
one of those is checked to 1e-12.

## Test suite

```
python3 -m pytest tests/ -q --tb=no -p no:cacheprovider --junit-xml=<path>
```

From the junit XML: **tests = 264, failures = 0, errors = 0, skipped = 0**.
The suite took between 40 s and 80 s across runs on two shared cores, depending
on how many sibling build agents were active. No test is marked `xfail`.
