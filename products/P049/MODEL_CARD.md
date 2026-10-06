# Model card - `linkoutage` short-horizon outage classifier

**This model is not certified for operational flight use.** Research-grade,
validation level 2. It is a demonstration that a two-parameter channel fit
beats a random forest on this problem, not a deployable component.

## Problem

Given the last 400 samples of received optical amplitude at 1 MHz (0.4 ms, two
correlation lengths), predict whether a **new** fade below an amplitude of 0.35
begins in the next 200 samples (0.2 ms, one correlation length). Decisions are
taken every 50 samples.

The base rate is 2.56 % over the whole record and 2.72 % on the test split.
That rate is the whole difficulty of the problem: accuracy is near-meaningless,
and a forecaster that emits the base rate and nothing else is already perfectly
calibrated.

## Baselines, implemented and validated before the learned model

| Baseline | What it is | Uses labels? |
|---|---|---|
| Constant base rate | emits the training base rate for every row | yes, one number |
| Analytic level-crossing rate | fits the lognormal AR(1) channel (scintillation index from `var(ln I)`, lag-one correlation from the autocorrelation) to the **training amplitude record**, then evaluates the exact conditional expected number of down-crossings in the horizon given the present log-amplitude and converts it to a probability by Poisson clumping | **no** |
| Logistic regression | L2-regularised, on the 14 standardised features | yes |

The constant base-rate baseline is included deliberately. On a rare-event
problem it is strong, it is perfectly calibrated by construction, its Brier
score equals the problem's uncertainty term exactly, and omitting it is the
standard way to convince oneself that a model has skill when it does not.

## Architecture

`sklearn.ensemble.RandomForestClassifier`, 200 trees, `min_samples_leaf = 5`,
no depth limit, `random_state = 4901`, `n_jobs = 2`. Fourteen tabular features;
no neural network, because PyTorch is unavailable in the build environment and
a 14-feature tabular problem of 48 000 rows does not need one.

`min_samples_leaf` was chosen on the **calibration** split from
`{5, 20, 60}` by Brier score (0.01805131, 0.01806293, 0.01816783); the test
split was not consulted. This is the only hyperparameter search performed, and
the three values it covered are reported in full.

## Features

All computed from the observation window `a[t-399 .. t]` only. Natural log of
amplitude throughout, because the channel is lognormal and a Gaussian-shaped
feature is the one a linear model can use.

`log_amp_last`, `log_amp_mean`, `log_amp_std`, `log_amp_min`, `log_amp_max`,
`log_amp_p10`, `log_amp_slope_full`, `log_amp_slope_short`,
`log_amp_mean_short`, `log_amp_diff_std`, `frac_below_threshold`,
`n_down_crossings_in_window`, `samples_since_down_crossing`, `in_fade_now`.

The analytic baseline is given column 0 only, which is the entire state its
model conditions on; it does not see the other thirteen.

## Dataset and splits

See `DATASET_CARD.md`. One 4 000 000-sample record (4 s at 1 MHz) from the
AR(1) lognormal channel at seed 4901, scintillation index 0.6, correlation time
0.2 ms. 79 989 decision instants, 2046 positive.

Split **temporally**, never randomly: 47 993 train (1219 positive), 11 998
calibration (279 positive), 19 974 test (543 positive), with 12 rows discarded
between splits so that the window and horizon either side of a boundary are
disjoint. A random split of a correlated series leaks: adjacent decision
instants share 350 of their 400 window samples and most of their horizon, so
shuffling puts near-duplicates of test rows into training and inflates every
metric.

**The effective sample size is the number of independent fade onsets, not the
row count.** On the test split that is 543 labelled onsets, and many of those
overlap. Brier differences in the fourth decimal place are not resolvable.

## Metrics on the held-out test split

From `validation/outage_classifier_output.txt`. Baselines first.

| Forecaster | Brier | Brier skill | Reliability | Resolution | ECE | MCE | log loss | AUC | AP | mean forecast |
|---|---|---|---|---|---|---|---|---|---|---|
| constant base rate | 0.0264495 | 0.0000 | 3.189e-06 | 0.000e+00 | 0.00179 | 0.00179 | 0.12488 | 0.5000 | 0.0272 | 0.02540 |
| analytic LCR (fit) | 0.0265626 | **-0.0043** | 4.092e-03 | 2.317e-03 | 0.03199 | 0.18886 | 0.10673 | **0.8730** | 0.3193 | 0.05913 |
| logistic regression | 0.0220142 | 0.1677 | 1.600e-05 | 2.304e-03 | 0.00268 | 0.00888 | 0.09127 | 0.8731 | 0.3015 | 0.02680 |
| random forest | 0.0223498 | 0.1550 | 8.354e-05 | 2.033e-03 | 0.00575 | 0.02561 | 0.09643 | 0.8466 | 0.2872 | 0.02854 |
| **analytic LCR (fit) + Platt** | **0.0217557** | **0.1775** | 8.436e-06 | 2.317e-03 | 0.00189 | 0.00713 | 0.09052 | 0.8730 | 0.3193 | 0.02556 |
| random forest + Platt | 0.0222801 | 0.1576 | 1.522e-05 | 2.033e-03 | 0.00332 | 0.00693 | 0.09573 | 0.8466 | 0.2872 | 0.02505 |

Test base rate 0.0271853. Brier skill score is measured against a constant
forecast at the **training** base rate.

**A baseline wins.** The analytic level-crossing-rate forecaster with a
two-parameter Platt recalibration has the lowest Brier score of the six. The
random forest is fourth of six on Brier and last of four distinct models on
AUC. Per the mission's standing rule, that is the published result; the forest
has not been retuned to change it.

Two further results worth stating plainly:

* The **raw** analytic forecaster has the best discrimination of all four
  models (AUC 0.8730, average precision 0.3193) and is *worse than doing
  nothing* on the Brier score (skill -0.0043), because its reliability term is
  two orders of magnitude worse than anyone else's: it over-forecasts by a
  factor of 2.2 (mean forecast 0.059 against a base rate of 0.027). The cause
  is the Poisson-clumping step that turns an expected crossing count into a
  probability; crossings of a correlated process arrive in bursts, so
  `1 - exp(-m)` overstates `P(at least one)`. Two parameters of recalibration
  remove it entirely and leave the best forecaster in the table. Reporting AUC
  alone would have made this model look like the winner; reporting Brier alone
  would have made it look like a failure. Both are reported.
* Accuracy at a 0.5 threshold is 0.970-0.973 for every model, against 0.9728
  for a forecaster that always says "no". The metric cannot distinguish them.

### Sensitivity to the horizon

| Horizon [samples] | base rate | constant | analytic + Platt | logistic | forest (60 trees) | winner |
|---|---|---|---|---|---|---|
| 50 | 0.00925 | 0.0102554 | 0.0069354 | 0.0072483 | 0.0071575 | analytic + Platt |
| 100 | 0.01506 | 0.0159091 | 0.0119962 | 0.0121753 | 0.0122914 | analytic + Platt |
| 200 | 0.02558 | 0.0264495 | 0.0217557 | 0.0220142 | 0.0224485 | analytic + Platt |
| 400 | 0.04525 | 0.0458742 | 0.0403902 | 0.0409730 | 0.0417103 | analytic + Platt |

The baseline wins at every horizon tested. The result is not an artefact of the
one horizon chosen for the headline.

## Uncertainty output

`RandomForestOutageClassifier.predict_with_uncertainty` returns
`(probability, standard deviation across trees)`. The spread measures
**ensemble disagreement**, which is an epistemic-uncertainty proxy: it is large
where the training data were sparse and the trees extrapolate differently. It
is **not** an interval on the event; a well-determined probability of 0.5 has
small tree spread and large outcome uncertainty.

On the test split the mean spread is 0.0630, the median 0.0434 and the maximum
0.3807; 0.12 % of rows have zero spread. Grouping rows into deciles of spread,
the calibration gap in the top decile (0.0246) is an order of magnitude larger
than in the bottom decile (0.0029), and the Spearman rank correlation between
mean spread and calibration gap over the ten deciles is 0.455. **With ten
points that correlation has a p-value of 0.187 and is not significant at any
conventional level.** The honest statement is that the spread is a plausible
confidence flag on this record and that this experiment does not establish it.
It is shipped with that caveat, not as a validated uncertainty.

The constant base-rate baseline carries its own uncertainty, the binomial
standard error of the fitted rate (0.000718 on 47 993 training rows), which is
also the yardstick for whether a difference in mean forecast between two models
is resolvable at all.

## Training procedure, compute and reproducibility

Everything is keyed on seed 4901 and the configuration above.

```bash
cd validation
python validate_outage_classifier.py     # the published table, about 100 s
python regenerate_outage_model.py        # refit, persist, verify, about 60 s
```

`regenerate_outage_model.py` writes `validation/outage_models.joblib` and
`validation/outage_dataset.npz`, neither of which is committed (both are in
`.gitignore`), and then verifies that the refitted models reproduce the
published test Brier scores. On the build machine all six reproduced to within
1e-7 against a stated tolerance of 5e-6.

**Compute budget: 2 CPU cores and 7.8 GiB of RAM, shared with four sibling
build agents.** The longest single fit is the 200-tree forest on 47 993 rows
and 14 features, about 18 s. `validate_outage_classifier.py` runs in about
100 s end to end, which includes four forest fits for the hyperparameter sweep
and four more for the horizon sweep. No run in this repository takes more than
three minutes.

PyTorch is not available in the build environment. No GPU was used and none is
needed.

## Failure cases

* **Non-stationary turbulence.** The channel parameters are constant over the
  whole 4 s record. A real link's scintillation index moves with elevation
  angle, time of day and weather. Both the analytic baseline and the forest
  would need refitting on a timescale this experiment does not probe, and the
  constant base-rate baseline would degrade exactly as badly as the record is
  non-stationary - its reliability term on a short 600 000-sample record is the
  squared gap between the training and test base rates, which on that record
  reaches a factor of two.
* **Strong turbulence.** The lognormal model is a weak-to-moderate
  scintillation model. At scintillation indices above about 1 the gamma-gamma
  distribution is standard and is not implemented here, so the analytic
  baseline would be fitting the wrong family.
* **Measured records.** The analytic baseline is fitted to the same model
  family that generated the data, which flatters it. On a real record it would
  be fitting an approximation and its advantage should shrink. How much is not
  known and is not claimed.
* **Very short horizons.** At a horizon of a few samples the onset is almost
  deterministic from the current state and every model converges; at horizons
  much longer than the correlation time the problem becomes the unconditional
  base rate and skill vanishes. The useful range is roughly 0.25 to 2
  correlation times, which is what the horizon sweep covers.
* **Deep thresholds.** At an amplitude threshold of 0.2 the base rate falls
  below 0.1 % and the test split holds too few onsets to resolve any of the
  differences in the table. The code will run; the numbers will not mean
  anything.

## Ethical and safety limits

This is a channel-statistics research tool. It makes no claim about any
specific link, terminal or mission. Nothing in it has been validated against
measured optical link data, because none was available in the build
environment. It must not be used to size margins, set availability commitments
or make any operational decision. Not flight-qualified, not certified, not
approved for operational aerospace use.
