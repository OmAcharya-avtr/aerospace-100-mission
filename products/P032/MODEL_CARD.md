# Model card — ConstelLink link-availability predictor

**This model is not certified for operational flight use.**

## Headline result, first

On the dataset this repository generates, **the logistic-regression baseline
is better calibrated than the learned model and has a better Brier score.**
The learned model wins neither. On a second, smaller pinned split the
*climatology* baseline has the lowest reliability term of the three.

Both orderings are measured, reproducible, and kept. They are the result of
this work, not an embarrassment to be worked around.

| predictor | Brier | **reliability REL** | resolution RES | ECE | accuracy @0.5 |
|---|---|---|---|---|---|
| climatology baseline | 0.15562 ± 0.02887 | 0.00645 ± 0.00487 | 0.03131 ± 0.00722 | 0.07025 ± 0.02991 | 0.8067 |
| **logistic baseline** | **0.07112 ± 0.01866** | **0.00260 ± 0.00151** | 0.11246 ± 0.01727 | **0.02583 ± 0.00777** | 0.8993 |
| learned (bagged GBM) | 0.07330 ± 0.01964 | 0.00516 ± 0.00284 | 0.11224 ± 0.01684 | 0.03929 ± 0.01064 | 0.8950 |

Mean ± sample standard deviation over five grouped splits. Lower is better for
Brier, REL and ECE; higher is better for RES. Source:
`validation/validate_calibration_output.txt`.

The logistic and learned bootstrap intervals on the Brier score overlap
substantially (0.07448-0.10249 against 0.07781-0.10702 on split seed 0), so
the Brier difference is **not** significant at this sample size. The
calibration difference across five seeds is the more robust part of the
result.

## Problem

Given a predicted contact opportunity — its geometry, its link type and a
weather forecast — estimate the probability that the contact will close at its
planned data rate. The output feeds a scheduling decision, so a
*mis-calibrated* probability corrupts the decision even when the thresholded
classification happens to be right. Calibration, not accuracy, is therefore
the figure of merit throughout.

## Baselines, implemented first

Both were written and measured before the learned model existed.

1. **Climatology** (`ClimatologyBaseline`). Predicts the training base rate of
   the row's stratum: link type × leg type × elevation band. This is the
   standard climatological reference forecast of forecast verification (Wilks
   2011, *Statistical Methods in the Atmospheric Sciences*, 3rd ed., Ch. 8).
   It is calibrated **by construction** on the training distribution, because
   it reports an observed frequency; its weakness is resolution, not
   reliability. Unseen strata fall back to the global training base rate, and
   the count of such rows is exposed.
2. **Logistic regression** (`LogisticBaseline`). L2-regularised logistic
   regression on standardised features. A generalised linear model with a
   canonical link, so its fitted probabilities are maximum-likelihood under
   the Bernoulli model. This is a genuinely competitive calibration reference,
   not a strawman — and on this data it wins.

## Learned model

`LinkAvailabilityModel`: a bagged ensemble of five
`sklearn.ensemble.HistGradientBoostingClassifier` members, each wrapped in
`sklearn.calibration.CalibratedClassifierCV` with Platt scaling on an inner
3-fold split.

| hyperparameter | value | why |
|---|---|---|
| `n_members` | 5 | enough spread for an uncertainty output inside the compute budget |
| `max_iter` | 120 | fits in under 1 s on one core |
| `max_depth` | 3 | the feature space has 9 columns and 1974 rows |
| `learning_rate` | 0.1 | sklearn default |
| `method` | `"sigmoid"` (Platt) | isotonic needs more data and overfits at this sample size |
| `calibration_folds` | 3 | inner CV for the calibrator |
| `seed` | caller-supplied | member seeds derive from it deterministically |

Calibration references: Platt 1999 (*Advances in Large Margin Classifiers*,
MIT Press) for sigmoid scaling; Zadrozny & Elkan 2002 (KDD) for isotonic;
Niculescu-Mizil & Caruana 2005 (ICML) for the finding that boosted trees need
such a correction because their raw scores are pushed toward the extremes.

Isotonic calibration was also measured, over the same five seeds: Brier
0.07354, REL 0.00424, ECE 0.03117. Slightly better calibrated than sigmoid and
slightly worse on Brier; still worse than the logistic baseline on both.
Sigmoid is the default because it is roughly twice as fast at this sample size
(5.6 s against 12.7 s for the five-seed sweep).

## Uncertainty output

`predict_std` returns the ensemble standard deviation of the five members'
probabilities. On the held-out split: mean 0.0411, max 0.2599.

**What it is:** epistemic spread — disagreement between members trained on
different bootstrap resamples.

**What it is not:** it does not capture the irreducible log-normal
scintillation fade in the data-generating process. The probability itself
expresses that. A near-zero spread at a probability of 0.5 is the correct
output for a contact whose outcome is genuinely a coin flip, and reading it as
"the model is confident the answer is 0.5" would be a mistake.

The two baselines return a zero spread from `predict_std`. That is a statement
that they have no epistemic spread to report, not a claim of certainty. A
parametric confidence interval on the logistic linear predictor could have
been returned instead; it was not, because presenting two different notions of
uncertainty under one name is worse than reporting none.

## Data

See `DATASET_CARD.md` in full. The short version, because it governs how every
number above should be read: the orbital geometry is real, the weather is
generated, and **the label is produced by this package's own capacity models**.
A model trained here learns this package's physics plus the observation noise
placed in front of it. This demonstrates a comparison method. It does not
demonstrate agreement with measured link outages, and no claim about real link
availability follows from any number in this card.

## Training and evaluation protocol

| item | value |
|---|---|
| dataset | 1974 contacts, 24/4/1 Walker shell at 550 km, 18 h, 5 ground stations |
| base rate | 0.7523 |
| split | **grouped by link**, 30 % test, five seeds (0-4) |
| train / test on seed 0 | 1457 / 517 |
| model seed | 1 (fixed across split seeds, so the split is the only varying factor) |
| bins for REL/RES/ECE | 10 equal-width |
| bootstrap | 2000 percentile resamples of forecast/outcome pairs |

Grouped splitting keeps every contact of a link entirely on one side.
Row-wise splitting leaks, and the leakage is measured rather than asserted:

| predictor | grouped Brier | row-wise Brier | optimism |
|---|---|---|---|
| climatology | 0.18479 | 0.17044 | +0.01435 |
| logistic | 0.08818 | 0.08568 | +0.00249 |
| learned | 0.09220 | 0.08546 | +0.00674 |

The learned model gains 2.7× more from leakage than the logistic baseline,
which is what a higher-capacity model is expected to do.

## Metrics, and why these ones

* **Brier score** (Brier 1950, *Monthly Weather Review* 78, 1-3) — a strictly
  proper score, so it cannot be improved by misreporting a believed
  probability.
* **Murphy decomposition** (Murphy 1973, *J. Applied Meteorology* 12, 595-600)
  into reliability − resolution + uncertainty. Reliability is the calibration
  term and is the headline.
* **Expected calibration error** — reported, but never alone: ECE is not a
  proper score, and a constant forecast at the base rate has an ECE near zero
  with no skill at all.
* **Log loss** — the second proper score, reported because it penalises
  confident errors differently from the Brier score.
* **Accuracy at 0.5** — reported last and explicitly demoted.

The three-term decomposition identity is exact only when every forecast inside
a bin has the same value. With 10 equal-width bins over continuous forecasts
there are two further components (Stephenson, Coelho & Jolliffe 2008, *Weather
and Forecasting* 23, 752-757), and the reported `identity_residual` is exactly
their sum — 5.4e-05 to 9.4e-04 here, which is real and is not called rounding
error. On distinct-value bins the identity closes to 3.9e-15, which is the
exactness check.

## Failure cases

1. **Unseen stratum.** The climatology baseline falls back to the global base
   rate and reports how many rows did so. The other two predictors extrapolate
   silently; nothing stops them.
2. **Single-class training split.** `LinkAvailabilityModel.fit` raises rather
   than training a degenerate calibrator. `grouped_split` raises if either
   side is single-class.
3. **Degenerate strata.** Two strata in the default configuration have a
   closure rate of exactly 1.0 (ground legs above 50 deg elevation). The
   climatology baseline then predicts 1.0 there, and a single counterexample
   in the test set produces an unbounded log loss contribution. This is why
   `log_loss_safe` clips, and why the log loss is reported next to the Brier
   score rather than instead of it.
4. **Small bins.** With 10 bins and 517 test rows some bins hold three or four
   samples. The reliability diagram draws Wilson 95 % intervals so such a bin
   is visibly uninformative instead of looking like a calibration failure.
5. **Correlation the bootstrap ignores.** Forecast/outcome pairs within one
   link are not independent, so the bootstrap intervals are optimistic. The
   grouped split removes the train/test leakage but not the within-test
   correlation.
6. **Out-of-distribution geometry.** The model has seen one Walker shell at
   one altitude. Nothing here says what it does at a different altitude,
   inclination or terminal design.

## Compute

Measured in the container that produced this card (one CPU core, shared):

| step | time |
|---|---|
| dataset generation, 18 h horizon, 1974 rows | 6.47 s |
| logistic baseline fit | 5.3 ms |
| learned model fit, 5 members, Platt | 0.95 s |
| learned model inference, 517 rows | 52.0 ms |
| five-seed comparison sweep, all three predictors | 5.6 s (sigmoid), 12.7 s (isotonic) |

Peak Python allocation during the learned fit: 3.8 MiB (`tracemalloc`, a lower
bound). No GPU. PyTorch is not used and is not available in the build
environment.

## Reproducibility

```bash
python validation/validate_calibration.py      # the five-seed comparison and the figure
python examples/example_calibration.py         # the reliability diagram in screenshots/
python -m constellink predict                  # the one-split table from the CLI
python -m pytest tests/test_regression.py -q   # the pinned scores
```

Seeds: dataset `20260401`, model `1`, split seeds `0-4`, bootstrap `7`. Every
one is a default or a named constant in the script.

The pinned regression scenario is a smaller 6 h, seed 7 dataset with a 3-member
model: climatology Brier 0.155990487 / REL 0.003488179; logistic 0.082848357 /
0.004595313; learned 0.089018047 / 0.012436083. On that split the climatology
baseline has the lowest reliability term, and
`tests/test_regression.py::test_the_pinned_ranking_is_the_measured_one` asserts
it, so the honesty claim in this card cannot drift away from the code.

## Ethical and safety limits

* Research-grade. Not flight-qualified, not certified, not approved for
  operational aerospace use. **This model is not certified for operational
  flight use.**
* Trained exclusively on synthetic data generated by this package. Using it to
  make a real scheduling decision would mean trusting a model of a model.
* The model outputs a probability, not a decision. Any threshold applied to it
  encodes a cost ratio that this package does not know.
* No personal data is involved at any point.

## Licence and credits

Released under AGPL-3.0 with the rest of the repository.
© 2026 OPTIMA Organisation.
