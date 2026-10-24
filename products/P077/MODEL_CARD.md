# Model card — relfiledelivery learned delivery-time predictor

**This model is not certified for operational flight use.**

Version 0.1.0 · Apache-2.0 · © 2026 OPTIMA Organisation · Status TESTING

Every number in this card was produced by `validation/validate_predictors.py`
in the session that wrote the card. The committed raw output is
`validation/validate_predictors_output.txt`.

## Problem

Predict the delivery time, in seconds, of one file transfer over one declared
link, and report a **calibrated interval** rather than a point estimate. The
inputs are declared link and file conditions; the output is a set of quantiles
of the delivery-time distribution conditional on those conditions.

The point of the model is not to replace the closed form in
`relfiledelivery.analytic`. The closed form is exact under its assumptions and
the model is not expected to beat it there. The question the model exists to
answer is whether a learned predictor earns its place in the regime where the
closed form stops being exact — correlated outages and contact windows — and
the answer is measured in both regimes and reported either way.

## Baselines, implemented first

Both baselines were written, tested and validated before any model was fitted.

1. **`analytic`** — the closed form of `relfiledelivery.analytic` evaluated at
   the 0.50 quantile for its point prediction, divided by the outage duty cycle
   and laid out over the declared contact windows. **It sees no training data at
   all**, has no fitted parameters, and nothing about it can be tuned on the
   benchmark. Its interval ends come from two exact marginals: the
   negative-binomial transmission count and the closed-form round count.
2. **`analytic_mean`** — the same formula with the *expectation* as its point
   prediction. Included because a closed form naturally produces an expectation
   and a point prediction should be a median, and the gap between them is a
   result rather than a detail.
3. **`mean_lookup`** — measured quantiles of delivery time divided by clean-pass
   time, binned by mode, loss probability and availability. This is the
   spreadsheet of past transfers an engineer actually has. Despite the name it
   reports the measured median, not the mean.

## Architecture

Seven independent `sklearn.ensemble.GradientBoostingRegressor` models with
`loss="quantile"`, one per quantile level in
`(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)`, fitted on the base-10 logarithm
of the delivery time. Quantiles commute with a monotone transform, so
exponentiating the prediction gives the quantile of the delivery time exactly
and not an approximation to it.

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 3 |
| `learning_rate` | 0.06 |
| `min_samples_leaf` | 12 |
| `subsample` | 0.9 |
| `random_state` | 20261024 |

No hyperparameter search was run. The values above are the first set tried and
they are recorded as such; a search against the held-out set would have made
the held-out set a validation set.

Independent per-level fits are not guaranteed to be monotone in the level. The
predictions are sorted across levels before being returned and **939 of 6762
adjacent level pairs (0.1389)** had to be sorted on the held-out set. That rate
is reported rather than hidden: it says the per-level fits disagree about the
shape of the conditional distribution in part of the feature space, and it is
consistent with the interval under-covering.

## Features

Fourteen, in declared order. File and link quantities plus three cheap derived
physical quantities.

| Feature | Units |
|---|---|
| `log10_pdu_count` | dimensionless |
| `log10_pdu_payload_bytes` | log bytes |
| `log10_file_bytes` | log bytes |
| `pdu_loss_probability` | dimensionless, [0, 1) |
| `log10_forward_rate_bps` | log bit/s |
| `log10_asymmetry` | dimensionless |
| `log10_one_way_light_time_s` | log seconds |
| `outage_duty` | dimensionless, (0, 1] |
| `log10_mean_down_s` | `log10(mean_down_s + 1)` |
| `contact_duty` | dimensionless, (0, 1] |
| `availability` | dimensionless, (0, 1] |
| `log10_clean_pass_s` | log seconds |
| `log10_turnaround_s` | log seconds |
| `mode_is_immediate_nak` | 0 or 1 |

**The analytic prediction is deliberately not a feature.** Feeding the closed
form in would turn the learned model into a residual corrector on top of the
baseline it is being compared against, and the comparison would be circular. A
residual model is a reasonable thing to build and it is not what this is.

## Dataset

See `DATASET_CARD.md`. In summary: scenarios are drawn from the seeded
generator in `relfiledelivery.dataset`, one simulated transfer per scenario, and
the realised delivery time of that single run is the target. One run, not an
average, because the model is asked for an interval and its training target has
to carry the run-to-run spread as well as the scenario-to-scenario spread.

| Split | Seeds | Scenarios | Usable (completed) | Dropped |
|---|---|---|---|---|
| train | from 100000, both regimes | 3000 | 2810 | 190 |
| test | from 900000, both regimes | 1200 | 1127 | 73 |

The seed ranges are disjoint. No flight data is used anywhere.

## Test-split strategy

Hold-out by seed range, not by random partition of a single generated table.
Train and test scenarios are generated independently from disjoint seeds, so no
test scenario can share a scenario draw with a training scenario. The scoring
is further split by regime, because the specification's prediction is about the
regimes and an aggregate number would hide it.

The asymmetry between predictors is stated rather than hidden: the learned model
and the lookup table see the generator through the training set and the
analytic predictor does not. That favours the learned model, and its win in the
correlated-outage regime should be read with that in mind.

## Metrics

Median absolute relative error of the point prediction, its 90th percentile,
interval coverage against nominal, and median interval width relative to the
observed time. The median rather than the mean, for the same reason the
delivery-time distribution is reported as quantiles.

| Predictor | Regime | Rows | median \|rel err\| | p90 \|rel err\| | Coverage | Nominal | Width/obs | Trained |
|---|---|---|---|---|---|---|---|---|
| analytic | all | 1127 | 0.174124 | 0.859613 | 0.5093 | 0.90 | 0.4327 | no |
| analytic | correlated_outage | 527 | 0.368928 | 1.274729 | 0.2562 | 0.90 | 0.4928 | no |
| analytic | no_outage | 600 | **0.051928** | 0.387599 | 0.7317 | 0.90 | 0.3970 | no |
| analytic_mean | all | 1127 | 0.184486 | 0.874741 | 0.5093 | 0.90 | 0.4327 | no |
| analytic_mean | correlated_outage | 527 | 0.381280 | 1.269200 | 0.2562 | 0.90 | 0.4928 | no |
| analytic_mean | no_outage | 600 | 0.104073 | 0.364563 | 0.7317 | 0.90 | 0.3970 | no |
| mean_lookup | all | 1127 | 0.535480 | 0.977916 | 0.8784 | 0.90 | 7.4233 | yes |
| mean_lookup | correlated_outage | 527 | 0.574066 | 1.406869 | 0.8672 | 0.90 | 5.2510 | yes |
| mean_lookup | no_outage | 600 | 0.489155 | 0.906005 | 0.8883 | 0.90 | 10.1253 | yes |
| learned_quantile | all | 1127 | 0.169898 | 0.575872 | 0.8146 | 0.90 | 0.8662 | yes |
| learned_quantile | correlated_outage | 527 | **0.292259** | 0.714166 | 0.8197 | 0.90 | 1.4190 | yes |
| learned_quantile | no_outage | 600 | 0.104687 | 0.347475 | 0.8100 | 0.90 | 0.5633 | yes |

### The headline result

- **In the no-outage regime the closed form wins by a factor of 2.0160**
  (0.051928 against 0.104687), using no training data. This is the
  specification's own prediction and it is published as measured.
- **In the correlated-outage regime the learned model wins by a factor of
  1.2623** (0.292259 against 0.368928). It earns its place, by a margin
  substantially smaller than the margin by which it loses in the other regime.
- The learned model also has the lowest 90th-percentile error in both regimes
  (0.347475 and 0.714166), so its advantage is larger in the tail of the error
  distribution than at its centre, which is what a quantile loss is for.
- **The lookup table loses to the closed form in both regimes.** An engineer's
  spreadsheet, binned on the variables this generator makes available, is not
  competitive: median absolute relative error 0.489155 against 0.051928 in the
  no-outage regime.

The analytic baseline was strengthened before this comparison was published. An
earlier version divided the closed form by the product of the contact and
outage duty cycles, which on the worked-example scenario gave 164.57 s against
a measured median of 26.80 s because the transfer fitted inside the first
contact and the contact duty cycle never bit. With that version the learned
model's win in the correlated-outage regime was a factor of 1.9036 rather than
1.2623. Benchmarking against a straw man measures nothing, and the weaker
figure is not the one reported.

## Uncertainty output

The interval between the 0.05 and 0.95 quantile models, returned by
`predict_interval(table, nominal=0.90)`.

**It is not calibrated.** Measured coverage at nominal 0.90 over 1127 held-out
scenarios is **0.8146**, against a standard error of a coverage estimate of
0.0089, so the shortfall of 0.0854 is about ten standard errors and is not
sampling noise. Per regime: 0.8100 in no-outage, 0.8197 in correlated-outage.

Neither baseline interval is calibrated either, and the closed form is much
worse: coverage **0.5093** at nominal 0.90, because it combines two exact but
*dependent* marginal quantiles at the same probability and the sum of quantiles
is not the quantile of a sum. The lookup table is the only one near nominal at
0.8784 and it pays with an interval **8.57 times wider** than the learned
model's (7.4233 against 0.8662 median width relative to the observed time).

Calibration curve, measured:

| Predictor | Nominal 0.50 | Nominal 0.80 | Nominal 0.90 |
|---|---|---|---|
| analytic | 0.2680 (width 0.1466) | 0.4321 (width 0.3326) | 0.5093 (width 0.4327) |
| mean_lookup | 0.4996 (width 1.5582) | 0.7941 (width 4.1216) | 0.8784 (width 7.4233) |
| learned_quantile | 0.4374 (width 0.3008) | 0.6868 (width 0.6153) | 0.8146 (width 0.8662) |

A conformal recalibration layer would be the obvious next step and is not
implemented here; the measured shortfall is reported instead.

## Failure cases

- **Scenarios that do not complete.** The model predicts delivery time
  conditional on completion and is never shown an incomplete run. 190 of 3000
  training scenarios and 73 of 1200 test scenarios were dropped for this reason.
  Nothing in this model predicts *whether* a transfer completes.
- **Interval under-coverage**, above, which is the main one.
- **Quantile crossing** at a rate of 0.1389 of adjacent level pairs.
- **Outside the generator's ranges.** Gradient-boosted trees are piecewise
  constant and do not extrapolate. A file an order of magnitude larger, a loss
  probability above 0.25 or a light time above 4 s is outside everything this
  model has seen, and the prediction will be the edge of the training range
  rather than an error.
- **A link whose loss is not independent-plus-Gilbert.** Every number here is a
  statement about that process.
- **The closed form's own failure is sharper and more predictable**: it is a
  step function of the contact plan and jumps by one window period when the
  contact time it needs crosses a window duration. On the worked-example
  scenario it reports 181.85 s against a measured median of 26.80 s for exactly
  that reason, while the simulation says 82.5 per cent of runs finish in the
  first contact.

## Reproducibility

```bash
python validation/validate_predictors.py
```

Seeds: training scenarios from 100000, test scenarios from 900000, both
regimes; the simulated run for scenario seed `s` uses seed `7919 * s + 13`; the
estimator `random_state` is 20261024. Two runs with these seeds produce
identical tables, asserted by `validation/validate_reproducibility.py` and by
`tests/test_dataset.py::TestTable::test_table_is_reproducible`.

Measured on Python 3.13.17, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1,
pandas 3.0.5. **scikit-learn is bounded to `>=1.9,<1.10`** in `pyproject.toml`
because the numbers above are properties of this estimator's fitted trees: a
minor release that changes the quantile-loss line search or the subsampling
order moves them without failing a single test.

## Compute used

Two CPU cores, no GPU, no accelerator. Scenario generation for the full
benchmark is about 13 seconds of wall clock and fitting the seven quantile
models about 29 seconds; the whole script is about 59 seconds. These are wall
clock on a shared container, they are **unstable**, and no conclusion in this
card is derived from them.

## Ethical and safety limits

- The model is trained entirely on synthetic data from a declared generator. It
  has never seen a real link and is not evidence about any real link.
- It must not be used to decide whether a transfer is safe to attempt, to size
  an operational margin without an independent check, or as input to any
  decision whose failure mode is loss of data or loss of vehicle.
- A prediction from it is a statement about `relfiledelivery.dataset`, and that
  module's equations are in its docstring so the statement can be checked.
- **This model is not certified for operational flight use.**
