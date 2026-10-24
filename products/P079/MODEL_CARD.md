# Model card: the learned delay-asymmetry corrector

**This model is not certified for operational flight use.**

Package `timetransfer`, module `timetransfer.asym.learned`. Validation label of
the product it belongs to: **Level 3, hardware-pending**. Nothing here was
trained on, or evaluated against, data from hardware.

---

## 1. The problem

A two-way timestamp exchange gives two observable combinations of three
unknowns. Writing `d_ab` and `d_ba` for the one-way delays and `theta` for the
remote clock offset:

```
t2 - t1 = d_ab + theta
t4 - t3 = d_ba - theta
```

The standard estimator `theta_hat = ((t2-t1) - (t4-t3))/2` is exact when
`d_ab == d_ba` and otherwise carries an error of exactly `A/2`, where
`A = d_ab - d_ba`. That is equation (5) of `timetransfer.twoway` and it is
verified to 1.084e-19 s in `validation/validate_symmetric_bias_output.txt`.

`A` splits into a static part, `f_ab - f_ba`, and a queueing part,
`q_ab - q_ba`. **The static part is not observable from timestamps**: it enters
the observable difference identically to `theta`. The queueing part is partly
observable, because each direction's excursion above its own delay floor shows
up in the dwell times.

The model's job is to predict `b = mean(A)/2` for a window of exchanges, which
subtracts directly off the offset estimate.

## 2. The baselines, implemented and scored first

Three closed-form estimators exist in `timetransfer.asym.baselines` and were
written, measured and published before any model was trained:

| Name | Prediction | Trained? |
|---|---|---|
| `symmetric_assumption` | `0` | no |
| `median_filter` | `(median(fwd_c) - median(rev_c)) / 2` | no |
| `mean_difference` | `(mean(fwd_c) - mean(rev_c)) / 2` | no |

`fwd_c` and `rev_c` are the dwell times centred on each direction's own window
minimum. `mean_difference` is included because the target is a *mean*; leaving
it out would have flattered the median filter, which the specification
predicted would be competitive.

A linear learned baseline, `learned_ridge`, is scored alongside, because the
honest question about a tree ensemble on sixteen features is whether a linear
combination of them would do.

## 3. Architecture

`LearnedAsymmetryCorrector`: three `sklearn.ensemble.GradientBoostingRegressor`
models (Friedman 2001), all with `n_estimators=200`, `max_depth=3`,
`learning_rate=0.05`, `min_samples_leaf=8`, `subsample=1.0`,
`random_state=20261024`:

- one with `loss="squared_error"` for the point prediction;
- two with `loss="quantile"` at `alpha = 0.1587` and `alpha = 0.8413`, whose
  half-difference is the reported per-row sigma.

`RidgeAsymmetryCorrector`: `StandardScaler` into `Ridge(alpha=1.0)`.

Inputs: the sixteen window features of `timetransfer.asym.features`, all in
seconds except the dimensionless counts. Output: one number, the predicted mean
asymmetry bias in seconds.

## 4. Dataset

See `DATASET_CARD.md`. In short: a seeded synthetic generator whose equations
are printed in `timetransfer/asym/data.py`, 1200 scenarios of 150 attempted
exchanges each, no real network trace anywhere.

**The single assumption that decides the result** is the `coupling` parameter,
which sets how strongly the unobservable static asymmetry follows the
observable congestion difference:

```
floor_asymmetry = coupling * base_scale * (congestion_ab - congestion_ba)
                  + independent_asymmetry
```

At `coupling = 0` the static asymmetry is unlearnable by construction. The
benchmark is therefore run at couplings 0.0, 1.0 and 2.0, and all three are
published, so the reader sees the curve rather than one point chosen after the
fact.

## 5. Training procedure

Scenario-ordered 70/30 split: the first 70 per cent of scenarios train, the
last 30 per cent are held out. The split is not shuffled, because each scenario
is an independent draw from the generator and an ordered split is reproducible
without a second seed. No hyperparameter search was run; the configuration in
`TrainingConfig` is the one every published number was measured with.

Compute: three ensembles of 200 trees over 840 rows and 16 features. Measured
fit time on the two-core container this was built in: **3.7050 s** for the
boosted model and **0.0026 s** for the ridge, from the table in
`validation/validate_asymmetry_benchmark_output.txt`, which is declared
unstable.

## 6. Metrics, on one held-out split, baselines included

Coupling 1.0, 360 held-out scenarios:

| Estimator | RMSE [s] | MAE [s] | median \|e\| [s] | p95 \|e\| [s] | Skill |
|---|---|---|---|---|---|
| `symmetric_assumption` | 2.525593e-04 | 1.992098e-04 | 1.584618e-04 | 4.785189e-04 | 0.0000 |
| `median_filter` | 1.512459e-04 | 1.211701e-04 | 1.059714e-04 | 3.054017e-04 | 0.4011 |
| `mean_difference` | 1.335423e-04 | 1.073300e-04 | 9.018790e-05 | 2.618734e-04 | 0.4712 |
| `learned_ridge` | **1.039986e-04** | 8.304100e-05 | 6.947085e-05 | 1.956540e-04 | **0.5882** |
| `learned_gbr` | 1.050239e-04 | 8.263485e-05 | 7.060012e-05 | 2.041022e-04 | 0.5842 |

Across the three couplings:

| Coupling | median_filter / learned_gbr | learned_gbr / learned_ridge |
|---|---|---|
| 0.0 | 1.0394 | 1.0273 |
| 1.0 | 1.4401 | 1.0099 |
| 2.0 | 1.8868 | 1.0088 |

### The verdict, stated plainly

**The gradient-boosted model loses to a standardised linear ridge on the same
sixteen features, at every coupling measured.** The margin is between 0.9 and
2.7 per cent of RMSE, in the ridge's favour. A tree ensemble, its
hyperparameters, its three-model structure and the scikit-learn minor-version
pin it forces are all cost; on this generator they buy nothing. If you are
going to correct the asymmetry, use the ridge.

**Whether the median filter matches the learned model depends on the
coupling.** At coupling 0.0 -- where the static asymmetry cannot be learned at
all -- the median filter is within 3.9 per cent of the boosted model, which is
the result the product specification predicted. At coupling 1.0 and 2.0 the
learned models are ahead by 44 and 89 per cent. The specification's prediction
is confirmed in the regime where nothing is learnable and not confirmed where
something is; both are published.

The feature importances say why. The boosted model puts 0.5386 of its
importance on `std_difference_s` and 0.3685 on `mean_difference_s`, and
`mean_difference_s / 2` *is* one of the closed-form baselines. The model's
advantage over that baseline comes from one extra feature.

## 7. Uncertainty output

Two outputs, and the distinction matters.

**The model's own output**, from `predict_with_uncertainty`, is half the width
of the quantile pair. Measured held-out coverage of the k = 1.96 band:
**0.8917 against a nominal 0.9500**, with 0.0000 quantile crossings
(`validation/validate_asymmetry_benchmark_output.txt` section 7c). **This band
under-covers and is not a confidence interval.** It is a dispersion estimate
and should be read as one.

**What the offset uncertainty budget actually consumes** is the corrector's
residual standard deviation measured on a *calibration* split -- scenarios the
model neither trained on nor is scored on -- fed to
`timetransfer.uncertainty.offset_uncertainty` as
`AsymmetrySource.ESTIMATED`. The coverage of the resulting offset interval is
measured in `validation/validate_uncertainty_coverage_output.txt`: **0.9525
against a nominal 0.9500**, Clopper-Pearson [0.9268, 0.9712], verdict
calibrated. The median filter treated the same way gives 0.9450, also
calibrated.

So: the model's internal band is not calibrated, and the product does not use
it for the thing that matters.

## 8. Failure cases

- **A path whose static asymmetry is uncorrelated with its queueing shape.**
  At `coupling = 0` the model recovers only the queueing part and the static
  part remains in the error. This is physics, not a modelling defect: the
  static asymmetry is not in the data.
- **A window shorter than `MIN_WINDOW = 8` exchanges.** The quantile features
  are meaningless and `window_features` raises `InsufficientDataError`.
- **A link whose delay floor drifts within the window.** Centring on the window
  minimum assumes a constant floor. A drifting floor leaks into the features as
  a spurious queueing excursion.
- **Delay-correlated loss.** The generator loses exchanges independently of
  their delay. A congested link does not, and the model has never seen that.
- **Anything outside the generator's parameter ranges** in `DATASET_CARD.md`.
  No extrapolation guarantee is offered and none should be assumed.
- **Any real network.** Every number here is a statement about a seeded
  generator.

## 9. Reproducibility

```bash
python validation/validate_asymmetry_benchmark.py
python validation/validate_uncertainty_coverage.py
```

Seeds: generator base seed 20261024 for the benchmark and 70079 for the
coverage study; scenario `i` uses `base_seed + i`. `random_state=20261024` on
every estimator, `subsample=1.0`, no shuffling anywhere.
`tests/test_reproducibility.py` asserts that a refit on the same split
reproduces the prediction vector and the sigma vector byte for byte -- on the
predictions, not on a pickle, because a pickle's stability is not part of
scikit-learn's public API contract. That is also why `scikit-learn` is bounded
to `>=1.9,<1.10` in `pyproject.toml`.

## 10. Compute used

Measured on the container this was built in: 2 available cores, no GPU.
Dataset generation for 1200 scenarios takes about 5 s; the three-coupling
benchmark including all five estimators takes about 35 s end to end. PyTorch is
not used and is not available.

## 11. Ethical and safety limits

- The model estimates a correction to a clock offset. It does not steer a
  clock, and nothing in this package should be in a control loop.
- Trained entirely on synthetic data from a generator whose equations are in
  the repository. It encodes that generator's assumptions, including the
  coupling assumption in section 4, and nothing about any real network.
- There is no personal data anywhere in the dataset or the model.
- **This model is not certified for operational flight use.** Research grade:
  not flight-qualified, not certified, not approved for operational aerospace
  use.

## 12. When not to use it

Use `symmetric_assumption` -- that is, no correction at all -- when you have
calibrated the path and know the asymmetry, or when the path is reciprocal by
construction. Use `median_filter` or `mean_difference` when you want a
correction with no training step, no model artefact and no version pin; on this
generator they take most of the available improvement. Use `learned_ridge` if
you want the rest of it. The boosted model is the one this card recommends
against, on its own measurements.
