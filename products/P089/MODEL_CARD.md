# Model card — `aligncal` learned alignment estimator

Version 0.1.0. Status: TESTING. Research grade.

**This model is not certified for operational flight use.**

Every number here comes from a script in `validation/` that was executed on
2026-10-29, and the script is named beside the number. Nothing is carried from
an earlier run.

## Problem

Given a short record of angular rates from two sensors with a declared mounting
geometry, estimate the fixed relative misalignment rotation between them, as a
three-component rotation vector in radians, with an uncertainty.

The case the learned component exists for is the **short record**. Over a long,
richly excited record the unbiased batch solution of `aligncal.batch` is the
maximum-likelihood estimator and its error is set by the Cramer-Rao bound of
equation (12). Over a 150-sample record that bound is comparable to the spread
of the misalignment population itself, and a biased estimator that shrinks
toward that population can have a smaller mean squared error. Whether a learned
estimator actually does, and whether it does better than the closed-form
shrinkage that an analytic method can perform with no training at all, is the
question this card answers.

## Baseline, implemented and validated first

Three baselines, in the order they were built.

**Baseline 1, the specified one.** `aligncal.batch.solve`: weighted batch
least squares over the whole record, equations (14) to (16), solved on the
observable subspace only. Validated in `validation/validate_recovery_grid.py`
over 864 records before any learned component existed: the standardised error of
equation (24) has mean **-0.073846** and standard deviation **0.959812** against
a standard normal, with no score above 2.7673 in 2290.

**Baseline 2.** `aligncal.recursive.run`: a multiplicative extended Kalman
filter, equations (19) to (22), scored against Baseline 1. Validated in
`validation/validate_recursive_agreement.py`: agreement to **0.052809** of the
Cramer-Rao one sigma on well-conditioned records, **0.352091** on full-rank
records of any conditioning.

**The strengthened baseline, reported beside the specified one.**
`aligncal.batch.solve_map`: the same batch solution with a zero-mean Gaussian
prior, equation (18), given the per-component standard deviation of the
generated misalignment population, 1.0 mrad. The mission's build guide requires
the strongest honest baseline to be used and the specified one reported beside
it. A learned estimator trained on a known misalignment population performs
shrinkage toward that population; comparing it only against the unbiased
solution would credit it with a win it has not earned.

## Architecture

| configuration | estimator | ensemble | uncertainty |
|---|---|---|---|
| `forest` | `RandomForestRegressor`, 200 trees, `min_samples_leaf=2`, `max_features=0.5` | the 200 trees | per-output standard deviation across trees |
| `forest_deep` | the same with `min_samples_leaf=1` and all features at every split | the 200 trees | the same |
| `bagged_ridge` | 40 `Ridge(alpha=1e-3)` fits on bootstrap resamples | the 40 members | standard deviation across members |

Multi-output: three outputs, the components of the misalignment vector in the
reference frame, in radians. Seeded with 20261029 throughout.

**No hyperparameter search was run.** Three declared configurations, all
reported. The numbers here are not the best achievable by a learned estimator on
this problem.

### The feature set deliberately contains the observability evidence

Twenty-four features, listed in `aligncal.features.FEATURE_NAMES`: the scaled
normal-equation right-hand side of (15), the six unique components of the
normalised Gramian of (8), the information scale, record length and duration,
three rate statistics, the three excitation shares of (9) and their entropy, the
three components of the dominant rotation axis, and two residual statistics.

That matters for the honesty of this card's headline result. If the learned
estimator produces a confident estimate for a degree of freedom the record
cannot support, it is **not** because it was starved of the evidence that the
degree of freedom is unsupported. It was given the excitation shares, the
dominant rotation axis and the Gramian structure -- every number the analytic
method uses to refuse.

What is **not** a feature: the pseudo-inverse solution `G^+ b`. Supplying it
would make the learned estimator a wrapper around the analytic baseline and the
comparison meaningless, and it would hide the shrinkage that is the only
mechanism by which a learned estimator can beat an unbiased solution here.

## Dataset

`aligncal.dataset.training_population`, documented in `DATASET_CARD.md`. Seeded
synthetic rate records; **no flight data is used anywhere**. Each record draws
its excitation profile uniformly from `single_axis`, `near_single_axis`,
`two_axis` and `three_axis`, its peak rate log-uniformly from 0.01 to 0.08
rad/s, and its per-axis sensor noise one sigma log-uniformly from 5e-5 to 8e-4
rad/s. The injected misalignment has a magnitude uniform on [0, 3] mrad and a
direction uniform on the sphere.

The training population **deliberately includes single-axis records**, so every
configuration was trained on data containing unobservable degrees of freedom and
had the opportunity to learn to refuse.

## Training procedure and splits

| split | seed | records | samples each | use |
|---|---|---|---|---|
| training | 20261029 | 2500 | 150 | fitting all three configurations and the trivial baseline |
| test | 70120261 | 600 | 150 | every score in §Metrics |
| control | 55120261 | 120 | 150 | the unobservable-subspace measurement |

The splits are disjoint by seed and no record appears in more than one. The
misalignment, the mounting, the excitation phases and the measurement noise are
all drawn from the same seeded generator, so a split is reproduced exactly by
re-running with its seed.

Fitting all four estimators takes **18.1860 s** on two cores
(`validate_learned.py` §5.6, declared unstable).

## Metrics

Errors are split by subspace throughout. The component of the truth outside the
span of the floor-clearing degrees of freedom is carried in full by every
estimator, including the one that correctly refuses to guess at it, so including
it in a comparison would flatten every difference and make a refusal look like a
defect.

### On the held-out short-record population, 600 records

Per-component RMSE in milliradians, from `validate_learned.py` §5.1.

| estimator | RMSE total | **RMSE observable** | RMSE unobservable |
|---|---|---|---|
| batch, eq (14)-(16) | 5.716038 | 5.703693 | 0.375465 |
| **batch MAP, eq (18)** | 0.792044 | **0.697326** | 0.375594 |
| learned `forest` | 0.853146 | 0.760152 | 0.387333 |
| learned `forest_deep` | 0.860938 | 0.766268 | 0.392489 |
| learned `bagged_ridge` | 0.959229 | 0.882258 | 0.376485 |
| trivial (training mean) | 1.010964 | 0.938906 | 0.374837 |

**The baseline wins.** The regularised analytic solution beats the best learned
configuration by 9.01 per cent, with no training, no features and no model, and
it does so on every one of the four excitation profiles.

The learned component is not useless, and that is reported too: it beats the
trivial baseline by 19.04 per cent and the specified unbiased Baseline 1 by a
factor of 7.5.

### What the analytic win is made of

The MAP baseline was given the correct prior scale. With a prior ten times too
tight it **loses** to the unbiased solution at all four record lengths measured
(1.002141 against 0.862279 mrad at 150 samples, 0.369287 against 0.075992 at
9600). The analytic win is a statement about knowing the population, not a free
lunch, and it is reported with that price attached.

### Uncertainty, measured

| configuration | 2-sigma coverage | median sigma | RMSE total | sigma / RMSE |
|---|---|---|---|---|
| `forest` | 0.916667 | 0.771451 mrad | 0.853146 mrad | 0.9042 |
| `forest_deep` | 0.939444 | 0.847459 mrad | 0.860938 mrad | 0.9843 |
| `bagged_ridge` | **0.187778** | **0.077611 mrad** | 0.959229 mrad | **0.0809** |
| trivial | 0.941111 | 1.019584 mrad | 1.010964 mrad | 1.0085 |
| batch, Cramer-Rao | **0.953064** | — | — | — |

Nominal two-sigma coverage is 0.954500. `bagged_ridge` is overconfident by a
factor of 12 on the whole population. The trivial estimator's interval is better
calibrated than two of the three learned configurations.

## Uncertainty and confidence output

Every configuration exposes a per-component standard deviation from the spread
of its ensemble members, and `Prediction.sigma_in_direction` gives that spread
along any requested direction, which is what makes the measurement below
possible. **It is a model-spread uncertainty, not a calibrated posterior
standard deviation,** and it is labelled as such everywhere it is reported.

## Failure cases

### The headline: a confident estimate for a degree of freedom the data cannot support

`validation/validate_unobservable.py`. 120 single-axis records with **2 mrad
injected entirely along each record's own dominant rotation axis** -- the
direction that, by equation (10), the record cannot support at any data volume.
Every record is verified rank deficient by its own Gramian first: 120 of 120.

Injected truth along that direction: median **412.5296 arcsec**.

| estimator | claim | \|error\| | its own 1 sigma | \|err\|/sigma | 2-sigma coverage |
|---|---|---|---|---|---|
| batch, eq (14)-(16) | — | 412.5296 | **infinite (refused)** | — | — |
| learned `forest` | -0.5320 | 413.0616 | 149.3864 | 2.7775 | **0.0750** |
| learned `forest_deep` | -1.6141 | 414.1437 | 169.0386 | 2.4939 | **0.2083** |
| learned `bagged_ridge` | -7.5118 | 420.0414 | **14.3540** | **29.0693** | **0.0000** |
| trivial baseline | -4.6774 | 417.2070 | 207.5599 | 1.9987 | 0.5083 |

All figures in arcseconds. **The specification's prediction is confirmed.** The
analytic method withholds the direction and reports an infinite bound on every
record; the sharpest learned configuration reports a one sigma of 14.3540
arcsec, is wrong by 29.0693 of it, and its two-sigma interval contains the truth
on none of the 120 records.

No learned configuration inflates its reported uncertainty by more than
**1.2364x** along the unsupported direction relative to its own best-observed
one, where the analytic ratio is infinite.

### It is specific to the unobservable direction

| configuration | 2-sigma coverage, unsupported direction | 2-sigma coverage, best-observed direction |
|---|---|---|
| `forest` | 0.0750 | 1.0000 |
| `forest_deep` | 0.2083 | 1.0000 |
| `bagged_ridge` | 0.0000 | 0.7250 |
| trivial | 0.5083 | 1.0000 |
| batch, Cramer-Rao | refuses | 0.9750 |

Along the best-observed direction every learned configuration covers at or above
nominal. Along the unsupported direction, the same models on the same records
collapse. The overconfidence appears exactly where the data stops supporting an
answer, and **nothing in the model's own output distinguishes the two cases**.

### Record length does not fix it

A factor of 32 in record length, from 150 to 4800 samples: the learned error
along the unsupported direction stays at 419.8 to 421.0 arcsec and its reported
sigma grows only from 14.380 to 17.847 arcsec. A degree of freedom the motion
never excited is not a sample-size problem.

### Other failure cases

- **Outside its training record length.** Trained at 150 samples and applied to
  2400-sample records, the best learned configuration reaches 0.558510 mrad
  against 0.394345 for the MAP baseline on the same population. The figure is
  reported as a measurement of what happens outside the training distribution,
  not as a fair comparison.
- **A mis-declared mounting.** The learned estimator has no more ability than
  the analytic one to separate a mounting that was declared wrong from a
  misalignment. Control C8 measures the confusion at 5.000000 mrad of estimate
  movement for a 5 mrad declaration error.
- **Sensor errors the generator does not model.** No bias, scale factor, angular
  random walk, g-sensitivity, quantisation, latency or time-tag error appears in
  the training data, and nothing here says what the model does when one is
  present.

## Reproducibility

```bash
python -m pytest tests/ -q --junitxml=junit_self.xml   # 307 tests
python validation/validate_learned.py                  # §5 of VALIDATION.md
python validation/validate_unobservable.py             # §4
python examples/learned_versus_baselines.py
python examples/learned_calibration.py
python examples/unobservable_subspace.py
python -m aligncal benchmark --train 1200 --test 300 --samples 150
python -m aligncal unobservable --profile single_axis --samples 150
```

Seeds: training 20261029, test 70120261, controls 55120261, grid 31420261.
Every estimator's own random state is the training seed.

## Compute used

Two CPU cores, 7.8 GiB, no GPU. **PyTorch is not installed and is not used.**
scikit-learn 1.9.1 and numpy 2.5.3.

| operation | measured |
|---|---|
| generate 2500 training records, build their features and fit all four estimators | 18.1860 s |
| predict 600 test records, `forest` | 0.0868 s |
| predict 600 test records, `forest_deep` | 0.1021 s |
| predict 600 test records, `bagged_ridge` | 0.0117 s |
| solve 600 records with the batch estimator, eq (14)-(16) | 1.1885 s |
| the whole of `validate_learned.py` | 49 s |

**These timings are unstable** -- two shared cores -- and no ratio derived from
them appears in any conclusion.

## Ethical and safety limits

- **This model is not certified for operational flight use.**
- Research grade. Not flight-qualified, not certified, not approved for
  operational aerospace use.
- An alignment estimate produced here must not be applied to flight software.
- An observable alignment is not a calibrated one. Observability is a statement
  about what the record supports, not about the instrument.
- Every number is measured against a seeded synthetic generator. None of it
  transfers to a real vehicle without re-measurement against that vehicle's own
  data and error model.
- The model's reported uncertainty is an ensemble spread. On the measurement in
  this card it does not contain the truth in a direction the data cannot
  support, and nothing in the model's output marks that case. If a learned
  estimator is used at all, the observability verdict of
  `aligncal.observability.assess` must be read first and acted on.
