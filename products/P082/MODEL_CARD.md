# Model card: the learned spatio-temporal correlation model

**This model is not certified for operational flight use.**

## Problem

Given a station pair, a seasonal regime and the marginal cloud climatology,
predict the **latent Gaussian-copula correlation** of the two stations' cloud
indicators. Everything downstream — the bivariate orthant of equation (8), the
`n`-station Monte Carlo of equation (10) — is shared with the baselines, so the
model's job is exactly the correlation and nothing else.

The reason that is the right target: a site-diversity availability figure is a
joint unavailability, and the only thing separating the independence assumption
from the truth is the dependence structure. Making all four model families
produce a correlation and pass it through the same copula isolates the
dependence model and reports in the units a network designer cares about.

## Baselines, implemented and validated first

The analytic baselines were built, validated against the generator's own ground
truth, and reported **before** the learned model was fitted. That ordering is
not a courtesy: an under-strength baseline inflates an apparent learned margin.

| model | what it is | reference |
| --- | --- | --- |
| `independent` | `rho = 0`. Baseline 1, equation (14). The thing being refuted. | ITU-R P.1815-1 and the P.618 site-diversity material both start from a joint statistic and warn that an independence assumption is optimistic |
| `exponential` | `rho_0 exp(-d/L)`, pooled over regimes. Baseline 2 **as specified**, equation (23). | Matern at smoothness one half; Stein 1999 chapter 2 |
| `per_regime` | the same, fitted independently per seasonal regime. The **strengthened** baseline, equation (24). | the mission build rule: where a named baseline measures weakly, use the strongest honest baseline and report the named one beside it |
| `learned` | the model this card describes, equations (35) to (38). | Friedman 2001 |

Baseline 3, the direct empirical joint rate of equation (20) with a moving-block
bootstrap, is not in the table: it is the **reference** the table is scored
against.

The specified baseline is **not** under strength on this record. Measured in
`validation/validate_correlation_output.txt`: pooled MAE 0.058410 against
per-regime 0.057737 in correlation units on the training cell of the
anisotropic record, a margin of 0.000674. Both are reported everywhere.

## Architecture

Three `sklearn.ensemble.GradientBoostingRegressor` models with the quantile
loss at declared levels `(0.1, 0.5, 0.9)`. The median model is the point
estimate; the outer two give an 80 per cent interval.

Gradient boosting rather than a neural network for three stated reasons: PyTorch
is not installed in the measured environment; the training set is 420 rows in
the declared configuration and a tree ensemble is the right capacity for that;
and the quantile loss gives a prediction interval directly rather than through a
separate calibration step.

| hyperparameter | value |
| --- | --- |
| trees | 240 |
| max depth | 3 |
| learning rate | 0.05 |
| subsample | 0.9 |
| random state | 12345 |
| quantile levels | 0.1, 0.5, 0.9 |

**These were chosen once, against the TRAIN cell only. No search was run against
either hold-out cell**, which would have made both hold-out scores meaningless.
`TrainingReport.tuned_against_holdout` carries that as a field rather than as a
sentence, and `validation/validate_learned_model.py` prints it.

## Features

Eight, listed in `sitediverse.features.FEATURE_NAMES`:

| # | feature | why |
| --- | --- | --- |
| 0 | `separation_km` | equation (1); the only spatial feature the baseline has |
| 1 | `along_km` | equation (3); invisible to the baseline |
| 2 | `cross_km` | equation (3); invisible to the baseline |
| 3 | `anisotropy_ratio` | `along / (along + cross + 1 km)` |
| 4 | `regime` | the declared seasonal quarter — the calendar, not the record |
| 5 | `rate_min` | the smaller of the two stations' long-term cloud rates |
| 6 | `rate_max` | the larger |
| 7 | `rate_product` | their product, which is equation (14) for the pair |

**Deliberately excluded**: station identity, latitude, longitude and altitude.
A model given the station identity memorises the training pairs and the held-out
SITES score then measures nothing. Altitude is excluded because the generator
does not use it, so including it would let the model fit noise with a predictor
that cannot generalise even in principle. Both exclusions decide a hold-out
result, so they are written down rather than left in the code.

## Dataset

See `DATASET_CARD.md`. In one line: a **synthetic** concurrent cloud record from
a declared generator with a known correlation structure; no real site
climatology; nothing transfers to a real network.

## Test-split strategy

Two hold-outs, **reported separately and never merged**, because a
spatio-temporal model can pass one and fail the other — and on this record it
does.

A row is `(station i, station j, regime, time half)`. The record is cut in half
by time at its midpoint and five of twenty stations are held out, one from each
regional group so that the held-out set spans the full separation range (126.0
km to 7299.5 km against the network's 91.9 km to 7299.5 km).

| cell | stations | time half | rows |
| --- | --- | --- | --- |
| TRAIN | both trained | first | 420 |
| held-out SITES | at least one held out | first | 340 |
| held-out TIME | both trained | second | 420 |
| *(unscored)* | at least one held out | second | 340 |

The fourth cell is **not** a headline score: it is a joint extrapolation in both
directions and a score on it cannot be attributed to either. It is computed and
printed in `validation/validate_learned_model_output.txt` so a reader can see it
exists and see that it is not mixed in.

**The one leak, stated.** The marginal climatology of
`sitediverse.climatology` is estimated once over the **whole** record and used
in both time halves. Estimating it twice would make the two halves' targets
incomparable, since each would be inverted against a different marginal model.
The leak is the climatology only, it is identical for every model in the
comparison including the baselines, and it is named here, in
`src/sitediverse/splits.py` and in README.md.

## Metrics

Mean absolute error in **joint unavailability** against the empirical reference
of equation (20), over the rows of a cell. Absolute and not squared: the
quantity is a probability bounded below by zero and a squared error is dominated
by the few near pairs with the largest joint rates, which every model gets
roughly right. The MAE in the implied **correlation** is reported beside it
because the two can disagree.

### Measured, on the anisotropic reference record

From `validation/validate_learned_model_output.txt`. Joint-unavailability MAE:

| cell | rows | independent | exponential | per_regime | learned |
| --- | --- | --- | --- | --- | --- |
| train | 420 | 0.018855 | 0.007910 | 0.007758 | 0.004633 |
| held-out SITES | 340 | 0.023843 | 0.012481 | 0.012509 | 0.010406 |
| held-out TIME | 420 | 0.020464 | 0.008923 | 0.008870 | 0.012329 |

### Measured, on the isotropic control record

Where the exponential baseline is **correctly specified by construction**:

| cell | rows | independent | exponential | per_regime | learned |
| --- | --- | --- | --- | --- | --- |
| train | 420 | 0.022081 | 0.007596 | 0.007510 | 0.004605 |
| held-out SITES | 340 | 0.023205 | 0.008736 | 0.008787 | 0.008023 |
| held-out TIME | 420 | 0.022532 | 0.008220 | 0.008382 | 0.011550 |

### The margins, equation (41)

Margin = baseline MAE minus learned MAE, so positive favours the learned model.
`per_regime` is the strongest baseline on the TRAIN cell of both records, so it
is the one quoted here; the `exponential` margins are within 0.00006 of these
and are in the validation output.

| record | cell | margin | 95% CI on the margin | CI contains 0 | reference half-width | margin inside it |
| --- | --- | --- | --- | --- | --- | --- |
| anisotropic | held-out SITES | **+0.002103** | [+0.000249, +0.002614] | no | 0.005952 | yes |
| anisotropic | held-out TIME | **-0.003459** | [-0.003248, -0.001384] | no | 0.005758 | yes |
| isotropic | held-out SITES | **+0.000764** | [-0.000604, +0.001409] | **yes** | 0.004892 | yes |
| isotropic | held-out TIME | **-0.003168** | [-0.002945, -0.001241] | no | 0.006598 | yes |

**The finding, stated plainly.** On every cell of both records the learned
margin is **smaller than the bootstrap half-width on the empirical reference**.
Where the baseline is correctly specified (isotropic, held-out SITES) the
learned model is not distinguishable from it at all: the paired interval
contains zero. Where the baseline is mis-specified (anisotropic, held-out SITES)
the learned model is better by 0.002103 — a third of the uncertainty in the
thing it is being scored against. On **held-out TIME, on both records, the
baseline beats the learned model**, by more than the learned model's best win.

That is the result. It is not a failure, it is not softened, and nothing was
retuned to change it.

### Against the generator's own correlation

The sharpest finding in this card, and the one a real record could never
produce. Scored against the **truth** rather than against the noisy implied
correlation, on held-out SITES of the anisotropic record
(`examples/correlation_models.py`):

| cell | independent | exponential | per_regime | learned |
| --- | --- | --- | --- | --- |
| train | 0.093279 | 0.012107 | **0.010578** | 0.043919 |
| held-out SITES | 0.122407 | 0.043280 | **0.043236** | 0.063107 |

The learned model is **closer to the noisy target and further from the truth**.
It is fitting the estimation noise in the implied correlation of equation (11),
whose own rms error against the truth is 0.073751. A comparison against an
empirical reference cannot see this; only a synthetic record can.

## Uncertainty output

`predict` returns a `CorrelationPrediction` carrying `lower`, `median` and
`upper`. **There is no method on that class returning a bare number**: a caller
who wants a point estimate reads `.median` and in doing so has the interval in
hand. `tests/test_learned.py` asserts that structurally.

The interval is an 80 per cent quantile-regression band on the conditional
distribution of the target given the features. It is therefore an estimate of
where a future implied correlation with these features would fall. It is **not**
a confidence interval on the true correlation, and it is **not calibrated by
construction**.

Measured coverage, anisotropic record:

| cell | coverage of the implied rho | coverage of the TRUE rho | median width | crossings |
| --- | --- | --- | --- | --- |
| train | 0.8000 | 0.8071 | 0.1398 | 8 of 420 |
| held-out SITES | 0.6147 | 0.7441 | 0.1371 | 43 of 340 |
| held-out TIME | 0.4643 | 0.8071 | 0.1398 | 8 of 420 |

The TRUE-correlation coverage is identical on the train and held-out-TIME rows,
and that is not a transcription error: **none of the eight features encodes the
time half**, so a held-out-TIME row gets the same prediction as its train-cell
twin. What differs between the two cells is the *target*, which is re-estimated
from the second half's epochs. Held-out TIME therefore tests whether a
correlation model fitted on the first half still reproduces the second half's
joint rates — not whether the model extrapolates in time, which it has no
mechanism to do. The distinction is stated here and in the validation output
because overstating it would overstate the result.

**The interval is badly under-covered out of sample, and that is reported
rather than claimed away.** Nominal is 0.80; held-out TIME reaches 0.4643. An
uncertainty output whose coverage is not measured is decoration, so the
measurement is in the validation output whichever way it comes out, and a
negative control asserts that the under-coverage is detected.

Three quantile models fitted independently can **cross**. Crossings are counted
and reported rather than repaired by sorting, because the count is a diagnostic
and a sorted interval hides it: 102 of 1520 rows on the anisotropic record
and 138 of 1520 on the isotropic one.

## Failure cases

- **Extrapolation in time.** The worst measured case. On held-out TIME the
  learned model is beaten by a two-parameter fit on both records, and its
  interval covers 46 per cent where it claims 80.
- **Fitting the target's noise.** Against the generator's truth the learned
  model is worse than the baseline even where it beats it against the empirical
  reference. The implied correlation of equation (11) has an rms error of
  0.073751 against the truth and the model is partly learning that.
- **Not constrained to be a correlation.** A tree ensemble can return a value
  outside `[-1, 1]`. The median is clamped into `[-0.95, 0.995]` before the
  copula sees it and the clamp count is reported; on the declared records it is
  0 of 1520.
- **No positive-semi-definiteness guarantee.** A correlation matrix assembled
  from learned pair predictions need not be a correlation matrix.
  `correlation.matrix_from_pairs` projects it and **reports** the least
  eigenvalue and the Frobenius distance moved.
- **Nothing in it knows the physics.** The model has no notion of advection, of
  cloud type or of the geometry beyond the eight features. It will extrapolate
  confidently outside the separation range it was fitted on.

## Compute

Measured on 2 cores, from `validation/validate_learned_model_output.txt`:

| stage | wall clock |
| --- | --- |
| generate the 8-year 20-station record and estimate its climatology | 0.072 s |
| build the 1520-row pair table (1520 copula inversions) | about 4 s |
| fit the three quantile models | 1.217 s |
| score and bootstrap 400 replicates | about 1 s |
| **the whole study, one record** | **6.710 s** |
| the whole validation suite, 14 scripts | 238.7 s |
| `probe_working_directories.py`, 70 runs | 278.2 s |

No training or Monte Carlo run in this repository approaches the declared
three-minute budget. No GPU is used, required or available.

## Reproducibility

```bash
python validation/validate_correlation.py      # the baselines, validated FIRST
python validation/validate_learned_model.py    # then the comparison
python -m sitediverse learned --record anisotropic
python examples/holdout_scores.py
python examples/correlation_models.py
python examples/learned_interval.py
```

Seeds: the record is `20260002` (anisotropic) or `20260003` (isotropic); the
model's `random_state` is `12345`; the margin bootstrap seed is `770001` with
400 replicates and a block of 64 epochs. Every figure is a deterministic
function of those. No model artifact is serialised or committed: every model is
refitted from its declared seed by the script that reports it.

Reproducible **within** a version of scikit-learn, not across versions.
`pyproject.toml` bounds `scikit-learn` to the minor version for exactly that
reason.

## Ethical and safety limits

The model predicts a correlation between two synthetic cloud-cover series. It
describes no person and contains no personal data.

This software is research grade: **not flight-qualified, not certified, not
approved for operational aerospace use.** **This model is not certified for
operational flight use.** A reader who takes a correlation or an availability
figure from this repository into a real link budget has made an error.
