# Model card — `calibaudit` recalibration maps

**This model is not certified for operational flight use.** Research-grade,
validation level 2. The learned components here are two classical, 1-D
recalibration maps, measured against the raw forecast as baseline. The
headline result is that **on small samples the baseline wins**, and on a
forecast that is already calibrated it wins at every sample size tested.

Every number in this card was produced by a script in `validation/`, executed
in the build container on 2026-10-10, with its raw stdout committed under
`validation/outputs/`. The producing script is named beside each table.

## Problem

Given `n` pairs of a forecast probability `f_i in [0, 1]` and a binary outcome
`o_i in {0, 1}`, learn a map `h: [0, 1] -> [0, 1]` from a training split such
that `h(f)` scores better than `f` on a held-out split, under the Brier score
and the logarithmic score.

This is a one-dimensional supervised problem with one input feature. There is
no feature engineering, no representation learning and no neural network. The
whole learned object is either two scalars or a monotone step function, and
the interesting question is not what it can represent but whether fitting it
at all is worth the estimation variance it adds.

## Baseline, implemented and validated before either learned map

| Baseline | What it is | Parameters fitted |
|---|---|---|
| **`RawForecast`** | the identity map, `h(f) = f` | none |

`RawForecast` is the baseline and it is not a straw man. A forecast that is
already calibrated cannot be improved by recalibration, so the identity map is
*optimal* in that case, and any map fitted on a finite split is strictly worse
in expectation. The baseline is evaluated on exactly the same held-out indices
as the learned maps, and the comparison is paired: `delta_brier` is computed
per bootstrap resample of the test set with both scores recomputed on the same
resample.

A second reference is available for skill scores, `base_rate_forecast`, the
constant climatological forecast. It is not used as the recalibration baseline
because it is not a recalibration of anything.

## Architecture

### `PlattScaling` — the two-parameter learned map

`h(f) = sigmoid(a * logit(f) + b)`, with `a` dimensionless and `b` in logit
units. Fitted by maximum likelihood on the Bernoulli log-likelihood with
`scipy.optimize.minimize(method="L-BFGS-B", jac=True)`, analytic gradient,
`maxiter=500`, `ftol=1e-14`, `gtol=1e-10`, started from `(a, b) = (1, 0)`,
which is the identity map. `logit` input is clipped to `[1e-12, 1 - 1e-12]`.

Reference: Platt, J. C. (1999), "Probabilistic outputs for support vector
machines and comparisons to regularized likelihood methods", in *Advances in
Large Margin Classifiers*, MIT Press, 61–74.

`target_smoothing=True` replaces the 0/1 targets with Platt's smoothed
targets `t+ = (N+ + 1)/(N+ + 2)` and `t- = 1/(N- + 2)`. It is **off by
default**, so what gets measured is the plain maximum-likelihood fit.

Two parameters means it cannot repair a non-monotone calibration curve and
cannot change the ranking of forecasts. Both limits are deliberate.

### `IsotonicCalibration` — the nonparametric learned map

A monotone step function fitted by pool-adjacent-violators, through
`sklearn.isotonic.IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
out_of_bounds="clip")`. This package **wraps rather than reimplements**: on a
1001-point grid its predictions are bit-identical to a direct sklearn fit,
worst difference **0.000000e+00** (`validation/validate_sklearn_interop.py`).

Reference: Zadrozny, B. and Elkan, C. (2002), "Transforming classifier scores
into accurate multiclass probability estimates", *KDD 2002*, 694–699,
doi:10.1145/775047.775151.

Nonparametric means it is the first to overfit, and it returns exactly 0 and
1 on pure blocks, which the logarithmic score punishes heavily. Both effects
are measured below.

### Why not PyTorch, and why not a bigger model

PyTorch is not installed in the build container and installing it is outside
the 2-core / 7.8 GiB compute envelope (ADR-005 amendment). It would make no
difference: this is a monotone 1-D regression on at most tens of thousands of
points, which a two-parameter logistic fit and a PAV solve already handle
exactly.

### A scikit-learn API note that matters to this exact task

`CalibratedClassifierCV(estimator, cv="prefit")` was the documented way to
recalibrate an already-fitted classifier. On scikit-learn 1.9.1 it raises
`InvalidParameterError`: *"The 'cv' parameter of CalibratedClassifierCV must
be an int in the range [2, inf), an object implementing 'split' and
'get_n_splits', an iterable or None. Got 'prefit' instead."* The replacement
is `CalibratedClassifierCV(FrozenEstimator(estimator))`, verified working in
`validation/outputs/validate_sklearn_interop_output.txt`. This package never
takes that path, because it fits maps on forecast values rather than wrapping
an estimator, but anyone doing the same job with sklearn on this version will.

## Uncertainty output

Both learned maps expose `predict_with_interval(forecasts, level=0.9)`, which
returns `(point, lower, upper)`. The interval is a pointwise percentile
interval over a bootstrap ensemble of maps refitted on resamples of the
training split, with the ensemble size set by `n_bootstrap` at construction.

It is the uncertainty **of the fitted map**, not of the outcome: it answers
"how well determined is this recalibrated probability by the data I fitted
on", which is the question a user needs before trusting a recalibrated number.
Measured behaviour, from `tests/test_recalibration.py`:

- interval width shrinks with training size (200 samples against 6400);
- isotonic's intervals are wider than Platt's at 300 training samples, which
  is the same overfitting fact as the harm rates below, seen from the other
  side.

From the worked example at n = 1200, `PlattScaling(n_bootstrap=200)`:

| forecast | recalibrated | 90 % interval on the map |
|---|---|---|
| 0.1 | 0.2015 | [0.1529, 0.2779] |
| 0.5 | 0.4634 | [0.4401, 0.4844] |
| 0.9 | 0.7472 | [0.6558, 0.8105] |

The interval at `f = 0.9` is 0.155 wide. A user quoting "0.75" from this map
without that width is overstating what 600 training cases support.

## Dataset

See [DATASET_CARD.md](DATASET_CARD.md). Every dataset in this repository is
synthetic, generated by committed code from fixed seeds. No data file is
committed; the largest file in the repository is a PNG.

The generator is `calibaudit.synthetic`: `p ~ Beta(a, b)`,
`o | p ~ Bernoulli(p)`, `f = g(p)` for a declared strictly increasing
distortion `g`. Because `f` is a deterministic monotone function of `p`, the
exact calibration curve is `E[o | f] = g^{-1}(f)`, which is what makes a
*measured* bias possible at all.

## Training procedure and test-split strategy

- **Split:** a single seeded random permutation, `test_fraction=0.5` by
  default. Not cross-validation, deliberately: the question being answered is
  what one practitioner with one dataset gets, which is the situation in which
  recalibration is actually applied. The cost is that the audit's own variance
  is part of the result, which is why `sample_size_sweep` reports `n_replicates`
  fresh datasets per cell rather than one.
- **Fitting:** each map is fitted on the training indices only. `RawForecast`
  is fitted on nothing, by construction.
- **Evaluation:** Brier score, logarithmic score, ECE and MCE on the held-out
  indices. `delta_brier` is `BS_method - BS_raw` on the same held-out samples;
  its interval is a paired percentile bootstrap over the test indices, with
  both scores recomputed on each resample.
- **Verdict:** `"improved"` when the whole interval lies below 0, `"worse"`
  when it lies entirely above 0, `"indistinguishable"` otherwise. There is no
  threshold to tune and no p-value to cross.
- **Hyperparameters:** none were searched. Platt has no hyperparameter beyond
  the optimiser tolerances; isotonic has none. Nothing in this card was
  selected by looking at a test split.
- **Compute:** the whole sweep below is 8 sample sizes × 80 replicates × 3
  methods, about 76 s on two contended cores.

## Metrics — does recalibration help?

`validation/validate_recalibration.py`, 8 sample sizes, 80 replicates each,
50/50 split, fresh data per replicate. `mean_dBS` is the mean held-out Brier
change against the raw baseline; **positive is worse than doing nothing**.
`harm_rate` is the fraction of replicates in which the method is strictly
worse than the raw forecast.

### Spec `overconfident` (population ECE 0.075204, there is something to fix)

| n_total | method | mean_dBS | sem | harm_rate |
|---|---|---|---|---|
| 60 | platt | **+0.005822** | 0.003050 | **0.512** |
| 60 | isotonic | **+0.016848** | 0.003634 | **0.713** |
| 100 | platt | +0.002458 | 0.001880 | 0.487 |
| 100 | isotonic | +0.011628 | 0.002369 | 0.688 |
| 200 | platt | −0.003078 | 0.000942 | 0.350 |
| 200 | isotonic | **+0.005541** | 0.001160 | **0.688** |
| 400 | platt | −0.005355 | 0.000653 | 0.150 |
| 400 | isotonic | **+0.000013** | 0.000825 | **0.525** |
| 1000 | platt | −0.005611 | 0.000383 | 0.062 |
| 1000 | isotonic | −0.003182 | 0.000401 | 0.150 |
| 2500 | platt | −0.006702 | 0.000249 | 0.000 |
| 2500 | isotonic | −0.005318 | 0.000259 | 0.013 |
| 6000 | platt | −0.006762 | 0.000145 | 0.000 |
| 6000 | isotonic | −0.005939 | 0.000150 | 0.000 |
| 15000 | platt | −0.006735 | 0.000093 | 0.000 |
| 15000 | isotonic | −0.006263 | 0.000093 | 0.000 |

**Crossing points: Platt at n_total = 200, isotonic at n_total = 1000.** That
ordering agrees with Niculescu-Mizil and Caruana (2005), *ICML*, 625–632,
which reports isotonic regression needing of order 1000 training samples
before it beats Platt scaling. The agreement is a check, not a coincidence
that was looked for afterwards: the expectation is stated in the validation
script before the measurement.

### Spec `calibrated` (population ECE exactly 0, nothing to fix) — the honest negative

| n_total | method | mean_dBS | sem | harm_rate |
|---|---|---|---|---|
| 60 | platt | +0.011345 | 0.002834 | 0.637 |
| 60 | isotonic | +0.022335 | 0.003569 | 0.775 |
| 200 | platt | +0.003725 | 0.000650 | 0.825 |
| 200 | isotonic | +0.012344 | 0.001072 | 0.963 |
| 1000 | platt | +0.000778 | 0.000146 | 0.700 |
| 1000 | isotonic | +0.003207 | 0.000293 | 0.925 |
| 6000 | platt | +0.000088 | 0.000022 | 0.713 |
| 6000 | isotonic | +0.000910 | 0.000053 | 0.988 |
| 15000 | platt | **+0.000048** | 0.000009 | **0.688** |
| 15000 | isotonic | **+0.000520** | 0.000023 | **1.000** |

**Both learned maps are worse than the baseline at every sample size tested,
and isotonic regression is worse in 100 % of 80 replicates at 15 000
samples.** The effect shrinks as `1/n`, as estimation variance must, but it
does not change sign, and it cannot: there is no miscalibration to remove, so
the only thing a fitted map can contribute is its own variance. This result is
retained verbatim per the mission's honest-negative policy. It is not an
argument against recalibration; it is the reason to audit before applying it,
which is what this package is for.

### Parameter recovery — the one exact check available

`PlattScaling` against the analytic inverse of each known distortion, at
n = 200 000 (`validation/validate_recalibration.py`):

| spec | exact `a` | exact `b` | worst fitted error |
|---|---|---|---|
| `calibrated` | 1.0 | 0.0 | `|da| = 0.01429`, `|db| = 0.00130` |
| `overconfident` | 0.6 | 0.0 | `|da| = 0.00858`, `|db| = 0.00130` |
| `underconfident` | 1.6 | 0.0 | `|da| = 0.02287`, `|db| = 0.00130` |
| `biased_high` | 1.0 | −0.6 | `|da| = 0.01429`, `|db| = 0.00727` |

Worst over all four at n = 200 000: `|da| = 0.02287`, `|db| = 0.00727`,
against a documented expectation of 0.03, which is about five standard errors
of a two-parameter logistic fit at this sample size.

### Agreement with scikit-learn

`validation/validate_sklearn_interop.py`:

| Comparison | Worst difference |
|---|---|
| `IsotonicCalibration` against a direct `IsotonicRegression` fit, 1001-point grid | **0.000000e+00** |
| `PlattScaling` against `LogisticRegression(C=1e8)` on `logit(f)`, 8 configurations | `|da| = 4.991e-09`, `|db| = 6.858e-09` |

## Failure cases, measured

1. **Small samples.** Both maps harm the held-out Brier score below a few
   hundred cases; isotonic harms in **71.3 %** of replicates at 60 samples and
   **68.8 %** at 200.
2. **An already-calibrated forecast.** Both maps are worse at every sample
   size tested, up to 15 000 (table above).
3. **Isotonic returns exact 0 and 1.** After a 200-sample fit, **24 of 200**
   held-out predictions were exactly 0 or 1 and **4 of those were wrong**,
   taking the held-out log score from **0.639511** to **1.272654** while the
   Brier score moved only from 0.221221 to 0.224179. The two proper scoring
   rules disagreed in sign about isotonic at **2 of 5** sample sizes.
4. **Forecasts of exactly 0 or 1 destroy the Platt fit.** On hard-thresholded
   forecasts the fitted slope collapsed from **0.559941** to **0.027358**,
   because `logit(0)` is clipped to −27.6 and enters the least-squares-like
   likelihood as a dominant outlier. Clip model outputs into the interior
   before recalibrating.
5. **Recalibration can improve the ECE while worsening the Brier score.** In
   the worked example Platt took the held-out ECE from **0.071500** to
   **0.050742** and the Brier score from **0.207283** to **0.208777**. A
   recalibration judged on ECE alone can be a regression on the score that
   matters.
6. **Non-monotone miscalibration is out of reach for Platt** by construction,
   and no spec in this repository exhibits it, so that limitation is stated
   rather than measured.

## Reproducibility

Exact commands, from the repository root, after `pip install -e ".[test]"`:

```bash
python validation/validate_recalibration.py
python validation/validate_sklearn_interop.py
python validation/worked_example.py
MPLBACKEND=Agg python examples/recalibration_audit.py
python -m pytest tests/test_recalibration.py tests/test_sklearn_interop.py -q
```

Seeds: the recalibration sweep uses `seed=56` and derives per-replicate seeds
as `56 * 1000003 + r + 1`; the audit inside it uses `56 * 7 + r + 1`. The
worked example uses `numpy.random.default_rng(2026)` and audit seed 3. Every
figure above is deterministic in those seeds and reproduces exactly, except
wall-clock times.

**Sample size is part of the seed.** `sample_forecast(spec, n, seed=s)` draws
each array in one call from one stream, so a draw of 200 is not a prefix of a
draw of 400 in the outcomes. Changing `n` changes every number.

## Compute used

Two shared, contended CPU cores, under 1 GiB resident. No GPU. Measured
wall-clock, single runs, moving 10–40 % between runs:

| Work | Time |
|---|---|
| `validate_recalibration.py` (3 specs × 8 sizes × 80 replicates × 3 methods) | about 76 s |
| `validate_sklearn_interop.py` | about 6 s |
| `examples/recalibration_audit.py` (7 sizes × 60 replicates) | about 24 s |
| the full test suite, 405 tests | 117 s, 136 s and 146 s in three timed runs |

Nothing in this repository approaches the mission's three-minute per-run
compute budget.

## Ethical and safety limits

- **Not certified for operational flight use.** Nothing in this repository is.
- The maps describe the sample they were fitted on. A recalibration fitted on
  one operating regime and applied in another is not validated by anything
  here, and this package cannot detect that it has happened.
- Every evaluation is on synthetic data whose generative model is known.
  Agreement with a closed-form population value is evidence about the
  implementation, not about any real forecasting system.
- `recalibration_audit` reports `"indistinguishable"` rather than picking a
  winner when the paired interval straddles zero. A caller who treats
  `"indistinguishable"` as `"improved"` is making a claim the package
  declined to make.
- No human-subject data, no personal data, no dual-use concern: the inputs are
  probabilities and binary outcomes.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.
