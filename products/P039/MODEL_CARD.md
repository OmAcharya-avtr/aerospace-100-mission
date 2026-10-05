# Model card — latencynet learned tail predictor

**This model is not certified for operational flight use.**

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use.

## Headline result, stated first

On the synthetic populations this model was developed against, **it is not the
best model, and on one of the two regimes it is worse than the analytic
baseline it was built to improve on.**

| regime | target | analytic sum-of-stages | OLS, 13 features | this model | verdict for this model |
|---|---|---|---|---|---|
| independent | p99 | 0.03655 | 0.03300 | 0.05167 | loses to both; paired p = 0.0009 against the analytic baseline |
| independent | p99.9 | 0.07086 | 0.05005 | 0.06080 | ahead of analytic but not significantly (p = 0.0988); loses to OLS |
| correlated | p99 | 0.15441 | 0.03155 | 0.05094 | beats analytic, p < 0.0001; loses to OLS |
| correlated | p99.9 | 0.22493 | 0.04107 | 0.05546 | beats analytic, p < 0.0001; loses to OLS |

Mean absolute log error on 150 held-out pipelines; lower is better. Source:
`validation/validate_model_comparison.py`, raw output in
`validation/validate_model_comparison_output.txt`.

Across a six-point hyperparameter sweep in both regimes (12 settings) this
model beat the analytic baseline on 6 — exactly the correlated-regime
settings — and beat the OLS baseline on **0 of 12**. The result is not an
artefact of the shipped hyperparameters, which are in fact the best of the six
in both regimes.

The honest conclusion: on this problem, as posed and as generated, a learned
tail predictor is not justified over a thirteen-feature ordinary least squares
fit. It does, however, confirm the specification's hypothesis — the learned
model helps only where stage latencies are dependent.

## Problem

Predict `ln q_p`, the natural log of the end-to-end latency quantile of a
serial staged pipeline at `p` in `{0.99, 0.999}`, from a short aligned
per-stage probe trace. The pipeline is one the model has never seen.

Why log space: end-to-end latency spans two orders of magnitude across the
population, and the engineering question is a relative error, not an absolute
one. A log-space error of 0.05 is a 5.1 % relative error.

Why a tail and not a mean: a deadline is violated by a single late pass.

## Baselines

Implemented first, and benchmarked against on the same held-out pipelines.

**Baseline 1 — analytic sum-of-stages** (`latencynet.analytic`). Two moment
identities, `E[S] = sum E[X_i]` and `Var[S] = sum Var[X_i]` under
independence (Casella & Berger 2002, Thm. 4.5.6 and Sec. 4.5), then a
Fenton-Wilkinson moment-matched lognormal quantile (Fenton 1960, *IRE
Transactions on Communications Systems* 8(1): 57-67). No parameters, no
training data, no seed. The identities are exact to **0.000e+00 relative
error** (`validation/validate_analytic_exactness.py`); the quantile map is
not, and under-predicts p99.9 by 2.36 % to 4.35 % depending on pipeline shape.
Interval: GUM propagation of probe sampling uncertainty (JCGM 100:2008,
Sec. 5.1.2).

**Baseline 2 — ordinary least squares** (`latencynet.linear`). OLS of
`ln q_p` on the thirteen probe features plus an intercept, solved by
`numpy.linalg.lstsq` on standardised columns. Interval: the exact Student-t
prediction interval `y_hat ± t·s·sqrt(1 + x'(X'X)^{-1}x)` (Draper & Smith
1998, Sec. 1.4).

**Diagnostic — covariance-aware analytic.** The same analytic model with the
probe's measured covariance matrix supplied, which makes equation (2) exact.
Not one of the two specified baselines; included so that any learned-model
advantage can be separated from the much cheaper statement "measure the
covariance". In the correlated regime it closes most of the gap on its own:
0.15441 → 0.03317 at p99.

## Architecture

Three `sklearn.ensemble.GradientBoostingRegressor` ensembles over the same
thirteen features:

| head | loss | purpose |
|---|---|---|
| point | `squared_error` | the point prediction of `ln q_p` |
| lower | `quantile`, `alpha = (1 - level)/2` | lower interval endpoint |
| upper | `quantile`, `alpha = (1 + level)/2` | upper interval endpoint |

Hyperparameters, fixed in `latencynet.learned.BoostingHyperparameters` and not
tuned against the test split:

| parameter | value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 3 |
| `learning_rate` | 0.05 |
| `min_samples_leaf` | 5 |
| `subsample` | 1.0 (so the fit is deterministic) |
| `random_state` | 20260403 |

Gradient boosting: Friedman (2001), *Annals of Statistics* 29(5): 1189-1232.
Quantile loss: Koenker & Bassett (1978), *Econometrica* 46(1): 33-50.

**Why trees and not a neural network.** PyTorch is not available in this build
environment. With 200 training pipelines and thirteen features, a boosted tree
ensemble of this size is the appropriate model class; a neural network would
have more parameters than training rows. The comparison therefore says nothing
about what a larger model trained on more data would do.

## Features

Thirteen, all computed from the probe trace only, never from the reference
sample that supplies the target. Full definitions in `latencynet.features`.

| idx | name | dependence-bearing |
|---|---|---|
| 0 | `n_stages` | |
| 1 | `log_sum_mean` | |
| 2 | `log_indep_sd` | |
| 3 | `log_dep_sd` | yes |
| 4 | `indep_cv` | |
| 5 | `dep_cv` | yes |
| 6 | `dependence_inflation` | yes |
| 7 | `mean_offdiag_corr` | yes |
| 8 | `max_offdiag_corr` | yes |
| 9 | `mean_stage_cv` | |
| 10 | `max_stage_cv` | |
| 11 | `mean_stage_skew` | |
| 12 | `log_sum_stage_p99` | |

The five dependence-bearing features are available to baseline 2 and to this
model, and deliberately not to baseline 1, because baseline 1 *is* the
independence assumption — that is the comparison the specification asks for.
Stage dependence is only observable at all because the probe trace is aligned
per pass; from marginal summaries alone the correlation is not identifiable and
no model of any kind could recover it.

Impurity-based importances of the shipped model at p99 (biased toward
high-cardinality features, Strobl et al. 2007, *BMC Bioinformatics* 8:25;
reported as what the model leaned on, not as a causal claim):

| feature | independent | correlated |
|---|---|---|
| `log_sum_stage_p99` | 0.8513 | 0.8439 |
| `log_sum_mean` | 0.0904 | 0.0925 |
| `log_indep_sd` | 0.0374 | 0.0034 |
| `log_dep_sd` | 0.0163 | 0.0561 |
| `max_stage_cv` | 0.0013 | 0.0006 |
| `mean_stage_skew` | 0.0009 | 0.0012 |

The model does find the dependence information when it is there —
`log_dep_sd` rises 3.4-fold and `log_indep_sd` collapses 11-fold between
regimes. It simply does not convert that into an advantage over a linear fit
on the same features.

## Dataset

Fully documented in [`DATASET_CARD.md`](DATASET_CARD.md). In summary: 410
synthetic pipelines per regime, two regimes, generated by
`latencynet.dataset.build_dataset` at seed 20260402. 2 to 6 lognormal stages
each, stage means log-uniform on [20, 500] us, stage coefficients of variation
uniform on [0.05, 0.60], and latent equicorrelation 0 (independent regime) or
uniform on [0.35, 0.90] (correlated regime). No real measurement anywhere.

## Training procedure

1. `build_dataset(regime, n_train=200, n_calibration=60, n_test=150,
   seed=20260402, n_probe=256, n_reference=25_000,
   n_reference_test=90_000)`.
2. Fit all three heads on the 200 training pipelines only.
3. Calibrate a split-conformal wrapper on the 60 calibration pipelines.
4. Score on the 150 test pipelines.

No early stopping, no validation-set hyperparameter search, no feature
selection. The hyperparameters above were written into the source before the
comparison was run and were not changed afterwards; the capacity sweep in
`validation/validate_model_comparison.py` is the check on that claim, and it is
reported in full rather than reduced to its best entry.

## Test-split strategy

**Split by pipeline, never by pass.** A held-out pipeline is one whose
parameters, probe trace and reference sample the model has never seen. No pass
from a test pipeline appears anywhere in fitting or calibration. Within a
pipeline, the probe trace and the reference sample use different seed streams
(`seed + 1` and `seed + 2`), so even the target is not computed from the model
input.

That is the only split that answers the operational question: will this work on
the next pipeline I profile.

## Metrics

| metric | definition | why |
|---|---|---|
| mean absolute log error | `mean abs(ln q_hat - ln q)` | to first order the mean relative error of the quantile |
| median absolute log error | the median of the same | so one badly mispredicted pipeline cannot decide a winner |
| signed bias | `mean(ln q_hat - ln q)` | negative means systematic under-prediction, the unsafe direction for a deadline |
| p90 absolute log error | 90th percentile of the absolute error | the bad case, not the typical one |
| interval coverage | fraction of held-out pipelines inside the interval, with binomial SE `sqrt(c(1-c)/n)` | an interval is worth nothing unless it delivers its claim |
| mean interval width in log space | `mean(ln upper - ln lower)` | the cost of that coverage, scale-free |
| paired t-test | on `abs(error_model) - abs(error_baseline)` over the same pipelines | whether a win is real |

Winner rule, fixed in `latencynet.compare` before the run: smallest mean
absolute log error on the test split, with significance at `p < 0.05` on the
paired test against the analytic baseline.

## Uncertainty output

The model does not emit a point estimate. Two intervals are available.

**Native** — the two quantile-loss heads. Measured coverage against a nominal
0.90 on the 4 rows evaluated: **0.520 to 0.733. This under-covers and is
reported as a failure.** Quantile regression carries no finite-sample coverage
guarantee and the heads are themselves estimates from 200 rows.

**Split-conformal** (`latencynet.conformal`) — the point prediction plus or
minus the `ceil((m+1)(1-alpha))`-th smallest absolute calibration residual.
Under exchangeability of calibration and test pipelines alone, and no
assumption whatever about the model, this has marginal coverage in
`[1-alpha, 1-alpha + 1/(m+1)]` (Vovk et al. 2005, Ch. 2; Lei et al. 2018,
*JASA* 113(523), Sec. 2.2). Measured on this model at nominal 0.90:

| regime | target | measured conformal coverage | binomial SE | mean log width |
|---|---|---|---|---|
| independent | p99 | 0.927 | 0.021 | 0.2474 |
| independent | p99.9 | 0.887 | 0.026 | 0.2908 |
| correlated | p99 | 0.927 | 0.021 | 0.2500 |
| correlated | p99.9 | 0.987 | 0.013 | 0.3641 |

Over all 12 conformal rows at the three levels measured, 7 fall inside a
two-sided two-sigma band and the one-sided guarantee holds on all but the
over-covering rows. For comparison, baseline 2's conformal intervals are
**about 1.6 times narrower at matched coverage** (mean log width 0.1220
against 0.1997, averaged over all 12 rows), which is the same conclusion as the
accuracy table in different units.

**Calibration caveats.** The conformal guarantee is marginal, not conditional:
the interval has the same width for every pipeline, so it over-covers easy
ones and under-covers hard ones. Its achievable coverage is quantised in steps
of `1/(m+1) = 0.0164` with the 60-pipeline calibration split. And it inherits
exchangeability: here that holds by construction because calibration and test
pipelines come from the same generator, and on a real fleet it would not hold
across a hardware revision.

## Failure cases

1. **Independent stages at p99.** The model loses to a parameter-free analytic
   identity, 0.05167 against 0.03655, with the paired test significant against
   it (p = 0.0009). Where the physics is exact, a learned model fitted to a few
   hundred noisy targets adds error rather than removing it.
2. **Against the linear baseline, everywhere.** 0 wins in 4 comparisons and 0
   in 12 hyperparameter settings. The target is close to linear in the log-scale
   features, and a closed-form fit on 200 rows beats an ensemble on the same
   rows.
3. **Native intervals under-cover** at every level measured (0.520 to 0.733
   against nominal 0.90). Do not use them; use the conformal wrapper.
4. **Pipelines outside the generator's range.** Stage counts above 6, stage
   coefficients of variation above 0.60, means outside [20, 500] us, negative
   stage correlation, or any dependence structure other than equicorrelated:
   all untested. The feature scaling is learned from the training population
   and tree models do not extrapolate — predictions outside the training hull
   are clamped to the boundary leaves, which will look confident and be wrong.
5. **Multimodal stage latencies.** A stage with a cache-hit and a cache-miss
   mode is not representable by the features, and no error is raised.
6. **Non-serial pipelines.** Overlap, queueing, preemption or contention
   between stages: entirely outside the model.
7. **Real hardware.** The model has never seen a real profile.

## Reproducibility

```bash
cd validation
PYTHONPATH=../src python3 validate_model_comparison.py
PYTHONPATH=../src python3 validate_interval_coverage.py
```

Deterministic given: dataset seed 20260402, `random_state` 20260403,
`subsample = 1.0`, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, Python
3.13.16. Fit determinism is pinned by a test
(`tests/test_learned.py::test_fit_is_deterministic`). Across scikit-learn
versions the tree-building tie-breaks can differ; the version is recorded here
for that reason.

## Compute used

| step | cost |
|---|---|
| three heads, 300 trees each, 200 rows × 13 features | about 1.2 s on one CPU core |
| full comparison, both regimes, both targets, plus the 12-setting sweep | about 31 s CPU |
| coverage measurement, 3 levels, both regimes, both targets | about 17 s CPU |
| peak memory | under 400 MB |

No GPU, no accelerator, no PyTorch. Total training compute for every model in
this product is under a minute of one core.

## Ethical and safety limits

- **This model is not certified for operational flight use.** It must not be
  used to set, verify or justify a deadline on any system where a timing
  violation has a safety consequence.
- It was trained and evaluated entirely on synthetic data from a generator
  whose assumptions — lognormal stage marginals, Gaussian-copula
  equicorrelated dependence, serial composition — are modelling choices and
  not established facts about real embedded pipelines. Its measured accuracy
  is accuracy against that generator and transfers no further.
- Its errors are **biased low** in the correlated regime when the analytic
  baseline is used instead (-14 % and -20 % at p99 and p99.9) and biased
  slightly high for this model itself (+0.0183 and +0.0151 in log space). A
  low-biased tail prediction is the dangerous direction: it reports a deadline
  as met when it is not.
- No personal data, no human subjects, no dual-use concern. The only ethical
  exposure is over-trust in a number, which is what the prediction intervals
  and this card exist to prevent.
- Where this model and the analytic baseline disagree, the recorded evidence
  says to use neither: use baseline 2.
