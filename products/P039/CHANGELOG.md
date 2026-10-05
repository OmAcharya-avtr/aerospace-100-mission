# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-05

First release. Validation level 2, research-grade. Status: TESTING.

### Added

- `latencynet.units` — scalar second/millisecond/microsecond/nanosecond
  conversions, so no call site has to write a bare `1e-6`.
- `latencynet.pipeline` — `StageSpec` and `PipelineSpec` for injected
  synthetic pipelines with lognormal or constant stage marginals, Gaussian
  copula dependence on the latent normals, closed-form injected mean,
  covariance and standard deviation, and a deterministic sampler whose
  generator-stream consumption order is part of the reproducibility contract.
- `latencynet.analytic` — baseline 1, the sum-of-stages model. The two moment
  identities (mean linearity, variance additivity), the Fenton-Wilkinson
  moment-matched lognormal quantile labelled as an approximation everywhere,
  and a native prediction interval from the GUM law of propagation of
  uncertainty over probe sampling error.
- `latencynet.features` — thirteen features from an aligned 256-pass probe
  trace, five of which carry stage-dependence information, with the
  identifiability argument for why alignment is what makes dependence
  observable at all.
- `latencynet.dataset` — synthetic pipeline populations in two dependence
  regimes, split by pipeline into train/calibration/test, with
  distribution-free order-statistic uncertainties recorded for every reference
  target so the noise floor of the comparison is visible.
- `latencynet.linear` — baseline 2, OLS on standardised features with the
  exact Student-t prediction interval.
- `latencynet.learned` — the learned predictor: three gradient-boosted tree
  ensembles (point plus two quantile-loss interval heads), deterministic at
  `subsample = 1.0`.
- `latencynet.conformal` — split-conformal intervals for any predictor, with
  the coverage quantisation `1/(m+1)` reported and an infinite radius returned
  honestly when the calibration split is too small to certify the level.
- `latencynet.tails` — two explicitly named quantile definitions with no
  changeable default, the analytic order-statistic standard error, the
  smallest sample size at which a tail probability is interior, and a
  convergence harness that fits the log-log rate against the predicted -1/2.
- `latencynet.metrics` — coverage with binomial and Wilson intervals,
  log-space accuracy with signed bias, and a paired t-test for model
  comparison.
- `latencynet.compare` — the three-way held-out comparison with a winner rule
  and significance test fixed before any result was looked at.
- CLI `python -m latencynet` with `analytic`, `sample`, `tail` and `compare`
  subcommands.
- 164 tests including twelve Hypothesis property tests for the algebraic
  identities; five validation scripts with their committed raw output and
  JSON; four examples, each producing a PNG in `screenshots/`.
- `MODEL_CARD.md` and `DATASET_CARD.md`.

### Findings recorded in this release

- The sum-of-stages moment identities are exact to **0.000e+00** relative
  error against a declared 3.553e-15 tolerance, in both dependence regimes, on
  200,000 injected draws.
- The Fenton-Wilkinson moment-to-quantile map **under-predicts p99.9 by
  2.36 % to 4.35 %** depending on pipeline shape. Under-prediction is the
  unsafe direction for a deadline.
- Tail-quantile RMSE converges at the order-statistic rate: fitted log-log
  slopes of -0.49583 ± 0.00971 (p99) and -0.52648 ± 0.01386 (p99.9) against a
  predicted -0.5. Below the theory's own validity condition `n(1-p) >= 5` the
  rate breaks down, measured at a ratio of 0.7102 at 0.2 expected
  exceedances.
- **The learned model does not beat the analytic baseline where stages are
  independent** (0.05167 against 0.03655 at p99, paired p = 0.0009 against
  it), and **does beat it where they are correlated** (0.05094 against
  0.15441, p < 0.0001). Both halves of the specification's stated expectation
  hold.
- **The linear-regression baseline wins all four comparisons**, and the
  learned model loses to it in 12 of 12 hyperparameter settings. On this
  dataset a learned tail predictor is not justified.
- **Every native prediction interval under-covers**: the analytic model's on
  12 of 12 rows, down to a measured 0.093 against a nominal 0.90; the learned
  model's on 4 of 4. The split-conformal interval holds its one-sided
  guarantee on 47 of 48 rows.
- **The P033 EdgeInfer cross-check DISAGREES on p50 by -5.8753 %**, outside
  the ±3 % band. The two implementations agree exactly (0.000e+00) on the
  injected cost model; the gap is accounted for to within +0.4174 % by P033's
  own published measurement-overhead ratio, because P033's p50 is a measured
  wall-clock median and this one is a sampled distributional median. The
  implementation was not changed to match.

### Known limitations in this release

- No measured latency anywhere in the validated path. The realism of the
  lognormal marginals and the Gaussian-copula dependence is a modelling choice
  and is not validated against hardware.
- Serial pipelines only: no overlap, queueing, preemption or contention.
- The analytic model's native interval covers probe sampling uncertainty and
  nothing else, so it is over-confident by construction. Use
  `ConformalPredictor`.
- The conformal guarantee is marginal, not conditional, and assumes
  exchangeability between calibration and test pipelines.
- Dependence is one-parameter equicorrelation; only positive values are
  generated in the correlated population.
- Stage marginals are unimodal, so a cache-hit/cache-miss stage cannot be
  represented.
- Reference targets carry 0.30 % (p99) and 0.88 % (p99.9) Monte Carlo relative
  standard error; differences below that are not resolvable.
- `latencynet.units` is scalar-only by design.
