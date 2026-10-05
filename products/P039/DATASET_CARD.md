# Dataset card — latencynet synthetic pipeline populations

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use.

## What this dataset is, and what it is not

Every number this product validates against comes from a **generated**
distribution with declared parameters and a fixed seed. **There is no measured
latency anywhere in the training, calibration or evaluation path.** No real
embedded pipeline, no real profile, no real hardware.

That is a deliberate choice and the reason is specific. The only host
available to this build is a shared single-CPU-core container running five
concurrent build jobs, where wall-clock timings of a 270 us pipeline move by
factors of 2 to 13 between runs; P033 EdgeInfer measured its own mean on the
same host at 2.19 times the injected value; this product reproduced the
effect independently and measured its own mean at 2.23 times the injected
value on one run of `validation/crosscheck_edgeinfer.py` and 1.08 times on
another run of the same script, same seed, minutes later. A measured target would make every
reported model error a statement about host load rather than about a model.
With an injected target the right answer is known, so an error is an error.

The cost of that choice is the central limitation of this dataset: **its
realism is unvalidated.** Whether real stage latencies are lognormal, whether
their dependence is Gaussian-copula equicorrelated, and whether end-to-end
latency really is the sum of stage latencies with no overlap are modelling
choices, not findings. Accuracy measured against this generator transfers to a
real pipeline only to the extent that those choices are right, and nothing
here establishes that they are.

## Generator

`latencynet.dataset.build_dataset` and `latencynet.pipeline`. Committed,
deterministic, re-runnable:

```bash
PYTHONPATH=src python3 -c "
from latencynet.dataset import build_dataset
ds = build_dataset('correlated', n_train=200, n_calibration=60, n_test=150,
                   seed=20260402, n_probe=256,
                   n_reference=25_000, n_reference_test=90_000)
print(ds.n_total, ds.regime)"
```

No data file is committed. At the validation sizes the reference samples total
of order 2 x 10^7 float64 latencies per regime, which is 160 MB and well over
the 1 MB commit threshold, so regeneration from the seed is the distribution
mechanism. Regeneration is bit-exact: the seed fully determines the pipeline
parameters, the probe trace and the reference sample.

## Per-pipeline generation

For pipeline `i`, seeded at `20260402 + 10i`:

| quantity | distribution | range |
|---|---|---|
| number of stages `K` | uniform integer | 2 to 6 inclusive |
| stage mean | log-uniform | 20 us to 500 us |
| stage coefficient of variation | uniform | 0.05 to 0.60 |
| stage standard deviation | `mean × cv` | derived |
| latent equicorrelation `rho` | 0 in the independent regime; uniform in the correlated regime | 0, or 0.35 to 0.90 |

Stage marginals are lognormal, parameterised from the declared mean and
standard deviation by `sigma^2 = ln(1 + cv^2)`, `mu = ln(mean) - sigma^2/2`
(Johnson, Kotz & Balakrishnan 1994, *Continuous Univariate Distributions*
Vol. 1, 2nd ed., Ch. 14). Dependence is injected with a Gaussian copula on the
latent normals, equicorrelated at `rho`, which for lognormal marginals gives
the closed-form covariance
`Cov(X_i, X_j) = m_i m_j (exp(rho sigma_i sigma_j) - 1)` (same source, Ch. 14
Sec. 4) — so the dependence is not only injected but analytically known.

Why lognormal: positive, right-skewed, and closed under products of
independent multiplicative effects, which is the usual argument for software
latency. It is a modelling convention, widely used, and not a law.

Why these ranges: a 20 to 500 us stage spans the plausible range for a
quantised inference kernel on an embedded target, and a coefficient of
variation of 0.05 to 0.60 spans "tightly bounded compute" to "noticeably
jittery". The correlated regime starts at 0.35 so that the independence
assumption is clearly, not marginally, wrong there.

Observed in the generated populations (`validate_model_comparison_output.txt`):

| regime | pipelines | latent rho: min / mean / max | stages: min / mean / max |
|---|---|---|---|
| independent | 410 | 0.000 / 0.000 / 0.000 | 2 / 3.94 / 6 |
| correlated | 410 | 0.354 / 0.623 / 0.900 | 2 / 3.94 / 6 |

## What each record contains

`latencynet.dataset.PipelineRecord`:

| field | what it is | who may see it |
|---|---|---|
| declared stage means, sds, `latent_rho` | the injected truth | nobody — diagnostics only |
| probe summary: per-stage sample mean, sd, fourth central moment, covariance matrix, p99, and the 13 features | a **256-pass aligned** per-stage trace, seeded at `seed + 1` | every model; the only model input |
| reference mean and quantiles at p99 and p99.9, with their distribution-free one-sigma uncertainties | a **25,000-pass (train, calibration) or 90,000-pass (test)** end-to-end sample, seeded at `seed + 2` | the target and the scorer only |

The probe stream and the reference stream are disjoint, so the target is never
computed from the model input. The declared parameters are never a feature:
every model sees only what a profiler could measure.

**Why 256 probe passes.** That is a few milliseconds of profiling for a 270 us
pipeline — cheap enough to be realistic. It is also noisy enough to matter: a
pairwise correlation estimated from 256 samples has a standard error of about
0.06, so in the independent regime a true `rho = 0` is observed as roughly
`0.00 ± 0.06`. That noise is part of the problem, not a defect in the dataset.

**Why 90,000 reference passes for the test split.** The target itself is a
Monte Carlo estimate and carries its own error. Measured on the test splits:

| target | mean relative Monte Carlo standard error of the target |
|---|---|
| p99 | 0.304 % (independent), 0.374 % (correlated) |
| p99.9 | 0.876 % (independent), 1.001 % (correlated) |

Those figures are printed by the comparison script and recorded in
`model_comparison.json`. **Model error differences below them are not
resolvable with this reference sample size**, and the comparison says so
rather than reading a winner out of the noise. The test split gets 3.6 times
the reference passes of the train split specifically so that target noise is
smaller where it would otherwise be mistaken for model error. The
uncertainties are distribution-free, from the binomial rank interval of the
order statistic (David & Nagaraja 2003, *Order Statistics*, 3rd ed., Sec. 7.1).

## Target

`ln q_p` for `p` in `{0.99, 0.999}`, in log-seconds. Log space because
end-to-end latency spans two orders of magnitude across the population and the
engineering question is relative.

## Splits

Split **by pipeline**: 200 train / 60 calibration / 150 test, assigned by
generation index so the split is reproducible from the seed alone. A held-out
pipeline is one whose parameters, probe trace and reference sample no model has
seen. No pass from a test pipeline appears anywhere in fitting or calibration.

The calibration split exists only to set the split-conformal radius, and its
size fixes the achievable coverage resolution at `1/(m+1) = 0.0164`.

Smaller sizes are used inside the test suite (32/14/14 pipelines, 3,000
reference passes) purely so the suite runs in seconds; no statistical claim is
made from those.

## Compute to regenerate

| population | reference draws | CPU time on one core |
|---|---|---|
| one regime at validation sizes | about 9 x 10^7 lognormal draws | about 4.5 s |
| both regimes | about 1.8 x 10^8 | about 9 s |
| the 4,000,000-pass convergence reference | 1.2 x 10^7 | about 0.6 s |

Peak memory under 400 MB. No GPU.

## Known limitations of this dataset

1. **No real measurement.** The realism of the lognormal marginals, the
   Gaussian-copula dependence and the serial sum is assumed, not validated.
   This is the limitation that matters most and no result here can fix it.
2. **One-parameter dependence.** Equicorrelated latent normals only. Real
   pipelines plausibly have structured dependence — adjacent stages correlated
   through a shared cache or thermal state, distant ones not — which this
   generator cannot produce.
3. **Positive correlation only in the correlated regime.** `rho` is drawn from
   [0.35, 0.90]. Negative stage correlation is supported by
   `latencynet.pipeline` and is exercised by a unit test, but no population is
   generated with it and no model is evaluated on it.
4. **Unimodal stage marginals.** No cache-hit/cache-miss bimodality, no
   page-fault outlier mode, no cold-start mode. Real latency traces frequently
   have all three.
5. **No non-stationarity.** Stage parameters are fixed per pipeline. No
   thermal drift, no frequency scaling, no interference from co-resident work.
6. **No queueing and no overlap.** End-to-end latency is the sum of stage
   latencies. A pipelined or concurrent implementation is outside the model.
7. **No censoring and no outliers of the kind a real profiler sees.** Nothing
   times out, nothing is dropped, nothing is preempted — which is exactly what
   dominated the one measured figure this product does report.
8. **Stage count capped at 6.** Deeper pipelines are untested, and the
   independence assumption degrades differently as `K` grows because the number
   of covariance terms grows as `K(K-1)/2`.
9. **The target has Monte Carlo noise**, quantified above: 0.30 % at p99 and
   0.88 % at p99.9 on the test split. It is reported, not removed.

## Licence and provenance

Entirely synthetic, generated by code in this repository. No third-party data,
no scraped data, no personal data, no human subjects. Licensed with the
package under Apache-2.0, copyright © 2026 OPTIMA Organisation.
