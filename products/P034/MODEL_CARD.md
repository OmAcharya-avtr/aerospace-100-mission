# Model card — faultinject campaign prioritiser

**Model:** `faultinject.prioritizer.CampaignPrioritizer` · **Version:** 0.1.0 ·
**Date:** 2026-10-05 · **Status: TESTING**

**This model is not certified for operational flight use.**

## Headline result, stated first

On the benchmark in `validation/validate_benchmark.py`, at a budget of 60
executions out of a 496-case pool, averaged over 24 runs (3 pools × 8 strategy
seeds):

| Strategy | Mean severe cases found | 95 % bootstrap interval |
|---|---|---|
| uniform random (baseline 1) | 16.583 | [15.208, 17.958] |
| coverage greedy (baseline 2) | 16.708 | [15.458, 18.000] |
| **kind-mean ablation, not learned** | **34.292** | [32.792, 35.750] |
| **learned (this model)** | **33.875** | [32.083, 35.583] |

Applying the decision rule — if the baseline's interval contains the other
strategy's mean score, the result is "no measurable advantage":

- **learned vs uniform random: measurable advantage.** 33.875 is far above the
  random interval's upper end of 17.958. Paired difference 17.292, interval
  [15.917, 18.625], excluding zero.
- **learned vs the kind-mean ablation: no measurable advantage.** 33.875 lies
  inside the ablation's interval [32.792, 35.750], and the ablation's mean lies
  inside the model's. The ablation is four lines of Python with no model in it.
- **coverage greedy vs uniform random: no measurable advantage** at finding
  severe faults. Paired difference 0.125, interval [−0.292, 0.543], containing
  zero. Coverage-greedy does reach more cells (0.2419 against 0.2266); that is a
  different objective.

**What that means.** The learned prioritiser roughly doubles the severe-fault
yield of random search on the same budget. Essentially all of that gain is
reproduced by ranking untried cases on the running mean severity of their fault
*kind*, which is knowledge a campaign engineer also has after twenty executions.
The model adds nothing measurable beyond it on this benchmark. The result is not
retuned, and the ablation is kept in the benchmark and in the figures.

## Problem

Given a campaign budget of `B` executions over a fault space of `N >> B`
candidate cases, order the untried cases so that the severe ones are executed
first. Severity is defined in `faultinject.severity` and a case is severe at
severity ≥ 0.6.

## Baselines, implemented first

1. **Uniform random search** without replacement. The reference: a method that
   cannot beat it should not ship.
2. **Coverage-greedy search**: always execute a case whose coverage cell is
   still uncovered, ties broken at random, falling back to uniform random once
   every reachable cell is covered.
3. **Kind-mean ablation** (not required by the specification, added for
   honesty): after the same warm-up, rank untried cases by the running mean
   severity of their fault kind.

All three were written and benchmarked before the learned model existed, and all
three remain in the benchmark and in `screenshots/search_comparison.png`.

## Architecture

`sklearn.ensemble.RandomForestRegressor`, 60 trees, `min_samples_leaf=2`,
`max_features=0.6`, `n_jobs=1`, fixed `random_state` drawn from the campaign
seed. A forest was chosen because it needs no feature scaling for a vector that
mixes one-hot indicators with normalised parameters, it fits in milliseconds on
the few dozen samples a warm-up provides, and the spread of its member trees
gives a per-case uncertainty without a second model. PyTorch is not available in
the build container, so no neural alternative was attempted; this is a
constraint, not a finding.

**Features (25).** 16 one-hot fault kind, 4 one-hot channel (`pos`, `vel`, `u`,
`bus`), start step as a fraction of the run, duration as a fraction of the run,
and three parameter slots each normalised to [0, 1] on that parameter's own
linear or log scale (unused slots are 0.0). Layout from
`faultinject.prioritizer.feature_names()`.

The one-hot kind block means most of the available signal is "which kind of
fault is this". That is exactly why the kind-mean ablation exists.

## Uncertainty output

`CampaignPrioritizer.predict` returns, per case, the mean over trees, the
standard deviation over trees, and `mean ± 1.959963984540054 · std` clipped to
[0, 1]. The acquisition used by the search is `mean + kappa · std`; the
benchmark uses `kappa = 0` (pure exploitation by predicted severity), which is
what the specification asks the learned strategy to do.

**Measured, not assumed:** empirical coverage of the nominal 95 % interval is
**0.947581** at a mean interval width of **0.830661**, on a severity range of
[0, 1]. The interval reaches its nominal coverage by being nearly
uninformative — it spans 83 % of the possible range. This is ensemble
*disagreement*, a proxy for epistemic uncertainty, not a calibrated predictive
interval. Use it to rank candidates for exploration; do not read it as a
probability.

## Dataset

See [DATASET_CARD.md](DATASET_CARD.md). In short: the training set is the set of
cases the campaign has already executed, with their observed severities. It is
generated inside the campaign, deterministically from the pool seed and the
strategy seed, and is never larger than the budget.

## Training procedure

Warm-up of 32 cases drawn uniformly at random (counted against the budget), then
fit; thereafter execute the highest-acquisition untried case and refit every 8
executions. The ranking is recomputed only when the model changes, which is
mathematically identical to recomputing the argmax at every pick.

Training cost per fit: about 10 ms on one core for a forest of 60 trees on ≤ 60
samples of 25 features.

## Test-split strategy

There is no fixed train/test split, because the setting is sequential
decision-making rather than batch supervised learning. The honest evaluation is
the campaign itself: the model is scored on the severity of the cases it chose
to execute, against two baselines spending the same budget on the same pool,
with the ground-truth severity of every pool case known in advance.

The interval-coverage figure above is measured on the **whole pool**, which
includes the ≤ 60 cases the model trained on — about 12 % of the pool. That
inflates the coverage slightly and the number should be read as an upper bound.

## Metrics

Primary: severe cases found per campaign budget, with percentile-bootstrap
confidence intervals over runs (5000 resamples) and a paired bootstrap on the
per-run difference against uniform random.

Secondary, reported in `validation/benchmark_output.txt`: mean severity of the
executed cases (learned 0.5764, kind-mean 0.5985, random 0.3258), final coverage
reached (learned 0.1892, coverage-greedy 0.2419, random 0.2266), interval
coverage and width.

Top impurity-based feature importances of the last fitted forest: `channel:u`
0.3415, `param0` 0.1135, `duration_frac` 0.1059, `start_frac` 0.0786,
`kind:numerical_nan` 0.0783, `kind:numerical_overflow` 0.0687.

## Failure cases

- **Against a simple kind-level heuristic, no measurable advantage.** Headline
  result above.
- **It trades coverage for severity.** The learned strategy reaches the lowest
  cell coverage of the four (0.1892 after 60 executions, against 0.2419 for
  coverage-greedy): it revisits the regions it believes are severe. If the
  campaign's objective is coverage, this is the wrong strategy.
- **Cold start.** With fewer than two executed cases `fit` raises; below roughly
  16 the ranking is dominated by whichever kinds happened to appear in the
  warm-up. The warm-up is a budget cost, not free.
- **The uncertainty is wide.** See above.
- **Extrapolation to unseen kinds is a mean.** A forest predicts the training
  mean for a fault kind it has never seen, so an untouched kind is ranked
  neither high nor low, just average. There is no mechanism that makes the model
  curious about unseen kinds unless `kappa > 0` is set.
- **Transfer is untested.** The model has only ever been fitted and evaluated on
  the one reference target. Nothing here says it would help on a different loop.

## Reproducibility

```bash
cd products/P034
python validation/validate_benchmark.py          # the full table above
python -m faultinject benchmark --budget 60 --pools 3 --seeds 8 --warmup 32
python -m faultinject campaign --strategy learned --budget 60 --warmup 32 \
    --pool-seed 1 --replicates 2 --seed 1001
```

Seeds: pools 1, 2, 3; strategy seeds `1000·(pool index) + s` for `s` in 0..7;
the forest's `random_state` is drawn from the strategy generator, so a campaign
is reproducible from its single seed. `numpy.random.default_rng` with PCG64 is
used throughout, with `[seed, 1]` for plant and measurement noise and
`[seed, 2]` for handler draws.

## Compute used

Whole benchmark, including the 1488 target executions that build the three
oracle tables and the 96 strategy runs: **26.7 s on one shared CPU core**. No
GPU. Peak memory well under 200 MB. Every number in this card came from that
run.

## Ethical and safety limits

- This model ranks synthetic fault-injection cases against a simulated control
  loop. It makes no prediction about a real vehicle, a real failure rate or a
  real hazard.
- **A high severity score is not a safety finding and a low one is not a
  clearance.** The score is a weighted combination of observed deviations with
  weights chosen by the author.
- **Using the prioritiser narrows coverage.** A campaign that only executes what
  the model ranks highly will systematically under-sample the fault space; the
  measured coverage drop is in the metrics above. If a campaign is evidence for
  anything, run the coverage-greedy baseline too.
- No personal data is involved at any point. All data is synthetic and generated
  by committed scripts.
- **Not certified for operational flight use.**

## Credits

MIT, © 2026 OPTIMA Organisation. The credits line for this product is in
README.md.
