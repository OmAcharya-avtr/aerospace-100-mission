# Model card — deadlinerisk contention-aware miss-rate predictor

Validation level: **Level 3, hardware-pending.** The mission coordinator decides
the level at Phase 9 against the measured evidence.

**This model is not certified for operational flight use.**

> **No measured campaign is present in this repository, so this model has not
> been fitted to any measured data and no metric in this card is a measured
> number.** `data/traces/` is empty. Every results slot carries the literal token
> `MEASURED_PENDING` and names the script that fills it. The architecture,
> baselines, split strategy, uncertainty definition and failure cases below are
> design decisions fixed before the data, and `tests/` exercises all of them
> against the committed deterministic synthetic generator, whose figures measure
> no board.

## Problem

Predict the deadline-miss rate of a periodic hard-real-time loop over a short
window, from **co-resident workload descriptors an operator could observe
without running the task**: the number of CPU and memory stressors, the measured
foreign CPU occupancy, our own and the stressors' occupancy, the load average,
the mean and minimum core clock, and the die temperature.

The point of restricting the inputs that way is that a predictor needing the
task's own response times is not a predictor: it is a measurement. These
descriptors are all available from `/proc` and `/sys` before the loop is
deployed.

## Baseline, implemented and validated first

Three baselines exist and all three were implemented and checked **before** the
learned component was written. The mission's build guide requires that order
because an under-strength baseline inflates an apparent learned win. With
`data/traces/` empty they have been checked only against the committed
deterministic synthetic generator, whose figures measure no board; checking them
against the measured traces is part of what
`validation/validate_measured_evt.py` and `validate_measured_campaign.py` do
when the campaign lands.

1. **Mean-plus-margin (Gaussian).** `P(resp > D)` under a Gaussian with the
   sample mean and standard deviation of the measured response times, plus the
   `k` for which `mean + k*sd = D`. This is the rule the product exists to
   measure the error of, not a candidate predictor.
2. **Generalised Pareto, peaks over threshold.** `zeta * (1 + xi*(D-u)/beta)
   ** (-1/xi)` with `zeta` the measured exceedance rate at threshold `u`, and
   the threshold chosen by a declared stability-window rule over 20 candidates.
3. **Counted rate** over all releases, with a moving-block bootstrap interval.
   This is the ground truth the learned component is scored against.

For the **forecasting** task the learned component is compared against two
predictors on the same splits:

- the **classical speed-scaling** predictor, `available_fraction = min(1, m/(n+1))`
  with `m = 6` cores and `n` co-resident workers, applied to the idle run's
  execution-time distribution. Fitted to nothing. The textbook
  processor-sharing capacity argument (Kleinrock 1976, "Queueing Systems,
  Volume 2", chapter 4; Funk, Goossens & Baruah 2001, RTSS, for the uniform
  multiprocessor form). Its assumptions — ideal fair sharing, no cache or
  bandwidth interference, no frequency scaling, no scheduling latency — are all
  false on a real SoC in the direction that makes it optimistic.
- the **constant-rate null model**, the training mean. A learned model that
  cannot beat this has learned nothing from the descriptors.

## Architecture

`sklearn.ensemble.RandomForestRegressor`, `n_estimators=300`, `max_depth=6`,
`random_state=920092`, `n_jobs=1`.

A forest rather than anything larger for three reasons that are hard constraints
and not preferences:

- the dataset is a few hundred windows from one campaign, so a model with many
  parameters would memorise windows;
- the build environment has two cores and a three-minute budget for any fit;
- a forest gives a spread across trees that is an honest, if conservative,
  uncertainty output without a second model.

`max_depth=6` caps the tree at 64 leaves against a few hundred rows, which is
the loosest cap that still forces leaves to average several windows.

## Dataset

Source: the hardware campaign that `data/campaign.py` writes into
`data/traces/`, on an NVIDIA Jetson Orin Nano Engineering Reference Developer Kit
Super. See `DATASET_CARD.md`, which records that the directory is currently
empty.

Construction: each run is tiled into 10 s windows. A window contributes the
descriptors above, averaged over its 1 Hz monitor samples, and its own counted
miss rate as the target, with releases and misses summed over the executed rows
whose **release** falls inside the window, so a dropped release is attributed to
the window of the release that was dropped. A window with fewer than 100
releases or with no monitor sample is discarded and the discard count is
reported.

MEASURED_PENDING — the hardware campaign had not written any file into `data/traces/` when this document was generated, so there is no measured number to put here and none was invented. This slot is filled by `validation/validate_measured_learned.py`, which refuses with a non-zero exit until `data/traces/CAMPAIGN_COMPLETE` exists.

## Training procedure

One fit per split. No hyperparameter search, no early stopping, no feature
selection, and no retuning after seeing a score — the mission's build guide
forbids retuning to make the model win, and the numbers below were not retuned.

MEASURED_PENDING — the hardware campaign had not written any file into `data/traces/` when this document was generated, so there is no measured number to put here and none was invented. This slot is filled by `validation/validate_measured_learned.py`, which refuses with a non-zero exit until `data/traces/CAMPAIGN_COMPLETE` exists.

## Metrics and test-split strategy

Scored **separately** on two splits, because they answer different questions.

- **Held-out workload mixes**, leave one run out. No window of a held-out run is
  ever in training, so the score measures generalisation to a contention mix the
  model has never seen. This is the harder and the more relevant split: an
  operator's co-resident load is not one of the ten this campaign ran.
- **Held-out time**, the last 30 per cent of every run's windows. Training sees
  only the earlier portion of each run, so the score measures forward prediction
  within mixes the model has seen.

Metrics: mean absolute error, root mean squared error and bias on the window
miss rate, all dimensionless, plus a one-sigma coverage figure for the predictors
that report an uncertainty.

## Results

MEASURED_PENDING — the hardware campaign had not written any file into `data/traces/` when this document was generated, so there is no measured number to put here and none was invented. This slot is filled by `validation/validate_measured_learned.py`, which refuses with a non-zero exit until `data/traces/CAMPAIGN_COMPLETE` exists.

## Uncertainty output

`ContentionMissPredictor.predict` returns a point estimate and a per-window
standard deviation with two named components:

- `sigma_model` — the standard deviation of the predictions across the forest's
  trees. A statement about the ensemble's internal disagreement.
- `sigma_target` — the binomial standard deviation of the window's own counted
  rate, `sqrt(p(1-p)/N)` at the predicted rate over the window's release count.
  This is the irreducible noise in the **target**: a window's counted miss rate
  is itself an estimate.

The reported `sigma` is the two combined in quadrature. It is a **lower bound on
the predictive uncertainty**: it does not include model misspecification. Its
calibration is reported as a measured coverage figure — the fraction of held-out
windows whose true rate lies within one reported sigma — rather than assumed.

MEASURED_PENDING — the hardware campaign had not written any file into `data/traces/` when this document was generated, so there is no measured number to put here and none was invented. This slot is filled by `validation/validate_measured_learned.py`, which refuses with a non-zero exit until `data/traces/CAMPAIGN_COMPLETE` exists.

## Failure cases

1. **Extrapolation.** A random forest averages leaf means, so its prediction
   cannot leave the range of its training targets. At a contention level more
   severe than anything in training it predicts the most severe rate it saw, and
   the reported sigma does not grow to signal that. `tests/test_predictor.py`
   asserts this confinement rather than treating it as a surprise.
2. **Collinear descriptors.** On this campaign the stressor count, the stressor
   occupancy and the load average move together by construction, because the
   stressors were the thing varied. The reported feature importances are a
   statement about this forest on this dataset and are not causal.
3. **A held-out mix with no neighbour in training.** The leave-one-run-out split
   removes an entire contention level, and the two extreme levels have a
   neighbour on one side only. Scores on those runs are the ones to read first.
4. **Windows with no misses.** A window whose counted rate is exactly zero is
   uninformative about the tail, and a campaign in which most windows are zero
   gives a forest whose best strategy is to predict near zero. The per-run
   target ranges are reported so a reader can see how much of the dataset is in
   that state.
5. **Nothing varies with ambient, board, payload or period**, because none of
   them varied. A model that generalises across mixes and across time on this
   campaign has been shown nothing about generalising across any of those.

MEASURED_PENDING — the hardware campaign had not written any file into `data/traces/` when this document was generated, so there is no measured number to put here and none was invented. This slot is filled by `validation/validate_measured_learned.py`, which refuses with a non-zero exit until `data/traces/CAMPAIGN_COMPLETE` exists.

## Reproducibility

Exact commands, from a cold clone:

```bash
pip install -e ".[dev]"
python validation/validate_measured_learned.py
python examples/learned_vs_classical.py
python -m deadlinerisk --traces data/traces learned
```

Seeds: the forest's `random_state` is `920092`; the analysis seed for every
bootstrap is `920092`; the synthetic fixture seed is `920092`; the negative
controls use `5150`. All are declared in code — `deadlinerisk.predictor.RANDOM_STATE`,
`validation/_bootstrap.SEED_ANALYSIS`, `SEED_FIXTURE` and `SEED_CONTROL` — so no
two reported numbers rest on different ones.

The dataset is rebuilt from the committed traces on every run. **No serialised
model is committed**: a pickled estimator would bind this repository to one
scikit-learn build, whose unpickling is not part of that library's public API
contract, and the fit takes under a second.

## Compute

MEASURED_PENDING — the hardware campaign had not written any file into `data/traces/` when this document was generated, so there is no measured number to put here and none was invented. This slot is filled by `validation/validate_measured_learned.py`, which refuses with a non-zero exit until `data/traces/CAMPAIGN_COMPLETE` exists.

Budget: the mission's build guide allows three minutes for any training or Monte
Carlo run on two cores. The dataset and model are sized for that, and
`validation/validate_measured_learned.py` section 2 times a fit and checks it
against the 180 s budget rather than asserting it is inside one. On the synthetic
campaign of comparable size, `tests/` fits the same forest repeatedly inside a
suite that completes in about thirteen seconds in total.

## Ethical and safety limits

- **This model is not certified for operational flight use.**
- It is research-grade: not flight-qualified, not certified, not approved for
  operational aerospace use.
- It predicts a **rate**, not a bound. No output of this model is an upper bound
  on a miss probability, and no safety argument may treat it as one.
- Its validity is confined to the scope in `README.md`: one board, one power
  mode, one uninstrumented ambient, one CPython CPU-only payload, one period,
  and the ten contention mixes the campaign runs. The board is shared and the
  measured foreign occupancy is reported wherever a figure is reported, rather
  than assumed absent.
- If a baseline beats the learned component on either split, that is the finding
  and it is published in this card, in `README.md` and in
  `validation/VALIDATION.md`. Nothing is retuned to change it.
