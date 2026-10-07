# Model card — `simplexguard` learned switch predictor

**This model is not certified for operational flight use.** Research-grade,
validation level 3. It is shipped as the losing side of a benchmark: the
measurement it exists to support is that an exact closed-form inequality beats
it on the inequality's own criterion, on both accuracy and latency. It is not a
deployable component and the README says so before it says anything else about
it.

## Problem

Given the state of a single-axis attitude loop and the reference it is tracking,
predict whether a Simplex runtime guard will hand control to the conservative
baseline controller. Two targets, because the honest answer differs between
them.

**Task A, lead 0.** Does the guard fire at this step. This is the switching
condition's own criterion: a closed-form inequality
`c_j^T (A x + B u) + h_W(c_j) <= d_j` over the 50 facets of the robust invariant
set. The exact computation is right by construction, so a classifier can at best
tie, and the only way it could earn a place is by being cheaper.

**Task B, lead 5.** Does the guard fire at any step in the next five. The exact
answer here is a worst-case reachability computation that answers "may fire
under *some* admissible disturbance sequence", which over-predicts a realised
episode by construction, so a learned model fitted to the realised label is
answering an easier question and has room to win.

Base rate: **0.081687** for Task A over the whole dataset, **0.083333** on the
test split; **0.148201** and **0.149477** for Task B.

## Baselines, implemented and validated before the learned model

| Baseline | What it is | Uses labels? |
|---|---|---|
| **Exact one-step guard condition** | the switching condition itself, evaluated on the saturated performance input | **no** |
| **Exact worst-case reachability, L = 5** | unrolls the unsaturated linear performance loop `L` steps and maximises each disturbance term over `W` by support function; answers "may fire within L" | **no** |
| **Exact nominal reachability, L = 5** | the same without the disturbance sum; answers "fires within L with no disturbance" | **no** |
| Persistence | fires now implies will fire within L | no |
| Always-negative at the training base rate | the degenerate forecaster, included because on a 8 %-base-rate problem accuracy is near-meaningless and an always-negative forecaster scores 0.91667 | yes, one number |
| L2 logistic regression | standardised, isotonic-calibrated; a measured lower bound on what a learned model can do against a 50-facet intersection | yes |

The three exact baselines are in `simplexguard/guard.py` and
`simplexguard/reachability.py`, both of which `simplexguard/predictor.py`
imports. The analytic answer is a dependency of the learned model's benchmark,
not an afterthought.

## Architecture

`sklearn.ensemble.RandomForestClassifier`, 150 trees, `min_samples_leaf = 8`, no
depth limit, `random_state = 5101`, `n_jobs = 2` for fitting and **`n_jobs = 1`
for inference**, wrapped in
`sklearn.calibration.CalibratedClassifierCV(FrozenEstimator(forest),
method="isotonic")` fitted on a held-out calibration split.

No neural network: PyTorch is unavailable in the build environment, and a
five-feature tabular problem of 31200 training rows does not need one.

`n_jobs = 1` at inference is not cosmetic. Measured in this container, a
150-tree forest costs **33.1 ms** per single-row `predict_proba` at `n_jobs = 2`
and **5.8 ms** at `n_jobs = 1`: joblib's per-call dispatch dominates, and a
runtime guard decides one step at a time. The model is given its best case.

**No hyperparameter search was performed.** The forest size was not tuned; four
sizes (20, 50, 150, 300) are reported in full in
`validation/validate_predictor.py` check 5 as a latency/accuracy curve, and the
best F1 among them (0.981250 at 50 trees) is still below the exact computation's
1.000000. Tuning further would not change the conclusion, and tuning until it
did would be the thing this portfolio exists not to do.

## Features

Five, all raw measurable quantities from the step the decision is taken at:

`theta` (rad), `theta_dot` (rad/s), `reference` (rad), `tracking_error`
(rad, `theta - reference`), `u_perf_unsaturated` (rad/s^2).

**The switching-condition margin is deliberately not a feature.** A model handed
the exact computation's output is a wrapper around it, not an alternative to it,
and benchmarking one against the thing it is wrapping would be meaningless.

## Dataset and splits

See `DATASET_CARD.md`. 40 simulated guarded episodes of 1200 steps each, from
the shipped illustrative plant at seed 5101, with the reference amplitude
jittered by ±25 % per episode.

Split **by episode**, never by row: 26 train (31200 rows), 6 calibration (7200),
8 test (9600) for Task A; 31070 / 7170 / 9560 for Task B, which discards the
last `lead` rows of each episode because their label is not yet determined.
Splitting by row would leak: consecutive rows of one episode are the same
trajectory, and a shuffled split puts near-duplicates of test rows into
training.

## Metrics on the held-out episodes

### Task A, lead 0 — the condition's own criterion

| Predictor | Precision | Recall | F1 | Brier | ECE | µs / decision |
|---|---|---|---|---|---|---|
| **Exact one-step condition** | **1.00000** | **1.00000** | **1.00000** | n/a | n/a | **8.68** |
| Forest, 150 trees, isotonic | 0.97764 | 0.98375 | 0.98069 | 0.002629 | 0.00258 | 7343.16 |
| Logistic, isotonic | undefined | 0.00000 | undefined | 0.072448 | 0.02798 | 819.25 |
| Always-negative at base rate | undefined | 0.00000 | undefined | 0.076467 | 0.00885 | — |

**The learned model loses, and the loss is published without retuning.** 18
false positives and 13 false negatives in 9600 held-out steps against zero of
each, and **846x** the latency. At `dt = 0.05 s` the exact condition uses
0.0174 % of the sample interval and the forest 14.69 %.

### Task B, lead 5 — anticipation

| Predictor | Precision | Recall | F1 | Brier | µs / decision |
|---|---|---|---|---|---|
| Exact worst-case, L = 5 | 0.43477 | 0.86284 | 0.57819 | n/a | 13.84 |
| Exact nominal, L = 5 | 0.44448 | 0.86284 | 0.58672 | n/a | 18.70 |
| Persistence | 0.85875 | 0.48076 | 0.61642 | n/a | — |
| **Forest, 150 trees, isotonic** | **0.96829** | 0.83345 | **0.89583** | 0.026397 | 7123.80 |

**The learned model wins on F1 here, on a different question.** The exact
predictor over-predicts by construction. It still costs 701x more per decision.

### Lead time (287 firing events on the test split)

| Predictor | mean | median | max | fraction unanticipated |
|---|---|---|---|---|
| Exact worst-case, L = 5 | **12.861** | 8.0 | 40.0 | **0.0000** |
| Forest, lead-5 target | 6.624 | 8.0 | 16.0 | 0.0418 |
| Forest, lead-0 target | 0.115 | 0.0 | 5.0 | 0.9686 |
| Exact one-step condition | 0.000 | 0.0 | 0.0 | 1.0000 |

### Latency against model size

| trees | F1 | Brier | µs / decision |
|---|---|---|---|
| 20 | 0.978301 | 0.002615 | 1717.9 |
| 50 | 0.981250 | 0.002374 | 3247.3 |
| 150 | 0.980685 | 0.002629 | 6723.7 |
| 300 | 0.977472 | 0.002760 | 12424.5 |
| exact | **1.000000** | n/a | **8.68** |

A 20-tree forest is already 198x the exact condition's latency, and the latency
rises only 7.23x from 20 to 300 trees, so the gap is scikit-learn's per-call
overhead and not tree traversal. **No model size closes it.**

## Uncertainty and confidence output

`predict_proba` returns an isotonic-calibrated probability, fitted on a split
the base estimator never saw. Measured on Task A: Brier **0.002629**, expected
calibration error **0.002583** over 10 equal-width bins, against Brier
**0.076467** for an always-negative forecaster at the training base rate.
Reliability bins are in `validation/VALIDATION.md` section 6 and in
`screenshots/predictor_benchmark.png`.

Neither exact predictor produces a probability at all, so neither has a Brier
score or a reliability curve. **This is the only structural advantage the
learned model has in this package**, and it is not enough to recommend it for
the switching decision.

## Failure cases

- **Near the switching boundary.** The exact decision surface is the
  intersection of 50 non-axis-aligned halfspaces in `(x, u)`. An axis-aligned
  tree ensemble approximates a tilted hyperplane with staircase error, so the 31
  errors on Task A are concentrated where the margin is near zero.
- **The logistic model fails outright** on Task A: no positive predictions at a
  0.5 threshold, which is what a single hyperplane does when asked to represent
  a 50-facet intersection.
- **Latency.** 7.2 ms per single-row decision is 14 % of a 50 ms sample
  interval, on a contended two-core container. A real 20 Hz loop with other work
  to do would not have it.
- **No generalisation claim.** The model is trained on episodes of one plant,
  one controller pair, one reference shape and one disturbance sampler. It has
  not been evaluated on a different plant, a different invariant set, a
  different reference, or a disturbance outside the declared bound, and nothing
  here suggests it would transfer.
- **The labels come from the realised guarded episodes**, so the model learns
  the behaviour of this guard and not the switching condition in general.

## Reproducibility

```bash
python validation/validate_predictor.py
MPLBACKEND=Agg python examples/predictor_benchmark.py
python -m simplexguard predict --episodes 40 --steps 1200 --lead 5 --trees 150
```

Seed 5101 throughout: episode `i` draws its disturbance from
`numpy.random.default_rng(5101 + 1000 * i)`, and the forest uses
`random_state = 5101`. The dataset is regenerated deterministically by
`build_dataset`, is not committed, and takes 4.1 s for 96000 steps.

## Compute used

Two shared CPU cores and 7.8 GiB, contended with four sibling build agents.
Fitting the 150-tree forest takes **1.7 s** for Task A and **4.0 s** for Task B.
The whole benchmark script, including dataset construction, four forests, two
logistic models and all latency measurements, runs in **36.0 s**. The longest
single model fit is **4.0 s**, and nothing here approaches the three-minute
budget.

## Ethical and safety limits

- **This model is not certified for operational flight use.**
- It must not be used to make a safety decision. The exact switching condition
  is cheaper and correct, and this package exists partly to make that
  comparison hard to ignore.
- Any use of a learned component in a runtime-assurance path needs the exact
  condition to remain the arbiter. A learned predictor can at most schedule,
  warn or pre-position; it cannot be the guard.
- The training data is simulated from a declared linear model. It contains no
  measurements of any physical system and no personal data.
