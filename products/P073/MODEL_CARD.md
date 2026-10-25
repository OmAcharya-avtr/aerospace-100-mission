# Model card: the contactsched conflict predictor

## Problem

Binary classification over ground-station contact requests. Positive means the
request will be **starved**: the proven optimum over the candidate grid, solved
with no fairness constraints, delivers fewer contacts than the request declared
it needs.

The purpose is renegotiation, not scheduling. A planner told at submission time
that a request is about to be dropped can shorten the contact, widen the allowed
antennas or lower the minimum count. A planner told after the schedule is
published can only complain.

Every feature is computable **before** the assignment is solved. A predictor
whose inputs include the schedule is not a predictor.

## Baseline, implemented and measured first

Three contention ratios are available before scheduling, defined in
`contactsched.contention`:

| Feature | Definition |
| --- | --- |
| `ratio_max` | largest apportioned-demand-over-union-supply ratio over the antenna-time bins the request touches |
| `ratio_weighted` | the same ratios averaged, weighted by the request's own apportioned demand in each bin |
| `demand_over_usable` | the request's declared requirement divided by the total length of windows long enough to hold one of its contacts |

Each is used two ways: a **ranking** form with no fitting at all, and a
one-feature **logistic regression** fitted on the training instances. A monotone
one-feature fit cannot change a ranking, so the two agree exactly on rank
statistics and differ only on calibration; that identity is asserted in
`tests/test_predictors.py` rather than assumed.

Which of the three is the baseline was decided by measurement. The product
specification named a demand-over-supply ratio *per station-window*; measured on
725 held-out requests it is the weakest of the three, at 0.6010 and 0.6228 area
under the ROC curve against 0.9230 for the per-request ratio, and on a smaller
108-row split both per-bin ratios fall below 0.5. Taking the specified ratio as
the baseline would have inflated the learned model's margin, so
`demand_over_usable` is the default baseline feature and all three are reported.

## Architecture

`sklearn.ensemble.HistGradientBoostingClassifier` over 23 features.

| Hyper-parameter | Value |
| --- | --- |
| `max_iter` | 200 |
| `learning_rate` | 0.06 |
| `max_leaf_nodes` | 15 |
| `min_samples_leaf` | 25 |
| `l2_regularization` | 1.0 |
| `early_stopping` | False |
| `random_state` | 0 |

No neural network is used; PyTorch is not available in the environment this
product was built in, and the problem does not need one at this dataset size.

## Dataset

Fully synthetic, from `contactsched.dataset.generate_instance`. See
`DATASET_CARD.md` for the generator's distributional choices, which are the
single largest limitation of every number below.

| Property | Value |
| --- | --- |
| Train instances | 240, seeds 0 to 239 |
| Test instances | 80, seeds 240 to 319 |
| Train rows | 2178, positive rate 0.312672 |
| Test rows | 725, positive rate 0.295172 |
| Features | 23 |
| Horizon | 43200 s |
| Label | the exact optimum delivers fewer than `min_contacts` |

**Split strategy.** By instance, never by request. Requests inside one instance
share its network, its horizon and its contention, so a per-request split would
leak and every score would be optimistic.
`contactsched.dataset.split_by_instance` raises on overlapping seed ranges.

**Dataset limitations.** The generator draws 3 to 4 antennas from a pool of 6,
4 to 6 spacecraft with mean motions in 13.9 to 15.5 rev/day, and 7 to 11
requests with priorities drawn 0.2/0.3/0.3/0.2 over 1 to 4. A real request mix
differs in every one of those respects. Nothing here is a statement about a real
ground-station network.

## Training procedure

One fit, no cross-validation, no hyper-parameter search on the test set. The
decision threshold is chosen by maximising F1 on the **training** rows and
applied unchanged to the test rows; the measured value is 0.461.

Reproduce with:

```bash
python validation/validate_predictors.py
```

## Metrics

Measured on 725 held-out requests from 80 instances in a disjoint seed range.
The learned model is compared against the best baseline **on each metric
separately**, because on this test set the ranking and logistic forms of
`demand_over_usable` tie on area under the ROC curve while the ranking form has
the lower Brier score. Nominating one baseline would have handed the learned
model an easier comparison than the baselines actually offer.

| Metric | Best baseline | Baseline | Learned | Margin |
| --- | --- | --- | --- | --- |
| area under the ROC curve | `ranking[demand_over_usable]` | 0.922952 | 0.984596 | +0.061644 |
| average precision | `ranking[demand_over_usable]` | 0.885048 | 0.969138 | +0.084090 |
| Brier score | `ranking[demand_over_usable]` | 0.098911 | 0.046725 | +0.052185 |
| **expected calibration error, 10 bins** | **`logistic[demand_over_usable]`** | **0.012237** | **0.027302** | **-0.015065** |
| F1 at the training-chosen threshold | `ranking[demand_over_usable]` | 0.733728 | 0.886256 | +0.152528 |
| area under the ROC curve, hard subset | `ranking[ratio_weighted]` | 0.838021 | 0.952907 | +0.114887 |

Margin is signed so that positive always means the learned model is better.

A constant forecast at the training base rate scores 0.5000 area under the ROC
curve and a Brier score of 0.2084, which is the floor any probability forecast
has to beat to be a forecast at all.

**The baseline wins on calibration.** That result stands. The learned model was
not retuned, re-split or removed to change it.

### Hard and easy subsets

604 of 2903 rows have a positive `capacity_deficit`: the request declared more
contacts than its own passes can carry, so it is unsatisfiable whatever the
competition does, and one feature says so directly. Those rows are a real part
of the label and are kept, but every score is reported both over all requests
and over the 581 test rows with a non-positive deficit, because reporting only
the first would flatter every model including the baselines. The hard-subset row
in the table above is that second measurement.

## Uncertainty and confidence output

`predict_proba` is the confidence this package exposes. It is **measured, not
asserted**: `validate_predictors.py` prints a ten-bin reliability table for the
learned model and for the strongest baseline, and reduces each to an expected
calibration error. The learned model is over-confident at the extremes, which is
the documented behaviour of a boosted ensemble (Niculescu-Mizil and Caruana,
2005), and it is worse calibrated than the one-feature logistic baseline.

Guidance, from the measurement: use the learned model's probability as a
**ranking** and the baseline's as a **forecast**, until you have re-measured on
your own traffic.

## Failure cases

- **Out-of-distribution request mixes.** Skill is skill on this generator.
  Priorities, durations and contact counts outside the drawn ranges are
  untested.
- **The confidence is not calibrated at the extremes.** A probability of 0.95
  from the learned model is not a 95 per cent chance of starvation on this test
  set; see the reliability table.
- **The label depends on the candidate grid.** A request that is starved on a
  240 s grid may be satisfiable on a finer one. The label is the exact optimum
  *over the grid*.
- **The label is method-dependent in the other direction too.** A request the
  exact optimum satisfies may still be starved by a greedy scheduler. Labelling
  against the exact optimum was deliberate -- it avoids teaching the model a
  heuristic's quirks -- but it means a positive prediction is a statement about
  the best achievable schedule, not about the one your scheduler will produce.
- **Instance-level features leak instance identity.** `instance_requests` and
  `instance_demand_over_supply` are constant within an instance. With an
  instance-level split they cannot leak the label, but they do let the model
  condition on instance difficulty, which is legitimate and is also why the
  split must stay at instance level.

## Reproducibility

- Every estimator takes an explicit `random_state`; the measured value is 0.
- Refitting on the same rows with the same seed produces bit-identical
  predictions, asserted in `validate_predictors.py`.
- The dataset rebuild is byte-identical, asserted in `validate_dataset.py`.
- `scikit-learn` is bounded `>=1.9,<1.10` in `pyproject.toml` because every
  number above is a property of fitted trees rather than of a documented API. A
  minor release that changes the histogram binning or the split ordering moves
  these numbers without failing a single test.

## Compute used

Measured on the shared two-core machine this product was built on. Wall clock is
unstable at the tens-of-per-cent level and appears here in a table only.

Wall clock only, in a table declared unstable, and used in no conclusion
anywhere. Ranges are what was observed across repeated runs in one build
session; the committed `validate_predictors_output.txt` records the single run
that produced it, which is the number to read if you want one.

| Step | Range observed this session |
| --- | --- |
| Dataset build, 320 instances including one exact solve each | 22.9 to 28.6 s |
| Learned model fit | 0.4 to 45.6 s |
| Permutation importance, 23 features, 5 repeats | 1.0 to 123.2 s |

Those ranges span two orders of magnitude on identical data with identical
seeds, because the machine is shared with two other build agents and because the
first fit in a fresh process pays scikit-learn's import and warm-up. That is
precisely why no timing in this repository appears in a prose conclusion. The
*result* -- which predictions the fit produces -- is bit-identical across every
one of those runs, which `validate_predictors.py` asserts.

No GPU. No PyTorch. Peak memory is dominated by the 2903-row dataframe.

## Feature importance

Permutation importance on the held-out rows, five permutations per feature,
scored by area under the ROC curve. The top five:

| Feature | Mean drop in AUC | Std |
| --- | --- | --- |
| `capacity_deficit` | 0.277247 | 0.017019 |
| `ratio_weighted` | 0.022479 | 0.002067 |
| `higher_priority_overlap_s` | 0.008144 | 0.001289 |
| `usable_seconds` | 0.003942 | 0.000781 |
| `priority` | 0.003052 | 0.000456 |

These describe **this fitted model on this generator** and are not evidence
about ground-station scheduling. Correlated features share credit arbitrarily
(Strobl et al., 2007), and `demand_over_usable` scores near zero here precisely
because `capacity_deficit` carries the same information in a sharper form -- not
because the ratio is uninformative, which the baseline table above disproves.

## Ethical and safety limits

- All training data is synthetic and generated by a committed, seeded script. No
  flight data, no operational request log and no catalogue element set is used
  anywhere in this repository.
- The model allocates a shared resource. A predictor used to decide which
  mission to drop is a policy instrument, and the fairness accounting in
  `contactsched.fairness` exists so that the policy's effect is visible rather
  than implicit. Using the prediction to pre-emptively drop requests, without
  the accounting, would make starvation less visible rather than more.
- **This model is not certified for operational flight use.**

## References

- Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision
  tree. *NeurIPS 30*.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of
  probability. *Monthly Weather Review* 78(1), 1-3.
- Niculescu-Mizil, A. and Caruana, R. (2005). Predicting good probabilities with
  supervised learning. *ICML 22*.
- Strobl, C., Boulesteix, A.-L., Zeileis, A. and Hothorn, T. (2007). Bias in
  random forest variable importance measures. *BMC Bioinformatics* 8.
- Jain, R., Chiu, D.-M. and Hawe, W. (1984). *A Quantitative Measure of Fairness
  and Discrimination for Resource Allocation in Shared Computer Systems*, DEC
  TR-301.
