# Model card — handoverloss per-mechanism handover-loss predictor

Validation level **2**: the evidence is agreement between a closed form, an
independently derived bound and a simulated trace. Nothing in this repository is
measured on hardware or on a real optical link.

**This model is not certified for operational flight use.**

Research-grade; not flight-qualified, not certified, and not approved for
operational aerospace use.

> **Every number in this card was produced this session by a script in
> `validation/`, and each number says which one.** The commands are in
> Reproducibility below.

## The headline, stated first because it is not flattering

**An analytic baseline beats this model on both held-out splits.**

| Split | baseline 1 (acquisition only) | **baseline 2 (analytic fluid)** | learned random forest |
|---|---|---|---|
| held-out station pairs, n=384 | 0.536458 | **0.653646** | 0.390625 |
| held-out time, n=576 | 0.506944 | **0.682292** | 0.600694 |

Dominance accuracy, from
`validation/validate_learned_vs_baselines_output.txt`. On the station-pair split
the learned component does not even reach baseline 1's always-acquisition floor,
and baseline 2's 95 per cent interval `[0.606771, 0.700521]` is disjoint from the
learned component's `[0.343750, 0.440104]`. **This result is published as the
finding. It was not retuned away and the learned component was not removed:** a
reader needs both numbers to know what the learned component bought, which here
is nothing on one split and less than the closed form on the other.

## Problem

From **pre-handover state only**, predict the three per-mechanism data losses of
the coming ground-station handover on an optical downlink — acquisition,
protocol restart, buffer drain — in bits, and hence **which of the three will
dominate**.

Dominance accuracy is the headline metric because it is the operator's decision:
buy faster acquisition, change the protocol, or add buffer. Loss magnitude is
reported beside it as the mean absolute error of `log10(1 + loss)`, which is the
scale the quantity spans.

## Inputs, and why they are restricted

The 18 columns of `handoverloss.corpus.FEATURE_COLUMNS`: the source and downlink
rates, the buffer capacity and its occupancy at the break, the derived
occupancy fraction, headroom in seconds and utilisation, the declared ramp and
restart durations, the four acquisition-distribution parameters, the nominal
elevation, the turbulence, fade-margin and sky-cover covariates, and the source
burst coefficient of variation.

**No realised duration and no realised loss is a feature.**
`check_feature_causality()` asserts that by name and
`tests/test_corpus.py::test_no_feature_is_a_post_handover_quantity` tests it.
A predictor that needed the realised acquisition time would not be a predictor;
it would be a measurement.

Both analytic baselines are evaluated from the same information, which
`tests/test_predictor.py::test_baselines_see_no_realised_duration` asserts by
perturbing the realised-duration column and requiring the baseline predictions
not to move.

## Baselines, implemented and validated FIRST

`validation/validate_baselines_against_truth.py` was written and run before the
learned component existed, because an under-strength baseline inflates an
apparent learned win. What it measured:

- **Baseline 1 is the exact restriction of baseline 2 to its first phase.** Over
  400 handovers spanning the declared envelope, deliberately skewed towards a
  nearly full buffer: **400 of 400 bit-identical**, worst relative discrepancy
  **0.000000e+00**. If this had failed, baseline 1 would have been weakened and
  every margin over it inflated.
- **Baseline 1 was also evaluated at its conservative setting**, the 95th
  percentile of the acquisition distribution rather than its mean, and both are
  reported. Mean setting log10 MAE **2.104258**; 95th-percentile setting
  **3.316295**. Neither is close to baseline 2.
- **Baseline 2, given the realised durations, reproduces the ground truth**:
  dominance agreement **0.983724**, median total ratio **0.994247** (5th/95th
  percentile 0.900798 / 1.072828), log10 MAE on the total **0.033592**. That is
  the closed form's own accuracy, separated from its forecasting error.

## Architecture

`sklearn.ensemble.RandomForestRegressor`, three outputs, trained on `log1p` of
each mechanism's loss in bits; predictions inverted with `expm1` and clipped at
zero. A forest rather than a linear model because the mechanism structure is
piecewise — each term is a `max(0, ...)` of a different combination of the
inputs — and a forest represents a piecewise-constant surface without being told
where the breakpoints are. Breiman, L., "Random Forests", *Machine Learning*,
volume 45, number 1, pages 5–32, 2001.

**Hyperparameters: `n_estimators=300`, `max_depth=20`, `min_samples_leaf=2`,
`random_state=96096`.** Chosen by **three-fold grouped cross-validation over
station pairs, on the TRAINING side of the station-pair split only.** The test
side was never consulted and no split was re-rolled. Six candidates, measured by
`validation/validate_learned_vs_baselines.py`:

| max_depth | min_samples_leaf | trees | grouped-CV dominance accuracy |
|---|---|---|---|
| 14 | 3 | 200 | 0.352431 |
| None | 1 | 200 | 0.350694 |
| 8 | 5 | 200 | 0.342882 |
| **20** | **2** | **300** | **0.360243** |
| None | 3 | 400 | 0.349826 |
| 6 | 8 | 200 | 0.352431 |

**The spread of 0.017361 across all six candidates matters more than the
winner.** Every candidate lands in a narrow band, so the weakness on held-out
station pairs is structural, not a tuning matter. See Failure cases.

## Uncertainty output

Not a point estimate. `Prediction` carries, per event and per mechanism:

- `log_mean` and **`log_sd`, the standard deviation ACROSS TREES** on the
  `log1p` scale — the forest's own disagreement, which is the uncertainty
  estimate;
- `low_bits` / `high_bits`, the `log_mean ± z · log_sd` band inverted to bits;
- and for the dominance decision, `dominance_confidence`, the **fraction of
  trees whose own three-output prediction has the same argmax**.

**The across-tree spread is a lower bound on predictive uncertainty.** It
captures model uncertainty and not the irreducible noise of the handover itself,
so it is expected to undercover, and it does. Wager, S., Hastie, T. and Efron,
B., "Confidence intervals for random forests: the jackknife and the
infinitesimal jackknife", *Journal of Machine Learning Research*, volume 15,
pages 1625–1651, 2014, is the reference both for why the raw spread understates
and for the jackknife corrections that are **not** applied here.

Achieved coverage of the nominal 90 per cent band, from
`validation/validate_learned_vs_baselines_output.txt`:

| Split | acquisition | restart | drain | all |
|---|---|---|---|---|
| held-out station pairs | 0.875000 | 0.835938 | 0.976562 | 0.895833 |
| held-out time | 0.802083 | 0.862847 | 0.873264 | 0.846065 |

**Use the measured figures, not the nominal level.**

`dominance_confidence` is a vote share, not a calibrated probability. Its
measured reliability on the held-out time split:

| vote share | n | accuracy |
|---|---|---|
| [0.00, 0.40) | 91 | 0.131868 |
| [0.40, 0.50) | 36 | 0.555556 |
| [0.50, 0.60) | 55 | 0.400000 |
| [0.60, 0.70) | 81 | 0.592593 |
| [0.70, 0.80) | 65 | 0.692308 |
| [0.80, 0.90) | 106 | 0.735849 |
| [0.90, 1.00) | 142 | 0.852113 |

Monotone above 0.6 and useless below 0.5. An operator should treat a vote share
below 0.5 as "the model does not know".

## Dataset

`data/corpus/handovers.csv`, 1536 synthetic handovers over 16 station pairs and
8 epochs, sha256
`3a0343a6c2c51a91eba0679856700f5495d8814c3bcdc94e6b710ba300c59706`. Fully
described in `DATASET_CARD.md`, including the stratification that was introduced
after a first pass produced a degenerate dominance mix.

**It is synthetic, and it is generated by a mechanism baseline 2 nearly
reproduces in closed form.** That is the single most important limitation of
every number in this card.

## Test-split strategy

Two splits, **scored separately and never averaged**, because they answer
different operational questions.

- **Held-out STATION PAIRS.** The four highest-numbered pairs of sixteen held
  out entirely: train 1152, test 384. Deterministic, with no shuffling, so the
  split is a property of the corpus and not of a seed. Answers: *can this be
  deployed at a station pair with no history.*
- **Held-out TIME.** The last three of eight epochs held out: train 960, test
  576. Every test event is later than every training event. Answers: *will this
  still be right next month.*

`Split.__post_init__` refuses a split whose sides overlap.

## Metrics

From `validation/validate_learned_vs_baselines_output.txt`.

**Held-out station pairs, n=384:**

| Predictor | dominance accuracy | 95% interval | log10 MAE (total) | band coverage |
|---|---|---|---|---|
| baseline 1 acquisition-only | 0.536458 | [0.486979, 0.585938] | 1.632816 | n/a |
| baseline 1 @ q0.95 | 0.536458 | [0.486979, 0.585938] | 1.294335 | n/a |
| **baseline 2 analytic fluid** | **0.653646** | [0.606771, 0.700521] | **0.932722** | n/a |
| baseline 2 @ q0.95 | 0.520833 | [0.473958, 0.570312] | 1.506500 | n/a |
| learned random forest | 0.390625 | [0.343750, 0.440104] | 1.707133 | 0.895833 |

**Held-out time, n=576:**

| Predictor | dominance accuracy | 95% interval | log10 MAE (total) | band coverage |
|---|---|---|---|---|
| baseline 1 acquisition-only | 0.506944 | [0.465278, 0.546875] | 1.795279 | n/a |
| baseline 1 @ q0.95 | 0.506944 | [0.465278, 0.546875] | 1.433896 | n/a |
| **baseline 2 analytic fluid** | **0.682292** | [0.644097, 0.718750] | **0.863551** | n/a |
| baseline 2 @ q0.95 | 0.550347 | [0.508681, 0.590278] | 1.389319 | n/a |
| learned random forest | 0.600694 | [0.560764, 0.640625] | 1.128893 | 0.846065 |

Per-mechanism log10 MAE, held-out time: the learned component is better than
baseline 2 on no mechanism (acquisition 2.266773 against 2.091804; restart
1.074787 against 0.794232; drain 0.817029 against 0.605632).

The `@q0.95` rows are baseline 1 and baseline 2 evaluated at the 95th percentile
of the acquisition distribution rather than at its mean, which is the
conservative engineer's setting. Both get worse on dominance accuracy and on
log10 MAE, except baseline 2 on the time split where the dominance accuracy falls
from 0.682292 to 0.550347. The conservative setting is reported so nobody can say
the refuted assumption was tested only in its weakest form.

## Failure cases

1. **A station pair the model has never seen.** Accuracy falls to 0.390625,
   below baseline 1's always-acquisition floor of 0.536458. The cause is
   structural: the features that identify a pair's configuration — rates, buffer,
   declared ramp and restart — are **constant within a pair**, so the forest has
   16 distinct configurations to learn from and cannot interpolate to a
   seventeenth. The hyperparameter sweep confirms this is not a tuning matter:
   all six candidates land within 0.017361 of each other.
2. **Low dominance confidence.** Below a 0.5 vote share the model's accuracy on
   held-out time is 0.131868 to 0.555556, worse than guessing the base rate. The
   confidence output is how a caller detects this, and it is why the model
   exposes one.
3. **Undercoverage on the acquisition mechanism.** 0.802083 against a nominal
   0.90 on held-out time, the worst of the three. The acquisition term has the
   heaviest tail (the retry structure) and the across-tree spread tracks it least
   well.
4. **Zero-loss handovers.** 216 of 1536 corpus events have no loss at all. The
   model predicts a small positive loss for most of them, and `argmax` over three
   near-zero numbers is meaningless. A caller should check the predicted total
   before reading the dominance.
5. **Anything outside the declared generator envelope.** The model has seen
   utilisation 0.039 to 0.970, restart 2.07 to 83.28 s, ramp 4 to 170 s nominal,
   and no `rho > 0.75` combined with a headroom above 120 s — that cell is empty
   in the corpus and is reported as structurally unmeasurable.

## Compute used

Measured this session on the two-core build machine, under contention from four
other concurrent build agents. Reported, never asserted as a relationship
between wall-clock readings.

| Step | Measured | Declared budget |
|---|---|---|
| Corpus generation, 1536 events × 2800 slots | 8.119 s | 60 s |
| Fit + predict + bootstrap, held-out station pairs | 0.812 s | 60 s |
| Fit + predict + bootstrap, held-out time | 0.758 s | 60 s |
| `validation/validate_learned_vs_baselines.py` whole script | 10.78 s | 600 s per script |
| Whole pytest suite, 222 tests | 6.75 s | 1800 s |

The two fit timings are single readings from the committed run of
`validation/validate_learned_vs_baselines.py`. Repeated runs this session gave
0.71 to 0.84 s. **No test or validation script asserts a relationship between
two wall-clock readings**, which cost this mission a whole product; the one
timing assertion in `tests/test_regressions.py` is a direction, `elapsed > 0`.

No model is serialised anywhere in this repository. The forest is refitted from
the committed corpus in under two seconds, and a pickled estimator would bind
the repository to one scikit-learn build whose unpickling format is not part of
that library's public API.

## Reproducibility

Exact commands, from the repository root, with the environment measured this
session: Python 3.13.18, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1,
pytest 9.1.1, hypothesis 6.170.15, ruff 0.16.9 (and 0.17.5 as a module).

```bash
python3 data/generate_corpus.py
python3 validation/validate_baselines_against_truth.py
python3 validation/validate_learned_vs_baselines.py
python3 examples/learned_vs_baselines.py
```

Seeds: corpus root seed `96096 2026` written as `96_096_2026`; forest
`random_state=96096`; analysis and bootstrap `96096`; negative controls `5150`;
trace `96096 2026`. All are declared in one place each —
`data/generate_corpus.py::ROOT_SEED`, `handoverloss.predictor`, and
`validation/_bootstrap.py`.

## Ethical and safety limits

- **This model is not certified for operational flight use.**
- It is fitted on synthetic data generated by a model of a handover, not on any
  measured handover. Its numbers describe this repository's generator and
  nothing else.
- An analytic baseline available in this same package outperforms it on both
  held-out splits. **A reader deciding what to deploy should deploy
  `fluid_decomposition`, not this model**, unless they have measured handover
  data on which the comparison comes out differently — which is precisely the
  measurement this repository does not have.
- Impurity-based feature importances are reported in
  `validation/validate_learned_vs_baselines_output.txt` as a description of the
  fitted forest. They are biased towards high-cardinality features and are not
  evidence about the physical mechanism. `headroom_s` carries 0.670922 of the
  importance on the station-pair split; that reflects the generator's structure,
  not a discovery.
