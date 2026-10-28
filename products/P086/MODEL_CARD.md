# Model card — `telemgap` learned anomaly classifier

**This model is not certified for operational flight use.**

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use.

## Problem

Given a decoded telemetry frame and one of its channels, decide whether that
channel's value is clean, **corrupt** (the bits are damaged) or **misparsed**
(the bits are intact and the reader applied the wrong decode).

The rules of `telemgap.classify` already answer this for every case where the
evidence is decisive. The learned model exists for the residual: the rows the
rules leave **undecided**, which on the measured split is 352 of 45,500 rows
(0.77 per cent), of which 289 are truly clean, 55 truly corrupt and 8 truly
misparsed. The largest part of that residual is frames whose error-control
field failed — which proves that *some* word in the frame is damaged and says
nothing about *which*.

## Baseline, implemented and validated first

Two rule baselines, both implemented and measured before the model was fitted,
in `validation/validate_rule_screens.py` and
`validation/validate_classification.py`.

`specified`
: the four screens the product specification names — timestamp monotonicity,
  expected frame cadence, counter continuity and declared value ranges — plus
  the frame error-control field and the reinterpretation test of
  `telemgap.reparse`, which are what make the three-way separation possible at
  all.

`strengthened`
: the above plus a robust local-residual screen, equation (5). This is the
  baseline the model is scored against, because the specified baseline is
  structurally blind to a bit error that leaves a value inside its declared
  range, and the build guide requires the strongest honest baseline.

Measured on the held-out split: the local-residual screen is worth +0.2092
corrupt recall. Both baselines are reported side by side in
`validation/VALIDATION.md` §2.

## Architecture

`sklearn.ensemble.RandomForestClassifier`, 120 trees, `max_depth=12`,
`min_samples_leaf=4`, `n_jobs=2`. Fifteen features, listed and documented one
per line in `telemgap/features.py`.

The choice is not a performance claim. The features are low-dimensional, mixed
continuous and binary, and the decision structure the rules encode is a shallow
conjunction of thresholds, which is the hypothesis class a forest represents
natively. A gradient-boosted ensemble was not used because it has no
uncertainty output that survives the per-member decomposition this model
reports. No neural model was used: **PyTorch is not installed in the measured
environment and nothing was installed to get it.**

**No hyperparameter search was run.** Every value above is fixed in the source
and was not tuned against any measurement.

### Two feature-set decisions that affect the honesty of the benchmark

**Channel identity is deliberately excluded.** The generator injects the
byteswap misparse on one named channel and the scale misparse on another, so a
channel-index feature would let the model recover the dataset's construction
rather than the physics. There is no channel one-hot, no channel index and no
channel-specific constant; every feature is normalised by a declared property
of its own channel.

**The model is given everything the rules use.** Features 8 to 10 and 13 are
the outputs of the reinterpretation test, which is the rules' own discriminator
between corrupt and misparsed. If the model loses to the rules it loses with
the rules' evidence in hand.

## Dataset

See `DATASET_CARD.md`. In short: a seeded synthetic generator,
`telemgap.dataset`, whose closed forms are printed in its module docstring.
**No flight data is used anywhere in this package and no number in it is taken
from a real spacecraft.**

Limitations that bear directly on the model: four channels only; one
decommutation form (equation 1); three injected misparse kinds; a corruption
model of one to three flipped bits with 60 per cent of them also failing the
frame error-control check. A real archive's anomaly mix is not this mix, and
the measured scores do not transfer to one.

## Training procedure and splits

Three disjoint realisations of the generator, 12,000 frames each:

| split | seed | purpose |
|---|---|---|
| training | 20261028 | fitting |
| test | 71120261 | every score reported |
| calibration | 44520261 | the uncertainty measurements only |

No frame is shared between them: they are independent draws from the generator,
not a partition of one record.

Three configurations are **declared** in `telemgap.learned.CONFIGURATIONS`,
all three fitted, and all three reported whatever they measure:

| configuration | clean-class handling | class weight | trained on |
|---|---|---|---|
| `recall_weighted` | subsampled to 10x the anomaly count | `balanced_subsample` | all rows |
| `unweighted` | none | none | all rows |
| `ambiguous_only` | none | none | the ambiguous residual rows only |

Scoring never subsamples and never reweights. Every reported precision and
recall is measured on the complete held-out population, because a precision
measured on a rebalanced test set is not a precision.

## Metrics

Per class: precision, recall, F1, undecided rate, and *surfaced rate* — the
share of a class either named correctly or explicitly flagged undecided, which
is the operational quantity for a tool a human reads. All five are defined as
equations (13) to (15b) in `telemgap/benchmark.py` and all five are reported
for every predictor.

### Result on the complete held-out population, 45,500 rows

Strengthened rules against the strongest learned configuration:

| class | metric | rules | learned | delta | winner |
|---|---|---|---|---|---|
| clean | recall | 0.9936 | 1.0000 | +0.0064 | learned |
| clean | precision | 0.9994 | 0.9987 | −0.0007 | rules |
| clean | surfaced | 1.0000 | 1.0000 | 0.0000 | tie |
| corrupt | recall | 0.4052 | 0.5098 | +0.1046 | learned |
| corrupt | precision | 0.9688 | 0.9750 | +0.0062 | learned |
| corrupt | surfaced | 0.7647 | 0.5098 | −0.2549 | rules |
| misparsed | recall | 0.9321 | 0.9815 | +0.0494 | learned |
| misparsed | precision | 0.9437 | 0.9086 | −0.0352 | rules |
| misparsed | surfaced | 0.9815 | 0.9815 | 0.0000 | tie |

Tally: rules 3, learned 4, tie 2.

The product specification predicted that the rules would win on every class
they cover. **On this population they did not**, and that is reported as
measured rather than reframed. The one large margin runs the rules' way: they
surface 0.7647 of the corrupt class against 0.5098, because they abstain on a
row they cannot decide and these learned configurations never do.

### Result on the ambiguous residual set, 352 rows

| predictor | accuracy | delta against the majority guess |
|---|---|---|
| majority-class guess (`clean` at 0.785714) | 0.821023 | — |
| learned, `recall_weighted` | 0.255682 | **−0.565341** |
| learned, `unweighted` | 0.906250 | +0.085227 |
| learned, `ambiguous_only` | **0.926136** | **+0.105114** |

Pooled across a 12-cell grid of anomaly rate against cadence jitter, 1,806
ambiguous rows: `ambiguous_only` 0.877076 against a majority guess of 0.803433.

**The imbalance handling decides the answer completely.** A model trained with
`class_weight="balanced_subsample"` on a subsampled clean class is 56.5
accuracy points *below* a constant on the population it exists for. A model
trained on the ambiguous rows alone with no reweighting is 10.5 points above
it. Nothing about the architecture changes between them, and both numbers are
published.

## Uncertainty and confidence output

Two numbers per prediction, which are different things and are labelled
differently.

`probability`
: the forest's vote share for the predicted class. It is a vote share, and a
  vote share is only a probability to the extent that it is calibrated. Its
  expected calibration error is measured on the independent third split and is
  **not post-hoc corrected**.

`dispersion`
: the standard deviation across the forest's individual trees of the predicted
  class's probability, accumulated by Welford's algorithm. This is an ensemble
  disagreement measure, not a confidence interval: the trees are not
  independent draws from a posterior. The only claim made for it is the area
  under the ROC curve of dispersion against the indicator of being wrong.

| configuration | rows | accuracy | ECE | dispersion AUC against wrong |
|---|---|---|---|---|
| recall_weighted | 47,832 | 0.990676 | 0.054133 | 0.965703 |
| unweighted | 47,832 | 0.998098 | 0.000781 | 0.886237 |
| ambiguous_only | 363 | 0.884298 | 0.027389 | 0.685877 |

An abstention threshold on the vote share is available and its trade is swept
rather than chosen: on the ambiguous set the `ambiguous_only` configuration
moves from 0.926136 accuracy at no abstention to 0.975610 at a 0.767045
abstention rate.

## Failure cases

1. **The recall-weighted configuration on the ambiguous set.** 0.255682
   accuracy against a majority guess of 0.821023. It predicts `corrupt` on 317
   of 352 rows at a precision of 0.1735. Measured and published.
2. **The corrupt class is the hard one for everything.** The best recall any
   predictor reaches on it is 0.5098, because 0.862371 of single-bit flips of
   an in-range word stay in range and are invisible to a declared-range screen.
   The model inherits that ceiling; it does not remove it.
3. **A misparse this package does not model is reported undecided**, never
   misparsed. The three modelled kinds are byteswap, signedness and a
   decade scale error. A polynomial or table calibration error has no
   alternative reading in `telemgap.reparse`.
4. **Out of distribution, nothing is claimed.** The model was fitted on a
   four-channel synthetic generator at a one-second cadence with 16-bit words.
   Its scores on any other record are unmeasured.
5. **The feature extractor requires raw words.** Without them the misparsed
   class is unreachable and `telemgap.features.build` raises rather than
   silently producing a degraded feature set.

## Reproducibility

```bash
python -m pytest tests/ -q
python validation/validate_rule_screens.py
python validation/validate_classification.py
python validation/validate_learned.py
python examples/rules_versus_learned.py
python examples/learned_confidence.py
python -m telemgap benchmark --samples 12000
```

Seeds: training 20261028, test 71120261, calibration 44520261; forest and
subsample seed 20261028, fixed in `LearnedClassifier.seed`. Sweep cell seeds
are derived deterministically from the cell indices inside each validation
script. Every number in this card comes from
`validation/validate_learned_output.txt`.

## Compute used

Generation, feature extraction, rule screening, three fits and three
predictions over three 12,000-frame splits plus a 12-cell swept grid:
**5.26 s wall clock on two cores**, against a three-minute budget. Peak memory
is dominated by the 45,524-row by 15-column float64 feature matrix, about 5.5
MB. No GPU. Timings are unstable and are quoted only here and in
`validation/VALIDATION.md`.

## Ethical and safety limits

- **This model is not certified for operational flight use.**
- It is not a flight-safety mechanism and must not be placed in a control path.
  Its output is an opinion about an archive, produced after the fact.
- A clean integrity report does not mean the telemetry is correct. It means no
  screen fired and the model did not object; both have measured miss rates and
  both are in `validation/VALIDATION.md`.
- The model never decides whether a gap may be interpolated across. That is
  `telemgap.policy`, which is rule-based, and the separation is deliberate: a
  learned model should not be the thing that grants permission to invent data.
- No personal data is involved. The dataset is synthetic and contains no flight
  data, so there is no provenance or licensing question about its contents.
