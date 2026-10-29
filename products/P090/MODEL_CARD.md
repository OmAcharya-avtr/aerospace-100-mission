# Model card — `tlecheck` learned screening classifier

**This model is not certified for operational flight use.**

Research grade. Every number in this card was produced by
`validation/validate_learned.py`, executed on 2026-10-29, whose raw output is
committed beside it as `validation/validate_learned_output.txt`.

## Problem

Given a two-line element set, the object's own prior element sets, and element
sets for the same object from other catalogues, predict which of three
consequence classes the element set falls into:

- `will_fail_loudly` — the propagator will refuse it or return a non-zero SGP4
  error code;
- `silently_wrong` — the propagator will report success and the resulting
  position will be more than a declared threshold away from the position the
  uncorrupted element set would give;
- `benign` — the propagator will report success and the position error will not
  exceed the threshold.

**The label is a measurement, not an annotation.** It is produced by propagating
the candidate and its uncorrupted reference with `sgp4` over the same absolute
time grid and differencing the positions, which is equation (11) of
`tlecheck.consequence`. The declared threshold is 1 km and it is swept in
`validation/validate_consequence.py`, where the class counts move from
124 silent at 0.1 km to 28 at 500 km on the same 420 element sets.

The class this model exists for is `silently_wrong`, because a silently bad
element set that passes screening is the whole failure this product addresses.

## Baseline, implemented and validated first

The baseline is `tlecheck.verdict`, a set of explicit screens with declared
thresholds. It was implemented, validated and strengthened **before** this model
was benchmarked, and it is the operational default of the package. It is given a
continuous score, equation (15), so it can be compared curve against curve
rather than point against curve; scoring a rule set only at its own operating
point understates it.

The baseline was changed five times in response to measurement, and the whole
history with what each change was worth is in `validation/VALIDATION.md`
section 7. The short version: the published learned margin is **+0.3661 recall
at a fixed false-positive rate**, and before the strengthening it was about
**+0.78** — a build-time figure, measured against code that is no longer in the
repository and labelled as such in `validation/VALIDATION.md` section 7.
Publishing the first figure would have overstated the learned contribution by
more than a factor of two.

A majority-class baseline is scored beside both, so that a model which has
learned nothing cannot look useful. Its area is 0.5000 by construction, because
its score is a constant.

## Architecture

`sklearn.ensemble.RandomForestClassifier`, 160 trees, maximum depth 12,
`min_samples_leaf` 2, `class_weight="balanced"`, `random_state=20261029`,
`n_jobs=1`.

A tree ensemble is chosen for three reasons about this problem rather than about
fashion. The features are a mixture of indicators, bounded angles and
heavy-tailed z-scores on wildly different scales, which trees do not care about.
The sample size is a few hundred objects per split, which rules out anything
that needs more. And the per-member votes give an uncertainty output without
extra machinery.

**PyTorch is not available in this environment and no neural network is used
anywhere in this repository.**

### Three declared feature configurations, all three reported

| configuration | features | what it is for |
|---|---|---|
| `elements_and_history` | 45 — element values, derived orbit geometry, checksum and parse validity, epoch age, the five history z-scores with infinity indicators, the continuity offset, median, scale, z-score, excess and resolution, and the cross-source separation | the default, and the one the specification names |
| `elements_only` | 21 — the above minus every history, continuity and cross-source feature | measures what the history contributes rather than asserting it |
| `with_rule_score` | 46 — `elements_and_history` plus equation (15) | answers the question the specification actually asks: given the rules, does a model add anything on top |

**The learned model is given every feature the rule baseline uses**, including
the continuity offset and the screen's own resolution. A learned win over a
baseline that was shown less information is not a result.

### Two feature decisions that affect the honesty of the benchmark

**The uncorrupted reference element set is never a feature and is not reachable
from `tlecheck.features`.** It does not exist in an operational setting, which
is exactly why screening is needed; it exists in this repository only to measure
consequence. No feature name contains `reference`, `truth` or `corruption`, and
a test asserts that.

**An infinite robust z-score is encoded as a declared cap of 1000.0 together
with a separate indicator column**, rather than being flattened into a large
finite number. An infinity is a real outcome — it is what a zero history scale
with a non-zero residual produces — and a tree ensemble cannot be fitted on one.
The indicator preserves the information that the value was infinite.

## Dataset

`tlecheck.dataset`, documented in full in `DATASET_CARD.md`. A seeded synthetic
catalogue: six orbit archetypes, element histories built with the SGP4
propagator itself, and eight corruption families with exact ground truth.

**This repository ships no element catalogue and that is deliberate.** The
authoritative source of element sets is an operator's catalogue, which is not
redistributable, not reproducible from a repository, and not something a
screening tool should bundle a stale copy of.

The consequence of the synthetic corpus is stated rather than implied: the
precision and recall figures here are measured against **this generator's**
corruption distribution, and a real one would give different numbers. The
figures in kilometres are different in kind — those are real measurements of
what `sgp4` does with these element sets.

## Training procedure and splits

Three splits on **disjoint object-number blocks** with different seeds, so no
object appears in two and the element text of one is not reachable from another.

| split | seed | object block | objects | loud | silent | benign |
|---|---|---|---|---|---|---|
| train | 20261029 | 10000+ | 420 | 20 | 122 | 278 |
| test | 20261030 | 40000+ | 420 | 18 | 112 | 290 |
| calibration | 20261031 | 70000+ | 419 | 16 | 127 | 276 |

The calibration split selected nothing and trained nothing. It is used only for
the uncertainty outputs, so those are measured on data that had no part in any
choice.

One object of the calibration split produced fewer than two element sets because
the propagator refused the rewind, and the generator omits it rather than
quietly replacing it with a different object. That is why the count is 419.

Nothing was tuned on the test split. The declared four-sigma thresholds, the
1 km silent threshold and the 7 day epoch requirement were fixed in advance; the
sweep in `validation/VALIDATION.md` section 4 shows that a threshold of 8.0
would score better on this corpus, and it was not adopted for that reason.

## Metrics

Recall on the silent class at a fixed false-positive rate is the headline,
because a screen that catches silent element sets by flagging everything has not
helped. The achieved rate is always reported beside the recall, because it is
usually below the target and quoting the target as if it were achieved would
overstate the result.

### The headline, test split, 420 objects

| predictor | recall | achieved fpr | area | accuracy |
|---|---|---|---|---|
| rules at their own threshold | 0.8661 | 0.3019 | 0.8677 | 0.7429 |
| rules swept, equation (15) | **0.4286** | 0.0455 | 0.8677 | 0.7429 |
| majority-class baseline | 0.0000 | 0.0000 | 0.5000 | 0.6905 |
| learned `elements_and_history` | 0.7768 | 0.0487 | 0.9583 | 0.8976 |
| learned `elements_only` | 0.5804 | 0.0487 | 0.7960 | 0.8238 |
| learned `with_rule_score` | **0.7946** | 0.0455 | 0.9629 | 0.9095 |

**The product specification predicted that the rule screens would match or beat
this model on every class they cover. They do not.** The learned model wins the
silent class by +0.3661 recall at a comparable false-positive rate, and wins on
accuracy and area as well. That is reported as measured and nothing was removed,
softened or retuned to change it.

The `elements_only` row is the measurement of what the history contributes:
removing the history, continuity and cross-source features costs 0.2142 recall
and 0.1669 of area. The history is doing most of the work, and the history is
also what the rule screens use.

### Per class, test split

| predictor | loud P | loud R | silent P | silent R | benign P | benign R |
|---|---|---|---|---|---|---|
| rules | 0.8500 | 0.9444 | 0.5105 | **0.8661** | **0.9429** | 0.6828 |
| majority | nan | 0.0000 | nan | 0.0000 | 0.6905 | 1.0000 |
| learned `elements_and_history` | 0.9000 | 1.0000 | 0.8723 | 0.7321 | 0.9052 | 0.9552 |
| learned `with_rule_score` | 0.8947 | 0.9444 | 0.8713 | 0.7857 | 0.9233 | 0.9552 |

Tally against the best learned configuration: **rules 2, learned 3, tie 1.**

### What this model is not asked, and the rules answer exactly

The integrity verdict. Over the test split the checksum screen reaches
precision **1.0000** and recall **1.0000** on the 57 element sets whose checksum
should be inconsistent, with **0** false positives in 363. Those are not
statistics: the checksum is a digit sum, with no threshold, no training set and
no error rate. Nothing in this model's output is an integrity verdict and a
caller who needs one could not get it from a classifier.

## Uncertainty and confidence output

Two outputs, both returned with every prediction.

- **Equation (19), ensemble dispersion.** The standard deviation across the 160
  member trees of the predicted probability of the predicted class.
- **Equation (20), decision margin.** The gap between the two largest class
  probabilities. A margin near zero is the model saying it cannot separate two
  classes for this element set, which a point prediction destroys.

Measured on the independent calibration split:

| group | n | mean margin | mean dispersion |
|---|---|---|---|
| correct | 384 | **0.6871** | 0.2827 |
| wrong | 35 | **0.2839** | 0.4211 |

The margin carries information: it is higher on predictions that turned out
right and the dispersion is higher on predictions that turned out wrong.
**It is not a probability of being right and nothing here calibrates it to one.**
Any uncertainty this model reports is an uncertainty about its own screening
decision; it is never an uncertainty about the elements, because the two-line
element format carries no covariance and the determinability report says so
before any screen runs.

## Failure cases

**1. It answers confidently where the analytic screen withholds its test.**
Equation (27) of `tlecheck.continuity` reports the continuity screen's own
resolution, and where that resolution exceeds the declared silent threshold the
screen withholds its absolute test rather than returning a verdict that is
noise. This model has no such mechanism.

| subset | n | errors | margin above 0.5 | fraction | mean margin |
|---|---|---|---|---|---|
| equation (27) resolves the threshold | 142 | 11 | **0** | 0.0000 | 0.7291 |
| equation (27) does **not** resolve it | 278 | 32 | **7** | **0.2188** | 0.6403 |

Seven of thirty-two errors on the unresolvable subset carry a margin above 0.5,
against none of eleven on the resolvable one. This is the failure mode the
mission's Batch 09 exists to look for: a confident answer about something the
data does not support.

**2. It loses badly on a corruption family it has not seen.** Retrained with one
corruption family withheld and scored on that family against the uncorrupted
records:

| withheld corruption | rules area | learned, withheld | learned, all | rules win |
|---|---|---|---|---|
| data_digit | 0.8285 | 0.9045 | 0.9409 | no |
| field_shift | 0.9065 | 0.9534 | 0.9425 | no |
| sign_flip | 0.9488 | 0.9813 | 0.9912 | no |
| implausible_eccentricity | 0.7858 | 0.9863 | 0.9910 | no |
| **stale_epoch** | **0.8601** | **0.6448** | 0.9438 | **yes, +0.2153** |
| **cross_source_divergence** | **0.9798** | **0.8335** | 0.9825 | **yes, +0.1463** |

The area recovers when the family is restored to training, so the drop is the
unseen-family effect and nothing else. **A real corruption is not drawn from the
distribution a model was trained on.** None of the rule screens was fitted to a
corruption distribution at all.

**3. Its silent-class recall is below the rules' own operating point.** 0.7946
against 0.8661. The learned model wins at a *fixed low false-positive rate*; the
rules, allowed their own 0.3019 rate, catch more. Which of those matters depends
on what a false alarm costs in the caller's operation, and this package does not
know that.

**4. It does not transfer to the one family where element-set continuity is
genuinely uninformative.** On the withheld `leo_decaying` family the rules take
the area 0.8442 against 0.8305. That is the orbit family whose continuity
resolution is 29.4676 km against a 1 km threshold.

## Reproducibility

```bash
python validation/validate_learned.py
```

Every corpus is regenerated deterministically from the seeds in the table above,
and `random_state=20261029` is fixed in `tlecheck.learned.LearnedClassifier`. A
test asserts that two fits on the same matrix give bit-identical probabilities.
Nothing is pickled and nothing is committed as a serialised artifact, so there is
no cross-version unpickling hazard; the model is refitted from the committed
generator every time.

The declared settings behind every number: 420 objects per split, 12 prior
element sets per object at 1.5 day cadence, a 1 day consequence horizon at 25
samples, a 1 km silent threshold, a 7 day epoch requirement, a target
false-positive rate of 0.05.

## Compute used

Two CPU cores. The three stages — screening, measuring and featurising all
three 420-object splits, fitting all three declared configurations, and
predicting the test split — are all of order seconds, and the whole benchmark is
well inside the mission's three-minute budget. The numbers are in
`validation/validate_learned_output.txt` section 10, where every line is tagged
`[unstable]`, and they are deliberately not copied here: a timing changes on
every re-run and a figure quoted in this card would be falsified by the next
one.

No GPU is used and none is available. PyTorch is not installed.

## Ethical and safety limits

- **This model is not certified for operational flight use.** It is research
  grade, it is not flight-qualified, and it must not be placed in a control path.
- **It must not be used as the only screen.** The operational default of this
  package is the rule baseline, which is explicit, auditable, and measurably
  more robust to a corruption it has not seen. The model is an addition to it.
- **A prediction from this model is not a statement about ephemeris accuracy.**
  Nothing in this package can establish that; the first limit of every report
  says so.
- **Its numbers do not transfer to a real catalogue without re-measurement.**
  The corruption distribution it was fitted to is synthetic and documented, and
  the two transfer experiments above show what an unfamiliar corruption costs.
- **It has no access to the uncorrupted element set and cannot be given one.**
  Any future version that is trained with one would be measuring something
  other than screening.
