# Model card — `cmdguard.ranker.RiskRanker`

**This model is not certified for operational flight use.**

Every number in this card comes from a script executed in the build session
that produced this repository, and each is attributed to the transcript it came
from. Reproduce them with the commands at the end.

## What the model is for

Triage, not detection. Detection is done by `cmdguard.trace.forward_execute`,
which is exact for a single linear sequence against the declared model and
costs milliseconds (`validation/validate_determinism_output.txt`, section 6).
What is expensive is the human review of a long sequence: a reviewer given four
hundred commands and a dozen findings still has to decide which commands to
read first.

The model orders the steps of a sequence by the probability that a step is
**involved in** a violation, meaning it either tripped one or is named in the
provenance trace of one. That is exactly the set a reviewer must read to
understand the findings.

## Problem framing

- **Unit of prediction:** one command occurrence in one candidate sequence.
- **Label:** `cmdguard.ranker.involved_mask`, derived from the forward walk.
- **Output:** `RiskScores.risk`, the forest's mean positive-class probability
  per step, and `RiskScores.spread`, the standard deviation of the per-tree
  probabilities.
- **Metric:** mean precision at k over held-out cases, with recall at k beside
  it and a paired percentile bootstrap over cases for every margin.

## Baselines it is scored against

| Baseline | What it is | Cost |
|---|---|---|
| `irreversibility_ordering` | Irreversible commands first, then by declared draw as a fraction of its ceiling. | One sort over a flag already in the command dictionary. |
| `resource_draw_ordering` | By largest declared draw as a fraction of its ceiling. | One sort. |
| `random_ordering` | A seeded shuffle. The control. | None. |

The specification for this product predicted that the flag ordering would match
or beat the learned model, and said that such a result is to be published
rather than retuned away. It did not, on this corpus, and the measured result is
below.

## Architecture

`sklearn.ensemble.RandomForestClassifier`, 200 trees, `max_depth=8`,
`class_weight="balanced"`, `random_state=20261024`, `n_jobs=1`.

`class_weight="balanced"` is set because the positive class is a minority — the
measured positive fraction of the training rows is **0.102769** — and an
unweighted forest on this corpus predicts the majority class everywhere, which
makes every ordering it produces a tie. The cost of that choice is calibration,
and it is measured rather than hidden; see **Uncertainty** below.

Twenty features, all static: computed from the command dictionary, the declared
initial state and the sequence's command and time columns. **No feature is
derived from the forward walk.** A feature that encoded the walk's own
conclusion would make the measured precision a statement about label leakage,
and the leak would be invisible in the score. The absence is asserted by test:
`tests/test_features.py::test_features_do_not_read_the_forward_walk` edits the
mode-machine transition relation, which changes the violations and which no
feature reads, and requires the feature matrix to be bit-identical.

The feature list with units is in the `cmdguard.features` module docstring.

## Training data

Seeded synthetic. See `DATASET_CARD.md`. **No flight data is used anywhere in
this repository**, and every result here is a statement about that generator.

| Quantity | Value | Source |
|---|---|---|
| Training seeds | 70 (0–69) | `validation/validate_ranker_output.txt` §1 |
| Training cases | 630 | same |
| Training rows (command occurrences) | 22750 | same, §5 |
| Positive rows | 2338 | same |
| Positive fraction | 0.102769 | same |
| Held-out seeds | 35 (100000–100034) | same, §1 |
| Held-out cases | 315 | same |
| Held-out command occurrences | 11375 | same |
| Held-out positive fraction | 0.089231 | same |

The train and held-out seed ranges are checked disjoint by
`cmdguard.benchmark.assert_disjoint`, which raises rather than warns. A held-out
score measured on a seed the model trained on is not a held-out score.

## Metrics, as measured

Mean precision and recall at k over 302 scored held-out cases; 13 of the 315
were excluded because the case carried no positive label or was shorter than k,
and the exclusion is label-driven and therefore identical for every ordering.
All figures from `validation/validate_ranker_output.txt` §2.

| Ordering | P@5 | P@10 | P@20 | R@5 | R@10 | R@20 |
|---|---|---|---|---|---|---|
| **learned, all features** | **0.270199** | **0.210265** | **0.144040** | 0.492905 | 0.698142 | 0.883113 |
| learned, timing features withheld | 0.213907 | 0.174503 | 0.135265 | 0.363070 | 0.529963 | 0.814182 |
| irreversibility flag | 0.156954 | 0.150331 | 0.102483 | 0.256661 | 0.449133 | 0.598273 |
| resource draw | 0.183444 | 0.157285 | 0.102152 | 0.264971 | 0.448804 | 0.595513 |
| random control | 0.071523 | 0.084106 | 0.098344 | 0.099441 | 0.232377 | 0.576683 |

Paired differences, learned minus baseline, by case, with a percentile
bootstrap 95 per cent interval over 4000 resamples:

| k | vs flag | vs resource draw |
|---|---|---|
| 5 | +0.113245 [+0.088079, +0.138411] | +0.086755 [+0.062252, +0.111921] |
| 10 | +0.059934 [+0.048344, +0.071854] | +0.052980 [+0.041060, +0.064901] |
| 20 | +0.041556 [+0.035762, +0.047355] | +0.041887 [+0.036093, +0.047848] |

With the three timing-shape features withheld:

| k | vs flag | vs resource draw |
|---|---|---|
| 5 | +0.056954 [+0.032450, +0.081457] | +0.030464 [+0.005298, +0.056291] |
| 10 | +0.024172 [+0.011589, +0.037086] | +0.017219 [+0.004967, +0.029470] |
| 20 | +0.032781 [+0.026325, +0.039073] | +0.033113 [+0.026656, +0.039404] |

**How to read this.** The learned model is ahead of both free orderings at every
cut-off, and every interval excludes zero. The absolute numbers are small: at
k=5 a precision of 0.270199 means that of the five commands put in front of a
reviewer, on average 1.35 were worth reading, against a base rate of 0.089231
and a mean of 3.22 involved steps per 36.1-step case. The recall column is the
one a reviewer cares about: the learned ordering's top ten covers 0.698142 of
the steps that matter.

**And the part that is an artefact.** Four of the nine defect families edit the
time column, so inter-command spacing carries the signature of the injection
process rather than information about commanding. Withholding `dt_prev_s`,
`dt_next_s` and `neighbours_within_60s` costs the learned model **0.056291** of
precision at k=5, **0.035762** at k=10 and **0.008775** at k=20 — roughly a
third of its margin at k=5. The ablated row is the conservative number and it
is the one to quote outside this generator.

**And a result in the baselines.** Against the random control, the free
orderings are ahead at k=5 (+0.085430 [+0.061589, +0.110596] for the flag) and
at k=10, but at k=20 the interval includes zero for both: +0.004139 [-0.002649,
+0.010762] for the flag and +0.003808 [-0.003146, +0.010596] for resource draw.
With twenty of thirty-six steps already shown, the free orderings are **not
distinguished from a shuffle** on this corpus.

Top feature importances (`validation/validate_ranker_output.txt` §5):

| Feature | Importance |
|---|---|
| `dt_prev_s` | 0.119267 |
| `dt_next_s` | 0.112697 |
| `power_draw_w` | 0.089800 |
| `draw_fraction_of_ceiling` | 0.081799 |
| `position_fraction` | 0.081796 |
| `n_preconditions` | 0.076939 |
| `irreversible` | 0.003546 |

The irreversibility flag is the **least** important feature to the forest and
the flag ordering still beats a shuffle at k=5. Both facts are in the
transcript and neither is edited.

## Uncertainty output

`RiskScores.spread` is the standard deviation across the forest's per-tree
positive-class probabilities. It is an **ensemble dispersion, not a calibrated
confidence interval**; see Wager, Hastie & Efron 2014 for what a calibrated
interval would require.

Measured on the held-out set (`validation/validate_ranker_output.txt` §6):

| Quantity | Value |
|---|---|
| Mean spread on involved steps | 0.167959 |
| Mean spread on uninvolved steps | 0.134671 |
| Mean risk on involved steps | 0.574477 |
| Mean risk on uninvolved steps | 0.405401 |

**The risk is not a probability.** The reliability table in the same section
shows a weighted mean absolute calibration gap of **0.331780**: in the
[0.4, 0.5) bin the mean predicted risk is 0.448457 and the observed involved
fraction is 0.079782. This is the expected consequence of
`class_weight="balanced"` and it is not corrected, because the model is used as
a ranking and a ranking is invariant under a monotone transform of its score. A
caller who wants a probability must calibrate it; this package does not.

## Failure cases

- **Short sequences.** Precision at k is undefined for a sequence shorter than
  k, and for a sequence with nothing to find. 13 of 315 held-out cases were
  excluded for one of those reasons and the count is printed rather than the
  cases being dropped silently.
- **Clean sequences.** On a sequence with no violation the ranking is
  meaningless and the metric is undefined. The gate and the forward walk are
  what answer that case.
- **A dictionary unlike the generator's.** Every feature is scaled in the units
  of the shipped dictionary — watts, megabits per second, seconds. A mission
  whose ceilings or hold durations are orders of magnitude different is outside
  anything measured here, and the model would have to be refitted on that
  mission's own sequences.
- **A defect population unlike the injected one.** The nine families are
  described in `cmdguard.defects`. A defect class not in that list is not
  represented in training and there is no measurement of how the model behaves
  on one.
- **Single cases are noisy.** `examples/risk_ranking.py` shows one defective
  sequence on which the learned ordering and the resource-draw ordering both
  score 0.200000 at k=5 and the flag ordering scores 0.000000. The aggregate is
  the result; one case is an illustration.

## Compute

Two cores, no GPU. From `validation/validate_ranker_output.txt` §1 and §5 and
`validation/validate_determinism_output.txt` §6 — all wall clock, all declared
unstable:

| Step | Measured |
|---|---|
| Corpus build, 70 + 35 seeds | 1.240 s |
| Fit, 200 trees on 22750 rows | 2.560 s |
| Scoring 315 held-out cases, four orderings | 8.623 s |
| Forward walk, 36-step sequence | 0.2658 ms |
| Forward walk, 240-step sequence | 2.3347 ms |

No step in this repository approaches the three-minute budget.

## Reproducibility

```bash
python validation/validate_ranker.py        # every number in this card
python examples/ranker_precision_at_k.py    # the figure, same seeds
```

Seeds: training 0–69, held-out 100000–100034, forest `random_state=20261024`,
bootstrap seed 4711, random-control seed 4711. Two fits with the same seed
produce bit-identical risks and spreads, asserted in
`validation/validate_determinism_output.txt` §5 and in
`tests/test_ranker.py::test_two_fits_with_the_same_seed_agree`.

## Ethical and safety limits

- The model ranks; it never decides. Nothing in this package blocks a sequence
  except the irreversible-command gate, which is a deterministic rule and
  contains no learned component.
- A low risk score is **not** evidence that a command is correct. The ranking
  carries no information the forward walk does not already have; it only
  reorders.
- **A sequence that this package reports clean has not been shown safe to
  uplink.** It has been shown consistent with the declarations it was given.
- This model is not certified for operational flight use.
