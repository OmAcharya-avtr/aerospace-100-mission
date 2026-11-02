# Model card — `hilverdict` deadline-miss consequence classifier

**Product:** P083 `hilverdict` · **Version:** 0.1.0 · **Status:** TESTING ·
**Validation level:** Level 3

**This model is not certified for operational flight use.**

Every number on this page comes from `validation/validate_learned_model.py`
and `validation/validate_negative_controls.py`, run in the session that wrote
the page, with the raw output committed beside each script. Nothing here is
quoted from an earlier run.

## The problem

A hardware-in-the-loop run records which controller iterations overran their
deadline. The question is which of those overruns measurably changed the
trajectory. The exact answer is one plant integration per overrun, which is
affordable for one run and not for a campaign. The learned component is for
the case where it is not affordable.

**The decision is asymmetric.** Dismissing an overrun that mattered publishes
a hardware-in-the-loop result whose timing fidelity was never established;
re-simulating one that did not costs some seconds of compute. So the model is
scored on **recall of consequential misses** first and on how much
re-simulation it saves second. Accuracy is not reported anywhere in this
product: the positive class is 5.6 per cent of the sample and accuracy would
reward dismissing everything.

## The baselines, implemented and validated first

Both baselines existed and were validated before any learned component was
written, and the learned component is benchmarked against them on the same
held-out runs.

- **Baseline 1 — the analytic certificate.** A closed-form upper bound on the
  divergence a miss of duration `tau` holding an input step `du` can cause in
  the declared linearisation (`hilverdict.certificate`, equation 13). A
  dismissal is a theorem, not an estimate. Validated in
  `validation/validate_certificate.py` and against four independent routes in
  `validation/validate_independent_bound.py` before the model was fitted.
- **Baseline 2 — direct re-simulation.** The declared nonlinear plant run with
  and without the miss (`hilverdict.resimulate`). This is the reference that
  produces every label on this page; it is not a baseline the model competes
  with, it is the truth the competition is scored against.

Two **trivial** baselines are also reported, because anything claiming to be
useful has to beat them: flag every miss, and threshold the overrun alone.

## Architecture

`sklearn.ensemble.GradientBoostingClassifier` with the logistic loss (Friedman
2001, "Greedy function approximation: a gradient boosting machine", *Annals of
Statistics* 29(5), 1189-1232). PyTorch is not installed in the measured
environment and is not a dependency of this package.

| hyperparameter | value |
| --- | --- |
| `n_estimators` | 150 |
| `learning_rate` | 0.08 |
| `max_depth` | 3 |
| `min_samples_leaf` | 4 |
| `subsample` | 0.9 |
| `positive_weight` (sample weight on the consequential class) | 8.0 |
| `random_state` | 20260301 |
| `abstention_band` | 0.05 |

**These are declared, not tuned. There is no hyperparameter search anywhere in
this repository.** A search scored on the same held-out runs that then report
the result would make the reported result a selection artifact, and the honest
alternative to a search with a third split is to declare the values and report
what they give.

## Features

Fourteen, all read off the log, none costing a simulation:

`overrun_periods`, `overrun_s`, `hold_periods`, `input_step_rad`,
`abs_command_rad`, `abs_alpha_rad`, `abs_rate_rad_s`, `tracking_error_rad`,
`remaining_fraction`, `command_activity_rad`, `misses_in_window`,
`consecutive_misses`, `local_utilisation`, `steps_to_switch`.

Two of them — `overrun_s` and `input_step_rad` — are **exactly** the two
numbers equation (13) is a function of. That is deliberate: the model is given
everything the proof uses and twelve further features besides, so if it loses
it loses on the merits.

## Dataset

Synthetic, from `hilverdict.synthetic`; full description in
[`DATASET_CARD.md`](DATASET_CARD.md). Labels come from baseline 2: a miss is
consequential when the re-simulated peak divergence reaches the declared
tolerance.

| split | seeds | runs | misses | consequential | rate |
| --- | --- | --- | --- | --- | --- |
| train | 1000-1071 | 72 | 1415 | 102 | 0.072085 |
| test | 2000-2044 | 45 | 841 | 47 | 0.055886 |

**The split is by RUN, never by miss.** Misses inside one run share its
trajectory and its timing trace and are not independent of each other; a split
by miss would leak. The soundness population used in
`validate_certificate.py` (seeds 3000-3044) is disjoint from both.

## Training procedure

One fit on the 1415 training misses, with the consequential class upweighted
by the declared factor of 8. No early stopping, no cross-validation, no
calibration step. The decision threshold is chosen on the **training** runs,
as the smallest flag rate reaching the declared recall target of 1, and
applied unchanged to the test runs.

That protocol is the single most important methodological point on this page.
An earlier revision of the validation script chose the threshold using the
test labels; it reached a recall of 1 at a flag rate of 0.0722 against the
certificate's 0.1193 and looked like a win. It reaches recall 1 **by
construction**, because the threshold was selected to do so on the data it is
scored on. The leaked version is retained in the results table, labelled as an
oracle no user can have, rather than deleted.

## Metrics

Primary: **recall of consequential misses**, with the exact Clopper-Pearson
binomial interval (Clopper & Pearson 1934, *Biometrika* 26(4), 404-413).
Secondary: flag rate, which is the fraction of misses sent to re-simulation.
Reported but never decisive: Brier score and calibration.

## Results, on 841 held-out misses

| method | recall | 95 % interval | consequential dismissed | flag rate | proof |
| --- | --- | --- | --- | --- | --- |
| **analytic certificate, eq (13)** | **1.0000** | [0.9245, 1.0000] | **0** | 0.1153 | **yes** |
| learned, threshold from train at recall 1 | **0.7234** | [0.5736, 0.8438] | **13** | 0.0428 | NO |
| learned, threshold from train at the certificate's flag rate | 1.0000 | [0.9245, 1.0000] | 0 | 0.1296 | NO |
| learned at the untuned 0.5 | 0.8936 | [0.7690, 0.9645] | 5 | 0.0595 | NO |
| learned with abstention referred | 0.7447 | [0.5965, 0.8606] | 12 | 0.0440 | NO |
| learned, ORACLE threshold from the test labels | 1.0000 | [0.9245, 1.0000] | 0 | 0.0654 | NO |
| trivial: flag every miss | 1.0000 | [0.9245, 1.0000] | 0 | 1.0000 | NO |
| trivial: overrun threshold from train at recall 1 | 0.9574 | [0.8546, 0.9948] | 2 | 0.6445 | NO |

**The analytic bound wins outright.** At the only threshold a user could
actually have chosen, the learned classifier dismisses 13 of 47 consequential
misses — the asymmetric error — while the certificate dismisses none and, for
the declared linearisation, cannot.

Two further measurements make the difference concrete rather than rhetorical.

**Stability.** Split the test population into three disjoint folds of 15 runs
and score both at one fixed threshold:

| fold | misses | consequential | certificate recall | learned recall |
| --- | --- | --- | --- | --- |
| 0 | 261 | 16 | 1.000000 | 0.812500 |
| 1 | 272 | 17 | 1.000000 | 0.588235 |
| 2 | 308 | 14 | 1.000000 | 0.785714 |

The learned recall moves by **0.224265** between folds. The certificate's does
not move, because it is not a statistic.

**The residual set — the one role the specification allows.** A model fitted
only on the training runs' undecided misses and scored only on the test runs':

| quantity | value |
| --- | --- |
| test residual misses | 97, of which **47** consequential (rate 0.484536) |
| threshold from the train residual | recall **0.765957** at flag rate 0.402062 |
| consequential misses it dismissed | **11** |
| oracle threshold from the test labels | recall 1.000000 at flag rate 0.670103 |
| **re-simulations saved at no cost in recall** | **0.000000** |

On the residual set the learned component saves nothing a user is entitled to
take: the saving is bought with exactly the error the scoring rule forbids.
The residual is where the certificate has already removed every easy case, and
what is left is 48.5 per cent consequential against a population rate of 5.6
per cent. A classifier asked for perfect recall on a set that is half positive
has little to work with, and the measurement says so.

## Cost

| policy | plant evaluations | fraction | consequential dismissed |
| --- | --- | --- | --- |
| re-simulate every miss | 1415712 | 1.000000 | 0.055886 as a rate |
| **certificate, then re-simulate the residual** | **185120** | **0.130761** | **0.000000** |

The certificate saves **0.869239** of the reference's cost at zero cost in
recall. The learned component is not on this table because it does not have an
operating point that saves anything without dismissing a consequential miss.

## Uncertainty and confidence output

`MissClassifier.predict_with_uncertainty()` returns the predicted probability,
the decision at the current threshold, the signed margin to that threshold,
and an **abstention** flag for probabilities inside a declared band around it.
There is no point-prediction method on the class: a bare label would hide the
margin, and the margin is what makes an asymmetric decision rule possible. An
abstention is a referral to re-simulation, never a dismissal.

Calibration is measured, not assumed: Brier score **0.010345** against the
base-rate predictor's **0.052763**, worst calibration gap over bins
**0.693204** in a bin holding a handful of misses. A Brier score better than
the base rate says the probabilities carry information. It does not say the
model is useful for this decision, which is what the results table measures.

## Failure cases

- **It dismisses consequential misses.** 13 of 47 at the train-chosen
  threshold, 5 of 47 at the untuned 0.5, 12 of 47 with the abstention band
  active, 11 of 47 on the residual set.
- **Its recall is unstable across folds**, by 0.224265 at one fixed threshold.
- **Its threshold does not transfer.** The threshold that reached recall 1 on
  72 training runs reaches 0.7234 on 45 held-out ones.
- **It learned the certificate's own inputs and nothing else.** The three
  largest impurity importances are `input_step_rad` **0.779548**, `overrun_s`
  **0.112442** and `overrun_periods` **0.076083**; the other eleven features
  sum to about 0.032. There was nothing else to find, which is the honest
  reading of why it cannot beat a closed form over the same two quantities.
- **It is useless with one class.** `fit` raises on a single-class label set
  rather than returning a model that predicts that class with probability 1.

## Negative controls on the learned pipeline

From `validation/validate_negative_controls.py`, scored as the flag rate each
model needs to reach recall 1 on held-out runs:

| control | declared | measured |
| --- | --- | --- |
| shuffling the training labels makes the model need a far larger flag rate | must fire | **0.901961** against the honest model's **0.058824** |
| blinding the model to `tau` and `du` makes it need a larger flag rate | must fire | **0.124183** against **0.058824** |
| fitting on a single-class label set raises | must fire | fired |
| asking for a recall above 1 raises | must fire | fired |

The first two were rewritten during the build. They originally compared recall
at a fixed 0.25 flag rate, where all three models reach recall 1 on this
population and the comparison is saturated, so the blinding control could not
fire. A control that cannot fire is a failed control, and it was replaced
rather than accepted.

## Reproducibility

```bash
python3 validation/validate_learned_model.py
python3 validation/validate_negative_controls.py
```

Seeds are in `validation/_bootstrap.py`: `TRAIN_SEEDS = range(1000, 1072)`,
`TEST_SEEDS = range(2000, 2045)`, `SOUNDNESS_SEEDS = range(3000, 3045)`. The
model's own seed is `ClassifierSpec.seed = 20260301`. No model artifact is
serialised or committed: every model is refitted from these seeds by the
script that reports it, so there is no pickle whose format would need version
bounding, and every run pays the fit cost.

## Compute

6.27 s wall clock for the whole learned study on 2 cores, of which 4.40 s is
generating and re-simulating the 2256 labelled misses -- 2.79 s for the
training population and 1.61 s for the test one, both printed in the script's
own output -- and the remainder is the two fits. No GPU. Peak resident memory **165.9 MB**, measured with
`resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` on the same script. The
wall-clock figures are unstable on shared hardware; the plant-evaluation
counts in the cost table are exact.

## Ethical and safety limits

- **This model is not certified for operational flight use.**
- It is trained and scored entirely on synthetic runs from a declared plant
  whose coefficients are a fiction with an engineering shape. Nothing here
  establishes behaviour on a real rig.
- It must not be used to dismiss a deadline miss in any setting where being
  wrong matters. The measured result is that it dismisses consequential misses
  at every operating point a user could actually select, and the analytic
  certificate in the same package does not.
- Its probabilities are calibrated on this synthetic population only. A
  probability it reports on a different plant, a different controller or a
  different metric means nothing.
