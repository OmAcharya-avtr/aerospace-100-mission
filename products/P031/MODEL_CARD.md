# Model card — `hilforge.predict.LearnedOverrunPredictor`

**Product:** P031 HilForge 0.1.0 · **Status:** TESTING
**Validation level:** 3, hardware-pending · **Licence:** AGPL-3.0-or-later
**Date:** 2026-10-04

**This model is not certified for operational flight use.**

Every number in this card comes from `validation/predictor_benchmark.py`, run
in the 0.1.0 build session, with raw stdout in
`validation/predictor_benchmark_output.txt`. Nothing here was estimated,
extrapolated or rounded in the model's favour.

---

## 1. Problem

Given the per-stage execution latencies of a periodic control loop up to and
including iteration `i`, predict whether **any** of iterations `i+1 … i+H`
will exceed the loop's deadline, with `H = 3`.

The intended use is load shedding: an iteration flagged early enough can skip
optional work — a coarser estimator, a skipped telemetry pack, a reduced
filter order — and meet its deadline instead of missing it. The flag is
therefore only useful if it arrives before the overrun and if its false-alarm
rate is low enough that the shed work is not shed constantly.

The window form (`H > 1`) rather than one-step-ahead is deliberate. With
`H = 1`, "lead time" is exactly 1 by construction for every true positive, and
the metric carries no information. With `H = 3` the lead time is measured.

**Label:** `y[i] = 1` if `max(d[i+1], …, d[i+H]) > D`, else 0, where `d` is the
per-iteration total duration [s] and `D` the relative deadline [s].

**Features:** 18 causal features over recent per-stage latencies — the current
and two lagged totals, rolling mean/max/std over 8, rolling mean/max over 32,
an EWMA at α = 0.30, a 4-step slope, the count of overruns in the last 8,
iterations since the last overrun, the deadline headroom and utilisation of the
current iteration, and the four current stage durations. Row `i` is a function
of iterations `≤ i` only, and
`tests/test_predict.py::test_features_are_causal` verifies that by perturbing
iteration 250 and checking that rows 0–249 are byte-identical.

---

## 2. Baselines — implemented first, and both deterministic

### Baseline 1: fixed threshold on the previous iteration

Flag iff `d[i] > θ`. One parameter, no state. `θ` is chosen by sweeping 199
quantiles of the training durations and taking the one that maximises training
F1; the whole sweep is retained in `FixedThresholdPredictor.sweep_` so the
choice can be audited rather than trusted.

Its probability output is the training conditional frequency
`P(y = 1 | d[i] > θ)` on one side and `P(y = 1 | d[i] ≤ θ)` on the other — a
genuine probability estimate that takes exactly two values, which is the
resolution a one-parameter rule has.

### Baseline 2: Markov-modulated stage-latency model

A two-state Markov chain over "calm" and "busy" regimes with a gamma duration
law in each:

1. **Regime labelling** — the training durations are split at the threshold
   that minimises within-class variance: Otsu's method (Otsu 1979, §3), used
   here on a 1-D latency histogram rather than an image, which is the same
   optimisation.
2. **Transition matrix** — maximum likelihood from transition counts,
   `P[i,j] = n_ij / Σ_j n_ij` (Billingsley 1961, Ch. 1).
3. **Duration law per regime** — gamma by method of moments,
   `shape = mean²/var`, `scale = var/mean`. Gamma rather than exponential
   because measured execution times are right-skewed with a tail heavier than
   exponential at small shape (Harchol-Balter 2013, Ch. 20).
4. **Prediction** — with the current regime `s` read off the same Otsu
   threshold, the regime distribution `h` steps ahead is `e_s Pʰ`
   (Chapman–Kolmogorov). `P(no overrun over H)` is taken as
   `Π_{h=1..H} Σ_z π_h[z] F_z(D)`.

**Stated approximation, and it matters:** that product treats successive
durations as independent given their regimes. They are not — the regime
sequence is shared — so the model **underestimates the probability of a run of
overruns**. This is a known bias of the baseline, not a bug, and it is visible
in §7: the Markov baseline flags least and misses most on the autocorrelated
family.

`md1_mean_wait_s` (Pollaczek–Khinchine for M/D/1, Kleinrock 1975 §5.6) is
provided and quoted as a reference point for why high utilisation produces a
long overrun tail. It is **not** used by either predictor, and a periodic
control loop satisfies none of M/D/1's assumptions.

---

## 3. Architecture

`sklearn.ensemble.HistGradientBoostingClassifier` with
`max_iter=160, max_depth=4, learning_rate=0.08, min_samples_leaf=40,
l2_regularization=1.0, early_stopping=False, random_state=20261004`, on the 18
features above.

Histogram-based gradient boosting was chosen because it is the strongest
tabular learner available in the build environment — **PyTorch is not
installed** — and because it fits in under a second on one CPU core, which the
compute budget requires. A small MLP on 18 tabular features would not add
anything a boosted ensemble cannot do.

**Calibration.** The training set is split **chronologically** into a fit slice
(75 %) and a calibration slice (25 %); a random split would leak the
autocorrelation the model exists to exploit. The classifier is fitted on the
fit slice, then an isotonic regression maps its raw probability onto a
calibrated one using the calibration slice (Zadrozny & Elkan 2002; the
underlying monotone regression is pool-adjacent-violators, Ayer et al. 1955).
The decision cut-off is chosen on the calibration slice by maximising F1 — the
same rule used for both baselines, so all three are compared at operating
points chosen the same way.

**Ranking score vs decision score.** `score()` returns the *raw* classifier
probability and `predict_proba()` the calibrated one. AUC is computed from the
raw score because isotonic calibration is a step function that ties large
groups of rows, which depresses AUC for a reason unrelated to ranking quality.
That choice is stated here because it affects a reported number.

---

## 4. Dataset

Full details in `DATASET_CARD.md`. In summary: synthetic per-stage latency
traces from a two-state Markov-modulated gamma model with a Bernoulli
exponential interrupt spike, generated by
`hilforge.predict.data.generate_trace` from a `numpy` PCG64 seed. Two families:

| Family | lag-1 autocorrelation of the overrun indicator | per-iteration overrun rate | utilisation |
|---|---|---|---|
| `jittery` | **0.02** | ~0.15 | 0.70 |
| `bursty` | **0.85** | ~0.15 | 0.79 |

The families are built to differ in exactly one property that matters —
persistence — at matched overrun rates, so a difference in predictor
performance between them is attributable.

**Limitation, stated plainly:** these are model outputs, not measurements. No
trace in this repository was recorded from a real loop, on a workstation or on
hardware. A real trace could have structure neither family captures (periodic
interference at a frequency unrelated to the loop rate, thermal throttling,
cache-state dependence, GC pauses correlated with the control law's own
allocations), and nothing here predicts how the three predictors would rank on
such a trace.

---

## 5. Training procedure

```
trace      = generate_trace(TraceConfig.preset(family, n_iterations=40000), seed=s)
train,test = split_trace(trace, train_fraction=0.6)      # chronological
X, y, idx  = build_dataset(part.stage_s, deadline_s, horizon=3, warmup=32)
```

* Chronological 60/40 split. Never random.
* `warmup=32` rows dropped at the start (the longest feature window) and
  `horizon=3` at the end (labels not yet determined). For a 40 000-iteration
  trace this gives 23 965 training rows and 15 965 test rows.
* No hyperparameter search. The five hyperparameters above were set once, from
  the compute budget and ordinary defaults, and not tuned against the test
  split. There is therefore no selection effect inflating the reported test
  metrics — and equally, no claim that these are the best hyperparameters.
* No resampling, no class weighting. Base rates are 0.17–0.39 depending on
  family and horizon, which is not imbalanced enough to need it.

**Test-split strategy.** The test part is the chronologically last 40 % of a
trace the model never saw. Three seeds (4242, 909090, 31337) give three
independent trace realisations per family, and every number in §7 is the mean
over those three.

---

## 6. Compute

| Quantity | Value |
|---|---|
| Hardware | one shared CPU core on a cloud container, no GPU, four other build agents running concurrently |
| RAM available | 7.8 GiB; peak use for the whole benchmark well under 1 GiB |
| Framework | scikit-learn 1.9.1; **PyTorch unavailable** |
| Total fitting time, all three predictors × 2 families × 3 seeds | **4.2 s** |
| A single learned-model fit (23 965 rows × 18 features, 160 trees) | ~0.5 s |
| Inference | one `predict_proba` over 15 965 rows is milliseconds; no per-iteration latency measurement was made, and none is claimed |

**No inference-latency claim is made.** A load-shedding predictor has to run
inside the loop it protects, so its own per-iteration cost is the first thing a
real deployment would need to know — and it is exactly the kind of number that
cannot honestly be produced on a contended shared core. It is not in this card
because it would not be worth anything.

---

## 7. Metrics — measured, as they came out

Mean over three seeds. `H = 3`, 15 965 test rows per seed per family. Operating
points chosen on the training set only. Full tables, per-seed, in
`validation/predictor_benchmark_output.txt`.

### 7a. The jittery family — nothing works

| predictor | false-alarm | missed | lead [it] | F1 | Brier | **ROC AUC** |
|---|---|---|---|---|---|---|
| fixed_threshold | 0.9887 | 0.0095 | 1.886 | 0.5604 | 0.23796 | **0.5041** |
| queueing_markov | 0.2867 | 0.7100 | 1.875 | 0.3337 | 0.24117 | **0.5016** |
| learned_hgb | 0.9935 | 0.0069 | 1.884 | 0.5604 | 0.23853 | **0.4966** |

*(F1-maximising operating point.)*

All three AUCs are within 0.01 of 0.50. **None of the three can rank the
iterations**, and the learned model is, by a hair, the worst of the three. That
is the correct answer for a family whose overrun indicator has lag-1
autocorrelation 0.02: the recent past does not contain the information, so no
amount of modelling extracts it.

The F1-maximising point degenerates to flagging ~99 % of iterations. With a
39 % positive rate under `H = 3`, flagging everything maximises F1. That is
reported rather than hidden; it is what an F1 criterion does on a high base
rate with no signal.

At a matched 15 % flag rate on the same family the learned model lands at an
achieved 5.1 % (its calibrated probability is too coarse to hit 15 %) and
misses 95.0 % of windows; the threshold achieves 15.1 % and misses 84.4 %. For
reference, flagging 15 % of iterations at random on a 39 % base rate misses
about 85 %. **The threshold is at the random-guessing level and the learned
model is below it.**

### 7b. The bursty family, matched flag rate (target 0.15) — the deciding table

| predictor | false-alarm | **missed** | lead [it] | F1 | **Brier** | AUC | achieved flag |
|---|---|---|---|---|---|---|---|
| fixed_threshold | 0.0127 | 0.3377 | 1.019 | 0.7701 | 0.06862 | 0.8558 | 0.1364 |
| queueing_markov | 0.0106 | 0.3219 | 1.017 | 0.7865 | 0.06983 | 0.8337 | 0.1378 |
| learned_hgb | 0.0098 | **0.3202** | 1.017 | 0.7892 | **0.05775** | **0.8583** | 0.1372 |

Also in the README's table, and in `validation/VALIDATION.md` §9, so the three
documents cannot drift apart.

**Missed-overrun rate:** learned 0.3202 against the threshold's 0.3377, a gap
of **0.0175**, against a typical 95 % Wilson half-width on that rate of
**0.0168**. The gap is marginally larger than the interval around either
number. That is a real but small improvement, the same order as its own
uncertainty, and not a decisive one.

**Brier score:** learned 0.05775 against 0.06862, a **15.8 % reduction**. This
is the clearer win and it is a win about calibration, not about decisions.

**ROC AUC:** learned 0.8583 against 0.8558 — a 0.0026 improvement on the
unrounded means (0.0025 as the table rounds), the smallest of its three
advantages.

**The queueing baseline is the most conservative of the three** — it flags
least (0.1378 achieved at a 0.15 target, and it cannot do better because its
score has two levels) and misses most. That is §2's stated independence
approximation doing exactly what it was said it would do.

### 7c. Lead time

~1.02 iterations on the bursty family, ~1.88 on the jittery one, for every
predictor.

This is not the bursty predictors warning late. Inside a burst the *next*
iteration is usually the one that misses, so a correct flag has a lead time of
1 by construction, and 1.02 means the flag is almost always about iteration
`i+1`. A lead time near the horizon mean (2.0 for `H = 3` over a uniform
position) is what a flag looks like when it carries no information about
*which* iteration will miss — the jittery flags are right that something will
overrun in the window and have no idea when.

For load shedding, one iteration of lead is the minimum useful amount and it is
what this gets you on the family where prediction works at all.

### 7d. What dominates the learned model

Permutation importance (Brier increase when a column is shuffled), bursty
family, seed 4242, held-out set:

| feature | ΔBrier |
|---|---|
| `total_max8_s` | **+0.044492** |
| `total_last_s` | +0.003730 |
| `control_last_s` | +0.001819 |
| `headroom_frac` | +0.001202 |
| `total_lag1_s` | +0.000698 |
| `total_mean32_s` | +0.000289 |
| `total_max32_s` | +0.000243 |
| `iters_since_overrun` | +0.000217 |

One feature is an order of magnitude above the next. The rolling maximum over
the last eight iterations is where the autocorrelation lives — which is what
the bursty family was built to have. **Seventeen of the eighteen features are
nearly idle**, and a two-feature model would probably do almost as well. That
is another argument against the ensemble rather than for it.

### 7e. The honest summary

**On the jittery family the fixed threshold is not beaten, because nothing
beats anything there.** All three predictors are at chance. This is the outcome
the specification expected, and it is kept as measured.

**On the bursty family the learned model is the best of the three**, by a margin
that is clear on calibration (−15.8 % Brier) and ranking (+0.0026 AUC) and
*within interval* on the metric that would justify deploying it
(missed-overrun rate at a matched flag rate).

**The recommendation this implies.** Do not put the learned model in a loop in
place of the threshold. An 18-feature boosted ensemble — effectively a
one-feature ensemble, per §7d — is a large amount of machinery for a
within-interval win on one synthetic trace family, and it carries an
unmeasured per-iteration inference cost into the loop it is supposed to
protect. The threshold has one parameter and can be read in an afternoon. The
defensible use of the learned model is its calibrated probability: if what you
need is "how likely is an overrun in the next three iterations", it is the only
one of the three that answers usefully. If what you need is "flag it or not",
use the threshold.

---

## 8. Uncertainty output

`predict_proba` returns the isotonically calibrated probability of an overrun
within the horizon. That **is** the uncertainty output, and it is held to a
falsifiable standard rather than asserted:

* **Brier score** on held-out data, reported in every table above.
  `tests/test_predict.py::test_learned_model_beats_a_constant_on_the_bursty_trace`
  requires it to beat the base-rate constant, which is the minimum bar for a
  probability to mean anything.
* **Reliability table**, `LearnedOverrunPredictor.reliability_table`, giving
  mean predicted against observed frequency per decile on held-out data. The
  table for the bursty family, seed 4242, is in
  `validation/predictor_benchmark_output.txt`; both families are plotted in
  `screenshots/overrun_predictor.png` (bottom right).

**Two limits on that output.** First, on the jittery family the reliability
curve is flat: the probability is well calibrated to the base rate and carries
no discrimination, which is a correctly calibrated useless forecast. Brier and
reliability together do not distinguish "calibrated and informative" from
"calibrated and uninformative" — AUC does, and it is 0.4966 there. Read all
three or none.

Second, the calibrated output is **coarse**: isotonic regression is a step
function, and on the jittery family it takes so few distinct values that no
cut-off gives a 15 % flag rate (the nearest achievable is 5.1 %). The output is
a probability, not a continuum.

---

## 9. Failure cases

1. **No autocorrelation in the overrun process.** The jittery family. AUC
   0.4966; the model learns nothing and there is nothing to learn. A user whose
   real trace looks like this should use the threshold and move on.
2. **Coarse calibrated output.** As above: the model cannot be placed at an
   arbitrary flag rate when its isotonic steps are wide. The achieved rate is
   always reported next to the target, in the metrics and in the CLI, so this
   failure is visible rather than silent.
3. **Single-feature dependence.** §7d. Shuffling `total_max8_s` costs
   +0.044 Brier; shuffling any other feature costs ≤ +0.004. The model is
   fragile to anything that changes the meaning of a rolling 8-iteration
   maximum — a different period, a different stage decomposition, a different
   sampling rate.
4. **No transfer claim across periods.** Two features (`headroom_frac`,
   `util_last`) are normalised by the deadline so the model is not nonsense at
   another period, but **no claim is made that it transfers**, and no
   cross-period evaluation was run. A model trained at 100 Hz should be
   retrained for 1 kHz.
5. **Horizon mismatch.** The label horizon must match
   `QueueingOverrunPredictor(horizon=…)` and the window matrix passed to
   `evaluate_predictor`. A mismatch produces a silently wrong lead time. The
   API requires `horizon` explicitly at every point rather than defaulting it
   globally, which makes the mismatch a visible argument rather than a hidden
   state.
6. **Regime degeneracy in the queueing baseline.** If the Otsu split puts
   every training sample in one class, the gamma fit for the empty regime falls
   back to the whole sample and the transition matrix row defaults to 0.5/0.5.
   The model still produces probabilities; they just carry no regime
   information. This is handled rather than crashed on, and
   `QueueingOverrunPredictor.describe()` prints the per-regime sample counts so
   it can be seen.
7. **Single-class fit slice.** `LearnedOverrunPredictor.fit` raises
   `ConfigurationError` if the fit slice has one class, rather than training a
   constant and reporting a perfect-looking score.
8. **Not evaluated on real hardware traces at all.** Every trace is synthetic.
   This is the failure case with the widest consequences and it is the reason
   this product is Level 3, hardware-pending.

---

## 10. Ethical and safety limits

This predictor decides whether a control loop sheds optional work. In a real
vehicle that decision is inside the control path, and a false negative means a
missed deadline in a loop that someone assumed would meet it.

* **This model is not certified for operational flight use.**
* It is research-grade. It is not flight-qualified, not certified, and not
  approved for operational aerospace use.
* It has never been evaluated on a trace from real hardware, so its measured
  false-alarm and missed-overrun rates do not bound its behaviour on one.
* Its own inference cost inside the loop was not measured, so the obvious
  question — does running the predictor cost more deadline margin than it
  saves — is open.
* On the evidence in §7 the recommendation is to use the one-parameter
  threshold for the decision. That recommendation is in this card rather than
  only in a footnote because the measured result supports it.
* Load shedding changes what the loop computes. Which work is optional is a
  systems-engineering decision, not a modelling one, and nothing in this
  package makes it.

---

## 11. Reproducibility

Exact commands and seeds. All three predictors, both families, three seeds:

```bash
cd products/P031
pip install -e ".[dev]"

# the full benchmark behind every number in §7
PYTHONPATH=src python3 validation/predictor_benchmark.py \
  > validation/predictor_benchmark_output.txt

# one family, one seed, from the CLI
PYTHONPATH=src python3 -m hilforge predict \
  --preset bursty --iterations 40000 --seed 4242 --horizon 3 --flag-rate 0.15

# the figure
PYTHONPATH=src python3 examples/overrun_predictor.py
```

| Knob | Value |
|---|---|
| trace seeds | 4242, 909090, 31337 |
| trace length | 40 000 iterations per family per seed |
| period / deadline | 0.010 s, implicit deadline `D = T` |
| horizon `H` | 3 iterations |
| split | chronological, `train_fraction=0.6` |
| warmup rows dropped | 32 |
| model seed (`random_state`) | 20261004 |
| model | `HistGradientBoostingClassifier(max_iter=160, max_depth=4, learning_rate=0.08, min_samples_leaf=40, l2_regularization=1.0, early_stopping=False)` |
| calibration | isotonic, trailing 25 % of training rows, chronological |
| cut-off rule | F1-maximising on the calibration slice, or `calibrate_flag_rate(X_train, 0.15)` |
| library versions | numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, Python 3.13.16 |

The trace generator is deterministic from its seed (`numpy` PCG64), and
`tests/test_regression.py::test_pinned_trace_statistics` pins the mean duration
and both overrun counts of a 5000-iteration trace from each family to 14
significant figures, so a change in the generator surfaces as a test failure
rather than as a quietly different benchmark.

---

## 12. References

- Ayer, M., Brunk, H. D., Ewing, G. M., Reid, W. T. and Silverman, E. (1955).
  "An Empirical Distribution Function for Sampling with Incomplete
  Information." *Annals of Mathematical Statistics* 26(4):641-647.
- Billingsley, P. (1961). *Statistical Inference for Markov Processes.*
  University of Chicago Press.
- Breiman, L. (2001). "Random Forests." *Machine Learning* 45(1):5-32.
- Brier, G. W. (1950). "Verification of Forecasts Expressed in Terms of
  Probability." *Monthly Weather Review* 78(1):1-3.
- Fischer, W. and Meier-Hellstern, K. (1993). "The Markov-modulated Poisson
  process (MMPP) cookbook." *Performance Evaluation* 18(2):149-171.
- Harchol-Balter, M. (2013). *Performance Modeling and Design of Computer
  Systems.* Cambridge University Press.
- Kleinrock, L. (1975). *Queueing Systems, Volume 1: Theory.* Wiley.
- Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms."
  *IEEE Transactions on Systems, Man, and Cybernetics* 9(1):62-66.
- Ross, S. M. (2014). *Introduction to Probability Models*, 11th ed. Academic
  Press.
- Wilson, E. B. (1927). "Probable Inference, the Law of Succession, and
  Statistical Inference." *Journal of the American Statistical Association*
  22(158):209-212.
- Zadrozny, B. and Elkan, C. (2002). "Transforming Classifier Scores into
  Accurate Multiclass Probability Estimates." *KDD '02*, 694-699.
