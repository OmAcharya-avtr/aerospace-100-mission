# Model card — telemdrift learned detector 0.1.0

**This model is not certified for operational flight use.**

Every number in this card was produced by a script in `validation/`, executed in
this container on 2026-10-10, with its raw output committed under
`validation/outputs/`. The two scripts that produce it are
[`validation/validate_learned.py`](validation/validate_learned.py) (the model's
own metrics) and
[`validation/validate_change_types.py`](validation/validate_change_types.py)
(the stream-level comparison against the analytic baselines).

## Problem

Given a trailing window of 50 standardised samples from a univariate telemetry
channel, score the proposition *"a change occurred in or just before this
window"*. The score is then thresholded to produce a streaming change detector,
which is scored on detection delay at a declared false-alarm rate — the same way
the five analytic detectors in this package are scored.

The model is not asked to classify the *kind* of change, to localise the change
point, or to output a probability a downstream system will act on numerically.
It is asked for a monotone ordering.

## Baseline, implemented first

Five analytic detectors — CUSUM, Page-Hinkley, EWMA, a windowed two-sample
Kolmogorov-Smirnov test, and an ADWIN-style adaptive window — were implemented,
validated against a published reference table (NIST handbook; see
`validation/VALIDATION.md` section 2) and calibrated to a measured ARL0 of 500
samples **before** any feature was extracted or any model fitted. The comparison
below is against them, at equal measured ARL0, on the same seeded streams.

## Architecture

`sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=8,
class_weight="balanced_subsample", n_jobs=1, random_state=58000)` on eight
features of a trailing 50-sample window.

PyTorch is not available in this build container and the container has two
cores, so a deeper model was never on the table. Nothing measured here suggests
one would help: the loss below is structural, not a capacity problem, and the
model already overfits at the window level with the capacity it has.

`n_jobs` is 1 deliberately. `predict_proba` on a single row measured **4.58x
slower** with `n_jobs = 2` than with `n_jobs = 1` on this container
(`validation/outputs/validate_cost.txt` section 4), reproducing a defect an
earlier session in this portfolio recorded, and the per-sample online cost is a
number this product publishes.

### Features

Eight, all dimensionless, all causal functions of the trailing window
(`telemdrift.features`):

| # | Feature | What it is for | Measured importance |
|---|---|---|---|
| 0 | window mean | the CUSUM/EWMA level statistic | **0.2108** |
| 4 | least-squares slope per sample | drift-ramp evidence | 0.1661 |
| 7 | range (max − min) | scale, maximally outlier-sensitive | 0.1514 |
| 1 | standard deviation | variance-change evidence | 0.1223 |
| 2 | second-half mean − first-half mean | localised level change | 0.1090 |
| 3 | log ratio of half-window std devs | variance-step evidence | 0.0891 |
| 6 | interquartile range | scale, robust to one outlier | 0.0862 |
| 5 | lag-one autocorrelation | dependence and transient shape | 0.0652 |

Features 6 and 7 were included as a pair on purpose: a transient spike moves the
range and leaves the IQR alone, a variance step moves both, so the pair is the
only information in the window that could separate them. Section "Transient
behaviour" below measures whether the model uses it. It does not use it enough.

## Dataset

See [DATASET_CARD.md](DATASET_CARD.md). In short: entirely synthetic, generated
by committed seeded scripts, never any real spacecraft telemetry.

| | Training | Held-out test |
|---|---|---|
| Seeds | 58001–58008 (8) | 58051–58056 (6), **disjoint** |
| Windows | 78 104 | 58 578 |
| Positive rate | 3.58 % | 3.58 % |
| Change specs | 7: mean step 0.5/1.0/2.0 σ, variance ×1.8/×3.0, drift 0.01/0.03 σ per sample | same |
| Negatives | stationary streams only | same |

### The split is by seed, and that is the most important choice here

Consecutive windows overlap by 49 of their 50 samples, so neighbouring rows are
near-copies. A random **row** split would put near-identical rows on both sides
and report a leaked number. The split is therefore by **seed**: a held-out
stream shares no sample with any training stream.

### Transients are held out of training

Transient-spike streams are **not** in the training set. The transient
experiment is the negative control for all six detectors, and giving the learned
detector supervision no analytic detector can receive would make the comparison
meaningless. A variant *with* transients in training is measured separately and
reported below.

## Training procedure and compute

```bash
python validation/validate_learned.py
```

Deterministic given the seeds: the training features regenerate bit-for-bit and
the refitted model produces identical held-out scores (both checked,
`validation/outputs/validate_learned.txt` section 5).

| | Measured |
|---|---|
| Training-set generation | 0.3 s |
| Fit | **12.6 s** on two cores |
| Hyperparameter search | **none** — the architecture was fixed before the first measurement and was not tuned against any result |
| Peak memory | well inside 7.8 GiB; the feature matrix is 78 104 × 8 float64 ≈ 5 MB |

Wall clock in this container moves 10-40 % between runs.

## Metrics

### Window-level, held-out

| Split | ROC-AUC | Average precision | Brier | n |
|---|---|---|---|---|
| train | 0.9411 | 0.6342 | 0.09157 | 78 104 |
| **held-out** | **0.8229** | **0.3912** | 0.10382 | 58 578 |

Average precision is the metric to read: the positive class is 3.58 % of windows
and ROC-AUC flatters a classifier on an imbalanced problem. No-skill average
precision equals the positive rate, so 0.3912 is **10.9x no-skill**.

### Stream-level, at equal measured ARL0 — this is the comparison that matters

ARL0: learned 543 ± 18 samples; analytic detectors 468 to 581. 300 seeded
replicates per cell, steady-state convention.

| Change | Best analytic | Learned | Loss | Ratio | Significance |
|---|---|---|---|---|---|
| mean step +1.0 σ | EWMA **9.2 ± 0.3** | 17.4 ± 2.8 | +8.2 samples | 1.88x | +2.9 σ |
| mean step +0.5 σ | EWMA **31.4 ± 1.4** | 45.5 ± 3.5 | +14.1 samples | 1.45x | +3.7 σ |
| variance ×2.0 | CUSUM **12.5 ± 0.7** | 16.5 ± 1.6 | +3.9 samples | 1.31x | +2.2 σ |
| drift 0.02 σ/sample | EWMA **33.2 ± 0.6** | 34.0 ± 0.8 | +0.8 samples | 1.02x | +0.8 σ |

**The learned detector loses on three of four change types at more than two
combined standard errors, ties on the fourth, and wins on none.** This result is
published as measured. It was not retuned.

### Why the baseline wins

Not an implementation accident. On an i.i.d. Gaussian stream with a mean shift
of known size, the CUSUM statistic *is* the sequential log-likelihood ratio: it
is the optimal detector for that problem in Lorden's minimax sense, and EWMA is
close to it. There is no information in a 50-sample window of such a stream that
a sufficient statistic has not already used, so the best a learned detector can
do is recover the same decision boundary — and it pays for the window twice:

1. **In delay.** The window must fill before the features move. A 50-sample
   window cannot respond in 9 samples the way a recursion with unbounded memory
   can.
2. **In cost.** One forest call per sample, measured at 7574 µs/sample online
   against 0.35 µs/sample for CUSUM with this scikit-learn API on this
   container.

A learned detector should be expected to win where the analytic model is wrong —
on a non-Gaussian, autocorrelated, multi-modal real channel. This package does
not have one, and says so rather than implying the result would transfer.

## Uncertainty and confidence output

The model exposes `LearnedDetector.last_score`, the forest's vote fraction, as
its confidence output. **It is measured to be badly miscalibrated** and nothing
in this package calls it a probability.

| bin | n | mean score | observed frequency | gap |
|---|---|---|---|---|
| [0.1,0.2) | 19 467 | 0.1534 | 0.0090 | −0.1444 |
| [0.5,0.6) | 3 070 | 0.5463 | 0.0612 | −0.4851 |
| [0.6,0.7) | 1 729 | 0.6430 | 0.1215 | **−0.5215** |
| [0.9,1.0) | 367 | 0.9381 | 0.8093 | −0.1289 |

Worst absolute gap **0.522**. A class-weighted forest on a 3.6 %-positive problem
produces an ordering, not a probability. The package uses it only as an
ordering, with its threshold moved until the measured ARL0 matches a declared
target — which requires no calibration at all. A reader who needs a calibrated
probability should recalibrate it; a sibling product in this portfolio
(P056 CalibAudit) does exactly that, and is not imported here.

## Failure cases, measured

1. **It overfits at the window level.** Train ROC-AUC 0.941, held-out 0.823, a
   gap of **+0.118** against a pre-declared expectation of under 0.05. This is a
   FAILED validation check, recorded as such. The split is by seed, so it is
   capacity, not leakage. Not retuned.
2. **It fires on transients.** 94.4 % of +4 σ, 20-sample excursions that are not
   changes, against a matched stationary baseline of 13.2 % — an attributable
   excess of **+81.2 pp**. It does take longer to fire than the analytic
   detectors (43.2 samples against CUSUM's 0.9), which is the only visible
   discrimination anywhere, and it is not enough to be useful.
3. **Explicit supervision barely helps.** Retrained with transient streams as
   negatives and recalibrated to the same ARL0 (519.5 ± 16.5), its transient
   firing rate falls from 94.4 % to **90.8 %**, a 3.6 pp improvement in the one
   thing the extra data was for.
4. **Its trade-off curve is non-monotone.** Its shortest measured delay (12.6
   samples) is at ARL0 = 289, not at its tightest threshold (16.8 samples at
   ARL0 = 96). Making it more sensitive makes it slower, because its own false
   alarms put it back into a 50-sample refractory period.
5. **It has never seen autocorrelated data.** Every detector in this package
   degrades under autocorrelation; the learned detector was additionally
   *trained* on i.i.d. data, so it has had no opportunity to learn the one thing
   that might distinguish it on a real channel.
6. **Online cost.** 7574 µs/sample with this scikit-learn API on this container,
   about four orders of magnitude above the analytic recursions. The Monte Carlo
   uses a batch score path that is 1113x cheaper and bit-identical (checked); a
   streaming deployment does not get that path.

## Reproducibility

```bash
# The model's own numbers, and the failed generalisation check.
python validation/validate_learned.py

# The stream-level comparison at equal ARL0.
python validation/validate_change_types.py

# The transient experiment, including the transient-trained variant.
python validation/validate_transient.py

# The trade-off curve, including the learned detector's sweep.
python validation/validate_tradeoff.py
```

Seeds: training 58001–58008, test 58051–58056, forest `random_state=58000`,
calibration 58101–58106, evaluation 58201–58208, ARL1 from 58400. All are in
`telemdrift.benchmark.BenchmarkConfig` and `telemdrift.learned`, not scattered
through scripts.

## Ethical and safety limits

- **This model is not certified for operational flight use.**
- It is trained and evaluated entirely on synthetic data it generated itself. No
  real spacecraft telemetry, no operator data, no personal data of any kind.
- It loses to a 1954 control chart on the problem it was trained for, at equal
  false-alarm rate. Any deployment decision should start from that.
- An alarm from this model means a vote fraction crossed a threshold. It does
  not mean a spacecraft is unwell, and the same alarm is raised by a transient
  excursion that resolves itself.
- The threshold is only meaningful at the false-alarm rate it was calibrated to,
  on a channel with the statistics it was calibrated on. Applied to an
  autocorrelated channel without recalibration, every detector in this package
  false-alarms far more often than its nominal rate; the measured factors are in
  `validation/VALIDATION.md` section 8.1.
