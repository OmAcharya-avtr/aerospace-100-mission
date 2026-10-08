# Model card — twininvalidate drift classifier 0.1.0

**This model is not certified for operational flight use.**

The headline result of this card is that the model it documents **should not
be deployed**, and the measurements that say so are below. It is published
because the measurement is the product, not the model.

---

## 1. Problem

Given a stream of normalised residuals from a declared linear-Gaussian digital
twin, raise an alarm as soon as possible after the asset stops obeying the
twin, subject to a declared in-control false-alarm rate.

The model is a binary classifier over a sliding window of residuals:
`P(the twin no longer describes the asset | the last 50 residuals)`. It is
turned into a monitor by thresholding that probability, exactly as the
analytic statistics are thresholded.

Inputs: 50 consecutive dimensionless normalised residuals `z`, reduced to nine
features. Output: one calibrated probability in `[0, 1]`.

## 2. Baseline

Three analytic sequential detectors, implemented **first** and benchmarked on
the same held-out streams at a **matched in-control ARL0**:

| Baseline | Source | Declared design constant |
|---|---|---|
| Two-sided tabular CUSUM | Page, E. S. (1954), *Biometrika* 41(1/2), 100–115 | reference `k = 0.25` (half a declared 0.5σ design shift) |
| EWMA, asymptotic standardisation | Roberts, S. W. (1959), *Technometrics* 1, 239–250 | `λ = 0.10` (Lucas & Saccucci 1990 range) |
| Windowed GLR for an unknown-onset mean shift | Willsky, A. S. & Jones, H. L. (1976), *IEEE TAC* 21, 108–112 | window 100 samples |

A fourth statistic, a CUSUM on `z² − 1`, is included as an **oracle** — it is
given the post-change residual variance in advance — to measure the cost of
the three baselines being mis-specified for a variance change. It is not one
of the declared baselines.

Every threshold, including the model's, is set from a declared in-control ARL0
target using in-control data only. No detection delay is visible to any
threshold-setting procedure.

## 3. Architecture

- `sklearn.ensemble.RandomForestClassifier`: 200 trees, `max_depth=12`,
  `min_samples_leaf=5`, `random_state=53007`.
- Probability calibration: `sklearn.calibration.CalibratedClassifierCV` with
  `method="isotonic"`, wrapping the fitted forest in
  `sklearn.frozen.FrozenEstimator`.
- After fitting, the forest's `n_jobs` is forced to 1.

Two environment notes, both defects rather than modelling choices:

- `CalibratedClassifierCV(base, cv="prefit")` raises `InvalidParameterError`
  on scikit-learn 1.9.1. `FrozenEstimator` is the supported route.
- A forest with `n_jobs > 1` is several times **slower** at single-row
  inference on this container, because thread dispatch dominates tree
  traversal. Single-row inference is what a streaming monitor does.

PyTorch is not available in the build container; nothing here needs it.

### Features (nine, dimensionless, chosen before any model was fitted)

| # | Feature | Intended for |
|---|---|---|
| 0 | window mean | a step in the residual mean |
| 1 | window standard deviation | a change in residual spread |
| 2 | mean \|z\| | spread, robust to the tail |
| 3 | max \|z\| | single large excursions |
| 4 | lag-1 autocorrelation | residual colouring, which a process-noise change produces |
| 5 | excess kurtosis | tail shape |
| 6 | trend (OLS slope × window) | a ramp |
| 7 | fraction \|z\| > 2 | tail mass |
| 8 | mean z² | the sufficient statistic for a scale change |

Features 0, 1, 2 and 8 are what the analytic baselines already use in some
form; 4, 5 and 6 are what they do not, and are where any genuine win would
have to come from. Measured impurity importances (descriptive only, biased
towards high-cardinality features):

```
mean 0.5513 · mean_square 0.0836 · std 0.0691 · mean_abs 0.0614 ·
lag1_autocorr 0.0593 · trend 0.0561 · max_abs 0.0540 ·
excess_kurtosis 0.0452 · frac_abs_gt_2 0.0199
```

The window mean carries more than half the importance, which is the first
indication that the model has mostly rediscovered a mean-shift test.

## 4. Dataset: source and limitations

**Entirely simulated.** No flight data, no hardware, no real asset. Generated
by committed seeded scripts (`src/twininvalidate/asset.py`,
`src/twininvalidate/datasets.py`); no data file is committed and regeneration
is deterministic. Full detail in [DATASET_CARD.md](DATASET_CARD.md).

- Source model: the declared single-axis attitude channel of
  `twininvalidate.twin.reference_twin()`, 20 Hz, `ω_n = 2.0 rad/s`,
  `ζ = 0.10`, `σ_a = 0.02 rad/s²/√Hz`, `σ_meas = 0.01 rad`.
- Positive class: three change kinds at **one magnitude each** — actuator gain
  −1 %, the same ramped over 400 samples, process noise × 2.
- Negative class: in-control streams from the same twin.
- 13 440 training windows and 6 720 calibration windows, both exactly balanced,
  subsampled with stride 10 from 240 and 120 distinct runs respectively.

Limitations that matter for how the model behaves:

1. **One sign only.** Every gain change in the training set is negative. The
   model does not know that a positive gain change is also a change, and the
   out-of-distribution results below show it does not detect one.
2. **One magnitude per kind.** The model is implicitly tuned to the declared
   magnitudes.
3. **Deliberate label noise.** Every post-onset window of the slow ramp is
   labelled 1, including early windows in which the ramp has barely begun and
   which are statistically indistinguishable from in-control. Removing them
   would make the training problem easier than the monitoring problem is.
4. **Window overlap.** Consecutive windows share 49 of 50 samples, so a
   random window split would leak almost completely. See the split strategy.
5. **Nothing transfers** to an asset whose residual is not white,
   unit-variance and Gaussian in control.

## 5. Training procedure

```bash
python validation/validate_classifier.py     # trains, calibrates and benchmarks
```

1. Generate training streams from seeds 53002 (in-control) and 53010–53012
   (changed), 600 samples per run.
2. Extract nine features over 50-sample windows, subsample with stride 10.
3. Fit the forest on the training windows.
4. Set `forest.n_jobs = 1`.
5. Fit isotonic calibration on the **disjoint** calibration set (seeds 53020
   in-control / 53030–53032 changed).
6. Set the alarm threshold by bracketing the declared in-control ARL0 on an
   in-control bank (seed 53001) — in-control data only.

Nothing in steps 1–6 sees a detection delay. No hyperparameter was changed
after any benchmark result was seen.

Fit wall clock: 4.8 s on 2 cores.

## 6. Test-split strategy

**By run, never by window**, and by seed family rather than by slicing one
dataset. Four disjoint families:

| Seeds | Role |
|---|---|
| 53002 (in-control), 53010–53012 (changed) | forest training |
| 53020 (in-control), 53030–53032 (changed) | isotonic calibration |
| 53001 | in-control bank for threshold calibration |
| 53110–53123, 53200–53202, 53300–53303 | evaluation, including out of distribution |

An earlier allocation put the training changed streams at 53003–53005 and the
calibration streams at 53004–53007, so the two families reused seed *numbers*.
No array was actually shared, because the families are generated at different
`n_runs` and `default_rng` then draws different values — but a disjointness
claim that is true by accident is not evidence, so the families were spaced
and the model retrained. The retraining moved every result **against** the
model; the earlier, more favourable numbers were discarded. See
`validation/VALIDATION.md`, error 4.

Each family is an independent noise realisation, not a different slice of one.
A random window split would be meaningless here: consecutive windows overlap
in 49 of 50 samples.

## 7. Metrics

### Probabilistic quality, 90 120 held-out windows, balanced

| Metric | Value |
|---|---|
| Brier score | **0.1589** (0 is perfect; 0.25 is the constant-0.5 forecast on a balanced set) |
| Expected calibration error, 10 equal-width bins | **0.0208** |
| Largest gap in a well-populated bin | −0.046 in the `[0.1, 0.2)` bin (3928 windows) |
| Largest gap overall | −0.252 in the `[0.0, 0.1)` bin, which holds 59 of 90 120 windows and is not usable evidence |

The Brier score cannot approach 0 here and the reason is in the labels. A
single 50-sample window of the declared parameter step carries a mean shift
of 0.404σ, i.e. about 2.9 standard errors of the window mean, so a
window-level decision is genuinely uncertain; and the slow-ramp labelling adds
deliberate irreducible noise.

### Detection delay against the baselines, declared ARL0 target 1000 samples

Steady-state protocol (change at sample 50, so the windowed model has a full
feature window), 200 runs × 2500 samples. The model's confidence is quantised
(1857 distinct levels), so it cannot be operated at ARL0 1000; both achievable
bracketing thresholds are shown. Cells: mean delay in samples / fraction of
runs detected.

| Scenario | CUSUM | EWMA | GLR | learned-cons (ARL0 1516) | learned-gen (ARL0 874) |
|---|---|---|---|---|---|
| parameter step, gain −1 % | **52 / 1.00** | 64 / 1.00 | 65 / 1.00 | 53 / 1.00 | 47 / 1.00 |
| slow ramp | 229 / 1.00 | 238 / 1.00 | 251 / 1.00 | **213 / 1.00** | 183 / 1.00 |
| noise variance × 2 | **200 / 1.00** | 225 / 1.00 | 277 / 1.00 | 445 / 0.99 | 338 / 1.00 |
| OOD: gain **+1 %**, sign flipped | 54 / 1.00 | 65 / 1.00 | 67 / 1.00 | 8 / **0.02** | 75 / **0.04** |
| OOD: gain −0.4 %, smaller | 225 / 1.00 | 277 / 1.00 | 312 / 1.00 | **174 / 1.00** | 136 / 1.00 |
| OOD: gain −3 %, larger | 20 / 1.00 | **19 / 1.00** | 20 / 1.00 | 26 / 1.00 | 23 / 1.00 |
| OOD: process noise × 5 | **71 / 1.00** | 71 / 1.00 | 76 / 1.00 | 184 / 1.00 | 169 / 1.00 |

At the comparable (conservative) threshold the model **loses 2 of 3 in
distribution** (+2 % on the step, +122 % on the variance change) and **3 of 4
out of distribution**, including one catastrophic failure. Its single win is
the slow ramp, by 7 %.

### Cost per decision

| | per decision | ratio | fraction of a 50 ms sample period |
|---|---|---|---|
| learned classifier, one window | 10.469 ms | 1712 × | 20.9 % |
| CUSUM, per sample | 0.00612 ms | 1 × | 0.012 % |

Container measurements (2 cores, 7.8 GiB, Python 3.13.16), not hardware
characteristics. They move 10–20 % between runs: across the build session the
classifier was observed between 10.5 and 11.4 ms per decision and the CUSUM
between 0.0039 and 0.0061 ms per sample, so the ratio is **of order 2000**
rather than the precise figure in the table.

## 8. Uncertainty and confidence output

The model's output **is** its uncertainty output: `DriftClassifier.confidence`
returns the isotonic-calibrated posterior probability that the twin no longer
describes the asset, given the last 50 residuals. It is not a point estimate
with an error bar bolted on.

Its reliability is measured, not assumed: ECE 0.0174 over ten bins, Brier
0.1589, and a reliability diagram in
`screenshots/classifier_benchmark.png` (right panel).

Two properties a user must know:

1. **The confidence is quantised.** Isotonic regression produces a step
   function; on the calibration bank the confidence takes 1857 distinct
   values, so the achievable in-control ARL0 jumps. The two levels bracketing
   a target of 1000 samples give ARL0 1516.2 and 874.4, a factor of 1.73
   apart: the model cannot be operated at the baselines' false-alarm rate at
   all. Both are reported rather than whichever flatters it.
2. **The confidence is calibrated on the training distribution only.** On the
   sign-flipped change it is confidently and consistently wrong: 98 % of runs
   never reach the alarm threshold at all. A calibrated probability is not a
   guarantee of correctness outside the distribution it was calibrated on, and
   this model is the demonstration.

## 9. Failure cases

| Failure | Measured |
|---|---|
| **Opposite-sign change of the same magnitude** | detects 2 % of runs; all three baselines detect 100 % |
| Process-noise change at the trained magnitude | 445 samples against the CUSUM's 200, +122 % |
| Process-noise change at 2.5× the trained magnitude | 184 samples against the CUSUM's 71, +159 % |
| Parameter step at the trained magnitude | 53 samples against the CUSUM's 52, +2 % |
| Large parameter step (−3 %, 3× trained) | 26 samples against the EWMA's 19, +37 % |
| Change shorter than the feature window | cannot alarm at all before 50 samples have arrived |
| Per-decision latency | of order 2000 × the CUSUM recursion (10.5 ms against 0.006 ms per sample) |
| Arbitrary false-alarm target | cannot be set; the confidence is quantised, bracketing levels 1.73 apart in ARL0 |
| Any residual that is not white unit-variance Gaussian in control | untested, and every threshold assumes it |

**The structural explanation of all of it.** The model's one surviving win —
the slow ramp, by 7 % — is a specification advantage, not a learning
advantage. Its training set
contains only negative gain changes of a known size, so it has learned an
effectively one-sided test tuned to that magnitude, while the two-sided CUSUM
spends half its false-alarm budget on the other direction and is tuned to a
declared 0.5σ shift rather than the true 0.404σ. On a Gaussian, white,
unit-variance residual the CUSUM accumulates the log-likelihood ratio itself;
a classifier on window summaries can approach that and cannot exceed it
except by being told something the CUSUM is not told. The out-of-distribution
results are what being told it costs.

**Nothing was retuned after these results were seen.**

## 10. Reproducibility

Exact commands and seeds:

```bash
git clone https://github.com/OmAcharya-avtr/twininvalidate.git
cd twininvalidate
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python validation/validate_classifier.py   # trains, calibrates, benchmarks; ~65 s
python examples/classifier_benchmark.py    # the figure; ~57 s
python -m twininvalidate benchmark         # the CLI version; ~52 s
```

| Constant | Value |
|---|---|
| Forest seed | 53007 |
| Training stream seeds | 53002 (in-control), 53010–53012 (changed) |
| Calibration stream seeds | 53020 (in-control), 53030–53032 (changed) |
| Threshold-calibration bank seed | 53001 |
| Evaluation seeds | 53110–53123, 53200–53202, 53300–53303 |
| Feature window | 50 samples |
| Training-window stride | 10 |
| Declared target in-control ARL0 | 1000 samples (50 s at 20 Hz) |

Determinism: identical `StreamSpec` values give bit-identical residual
streams; the forest and the calibrator are seeded. Across scikit-learn or
NumPy versions the forest's tie-breaking may differ, so the delays may move at
the level of the standard errors quoted; the qualitative results — the
sign-flip collapse above all — do not depend on that.

## 11. Compute used

| Step | Wall clock, 2 cores |
|---|---|
| Generate training + calibration windows | 1.7 s |
| Fit forest (13 440 windows, 200 trees) | 4.8 s |
| Isotonic calibration | included above |
| Confidence path, 200 × 2500 samples | about 6 s |
| Full benchmark script | 65 s |

Total energy and hardware: a 2-core, 7.8 GiB container. No GPU, no
accelerator, no distributed training. The whole model is about 1 MB of fitted
trees.

## 12. Ethical and safety limits

- **This model is not certified for operational flight use.**
- It is research-grade: not flight-qualified, not certified, not approved for
  operational aerospace use.
- It must not be used as a monitor. Its documented failure on an opposite-sign
  change means it would silently miss an entire class of asset change. The
  measured recommendation of this repository is **use the CUSUM**.
- An alarm from any detector in this package, learned or analytic, means the
  twin and the asset disagree. It does **not** identify a fault, locate one,
  or attribute one to a cause. See `twininvalidate.ambiguity`, where two
  physically different situations produce residual streams identical to
  6.17e-14.
- No personal data is involved. The training data is simulated from a
  declared model and contains no measurement of any real system.
- The model's confidence output is well calibrated on its training
  distribution and misleading outside it. Anyone presenting that confidence to
  an operator must say which it is.

## 13. References

As listed in [validation/VALIDATION.md](validation/VALIDATION.md) §8, all
verified 2026-10-08. The ones specific to this card:

- Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika* 41(1/2),
  100–115. DOI 10.1093/biomet/41.1-2.100.
- Roberts, S. W. (1959). "Control Chart Tests Based on Geometric Moving
  Averages." *Technometrics* 1, 239–250. DOI 10.1080/00401706.1959.10489860.
- Willsky, A. S. & Jones, H. L. (1976). "A Generalized Likelihood Ratio
  Approach to the Detection and Estimation of Jumps in Linear Systems."
  *IEEE Transactions on Automatic Control* 21, 108–112.
- Lorden, G. (1971). "Procedures for reacting to a change in distribution."
  *Annals of Mathematical Statistics* 42(6), 1897–1908.
- Basseville, M. & Nikiforov, I. V. (1993). *Detection of Abrupt Changes:
  Theory and Application*. Prentice-Hall.
- Brier, G. W. (1950). "Verification of forecasts expressed in terms of
  probability." *Monthly Weather Review* 78(1).
