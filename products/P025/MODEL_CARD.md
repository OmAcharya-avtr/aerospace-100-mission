# Model Card — FDIScope residual fault classifier v0.1.0

**This model is not certified for operational flight use.**

**Every residual, scenario and label used to train or evaluate this model is
SIMULATED — see `DATASET_CARD.md`. No flight telemetry, no measured
innovation sequence and no on-orbit fault log is used anywhere in this
package.**

## Problem

Given a window of 100 normalised Kalman-filter residuals from a spacecraft
attitude control loop, decide whether a fault is present and, if so, which of
seven fault modes it is: sensor bias, sensor drift, sensor stuck, sensor
dropout, actuator loss of effectiveness, actuator stuck, actuator runaway.
The model outputs a class and a confidence. Detection uses the same model:
the score is `1 - P(none)`.

## Classical baselines, implemented and validated first

Four classical detectors and one classical isolator were written, tested and
validated **before** any learned component, and all of them are benchmarked
against the model on identical held-out windows
(`validation/detection_benchmark.py`, `validation/isolation_confusion.py`):

1. **`chi2_short`** — sliding chi-squared test on the normalised innovation
   squared, 25-sample window, threshold `chi2.isf(alpha, 50)`. Its measured
   false-alarm rate matches its design value across two decades of `alpha`
   (`validation/VALIDATION.md`, V1-A2).
2. **`chi2_long`** — the same test with a 100-sample window.
3. **`cusum`** — four one-sided CUSUMs, one per residual channel and sign
   (Page 1954), with the threshold from the Siegmund `ARL0` expression. Its
   measured run length matches that expression to within 3 % in the exact
   change-point model (V2-B1a).
4. **`glr`** — a bank of generalised-likelihood-ratio tests matched to the
   model-derived residual signature of each fault (Willsky 1976). This is the
   **strong competitor**: it is a classical, model-based isolator with no
   fitted parameters at all, and it wins on detection.

## Architecture

- `sklearn.ensemble.RandomForestClassifier(n_estimators=150, max_depth=12,
  min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=1)`
  (`fdiscope.classifier.FaultClassifier`). No PyTorch, no GPU.
- **Inputs, sixteen features** (`fdiscope.features.window_features`), all
  computed from the normalised residual window and **nothing else** — no true
  state, no fault label, no plant parameter enters the feature vector, so the
  model sees exactly the information the classical tests see:

  | # | Feature | Expected value under `H0` |
  |---:|---|---|
  | 0, 6 | `mean_ch{c}` | 0 |
  | 1, 7 | `std_ch{c}` | 1 |
  | 2, 8 | `slope_ch{c}` (per window) | 0 |
  | 3, 9 | `autocorr1_ch{c}` | 0 |
  | 4, 10 | `max_abs_ch{c}` | E[max of W standard normals] |
  | 5, 11 | `cusum_range_ch{c}` | range of a random walk / sqrt(W) |
  | 12 | `mean_nis` | 2 |
  | 13 | `max_nis` | — |
  | 14 | `corr_01` | 0 |
  | 15 | `exceed_frac` | 0.01 |

- **Target**: the scenario's fault class for windows starting 0, 10, 25 and 50
  samples after onset; `FaultType.NONE` for windows 150 and 300 samples
  *before* onset, which are genuinely fault-free.
- **Outputs**: `predict_with_confidence` returns the class and the winning
  vote fraction; `detection_score` returns `1 - P(none)`.

## Dataset

Entirely synthetic. 240 training scenarios (seeds 1000–1239) giving 1440
feature rows × 16 features, and 240 held-out scenarios (seeds 5000–5239).
Classes are exactly balanced by construction, 30 per class per set.
Regenerate with `python data/generate_dataset.py`. See `DATASET_CARD.md` for
the sampling ranges and their limitations.

## Test-split strategy

Scenario seeds are disjoint by construction: 1000–1239 train, 5000–5239 held
out, 9000–9149 threshold calibration (all fault-free), 12000–12149 held-out
false-alarm measurement (all fault-free). Every window of a scenario stays on
the same side of the split, so there is no window-level leakage. The
classifier's decision threshold and the GLR bank's are calibrated on the
9000-block only. The held-out set is scored once per method; no held-out
result was used to reselect a hyperparameter, a seed or a tolerance.

## Training procedure

```bash
python validation/detection_benchmark.py    # trains and evaluates end to end, ~63 s
python validation/isolation_confusion.py    # the isolation half, ~26 s
python data/generate_dataset.py             # writes the same dataset to CSV, ~17 s
```

Simulation settings for every run: `n_steps=2000`, `dt=0.1 s`, onset drawn
uniformly from samples 600–1300, sinusoidal attitude reference of 0.02 rad
amplitude and 60 s period. No hyperparameter search was performed against the
held-out set; the forest size was fixed a priori as "small enough to fit and
score inside the compute budget, large enough for a stable vote fraction".

## Metrics

**Detection**, at a common per-run false-alarm probability of 10 % calibrated
on the fault-free 9000-block (`validation/VALIDATION.md`, V3):

| method | measured FAR/run | detection rate | mean delay [samples] | 95 % CI | median | AUC |
|---|---:|---:|---:|---|---:|---:|
| `chi2_short` | 0.1167 | 0.9857 | 71.42 | [59.81, 83.02] | 49.0 | — |
| `chi2_long` | 0.1222 | 0.9857 | 75.76 | [63.88, 87.64] | 57.0 | 0.9411 |
| `cusum` | 0.1389 | **1.0000** | **54.33** | [44.90, 63.76] | **33.0** | 0.9459 |
| `glr` | 0.0389 | **1.0000** | **54.27** | [46.85, 61.69] | 41.0 | **0.9751** |
| **`learned`** | 0.1500 | 0.9952 | 56.58 | [48.25, 64.90] | 42.0 | 0.9695 |

**Isolation**, on the identical `[onset, onset + 100)` window of all 240
held-out runs. The full 8×8 confusion matrices are in
`validation/VALIDATION.md`, V4-D2, and are not summarised here:

| method | accuracy | 95 % Wilson |
|---|---:|---|
| `glr` | 0.4667 | [0.4046, 0.5298] |
| **`learned`** | **0.6958** | [0.6349, 0.7506] |

### The honest summary

**The learned classifier wins the isolation problem and loses the detection
problem.** Its isolation accuracy is 0.6958 against the classical GLR bank's
0.4667, and it wins on six of the seven fault classes. But on detection it is
slower than both sequential classical tests (56.58 samples against 54.33 for
the CUSUM and 54.27 for the GLR bank) while running at a *higher* measured
false-alarm rate (0.150 against 0.139 and 0.039), and its AUC of 0.9695 is
below the classical GLR bank's 0.9751 at every operating point measured. It is
never the fastest method on any individual fault class.

If you need to know *that* something has failed, use the CUSUM: it is faster,
it misses nothing, and its threshold comes from a formula rather than from a
calibration set. If you need to know *what* has failed, the classifier is
worth its complexity — except for loss of effectiveness, where the classical
bank's recall is 0.4667 against the classifier's 0.1000.

## Uncertainty / confidence output

`predict_with_confidence` returns the fraction of trees voting for the winning
class. Measured calibration on the held-out set
(`validation/VALIDATION.md`, V4-D4):

| confidence bucket | n | accuracy | mean confidence | gap |
|---|---:|---:|---:|---:|
| [0.00, 0.40) | 36 | 0.4722 | 0.3428 | −0.1294 |
| [0.40, 0.60) | 78 | 0.6410 | 0.5014 | −0.1396 |
| [0.60, 0.80) | 57 | 0.8246 | 0.7066 | −0.1180 |
| [0.80, 0.90) | 47 | 0.7872 | 0.8495 | +0.0623 |
| [0.90, 1.00) | 22 | 0.7273 | 0.9259 | +0.1987 |

**This is an ensemble-agreement heuristic, not a calibrated probability.** It
is under-confident below 0.8 and over-confident above it, with gaps up to
0.20. A confidence of 0.93 means 93 % of the trees agree, not that the
prediction is right 93 % of the time — the measured accuracy in that bucket is
0.7273. No isotonic or Platt recalibration was fitted; treat it as a relative
ranking of trustworthiness only.

For comparison the classical GLR bank's posterior is far worse: 137 of its 182
declarations sit in the top bucket with mean confidence 0.9973 and accuracy
0.4599, a gap of **+0.5375**. Neither confidence should be relied on.

## Failure cases

1. **It loses to both classical sequential detectors on detection delay and to
   the classical GLR bank on AUC.** See the metrics table. It also runs at a
   higher measured false-alarm rate than the operating point it was calibrated
   to, which flatters those delays.
2. **Loss of effectiveness is its worst class by a wide margin**: recall
   0.1000, against 0.4667 for the classical bank. It sends 14 of 30 such cases
   to `actuator_stuck` and 6 to `none`.
3. **It cannot separate the actuator faults reliably.** Recalls 0.1000, 0.5333
   and 0.3333 for LOE, stuck and runaway. A stuck actuator latched near the
   value a settled controller would have commanded anyway is close to
   invisible, and the sampler deliberately keeps those cases in.
4. **Its window is 100 samples long, which is a floor on its detection
   delay.** A CUSUM updates every sample and has no such floor; that is most of
   the delay gap.
5. **It has no threshold formula.** Detection requires a fault-free
   calibration set, and the threshold so obtained transferred at 0.150 against
   a 0.10 target on held-out data (`validation/VALIDATION.md`, V3-C1 **FAIL**).
   The chi-squared and CUSUM thresholds need no data and landed inside their
   intervals.
6. **Isolation assumes the onset sample is known.** Shifting the window 50
   samples early costs 49 % of its accuracy (V4-D3). The benchmark shares this
   idealisation with the classical bank, so the comparison is fair, but the
   absolute number is optimistic for both.
7. **No out-of-distribution guard at all.** A residual window from a different
   plant, a different sampling rate or a mismatched filter still returns a
   class and a confidence, with nothing to say it is outside the training
   distribution.
8. **240 training scenarios with all rows of a scenario sharing one label.**
   The 1440 rows are far from 1440 independent observations; the effective
   sample size is closer to 240.
9. **Exactly one fault at a time.** Simultaneous faults are neither trained
   nor tested, and the eight-class softmax cannot express them.

## Reproducibility

```bash
python -m pytest tests/ -q                  # 435 passed
python validation/detection_benchmark.py    # detection half, ~63 s
python validation/isolation_confusion.py    # isolation half, ~26 s
python data/generate_dataset.py             # the training CSVs + sha256 manifest
```

Seeds: training scenarios 1000–1239, held out 5000–5239, calibration
9000–9149, held-out fault-free 12000–12149, `random_state=0` for the forest.
Identical seeds give bit-identical predictions, checked by
`tests/test_classifier.py::TestReproducibility::test_same_seed_gives_identical_probabilities`,
and pinned reference values live in `tests/test_benchmark_regression.py`.
`data/dataset_manifest.txt` records a SHA-256 of each generated CSV.

## Compute used

2 CPU cores, no GPU, `n_jobs=1` throughout. The full campaign — 240 + 240
scenario simulations, 150 calibration runs, 150 held-out fault-free runs, the
signature bank and the forest fit — takes **27 s** to build; the detection
benchmark on top of it takes 36 s and the isolation analysis 0.5 s. The forest
fit itself is under a second on 1440 rows. Peak memory stays under 400 MB.
The whole model is 150 trees of depth ≤ 12.

## Ethical and safety limits

- **This model is not certified for operational flight use.** It must not be
  used to declare a fault on a real spacecraft, to trigger a reconfiguration,
  or to support any go/no-go decision.
- It was trained and evaluated entirely inside a simulation whose filter model
  is *exactly* the plant model. Real filters are mismatched, and a mismatched
  filter's innovation is not white — which invalidates the distributional
  assumption every method here rests on, the classical ones included. Nothing
  in this repository measures that gap.
- It has no failure detection of its own. Outside the training distribution it
  still returns a class and a confidence.
- The recommended default in this repository is the **classical** chi-squared
  or CUSUM test for detection, with this model used, if at all, only for
  isolation after a classical detector has fired.
- No personal, proprietary or export-controlled data is involved anywhere:
  every input is generated by a committed seeded script.

## References

- Basseville, M. and Nikiforov, I. V., *Detection of Abrupt Changes: Theory
  and Application*, Prentice-Hall, 1993.
- Willsky, A. S., "A Survey of Design Methods for Failure Detection in Dynamic
  Systems", *Automatica*, 12(6), 1976, pp. 601–611.
- Gertler, J., *Fault Detection and Diagnosis in Engineering Systems*, Marcel
  Dekker, 1998.
- Chen, J. and Patton, R. J., *Robust Model-Based Fault Diagnosis for Dynamic
  Systems*, Kluwer, 1999.
- Page, E. S., "Continuous Inspection Schemes", *Biometrika*, 41(1/2), 1954,
  pp. 100–115.
- Siegmund, D., *Sequential Analysis: Tests and Confidence Intervals*,
  Springer, 1985.
