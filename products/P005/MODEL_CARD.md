# Model Card — JitterScope `NominalModel` (autoencoder-equivalent anomaly detector)

**Package:** `jitterscope` 0.1.0 · **Date:** 2026-08-06 · **License:** Apache-2.0
**Status:** TESTING · **Validation level:** 2 (Research)

> **This model is not certified for operational flight use.**

## 1. Problem

Detect anomalous behaviour in single-channel platform vibration / pointing-jitter
telemetry, where "anomalous" means the short-term power spectral density departs
from the spectrum of a nominal reference period. Target signatures: a new tone
appearing, broadband energy shifting in a band, and intermittent transients.
Output is a per-window anomaly score, a binary flag against a calibrated
threshold, and a confidence value.

This is unsupervised, one-class anomaly detection: the model is trained only on
nominal data and never sees a labeled fault during training.

## 2. Baseline (implemented and evaluated first)

`BandZScoreBaseline` — per-band log-energy z-score thresholding. Fits the mean
and standard deviation of each log-PSD bin on nominal windows; the anomaly score
of a window is `max_b |(x_b − μ_b)/σ_b|`. This is the standard control-chart
limit check used in vibration condition monitoring (Randall 2011,
*Vibration-based Condition Monitoring*, Wiley, ch. 3; ISO 10816 uses fixed band
limits in the same spirit).

The baseline was built first, and the ML model is benchmarked against it on the
same features, the same threshold rule, and the same held-out labeled data.

## 3. Architecture

`sklearn.neural_network.MLPRegressor` trained as an autoencoder — it reconstructs
its own input through a bottleneck (PyTorch is not available in this environment;
this is the autoencoder-equivalent construction).

| Property | Value |
|---|---|
| Input / output dimension | 24 (log-spaced log-PSD bins, 1 Hz → Nyquist) |
| Hidden layers | 16 → **6** → 16 (bottleneck 6 ≪ 24 forces compression) |
| Activation | tanh |
| Solver | Adam, `alpha = 1e-3` (L2), `max_iter = 3000`, `tol = 1e-6` |
| Preprocessing | `StandardScaler` fitted on the training split |
| Anomaly score | Per-window reconstruction MSE in standardized units |
| Parameters | ~1 000 weights |

Reconstruction-error anomaly scoring follows Sakurada & Yairi 2014 (MLSDA
workshop, "Anomaly Detection Using Autoencoders with Nonlinear Dimensionality
Reduction"); the autoencoder principle follows Hinton & Salakhutdinov 2006,
*Science* 313:504–507.

### Features

Telemetry is segmented into 1 s windows with 50 % overlap. Each window gets a
Welch PSD (`nperseg = 256`, Hann, 50 % overlap), averaged into 24
logarithmically-spaced frequency bins, then `log10`. Log-energy band features are
standard in vibration condition monitoring (Randall 2011, ch. 3). Log spacing
gives roughly equal weight per decade, matching how structural modes and
low-frequency drift distribute.

## 4. Dataset

Synthetic and idealized — see `DATASET_CARD.md` for full composition, fault
definitions, and the list of unmodeled effects. Summary:

- **Train:** 4 nominal records, 60 s @ 1 kHz, seeds 1000–1003 → 476 windows × 24 bins.
- **Test:** 24 records (6 nominal, 6 `new_tone`, 6 `band_shift`, 6 `transient`),
  seeds 2000–2023 → 2856 windows, 952 faulty.

**Limitations of the data:** no nonstationarity, no multi-axis coupling, no
structural transfer function, no sensor artifacts or dropouts, no control-loop
shaping, Gaussian-only statistics, and clean step-onset faults. Performance below
does not transfer to real telemetry.

## 5. Test-split strategy

Three-way separation by seed, with no leakage:

1. **Training records** (seeds 1000–1003) — nominal only, used to fit features.
2. **Within training, a seeded 70/30 split**: the MLP is fitted on 70 % of nominal
   windows; the remaining 30 % is a **held-out calibration set** used solely to
   set the threshold and the confidence reference distribution. This matters:
   reconstruction error on the fit set underestimates error on unseen data, so an
   in-sample threshold would be optimistically low and produce excess false
   alarms. The baseline uses the same calibration convention.
3. **Test records** (seeds 2000–2023) — never seen in any form during fitting or
   calibration.

## 6. Training procedure and compute

Threshold rule (both models): the 0.995 quantile of held-out nominal scores.

| Item | Value |
|---|---|
| MLP fit time | **2.452 s** |
| Baseline fit time | **0.001 s** |
| Full benchmark (fit + 24 test records + metrics) | **6.4 s** |
| Hardware | 2 CPU cores, no GPU |
| Convergence | Adam stops at `max_iter`; the `ConvergenceWarning` is suppressed deliberately because exact loss convergence is not required — the threshold is calibrated post-hoc on held-out scores |

Well inside the 3-minute compute budget.

## 7. Metrics (from an actual run — `validation/val_detector_output.txt`)

At the fixed q0.995 operating threshold, on 2856 test windows (952 faulty):

| Model | Precision | Recall | F1 | TP | FP | FN | ROC AUC |
|---|---|---|---|---|---|---|---|
| **Baseline band z-score** | **0.9900** | 0.9401 | **0.9644** | 895 | 9 | 57 | **0.9797** |
| MLP autoencoder | 0.9751 | **0.9454** | 0.9600 | 900 | 23 | 52 | 0.9784 |

Per-fault-type recall:

| Fault type | Faulty windows | Baseline | MLP |
|---|---|---|---|
| `new_tone` | 360 | 1.0000 | 1.0000 |
| `band_shift` | 360 | 1.0000 | 1.0000 |
| `transient` | 232 | 0.7543 | 0.7759 |

False alarms on 714 purely nominal test windows: baseline 4 (0.56 %),
MLP 10 (1.40 %).

### Verdict: the baseline wins

**The classical baseline outperforms the ML model** on F1 (0.9644 vs 0.9600) and
ROC AUC (0.9797 vs 0.9784), fits 2 400× faster, and hits its design false-alarm
rate (0.56 % against a 0.5 % target) where the MLP runs 2.8× hot at 1.40 %.

The margin is under 1 % on both metrics. Both models detect `new_tone` and
`band_shift` perfectly; the whole difference is in `transient`, where the MLP
recovers marginally more true positives (0.7759 vs 0.7543) at the cost of 14 more
false alarms. The honest reading is that the two are near-equivalent on this data
and the baseline is preferable on cost and calibration.

**Why:** the injected faults are additive-energy signatures that raise log-PSD in
specific bins — precisely the alternative hypothesis a per-bin z-score is optimal
against. The autoencoder has no structural advantage here.

The MLP is retained for the case of *correlated multi-band shape changes* that a
per-bin marginal test cannot see. **That case is not demonstrated in this
benchmark and remains an untested claim.** The CLI therefore defaults to
`--detector baseline`.

## 8. Uncertainty / confidence output

Required by the AI-product rules and provided: `NominalModel.confidence(scores)`
returns the **empirical CDF of the score under the held-out nominal
distribution** — the fraction of calibration windows scoring below the observed
value, in [0, 1]. `DetectionResult.confidence` carries it per window.

Measured separation: mean confidence **0.5066** on nominal windows,
**0.9781** on faulty windows.

Interpretation and its limit: this is a calibrated *how abnormal relative to
nominal* measure, **not** a probability that a fault is present. It saturates at
1.0 once a score exceeds every calibration score, so it cannot rank the severity
of strong anomalies against each other — use the raw score for that. It carries
no epistemic uncertainty about the model itself.

## 9. Failure cases (observed and expected)

- **Transients are under-detected** (recall 0.75–0.78, both models). A ~0.25 s
  burst is diluted by the 1 s window average. Shorter windows or a kurtosis /
  crest-factor feature would address this; neither is implemented.
- **MLP false-alarm rate runs ~2.8× above its design target** because the 30 %
  calibration split under-represents nominal variability with only 476 training
  windows.
- **Nonstationary nominal operation breaks both models.** A slew, a wheel speed
  change, or a thermal transient shifts the nominal spectrum and will be flagged
  as anomalous. This is the dominant expected failure mode on real telemetry.
- **A fault present during the training period is learned as nominal** and will
  never be flagged. The CLI's `--train-frac` convention (leading fraction assumed
  nominal) makes this the operator's responsibility to verify.
- **Faults that preserve the band energy distribution are invisible** — e.g. a
  phase-only change, or a tone that shifts within a single bin.
- **Single channel only.** Cross-axis-only anomalies cannot be represented.
- **Out-of-distribution sample rates or window lengths** silently change the
  feature meaning; the same `FeatureExtractor` configuration must be used at fit
  and detect time (the API requires passing it explicitly).

## 10. Reproducibility

Deterministic given the seeds. Exact commands:

```bash
# Full benchmark, all numbers in section 7 (~6 s, 2 CPU cores)
python validation/val_detector.py     # writes validation/val_detector_output.txt

# Reproducibility regression test (same seed -> identical scores and threshold)
python -m pytest tests/test_detect.py::test_mlp_reproducibility -q
```

Seeds: training records 1000–1003; test records 2000–2023; MLP
`random_state = 0`, which fixes both weight initialization and the
fit/calibration split permutation. Environment: Python 3.11, numpy 2.4.4,
scipy 1.17.1, scikit-learn 1.8.0.

Determinism caveat: bit-identical scores depend on the BLAS build and platform
floating-point behaviour. Metrics are stable to the reported precision across
runs on the same machine.

## 11. Ethical and safety limits

- **This model is not certified for operational flight use.** It is
  research-grade software, not flight-qualified and not approved for operational
  aerospace use.
- Trained exclusively on synthetic data. It has never seen real telemetry, and no
  claim about real-hardware performance is supported by this work.
- Do not use it as the sole basis for a fault declaration, a safe-mode entry, an
  autonomous vehicle action, or a hardware disposition decision. Any flag is a
  cue for human engineering review.
- No personal, human-subject, or export-controlled data is involved.
- The confidence output must not be reported as a probability of failure to
  downstream decision-makers.
- Threshold choice is a safety-relevant decision: the default q0.995 trades a
  ~0.5 % false-alarm rate for sensitivity, which is a demonstration setting, not
  a mission-tuned one.
