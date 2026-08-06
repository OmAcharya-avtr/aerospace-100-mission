# JitterScope — Validation Evidence

**Validation level:** 2 (Research) · **Package:** `jitterscope` 0.1.0 · **Date of run:** 2026-08-06

Every number below was produced by running the scripts in this directory in this
session on the target environment (Python 3.11, numpy 2.4.4, scipy 1.17.1,
scikit-learn 1.8.0, 2 CPU cores). Raw script output is saved verbatim alongside
each script. No tolerance was adjusted after seeing a result.

| Script | Raw output | Subject | Result |
|---|---|---|---|
| `val_psd.py` | `val_psd_output.txt` | Welch PSD known answers | PASS |
| `val_pointing.py` | `val_pointing_output.txt` | Pointing-loss closed form vs Monte Carlo | PASS |
| `val_detector.py` | `val_detector_output.txt` | Detector precision/recall/F1 vs baseline | PASS (baseline wins) |

Reproduce all three:

```bash
python validation/val_psd.py
python validation/val_pointing.py
python validation/val_detector.py
```

---

## 1. Welch PSD known-answer tests

Reference: Bendat & Piersol 2010, *Random Data: Analysis and Measurement
Procedures*, 4th ed., ch. 5 (Parseval / PSD normalization); Welch 1967,
*IEEE Trans. Audio Electroacoust.* 15(2):70–73 (averaged periodogram method).

### 1a. Pure sinusoid — analytic answer

Signal `x(t) = A sin(2π f₀ t)`, `A = 2.0e-6 rad`, `f₀ = 80.000 Hz`, 20 s at
1000 Hz. Analytic: total power `A²/2 = 2.000000e-12 rad²`, band RMS
`A/√2 = 1.414214e-6 rad`, all power concentrated at `f₀`.
Welch settings: `nperseg = 4096`, Hann window, 50 % overlap, resolution
`df = 0.2441 Hz`.

| Quantity | Measured | Expected | Error |
|---|---|---|---|
| Peak frequency | 80.0781 Hz | 80.0000 Hz | 0.0781 Hz = **0.320 bins** |
| Integrated power `∫S df` | 2.000001e-12 rad² | 2.000000e-12 rad² | rel **5.241e-07** |
| Band RMS 70–90 Hz | 1.414214e-06 rad | 1.414214e-06 rad | rel **8.374e-11** |
| Band RMS 200–400 Hz (leakage floor) | 2.547916e-14 rad | ~0 | 1.802e-08 × tone band |

Peak offset is 0.32 of one frequency bin, i.e. the tone falls between bins —
expected behaviour for a non-bin-centred frequency, not an error. Pass criteria
(peak within 1 bin, power relative error < 1e-2): **PASS**.

### 1b. White Gaussian noise — analytic answer

`σ = 1.0e-6 rad`, 200 000 samples (200 s) at 1000 Hz, seed 20260806.
Analytic one-sided PSD level `σ²/(f_s/2) = 2.000000e-15 rad²/Hz`; integrated PSD
must equal the sample variance (Parseval). 96 averaged segments.

| Quantity | Measured | Expected | Error |
|---|---|---|---|
| PSD level (median over bins) | 1.987657e-15 rad²/Hz | 2.000000e-15 | rel **6.171e-03** |
| PSD level (mean over bins) | 1.991950e-15 rad²/Hz | 2.000000e-15 | rel **4.025e-03** |
| Bin-to-bin flatness (std/mean) | 0.1048 | 0.1021 (χ²₂ₖ, k = 96) | rel 2.6e-02 |
| `∫S df` vs sample variance | 9.956535e-13 rad² | 9.958205e-13 rad² | rel **1.677e-04** |
| Cumulative RMS total | 9.978244e-07 rad | 9.979081e-07 rad (sample std) | rel 8.4e-05 |

The measured estimator scatter (0.1048) matches the theoretical χ² scatter for 96
averaged segments (0.1021) to 2.6 %, confirming the variance-reduction behaviour
of the Welch average is as documented. Pass criteria (level and Parseval relative
error < 5e-2): **PASS**.

A companion property-based test (`tests/test_psd.py::test_parseval_consistency_property`,
Hypothesis, 25 random seeds/amplitudes) checks Parseval to 10 % on random Gaussian
records — a statistical tolerance justified by the ~63-segment Welch average.

---

## 2. Gaussian-beam pointing loss: closed form vs Monte Carlo

**Claim under test.** For a far-field TEM00 Gaussian beam of 1/e² half-angle
divergence `θ_div`, instantaneous loss `L(θ) = exp(-2θ²/θ_div²)`
(Siegman 1986, *Lasers*, ch. 17). With zero-mean Gaussian jitter of per-axis
standard deviation `σ_θ` on two independent axes, the radial error is Rayleigh
distributed and the Gaussian integral gives

```
⟨L⟩ = 1 / (1 + 4 (σ_θ / θ_div)²)
```

This is the point-receiver limit of the pointing-error model of Farid &
Hranilovic 2007, *J. Lightwave Technol.* 25(7):1702–1710; see also Andrews &
Phillips 2005, *Laser Beam Propagation through Random Media*, 2nd ed., ch. 12.

**Method.** Independent seeded Monte Carlo, `θ_div = 12.0 µrad`, 1 000 000 draws
per axis per case, base seed 20260806. The MC standard error of the mean is
reported so agreement can be judged against sampling noise rather than an
arbitrary tolerance.

| σ/θ_div | σ [µrad] | Closed form | Monte Carlo | MC std err | Rel. error | \|err\|/SE |
|---|---|---|---|---|---|---|
| 0.00 | 0.000 | 1.000000 | 1.000000 | 0.00e+00 | 0.000e+00 | 0.00 |
| 0.10 | 1.200 | 0.961538 | 0.961488 | 3.71e-05 | 5.266e-05 | 1.37 |
| 0.25 | 3.000 | 0.800000 | 0.799911 | 1.63e-04 | 1.114e-04 | 0.55 |
| 0.50 | 6.000 | 0.500000 | 0.499917 | 2.89e-04 | 1.659e-04 | 0.29 |
| 0.75 | 9.000 | 0.307692 | 0.307627 | 2.95e-04 | 2.123e-04 | 0.22 |
| 1.00 | 12.000 | 0.200000 | 0.199940 | 2.67e-04 | 3.020e-04 | 0.23 |
| 1.50 | 18.000 | 0.100000 | 0.100241 | 2.07e-04 | 2.405e-03 | 1.16 |
| 2.00 | 24.000 | 0.058824 | 0.058584 | 1.64e-04 | 4.068e-03 | 1.46 |

**Maximum relative error across all cases: 4.068e-03** (at σ/θ_div = 2.0, the
deepest-fade case where the MC estimator has the poorest relative efficiency).
Every deviation is **within 1.5 Monte Carlo standard errors**, i.e. consistent
with pure sampling noise — the closed form is confirmed, not merely
approximately matched. Pass criterion (max relative error < 1e-2): **PASS**.

Engineering form of the same result:

| σ/θ_div | ⟨L⟩ | Loss [dB] |
|---|---|---|
| 0.10 | 0.9615 | 0.17 |
| 0.25 | 0.8000 | 0.97 |
| 0.50 | 0.5000 | 3.01 |
| 1.00 | 0.2000 | 6.99 |

**Validity range.** No static boresight bias; isotropic, equal-variance jitter on
both axes; unobscured TEM00 beam; point receiver in the far field; no atmospheric
scintillation or beam wander. Biased or anisotropic jitter breaks the closed form
and requires numerical integration.

A seeded regression test of the same comparison (200 k draws, 1 % tolerance ≈ 4.5σ)
runs in `tests/test_pointing.py::test_closed_form_matches_monte_carlo`.

---

## 3. Anomaly detector benchmark on a labeled synthetic fault set

**Design.** The classical baseline was implemented and fitted first. Both
detectors consume identical features (24 log-spaced log-PSD bins, 1 s windows,
50 % overlap), use the identical threshold rule (q0.995 of held-out nominal
scores), and are evaluated on the identical labeled test set, so the comparison
isolates the model.

- **Training:** 4 nominal records, 60 s @ 1 kHz, seeds 1000–1003 → feature
  matrix (476 windows × 24 bins).
- **Test set:** 24 records, 60 s each — 6 nominal, 6 `new_tone`, 6 `band_shift`,
  6 `transient` (seeds 2000+). **2856 windows total: 952 faulty, 1904 nominal.**
- **Labeling:** a window is positive if any sample in it carries the generator's
  fault mask. Test records are disjoint from training records by seed.
- **MLP:** `MLPRegressor(24 → 16 → 6 → 16 → 24)`, tanh, Adam, `alpha=1e-3`,
  `max_iter=3000`, `random_state=0`; trained to reconstruct standardized nominal
  features; score = reconstruction MSE. Thresholds: baseline 3.9906 (max |z|),
  MLP 1.0099 (MSE).

### Overall metrics at the fixed q0.995 operating threshold

| Model | Precision | Recall | F1 | TP | FP | FN | ROC AUC |
|---|---|---|---|---|---|---|---|
| **Baseline band z-score** | **0.9900** | 0.9401 | **0.9644** | 895 | 9 | 57 | **0.9797** |
| MLP autoencoder | 0.9751 | **0.9454** | 0.9600 | 900 | 23 | 52 | 0.9784 |

### Per-fault-type recall

| Fault type | Faulty windows | Baseline recall | MLP recall |
|---|---|---|---|
| `new_tone` | 360 | 1.0000 | 1.0000 |
| `band_shift` | 360 | 1.0000 | 1.0000 |
| `transient` | 232 | 0.7543 | 0.7759 |

### False alarms on the 6 purely nominal test records (714 windows)

| Model | False alarms | Rate |
|---|---|---|
| Baseline | 4 | 0.56 % |
| MLP | 10 | 1.40 % |

The baseline's 0.56 % false-alarm rate matches the 0.5 % design expectation of a
q0.995 threshold. The MLP's 1.40 % is ~2.8× the design target: its held-out
calibration split (30 % of nominal windows) under-represents nominal variability
relative to the per-bin Gaussian statistics the baseline fits. This is a
documented model limitation, not a tuning artifact.

### Confidence (uncertainty) output

The MLP's required confidence channel is the empirical nominal CDF of the score.
Mean confidence on nominal windows **0.5066**; on faulty windows **0.9781** —
the channel separates the two populations and is usable for triage ranking.

### Verdict — stated plainly

**The classical baseline wins.** F1 0.9644 vs 0.9600 (Δ = +0.0044 for the
baseline); ROC AUC 0.9797 vs 0.9784 (Δ = +0.0013 for the baseline). The baseline
also fits in 0.001 s versus 2.452 s for the MLP, and has no hyperparameters
beyond the threshold quantile.

The margin is small (well under 1 % on both metrics) and both models detect
`new_tone` and `band_shift` faults perfectly; the entire difference lives in the
`transient` class, where the MLP recovers slightly more true positives (0.7759 vs
0.7543) at the cost of 14 extra false alarms. Neither model is a clear winner on
transients, because a short burst occupies a small fraction of a 1 s window and is
diluted by the window average — a feature-resolution limitation shared by both.

**Why the baseline wins:** the injected faults are additive-energy signatures
that raise log-PSD in specific bins, which is exactly the alternative hypothesis
the per-bin z-score is optimal against. The autoencoder has no structural
advantage on this data and pays for extra capacity with a higher false-alarm rate.
The MLP is retained in the package for the case of *correlated multi-band shape
changes* that a per-bin marginal test cannot see — **that case is not demonstrated
in this benchmark** and remains an untested claim.

**Recommendation for users:** use `--detector baseline` (the CLI default) unless
you have evidence that your fault signatures are multi-band correlated.

Total benchmark runtime: **6.4 s** on 2 CPU cores (well inside the 3-minute budget).

---

## 4. What was NOT validated

- No comparison against real flight or ground-test vibration telemetry. All data
  is synthetic and idealized (see `DATASET_CARD.md`).
- No validation against a published reaction-wheel disturbance dataset or a
  microvibration test campaign; the generator's harmonic amplitudes are
  illustrative, not fitted to hardware.
- The pointing-loss model is validated only against Monte Carlo of its own stated
  assumptions — this confirms the algebra, not the physical adequacy of the
  point-receiver far-field model for any particular link.
- No validation of detector behaviour under nonstationary nominal conditions
  (slew, thermal transients, wheel speed sweeps), which would violate the
  stationarity assumption behind both detectors.
- The MLP's claimed advantage on correlated multi-band anomalies is untested.
