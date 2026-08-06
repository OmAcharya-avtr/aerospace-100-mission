# JitterScope

**Status:** TESTING · **Class:** medium · **Validation level:** 2 (Research) · **AI:** yes

## Executive overview

JitterScope characterizes platform jitter and vibration for optical pointing
applications, and screens the same telemetry for spectral anomalies.

It does four things:

1. **Spectral characterization** — Welch PSD estimation with the windowing and
   averaging parameters exposed and documented, not hidden behind defaults.
2. **Jitter budgets** — band-limited RMS jitter by integrating the PSD
   (σ² = ∫S df), plus the cumulative RMS curve that shows which bands actually
   spend the error budget.
3. **Link impact** — conversion of RMS jitter to average Gaussian-beam pointing
   loss via a closed form verified against Monte Carlo to within 1.5 sampling
   standard errors.
4. **Anomaly detection** — a per-band z-score baseline and an
   autoencoder-equivalent neural model, both producing a score, a flag, and a
   confidence value, benchmarked head-to-head on labeled synthetic faults.

On the benchmark in this package **the classical baseline beats the neural
model** (F1 0.9644 vs 0.9600). That result is reported as measured; see
[AI model details](#ai-model-details).

## Aerospace problem

Optical payloads — laser communication terminals, imaging instruments, optical
metrology — fail their pointing budget because of platform micro-vibration.
Reaction wheels, cryocoolers, gimbals, and thermal snap inject disturbance that
reaches the line of sight as jitter. Two questions follow, and this package
answers both:

- **How much does it cost the link?** A jitter budget is only meaningful once it
  becomes an optical penalty. RMS jitter that is a quarter of the beam divergence
  costs about 1 dB; jitter equal to the divergence costs about 7 dB. Which
  frequency band is responsible determines whether the fix is a fine-steering
  mirror, an isolator, or a wheel speed restriction.
- **Is the platform still behaving the way it did?** Vibration signatures drift
  as bearings wear, isolators degrade, and mechanisms age. A new tone or a shift
  in band energy is an early indicator, and it appears in the spectrum well
  before it appears in a mean or a peak-to-peak limit check.

## Intended users

- **Optomechanical / pointing engineers** building line-of-sight jitter budgets
  and allocating between structure, isolation, and active control.
- **Laser communication link engineers** converting a pointing budget into a
  link-budget penalty.
- **Telemetry and spacecraft operations analysts** screening vibration channels
  for spectral change over a mission.
- **Students and researchers** in spacecraft dynamics and free-space optics.

Users are assumed to understand PSD estimation and to treat every output as an
engineering input requiring review, not an answer.

## Engineering theory

### Power spectral density (Welch's method)

For a zero-mean wide-sense-stationary process `x(t)`, the one-sided PSD `S(f)`
satisfies the Wiener–Khinchin / Parseval relation

```
σ² = var(x) = ∫₀^∞ S(f) df
```

**Source:** Bendat & Piersol 2010, *Random Data: Analysis and Measurement
Procedures*, 4th ed., ch. 5.
**Units:** if `x` is a pointing angle in rad, `S(f)` is rad²/Hz.
**Method:** Welch 1967, *IEEE Trans. Audio Electroacoust.* 15(2):70–73 — the
record is split into overlapping segments, each windowed and periodogram-averaged,
trading frequency resolution (`Δf = f_s/nperseg`) for estimator variance. With
`k` averaged segments the estimator scatter falls as `1/√k`; this is verified in
[Validation](#validation) (measured 0.1048 against the 0.1021 χ² prediction for
k = 96).
**Window:** Hann by default, −31.5 dB first sidelobe (Harris 1978,
*Proc. IEEE* 66(1):51–83); 50 % overlap is the standard variance/efficiency
compromise for Hann.
**Assumptions and validity:** stationarity over the record and adequate sampling
(no aliasing). Accuracy degrades for strongly nonstationary data — a slew or a
wheel run-up violates the premise.

### Band-limited and cumulative RMS jitter

```
σ_band = √( ∫_{f₁}^{f₂} S(f) df )          σ_c(f) = √( ∫₀^{f} S(ν) dν )
```

**Source:** same Parseval relation (Bendat & Piersol 2010, ch. 5); standard
practice in jitter budgeting. **Units:** rad RMS for a rad input.
Integration is trapezoidal on the PSD grid with band edges included by linear
interpolation, so accuracy is bounded by the Welch resolution `f_s/nperseg`.
The cumulative curve is the diagnostic: its plateaus and steps identify which
bands own the budget.

### Jitter → Gaussian-beam pointing loss

Far-field intensity of a fundamental Gaussian beam versus off-axis angle θ:

```
I(θ) = I₀ exp(−2θ² / θ_div²)      →      L(θ) = exp(−2θ² / θ_div²)
```

where `θ_div` is the **1/e² half-angle divergence** in rad
(**source:** Siegman 1986, *Lasers*, ch. 17).

For zero-mean Gaussian jitter with per-axis standard deviation `σ_θ` on two
independent axes, the radial error θ is Rayleigh distributed,
`p(θ) = (θ/σ²)exp(−θ²/2σ²)`, and the Gaussian integral gives the closed form

```
⟨L_p⟩ = ∫₀^∞ exp(−2θ²/θ_div²) p(θ) dθ  =  1 / (1 + 4 (σ_θ / θ_div)²)
```

**Source:** this is the point-receiver limit of the pointing-error model of
Farid & Hranilovic 2007, *J. Lightwave Technol.* 25(7):1702–1710; see also
Andrews & Phillips 2005, *Laser Beam Propagation through Random Media*, 2nd ed.,
ch. 12 on pointing-error statistics.
**Units:** `σ_θ` and `θ_div` in rad; `⟨L_p⟩` is dimensionless mean normalized
received power in (0, 1]. dB penalty is `−10 log₁₀⟨L_p⟩`.
**Verified:** against 10⁶-draw Monte Carlo at eight jitter ratios; every case
agrees within 1.5 MC standard errors, max relative error 4.068e-03. See
[Validation](#validation).
**Assumptions and validity range:** unobscured TEM00 beam; far field; point
receiver (aperture ≪ beam footprint); zero static boresight bias; isotropic,
equal-variance jitter on both axes; no atmospheric scintillation or beam wander.
**Biased or anisotropic jitter breaks the closed form** and requires numerical
integration — the function does not detect this and will return a wrong answer if
misapplied.

### Anomaly detection

Both detectors reduce a 1 s window to a 24-element log-PSD feature vector
(log-spaced bins, 1 Hz → Nyquist). Log-energy band features are standard in
vibration condition monitoring (Randall 2011, *Vibration-based Condition
Monitoring*, Wiley, ch. 3).

- **Baseline** — per-bin Gaussian z-score, window score `max_b |(x_b−μ_b)/σ_b|`;
  the control-chart limit check of Randall 2011, ch. 3, in the spirit of the
  fixed band limits of ISO 10816.
- **Neural model** — reconstruction error of a bottleneck `MLPRegressor`
  autoencoder (Hinton & Salakhutdinov 2006, *Science* 313:504–507; reconstruction
  -error scoring per Sakurada & Yairi 2014, MLSDA workshop).

Both threshold at the 0.995 quantile of **held-out** nominal scores.

## Architecture

```
src/jitterscope/
├── psd.py         Welch PSD, band_rms, cumulative_rms, input/NaN policy
├── pointing.py    pointing_loss_avg (closed form) + pointing_loss_avg_mc (MC check)
├── telemetry.py   seeded synthetic vibration generator + fault injection
├── detect.py      FeatureExtractor, BandZScoreBaseline, NominalModel, detect
└── __main__.py    argparse CLI: python -m jitterscope analyze
```

Data flows one way: `telemetry → psd → features → model → DetectionResult`.
`psd.py` and `pointing.py` are pure numeric functions with no ML dependency, so
the signal core is usable on its own. No cross-product imports.

## Installation

Requires Python 3.11 with numpy, scipy, scikit-learn, and matplotlib.

```bash
cd products/P005
pip install -e .            # or: export PYTHONPATH=$PWD/src
pip install -e ".[test]"    # adds pytest + hypothesis
```

## Quick start

```python
import numpy as np
from jitterscope import (generate_telemetry, psd, band_rms, cumulative_rms,
                         pointing_loss_avg, FeatureExtractor,
                         BandZScoreBaseline, detect)

# 60 s of synthetic nominal telemetry at 1 kHz (wheel tones at 45/90/135 Hz)
t, x, _ = generate_telemetry(duration_s=60.0, fs=1000.0, seed=2026)

# Spectrum and jitter budget
f, pxx = psd(x, fs=1000.0, nperseg=4096)          # Welch knobs are yours
rms = band_rms((f, pxx), [(0.5, 10), (10, 40), (40, 100), (100, 500)])
_, cum = cumulative_rms((f, pxx))
print(rms * 1e6, "µrad per band;  total", cum[-1] * 1e6, "µrad")

# Link impact for a 10 µrad divergence beam (per-axis sigma)
loss = pointing_loss_avg(cum[-1] / np.sqrt(2), theta_div=10e-6)
print(f"pointing loss {loss:.4f} = {-10 * np.log10(loss):.2f} dB")

# Anomaly detection: fit on nominal, score a faulty record
ext = FeatureExtractor(fs=1000.0)
model = BandZScoreBaseline().fit(ext.transform(x)[0])
_, x_bad, _ = generate_telemetry(60.0, 1000.0, seed=777, faults=[
    {"kind": "new_tone", "t_start": 20.0, "freq_hz": 137.0, "rms": 1e-6}])
res = detect(x_bad, model=model, extractor=ext)
print(f"{res.n_anomalous}/{res.scores.size} windows flagged")
```

### CLI

```bash
python -m jitterscope analyze --input telemetry.csv --fs 1000
```

Actual output on a 30 s record with a 310 Hz tone injected at t = 20 s:

```
jitterscope analyze: telemetry.csv
  samples: 30000  fs: 1000 Hz  duration: 30.00 s
  mean: -8.4833e-08  std: 1.5519e-06 (signal units)

PSD (Welch, hann, nperseg=1024, 50% overlap)
  resolution: 0.977 Hz   total RMS (Parseval): 1.5099e-06
  dominant peak: 309.57 Hz at 6.777e-13 u^2/Hz

Band-limited RMS (sigma = sqrt(int PSD df)):
  band [Hz]          RMS
      0.0-   10.0   6.8034e-07
     10.0-   50.0   5.8193e-07
     50.0-  200.0   3.6139e-07
    200.0-  500.0   1.1609e-06

Anomaly report (baseline detector, threshold = q0.995 of nominal scores = 3.508)
  training on first 50% of record (ASSUMED nominal)
  windows: 59   flagged: 21
  t_center [s]    score        confidence
       12.50    3.599       0.966
       20.00    49.08       1.000
       20.50    57.49       1.000
       ...
```

The CLI assumes the leading `--train-frac` of the record is nominal. **That
assumption is the operator's to verify** — a fault present during the training
window is learned as normal and never flagged.

## Configuration

| API | Key parameters |
|---|---|
| `psd(x, fs, **welch_kw)` | `nperseg` (resolution `fs/nperseg`), `window` (`"hann"`), `noverlap` (50 %), `detrend` (`"constant"`), `average` (`"mean"`, or `"median"` for robustness to transients) — all forwarded to `scipy.signal.welch` |
| `band_rms(psd, bands)` | `bands` as a list of `(f_lo, f_hi)` in Hz |
| `pointing_loss_avg(sigma_theta, theta_div)` | both in rad; `theta_div` is the 1/e² **half**-angle |
| `FeatureExtractor` | `fs`, `window_s` (1.0), `overlap` (0.5), `n_bins` (24), `f_min` (1.0) |
| `NominalModel` | `hidden` ((16,6,16)), `quantile` (0.995), `seed`, `max_iter`, `calib_frac` (0.3), `alpha` |
| `BandZScoreBaseline` | `quantile` (0.995) |
| `detect(telemetry, threshold, *, model, extractor)` | `threshold=None` uses the model's calibrated `threshold_` |

CLI flags: `--input`, `--fs`, `--bands`, `--nperseg`, `--detector {baseline,mlp}`,
`--train-frac`, `--window-s`, `--quantile`.

**NaN policy (explicit):** non-finite samples raise `ValueError` at every entry
point. Nothing is imputed or silently dropped, because interpolating over a
telemetry dropout fabricates spectral content and can hide the very event worth
detecting. Callers must clean gaps deliberately. Tested in
`tests/test_psd.py::TestInputValidation`.

## Examples

Both scripts run standalone and write PNGs to `screenshots/`.

```bash
cd examples
python psd_cumulative_rms.py     # -> ../screenshots/psd_cumulative_rms.png
python anomaly_timeline.py       # -> ../screenshots/anomaly_timeline.png
```

**`psd_cumulative_rms.py`** — PSD with the 45/90/135 Hz wheel harmonics marked,
plus the cumulative RMS curve. Measured output: band RMS 0.683 / 0.270 / 0.579 /
0.243 / 0.216 µrad across 0.5–10, 10–40, 40–100, 100–250, 250–500 Hz; total
1.019 µrad; pointing loss 0.9797 (0.09 dB) for a 10 µrad divergence. The
cumulative curve steps visibly at each wheel harmonic — that is the diagnostic.

**`anomaly_timeline.py`** — anomaly-score timelines for both detectors against a
record with three injected faults (new tone at 20 s, band shift at 35 s,
transients from 48 s), with onsets marked and flagged windows highlighted.

## Validation

Full evidence, method, and raw script output: **[`validation/VALIDATION.md`](validation/VALIDATION.md)**.
All numbers below come from running the scripts in this session; raw output is
saved in `validation/*_output.txt`. Nothing was tuned after seeing a result.

| Check | Result | Reference |
|---|---|---|
| Sinusoid integrated power | 2.000001e-12 vs 2.000000e-12 rad² — **rel err 5.24e-07** | Parseval, Bendat & Piersol 2010 ch. 5 |
| Sinusoid peak frequency | 80.0781 Hz vs 80.0000 Hz — **0.320 bins** (Δf = 0.2441 Hz) | Welch 1967 |
| White-noise PSD level | 1.987657e-15 vs 2.000000e-15 rad²/Hz — **rel err 6.17e-03** | σ²/(f_s/2) |
| White-noise Parseval | ∫S df 9.956535e-13 vs variance 9.958205e-13 rad² — **rel err 1.68e-04** | Bendat & Piersol 2010 |
| Estimator scatter | measured 0.1048 vs χ² prediction 0.1021 (k = 96) | Welch variance theory |
| **Pointing loss vs Monte Carlo** | **max rel err 4.068e-03** over 8 ratios, 10⁶ draws each; every case **within 1.5 MC standard errors** | Farid & Hranilovic 2007; Andrews & Phillips 2005 ch. 12 |
| Detector F1 (baseline / MLP) | **0.9644 / 0.9600** on 2856 labeled windows | `val_detector.py` |

Pointing-loss closed form against Monte Carlo, `θ_div = 12 µrad`:

| σ/θ_div | Closed form | Monte Carlo | Rel. error | \|err\|/SE |
|---|---|---|---|---|
| 0.25 | 0.800000 | 0.799911 | 1.114e-04 | 0.55 |
| 0.50 | 0.500000 | 0.499917 | 1.659e-04 | 0.29 |
| 1.00 | 0.200000 | 0.199940 | 3.020e-04 | 0.23 |
| 2.00 | 0.058824 | 0.058584 | 4.068e-03 | 1.46 |

Every deviation is consistent with pure sampling noise, so the closed form is
confirmed rather than merely approximated.

**Not validated:** no comparison against real flight or ground-test telemetry; no
comparison against a published reaction-wheel disturbance dataset; no validation
under nonstationary conditions; the neural model's claimed advantage on
correlated multi-band anomalies is untested. See VALIDATION.md §4.

## Benchmark results

Detector benchmark on 24 labeled synthetic records (2856 windows, 952 faulty;
train seeds 1000–1003, test seeds 2000–2023, disjoint):

| Model | Precision | Recall | F1 | TP | FP | FN | ROC AUC | Fit time |
|---|---|---|---|---|---|---|---|---|
| **Baseline band z-score** | **0.9900** | 0.9401 | **0.9644** | 895 | 9 | 57 | **0.9797** | **0.001 s** |
| MLP autoencoder | 0.9751 | **0.9454** | 0.9600 | 900 | 23 | 52 | 0.9784 | 2.452 s |

Per-fault recall — `new_tone` 1.000/1.000, `band_shift` 1.000/1.000,
`transient` 0.7543/0.7759 (baseline/MLP). False alarms on 714 nominal windows:
baseline 4 (0.56 %), MLP 10 (1.40 %).

Runtime: full benchmark 6.4 s; end-to-end pipeline (generate 60 s + fit + score
40 s) ~3 s; PSD of 10⁶ samples ~50 ms. All on 2 CPU cores, no GPU, far inside the
3-minute budget. A regression test guards a 30 s pipeline ceiling.

## AI model details

Full card: **[`MODEL_CARD.md`](MODEL_CARD.md)** · Dataset: **[`DATASET_CARD.md`](DATASET_CARD.md)**

> **This model is not certified for operational flight use.**

- **Baseline first.** `BandZScoreBaseline` (per-band log-energy z-score,
  Randall 2011 ch. 3) was implemented and calibrated before the neural model, and
  both are benchmarked on identical features, threshold rule, and held-out data.
- **Architecture.** `MLPRegressor(24 → 16 → **6** → 16 → 24)`, tanh, Adam,
  `alpha=1e-3`, `max_iter=3000` — an autoencoder-equivalent construction
  (PyTorch is unavailable in this environment). Score = reconstruction MSE on
  standardized log-PSD features. ~1000 parameters.
- **Dataset.** Entirely synthetic, generated by committed seeded code; no data
  files committed. 4 nominal training records; 24 test records across 4 classes.
  Unmodeled effects — nonstationarity, multi-axis coupling, structural transfer
  functions, sensor dropouts, control-loop shaping, non-Gaussian statistics — are
  enumerated in the dataset card. **Performance here does not transfer to real
  telemetry.**
- **Test split.** Three-way by seed. Within training, a seeded 70/30 split holds
  out 30 % of nominal windows purely for threshold calibration, because in-sample
  reconstruction error underestimates unseen error and would set an optimistically
  low threshold. Test records (2000–2023) are never seen in fitting or calibration.
- **Metrics.** See [Benchmark results](#benchmark-results).
- **Uncertainty output.** `confidence()` returns the empirical CDF of the score
  under the held-out nominal distribution, in [0, 1]; measured mean 0.5066 on
  nominal windows vs 0.9781 on faulty ones. It is a calibrated *how abnormal*
  measure, **not** a probability of failure, and it saturates at 1.0 so it cannot
  rank severity among strong anomalies.
- **Failure cases.** Transients under-detected (recall ~0.75–0.78; a 0.25 s burst
  is diluted by a 1 s window); MLP false-alarm rate ~2.8× its design target;
  nonstationary nominal operation (slews, wheel run-ups, thermal snap) is flagged
  as anomalous; a fault present during training is learned as nominal; faults that
  preserve band energy distribution are invisible; single channel only.
- **Reproducibility.** `python validation/val_detector.py` (~6 s). Seeds:
  train 1000–1003, test 2000–2023, `random_state=0` (fixes both weight init and
  the calibration split). Environment: Python 3.11, numpy 2.4.4, scipy 1.17.1,
  scikit-learn 1.8.0.

### The honest result

**The classical baseline wins:** F1 0.9644 vs 0.9600, ROC AUC 0.9797 vs 0.9784,
2400× faster to fit, and it hits its design false-alarm rate (0.56 % against a
0.5 % target) where the MLP runs hot at 1.40 %. The margin is under 1 % on both
metrics, and both models detect `new_tone` and `band_shift` perfectly — the
entire difference sits in the `transient` class, where the MLP buys 0.02 recall
with 14 extra false alarms.

The reason is structural: the injected faults are additive-energy signatures that
raise log-PSD in specific bins, which is exactly the alternative hypothesis a
per-bin z-score is optimal against. The autoencoder has no advantage to exploit
here. It is retained for correlated multi-band shape changes that a per-bin
marginal test cannot see — **an untested claim on this dataset**. The CLI
therefore defaults to `--detector baseline`.

## Hardware requirements

CPU-only; no GPU. Developed and validated on 2 cores. Peak memory under ~200 MB
for a 10⁶-sample record. Python 3.11 with numpy ≥ 1.26, scipy ≥ 1.11,
scikit-learn ≥ 1.3, matplotlib ≥ 3.8. Full test suite ~16 s; full validation
suite ~30 s.

## Limitations

- **Synthetic data only.** Nothing here has been checked against real flight,
  ground-test, or laboratory vibration telemetry. Every performance number
  describes behaviour on idealized synthetic signals.
- **Stationarity is assumed everywhere.** Welch estimation and both detectors
  assume the nominal spectrum is stationary. Slews, wheel speed sweeps, thermal
  transients, and mode changes violate this and will be misread — this is the
  dominant expected failure mode on real telemetry.
- **Single channel.** No multi-axis processing, no cross-spectra, no coherence,
  no structural transfer function from disturbance source to line of sight.
- **Pointing loss is point-receiver and unbiased-jitter only.** Static boresight
  bias, anisotropic jitter, aperture averaging, beam truncation, obscuration, and
  atmospheric effects are outside the model. The function cannot detect misuse.
- **Transient detection is weak** (recall ~0.75–0.78) because sub-second bursts
  are averaged away in 1 s windows. No kurtosis or crest-factor feature is
  implemented.
- **The neural model does not beat the baseline** on the available evidence, and
  its motivating case (correlated multi-band anomalies) is untested.
- **Confidence is not a failure probability** and saturates at 1.0.
- **No gap handling.** NaN/Inf raises rather than imputing — deliberate, but it
  means real telemetry with dropouts needs preprocessing this package does not
  provide.
- **The CLI's nominal-training assumption is unverified by the tool.** It trusts
  the operator that the leading `--train-frac` is fault-free.
- Deviations from the build guide: none.

## Safety statement

This software is research-grade. It is not flight-qualified, not certified, and
not approved for operational aerospace use. Outputs are engineering aids for human
review, never a basis for autonomous action, fault declaration, safe-mode entry,
or hardware disposition.

## Roadmap

- Multi-axis input with cross-spectral density and coherence.
- Time-frequency features (spectrogram, kurtosis, crest factor) to fix transient
  recall.
- Nonstationarity handling: wheel-speed-conditioned nominal models, order tracking.
- Validation against a published reaction-wheel microvibration dataset.
- Aperture-averaged and biased-jitter pointing loss by numerical integration.
- Sequential detection (CUSUM) over window scores to reduce false alarms.

## License

Apache-2.0. See [`LICENSE`](LICENSE). Copyright © 2026 OPTIMA Organisation.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

## Citation

```bibtex
@software{jitterscope_2026,
  title  = {JitterScope: platform jitter and vibration characterization
            for optical pointing with telemetry anomaly detection},
  author = {{OPTIMA Organisation}},
  year   = {2026},
  version = {0.1.0},
  license = {Apache-2.0}
}
```

Key references implemented in this package:

- Welch, P. D. (1967). *IEEE Trans. Audio Electroacoust.* 15(2):70–73.
- Bendat, J. S. & Piersol, A. G. (2010). *Random Data: Analysis and Measurement Procedures*, 4th ed. Wiley.
- Harris, F. J. (1978). *Proc. IEEE* 66(1):51–83.
- Siegman, A. E. (1986). *Lasers*. University Science Books.
- Farid, A. A. & Hranilovic, S. (2007). *J. Lightwave Technol.* 25(7):1702–1710.
- Andrews, L. C. & Phillips, R. L. (2005). *Laser Beam Propagation through Random Media*, 2nd ed. SPIE Press.
- Masterson, R. A., Miller, D. W. & Grogan, R. L. (2002). *J. Sound Vib.* 249(3):575–598.
- Kasdin, N. J. (1995). *Proc. IEEE* 83(5):802–827.
- Randall, R. B. (2011). *Vibration-based Condition Monitoring*. Wiley.
- Hinton, G. E. & Salakhutdinov, R. R. (2006). *Science* 313:504–507.
- Sakurada, M. & Yairi, T. (2014). *Proc. MLSDA 2014 Workshop*, ACM.
