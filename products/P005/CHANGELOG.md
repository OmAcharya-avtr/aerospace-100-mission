# Changelog

All notable changes to JitterScope are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-06

Initial release. Status: TESTING. Validation level 2 (Research).

### Added

**Signal core**
- `psd(x, fs, **welch_kw)` — one-sided Welch PSD (Welch 1967) with `nperseg`,
  `window`, `noverlap`, `detrend`, and `average` exposed and documented.
- `band_rms(psd, bands)` — band-limited RMS jitter by PSD integration
  (σ² = ∫S df), with band edges interpolated onto the PSD grid.
- `cumulative_rms(psd)` — cumulative RMS curve σ_c(f) for jitter budgeting.
- Explicit NaN/Inf policy: non-finite input raises `ValueError`; nothing is
  imputed or silently dropped.

**Pointing loss**
- `pointing_loss_avg(sigma_theta, theta_div)` — closed-form average Gaussian-beam
  pointing loss `⟨L⟩ = 1/(1 + 4(σ_θ/θ_div)²)` for radial Gaussian jitter
  (point-receiver limit of Farid & Hranilovic 2007; Andrews & Phillips 2005 ch. 12).
- `pointing_loss_avg_mc(...)` — seeded Monte Carlo cross-check of the same quantity.

**Synthetic telemetry**
- `generate_telemetry(...)` — seeded, deterministic multi-band vibration process:
  white base noise, ~1/f² colored noise (Kasdin 1995), and reaction-wheel-like
  harmonics (Masterson et al. 2002), with injectable `new_tone`, `band_shift`, and
  `transient` fault signatures and a per-sample ground-truth fault mask.

**Anomaly detection**
- `FeatureExtractor` — windowed log-spaced log-PSD feature vectors.
- `BandZScoreBaseline` — classical per-band z-score baseline (implemented first).
- `NominalModel` — autoencoder-equivalent `MLPRegressor` (24→16→6→16→24) scored by
  reconstruction error, with a held-out calibration split for thresholding.
- `detect(...)` returning `DetectionResult` with per-window scores, flags, and a
  confidence (empirical nominal CDF) channel on both models.

**Interfaces and artifacts**
- CLI: `python -m jitterscope analyze --input telemetry.csv --fs 1000`, with
  `--bands`, `--nperseg`, `--detector`, `--train-frac`, `--window-s`, `--quantile`.
- Examples producing PNGs: `psd_cumulative_rms.py`, `anomaly_timeline.py`.
- 53 tests (PSD known-answers, Hypothesis Parseval property, seeded pointing-loss
  vs Monte Carlo, detector reproducibility, input validation, end-to-end
  integration, runtime benchmarks).
- `validation/` with three rerunnable scripts and their saved raw output;
  `VALIDATION.md`, `MODEL_CARD.md`, `DATASET_CARD.md`.

### Validation summary

- Sinusoid integrated power: 2.000001e-12 vs 2.000000e-12 rad² (rel err 5.24e-07);
  peak within 0.320 frequency bins.
- White-noise PSD level rel err 6.17e-03; Parseval rel err 1.68e-04; estimator
  scatter 0.1048 vs χ² prediction 0.1021.
- Pointing loss vs 10⁶-draw Monte Carlo: max relative error 4.068e-03 across eight
  jitter ratios; all cases within 1.5 MC standard errors.
- Detector benchmark on 2856 labeled windows: baseline F1 **0.9644**
  (P 0.9900 / R 0.9401, AUC 0.9797) vs MLP F1 **0.9600**
  (P 0.9751 / R 0.9454, AUC 0.9784).

### Known limitations

- The classical baseline outperforms the neural model on the benchmark; reported
  as measured. The MLP's motivating case (correlated multi-band anomalies) is
  untested.
- Synthetic data only; no validation against real flight or ground-test telemetry.
- Stationarity assumed throughout; slews and wheel run-ups will be misread.
- Single channel; no multi-axis or cross-spectral processing.
- Transient recall ~0.75–0.78 (sub-second bursts diluted by 1 s windows).
- Pointing loss valid only for unbiased isotropic jitter and a point receiver.

[0.1.0]: https://example.invalid/optima/jitterscope/releases/tag/v0.1.0
