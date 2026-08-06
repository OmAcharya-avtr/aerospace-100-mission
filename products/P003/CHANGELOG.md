# Changelog

All notable changes to ScintiNet are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-06

Initial release. Status: TESTING.

### Added

- **Analytic core** (`scintinet.rytov`): plane- and spherical-wave Rytov
  variance σ_R² = 1.23 Cn² k^(7/6) L^(11/6) (0.50 coefficient for spherical),
  weak-fluctuation scintillation index, and the Andrews (1992) circular
  aperture-averaging factor A = [1 + 1.062 kD²/(4L)]^(−7/6). Sources cited
  in every docstring with units, assumptions and validity range.
- **Split-step simulator** (`scintinet.simulator`): paraxial angular-spectrum
  propagation between FFT-synthesised Kolmogorov phase screens, with
  documented and run-time-enforced sampling rules (Fresnel-scale resolution,
  domain size, per-screen Fried parameter). Point and aperture-averaged σ_I²
  estimation. Fully seeded and reproducible.
- **Surrogate** (`scintinet.surrogate`): 5-member `MLPRegressor` ensemble
  predicting σ_I² from (Cn², L, λ, D), with `predict(..., return_std=True)`
  ensemble-spread uncertainty, plus the `rytov_baseline` analytic reference
  on the identical interface.
- **Validation campaign** (`validation/run_campaign.py`): reduced-scale seeded
  simulation campaign producing `validation/dataset.csv` (54 rows) in 22.6 s.
- **Validation evidence**: `validation/VALIDATION.md` plus raw script outputs
  `sim_vs_theory.txt`, `benchmark_results.txt`, `campaign_log.txt`.
- **Examples**: `sweep_sigma_i2.py` (simulation vs theory vs surrogate with
  uncertainty band) and `phase_screen_speckle.py` (phase screen + speckle
  field), both producing PNGs in `screenshots/`.
- **Documentation**: `README.md`, `MODEL_CARD.md`, `DATASET_CARD.md`.
- **Tests**: 50 tests covering hand-checked known answers, Hypothesis
  property tests, simulator sanity (energy conservation, zero-turbulence),
  seeded reproducibility, surrogate pipeline, input validation, an
  end-to-end integration test, and a benchmark/regression test.

### Known issues

- FFT phase screens have no subharmonic augmentation, so sub-fundamental
  spatial frequencies are missing. Measured effect: simulated σ_I² runs ~2 %
  low for a point receiver and ~15 % low averaged over 50–100 mm apertures
  relative to Andrews theory. Documented in `validation/VALIDATION.md` §V3.
- The MLP surrogate does **not** beat the Rytov analytic baseline in-regime
  (RMSE log10 0.0781 vs 0.0429 on 14 held-out points). Reported as measured;
  see `MODEL_CARD.md` for when a surrogate is worthwhile.
- Dataset rows sharing a simulation seed (different apertures) are
  correlated; the benchmark's random row split leaks mildly across
  train/test. A group split would be stricter.
- Weak-fluctuation regime only (σ_R² ≲ 1). Plane wave, horizontal
  homogeneous path, Kolmogorov spectrum with no inner/outer scale.
