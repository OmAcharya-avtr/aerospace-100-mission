# Changelog

All notable changes to BeamTwin are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-08-06

Initial MVP release. Status: **TESTING**.

### Added

**`beamtwin.budget` — deterministic link budget**
- Gaussian-beam propagation: divergence half-angle, Rayleigh range, beam radius (Saleh & Teich 2007).
- Geometric capture fraction on a centred circular aperture, `eta = 1 - exp(-2a^2/w^2)`.
- Static pointing loss (point-receiver approximation).
- Beer-Lambert atmospheric attenuation, plus a Kim-model helper converting visibility to dB/km (Kim et al. 2001, SPIE 4214) including the wavelength-independent dense-fog branch.
- `LinkParams` / `LinkBudget` dataclasses with validation and a `margin_negative` flag.
- dBm/watt and fraction/dB conversion helpers.

**`beamtwin.channel` — stochastic atmospheric channel**
- Plane-wave Rytov variance and lognormal scintillation (Andrews & Phillips 2005), with a `weak_regime_valid` flag when `sigma_R^2 >= 1`.
- Gaussian per-axis pointing jitter with the closed-form mean pointing loss `1/(1 + 4 sigma_d^2/w^2)`.
- Vectorised, seeded Monte Carlo of combined received power (peak 1.3e7 samples/s measured).

**`beamtwin.stats` — fade statistics**
- Fade probability with 95 % Wilson score confidence interval.
- Closed-form lognormal fade-probability baseline (Q-function in the log domain).
- Fade-margin percentiles, mean, and variance.

**`beamtwin.surrogate` — ML surrogate (AI component)**
- 5-member bootstrap `GradientBoostingRegressor` ensemble predicting `log10 P_fade` from 5 features.
- Ensemble-spread uncertainty output and a training-domain extrapolation flag.
- Committed trained model (`models/surrogate.joblib`) and deterministic dataset generation.

**`beamtwin.scenario` and CLI**
- YAML scenario loader with strict validation (unknown keys rejected).
- `python -m beamtwin run scenario.yaml` — text + JSON twin report.
- `python -m beamtwin sweep` — parameter sweeps with PNG output.

**Documentation and evidence**
- `docs/REQUIREMENTS.md` — 19 numbered requirements with a verification matrix and the error-handling policy.
- `validation/VALIDATION.md` — hand-checked budget, analytic limit cases, uncertainty analysis, and an explicit "what was NOT validated" section.
- `MODEL_CARD.md`, `DATASET_CARD.md`.
- Four rerunnable validation scripts with saved raw output.
- Three runnable examples producing PNGs in `screenshots/`.
- 251 tests (unit, integration, regression, performance, failure-mode, configuration, property-based, reproducibility).

### Known limitations at 0.1.0

- No validation against measured FSO link data — all evidence is internal consistency or agreement with closed-form theory.
- Surrogate uncertainty is **not calibrated** (±2σ covers 39.9 % of held-out cases, not 95 %); usable only as a relative confidence ranking.
- Fade probabilities below ~1e-4 are unresolvable at default sample counts; the surrogate's target is floored there.
- Strong-turbulence regime (`sigma_R^2 >= 1`) is detected and flagged but not modelled.
- No aperture averaging, beam wander, temporal correlation, or fade-duration statistics.
- Surrogate trained at 1550 nm only; wavelength is not a feature.

See `README.md` § Limitations and `validation/VALIDATION.md` §7 for the complete list.

[0.1.0]: https://example.invalid/beamtwin/releases/0.1.0
