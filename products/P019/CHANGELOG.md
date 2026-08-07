# Changelog

All notable changes to CnCast are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-07

Initial release. Status: TESTING.

### Added — baselines (implemented first)

* `cncast.baselines.hufnagel_valley` — Hufnagel (1974) / Valley (1980) profile
  with pseudowind and ground-level Cn² as explicit arguments; validity 0–20 km AGL.
* `cncast.baselines.hv57` — the HV 5/7 parameterisation (v = 21 m/s,
  A = 1.7e-14 m^-2/3). Verified to give r₀ = 4.9624 cm and θ₀ = 7.0109 µrad at
  500 nm, zenith.
* `cncast.baselines.slc_day` / `slc_night` — Beland (1993) piecewise AMOS fits,
  validity 0–20.5 km / 0–20 km above the site, identically zero above.
* `cncast.baselines.bufton_wind` / `rms_high_altitude_wind` — Bufton (1973) wind
  profile and its 5–20 km rms.

### Added — derived quantities

* `cncast.seeing` — turbulence moments, Fried parameter r₀, isoplanatic angle
  θ₀, Greenwood frequency f_G, long-exposure seeing FWHM, each with source,
  units, assumptions and validity range in its docstring, and explicit zenith
  handling via the plane-parallel sec ζ law.

### Added — synthetic dataset

* `cncast.dataset` — seeded scenario generator (surface meteorology + latent
  profile state) and pure `profile_cn2` truth function, scenario-level splits,
  and feature construction. Master seed 20260807. See `DATASET_CARD.md`; the
  data is synthetic and no measurements are used anywhere.

### Added — learned model

* `cncast.model.CnCastModel` — three quantile gradient-boosting regressors
  (α = 0.05 / 0.50 / 0.95) with pointwise quantile-crossing repair, plus split
  conformalised quantile regression (Romano et al. 2019) for calibrated
  intervals. Every prediction carries lower/median/upper bounds and an
  `extrapolating` flag.
* `Hv57Baseline`, `SlcBaseline`, `ClimatologyBaseline` — comparators scored
  alongside the learned model on the same held-out rows.
* `train_default_model` — the single seeded recipe used by the validation
  scripts, examples and CLI.

### Added — CLI, examples, validation

* `python -m cncast baseline|predict`.
* `examples/profile_with_intervals.py`, `examples/r0_comparison.py` — both run,
  both save PNGs to `screenshots/` with the Agg backend.
* `validation/validate_baselines.py`, `validation/benchmark_ml.py` with their
  committed raw output, and `validation/VALIDATION.md` (Level 2) including a
  hand check of r₀ from a predicted profile with the arithmetic shown.
* `MODEL_CARD.md` (fifteen items), `DATASET_CARD.md`.

### Measured in this release (held-out, actual run)

* Learned model RMSE 0.2095 dex vs HV 5/7 0.5665 dex and training climatology
  0.3102 dex — see the honest reading of that result in `MODEL_CARD.md` §7.
* Interval coverage 0.8988 against 0.900 nominal after conformal calibration
  (0.8033 before).
* Fit + calibration 15.9 s on 2 CPU cores against a 120 s budget.

### Known limitations at 0.1.0

Synthetic training data; no forecast horizon; site-agnostic; plane-wave and
plane-parallel geometry only; only the 90 % interval is calibrated. Full list in
`README.md` § Limitations and `MODEL_CARD.md` §10.
