# Changelog

All notable changes to TurbScope are documented in this file.

## [0.1.0] - 2026-08-29

Initial release.

### Added

* `turbscope.scintillometer`: Rytov weak-fluctuation forward model
  (`rytov_variance`, plane/spherical wave coefficients), the closed-form
  weak-regime inversion baseline (`invert_cn2_weak`), a heuristic weak-to-
  saturated scintillation-index bridging model
  (`scintillation_index_full`), and a multi-root inversion
  (`invert_cn2_all_roots`) that demonstrates and quantifies the saturation
  failure mode.
* `turbscope.dimm`: DIMM differential-motion forward model
  (`differential_variance`, longitudinal/transverse), Fried-parameter
  conversions, and closed-form inversion (`invert_cn2_from_variance`).
* `turbscope.inversion`: classical closed-form inversion WITH exact linear
  uncertainty propagation, and inverse-variance multi-sensor fusion.
* `turbscope.synthetic` / `turbscope.dataset`: seeded synthetic scenario and
  multi-sensor measurement generator, feature-table construction with
  scenario-grouped splits.
* `turbscope.model`: `TurbScopeModel` (quantile-GBR multi-sensor regressor
  with split-conformal-calibrated prediction intervals) plus three baseline
  comparators (`ScintillometerWeakBaseline`, `DimmOnlyBaseline`,
  `MeanTrainingBaseline`).
* `python -m turbscope` CLI: `forward`, `invert`, `predict` subcommands.
* 122 tests: unit, input-validation, hand-derived known-answer, edge-case,
  Hypothesis property, integration and CLI tests. All pass.
* Validation suite: `round_trip_recovery.py` (weak-regime recovery, median
  3.9% relative error with realistic sensor noise), `saturation_regime.py`
  (saturation demonstration and quantification, median 89.5% baseline error
  outside the weak regime), `benchmark_ml.py` (learned model vs three
  closed-form baselines, prediction-interval coverage). Raw outputs
  committed alongside `VALIDATION.md`.
* Two runnable examples producing `screenshots/*.png`.
* `MODEL_CARD.md`, `DATASET_CARD.md` per mission AI-product requirements,
  including the honest finding that the DIMM-only closed-form baseline
  beats the learned model in this synthetic design (the mandated
  scintillometer-only baseline does not).
