# Changelog

All notable changes to WaveLab are documented in this file.

## [0.1.0] - 2026-08-29

Initial release.

### Added

- `wavelab.zernike`: Noll-indexed Zernike polynomials, analytic gradients,
  basis/slope interaction matrices, and least-squares fitting of sampled
  wavefronts to coefficients.
- `wavelab.geometry`: circular-pupil `PupilGrid`, Hudgin (1977) and Fried
  (1977) zonal finite-difference geometry matrices, waffle-pattern
  diagnostic, and unconstrained-point pruning.
- `wavelab.linalg`: Tikhonov and truncated-SVD regularized least-squares
  solvers, null-space extraction, and analytic noise-propagation
  coefficients.
- `wavelab.modal.ModalReconstructor`: regularized modal (Zernike) slope-to-
  coefficient least-squares baseline, with subaperture-dropout row selection.
- `wavelab.zonal.ZonalReconstructor`: regularized zonal (Hudgin/Fried) slope-
  to-phase least-squares reconstructor, with explicit piston/waffle null-
  space handling.
- `wavelab.screens.kolmogorov_screen`: synthetic Kolmogorov phase screens via
  the FFT method (Roddier 1981 PSD, McGlamery 1976 / Schmidt 2010 algorithm).
- `wavelab.noise`: photon-shot-noise slope model and subaperture-dropout
  sampling.
- `wavelab.dataset`: deterministic synthetic slopes-to-Zernike dataset
  generation for training and benchmarking.
- `wavelab.ml.ZernikeSlopeEnsemble`: learned slopes-to-Zernike reconstructor,
  an ensemble of `MLPRegressor` networks with ensemble-spread uncertainty.
- `python -m wavelab` CLI: `geometry`, `reconstruct`, `demo-benchmark`
  subcommands.
- Validation suite (`validation/`): noise-free recovery for all three
  reconstructors, photon-flux noise-propagation check against the analytic
  coefficient, and a learned-vs-baseline benchmark across photon flux and
  subaperture-dropout rate, all with saved raw output.
- 180 tests (`tests/`): unit, input-validation, hand-calculated known-answer,
  edge-case, Hypothesis property, integration, and pinned-seed regression
  tests.
- `MODEL_CARD.md`, `DATASET_CARD.md`, `README.md`.

### Known limitations (see README "Limitations")

- The learned reconstructor loses to the regularized least-squares baseline
  at every tested operating point except very high subaperture dropout
  (measured >= 60% in this benchmark), where the baseline's fixed
  regularization strength becomes unstable on a severely under-determined
  system; this is reported as measured, not tuned toward a preferred
  narrative.
- The learned ensemble's uncertainty output is not a calibrated 1-sigma
  error bar (measured ratio in `MODEL_CARD.md`).
- `kolmogorov_screen` does not implement subharmonic compensation and
  therefore under-represents low-order turbulent content (Lane, Glindemann &
  Dainty 1992).
