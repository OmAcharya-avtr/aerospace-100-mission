# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-16

First release. Validation level 2, research grade.

### Added

- `degradepredict.limits`: declared scalar limits with a direction and an
  explicit tolerance band, a three-valued crossing test, and linear
  interpolation of the first crossing time of a sampled trajectory.
- `degradepredict.generate`: seeded synthetic degradation trajectories from
  three declared mean shapes (linear, exponential, and the sublinear power-law
  capacity fade of Wang et al. 2011) with known crossing times, independent
  right censoring, and the operational selection that a unit which has already
  crossed is not a forecasting case.
- `degradepredict.baselines`: ordinary least squares of a transformed metric on
  a transformed time, inverted for the crossing time, with both the
  delta-method interval and the exact Fieller (1954) confidence set, including
  the unbounded case.
- `degradepredict.physics`: the power-law capacity-fade baseline with a
  declared, cited fade exponent and an optional profiling grid.
- `degradepredict.censoring`: Kaplan-Meier estimation and inverse probability
  of censoring weights.
- `degradepredict.features`: nineteen lag and shape features from one telemetry
  window, with declared units and no label leakage.
- `degradepredict.learned`: gradient-boosted quantile regression on log
  remaining life, producing a distribution over the crossing time, with IPCW
  weighting and a complete-case ablation.
- `degradepredict.metrics`: pinball loss, CRPS by the quantile representation,
  interval coverage reported separately for complete and right-censored units,
  the field-measurable falsification rate, PIT values, calibration curves and
  the Kolmogorov-Smirnov distance from uniformity.
- `degradepredict.benchmark`: the head-to-head harness, with a declared
  uninformative fallback for windows a baseline cannot fit and a count of how
  often it was used.
- CLI `python -m degradepredict` with `generate`, `fit` and `benchmark`.
- Seven validation scripts with their raw output, seven examples producing
  seven figures, 186 tests.

### Known results reported rather than removed

- The learned forecaster wins the CRPS comparison on complete trajectories and
  **loses to the exponential baseline on right-censored trajectories**, which
  are 57 percent of the shipped test population. Both results are in the README
  and the model card.
- No model in this repository is calibrated. At a nominal 0.90, the best
  censored coverage measured is 0.3392.
