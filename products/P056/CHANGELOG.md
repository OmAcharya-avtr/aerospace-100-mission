# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-10

First release. Status: TESTING. Validation level 2.

### Added

- `scores`: `brier_score` (Brier 1950), `log_score` (Good 1952), both skill
  scores against an explicit reference forecast array, the constant and
  base-rate reference forecasts, and `check_forecasts`, the single validator
  every entry point uses.
- `binning`: `equal_width` and `equal_mass` bin edges and assignment, with the
  tie and empty-bin behaviour of each documented rather than discovered.
- `decomposition`: `murphy_decomposition`, the exact three-term partition over
  distinct forecast values, and `binned_decomposition`, which carries the
  exact five-term identity `BS = REL - RES + UNC + WBV - 2 WBC` and reports
  `three_term_residual` — the quantity a three-term binned report drops.
- `ece`: the plug-in expected and maximum calibration errors;
  `null_ece_distribution`, the estimator's own noise under the calibrated
  null; `debiased_ece`, which subtracts that bias and returns a
  parametric-bootstrap p-value alongside; and `ece_bias_curve`, which measures
  the estimator's bias against a population value known in closed form and
  fits its `sqrt(B/n)` scaling law.
- `reliability`: reliability curves, pointwise percentile bootstrap bands with
  bin edges fixed from the original sample, and `band_coverage`, which
  measures what the bands actually cover against the exact calibration curve
  of a synthetic forecaster, pointwise and simultaneously.
- `recalibration`: `RawForecast` (the baseline, implemented first),
  `PlattScaling` (maximum-likelihood two-parameter logistic map) and
  `IsotonicCalibration` (a thin wrapper over
  `sklearn.isotonic.IsotonicRegression`), each with a bootstrap-ensemble
  uncertainty output through `predict_with_interval`;
  `recalibration_audit`, a held-out paired comparison with an explicit
  verdict; and `sample_size_sweep`, which reports the harm rate against
  sample size.
- `synthetic`: six forecasters whose population Brier score, Murphy terms,
  ECE and logarithmic score are known in closed form or by quadrature with its
  own error estimate, three of them exactly calibrated.
- `plotting`: four Agg-only figure builders, one per example.
- CLI `python -m calibaudit` with `specs`, `decompose`, `ece`, `ece-bias`,
  `reliability` and `recalibrate`, text and JSON output, and a non-zero exit
  from `recalibrate --require-improvement` when no learned recalibrator beats
  the raw baseline.
- Five runnable examples, each writing one PNG into `screenshots/`.
- Ten validation scripts with their raw stdout committed under
  `validation/outputs/`, including `validate_cli.py`, which re-runs every
  command quoted in the documentation and fails if a documented command is
  neither executed nor excused.
- 405 tests: unit, input validation, known-answer with the hand arithmetic in
  the test bodies, edge cases, Hypothesis property tests for the decomposition
  identities, scikit-learn agreement tests, plotting smoke tests and CLI
  subprocess tests asserting on `returncode`.

### Findings published at this release

- The ECE of a **perfectly calibrated** forecaster is measured as 0.054061 at
  5 bins and 200 samples, rising to 0.217391 at 100 bins, where the population
  value is exactly 0. The bias follows `sqrt(B/n)` with a fitted exponent of
  0.496599 against the predicted 0.5.
- The parametric-bootstrap debiasing improves the error on all 9 calibrated
  configurations tested and **worsens it on all 9 miscalibrated ones**, by up
  to 57.9x. `p_value` must be read before the corrected number is quoted.
- Bootstrap band coverage is **below** the nominal 0.90 in all 12
  configurations tested, mean 0.8381, and simultaneous coverage is 0.0000 to
  0.3750. The worst single bin reached 0.3000.
- **Honest negative:** on a forecast that is already calibrated, neither Platt
  scaling nor isotonic regression improves the held-out Brier score at any
  sample size from 60 to 15 000, and isotonic is worse in 100 % of 80
  replicates at 15 000.
- **Honest negative:** on an overconfident forecast, Platt scaling harms the
  held-out score in 51.2 % of replicates at 60 total samples and isotonic in
  71.3 %. The crossing points are 200 and 1000 total samples respectively.

### Known limitations at this release

See the Limitations section of README.md. The ones that bite first: binary
forecasts only; the ECE debiasing is only trustworthy near the calibrated
null; the bootstrap bands undercover and their edge bins badly; and forecasts
of exactly 0 or 1 destroy the Platt fit and dominate the logarithmic score.
