# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-10

First release. Status: TESTING. Validation level 2.

### Added

- `physics`: the closed-form steady level-flight energy model for a small
  fixed-wing electric UAV (parasite plus induced power, propulsive efficiency
  with a quadratic fall-off away from the design airspeed, constant avionics
  draw), with units, assumptions and validity ranges on every function.
- `shift`: `CovariateShift`, a declared Gaussian mean shift in all-up mass and
  headwind component, with the **exact** closed-form likelihood ratio and a
  `scaled(fraction)` constructor for building deliberately wrong weights.
- `data`: deterministic synthetic dataset generation with multiplicative
  (heteroscedastic) observation noise, and `make_audit_split` for one audit
  replicate.
- `baseline`: `PhysicsRegressor`, the analytic point-prediction baseline, four
  coefficients by bounded nonlinear least squares on the closed-form model;
  and `GaussianResidualInterval`, the parametric interval baseline from the
  model's own residual variance. Both written and benchmarked before any
  learned component existed.
- `conformal`: `SplitConformal`, `MondrianConformal` (class-conditional) and
  `WeightedConformal` (likelihood-ratio weights, with an explicit infinite
  interval when the test weight takes more than `alpha` of the mass), the
  `weighted_quantile` primitive, tercile binning helpers, and
  `ConformalizedRegressor`, the wrapper that gives a point predictor an
  interval output.
- `bounds`: the finite-sample split-conformal coverage bound
  `ceil((n+1)(1-alpha))/(n+1)`, Clopper-Pearson binomial intervals and Kish's
  effective sample size.
- `learned`: `LearnedRegressor`, a gradient-boosted tree ensemble, and
  `LearnedWeightEstimator`, a logistic density-ratio estimate of the
  likelihood ratio from unlabelled covariates, with its calibration-versus-test
  AUC as a confidence output and a reported clipping fraction.
- `audit`: `coverage_audit` (the deliverable), `breaking_point_sweep` and
  `stratified_coverage`, all reporting replicate-level confidence intervals
  alongside the too-narrow pooled Clopper-Pearson interval.
- `plotting`: six Agg-only figure functions, each taking an already-computed
  result so a figure cannot disagree with the numbers.
- CLI `python -m conformalband` with `info`, `bound`, `baseline`, `audit`,
  `breaking-point` and `strata`, text and JSON output, and a documented
  non-zero exit on a refused request or, with `--fail-on-undercoverage`, on a
  measured under-coverage.
- Six runnable examples, each writing a PNG into `screenshots/`.
- Eight validation scripts with their raw output committed beside them.
- 440 tests: unit, input validation, known-answer with the hand arithmetic in
  the test comments, edge cases, derandomised Hypothesis property tests,
  integration tests on the audit, and CLI subprocess tests asserting exit
  status.

### Known limitations at this release

See the Limitations section of README.md. The ones that bite first: the
parametric baseline is **wider**, not tighter, than split conformal in
distribution, contradicting the product specification; the analytic physics
baseline beats the learned model at every shift severity; marginal split
conformal covers 0.9597 in the lowest tercile of predicted energy and 0.8192 in
the highest with no shift at all; and weighted conformal loses the guarantee
when the declared shift is understated by 10 to 15 per cent.
