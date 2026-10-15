# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-15

First release. Status: TESTING. Validation level 3, research grade.

### Added

- **Identifiability analysis, run before any fitting.** Weighted
  log-parameter sensitivity matrix, Fisher information, numerical rank,
  condition number, Brun et al. (2001) collinearity index and per-parameter
  sensitivity measure, and the unconstrained parameter *combinations* written
  out as combinations rather than reported as a warning.
- **`null_direction_loading`**, which puts the infinities back into a
  rank-deficient covariance so that a pseudo-inverse cannot report a finite
  standard deviation for a parameter carrying no information.
- **Weighted nonlinear least squares in log-parameter coordinates**, which for
  this noise model is maximum likelihood, with the Gauss-Newton covariance,
  correlation, reduced chi-squared and a `caveats()` list that travels with
  every reported value. A divergence guard scores parameter vectors the
  fixed-step integrator rejects and counts them.
- **Profile-likelihood intervals** by grid walk plus bisection refinement, with
  open sides reported as open and the searched range reported with them.
- **Two twin models.** A single-node linear model with closed-form solution
  *and* closed-form sensitivities, which exists so that the differentiation and
  the identifiability analysis have exactly known answers to be checked
  against; and the two-node radiative network the package is built around.
- **Five experiment designs**, three deliberately defective: a structurally
  non-identifiable one whose null direction is known in closed form, a
  practically non-identifiable one that passes a rank test, and a
  strongly-coupled single-channel one.
- **A learned residual (discrepancy) model**, implemented last: Gaussian-process
  regression on features computed from the physics run alone, returning a mean
  and a predictive standard deviation, benchmarked against physics-only on
  held-out trajectories.
- **Two absorption measurements**: a subspace decomposition of the learned
  residual field into directions the parameters could have produced and the data
  can or cannot constrain, and a parameter-drift report that refits the physics
  with the discrepancy present.
- **Command-line interface** `python -m twincalib` with `designs`, `identify`,
  `fit`, `profile` and `residual`, whose exit status distinguishes "ran and
  nothing came out negative" from "ran and something did".
- Nine validation scripts with committed raw output, six plotting examples with
  committed PNGs, and 219 tests.

### Measured, and reported because it is unflattering

- The learned residual model improves held-out prediction by 39 per cent on the
  sound design and **makes the physical parameters 35 per cent worse** in
  root-mean-square log error, on the same experiment. It makes them worse on
  three of the four designs measured.
- Its predictive interval **over-covers** on every design: 0.9709 to 0.9933
  against a nominal 0.9500. It is conservative and therefore not calibrated.
- Profile-interval coverage falls from **0.9306** to **0.3750** once the declared
  model-form discrepancy is present, which is what a confidence interval means
  when the model is wrong.
- Refining the integration step moves the trajectory by **0.626 K** because the
  eclipse edge is not a multiple of the step. The integrator's own error at the
  same comparison is 4.76e-06 K. The input grid is part of the model.
- The absorption *ratio* between designs is not a stable number: it swings from
  3.34 to 96.74 across six configurations. The *separation* is stable and is
  what the README cites.

### Known limitations

See README "Limitations" for all fifteen, including the one-minor-version
dependency pins and why they cannot be widened without re-measuring.
