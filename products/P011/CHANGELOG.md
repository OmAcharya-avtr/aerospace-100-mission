# Changelog

All notable changes to WaveForge are recorded here.

## 0.1.0 — 2026-08-29

Initial release.

### Added

- `pupil`: circular sampling grid with optional central obscuration, piston
  removal, spatial variance and RMS, and a numerical Strehl ratio from the
  complex pupil field.
- `zernike`: Noll (1976) indexing as an exact integer map in both directions,
  radial polynomials from the Born & Wolf factorial sum, orthonormal modes in
  polar and Cartesian coordinates, closed-form Cartesian gradients with the
  `1/rho` singularity removed at the coefficient level, and least-squares modal
  fitting with an explicit mask.
- `statistics`: Kolmogorov PSD in cyclic frequency, structure function with the
  constant at full precision (6.883877, not the rounded 6.88), Fried parameter
  from a `Cn^2` path integral, Greenwood frequency and coherence time, analytic
  Kolmogorov Zernike variances derived from Noll's Eqs. 8 and 18 with log-gamma
  arithmetic so large orders do not overflow, residual variances `Delta_J` by
  direct summation, Noll's large-`J` asymptote, and Noll's published table as
  reference data only.
- `atmosphere`: Fourier phase screens with optional Lane et al. (1992)
  subharmonic augmentation and a von Karman outer scale, an exact band-limited
  structure function for the sampling grid so that implementation error and
  method bias can be separated, an empirical structure-function estimator, and
  a frozen-flow atmosphere that refuses to wrap its non-periodic subharmonic
  component.
- `sensor`: Shack-Hartmann slope operator built by explicit central differences
  per subaperture, fill-fraction masking, Rousset (1999) photon and read-noise
  variances converted to rad/m, noise-equivalent angle, and per-frame
  subaperture dropout with validity flags.
- `dm`: Gaussian influence functions parameterised by nearest-neighbour
  coupling, optional margin actuator rings, symmetric stroke limiting with a
  saturation fraction, and regularised least-squares fitting with a cached
  Cholesky factorisation.
- `control`: leaky integrator with an explicit latency buffer, analytic error
  rejection and noise transfer functions, closed-loop stability limit by
  bisection on the pole moduli, and noise variance amplification from the
  impulse response.
- `errorbudget`: fitting error with the ideal-filter coefficient derived rather
  than quoted, pure-delay and servo-bandwidth temporal terms kept separate,
  exact noise propagation through the actual reconstructor, both Marechal
  Strehl forms and their inverse, and an additive budget container.
- `loop`: `AOConfig`, `AOSystem` and `LoopResult` — the end-to-end simulation,
  with a truncated-SVD reconstructor, pseudo-open-loop slope reconstruction,
  optional predictive control, divergence detection, and per-run gain and
  latency overrides.
- `predictor`: a pure-delay baseline and a bagged ridge (or MLP) forecaster of
  pseudo-open-loop slopes with a per-slope predictive standard deviation from
  ensemble spread plus out-of-bag residual variance.
- `datasets`: deterministic, seed-partitioned generation of training and test
  slope sequences with a provenance summary; nothing is written to disk.
- CLI `python -m waveforge` with `noll`, `screen`, `budget`, `loop` and
  `predict` subcommands.
- `docs/REQUIREMENTS.md`: 18 numbered requirements, a verification matrix
  naming the test or validation script for each, and seven explicitly
  unimplemented items.
- Six validation scripts with their raw transcripts, five example scripts and
  the five figures they produce, 635 passing tests including Hypothesis
  property tests, a pinned-seed regression suite, performance budgets and
  failure-mode tests, and a ruff-clean source tree.

### Validation highlights

- Zernike orthonormality under exact quadrature: worst deviation 6.2e-15.
- Noll residual variances against Table IV, `J = 1..21`: worst 0.53 %.
- Two independent derivations of the total piston-removed variance agree to
  0.033 % and both sit 0.25 % above Noll's published third significant figure;
  reported, not tuned away.
- Deformable-mirror fitting coefficient at 33 x 33 actuators, corrected for the
  screens' band limit: 0.273 against the published 0.28.
- Scalar rejection transfer function against the analytic expression: 1.7e-14.
- Stability limits 2, 1 and 0.618034 reproduced to 3.5e-10.
- Screens against the exact discrete-spectrum expectation: 0.50 %.
- Learned predictive controller against a gain-tuned classical integrator on
  held-out screens: 1.25x lower residual variance at one frame of latency and
  3.45x at four frames. It loses by 17 % at twice the training wind speed and
  by 5.9x if deployed on a noisier sensor than it was trained on; both failures
  are reported in `MODEL_CARD.md` and `validation/VALIDATION.md`.

### Known deviations

- Per-mode Kolmogorov variances recovered by differencing Noll's rounded table
  entries are consistent with this package for 13 of 20 modes; the other seven
  differ in one direction by the 0.25 % normalisation offset above.
- The extended Marechal approximation underestimates the Strehl of genuine
  closed-loop residuals by up to 26.7 %.
