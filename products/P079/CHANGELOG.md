# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-24

First release. Validation label: **Level 3, hardware-pending**.

### Added

- **Baseline 1, the symmetric-delay two-way offset estimator**
  (`timetransfer.twoway`), with the asymmetry-induced bias `A/2` derived in
  closed form and verified to a worst departure of 1.084e-19 s against a
  declared asymmetry sweep, and a fitted slope of 0.500000000000.
- **Baseline 2, least-squares and Kalman estimation of offset, skew and drift**
  (`timetransfer.skew`), with the parameter covariance, the two-state clock
  model of Zucca and Tavella (2005), a measurement variance taken from the data
  and a consistency diagnostic reported rather than tuned against.
- **Overlapping Allan deviation over tau with a confidence interval at each
  tau** (`timetransfer.allan`), implemented here because `allantools` is not
  installed in the environment this product was measured in, and verified
  against three closed-form noise processes (`timetransfer.noise`): fitted
  slopes -0.9998, -0.4834 and +0.5194 against theory -1, -1/2 and +1/2.
- **A measured uncertainty on the offset estimate** (`timetransfer.uncertainty`)
  with three components -- statistical with a serial-correlation inflation
  measured from the residuals, quantisation at `delta/sqrt(12)`, and asymmetry
  -- and `measure_coverage` for measuring the result against nominal.
- **The learned delay-asymmetry corrector and the three analytic estimators it
  is benchmarked against** (`timetransfer.asym`), with the analytic ones
  implemented and scored first.
- **Hardware abstraction layer** (`timetransfer.hal`, `timetransfer.backends`)
  with a simulated backend, a device backend over a five-method driver shim,
  and loopback, recorded, absent and guarded driver stubs.
- **Seeded simulation mode** (`timetransfer.clockmodel`,
  `timetransfer.linkmodel`) with declared drift, jitter, asymmetry and outages.
- **Dry-run mode** (`timetransfer.dryrun`) replaying a committed recording with
  counted driver operations and a negative control that must fire.
- **Deployment and recovery** (`timetransfer.deploy`, `docs/DEPLOYMENT.md`):
  ten executable pre-flight checks, a run guard that restores and verifies, and
  a truncated-estimate path for losing the exchange mid-window.
- **Benchmark harness** (`timetransfer.bench`) with a measured host table and a
  Level 4 hardware table whose five rows are empty and marked `unmeasured`,
  guarded by three refusals that raise `HardwareClaimError`.
- **Command-line interface** (`python -m timetransfer`): `version`, `info`,
  `simulate`, `allan`, `record`, `dryrun`, `preflight`, `parity`, `bench`.
- 335 tests, including Hypothesis property tests over the estimator identities
  and named regression tests with literal inputs for every defect found.
- Fourteen validation scripts with their committed raw output, seven examples
  producing seven screenshots, `docs/REQUIREMENTS.md`, `MODEL_CARD.md` and
  `DATASET_CARD.md`.

### Measured, and reported as measured

- An offset interval with no asymmetry term covers **0.1200** against a nominal
  0.9500. With a declared bound it covers 0.9900; with either corrector applied
  it covers 0.9450 and 0.9525.
- **The gradient-boosted asymmetry corrector loses to a standardised linear
  ridge on the same sixteen features at every coupling measured** (RMSE ratios
  1.0273, 1.0099, 1.0088). The boosted model's own uncertainty band under-covers
  at 0.8917 against 0.9500.
- The median-filtered estimator matches the learned model where nothing is
  learnable (ratio 1.0394 at coupling 0) and loses where something is (1.4401
  and 1.8868). The product specification predicted a match; the measurement
  confirms it in one regime and not the other, and both are published.
- The Kalman filter is over-confident when its measurement variance comes from
  the data: mean normalised innovation 2.53 to 2.65 with jitter, 8.788e+04
  without.

### Fixed during the build

- `BackendInfo.is_hardware` did not strip a `guarded-` prefix, so a recorded
  driver wrapped in `GuardedDriver` reported itself as hardware and dry-run
  mode claimed 240 hardware operations. Pinned as
  `tests/test_regressions.py::test_regression_guarded_stub_is_not_hardware`.
- `fit_least_squares` over identical epochs returned a skew of exactly zero
  with an uncertainty of exactly zero, via the pseudo-inverse of a rank-deficient
  design matrix. Found by a Hypothesis probe; now refused. Pinned as
  `tests/test_regressions.py::test_regression_identical_epochs_are_refused`.
- `measurement_variance_from_data` had a fixed, physically meaningless floor.
  The floor is now a parameter, its default is documented as not meaningful,
  and the measured consequence of leaving it alone on a quiet link is published.

### Not in this release

- Any number measured on hardware. The Level 4 table is empty and this product
  is labelled Level 3, hardware-pending. See `docs/DEPLOYMENT.md` section 0 for
  the five figures that are missing and what unblocks them.
