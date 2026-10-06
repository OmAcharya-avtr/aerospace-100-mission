# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-06

First release. Validation level 3, hardware-pending. Status: TESTING.

### Added

- `photoncount.poisson` - semiclassical photon-counting statistics: mean counts
  from optical power, wavelength, slot length and detection efficiency with SI
  defining constants; additive signal, background and dark rates; threshold
  detection and missed-detection probabilities; counting SNR; the Fano factor;
  and the exact (Garwood) two-sided interval for a Poisson mean from one count.
- `photoncount.webb` - the Webb approximation for avalanche-photodiode
  counting statistics in its gain-normalised form, with its first three moments
  given in closed form (`m`, `mF`, `3mF(F-1)`) and verified against quadrature
  of the density; the exact Gaussian limit at `F = 1`; integer binning for
  comparison with a Poisson pmf; inverse-CDF sampling; and the excess-noise
  relation `F = kG + (2 - 1/G)(1 - k)`.
- `photoncount.ppm` - M-ary PPM slot statistics: slot means, sampling, the
  exact hard-decision symbol error probability by summation with uniform
  tie-breaking and a reported truncation tail mass, the all-slots-empty
  probability, the erasure-channel description of the background-free case, the
  per-slot Poisson log-likelihood derived in the module docstring, symbol
  posteriors, maximum-likelihood decisions, and bit log-likelihood ratios for a
  natural-binary mapping.
- `photoncount.deadtime` - the paralyzable and non-paralyzable relations, both
  inverses, the paralyzable maximum `(1/τ, 1/(eτ))`, both Lambert branches with
  the branch choice mandatory, observability checks, and loss and live-time
  fractions kept distinct per model.
- `photoncount.afterpulse` - afterpulsing as a Poisson cluster process in a
  first-order and a cascading variant, with closed-form cluster moments, Fano
  factors `(1+3p)/(1+p)` and `(1+p)/(1-p)`, exact rate inverses, and the
  effective probability `p exp(-τ/t_ap)` behind a dead time.
- `photoncount.capacity` - the exact background-free PPM erasure capacity, the
  exact hard-decision capacity of the induced symmetric channel, a Monte Carlo
  soft-decision achievable rate with a standard error, photons per bit, and the
  `1/log2(M)` PPM limit. No bound that could not be verified in this
  environment is cited or evaluated.
- `photoncount.simulate` - an event-level detector simulator: a Poisson primary
  process gated by a fixed or extending dead time, with afterpulses scheduled on
  registration and themselves subject to the gate. Reproduces every rate law it
  generalises and quantifies the non-commutation of the two effects.
- `photoncount.hal` - one backend contract, a fully implemented simulated
  backend, and a device backend carrying the documented contract that raises
  `NotImplementedError` naming what is missing. Simulation and dry-run modes.
  `Acquisition.is_measurement` is the single provenance predicate.
- `photoncount.ops` - preflight checks, run capture and backout as runnable
  functions: eight named preflight checks, a JSON run record carrying the
  backend description and the simulated flag, a crash-safe run journal, and an
  idempotent backout that converts a half-finished run to an aborted record
  without deleting anything. No function emits or stores an absolute path.
- `photoncount.dataset` and `photoncount.correction` - the synthetic dataset for
  the learned rate correction and the correction itself: four closed-form
  baselines implemented and measured first, then three gradient-boosted quantile
  regressors giving a point estimate and a 5-95 interval, with per-regime error
  reporting.
- CLI `python -m photoncount` with `ppm`, `deadtime`, `webb`, `capacity`,
  `acquire` and `correct` subcommands, all emitting JSON.
- 302 tests, including Hypothesis property tests and one contract suite
  parameterised over both HAL backends; nine validation scripts with their
  committed raw output; four examples, each producing a PNG in `screenshots/`;
  a benchmark harness that writes its method and environment into its own output
  file; and `docs/REQUIREMENTS.md` naming the test that exercises each of 66
  numbered requirements.

### Known limitations in this release

- **Past the paralyzable maximum nothing works.** At `n τ > 1.5` on a
  paralyzable detector every estimator in this package, closed-form and learned,
  has a median relative error between 0.83 and 0.86. The forward map is
  two-valued there. Preflight fails a run planned at or above `1/τ`.
- **The learned correction loses to the closed forms where afterpulsing is
  negligible** (`p <= 0.02`): 0.0243 median relative error against 0.0143 for
  the matched closed form, both near the 0.0097 counting-noise floor.
- **The Fano factor contributes nothing** at the counting statistics used.
  Flattening it to a constant leaves the learned model slightly better.
- Every detector parameter is an input, not a measurement. No dead time,
  afterpulse probability, dark-count rate or timestamp resolution in this
  repository came from a detector.
- The dataset holds the counting statistics roughly constant at about 5000
  counts per row, so nothing here says how any method behaves at very low or
  very high counts.
- Afterpulse release delays are a single exponential; real SPADs show
  multi-exponential or power-law tails.
- The Webb implementation is the continuous approximation; it reaches the
  Poisson pmf only at rate `m^-1/2` and has the wrong third central moment at
  `F = 1`.
- Only single-pulse M-ary PPM is implemented; multipulse PPM is not.
- `commpy` and `crcmod` do not install in the build environment, so neither was
  benchmarked against; they are named in the alternatives table on the strength
  of their own published descriptions only.
- All timing, memory and throughput figures come from a shared two-core cloud
  container. **No Level 4 claim may be built on them.**
