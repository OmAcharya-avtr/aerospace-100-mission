# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-21

First release. Status: TESTING. Validation level 3, research grade.

### Added

- `conjprob.geometry`: time of closest approach, miss vector, deterministic
  encounter-plane basis, marginal projection of a 3x3 position covariance onto
  the plane, principal frame, Mahalanobis distance.
- `conjprob.analytic`: three independently implemented analytic collision
  probabilities — Foster-Estes by adaptive quadrature of the error-function
  reduction, Chan's equivalent-area series, Alfano's fixed-node Simpson
  evaluation — plus the exact Rayleigh closed form at zero miss distance and
  the leading-order small-hard-body approximation.
- `conjprob.quadrature`: an independent 2-D polar quadrature reference that uses
  neither the error-function reduction nor the eigendecomposition.
- `conjprob.montecarlo`: 3-D Monte Carlo over the relative state with
  Clopper-Pearson intervals, in three variants — unbounded time, finite window,
  and finite window with a declared constant relative acceleration.
- `conjprob.dilution`: the dilution diagnosis, the closed-form maximum Pc over
  covariance scale, the Pc-against-scale curve, and `DilutionWarning`, raised by
  the library itself.
- `conjprob.nonlinear`: measured breakdown of the rectilinear assumption against
  the finite-window and curvature parameters, with the gravity-gradient
  acceleration scale for reference.
- `conjprob.regimes`: the dimensionless parameterisation and measured validity
  boundary curves for each analytic method against the quadrature reference.
- `conjprob.report`: `assess_conjunction`, which returns every method, their
  disagreement, the dilution diagnosis and the regime flags, and deliberately
  has no scalar `pc` attribute.
- `conjprob.features`, `conjprob.screens`, `conjprob.screening`,
  `conjprob.dataset`: cheap screening features, three analytic screens
  implemented before any learned model, a random-forest screen with a per-tree
  dispersion output, and a committed deterministic synthetic population in three
  declared regimes.
- `conjprob.frames`, `conjprob.cases`, `conjprob.tle`: radial/along-track/
  cross-track covariance declaration, YAML case loading, and an optional TLE
  state entry point over `sgp4`.
- Command-line interface with `assess`, `dilution`, `regimes` and `screen`.
- Ten validation scripts with their raw output, seven examples producing the
  committed screenshots, 269 tests.

### Fixed during the build, found by property-based testing

These were defects in this package's own first implementation. Each is pinned as
a named regression test in `tests/test_regressions.py` with its literal inputs.

- An absolute quadrature tolerance of `1e-14` destroyed scale invariance for any
  Pc near or below it: a geometry and the same geometry with every length
  multiplied by 19 disagreed by 5.3 per cent, and both values were wrong. The
  tolerance is now purely relative.
- The published error-function form of the chord mass,
  `0.5 [erf((h - y)/s) + erf((h + y)/s)]`, cancels to exactly zero when the
  whole chord is many standard deviations from the mean. At `sigma_y = 3 m`,
  `ybar = 68 m`, `h = 1 m` the correct mass is 8.767418031863e-111 and the
  published form returned 0.0. Both Foster-Estes and Alfano now use a complementary-error-
  function branch.
- Chan's published finite exponential sum loses up to twelve significant figures
  to cancellation. At `sigma = 1 m`, `ybar = 8 m`, `R = 1 m` it was 5.2e-4 wrong
  against the non-central chi-square distribution function. The bracket is now
  evaluated as a regularised incomplete gamma function.
- Chan's series needs an order that grows as `M^2 / 2`; the order of 3 the
  original papers recommend is 2.0e-6 wrong at `M = 9` and 15 per cent wrong at
  `M = 12` with `R / sigma = 0.5`. `chan_recommended_order` was added and is the
  default in `assess_conjunction`.
- Chan's series as written accumulates `(v/2)^m / m!` and multiplies by
  `exp(-v/2)` at the end. At `sigma = 437 m`, `xbar = 16515 m`, `R = 1 m` the
  accumulation overflows to infinity while the prefactor underflows to zero and
  the product is `nan`. The term weights are now evaluated in logarithms as a
  Poisson distribution, and the case returns 1.92739845e-316 in agreement with
  the Foster-Estes path to 2.563381e-08 relative.
- A subnormal miss component left the unsquared Mahalanobis hypotenuse positive
  while its square underflowed to zero, turning a guarded condition in
  `max_probability` into a division by zero.

### Known failing check, published rather than repaired

- The learned screen's predicted probabilities are miscalibrated: the worst
  reliability-bin gap is 0.272486 against a declared tolerance of 0.15
  (`validation/validate_screening_output.txt`, check 4). It is not repaired
  because recall at a fixed false-positive rate depends only on the ordering of
  the scores and is invariant under any strictly monotone recalibration, so a
  repair would move the Brier score and change nothing a user of the screen
  experiences. See `MODEL_CARD.md`.
