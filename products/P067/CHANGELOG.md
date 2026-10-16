# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-16

First release. Status: TESTING.

### Added

- Pointing-uncertainty geometry: isotropic 2-D Gaussian prior with an exact
  Rayleigh radial law and exact rectangle masses, terminal beam and
  detector-field geometry, and a square-cell partition of the search region
  whose cells are inscribed in the footprint disc.
- Four scan-pattern path generators (Archimedean spiral, boustrophedon raster,
  Lissajous, seeded random restart) and the cell visit orders they induce, so
  the four patterns are compared on the same cell set.
- Photon-counting dwell detection: Poisson threshold detection with a
  Gaussian-beam shape factor and an optional lognormal scintillation average
  by Gauss-Hermite quadrature, with the false-alarm probability from the same
  model.
- Analytic expected acquisition time and deadline probability under four
  stated assumptions, a Monte Carlo twin that samples exactly those
  assumptions, and a continuous-geometry Monte Carlo that measures what the
  cell abstraction costs.
- Dwell-time trade with a grid optimum, and sensitivity of that optimum to
  the count threshold, the verification time and the Rytov variance.
- Geometric closed form for the coverage of a fully-covering Archimedean
  spiral, with an explicit validity predicate.
- Bayesian posterior update over cells and a Bayesian-greedy adaptive policy
  (the analytic baseline), a fixed-pattern policy, and a learned
  gradient-boosted scan-order policy with a (0.1, 0.9) prediction interval on
  its remaining-time estimate.
- Paired held-out benchmark over 300 declared scenarios, a shortlist
  ablation, and interval calibration.
- `python -m acqsearch` with `patterns`, `expected`, `dwell`, `coverage`,
  `benchmark` and `scenarios` subcommands.
- Three runnable examples writing PNGs into `screenshots/`, eight validation
  scripts with their raw output, and a 242-test suite counted from junit XML.

### Measured

- The Bayesian-greedy baseline and the fixed spiral pattern both beat the
  learned policy on held-out scenarios, by +0.3359 s and +0.3455 s of mean
  censored acquisition time at z = +7.53 and z = +6.51. The result is
  reported in README.md, MODEL_CARD.md and VALIDATION.md in the same terms
  and is pinned in `tests/test_benchmark.py`.
