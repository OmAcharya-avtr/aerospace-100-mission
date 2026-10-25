# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-25

First release. Research grade: not flight-qualified, not certified, not
approved for operational aerospace use.

### Added

- `contactsched.geometry`: WGS-84 geodetic conversion, an independent IAU-1982
  GMST implementation measured against `sgp4.propagation.gstime` to 2.211e-11
  rad over 120 years, topocentric south-east-zenith look angles, two
  independent geocentric elevation derivations, and a closed-form upper bound on
  pass duration used as a falsifier.
- `contactsched.tle`: construction and independent parsing of checksummed
  two-line element sets from declared orbital elements. No catalogue element set
  is quoted anywhere in this repository.
- `contactsched.ephemeris`: a declared time base and batched propagation through
  `sgp4`, with propagation errors raised rather than returned.
- `contactsched.station`: ground stations with azimuth-dependent horizon masks
  and antenna turnaround times.
- `contactsched.visibility`: contact windows by coarse scan plus bisection on
  the mask margin, with the clipped and mask-limited boundaries flagged, and
  union supply per antenna.
- `contactsched.problem`: contact requests, discretised candidate placements,
  the conflict predicate, feasibility checking and shortfall measurement, kept
  as separate functions so that a starved schedule is feasible and short rather
  than infeasible.
- `contactsched.greedy`: baseline 1, priority-ordered first-fit, with its three
  documented failure modes exhibited on hand-built instances in the test suite.
- `contactsched.exact`: baseline 2, an exact clique mixed-integer program on
  HiGHS through `scipy.optimize.milp`, with declared fairness constraints and an
  infeasibility diagnostic that names the binding floor and its standalone
  capacity.
- `contactsched.bruteforce`: literal `2 ** n` enumeration, sharing no
  formulation with the exact method, used only to falsify it.
- `contactsched.contention`: apportioned demand over union supply per
  antenna-time bin.
- `contactsched.fairness`: per-request and per-mission satisfaction, Jain's
  index, the max-min share, and priority-inversion counting.
- `contactsched.features`, `.dataset`, `.predictors`: 23 pre-scheduling
  features, a seeded synthetic instance generator with exact-solve labels, three
  contention-ratio baselines in ranking and logistic form, and a
  gradient-boosted conflict predictor with a measured confidence output.
- `contactsched.benchmark`, `.report`, `.__main__`: comparison harnesses, text
  rendering, and a command line interface with seven subcommands.
- 611 tests, counted from junit XML and cross-checked against a literal
  `testcase` element count.
- Twelve validation scripts with committed raw output, seven examples producing
  the seven figures embedded in `README.md`, `MODEL_CARD.md`,
  `DATASET_CARD.md`, `validation/VALIDATION.md` and `docs/METHODS.md`.

### Changed from the product specification

- **The exact baseline runs on HiGHS through `scipy.optimize.milp`, not on
  PuLP.** The specification named a PuLP program. Measured in the environment
  this product was built in, `pulp` 4.1.0 returns an empty list from
  `listSolvers(onlyAvailable=True)` and `PULP_CBC_CMD` is absent from
  `listSolvers()` entirely, CBC having been unbundled in the 4.x series, so a
  PuLP-backed solve would raise rather than return a schedule. `pulp` is not a
  dependency; it is named among the alternatives in `README.md` with the
  measurement. Reproduce with `python -m contactsched backends`.
- **The conflict predictor's baseline is the per-request demand ratio, not the
  per-station-window ratio the specification named.** The per-bin ratio was
  measured first and is the weakest of the three available, at 0.6010 and 0.6228
  area under the ROC curve against 0.9230; on a smaller split it falls below
  0.5. All three are reported.
- **The cross-check against an independent geometric computation is a second
  derivation written in this package**, because `astropy` is not installed in
  this environment. No number here is a cross-check against `astropy`, `pulp` or
  OR-Tools, and every validation output says so in its header.

### Fixed during the build

- `jain_index` raised `ZeroDivisionError` for a share of 5e-324, because the sum
  of squares underflowed to zero while the sum did not. Found by Hypothesis;
  pinned as a regression test. Shares are now normalised by their maximum before
  squaring.
- `look_angles_deg` returned an azimuth of exactly 360.0, outside its documented
  half-open range, because a tiny negative angle reduced modulo 360 evaluates to
  360.0 in double precision. Found by Hypothesis; pinned.
- `HorizonMask` was not periodic in azimuth when two breakpoints shared an
  azimuth, because `numpy.interp` resolved the tie by position in its input
  array. Found by Hypothesis; duplicates are now collapsed to their maximum
  elevation, which is the conservative choice for an obstruction, and the case is
  pinned.
- `MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG` was 0.19242218 and the closed form is
  0.19242430115956036. Found by its own known-answer test, with the bound check
  that uses it failing by 1.0e-06 deg at the same time.
- `probe_working_directories.py` reported itself twice: `runpy.run_path` does not
  put a script's directory on `sys.path` the way `python script.py` does, and the
  absolute-path scan matched its own pattern literals and its own previous
  output file. Patterns are now assembled from `os.sep`, offenders are reported
  by pattern name only, and the output file is deleted immediately before the
  scan.
- `.github/workflows/tests.yml` reported itself: the CI step that forbids an
  absolute path in any tracked file contained one in its own grep pattern. The
  pattern is now assembled from a shell separator variable.

### Known limitations at this release

Recorded in full under Limitations in `README.md`. The load-bearing ones: UT1
against UTC, polar motion and the equation of the equinoxes are omitted; the
optimum is exact over a discretised candidate grid; every instance is synthetic
and the learned model's skill is skill on the generator; the confidence output
is less well calibrated than the one-feature baseline; and every dependency is
bounded on both sides, so a future minor release in any of those ranges cannot
be taken without a port and a full re-measure.
