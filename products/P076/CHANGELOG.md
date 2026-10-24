# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 — 2026-10-24

First release. Status TESTING: the build is complete and the quality review and
batch review have not happened yet.

### Added

- `pointahead.angle`: the point-ahead angle two ways. `closed_form_point_ahead`
  implements the first-order closed form `theta = 2 |v_rel_perp| / c`;
  `reference_point_ahead` solves both light times by iteration against a
  declared trajectory model and applies the exact relativistic aberration with
  the local terminal's velocity. `score_closed_form` scores the first against
  the second on three error metrics, and `decompose_point_ahead` attributes the
  magnitude to the light-time lead and the two aberration terms.
- `pointahead.budget`: an attributed error budget of six individually
  checkable lines — relative velocity knowledge, two ephemeris staleness lines,
  the light-time iteration residual, the aberration model error and the
  closed-form model error — with the beam-divergence context that says whether
  the total matters. `verify_sensitivities` checks every analytic line against
  a numerical propagation of its own input.
- `pointahead.aberration`: four implementations of the aberration
  transformation, one approximate and three exact, derived from the
  trigonometric relation, a four-momentum boost and a coordinate transformation
  of two events on the photon worldline.
- `pointahead.lighttime`: the iterated light-time solution with its full
  iterate history and residual, plus the exact quadratic root for a
  constant-velocity remote trajectory, which is what the iteration is checked
  against.
- `pointahead.kepler`: universal-variable two-body propagation with Stumpff
  functions, series-expanded near zero, and exact reduction modulo the orbital
  period for elliptical orbits.
- `pointahead.divergence`: Gaussian far-field divergence, pointing loss in
  decibels and its inverse, the affordable error for a declared loss
  allocation.
- `pointahead.scenario`: three named crosslink geometries and seeded families
  of unoccluded random ones.
- `pointahead.tle`: `sgp4`-backed terminal states from two-line element sets,
  with the TEME frame label carried through.
- `pointahead.cases`: a YAML case schema carrying the geometry, the beam and
  the declared budget configuration together, with unknown keys refused rather
  than ignored.
- `pointahead.report` and a `python -m pointahead` CLI with `info`, `budget`,
  `compare`, `lighttime` and `scenario` subcommands.
- 272 tests, counted from junit XML. Hypothesis property tests over the
  algebraic identities, and eleven pinned regressions with literal inputs for
  every defect this build found.
- Twelve validation scripts in `validation/`, each writing its raw output
  beside itself. 105 checks pass and 1 fails; the failure is diagnosed in
  `validation/VALIDATION.md` and left in place.
- Six examples producing the PNGs embedded in the README.

### Known at release

- `astropy` is not installed in the environment this build was measured in and
  installing it was not permitted, so the aberration cross-check the batch
  specification asks for was replaced by three internal exact implementations
  derived three different ways. No number in this repository came from
  `astropy`. Recorded as deviation 1 in `validation/VALIDATION.md`.
- Only one point in each declared dependency range was executed, for the same
  reason. Recorded as deviation 2.
- The `light_time_iteration_residual` budget line is unverifiable at iteration
  cap zero, where the predicted error falls below the float64 direction
  resolution. 36 of 1584 checked combinations fail there, and the check is
  reported as failed.
