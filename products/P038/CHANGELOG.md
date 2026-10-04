# Changelog

All notable changes to dopplerkit are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[semantic](https://semver.org/).

## [0.1.0] — 2026-10-04

First release. Validation level 1 (closed-form known answers and internal
consistency). Status: TESTING.

### Added

- **One sign convention, stated at every interface.** Range-rate positive when
  receding; Doppler positive (up-shift) when approaching;
  `Delta_f = -f_c * rho_dot / c`; pre-compensation offset is the negative of the
  Doppler shift. Carried in every public docstring, on `DopplerObservable` and
  `DopplerProfile` result objects, in the README API table, and in a dedicated
  `python -m dopplerkit convention` subcommand.
- `dopplerkit.geometry` — `State`, `slant_range_m`, `range_rate_mps` (invariant
  under reversing the link direction), `range_acceleration_mps2`,
  `range_rate_finite_difference_mps`.
- `dopplerkit.doppler` — `one_way_doppler_hz`, `two_way_doppler_hz` (exactly
  twice one-way at unit turnaround ratio), `two_way_doppler_two_leg_hz`
  (light-time-separated legs, returns the dropped O(beta^2) cross term),
  `doppler_rate_hz_per_s`, `precompensation_offset_hz` (first-order and exact),
  `relativistic_fraction_second_order`, `relativistic_correction_hz`,
  `gravitational_shift_fraction`, `LinkDirection`, `DopplerObservable`.
- `dopplerkit.lighttime` — `down_leg_light_time`, `up_leg_light_time`,
  `two_way_light_time`, each reporting iteration count and final residual rather
  than only a converged flag.
- `dopplerkit.passprofile` — `compute_profile`, `DopplerProfile`,
  `time_of_closest_approach_s`, `doppler_zero_crossing_s` (independent methods,
  so their agreement is evidence).
- `dopplerkit.analytic` — `CircularOverheadPass`, the closed-form reference used
  for every known-answer check.
- `dopplerkit.frames` — ground-station state in TEME with its `omega_E x r`
  rotation velocity and centripetal acceleration, geodetic site position, IAU
  1982 GMST, `sgp4` state conversion, topocentric elevation.
- `dopplerkit.constants` — SI and WGS-84 constants with their sources.
- CLI `python -m dopplerkit` with `convention`, `analytic`, `tle` and
  `relativistic` subcommands; JSON output on the first two; exit code 2 with the
  offending option named on invalid input.
- 146 tests, including 23 dedicated sign assertions and 12 Hypothesis
  properties.
- Five validation scripts with committed raw output, and four examples that
  produce the README screenshots.

### Known limitations at this release

- Order retained is O(beta^1). The O(beta^2) terms are reported, never applied.
- Shapiro delay, troposphere, ionosphere, transponder group delay and oscillator
  drift are not modelled at all.
- Doppler rate from the `sgp4` path carries only the kinematic term, because
  `satellite_state_teme` leaves the acceleration at zero.
- The frame reduction is scheduling-class (GMST only, UT1 taken as UTC).
- The TLE-path numbers are self-consistency regression values, not a comparison
  against an independent ephemeris.

[0.1.0]: https://github.com/OmAcharya-avtr/dopplerkit/releases/tag/v0.1.0
