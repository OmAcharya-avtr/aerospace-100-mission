# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-29

First release. Status: TESTING. Validation level 2, research grade.

### Added

- `refractpath.atmosphere`: `AtmosphericState`, `Observer` and
  `LayeredProfile`, with the hydrostatic layer solution of equation (1), the
  US Standard Atmosphere 1976 lapse-rate structure of equation (2) **anchored
  on the observer rather than on mean sea level**, the exponential vapour
  profile of equation (3) and the Buck (1981) saturation form of equation (4).
  The surface lapse rate and an optional shallow surface-inversion layer are
  exposed, because they are the dominant uncontrolled input near the horizon
  and a product that hides them cannot report its own validity floor honestly.
- `refractpath.refractivity`: four independently published formulations of
  `n - 1` — the Explanatory Supplement form of equation (9), Edlén (1966) of
  equation (5), Birch and Downs (1994) of equation (6), and Ciddor (1996) of
  equations (10) to (12), the last computing air densities with the BIPM-1991
  equation of state rather than an empirical scaling. Each carries its
  published wavelength range and the source of that range.
- `refractpath.models`: baseline 1, the closed forms that are actually used.
  The exact plane-parallel solution of equation (13) with its analytic floor
  `90 - arcsin(1/n)`; Oriani's two-term series of equation (14) with the
  truncation ratio and breakdown elevation of equation (14a) computed in closed
  form; Bennett (1982) of equation (15); Sæmundsson (1986) of equation (16),
  inverted numerically from its published true-elevation direction onto the
  apparent elevation; and the density scaling of equation (17).
- `refractpath.raytrace`: baseline 2, a ray trace through a layered spherically
  symmetric refractivity profile. Bouguer's invariant of equation (18), the
  refraction integral of equation (19), the telescoping shell discretisation of
  equation (20) evaluated at shell midpoints, and the cancellation-free form of
  `cos^2(theta)` in equation (21).
- `refractpath.disagreement`: equation (22), the cross-model spread, and
  `validity_floor`, which **measures** the elevation at which that spread
  crosses a declared pointing tolerance by bisection and reports the bracket,
  the crossing count and whether the curve was monotone.
- `refractpath.verdict`: a `ValidityVerdict` returned as the **first field of
  every result**, computed before the correction it gates. Three labels,
  `usable`, `degraded` and `refuse`, with the reasons named; the profile
  envelope of equation (23); both computed model floors; the per-formulation
  wavelength-range report; and five structural limits the library cannot
  determine from its inputs at any elevation.
- `refractpath.references` and `refractpath.reference_checks`: six external
  published references in three declared kinds, with their conditions,
  citations and tolerance provenance, and the checks against them as library
  code so that the shipped behaviour and the committed transcript cannot
  disagree.
- A CLI with `info`, `refract`, `curve`, `floor`, `dispersion` and
  `references`.
- Eleven validation scripts with their raw output, eight example figures, and
  392 tests including Hypothesis property tests and eleven pinned regressions.

### Measured and reported as found

- **Bennett (1982)'s own published accuracy claim of 0.07 arcminutes is not
  reproduced** against this package's ray trace. Measured maximum departure
  34.0737 arcsec at zero elevation, 63 of 181 grid points outside the claim,
  and the claim holds at and above 19.1361 degrees. The claim was made against
  Garfinkel (1967), which this package does not implement, so the miss measures
  the difference between two reference atmospheres and not an error in the
  published formula. **The tolerance was not widened.**
- **This product ships no transcribed refraction table.** None was available to
  this build and inventing plausible entries would be worse than having none.
  That is the principal weakness of the external verification and it is stated
  in the README, in `validation/VALIDATION.md` and in the module docstring of
  `refractpath.references`.
- **The two published navigational fits overstate the refraction relative to
  the layered integral at every one of 54 swept states above 15 degrees**, by
  up to 2.8670 and 4.1900 arcsec at 45 degrees. Their measured usable range at
  a 10 arcsec budget is 15 and 30 degrees against a published range of 0 to 90.
- **Below about 7 degrees the sign of those fits' error is not constant across
  the state grid.** At 5 degrees Bennett's signed error runs from -22.194 to
  +20.229 arcsec, and at the horizon it spans 489.64 arcsec across the 54
  states. A departure with no constant sign cannot be removed by a scale
  factor.
- **Equation (22) is not monotone.** At a 10 arcsec tolerance the scan finds
  three crossings, and a search that assumed one and walked up from the horizon
  would understate the floor of 11.2448 degrees by a factor of 9.179. This
  package always takes the highest crossing.
- **Below one degree the profile nobody measured is the larger term**, by
  factors of 6.9 to 8.8. Three physically attainable surface lapse rates move
  the horizon refraction by 300.0181 arcsec and move the 45-degree refraction
  by 0.0014 arcsec. A 25 m surface inversion of +20 K/km moves the horizon
  refraction by +137.001 arcsec.
- **Equations (15) and (16) predict exactly zero atmospheric dispersion**,
  because the published formulae take no wavelength. The ray trace puts the
  400-to-700 nm difference at 1.4571 arcsec at 45 degrees, exceeding one
  arcsecond below 55.426 degrees.
- **The four refractivity formulations agree on dry air and disagree on
  humidity**, by 11.77 times the dry-air spread at 40 C and saturation.
- **The Oriani series and the ray trace agree to 0.0007 arcsec at 45 degrees**
  over all 54 states, which is the strongest internal consistency result here.
- **The independently derived bracket rejects the exact plane-parallel answer**
  at every elevation tested, which is how the bracket was shown to
  discriminate rather than merely be wide.
- **The constant-of-refraction check rests on a wavelength this product
  chose.** The published figure is given for the visible band without one, and
  the measured coefficient spans 0.83924 arcsec across the visible, 3.36 times
  the check's tolerance.
- **Nine controls must fire and nine fired; five must not fire and none did.**

### Defects found during the build and pinned as regressions

- A trapezoid rule over shell **boundaries** evaluates `tan(theta)` at the
  observer, and for a ray launched exactly horizontally the observer is the
  tangent point, so the horizon refraction came out as 8.0e138 arcmin.
  Equation (20) evaluates at shell **midpoints**, which never touches it.
- Computing `1 - sin^2(theta)` directly loses all precision when `sin(theta)`
  is within 1e-14 of unity, which is the first few metres of a horizon ray.
  Equation (21) factorises it.
- Without the equation (14a) guard, the Oriani series returns about -1.26e7
  arcsec at 0.1 degrees and about -2.9e47 arcsec at exactly 0 degrees in
  floating point, which made equation (22) a measurement of a floating-point
  artefact rather than of a disagreement.
- The validity floor took the lowest tolerance crossing, understating it by a
  factor of 9.179 at a 10 arcsec tolerance.
- The test suite first asserted the wrong sign for the observer-height
  dependence. At fixed station pressure a larger radius means a smaller
  curvature correction, so the refraction **rises** toward the plane-parallel
  limit.
- Two validation checks were first written as assertions that the measurement
  then falsified — "Bennett departs by more than 1 arcsec at 45 degrees over
  every state" and "the profile envelope exceeds the cross-model spread at or
  below 2 degrees". Both were replaced by the statement the measurement
  supports, not by a widened threshold.

### Hypothesis counterexamples, pinned with their literal inputs

- A 1e-3 bound on `R(2p)/R(p) - 2` falsified at 807 hPa and -2 C, then a
  widened 2e-3 bound falsified at 1002 hPa and -41 C. Both are the known 1e-8 p
  term of equation (7), so the property was replaced by the **exact algebraic
  ratio** rather than by a third tolerance.
- A strict `raytrace < plane_parallel` falsified at exactly 90 degrees, where
  both are identically zero.
- "The Oriani breakdown is always above the plane-parallel floor" falsified at
  1079 hPa, -59 C and 1.0 um. The ordering reverses exactly where
  `beta = 2.5 kappa`, which is now the asserted property.

### Known limitations

See `README.md`. The dominant ones: one external check fails and is published
as failed; there is no transcribed refraction table and no comparison against a
measurement; `astropy`, `ephem` and `pyerfa` are not installed in the measured
environment and nothing here is compared against them; everything above the
observer is invented from one surface measurement; the ray trace assumes
spherical symmetry; and the declared dependency ranges were not run at both
ends because the build environment is a shared pre-pinned interpreter this
product is forbidden to modify.

[0.1.0]: https://github.com/OmAcharya-avtr/refractpath/releases/tag/v0.1.0
