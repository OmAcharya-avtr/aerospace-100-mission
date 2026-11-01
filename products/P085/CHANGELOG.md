# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-11-01

First release. Status: TESTING. Validation level 2, research grade.

### Added

- `pointmodel.terms`: the nine-term classical alt-azimuth pointing model —
  azimuth and elevation encoder index, collimation, two azimuth-axis tilt
  components, axis non-perpendicularity, two gravitational flexure terms and
  an atmospheric refraction term — with every column's functional form
  **derived in the module** rather than copied, because a sign taken on trust
  is a defect waiting to happen.
- `pointmodel.geometry`: an exact composition of rotation matrices for the same
  mount, sharing no algebra with the design matrix, and
  `linearisation_bound()`, the operator-norm Taylor remainder of a product of
  matrix exponentials, which bounds the difference between them without a
  fitted constant.
- `pointmodel.refraction`: Bennett (1982) as published by Meeus (1998) chapter
  16 equation (16.4), and an independent numerical integration of the
  refraction integral through a US Standard Atmosphere 1976 profile, written
  here. Two declared refractivity constants, radio dry-air and optical at
  550 nm, with the choice an argument and never a hidden default.
- `pointmodel.wls`: weighted least squares through the SVD of the whitened
  design matrix, with a full covariance, Student-t or normal interval
  multipliers under a declared error model, and per-term variance inflation
  computed from the full singular-value spectrum. `normal_equation_fit()` is
  shipped **only** to demonstrate the failure of the textbook form and is
  marked as such.
- `pointmodel.identifiability`: the rank analysis. Near-null directions named
  in terms of the physical coefficients, a complete orthogonal
  reparameterisation into identifiable linear combinations with exact standard
  errors, two declared thresholds, and a refusal to report a per-term split
  along any direction below the null threshold.
- `pointmodel.robust`: Huber M-estimation by IRLS with the asymptotic sandwich
  covariance, and a RANSAC-style consensus fit sampling whole observations
  rather than rows. Neither is preferred over least squares; what is reported
  is their disagreement, measured in the prediction space that is identifiable
  whatever the geometry.
- `pointmodel.synthetic`: five declared observing geometries and a residual-set
  generator whose truth comes from the exact rotation model, so a recovery test
  is not fitted with the algebra that produced the data.
- `pointmodel.budget`: `decompose()`, which returns the identifiability verdict
  first and the coefficients second.
- A CLI with `terms`, `identify`, `compare`, `decompose`, `outliers` and
  `refraction`.
- Ten validation scripts with their raw output, ten example figures, and 241
  tests including Hypothesis property tests and pinned regressions.

### Measured and reported as found

- **A single-azimuth sweep produces a least-squares split that is repeatable
  and wrong.** Over 200 noise realisations with the geometry fixed, the
  azimuth-axis tilt component `AW` has an empirical scatter of 2.321 arcsec, a
  reported standard error of 2.534, and a bias of **+16.125 arcsec**. A
  repeatable wrong number is worse than a noisy one, because it reads as a
  measurement.
- **The textbook normal-equation solver reports formal standard errors
  294,000 times larger than the actual scatter** on the same geometry — 1.201e+06
  arcsec against a measured 4.084 — because `A^T W A` squares a condition
  number already at 2.1e15.
- **A 43–47 degree elevation band makes seven of the nine terms
  unidentifiable**, with a condition number of 6.49e5 and three named null
  directions, one of which is `+0.707 CA − 0.500 NPAE − 0.500 IA` to three
  decimal places. The post-fit residual on that run is 1.91 arcsec against a
  declared sigma of 2.0 — the fit looks good by every ordinary measure and the
  elevation index is wrong by 55,786 arcsec.
- **`two_az` has two azimuths and a condition number of 1.29e2, within 0.1 per
  cent of the full sky's 1.29e2.** A geometry cannot be judged by how few
  pointings it has.
- **Even on a well-scheduled run, `IE`, `TF`, `TS` and `RF` share a weakly
  determined direction** at a singular-value ratio of 7.760e-03. That is a
  property of the term set, not of a bad schedule, and it is why refraction
  coefficients fitted from pointing data disagree with refraction coefficients
  computed from the atmosphere.
- **The refraction cross-check does not fully pass with the radio dry-air
  constant.** The maximum absolute disagreement with the independent ray trace
  is 4.1270 arcsec with the optical constant, inside the 4.2 arcsec Meeus
  quotes for equation (16.4) with 1.7 per cent of margin; and 8.4736 arcsec
  with the radio dry-air constant, which exceeds it by a factor of 2.02. The
  excess is the refractivity constant: the two differ by 1.7508 per cent and
  the measured mean ratios differ by 1.7543 per cent.
- **The normalised refraction shape, which is what the design matrix uses,
  agrees to 1.96 per cent** over 10 to 85 degrees, worst at the low end.
- **RANSAC collapses between 10 and 20 per cent contamination** — worst error
  5.45 arcsec at 10 per cent and 52.69 at 20 — and the breakdown point is
  derivable from its own parameters rather than discovered: with 27-observation
  subsets and 200 trials the expected number of clean subsets falls below one
  at a contamination of 0.1784. Huber degrades gradually instead: 6.88 arcsec
  at 20 per cent, 12.63 at 35, 83.17 at 50.
- **Omitting the azimuth-axis tilt from the model barely biases the other
  coefficients at all** — `|bias|/se` of 3.36 for `AN` and 1.01 for `AW`
  against 152 for `IA` — because the tilt columns are nearly orthogonal to
  every other column on a full-sky geometry. A missing tilt term shows up in
  the residual and not in the coefficients, and a control that tested only one
  of the two routes would have missed half the term set.
- **The rank analysis refuses more than it strictly must.** Scored against a
  truth-anchored criterion over five geometries and nine terms, its recall is
  1.0000 under both declared criteria and its precision is 0.9286 at 2 declared
  sigmas and 0.5000 at 10. It is a conservative refusal and the figure is
  reported rather than chosen.
- **The design matrix satisfies its bound with a worst ratio between 0.2627 and
  0.3888 across five decades** of misalignment amplitude, so the bound is
  neither violated nor vacuous; and its resolution is stated: at 10 arcsec
  coefficients it detects a column error of 0.12 to 0.38 per cent, and at
  100 arcsec coefficients a one-part-per-million column error is thirty
  thousand times inside it and cannot be seen.
- **Eighty-three negative controls across four scripts, of which 61 were
  declared must-fire and 61 fired, and 22 were declared must-not-fire and none
  fired.**

### Defects found during the build and pinned as regressions

- **Variance inflation computed from the truncated pseudo-inverse hid the
  degeneracy.** The first implementation took the inflation factors from the
  same truncated inverse the estimator uses, which sets the directions below
  the rank cut to zero, and reported a largest VIF of 1304 on a geometry that
  is exactly rank deficient by two. The factors now use the full spectrum and
  the same case reports 5.26e29. Pinned in `tests/test_regressions.py`.
- **Bennett's formula returns a negative refraction above 89.9225 degrees.**
  The tangent argument `h + 7.31/(h + 4.4)` reaches 90 degrees at the positive
  root of `h^2 − 85.6h − 388.69 = 0`. Caught while writing the design matrix,
  where the refraction column would have changed sign near the zenith and a
  fit would have attributed a positive refraction to a negative coefficient.
  The formula is refused above 89.9 degrees and the quadratic is solved in the
  test suite.
- **An absolute 1e-12 tolerance on the fit's equivariance identity is wrong.**
  Hypothesis falsified it at 2000 examples with coefficients `[0, 0, 0, 0, 0,
  0, 0, 0, 60.0]` and an offset of 19.0, giving a discrepancy of
  1.2363443602225743e-12. The bound now follows the conditioning of the solve,
  `1e-12 * condition_number * scale`, and the counterexample is pinned with
  literal inputs.
- **The first negative-control design could not detect a 1 per cent column
  error.** At 100 arcsec coefficients the linearisation bound admits about 3
  arcsec of sky error, which is more than a 1 per cent error produces, so nine
  of eighteen controls did not fire. The bound's resolution is now stated, the
  scale controls are run at 10 arcsec coefficients where the bound is tight,
  and a deliberately undetectable one-part-per-million corruption is added as a
  must-NOT-fire control.
- **The `pytest.approx` audit.** 0 of 241 tests depend on the hidden 1e-12
  absolute default. Forcing every explicit `abs=` to zero fails 9 of 241, six
  of which compare against an exact zero where a relative bound has no meaning
  and three of which are named individually in
  `validation/validate_approx_audit_output.txt` with their reasons.

### Known limitations

See `README.md`. The dominant ones: every number is measured against synthetic
residual sets generated by this package's own exact rotation model and not
against any real instrument; the term set is fixed at nine and a mount with a
tenth error source will have it absorbed by the nine; the first-order model
breaks down above about 1000 arcsec of total misalignment; the refraction
column rests on an optical fit used, in the radio case, outside the wavelength
it was fitted for; and the declared dependency ranges were measured at one
version each because the build environment forbids installing a second.

[0.1.0]: https://github.com/OmAcharya-avtr/pointmodel/releases/tag/v0.1.0
