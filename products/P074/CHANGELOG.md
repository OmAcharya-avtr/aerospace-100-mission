# Changelog

All notable changes to orbitdrift are recorded here. Research-grade software;
not flight-qualified, not certified, not approved for operational aerospace use.

## 0.1.0 — 2026-10-25

First release. Status: TESTING.

### Added

- `orbitdrift.divergence` — the two baselines. `model_divergence` compares SGP4
  against the independent numerical force model, seeded from the SGP4 state at
  the element-set epoch so the curve is zero at zero age.
  `self_consistency_divergence` compares one object's earlier element set
  against its later one. `larger_error_lower_bound_km` returns half the
  separation and the module derives why that, and nothing more, is what the
  measurement supports.
- `orbitdrift.forces` — the declared force model: two-body, `J2` to `J4` zonal
  as the analytic gradient of a published potential, drag on the static
  piecewise-exponential atmosphere of Vallado 2013 Table 8-4, and luni-solar
  point masses from low-precision analytic series. WGS72 constants, the ones
  SGP4 uses, so the divergence is attributable to model structure.
- `orbitdrift.integrate` — classical fourth-order Runge-Kutta, vectorised across
  satellites rather than across time, with a tabulated luni-solar ephemeris.
  `invariant_residuals` reports the energy, `|h|` and `h_z` residuals and the
  module states which of the three the zonal field actually conserves.
- `orbitdrift.kepler` — closed-form two-body propagation and element
  conversions, so the integrator has an independently derived reference to be
  checked against rather than merely compared with.
- `orbitdrift.frames` — Greenwich mean sidereal time (IAU 1982), TEME to ECEF,
  Bowring geodetic height, the radial/in-track/cross-track basis, and
  low-precision Sun and Moon series. Nothing here is cross-checked against
  `astropy`, which is not installed.
- `orbitdrift.vectors` — an independent parser for the published SGP4
  verification vectors that ship inside the installed `sgp4` distribution, and a
  numeric comparison against them.
- `orbitdrift.growth` — the analytic growth forms: linear, quadratic, free power
  law and drag-dominated exponential, each derived from a stated dominant error
  term. Fitted in log space, with rank deficiency refused rather than resolved
  by a pseudo-inverse.
- `orbitdrift.predictors` — the population predictors: a rule of thumb, five
  analytic log-linear forms, a gradient-boosted quantile model, and a split
  conformal wrapper for any of them. `leverage` measures the distance from the
  collinear feature manifold the fit is valid on.
- `orbitdrift.calibration` — Clopper-Pearson intervals, interval metrics
  including the Winkler score, object-level coverage, and a stratified coverage
  table that flags cells inconsistent with the nominal level.
- `orbitdrift.benchmark` — the two benchmark tasks, with the analytic baselines
  fitted and measured before the learned model is constructed, and three
  rankings: overall, calibrated, and not-under-covering.
- `orbitdrift.staleness` — the decision the product exists for: the epoch age at
  which a declared position tolerance is reached, reported as both a median
  horizon and a conservative horizon, with the measured coverage of the interval
  attached.
- `orbitdrift.guard` and `orbitdrift.report` — the covariance prohibition,
  enforced by a purpose screen, a serialised-key audit and two methods that
  exist only to refuse, each cross-referencing P071 ConjProb.
- `orbitdrift.dataset` — the deterministic synthetic population, written out as
  real two-line element sets, and the vectorised divergence table built from it.
- A command-line interface: `version`, `vectors`, `atmosphere`, `drift` and
  `stale`.
- 273 tests, 20 validation scripts with their committed raw output, 7 examples
  with their screenshots, `docs/REQUIREMENTS.md` for Level 3,
  `validation/VALIDATION.md`, `MODEL_CARD.md` and `DATASET_CARD.md`.

### Measured results recorded at this release

- SGP4 reproduced against its own published vectors: 33 cases, 666 rows, worst
  position disagreement 1.154503e-07 km, worst velocity disagreement
  4.996954e-10 km/s.
- Energy and `h_z` conserved to 1.7174e-09 and 8.5745e-10 relative at 360 steps
  per revolution over six revolutions; `|h|` **not** conserved under `J2`, with
  a step-invariant residual of 5.125159e-04.
- Convergence order 4.3701 to 4.7123 by Richardson, with no reference solution.
- The fitted growth exponent above 1100 km is 0.9685 +/- 0.0628, which is the
  linear-in-time regime the physics predicts for a constant mean-motion error.
- On the extrapolation task the best point prediction is analytic
  (`own_curve_drag_exponential`, 0.1299 median absolute log error, against the
  best learned at 0.1391), the only calibrated interval is analytic, and **no
  learned predictor reaches its nominal coverage.**
- On the cold-prediction task the learned model wins the point prediction
  (0.5942 against 0.7410) and **every predictor under-covers at 0.90**, so that
  table has no operationally usable row.
- An analytic fit's own in-sample residual spread covers 0.3069 of held-out
  extrapolation rows against a claimed 0.90. Conformal calibration raises it to
  0.9474.

### Defects found and fixed during construction

- **The drag unit conversion divided by 1000 where it had to multiply**, a factor
  of 1e6, so the drag term was invisible. Found by a negative control — switching
  the term off and requiring the divergence to change, which it did by 2.09e-07
  relative — after a unit test that recomputed the same expression had passed
  throughout. Now pinned by an independent SI evaluation and an
  order-of-magnitude check against a figure from the literature.
- **The published-vector parser keyed expected blocks by catalogue number**,
  merging the two cases that share catalogue 20413 and inflating the row count
  from 667 to 763. Blocks are now matched positionally and the row count is
  pinned.
- **The integration step was derived from the shortest period in each batch**, so
  a subset recomputation did not reproduce the committed rows. The step is now a
  declared input recorded in the table's provenance string.
- **The synthetic self-consistency pair advanced the later element set's angles
  by the Kozai mean motion**, which is not SGP4's secular rate, so the
  identical-pair control came out 6882 km apart. Fixed using `Satrec.mdot`,
  `argpdot` and `nodedot`; the identical pair now separates by 7.69e-10 km.
- **`Satrec.twoline2rv` accepts malformed element sets without raising** —
  measured on five inputs, none raised — so `load_population` validates line
  length, ASCII and the catalogue-number match itself.
- **The log-linear predictor extrapolates catastrophically off the collinear
  feature manifold**: a hand-assembled 1300 km feature vector drew a prediction
  of 2.5e8 km against the 17.7 km measured for a real object. Diagnosed rather
  than fixed, with `AnalyticGrowthPredictor.leverage` and a documented
  instruction to build features from an actual element set.
- **A geodetic-height test asserted a tolerance Bowring's single-pass formula
  cannot deliver at orbital altitude.** The claim was corrected to the measured
  height dependence rather than the tolerance loosened silently.
- **A drag unit test used the density at a round 400 km** while the function
  computes 399.998 km, because the state is built from the WGS72 radius and the
  height from the WGS84 ellipsoid. The two-metre difference between the
  reference radii is now stated in the test.
- **Hypothesis falsified a flat 1e-9 radian tolerance on the recovered
  inclination**, at `a = 6600` km, `e = 0` and `i = 5.960464477539063e-08`.
  `arccos(h_z/|h|)` has derivative `-1/sin(i)`, so the recovery's accuracy floor
  near zero inclination is `eps/sin(i)`, about 3e-08 radian there. The tolerance
  is now derived from that conditioning, the limitation is documented in
  `rv_to_elements`, and the counterexample is pinned with literal inputs. No
  hand-written case used an inclination below one degree.

### Known limitations at this release

Recorded in full in `README.md` and `docs/REQUIREMENTS.md` §10. The three that
most constrain what the numbers mean:

- Error against truth is not measured and cannot be. There are no precision
  ephemerides and no element-set archive in this environment.
- The atmosphere model is static, with no solar index. Published solar-cycle
  density variation at 400 km is about an order of magnitude.
- For 3 of 192 objects the integrator's truncation floor is more than 10 per cent
  of the divergence reported, reaching 18.5 per cent for the least divergent one.
  Those rows should be read as "divergence below about a kilometre".

### Dependency ranges

Every range is bounded on both sides at the versions every number here was
measured against: `numpy>=2.5,<2.6`, `scipy>=1.18,<1.19`,
`scikit-learn>=1.9,<1.10`, `sgp4>=2.27,<2.28`. The cost is real and is recorded
under Limitations: this package cannot take a `numpy` 2.6, a `scipy` 1.19, a
`scikit-learn` 1.10 or an `sgp4` 2.28 release without a port and a
re-measurement. The `sgp4` ceiling is the tightest for a reason beyond API
churn: the published verification vectors are read out of that distribution as
package data, and a release that revised either file would change a published
number here.

## Licence

Apache-2.0, copyright 2026 OPTIMA Organisation. The credits line required by
the mission appears once, in README.md under Credits.
