# Validation — conjprob 0.1.0

**Validation level 3. Research grade. Not flight-qualified, not certified, not
approved for operational aerospace use. No output of this package is suitable as
an input to a manoeuvre decision.**

Every number in this file, in `README.md` and in `MODEL_CARD.md` was produced by
running one of the ten scripts in this directory in the session that wrote the
document. Each script writes its raw output beside itself as
`<script>_output.txt`, and the file names are given in every table below.
Nothing here is carried from an earlier session and nothing is estimated.

## Environment measured against

| Component | Version |
|---|---|
| Python | 3.13.16 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| sgp4 | 2.27 |
| pytest | 9.1.1 |
| hypothesis | 6.170.5 |
| ruff | 0.17.1 |
| CPU cores | 2, shared |

Every dependency is bounded on both sides in `pyproject.toml` at the version it
was measured against. A floor without a ceiling is not a dependency
specification: a product measured here and resolved later against a newer minor
version is a product whose published numbers no longer describe it.

## How to reproduce all of it

```bash
python -m pytest tests/ -q --junitxml=junit.xml
ruff check src/ tests/ examples/ validation/
python validation/validate_geometry.py
python validation/validate_analytic.py
python validation/validate_regimes.py
python validation/validate_montecarlo.py
python validation/validate_dilution.py
python validation/validate_nonlinear.py
python validation/validate_screening.py
python validation/validate_tle.py
python validation/validate_cli.py
python validation/worked_example.py
```

Each script changes its working directory to the repository root before doing
anything, so all of them run identically from any directory. That was verified
rather than assumed: all ten were executed from the filesystem root, the system
temporary directory, the product root and `validation/`, and all forty runs
exited zero.

## Test suite

Counted from `junit.xml`, not from pytest's standard output, because a suite that
collects nothing prints success.

| Quantity | Value |
|---|---|
| `testsuite` attribute `tests` | 269 |
| `testcase` elements, counted independently | 269 |
| failures | 0 |
| errors | 0 |
| skipped | 0 |
| wall clock | 8.992 s, shared two-core container, unstable between runs |

`ruff check src/ tests/ examples/ validation/` with `line-length = 100` and the
`E`, `F`, `W`, `I`, `UP`, `B` rule sets: **All checks passed**, exit 0.

## The references the analytic methods are scored against

Three independent references, in increasing order of independence.

1. **An exact closed form.** At zero miss distance with a circular covariance,
   `Pc = 1 - exp(-R^2 / (2 sigma^2))`, exactly, for every `R / sigma`.
2. **An exact distribution function in a different library.** For any circular
   encounter-plane covariance the collision probability is the cumulative
   distribution function of a non-central chi-square variate with two degrees of
   freedom: with `u = R^2 / sigma^2` and `v = d^2 / sigma^2`,
   `Pc = P(chi'^2_2(v) <= u)`, evaluated by `scipy.stats.ncx2`. This is a
   separately implemented special function and is the strongest offline check
   available.
3. **An independent quadrature and an independent Monte Carlo** in this package,
   which share no algebra with the analytic paths beyond the definition of the
   integral: `conjprob.quadrature.pc_quadrature` integrates the bivariate normal
   over the hard-body disc in polar coordinates with the full covariance, using
   neither the error-function reduction nor the eigendecomposition, and
   `conjprob.montecarlo.pc_monte_carlo` samples the 3-D relative state and so
   validates the encounter-plane projection itself rather than only the integral.

An honest note on independence that the measurement forced: `pc_foster` and
`pc_alfano` are two quadratures of the *same* one-dimensional reduction, so their
agreement is a numerical check and not an independent derivation. `pc_chan` is
independent algebra. This is stated in the module docstring as well.

## 1. Analytic methods against an exact distribution function

`validate_analytic_output.txt`, check 2. Grid: Mahalanobis distance in
{0.25, 0.5, 1, 2, 4, 8} and `R / sigma` in {0.01, 0.1, 0.5, 1, 2}, circular
covariance, 30 cells. Chan evaluated at `chan_recommended_order(M)`.

| Method | Worst relative error vs `scipy.stats.ncx2` | Tolerance | Result |
|---|---|---|---|
| `pc_foster` | 6.661338e-16 | 1e-9 | PASS |
| `pc_chan` | 1.110223e-15 | 1e-9 | PASS |
| `pc_quadrature` | 5.551115e-16 | 1e-9 | PASS |
| `pc_alfano` (2001 nodes) | 1.479235e-04 | 5e-4 | PASS |
| `pc_small_body_approximation` | 9.999469e-01 | none — this is a leading-order approximation and its error is the measurement | reported |

Alfano's tolerance of 5e-4 is derived from its measured convergence order (check
4 below) and the node count used, not chosen to make this grid pass.

The Rayleigh closed form at zero miss distance (check 1) is reproduced by
`pc_foster` to a worst relative error of 3.330669e-16 over `R / sigma` from 0.01
to 8, by `pc_chan` **at order 0** to 8.881784e-16 — exact, because the series has
a single term when the miss vector is zero — and by `pc_alfano` at 2001 nodes to
4.622818e-06.

## 2. Alfano's measured convergence order

`validate_analytic_output.txt`, check 4. Geometry `sigma = (300, 54) m`,
`miss = (120, 30) m`, `R = 12 m`.

| Nodes | Relative error | Measured order |
|---|---|---|
| 51 | 1.176e-03 | |
| 101 | 4.154e-04 | 1.501 |
| 201 | 1.468e-04 | 1.501 |
| 401 | 5.189e-05 | 1.500 |
| 801 | 1.834e-05 | 1.500 |
| 1601 | 6.485e-06 | 1.500 |
| 3201 | 2.293e-06 | 1.500 |

**Mean measured convergence order 1.5004 against Simpson's nominal 4.** The
half-chord `sqrt(R^2 - x^2)` has an infinite derivative at both endpoints of the
integration interval, which caps the attainable order at 3/2. This is a property
of the published fixed-node scheme and is reported rather than worked around. The
relative error at the package default of 201 nodes is **1.468000e-04**.

## 3. Chan's series order

`validate_analytic_output.txt`, check 3. The series weights its terms by a
Poisson distribution of mean `M^2 / 2`, so a fixed order is adequate only up to a
fixed Mahalanobis distance. Relative error of the default order of 3 against
`scipy.stats.ncx2`:

| M | at `R/sigma = 0.5` | at `R/sigma = 3` |
|---|---|---|
| 1 | 4.937e-09 | 8.243e-04 |
| 2 | 1.163e-06 | 7.039e-02 |
| 4 | 2.182e-04 | 7.573e-01 |
| 8 | 2.046e-02 | 9.997e-01 |
| 12 | 1.515e-01 | 1.000e+00 |

At `chan_recommended_order(M)` the worst relative error over the whole sweep is
1.332e-15. `conjprob.report.assess_conjunction` therefore uses the recommendation
and not a fixed order, so that the Chan error it reports is the method's own
equivalent-area error rather than an artefact of a truncation chosen here.

## 4. Measured validity regimes

`validate_regimes_output.txt`. Grid: 60 logarithmic points in `R / sigma_eff`
from 1e-3 to 6.31, aspect ratios {1, 2, 5, 10, 20, 30}, Mahalanobis distances
{0.25, 0.5, 1, 2, 4}. Reference: `pc_quadrature`.

Worst absolute relative error anywhere on the 1800-point grid:

| Method | Worst relative error |
|---|---|
| `pc_foster` | 1.408800e-13 |
| `pc_alfano` (201 nodes) | 2.453471e-03 |
| `pc_chan` | 2.906285e+01 |
| `pc_small_body_approximation` | 3.933262e+00 |

Chan is wrong by a factor of 30 and the leading-order approximation by a factor
of 5 somewhere on this grid. **That is the product's reason for existing: a
single scalar from one of these methods, with no regime statement beside it, can
be wrong by more than an order of magnitude.**

Measured `R / sigma_eff` at which the relative error first exceeds **1 per cent**.
Blank means the tolerance was never exceeded anywhere in the swept range.

`pc_chan`:

| aspect | M=0.25 | M=0.5 | M=1 | M=2 | M=4 |
|---|---|---|---|---|---|
| 1 | — | — | — | — | — |
| 2 | 0.4142 | 0.4364 | 0.5609 | 0.457 | 0.1541 |
| 5 | 0.1608 | 0.1691 | 0.2221 | 0.1623 | 0.06002 |
| 10 | 0.101 | 0.1061 | 0.1397 | 0.1012 | 0.03766 |
| 20 | 0.0675 | 0.07106 | 0.09358 | 0.0675 | 0.02524 |
| 30 | 0.05426 | 0.057 | 0.07519 | 0.0542 | 0.02024 |

`pc_small_body_approximation`:

| aspect | M=0.25 | M=0.5 | M=1 | M=2 | M=4 |
|---|---|---|---|---|---|
| 1 | 0.2025 | 0.2131 | 0.2818 | 0.2013 | 0.07576 |
| 2 | 0.1814 | 0.1905 | 0.2516 | 0.1807 | 0.06771 |
| 5 | 0.1257 | 0.1325 | 0.1746 | 0.1253 | 0.04707 |
| 10 | 0.09019 | 0.09492 | 0.1251 | 0.09 | 0.03372 |
| 20 | 0.06414 | 0.06734 | 0.08885 | 0.064 | 0.02392 |
| 30 | 0.05232 | 0.05515 | 0.07269 | 0.05221 | 0.01958 |

`pc_foster` and `pc_alfano` never exceed 1 per cent in any of the 30 cells.
At a **0.1 per cent** tolerance Alfano does: its boundary spans 1.941 to 5.149
and 20 of 30 cells still never exceed it.

Two readings worth stating:

- **Chan's equivalent-area substitution is exact at aspect ratio 1** — the
  tolerance is never exceeded in that whole row — and its boundary collapses by
  a factor of 7 between aspect 2 and aspect 30. The equivalent-area step is the
  approximation, and the aspect ratio is what it costs.
- The small-hard-body boundary is nearly flat in aspect ratio and is set almost
  entirely by the hard-body size, which is what the leading-order expansion
  predicts.

The quadrature reference's own reported absolute error, relative to its value, is
below 8.384586e-10 over a sample spanning `R/sigma_eff` from 1e-3 to 5 and `M`
from 0.5 to 6 (check 4).

## 5. Monte Carlo reference

`validate_montecarlo_output.txt`.

### Convergence, case `leo-crossing`, analytic value 3.5934922751e-03

| Samples | Hits | Pc | 95 per cent interval | Contains the analytic value |
|---|---|---|---|---|
| 10000 | 37 | 3.70000000e-03 | [2.60643901e-03, 5.09639125e-03] | True |
| 30000 | 100 | 3.33333333e-03 | [2.71293107e-03, 4.05276430e-03] | True |
| 100000 | 367 | 3.67000000e-03 | [3.30467291e-03, 4.06456209e-03] | True |
| 300000 | 1139 | 3.79666667e-03 | [3.57973408e-03, 4.02326946e-03] | True |
| 1000000 | 3621 | 3.62100000e-03 | [3.50421307e-03, 3.74067643e-03] | True |
| 3000000 | 10816 | 3.60533333e-03 | [3.53782333e-03, 3.67380427e-03] | True |

6 of 6 intervals contain the analytic value. **Measured log-log slope of the
interval width against the sample count: -0.505200**, against the -0.5 expected
for a binomial estimator, tolerance 0.05.

### Measured interval coverage

This is the only honest check of an interval, and it was done rather than
assumed: 300 independent seeds, 100000 samples each.

| Quantity | Value |
|---|---|
| nominal coverage | 0.95 |
| **measured coverage** | **0.960000** (288 of 300) |
| 95 per cent interval on the coverage itself | [0.931171, 0.979163] |
| relative bias of the mean of the 300 estimates | +1.208033e-03 |
| standard error of that mean | 2.999838e-03 relative |

The measured coverage is at or above nominal, which is what a Clopper-Pearson
interval should do, and the bias is within one standard error of zero.

### Practical floor

Samples needed for a 95 per cent interval of width `0.1 Pc` follow `1537 / Pc`.
At the measured rate of about 3.3e6 samples per second this gives 1.537e+05
samples and 0.0007763 minutes at `Pc = 1e-2`, 1.537e+08 samples and 0.7763
minutes at `Pc = 1e-5`, and 7.763 minutes at `Pc = 1e-6`. **Monte Carlo is usable
as a reference down to about `Pc = 1e-5` inside this product's three-minute budget
and not below**, which is why the shipped `far-miss-tight-covariance` case at
`Pc = 2.65e-35` has no Monte Carlo row. Sample rates are wall clock on a shared
two-core container and are unstable; the scaling is the point, not the number.

## 6. Dilution

`validate_dilution_output.txt`.

| Check | Measured | Tolerance | Result |
|---|---|---|---|
| closed-form maximum against `R^2 / (e d^2)` (Alfano 2005b), 9 geometries | worst relative difference 4.440892e-16 | 1e-13 | PASS |
| closed form against the numerical maximum of the Foster-Estes Pc, `R/sigma_eff <= 0.01` | worst relative error 2.882619e-04 | 1e-3 | PASS |
| scale at the maximum against `M / sqrt(2)`, 7 Mahalanobis distances | worst relative difference 6.257130e-04 | 1e-3 | PASS |
| the Pc-against-scale curve has a single interior maximum, strictly rising left and strictly falling right, over 241 sampled scales | maximum 7.20571701e-03 at scale 0.482318 | monotone on both branches | PASS |
| `DilutionWarning` raised by the library on the two dilution-branch cases and on neither other case | 2 raised, 1 not raised, matching the measured branch flag | exact match | PASS |

The error of the closed-form maximum scales as `(R/sigma_eff)^2`, which is the
order of the neglected term; at `R/sigma_eff = 0.01` the measured
`|relative error| / (R/sigma_eff)^2` is 2.882619 at aspect 5 and 0.000050 at
aspect 1.

### The dilution region on the shipped case

`worked_example_output.txt`. The declared covariance of `leo-crossing` gives
`Pc = 3.593492e-03`. The maximum attainable by varying the covariance scale is
**7.209827e-03, at scale 0.473905** — that is, with a covariance about half the
size. The declared number is therefore **2.0064 times smaller than the worst
case**, and `dPc / d ln(scale) = -5.480804e-03`, negative: inflating this
covariance would make the reported probability *smaller*. The Mahalanobis
distance is 0.685422, below `sqrt(2)`.

**A small Pc here is not evidence of a small risk, and the library says so by
raising a warning rather than by setting a flag the caller may not read.**

## 7. Breakdown of the rectilinear 2-D formulation

`validate_nonlinear_output.txt`. Case `large-body-tight-covariance`, chosen
because its Pc is of order 0.5 so a Monte Carlo interval under one per cent wide
costs 60000 samples and the effect is resolvable.

### Finite screening window

| T / t_cross | Pc in the window | Ratio to unbounded | 95 per cent interval on the ratio |
|---|---|---|---|
| 0.05 | 0.37927000 | 0.791680 | [0.783654, 0.799786] |
| 0.10 | 0.38747000 | 0.808796 | [0.800674, 0.816999] |
| 0.20 | 0.40271000 | 0.840608 | [0.832310, 0.848986] |
| 0.50 | 0.43747500 | 0.913176 | [0.904495, 0.921938] |
| 1.00 | 0.46640000 | 0.973553 | [0.964571, 0.982618] |
| 2.00 | 0.47843500 | 0.998675 | [0.989572, 1.007861] |
| 4.00 | 0.47907000 | 1.000000 | [0.990891, 1.009193] |
| 8.00 | 0.47907000 | 1.000000 | [0.990891, 1.009193] |

The ratios are non-decreasing in window length and never exceed one. The 2-D
formulation integrates over all time, so at half a crossing time it **overstates
the probability of a conjunction inside the window by a factor of 1.0951**.

### Curvature

A declared constant relative acceleration along the miss direction, over a window
of 8 crossing times. `kappa = |a| T^2 / (2 sigma_eff)`.

| kappa | Monte Carlo Pc | Relative error of the 2-D value | Interval |
|---|---|---|---|
| 1.0000e-03 | 0.479150 | -0.000668 | [-0.008955, +0.007754] |
| 1.0000e+00 | 0.477483 | +0.002820 | [-0.005524, +0.011300] |
| 1.0000e+01 | 0.469117 | +0.020705 | [+0.012069, +0.029483] |
| 1.0000e+02 | 0.457683 | +0.046203 | [+0.037146, +0.055410] |
| 1.0000e+03 | 0.429050 | +0.116023 | [+0.105786, +0.126433] |
| 1.0000e+04 | 0.417983 | +0.145571 | [+0.134823, +0.156503] |

**Measured: the rectilinear 2-D error reaches 1 per cent at kappa = 3.288387 and
10 per cent at kappa = 6.702773e+02.**

The curved estimator discretises time, which can only underestimate Pc, so its
own bias was measured by refining the grid at fixed seed: 9 nodes gives
-1.030660e-01, 33 nodes -6.821331e-03, 129 nodes with no refinement
-3.197499e-04, and the package default of 129 nodes with 4 refinements agrees
with a 513-node, 6-refinement grid to **+0.000000e+00**.

### What the measured kappa means in a quantity an operator has

`kappa = |a| w^2 sigma_eff / (2 |v|^2)` with `w = 8`, so
`|v| = w sqrt(|a| sigma_eff / (2 kappa))`. Taking `|a|` as the gravity-gradient
scale `mu |dr| / r^3` at `r = 7078137 m`:

| sigma_eff | separation | gravity gradient | speed at 1 per cent | speed at 10 per cent |
|---|---|---|---|---|
| 18.5966 m | 21.4056 m | 2.40607e-05 m/s^2 | 0.065986 m/s | 0.004622 m/s |
| 200 m | 300 m | 3.37211e-04 m/s^2 | 0.810120 m/s | 0.056743 m/s |
| 1000 m | 500 m | 5.62019e-04 m/s^2 | 2.338616 m/s | 0.163803 m/s |

A typical low Earth conjunction has a relative speed of several kilometres per
second, four orders of magnitude above these figures, **so the rectilinear
assumption is excellent there**. It is the slow encounter — metres per second and
below, the geostationary co-location case — where the 2-D formulation in this
package should not be used. Patera (2001) is the established method for a curved
trajectory and is not implemented here.

## 8. The learned screen against the analytic screens

`validate_screening_output.txt`. Full tables are in `MODEL_CARD.md`. Recall at an
achieved false-positive rate of about 1 per cent, 12000 test conjunctions per
regime, model trained on `nominal` only:

| Screen | `nominal` | `homogeneous` | `dilution` |
|---|---|---|---|
| `miss_distance` | 0.30112 | 0.91280 | 0.02954 |
| `mahalanobis` | 0.03436 | 0.83720 | 0.01297 |
| `small_body` (one-line closed form) | 0.97733 | **1.00000** | **1.00000** |
| `learned` (random forest) | **0.99284** | 0.98840 | 0.99784 |

Paired exact binomial tests on the discordant positives: the learned model's win
on `nominal` is real (85 against 20, p = 1.008e-10); the closed form's win on
`homogeneous` is real (35 against 0, p = 5.821e-11); on `dilution` the 3
discordant pairs do not resolve a difference (p = 0.25).

**The honest verdict: a closed-form expression that fits on one line, uses the
same features and has no training step is the better screen on two of the three
populations measured, and the learned model is ahead only on the population it
was trained on.**

**The specification's prediction that a Mahalanobis threshold would match the
learned model does not hold on the broad population**, where it is the worst of
the three analytic screens at 0.03436 recall. The reason was measured rather than
argued: the Mahalanobis distance is invariant to `R / sigma_eff`, which spans
3.92251 decades on `nominal` and 0.26125 decades on `homogeneous`, and
`R / sigma_eff` is half of the leading-order Pc. On `homogeneous` the same screen
recovers to 0.83720. The prediction holds under a condition the specification did
not state. Nothing was retuned after this was measured.

### A check that FAILED

| Check | Measured | Tolerance | Result |
|---|---|---|---|
| calibration of the learned probabilities, worst reliability-bin gap | **0.272486** | 0.15 | **FAIL** |

It is published and not repaired. Recall at a fixed false-positive rate depends
only on the ordering of the scores and is invariant under any strictly monotone
recalibration, so an isotonic or Platt fit would move the Brier score and the
reliability table and would not move a single recall number above. The
consequence for a user is in `MODEL_CARD.md`: **the predicted probability is
usable as a ranking and is not usable as a probability.**

### The uncertainty output

Per-tree dispersion is 4.894, 3.121 and 7.932 times larger on the cases the model
gets wrong than on the cases it gets right, on the three regimes. It ranks errors
and is not a confidence interval.

### Features that cannot matter

0.057305 of the forest's total importance is assigned to the relative speed and
the out-of-plane standard deviation, both of which provably cancel out of the
short-encounter integral. The model is partly fitting generator structure, and
that is one reason the closed form generalises better.

## 9. Geometry

`validate_geometry_output.txt`.

| Check | Measured | Tolerance | Result |
|---|---|---|---|
| time of closest approach against a hand calculation (head-on at 10 m/s from 100 m, so tca = 10 s, miss = 0) | tca 10.000000000000000 s, miss 0.000000e+00 m | 1e-12 s, 1e-9 m | PASS |
| miss vector perpendicular to the relative velocity, 200 random geometries, normalised residual | 0.000000e+00 | 1e-12 | PASS |
| in-plane trace preserved by the projection of an isotropic covariance, 200 cases | 0.000000e+00 | 1e-12 | PASS |
| Pc invariance under a rigid rotation of the whole problem, 32 rotations on each of 3 cases | worst 4.070440e-10 | 1e-6 | PASS |
| Pc invariance under a common rescaling of every length, factors 1e-4 to 1e4 | worst 5.995204e-15 | 1e-9 | PASS |
| Mahalanobis distance from the principal frame against `sqrt(m^T C^-1 m)`, 200 correlated covariances | worst 3.155254e-13 | 1e-9 | PASS |

## 10. The TLE entry point

`validate_tle_output.txt`. Reference: `SGP4-VER.TLE` and `tcppver.out`, the
official verification cases published with Vallado et al. (2006), AIAA 2006-6753,
which ship inside the installed `sgp4` package. Satellite 00005, 31 published
epochs.

| Check | Measured | Tolerance | Result |
|---|---|---|---|
| position against the published values | worst 6.819541e-06 m | 1e-4 m | PASS |
| velocity against the published values | worst 7.704807e-07 m/s | 1e-5 m/s | PASS |
| closest-approach search: spread of the refined minimum over coarse-scan steps from 4 to 2000 | 5.596563e-08 m | 1e-6 m | PASS |
| `synthetic_secondary`: declared offset magnitude and speed preserved under the velocity rotation | measured offset 120.830459736 m against a declared 120.830459736 m; primary and secondary speeds equal to 1e-12 relative | 1e-9, 1e-12 | PASS |

This check does **not** establish that SGP4 is correct — that is the `sgp4`
package's own verification, cited not claimed. It establishes that this product's
use of the API reproduces the published output.

One measured correction to the package's own documentation came out of it: the
relative speed produced by `synthetic_secondary` is
`2 |v_perp| sin(rotation / 2)` and not `2 |v| sin(rotation / 2)`, because the
rotation is about the radial direction and the orbit is eccentric so the velocity
has a radial component the rotation does not move. The naive form is 0.28 per
cent wrong on this case and the docstring now states the corrected one.

## 11. Command-line interface

`validate_cli_output.txt`. Nine invocations, each checked for its exit status and
for required fragments in its output, including the two that must fail: a missing
case file and an unknown generator regime both return 2 and print a message
naming the problem. The full `assess` report for the shipped `leo-crossing` case
is reproduced verbatim in the output file. A scan of that output for tokens
beginning with a path separator returns **0**.

## 12. Defects this package's own first implementation contained

All six were found by the property-based tests in `tests/test_properties.py`, not
by the hand-written tests, and all six are now pinned as named regression tests
with literal inputs in `tests/test_regressions.py` — seven tests in total — because
`.hypothesis/` is gitignored and a counterexample found by one run is otherwise
lost. Four are re-measured in `validate_analytic_output.txt`, check 5, and one in
`validate_dilution_output.txt`, check 6.

| Defect | Found at | Wrong by | Fix |
|---|---|---|---|
| An absolute quadrature tolerance of 1e-14 stopped the adaptive integrator on the first subdivision for any Pc at or below it, destroying scale invariance. | `sigma = (2160, 2134) m`, `miss = (2, 18061) m`, `R = 200 m`, scaled by 19 | the two gave 1.2142e-18 and 1.2791e-18, a 5.3 per cent disagreement in a quantity invariant by construction, and both were wrong: the correct value is 1.305824127767e-18, now reproduced by both to 7.327472e-15 relative | `epsabs` defaults to 0, so the tolerance is purely relative |
| The published error-function form of the chord mass cancels to exactly zero when the whole chord is many standard deviations from the mean. | `sigma_y = 3 m`, `ybar = 68 m`, `h = 1 m` | returned 0.0 where the correct mass is 8.767418031863e-111, giving Pc = 0 where the 2-D quadrature gives 1.007060160815e-111 | a complementary-error-function branch on whichever side keeps both arguments positive; the same defect was present in `pc_alfano` |
| Chan's published finite exponential sum is a difference of two numbers that both approach one and loses up to twelve significant figures. | `sigma = 1 m`, `ybar = 8 m`, `R = 1 m` | 4.273235284733e-13 against an exact 4.271014885290e-13, a relative error of 5.2e-4; now 4.440892e-16 | the bracket is evaluated as `gammainc(m + 1, u / 2)` |
| Chan's series needs an order growing as `M^2 / 2`; the order of 3 the papers recommend is not enough. | `sigma = 1 m`, `ybar = 9 m`, `R = 9 m`, order 60 | 2.0e-6 relative; at order 3 and `M = 12` it is 15 per cent | `chan_recommended_order` added and made the default in `assess_conjunction` |
| Chan's series accumulates `(v/2)^m / m!` and multiplies by `exp(-v/2)` at the end; the accumulation overflows while the prefactor underflows. | `sigma = 437 m`, `xbar = 16515 m`, `R = 1 m`, so `v / 2 = 714` | returned `nan`; the correct value is 1.92739845e-316, now agreeing with Foster-Estes to 2.563381e-08 relative | the term weights are evaluated in logarithms as a Poisson distribution |
| A subnormal miss component leaves the unsquared Mahalanobis hypotenuse positive while its square underflows to zero. | `xbar = 2.2250738585072014e-308 m`, `sigma = 1 m` | division by zero in `max_probability` | the squared quantity is the guarded one and the error message names the components |

Stating this plainly: **four of the six are defects in the published forms of the
equations as literally transcribed**, not in the transcription. Two of the three
analytic methods in this package return visibly wrong answers — zero, `nan`, or
five significant figures of error — if written the way the source papers write
them, in regimes an operational tool reaches. That is a result of this build and
it is the reason the regression tests exist.

The sixth defect, and the check that found it, were added *after* the first draft
of this document was written: a single additional property-based run with a fresh
database found the `nan`. Five runs at 60 examples and one at 500 examples per
property have since passed with no new counterexample, which is evidence and not
a proof.

## 13. What this validation does not establish

- **That any output is suitable for a manoeuvre decision.** It is not, and no
  amount of method agreement would make it so.
- **That the covariance is right.** Every number in this document is conditional
  on the supplied covariance. In practice the dominant error in a collision
  probability is that the covariance is wrong or misreported, and nothing here
  can measure that. This is the single largest gap between this product and
  operational conjunction assessment.
- **That the learned screen transfers to a real catalogue.** It was trained and
  tested on a declared synthetic generator. See `DATASET_CARD.md`.
- **That Pc below about 1e-5 has been checked against Monte Carlo.** It has not;
  the sample count is outside the compute budget. Below that, the evidence is the
  agreement of the analytic methods with the quadrature reference and with
  `scipy.stats.ncx2`, which holds to machine precision but is not an independent
  sampling check.
- **That a curved-trajectory Pc is computed anywhere.** The curvature breakdown
  is measured and the remedy is cited, not implemented.
