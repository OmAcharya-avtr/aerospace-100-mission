# Validation — orbitdrift (P074)

Validation Level 3. Every number in this document was produced by a script in
this directory, executed in the session that wrote the document, and the raw
output of each run is committed beside it as `<script>_output.txt`. The
requirement each check answers is in `docs/REQUIREMENTS.md`.

**Checks that failed, and claims that are only partly satisfied, are in this
document.** The two that matter are in §4.4 and §10.

Environment every number was measured in. Two interpreters were used and they
agree on everything numerical and differ only on test and plotting tooling; the
committed `*_output.txt` files were last written by the system interpreter, and
the suite and gate conditions were verified under both:

| package | system `python3` (the gate's) | pinned venv |
|---|---|---|
| python | 3.13.17 | 3.13.17 |
| numpy | 2.5.3 | 2.5.3 |
| scipy | 1.18.1 | 1.18.1 |
| scikit-learn | 1.9.1 | 1.9.1 |
| sgp4 | 2.27 | 2.27 |
| pytest | 9.1.1 | 8.4.2 |
| hypothesis | 6.170.9 | 6.140.3 |
| matplotlib | 3.11.2 | 3.10.8 |
| ruff | 0.16.9 | 0.14.2 |
| astropy | not installed | not installed |
| torch | not installed | not installed |

Nothing here is cross-checked against `astropy` or `torch`. Two cores.

## 0. The claim being validated

This package does not measure the error of a two-line element set. It cannot:
there are no precision ephemerides and no element-set archive in this
environment, so there is nothing to call truth. It measures **model
divergence** — how fast two propagations of the same object separate — and the
only rigorous statement that supports is a triangle inequality.

If two propagators disagree by `D` at some time, and `e1` and `e2` are their
unknown errors against truth, then `|e1| + |e2| >= D`, so
`max(|e1|, |e2|) >= D/2`. That is a lower bound on **the larger** of the two
models' position errors and nothing more.

`validate_lower_bound.py` checks it against a case where truth is known in
closed form, verifies that it does **not** bound either model individually, and
shows the same inequality written as `D` rather than `D/2` failing on the same
data.

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| `max(\|e1\|,\|e2\|) >= D/2` over 20 model pairs x 401 samples | closed-form two-body truth from `orbitdrift.kepler` | 0 violations, worst margin 0.000e+00 km | exact inequality, 1e-9 km slack |
| the bound is not vacuous | — | largest bound 4132.1649 km | > 1 km |
| a perfect model's error is below the bound | — | true at all 401 samples | — |
| the wrong-way-round bound `D` is violated | — | 401 of 401 samples | control must fire |
| the correct bound is not violated on that data | — | 0 of 401 samples | — |

## 1. SGP4 reproduced against its own published test vectors

Source: `SGP4-VER.TLE` and `tcppver.out`, **package data inside the installed
`sgp4` 2.27 distribution**. They are the verification set published with
Vallado, Crawford, Hujsak and Kelso (2006), "Revisiting Spacetrack Report #3",
AIAA 2006-6753, and the accompanying `testcpp.cpp` program; the distribution
carries them unchanged. The parser is `orbitdrift.vectors`, this package's own;
`sgp4.tests` is **not** imported, because it regenerates `tcppver.out` as text
and compares strings, so a bug shared between its generator and its formatter
would be invisible to it.

| Check | Result | Tolerance |
|---|---|---|
| cases parsed | 33 | 33 |
| published data rows accounted for | 666 compared + 1 skipped = 667 | 667 |
| worst position disagreement | **1.154503e-07 km** | < 1e-6 km |
| worst velocity disagreement | **4.996954e-10 km/s** | < 1e-8 km/s |
| median per-case worst position disagreement | 2.728484e-12 km | — |
| rows skipped on a propagator error code | 1, SGP4 code 3, named | counted, not dropped |
| negative control: inclination edited 34.2682 -> 44.2682 deg | 1275.386193 km | must exceed 100 km |

The file prints position to 8 decimal places of a kilometre and velocity to 9
of a km/s, so no comparison can be tighter than about 5e-9 km and 5e-10 km/s
from the file's own rounding. The measured disagreement is a small multiple of
that.

The skipped row is not a defect: `tcppver.out` handles a satellite whose
propagation fails at zero minutes by repeating the previous satellite's last
data line, a quirk of the reference program, and comparing that row would
report a spurious disagreement of thousands of kilometres.

**A parser defect this check found.** `SGP4-VER.TLE` lists catalogue 20413
twice with different propagation schedules. Keying the expected blocks by
catalogue number merged them and compared each element set against the union,
inflating the row count from 667 to 763 and silently comparing one satellite
against another test case's rows. The blocks are matched positionally, and
`tests/test_vectors.py` pins the row count.

## 2. Conservation of energy and angular momentum

`validate_conservation.py`. The question is not whether the invariants are
conserved but **which residual is the integrator and which is the physics**,
and the discrimination is made without any reference solution: halving the step
must shrink anything numerical and leave anything physical alone.

With only the central term, the specific energy, `|h|` and `h_z` are all exact
constants. With the zonal field on, the potential is still time-independent and
still axially symmetric about z, so energy and `h_z` remain exact constants —
but `|h|` is **not**: `J2` torques the orbit about any equatorial axis, so the
angular momentum vector precesses about z at constant `h_z` while its magnitude
oscillates.

The declared tolerance is step-dependent because the residual is:
`6e-9 * (360/steps_per_rev)^5`, which is the measured `h^5` scaling with a
factor-of-three margin over the worst value at 360 steps per revolution.

Two-body plus `J2`-`J4`, 6 revolutions, worst over three orbits
(500 km `e`=0.001 `i`=51.6; 350 km `e`=0.02 `i`=98.0; 1400 km `e`=0.005 `i`=20.0):

| steps/rev | declared tolerance | worst energy | worst `h_z` | `|h|` (500 km case) |
|---|---|---|---|---|
| 180 | 5.5556e-07 | 5.4961e-08 | 2.7443e-08 | 5.1172e-04 |
| 360 | 6.0000e-09 | 1.7174e-09 | 8.5745e-10 | 5.1225e-04 |
| 720 | 1.8750e-10 | 5.5983e-11 | 2.7922e-11 | 5.1227e-04 |
| 1440 | 5.8594e-12 | 1.9809e-12 | 9.8930e-13 | 5.1227e-04 |

| Discrimination | Result | What it means |
|---|---|---|
| energy residual ratio per halving, three cases | 32.00/31.99/31.92, 32.01/30.34/28.26, 32.00/32.00/32.02 | `h^5`: the integrator |
| `|h|` residual relative spread across four step sizes | 5.007e-05, 1.787e-06, 3.401e-04 | step-invariant: the `J2` torque |
| energy with drag on | every sampled difference negative, largest -6.121e-05; total -7.072135e-03 km^2/s^2 | dissipative, as the sign of `a . v` requires |

**Negative control, anchored on the true value.** A check that accepts a
quantity within `b` of a reference is only *guaranteed* to fire on an offset
strictly greater than `2b`, because the unperturbed quantity could already sit
anywhere in `[-b, b]`. The table below separates what is guaranteed from what
happened:

| offset | residual | fires | guaranteed to fire |
|---|---|---|---|
| 0.5 b | 1.8693e-09 | no | no |
| 1.0 b | 4.8693e-09 | no | no |
| 2.0 b | 1.0869e-08 | yes | no |
| 3.0 b | 1.6869e-08 | yes | yes |

Unperturbed residual 1.1307e-09, which is 0.1885 b. The 0.5 b row is the point
of the table: a negative control built on an offset of half the tolerance would
have silently passed a check that was not testing anything.

## 3. Integrator convergence under step refinement

`validate_step_refinement.py`. Two independent routes to the same order.

**Richardson**, which needs no reference solution at all:
`p = log2((x_h - x_h/2)/(x_h/2 - x_h/4))` from the final position vector.

| case | 90/180/360 | 180/360/720 | 360/720/1440 |
|---|---|---|---|
| 500 km `e`=0.001 | 4.6990 | 4.5386 | 4.3701 |
| 350 km `e`=0.02 | 4.7123 | 4.5542 | 4.3849 |
| 1400 km `e`=0.005 | 4.7018 | 4.5418 | 4.3731 |

**Against the closed form**, error ratio per halving (16 expected for a
fourth-order method asymptotically):

| case | ratios |
|---|---|
| 500 km `e`=0.001 | 25.850, 23.111, 20.571, 18.666 |
| 350 km `e`=0.02 | 26.093, 23.363, 20.781, 18.801 |
| 1400 km `e`=0.005 | 25.900, 23.162, 20.614, 18.698 |

The measured order approaches 4 from above and the ratio falls toward 16 as the
step shrinks. It is reported as measured rather than asserted to be exactly 4,
because it is not exactly 4.

**With drag on**, the published atmosphere is 28 independent exponential fits
and its band joins are discontinuous, which can cost a fixed-step method its
order. Measured jumps are at most 0.001362 (§5), and the orders with drag on —
4.6978, 4.5370, 4.3688 — are indistinguishable from the drag-free orders above.
The mechanism is real and not material at this jump size.

## 4. The truncation floor against the signal it has to measure

Every divergence curve in this repository was produced at 360 steps per
revolution. At that step, over 7 days, the integrator's own error against the
closed-form two-body solution is:

| case | truncation floor |
|---|---|
| 500 km `e`=0.001 `i`=51.6 | 1.135041e-01 km |
| 350 km `e`=0.02 `i`=98.0 | 1.168030e-01 km |
| 1400 km `e`=0.005 `i`=20.0 | 9.004421e-02 km |

Against the divergence of the committed population at 6.79 days, using the
worst of the three floors (1.168030e-01 km):

| statistic | value |
|---|---|
| smallest divergence over 192 objects | 0.6315 km |
| largest divergence over 192 objects | 3540.2509 km |
| median floor / divergence | 2.798e-03 |
| 75th percentile | 7.699e-03 |
| 90th percentile | 1.593e-02 |
| 95th percentile | 2.336e-02 |
| 99th percentile | 1.041e-01 |
| worst | **1.850e-01** |
| objects where the floor exceeds 1 per cent of their divergence | 34 of 192 |
| objects where the floor exceeds 10 per cent of their divergence | 3 of 192 |

### 4.4 A claim that is only partly satisfied

The requirement was that the truncation floor be below 1 per cent of every
measured divergence. **It is not.** It is below 1 per cent for three quarters of
the population and reaches 18.5 per cent for the least divergent object. For
the handful of objects whose seven-day divergence is under about a kilometre —
high altitude, tiny `B*`, almost nothing for the two models to disagree about —
the integrator's own error is a double-digit percentage of the number reported,
and those rows should be read as *divergence below about a kilometre* rather
than as the value printed. The three objects concerned are named in
`validate_step_refinement_output.txt` and the limitation is in `README.md`.

## 5. The declared atmosphere model

`validate_atmosphere.py`. Vallado 2013, *Fundamentals of Astrodynamics and
Applications*, 4th ed., Table 8-4, after CIRA-72 and the US Standard Atmosphere
1976. 28 bands, each an independent exponential fit.

| Check | Result |
|---|---|
| every base altitude returns its tabulated density | worst relative departure 0.000e+00 |
| density falls with height over 0 to 1500 km | 0 of 150001 samples rise |
| largest band-join relative jump | 0.001362 |
| drag at 1000 km at `C_D A/m` = 1e-2 m^2/kg | 7.397e-13 km/s^2 |

The model is **static**: no solar index, no diurnal bulge, no geomagnetic term.
Published solar-cycle variation of thermospheric density at 400 km is about an
order of magnitude between solar minimum and maximum, and this model sits at one
unspecified point in that range. Every drag-driven divergence number in this
repository inherits that factor.

## 6. The independent dynamical model's other terms

### 6.1 Zonal gravity against the gradient of its own potential

`tests/test_forces.py`. The acceleration is coded from the differentiated
Legendre form and the potential from the undifferentiated one, so these are two
different expressions for the same quantity. The comparison is against a
central finite difference of the potential, with the tolerance derived from the
method's roundoff floor `eps |V| / h` rather than tuned. With the two-body term
removed from both sides — so that `J2` is the whole signal rather than 1e-3 of
it — the agreement is 5e-5 of the perturbing acceleration's norm.

### 6.2 Luni-solar third body against a cancellation-free expansion

`validate_third_body.py`. The implemented form `mu_b (d/|d|^3 - r_b/|r_b|^3)`
cancels two terms of order `mu_b/|r_b|^2` to produce a result of order
`2 mu_b |r| / |r_b|^3`. The check is against the leading term of the series
expansion, which has no such cancellation in it, and the agreement must be of
the order of the neglected term, `eps = |r|/|r_b|`:

| body | `|r|` km | measured relative difference | `eps` | ratio |
|---|---|---|---|---|
| Sun | 6800 | 6.934e-05 | 4.623e-05 | 1.4999 |
| Sun | 42164 | 4.299e-04 | 2.866e-04 | 1.5000 |
| Moon | 6800 | 2.589e-02 | 1.690e-02 | 1.5323 |
| Moon | 42164 | 1.813e-01 | 1.048e-01 | 1.7309 |

The check is two-sided: a difference much larger than `eps` would mean the
implemented form loses more precision than the cancellation accounts for, and a
difference much smaller would mean the series is not actually being truncated,
which would make this a comparison of two identical expressions. At the
geocentre the two terms cancel exactly, measured at 0.000e+00 km/s^2. The
negative control — the series with the quadrupole sign flipped — disagrees by
1.031527 relative.

### 6.3 The `B*` inversion, and how much of a curve rests on it

`validate_bstar_sensitivity.py`. Drag needs `C_D A/m` and a two-line element set
carries only `B*`. From `B* = (1/2) C_D (A/m) rho_0` with
`rho_0 = 0.157 kg/(m^2 ER)`, the product drag actually needs is
`C_D A/m = 2 B* / rho_0 = 12.7389 B*` m^2/kg, which requires **no assumed drag
coefficient** because `C_D` cancels. What remains an assumption is `rho_0` and
the premise that an operational `B*` is a physical ballistic parameter.

Measured over 8 objects, a factor of two in `C_D A/m` in either direction moves
the reported 2-day divergence by this fraction of itself:

| statistic | value |
|---|---|
| median | 0.0002 |
| worst | 0.5909 |
| correlation of log sensitivity with the log drag parameter | **+0.9626** |
| heaviest drag parameter (-13.662) | 0.395637 |
| lightest drag parameter (-19.844) | 0.000001 |

The sensitivity is not a function of altitude: the population draws `B*`
independently of altitude, so a 500 km object with `B*` = 1e-6 has almost no
drag and almost no sensitivity. The correlation above is against
`log10(C_D A/m * rho_perigee)`, the combination that appears in the drag
acceleration.

## 7. Frames and transforms

`validate_frames.py`. `astropy` is not installed and nothing here is
cross-checked against it.

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| GMST at J2000.0 | published 280.460618370 deg | 280.460618375 deg, difference 5.000e-09 | < 1e-6 deg |
| sidereal excess over one solar day | published 0.9856473 deg | difference 6.629e-08 deg | < 1e-5 deg |
| rotation orthogonality and determinant | — | 2.220e-16 | < 1e-15 |
| TEME/ECEF round trip | — | 3.638e-12 km | < 1e-9 km |
| geodetic height against an independent fixed-point solve | — | 0.000 mm at 0 km, 1.219 mm at 400 km, 8.771 mm at 1200 km, 15.488 mm at 1700 km, 35.773 mm at 3000 km | < 20 mm up to 1700 km |
| RIC basis orthonormal, right-handed, isometric | — | 1.164e-15 | < 1e-12 |
| solar distance at J2000.0 | Earth near perihelion in early January | 0.983308 AU | 0.980 to 0.986 AU |
| lunar distance over 2000 days | published perigee-apogee 356400 to 406700 km | 357350 to 406018 km | inside 352000 to 410000 km |
| lunar daily motion | published 13.176 deg/day | 11.9039 deg/day | 11 to 15 deg/day |
| in-track is the largest component | physical prediction | 0.996466 of 5376 rows | > 0.99 |
| components add in quadrature to the total | algebraic identity | 6.440e-16 | < 1e-10 |
| negative control: radial and in-track axes swapped | — | 0.000000 | control must fire |

Bowring's geodetic formula is a single pass built for points near the surface,
so its departure grows with height. It is reported as a function of height
rather than quoted from the paper. The only consumer is the atmosphere model,
whose density is uncertain by a factor of several at these altitudes.

## 8. Baseline 1: SGP4 self-consistency

`validate_self_consistency.py`. Both sides are SGP4, so any error common to the
theory — the atmosphere model, the truncation of the Brouwer-Lyddane expansion,
the absence of solar radiation pressure — cancels out of the comparison. It
measures the part of the error that came from the *fit*, not the part that came
from the *model*.

**The one real pair in this environment.** Catalogue 5, Vanguard 1, at epochs
2000-179 and 2020-287. Both element sets are transcribed into the script and
their checksums are recomputed and verified there. The epoch gap is 7412.4184
days, 20.29 years, and the relative change in mean motion over it is 2.263455e-03.
The resulting separation ranges from 4315.5 to 16709.2 km over 248 reported
ages. **That magnitude is meaningless as a staleness figure and is not quoted
as one anywhere.** It is in the repository because it is the only real
same-object element-set pair available and running the code path on real data is
worth more than not running it.

**Synthetic pairs with geometrically derivable separations.** Two element sets
are built from explicitly chosen mean elements. Advancing the later set's angles
by the Kozai mean motion does **not** make the two describe the same orbit:
SGP4's secular rates for the mean anomaly, the argument of perigee and the right
ascension include the `J2` corrections. A first version did that and the
identical pair came out 6882 km apart, the orbit radius. `Satrec` exposes
`mdot`, `argpdot` and `nodedot`, and advancing by those gives an exact match.

| Experiment | Predicted | Measured | Ratio |
|---|---|---|---|
| mean-anomaly offset `dM` = 1e-5 | chord 2 a sin(dM/2) = 0.068781 km | 0.068778 km | 0.99996 |
| `dM` = 1e-2 | 68.781063 km | 68.778550 km | 0.99996 |
| mean-motion offset, growth rate | `a (mdot_late - mdot_early)` | fitted slope | 0.99655 at all three offsets, `r^2` = 0.999980 |
| identical pair | 0 | **7.691751e-10 km** | — |
| separation constant in time for a pure `dM` | relative spread 0 | 0.0000 over 12 ages | — |

## 9. The analytic baseline, validated before anything is learned

`validate_growth_baseline.py`, which runs before `validate_predictors.py`.

The physics predicts linear-in-time growth where drag is negligible (a constant
mean-motion error gives an along-track error `-(3/2) n (da/a) t`) and
quadratic-in-time growth where drag dominates (a constant error in the *rate* of
change of semi-major axis integrates to `t^2`).

Per-object free-exponent fits over the committed population:

| perigee band | n | mean p | sd p | median log-residual spread | best-form counts |
|---|---|---|---|---|---|
| 300-450 km | 42 | 1.2683 | 0.3914 | 0.21484 | drag_exponential 23, power 18, quadratic 1 |
| 450-600 km | 32 | 1.0249 | 0.3590 | 0.10374 | power 23, drag_exponential 8, quadratic 1 |
| 600-800 km | 31 | 1.1197 | 0.3383 | 0.09587 | power 19, drag_exponential 12 |
| 800-1100 km | 41 | 1.0175 | 0.2020 | 0.05525 | drag_exponential 12, power 29 |
| 1100-1700 km | 46 | **0.9685** | **0.0628** | 0.04165 | power 35, drag_exponential 11 |

| Check | Result | Tolerance |
|---|---|---|
| exponent above 1100 km equals the predicted linear value 1 | 0.9685 +/- 0.0628 | within 0.1 of 1 |
| spread above 1100 km is small, so it is one regime not an average of two | sd 0.0628 | < 0.12 |
| exponent below 450 km exceeds the one above 1100 km | 1.2683 against 0.9685 | difference > 0.1 |
| correlation of p with log perigee altitude | -0.2914 | < -0.15 |
| correlation of p with the log drag parameter | **+0.5090** | > 0.15 |
| median log-residual spread of the free power law | 0.07469 | < 0.15 |
| negative control: shuffled drift column | 1.14177 against 0.07469 | > 5x worse |

**This is the specification's predicted outcome and it is confirmed.** Above
1100 km the growth is linear in time to within the fitted scatter, which is
exactly what a constant mean-motion error gives.

Form-by-form fit quality over all 192 objects:

| form | median | mean | 90th pct | wins |
|---|---|---|---|---|
| linear | 0.1329 | 0.3344 | 1.0294 | 0 |
| quadratic | 1.1730 | 1.1276 | 1.3970 | 2 |
| power | **0.0747** | 0.2136 | 0.7515 | 124 |
| drag_exponential | 0.0924 | 0.2267 | 0.6807 | 66 |

## 10. The analytic baseline against the learned predictor

`validate_predictors.py`. Nominal level 0.90. Split by **object**: 134 training
and 58 test objects, disjoint. Splitting by row would put an object's early ages
in training and its late ages in test, where the features are identical and only
the age column differs.

How the winner is decided, and why it is not the narrowest interval: narrowing
an interval always improves its score until it stops covering, so width is only
comparable among predictors whose measured coverage is consistent with the level
they claim. Three rankings are reported.

### 10.1 Task 1 — cold prediction from elements and epoch age

1624 held-out rows over 58 objects.

| predictor | kind | coverage | Clopper-Pearson 95 % | median log width | interval score km | median \|log error\| |
|---|---|---|---|---|---|---|
| rule_of_thumb | analytic | 0.3017 | [0.2795, 0.3247] | 2.1972 | 1352.77 | 1.6282 |
| analytic_linear | analytic | 0.8799 | [0.8631, 0.8953] | 3.5433 | 520.85 | 0.7843 |
| analytic_linear_conformal | analytic | 0.8744 | [0.8573, 0.8901] | 3.5384 | 565.42 | 0.7550 |
| analytic_quadratic | analytic | 0.8898 | [0.8735, 0.9046] | 4.9060 | 621.35 | 0.8597 |
| analytic_quadratic_conformal | analytic | 0.8645 | [0.8469, 0.8808] | 4.7010 | 644.60 | 0.9381 |
| analytic_power | analytic | 0.8793 | [0.8625, 0.8948] | 3.5300 | 508.44 | 0.7877 |
| analytic_power_conformal | analytic | 0.8695 | [0.8521, 0.8855] | 3.5120 | 556.49 | 0.7410 |
| analytic_power_altitude | analytic | 0.8861 | [0.8696, 0.9011] | 3.5129 | **477.60** | 0.7930 |
| analytic_power_altitude_conformal | analytic | 0.8768 | [0.8599, 0.8924] | 3.5209 | 531.88 | 0.7535 |
| analytic_drag_exponential | analytic | 0.8725 | [0.8553, 0.8884] | 3.5079 | 493.83 | 0.8044 |
| analytic_drag_exponential_conformal | analytic | 0.8688 | [0.8514, 0.8849] | 3.4891 | 543.85 | 0.7512 |
| learned_gbr_quantile | learned | 0.6041 | [0.5798, 0.6280] | 2.0559 | 349.65 | **0.5942** |
| learned_gbr_conformal | learned | 0.8849 | [0.8683, 0.9000] | 3.0685 | **270.64** | 0.6014 |

Verdict as measured:

- Lowest interval score overall: `learned_gbr_conformal`, 270.64 km, coverage 0.8849.
- Best point prediction: `learned_gbr_quantile`, median absolute log error 0.5942, against the best analytic `analytic_power_conformal` at 0.7410. **The learned model wins the point prediction on this task.**
- Calibrated predictors (Clopper-Pearson interval contains 0.90): `analytic_quadratic` and `analytic_power_altitude`. Best of those: `analytic_power_altitude`, 477.60 km.
- **No learned predictor is calibrated on this task.**
- **Every predictor under-covers at 0.90.** For a staleness decision that means every horizon in this table is late, and the table has no operationally usable row.

### 10.2 Task 2 — extrapolation beyond a two-day fitting horizon

This is the realistic task. The operator holds the element set and can run the
divergence computation over the first two days in seconds. 1536 rows inside the
horizon, 3840 beyond it, 1160 measured on 58 held-out objects. The analytic
forms are fitted to each object's own inside-horizon curve; the learned model is
trained on the training objects' beyond-horizon rows and is additionally given
that same fit's amplitude, exponent and residual spread as three features, so it
is not handicapped by being denied what the baseline has.

Best-fitting form inside the horizon, per object: power 146, drag_exponential 36,
linear 10.

| predictor | kind | coverage | Clopper-Pearson 95 % | median log width | interval score km | median \|log error\| |
|---|---|---|---|---|---|---|
| own_curve_linear | analytic | 0.6155 | [0.5868, 0.6436] | 0.4874 | 570.29 | 0.1554 |
| own_curve_linear_conformal | analytic | **0.8991** | [0.8804, 0.9159] | 3.3315 | 554.32 | 0.1554 |
| own_curve_quadratic | analytic | 0.7690 | [0.7436, 0.7929] | 4.4780 | 3455.24 | 1.8389 |
| own_curve_quadratic_conformal | analytic | 0.8707 | [0.8500, 0.8895] | 4.8853 | 4854.34 | 1.8389 |
| own_curve_power | analytic | 0.3069 | [0.2804, 0.3343] | 0.2282 | 280.55 | 0.1655 |
| own_curve_power_conformal | analytic | **0.9474** | [0.9330, 0.9595] | 2.8729 | **281.53** | 0.1655 |
| own_curve_drag_exponential | analytic | 0.6621 | [0.6340, 0.6893] | 0.4012 | 874131.70 | **0.1299** |
| own_curve_drag_exponential_conformal | analytic | 0.8776 | [0.8573, 0.8959] | 3.4852 | 651578.31 | 0.1299 |
| learned_with_curve_features | learned | 0.6207 | [0.5921, 0.6487] | 0.6262 | 200.91 | 0.1391 |
| learned_with_curve_features_conformal | learned | 0.8681 | [0.8473, 0.8871] | 1.1147 | 164.44 | 0.1552 |
| learned_features_only | learned | 0.5974 | [0.5685, 0.6258] | 1.8424 | 446.79 | 0.5439 |
| learned_features_only_conformal | learned | 0.8750 | [0.8546, 0.8935] | 3.0153 | 310.37 | 0.6488 |

Verdict as measured:

- **Best point prediction is analytic**: `own_curve_drag_exponential` at a median absolute log error of 0.1299, against the best learned `learned_with_curve_features` at 0.1391. The analytic form beats the learned model by 7 per cent on the point prediction.
- **The only calibrated predictor is analytic**: `own_curve_linear_conformal`, coverage 0.8991 [0.8804, 0.9159], interval score 554.32 km. No learned interval's confidence interval contains 0.90.
- **The only predictor that does not under-cover is analytic**: `own_curve_power_conformal`, coverage 0.9474, interval score 281.53 km. **No learned predictor reaches its nominal coverage**, so a staleness horizon taken from one would be late.
- The lowest interval score overall is `learned_with_curve_features_conformal` at 164.44 km — bought with a coverage of 0.8681 against a claimed 0.90.

### 10.3 What the object's own curve is worth, separately from the learning

Two learned models, identical except that one sees the object's own
inside-horizon fit:

| | median \|log error\| | interval score km |
|---|---|---|
| with the curve | 0.1391 | 200.91 |
| without the curve | 0.5439 | 446.79 |
| ratio | **3.9095** | 2.2238 |

The information is in the curve, not in the learning. A learned model denied the
curve is four times worse at the point prediction than one given it, and the
analytic extrapolation of that same curve beats both.

### 10.4 The central measured fact about nominal intervals

An analytic fit's own in-sample residual spread is what a practitioner would
naturally use as an interval. Measured on held-out objects beyond the horizon:

| form | nominal interval coverage | conformal coverage |
|---|---|---|
| linear | 0.6155 | 0.8991 |
| quadratic | 0.7690 | 0.8707 |
| power | **0.3069** | 0.9474 |
| drag_exponential | 0.6621 | 0.8776 |

A fit's in-sample residuals say nothing about how it behaves outside the range
it was fitted on. The power-law form, which fits best inside the horizon, has
the *worst* nominal coverage outside it: 0.3069 against a claimed 0.90. This is
the single most important measured fact in the product and the reason it
measures coverage instead of quoting it.

## 11. The coverage table, including where it is miscalibrated

`validate_coverage_table.py`. 65 cells across four tables are inconsistent with
the nominal level at 95 per cent confidence.

**The rows are grouped.** Twenty-eight epoch ages of each object share its
features and its physics, so the effective sample size of a cell is closer to
the number of objects in it than to the number of rows, and the Clopper-Pearson
interval computed on the row count is **too narrow**. The object-level coverage
in `validate_predictors_output.txt` is the conservative counterpart; neither is
presented as the answer.

Cold prediction, coverage by perigee altitude band:

| predictor | n per cell | <400 km | 400-600 | 600-800 | 800-1100 | >1100 |
|---|---|---|---|---|---|---|
| | | 364 | 336 | 168 | 308 | 448 |
| rule_of_thumb | | 0.2225 under | 0.3333 under | 0.2738 under | 0.1851 under | 0.4330 under |
| analytic_linear | | 0.7665 under | 0.8988 ok | 0.8095 under | 0.9416 over | 0.9420 over |
| analytic_power_altitude | | 0.7912 under | 0.8929 ok | 0.8155 under | 0.9416 over | 0.9464 over |
| analytic_power_altitude_conformal | | 0.7802 under | 0.8929 ok | 0.8095 under | 0.9221 ok | 0.9375 over |
| learned_gbr_quantile | | 0.5110 under | 0.6369 under | 0.4226 under | 0.8279 under | 0.5692 under |
| learned_gbr_conformal | | 0.9011 ok | 0.9494 over | 0.6607 under | 0.8896 ok | 0.9040 ok |

Cells inconsistent with 0.90, out of 9 (5 altitude bands plus 4 age bands):

| predictor | cold task | extrapolation task |
|---|---|---|
| rule_of_thumb | 9 of 9 (9 under) | — |
| analytic_linear | 6 of 9 (3 under) | — |
| analytic_power_altitude | 6 of 9 (3 under) | — |
| analytic_power_altitude_conformal | 4 of 9 (3 under) | — |
| learned_gbr_quantile | 9 of 9 (9 under) | — |
| learned_gbr_conformal | 2 of 9 (1 under) | — |
| own_curve_power | — | 7 of 7 (7 under) |
| own_curve_power_conformal | — | 5 of 7 (**0 under**) |
| own_curve_linear_conformal | — | 4 of 7 (2 under) |
| own_curve_drag_exponential_conformal | — | 4 of 7 (2 under) |
| learned_with_curve_features | — | 7 of 7 (7 under) |
| learned_with_curve_features_conformal | — | 2 of 7 (2 under) |

**The conditional miscalibration conformal prediction does not promise.** Split
conformal prediction gives marginal coverage at or above the nominal level for
exchangeable data. It promises nothing about coverage conditional on a feature,
and these tables show it failing conditionally while delivering marginally:

| task | predictor | marginal | altitude cells span | spread |
|---|---|---|---|---|
| cold | analytic_power_altitude_conformal | 0.8768 | 0.7802 to 0.9375 | 0.1573 |
| extrapolation | own_curve_power_conformal | 0.9474 | 0.8909 to 0.9769 | 0.0860 |

That is not a defect in the method; it is the method's stated scope. A product
reporting only the marginal number would be hiding it.

`own_curve_power_conformal` is the row to read for an operational decision: 5 of
its 7 cells are inconsistent with 0.90, but **none of them under-covers**. Its
intervals are wider than nominal in most strata, which for a staleness horizon
is a cost rather than a defect; an under-covering interval gives a horizon that
is late, which is.

## 12. Determinism and reproducibility

`validate_reproducibility.py`.

| Claim | Result |
|---|---|
| the committed population is reproduced byte for byte from its seed | sha256 `c1a862cbc73c58bc7b2c461f34aea5b58babba375888cf12cc18fbc95abfa90a` matches |
| a different seed gives a different population | confirmed |
| two runs of the same computation agree bit for bit | worst difference 0.000e+00 |
| a pinned-step subset reproduces the committed rows | worst relative 8.305428e-09 |
| an unpinned step changes the rows measurably | worst relative 1.141518e-02 |
| the cold-prediction benchmark is bit-reproducible | 0 mismatched rows of 13 |
| the extrapolation benchmark is bit-reproducible | 0 mismatched rows of 12 |
| a different split seed changes the measured coverage | 13 of 13 rows change |

**Why the subset comparison is not bit-exact**, stated rather than absorbed into
a loose tolerance: the luni-solar ephemeris is tabulated over the propagation
span, so a one-day run evaluates the trigonometric series on 1442 epochs where
the seven-day committed run used 10082. numpy's vectorised transcendental loops
are not required to give bit-identical results for arrays of different length —
the tail of a SIMD loop is handled differently from its body — and a few
thousand Runge-Kutta steps amplify that last-bit difference to 8.3e-09 relative.

**Why the integration step is an input.** An earlier version derived it from the
shortest period in each *batch*, which made the result depend on which other
objects happened to share an epoch: recomputing three objects on their own then
differed from the committed table in the fourth significant figure. That is the
truncation floor showing through a reproducibility claim. The step is now
`reference_period_s / steps_per_rev` with the reference recorded in the table's
own `detail` string.

## 13. The covariance prohibition

`validate_guard.py`. 23 refusals fire and 15 legitimate calls are accepted. Both
counts matter: a guard that refused everything would make the first large and
the second zero.

| Guard | Fires on | Passes on |
|---|---|---|
| `check_purpose` | 13 purposes naming a covariance, a conjunction, a collision, `Pc`, a CDM, a close approach, a miss distance, screening or an encounter | 8 legitimate purposes including `spacecraft operations` and `pc12345 is an identifier, not the word` |
| `audit_mapping_for_covariance_keys` | 8 key spellings, at any nesting depth, naming the path | 7 ordinary keys including `recovery`, which contains the letters of `cov` but not as a word |
| `DriftReport.as_covariance` | always | — |
| `StalenessVerdict.as_covariance` | always | — |
| `DriftReport.to_dict` self-audit | a covariance key inserted into its own payload | the payload's 10 real keys |

Every refusal names **P071 ConjProb**, states that a conjunction assessment
needs a covariance from orbit determination — a filter with measurement
residuals, or a published covariance in a conjunction data message — and states
that this package has no measurements.

## 14. Suite hygiene

### 14.1 The `pytest.approx` audit

`validate_approx_audit.py`. `pytest.approx(expected, rel=r)` carries a **default
absolute tolerance of 1e-12** and passes when *either* bound is satisfied, so
for a quantity of order 1e-5 every relative tolerance tighter than 1e-7 is
silently replaced by the absolute one. Conservation residuals, of order 1e-9
here, are exactly where that bites.

| Measurement | Result |
|---|---|
| `pytest.approx` calls in the suite, found by AST | 67 |
| with `rel=` and an explicit `abs=` | 67 |
| with `rel=` and **no** `abs=` (the default would apply) | **0** |
| assertions relying on the 1e-12 default | **0** |
| assertions whose explicit absolute tolerance is load-bearing (forcing `abs=0.0` everywhere) | 17 |
| assertions that would change verdict if the explicit `abs=` were removed | 10 under hypothesis 6.170.9, 11 under 6.140.3 |

The first number is what the audit asks for and it is zero under both
interpreters. Experiment (c) differs by one assertion between the two
hypothesis versions, because a property test draws different examples and one
of them lands where the default 1e-12 tolerance would or would not have
sufficed; the deterministic experiment (b) gives 17 under both. The last two are why
that is worth something: there are assertions in this suite whose absolute
tolerance matters, so a suite that had relied on the default would have had that
many assertions testing `|x - y| < 1e-12` instead of what they say they test.

A regular expression over the source also matched the occurrences of
`pytest.approx(expected, rel=r)` inside the docstrings explaining the problem,
and the first version of the scan reported its own docstring as an offender. The
scan parses instead.

### 14.2 Working-directory independence

`validate_cwd_independence.py`. 26 scripts run from `/`, a temporary directory
outside the repository, the product root, and the script's own directory: 104
invocations, **0 non-zero exits**.

This is the check that four products in one batch of this mission failed at the
release gate after passing every check in their own build, because the build
agent, its self-report and the coordinator's re-measurement had all run the
scripts from the product root — three independent verifications sharing an
assumption rather than a method.

### 14.3 Defects this validation found in this product

| Defect | How it was found | Status |
|---|---|---|
| The drag unit conversion divided by 1000 where it had to multiply, a factor of 1e6, so the drag term was invisible. | A **negative control**: switching the drag term off and requiring the divergence curve to change. It changed by 2.09e-07 relative, which is to say not at all. The unit test that recomputed the same expression passed throughout. | fixed; pinned by an absolute magnitude from an independent SI evaluation in `tests/test_regressions.py`, and by an order-of-magnitude check against a figure from the literature |
| The published-vector parser keyed expected blocks by catalogue number, merging the two cases that share catalogue 20413. | The row count: 763 against the file's 667. | fixed; blocks matched positionally, row count pinned |
| The integration step was derived from the shortest period in each batch, so a subset recomputation did not reproduce the committed rows. | A subset-reproduction test, which failed in the fourth significant figure. | fixed; the step is now a declared input |
| The synthetic self-consistency pair advanced the later element set's angles by the Kozai mean motion, which is not SGP4's secular rate. | The identical-pair negative control came out 6882 km apart, the orbit radius. | fixed using `Satrec.mdot`, `argpdot` and `nodedot`; identical pair now 7.69e-10 km |
| The log-linear predictor extrapolates catastrophically off the collinear feature manifold: a hand-assembled 1300 km feature vector drew a prediction of 2.5e8 km against the 17.7 km measured for a real object. | Building an example from hand-assembled feature vectors instead of real ones. | diagnosed rather than fixed: `AnalyticGrowthPredictor.leverage` measures the distance from the training manifold (training maximum 7.04, held-out maximum 8.45, the inconsistent vector 259), and the documentation says to build features with `features_from_satrec` from an actual element set |
| The geodetic height test asserted a 2 mm tolerance that Bowring's single-pass formula cannot deliver at orbital altitude. | The test failed at 7.7 mm. | the claim was corrected to the measured height dependence rather than the tolerance loosened silently |
| A drag unit test used the density at a round 400 km while the function computes 399.998 km, because the state is built from the WGS72 radius and the height from the WGS84 ellipsoid. | The test failed by 3.4e-5 relative. | the two-metre difference between the reference radii is now stated in the test |
| The element round trip's inclination tolerance was a flat 1e-9 radian, which the recovery cannot deliver for a nearly equatorial orbit: inclination comes back as `arccos(h_z/|h|)` and `d(arccos x)/dx = -1/sin(i)`, so a few machine epsilons of relative error in the ratio become `eps/sin(i)` radian of absolute error. | **Hypothesis**, at `a = 6600` km, `e = 0`, `i = 5.960464477539063e-08` radian with every angle zero, where the round trip recovers 6.143906154658886e-08, a departure of 1.83e-09. No hand-written case used an inclination below a degree. | the tolerance is now derived from the conditioning, the limitation is stated in `rv_to_elements`, and the counterexample is pinned with literal inputs in `tests/test_regressions.py` |

### 14.4 What Hypothesis found that the hand-written tests did not

Hypothesis is used in `tests/test_frames.py`, `tests/test_kepler.py`,
`tests/test_forces.py`, `tests/test_growth.py`, `tests/test_calibration.py` and
`tests/test_divergence.py`, on properties that are algebraic identities or
independently derived bounds rather than restatements of the implementation.

- `test_element_round_trip_closes` generates inclinations down to 6e-08 radian.
  It falsified a flat 1e-9 radian tolerance on the recovered inclination, which
  the recovery cannot deliver for a nearly equatorial orbit because
  `arccos(h_z/|h|)` has derivative `-1/sin(i)`. The tolerance is now derived
  from that conditioning and the counterexample is pinned. No hand-written case
  in this suite used an inclination below one degree, and none would have.
- `test_fit_never_returns_a_non_finite_parameter_or_a_fabricated_zero` generates
  age and drift lists of mismatched and degenerate shapes. It drove the
  rank-deficiency refusal in `fit_power_law`: `numpy.linalg.lstsq` on a design
  matrix whose age column is constant returns the least-norm solution, an
  exponent of exactly 0.0 with a standard error of exactly 0.0, which is a
  confident-looking answer to a question the data cannot answer. The hand-written
  tests used three or four well-formed ages and never produced a repeated one.
  The counterexample is pinned with literal inputs in
  `tests/test_regressions.py`, because `.hypothesis/` is gitignored and a
  falsifying example found by one run is lost on a fresh clone.
- `test_clopper_pearson_always_brackets_the_proportion` explores the boundary
  counts 0 and `n` at every confidence level, where a naive implementation
  returns a NaN or an inverted interval. The hand-written known-answer test
  covered 8 of 10 and the two boundaries at one confidence level.
- `test_noiseless_power_law_is_always_recovered` explores amplitudes over seven
  decades and exponents from 0.2 to 3.0, against the two or three a hand-written
  test picks.

The honest accounting: Hypothesis found **two** things the hand-written tests
did not. The rank-deficient fit is a defect in the library — a pseudo-inverse
returning a confident-looking exponent of exactly zero — and the inclination
tolerance is a defect in the suite, an assertion asking for an accuracy the
formula cannot deliver in a regime the hand-written cases never visited. Both
are now pinned as explicit regressions with literal inputs. The other properties
Hypothesis covers were already correct, and its value there is the breadth of
the search rather than a discovery.

## 15. Reproducing every number in this document

From the product root, with `PYTHONPATH=src` and `MPLBACKEND=Agg`:

```bash
python validation/regenerate_dataset.py          # about 3 minutes; writes fixtures/
python validation/validate_sgp4_vectors.py
python validation/validate_frames.py
python validation/validate_atmosphere.py
python validation/validate_third_body.py
python validation/validate_kepler_reference.py
python validation/validate_conservation.py
python validation/validate_step_refinement.py
python validation/validate_bstar_sensitivity.py
python validation/validate_lower_bound.py
python validation/validate_self_consistency.py
python validation/validate_growth_baseline.py    # the baseline, before the benchmark
python validation/validate_predictors.py
python validation/validate_coverage_table.py
python validation/validate_reproducibility.py
python validation/validate_guard.py
python validation/validate_approx_audit.py       # about 4 minutes; runs the suite 3 times
python validation/worked_example.py
python validation/validate_cli.py
python validation/validate_cwd_independence.py   # about 6 minutes; 104 invocations
for example in examples/*.py; do python "$example"; done   # regenerates screenshots/
```

The example runs come last because `validate_cwd_independence.py` redraws the
screenshots from a reduced subset while probing exit statuses.

## Licence

Apache-2.0, copyright 2026 OPTIMA Organisation. The credits line required by
the mission appears once, in README.md under Credits.
