# Validation

Validation level 2. Research grade. Not flight-qualified, not certified, not
approved for operational aerospace use.

Every number in this document was produced by a script in this directory that
was executed in the session that wrote it, and the raw output of each script is
committed beside it as `<script>_output.txt`. No figure here is carried over,
estimated or rounded by hand.

**One check fails and it is the external one.** Bennett's own published
accuracy claim of 0.07 arcminutes is not reproduced against this package's ray
trace below about 19 degrees of apparent elevation. It is reported as failed in
`validate_published_references_output.txt`, the tolerance was not widened, and
what the miss does and does not prove is stated in section 1.

## What was run

| script | checks | failed | what it measures |
|---|---|---|---|
| `validate_published_references.py` | 6 | **1** | the external cross-check against published values |
| `validate_raytrace_convergence.py` | 7 | 0 | the ray trace's own numerical error budget |
| `validate_model_grid.py` | 16 | 0 | every closed form against the ray trace over 54 swept states |
| `validate_disagreement_floor.py` | 3 | 0 | equation (22) and the measured tolerance crossing |
| `validate_dispersion.py` | 5 | 0 | the four refractivity formulations across the optical band |
| `validate_profile_sensitivity.py` | 5 | 0 | equation (23), the profile nobody measured |
| `validate_low_elevation_failure.py` | 4 | 0 | where each closed form stops working, exhibited |
| `validate_controls.py` | 14 | 0 | nine controls that must fire and five that must not |
| `validate_cli.py` | 17 | 0 | every command line quoted in the README, as subprocesses |
| `worked_example.py` | 1 | 0 | the README worked example |
| `probe_working_directories.py` | 1 | 0 | 18 scripts from 5 working directories, 90 runs |
| **total** | **79** | **1** | |

Test suite: **392 tests, 0 failures, 0 errors, 0 skipped**, counted from the
`tests` attribute of the junit XML and cross-checked against a literal count of
`testcase` elements, which agree at 392. All 107 `pytest.approx` calls in the
suite pass an explicit `abs=`; re-running the suite with a plugin that makes a
bare `rel=` raise reported **0 assertions depending on the hidden 1e-12
absolute default**.

Environment, recorded in the header of every output file: Python 3.13.18,
numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, pytest 9.1.1, hypothesis
6.170.13, ruff 0.17.3 under `python3 -m ruff` and 0.16.9 on `PATH`, 2 CPU
cores. `astropy`, `ephem` and `pyerfa` are **not installed** and no number here
is compared against any of them.

## 1. The external cross-check, and the one that fails

`validate_published_references.py`. Six references in three kinds, because the
three kinds are not equally strong and collapsing them would be dishonest: a
`published_value` is a number stated in a source; a `published_claim` is an
accuracy claim an author made about their own formula relative to a computation
this package does not implement; a `convention` is a figure adopted as a working
convention rather than measured.

| reference | kind | published | measured | difference | tolerance | result |
|---|---|---|---|---|---|---|
| refractivity of standard air at 633 nm | published_value | 2.76530e-04 | 2.765288675e-04 | -1.133e-09 | 5e-9 | PASS |
| density of standard dry air, kg/m^3 | published_value | 1.2250 | 1.2254582 | +0.0004582 | 0.001 | PASS |
| saturation vapour pressure at 20 C, Pa | published_value | 2339 | 2338.3400 | -0.66 | 2.0 | PASS |
| constant of refraction, arcsec | published_value | 58.25 | 58.247503 | -0.002497 | 0.25 | PASS |
| horizon refraction, arcsec | convention | 2040 | 2041.7165 | +1.717 | 60 | PASS |
| **Bennett (1982) maximum error, arcsec** | **published_claim** | **4.20** | **34.0737** | **+29.87** | **0** | **FAIL** |

**The failure, in full.** Bennett (1982) states a maximum error of 0.07
arcminutes over 0 to 90 degrees relative to Garfinkel (1967). Measured against
this package's ray trace at 1010 hPa, 10 C, dry and 550 nm over 181 elevations:
the maximum absolute departure is **34.0737 arcsec = 0.5679 arcmin at zero
elevation**, **63 of 181 grid points** lie outside the claim, and the claim
**holds at and above 19.1361 degrees** and fails below it.

What this does **not** show is that Bennett's formula is wrong. This package
does not implement Garfinkel (1967), so the comparison is against a different
reference atmosphere. What it does show is that the two reference atmospheres
differ by more than 0.07 arcmin wherever it misses, and section 6 measures how
much freedom there is in that choice: the three declared surface lapse rates of
equation (23) move the horizon refraction by 301.3 arcsec on their own at the
conditions of that check, and the full swept set of surface lapse rates moves
it by 1245.4164 arcsec. A 34 arcsec difference between two atmosphere models at
zero elevation sits inside that freedom and this package cannot attribute it. The wavelength choice alone moves the measured departure: at
633 nm instead of 550 nm it is 44.500 arcsec, +10.427 arcsec.

**What this product does not have.** There is no transcription of a printed
refraction table in this repository. None was available to this build and
inventing plausible table entries would be worse than having none. That is the
principal weakness of the external verification and it is stated here rather
than left to be inferred.

Two of the passing checks carry their own caveats, printed in the output file:

* The **constant of refraction** is published for the visible band without a
  stated wavelength. 550 nm is this product's choice. The measured coefficient
  is 58.81265 arcsec at 450 nm and 57.97341 arcsec at 633 nm, a span of
  0.83924 arcsec across the visible, which is **3.36 times the tolerance of the
  check**. Agreement at the midpoint therefore rests on the wavelength this
  product picked for a figure that is published without one.
* The **saturation vapour pressure** check compares the **pure-water**
  saturation forms, because that is what the published 2339 Pa is: equation (4)
  gives 2338.3400 Pa and the bare IAPWS form of equation (12) gives 2339.1632
  Pa. Applying the Ciddor enhancement factor at 101325 Pa raises the effective
  partial pressure in air to **2348.5798 Pa**, +9.4165 Pa above the pure-water
  figure. That is a real and correct difference, not an error, and it is
  reported rather than compared against a reference published for a different
  quantity.
* The **horizon convention** is a rounded figure adopted for rise-and-set
  computations, and the profile envelope at that elevation and those conditions
  is 301.3 arcsec, **5.02 times the check's own tolerance**. Agreement there
  says very little.

## 2. The ray trace's numerical error budget

`validate_raytrace_convergence.py`. Before the closed forms can be scored
against the ray trace, the ray trace's own numerical error has to be smaller
than the differences being measured.

| quantity | measured | declared bound |
|---|---|---|
| grid residual, 10,000 against 40,000 shells, above the horizon | worst **9.299e-05 arcsec** at 0.2 deg | 1e-3 arcsec |
| grid residual at zero elevation | **6.077e-03 arcsec** | 2e-2 arcsec |
| truncation, 90 km against 110 km ceiling | worst **4.851e-04 arcsec** | 1e-2 arcsec |
| flat-Earth limit against the exact Snell solution | worst **3.506e-06 relative** | 2e-4 |

The horizon converges more slowly than anywhere else because the integrand has
an integrable inverse-square-root singularity at the observer. The grid
exponent `q` of equation (20) matters only there.

**The independently derived bracket.** `tan(theta)` decreases monotonically
along the ray and the total measure of the integration variable is
`ln n_0 - ln n_top`, so the mean value theorem brackets the refraction between
`tan(theta_top)` and `tan(theta_0)` times that measure. That argument shares no
mathematics with the shell sum of equation (20).

| elevation (deg) | lower | measured | upper | width/value | verdict |
|---|---|---|---|---|---|
| 60 | 32.9536 | 33.5080 | 33.5597 | 0.0181 | inside |
| 45 | 56.5733 | 57.9931 | 58.1271 | 0.0268 | inside |
| 30 | 95.5016 | 100.2179 | 100.6790 | 0.0517 | inside |
| 20 | 143.5330 | 158.1608 | 159.7028 | 0.1022 | inside |
| 10 | 237.7035 | 318.1216 | 329.6550 | 0.2890 | inside |
| 5 | 307.6048 | 589.6477 | 664.3955 | 0.6051 | inside |
| 2 | 340.7794 | 1089.1223 | 1664.5416 | 1.2154 | inside |

The bracket is not vacuous and it was verified to discriminate: it **rejects
the exact plane-parallel answer** at every elevation tested, by +0.00636 arcsec
at 60 degrees rising to +0.19306 arcsec at 20 degrees, because
`ln(1 + kappa) tan z` is strictly smaller than `kappa tan z`; and it rejects a
ray trace scaled by 2.5 times the bracket's own relative width at 6 of 6
elevations. The injection is scaled by the **conditioning of the bound**, not by
an absolute constant.

## 3. Every closed form against the ray trace, swept

`validate_model_grid.py`. Fourteen elevations crossed with 54 atmospheric
states — 3 pressures, 3 temperatures, 2 humidities, 3 wavelengths — for 756
cells.

Maximum absolute error against the ray trace, arcsec, over all 54 states:

| elevation | plane_parallel | oriani_series | bennett_1982 | saemundsson_1986 | mean reference |
|---|---|---|---|---|---|
| 90 | 0.0000 | 0.0000 | 0.0963 | 0.1374 | 0.000 |
| 60 | 0.0603 | 0.0003 | 1.6545 | 2.4177 | 29.900 |
| 45 | 0.1563 | 0.0007 | 2.8670 | 4.1900 | 51.749 |
| 30 | 0.5390 | 0.0057 | 4.7724 | 6.9870 | 89.431 |
| 20 | 1.8104 | 0.0450 | 6.9185 | 10.1339 | 141.151 |
| 15 | 4.2400 | 0.1854 | 8.8968 | 12.2692 | 190.424 |
| 10 | 13.8348 | 1.2983 | 12.1190 | 15.4135 | 284.049 |
| 5 | 95.8933 | 29.8737 | 22.1935 | 21.4150 | 527.252 |
| 2 | 1035.9934 | 1040.2506 | 85.8231 | 68.0302 | 977.621 |
| 1 | undefined | undefined | 147.9814 | 104.9989 | 1301.564 |
| 0 | undefined | undefined | 300.8100 | 208.1177 | 1841.692 |

The **sign** of the error is the finding, not only its size. Above 15 degrees
the plane-parallel model and both published navigational fits **overstate** the
refraction relative to the layered integral at every one of the 54 states, and
the Oriani series understates it by a few hundredths of an arcsecond. Below
about 7 degrees the sign of the published fits' error stops being constant
across the state grid: at 5 degrees Bennett's signed error runs from -22.194 to
+20.229 arcsec. **A departure with no constant sign cannot be removed by a
scale factor.**

At zero elevation Bennett's signed error spans **489.64 arcsec across the 54
states**, from -300.81 to +188.83. Equation (17) carries no wavelength and only
a density ratio, and at the horizon that is not enough.

Measured usable range, the lowest elevation above which each method's maximum
error over all 54 states stays at or below 10 arcsec:

| method | equation | declared floor | measured floor | measured against declared |
|---|---|---|---|---|
| plane_parallel | (13) | 30 deg | 15 deg | wider |
| oriani_series | (14) | 15 deg | 7 deg | wider |
| bennett_1982 | (15) | 0 deg | 15 deg | **narrower** |
| saemundsson_1986 | (16) | 0 deg | 30 deg | **narrower** |

Both published fits are published for the whole sky and both are measured to be
usable, at a 10 arcsec budget against this package's ray trace, only over a
part of it. This is a measurement against a different reference atmosphere, not
a claim about their accuracy against their own.

**The Oriani series and the ray trace agree to 0.0007 arcsec at 45 degrees** and
to 0.1854 arcsec at 15 degrees, over all 54 states. A truncated asymptotic
series and a numerical shell integral are independent derivations, and this is
the strongest internal consistency result in the package.

## 4. The disagreement curve and the measured validity floor

`validate_disagreement_floor.py`. Equation (22) is the range of the five
methods over the members defined at each elevation.

| elevation (deg) | spread (arcsec) | widest pair |
|---|---|---|
| 90 | 0.1157 | saemundsson_1986 / plane_parallel |
| 60 | 1.6144 | oriani_series / saemundsson_1986 |
| 45 | 2.8049 | oriani_series / saemundsson_1986 |
| 30 | 4.6666 | oriani_series / saemundsson_1986 |
| 20 | 6.7455 | oriani_series / saemundsson_1986 |
| 10 | 14.2224 | oriani_series / plane_parallel |
| 5 | 114.1489 | oriani_series / plane_parallel |
| 2 | 1839.9769 | oriani_series / plane_parallel |
| 1 | 14.7497 | raytrace / saemundsson_1986 |
| 0 | 34.0737 | raytrace / bennett_1982 |

**The curve is not monotone, and that is structural.** At a 10 arcsec tolerance
the scan finds **3 crossings** on the 181-point grid: between 1.225 and 1.344
degrees, between 1.344 and 1.469 degrees, and between 11.025 and 11.37778
degrees. The drops are members **leaving** the ensemble at their own validity
limits. A search that assumed one crossing and walked up from the horizon would
stop at **1.225 degrees** and understate the floor of **11.2448 degrees** by a
factor of **9.179**. This package always takes the highest crossing.

Measured floor, in degrees of apparent elevation, by declared tolerance and
atmospheric state:

| state | 1 arcsec | 5 arcsec | 10 arcsec | 30 arcsec | 60 arcsec |
|---|---|---|---|---|---|
| 1010 hPa, 10 C, dry, 550 nm | 69.789 | 28.050 | 11.245 | 7.799 | 6.192 |
| 1035 hPa, -25 C, dry, 550 nm | 72.090 | 31.950 | 11.340 | 7.867 | 6.247 |
| 1005 hPa, 35 C, 90 pct, 550 nm | 70.611 | 29.607 | 13.259 | 7.767 | 6.167 |
| 700 hPa, 0 C, dry, 550 nm | 63.353 | 18.684 | 9.950 | 6.901 | 5.478 |
| 1010 hPa, 10 C, dry, 400 nm | 60.472 | 14.249 | 11.311 | 7.845 | 6.229 |
| 1010 hPa, 10 C, dry, 1000 nm | 73.676 | 35.124 | 17.152 | 7.765 | 6.165 |

The bisection is auditable: at 1010 hPa, 10 C, dry, 550 nm and 10 arcsec the
bracket is [11.244797, 11.244883] degrees, width 8.613e-05 degrees, with the
spread at 10.000223 arcsec below and 9.999993 arcsec above, after 12
iterations.

## 5. Where each closed form stops working

`validate_low_elevation_failure.py`. Three failure modes, and they are not the
same thing.

| limit | value | how it is obtained |
|---|---|---|
| equation (13) analytic floor | 1.360182 deg | closed form, `90 - arcsin(1/n)`; the equation has no solution below it |
| equation (14a) breakdown | 1.950707 deg | closed form from the series coefficients; the truncation inverts |
| equations (15) and (16) | none | both remain finite and smooth to the horizon |

What each form returns below its own limit, arcsec, at 1010 hPa, 10 C, dry,
550 nm:

| elevation | eq (13) | eq (14) guarded | eq (14) unguarded | eq (15) | eq (16) | eq (19) |
|---|---|---|---|---|---|---|
| 5.00 | 677.207 | 563.058 | 563.058 | 592.989 | 595.228 | 589.649 |
| 3.00 | 1172.946 | 639.945 | 639.945 | 860.665 | 860.172 | 858.911 |
| 2.00 | 1920.986 | 81.009 | 81.009 | 1092.965 | 1091.204 | 1089.098 |
| 1.75 | 2335.696 | undefined | **-461.252** | 1168.826 | 1167.086 | 1163.440 |
| 1.50 | 3123.220 | undefined | **-1533.802** | 1254.162 | 1252.734 | 1246.617 |
| 1.00 | undefined | undefined | **-9338.314** | 1459.747 | 1460.175 | 1445.425 |
| 0.25 | undefined | undefined | **-797459.678** | 1886.118 | 1889.205 | 1856.009 |
| 0.00 | undefined | undefined | undefined | 2068.652 | 2065.952 | 2034.578 |

Equation (14) does not fade out; it **reverses**. Equation (13) diverges upward
instead, reaching 1.764 times the layered integral at 2 degrees before its own
floor cuts it off.

**The dangerous mode is the third one.** Equations (15) and (16) stay within
**1.687 per cent** of the layered integral all the way to the horizon and
remain monotone and smooth. There is nothing in their output to tell a caller
that the quantity they are being compared against is itself uncertain by five
to twenty arcminutes there.

## 6. Equation (23): the profile nobody measured

`validate_profile_sensitivity.py`.

Range of the ray-trace refraction over the swept surface lapse rates, -9.8 to
+30 K/km applied to the whole troposphere:

| elevation | range (arcsec) | as arcmin |
|---|---|---|
| 45 deg | 0.1489 | 0.0025 |
| 10 deg | 0.4805 | 0.0080 |
| 5 deg | 12.3676 | 0.2061 |
| 1 deg | 358.4139 | 5.9736 |
| 0 deg | 1245.4164 | 20.7569 |

Over the **three declared envelope members** of equation (23), which are
physically attainable, the range at 45 degrees is 0.00143 arcsec and at zero
elevation 300.018 arcsec.

A **25 m surface inversion of +20 K/km** moves the horizon refraction by
**+137.001 arcsec** against the US Standard Atmosphere 1976 profile, and is
invisible at 45 degrees.

**The headline comparison**, equation (22) against equation (23):

| elevation | eq (22) spread | eq (23) envelope | larger term |
|---|---|---|---|
| 45 | 2.8049 | 0.0014 | cross-model |
| 20 | 6.7455 | 0.0061 | cross-model |
| 10 | 14.2224 | 0.2336 | cross-model |
| 5 | 114.1489 | 3.5557 | cross-model |
| 2 | 1839.9769 | 38.7125 | cross-model |
| 1 | 14.7497 | 102.4791 | **profile** |
| 0.5 | 27.9254 | 172.9883 | **profile** |
| 0 | 34.0737 | 300.0181 | **profile** |

Between 1 and 5 degrees the cross-model spread is the larger term only because
the plane-parallel model and the Oriani series are diverging there. Among the
three methods that remain defined below 1.36 degrees the profile term already
dominates, by factors of 6.9 to 8.8.

Equation (23) is a **lower** bound on the profile uncertainty, not an upper
one: a real surface inversion is steeper than isothermal and the three members
are a declared bracket, not a distribution.

## 7. Wavelength dependence

`validate_dispersion.py`.

* The four formulations agree on dry air to **9.748e-08 in n** at worst over
  0.32 to 1.69 um, which is 0.01174 arcsec of refraction at 45 degrees inside
  the band where all four are published valid, and 0.02006 arcsec at 0.32 um
  where one of them is outside its published range.
* They disagree on **humidity** by **11.77 times** the dry-air spread at 40 C
  and saturation: 1.3405e-06 against 1.1393e-07. The Explanatory Supplement
  humidity term of equation (9) is the outlier.
* **Atmospheric dispersion**, the refraction difference between 400 nm and
  700 nm, measured by the ray trace: 0.8418 arcsec at 60 degrees, 1.4571 at 45,
  3.9775 at 20, 15.0043 at 5. It exceeds 1 arcsec below **55.426 degrees**.
* **Equations (15) and (16) predict exactly zero atmospheric dispersion**, at
  every elevation, because the published formulae take no wavelength. This is a
  different class of limitation from being a few arcseconds out.
* Saturation at 30 C **lowers** the optical refraction by 0.3008 arcsec at 45
  degrees, 3.3865 arcsec at 5 degrees and 25.0510 arcsec at zero elevation.
  Water vapour raises radio refractivity; this product does not cover the radio
  band.

## 8. Controls

`validate_controls.py`. Nine controls must fire and five must not. **All nine
fired and none of the five did.**

| control | anchored on | result |
|---|---|---|
| C1 eq (13) refuses below its analytic floor | `90 - arcsin(1/n)`, computed from the refractivity | fired |
| C2 eq (14) refuses below its computed breakdown | equation (14a) in closed form | fired |
| C3 bracket rejects a 6.7% error at 45 deg | the bracket's own relative width of 2.68% | fired |
| C4 bracket rejects a 25.6% error at 20 deg | its width of 10.22% | fired |
| C5 bracket rejects a 72.3% error at 10 deg | its width of 28.90% | fired |
| C6 bracket rejects the exact plane-parallel answer | a model the package ships | fired |
| C7 the verdict refuses at 2 deg, 10 arcsec | the measured floor | fired |
| C8 the envelope alone refuses with the spread set to zero | the measured envelope | fired |
| C9 a 1e-6 hPa atmosphere gives 5.763e-07 arcsec | the vacuum | fired |
| N1 bracket does not reject the ray trace, 7 elevations | — | did not fire |
| N2 the verdict does not refuse at 80 deg, 10 arcsec | — | did not fire |
| N3 a 200 m +20 K/km inversion does not refuse at 80 deg | — | did not fire |
| N4 the eq (14a) guard does not refuse above the breakdown | — | did not fire |
| N5 the verdict label is unchanged over 400 to 633 nm | the published ranges | did not fire |

No control's criterion is a seed, and every injected error is scaled by the
conditioning of the bound it is injected into rather than by an absolute
constant.

## 9. Command line and working directories

`validate_cli.py`: 15 invocations run as subprocesses with a temporary `HOME`
and `MPLCONFIGDIR` outside the repository and `PYTHONDONTWRITEBYTECODE=1`. All
15 returned the expected exit status and the expected stdout fragment. The two
pollution checks are deterministic: the subprocess environment is asserted to
point `HOME` and `MPLCONFIGDIR` outside the repository and to disable bytecode
writing, and no `.config` directory is asserted to exist anywhere in the tree.

A tree-wide differential over cache directories was tried first and is **not** a
check, because it has a race that has nothing to do with the defect it is
looking for: it failed once in 90 probe runs during this build, caused by a
`ruff check` the build agent ran in another shell while the probe was in
flight. The differential is still printed, as information.

`python -m refractpath references` **exits 1 by design**: one of the six
external reference checks does not pass and the command reports it. An exit 0
there would mean the failure had been hidden.

`probe_working_directories.py`: 18 scripts run from the filesystem root, the
system temporary directory, the repository root, `validation/` (which is what
the release gate uses) and `src/`. **0 non-zero exit statuses out of 90 runs.**

## 10. Defects found during this build

Each was found, fixed, and pinned as a named regression in
`tests/test_regressions.py` with the literal inputs in the test body.

* A trapezoid rule over shell **boundaries** evaluates `tan(theta)` at the
  observer, and for a ray launched exactly horizontally the observer is the
  tangent point, so the horizon refraction came out as 8.0e138 arcmin.
  Equation (20) evaluates at shell **midpoints**, which never touches it.
* Computing `1 - sin^2(theta)` directly loses all precision when `sin(theta)`
  is within 1e-14 of unity, which is the first few metres of a horizon ray.
  Equation (21) factorises it into terms with no catastrophic cancellation.
* Without the equation (14a) guard the Oriani series returns about -1.26e7
  arcsec at 0.1 degrees and about -2.9e47 arcsec at exactly 0 degrees in
  floating point, which made equation (22) a measurement of a floating-point
  artefact rather than of a disagreement.
* The validity floor took the lowest tolerance crossing, which understates it
  by a factor of 9.179 at a 10 arcsec tolerance.
* The test suite first asserted the wrong sign for the observer-height
  dependence. At fixed station pressure a larger radius means a smaller
  curvature correction, so the refraction **rises** toward the plane-parallel
  limit: 589.6488 arcsec at sea level against 589.6995 arcsec at 5 km, at 5
  degrees.

Hypothesis found four falsifying examples, all pinned:

* A 1e-3 bound on `R(2p)/R(p) - 2` falsified at 807 hPa and -2 C (measured
  1.00089e-3), then a widened 2e-3 bound falsified at 1002 hPa and -41 C
  (measured 2.00103e-3). Both are the known 1e-8 p term of equation (7), so
  the property was replaced by the **exact algebraic ratio** rather than by a
  third tolerance.
* A strict `raytrace < plane_parallel` falsified at exactly 90 degrees, where
  both are identically zero.
* "The Oriani breakdown is always above the plane-parallel floor" falsified at
  1079 hPa, -59 C and 1.0 um, where the two are 1.6067910 and 1.6069088
  degrees. The ordering reverses exactly where `beta = 2.5 kappa`, which is now
  the asserted property.

## What is not validated here

* **No comparison against a measurement.** Every number above compares one
  computation against another computation or against a published number. There
  is no theodolite data in this repository and none was available.
* **No comparison against `astropy`, `ephem` or `erfa`.** None of the three is
  installed in the measured environment and nothing was installed to get them.
* **No transcribed refraction table.** See section 1.
* **No horizontal gradient, no turbulence, no radio band.** The ray trace of
  equation (19) assumes spherical symmetry and computes the mean refraction
  only.
* **Both ends of the declared dependency ranges were not run.** The build
  environment is a shared pre-pinned interpreter this product is forbidden to
  modify, and installing a second numpy or scipy would have required running
  `pip`. Both ranges are bounded at the single version each was measured at.
  The ruff range is the exception: 0.16.9 and 0.17.3 were both available and
  the source measured clean under both.
