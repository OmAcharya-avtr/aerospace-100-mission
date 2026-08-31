# disturbtorque 0.1.0 — Validation evidence (Level 2, Research)

Every number in this document was produced by running the scripts in this directory in
the build session on 2026-08-31 (Python 3.11.15, numpy 2.4.4, scipy 1.17.1, 2 CPU
cores). Each script is rerunnable from this directory with `python3 <script>.py` and
writes its raw stdout next to itself; the `_output.txt` files are the primary evidence
and this document summarises them. There is no randomness anywhere in this package, so
every number below reproduces exactly, without seeds.

| Script | Raw output | Wall time | Result |
|---|---|---:|---|
| `hand_calculations.py` | `hand_calculations_output.txt` | 0.46 s | 27 passed, 0 failed |
| `atmosphere_table_continuity.py` | `atmosphere_table_continuity_output.txt` | 0.39 s | 7 passed, 0 failed |
| `orbit_geometry.py` | `orbit_geometry_output.txt` | 1.43 s | 29 passed, 0 failed |
| `momentum_integration.py` | `momentum_integration_output.txt` | 15.88 s | 20 passed, 0 failed |
| `leo_smallsat_magnitudes.py` | `leo_smallsat_magnitudes_output.txt` | 9.15 s | 7 passed, **1 reported FAILURE** |

Total validation wall time **27.3 s**, well inside the 3-minute compute budget.

## Citation policy

Citations are given at work level only (author, title, year where known). **No page,
equation, table or figure number is quoted anywhere in this product**, because no
physical copy of any reference was consulted during the build. The direct consequence
is stated plainly in section 5: the external magnitude comparison can only be made
against an order-of-magnitude *band*, not against a specific published number, and it is
therefore a weak check. The strong evidence in this product is the closed-form and
hand-arithmetic agreement of sections 1 to 4.

References used:

* Wertz, J. R. (ed.), *Spacecraft Attitude Determination and Control*, 1978.
* Larson, W. J., and Wertz, J. R. (eds.), *Space Mission Analysis and Design*.
* Vallado, D. A., *Fundamentals of Astrodynamics and Applications*.
* Hughes, P. C., *Spacecraft Attitude Dynamics*.

---

## 1. Each torque expression against hand arithmetic

`hand_calculations.py`. All four expressions are evaluated for a geometry with a closed
form, and the arithmetic is written out in the script and reproduced in the raw output.

### 1.1 Gravity gradient — PASS

`I = diag(10, 20, 30) kg m^2`, `R = 7.000000e6 m`, nadir at 45 deg in the body y-z plane.

```
R^3            = 3.43e20 m^3
3 mu / R^3     = 3 * 3.986004418e14 / 3.43e20 = 1.1958013254e15 / 3.43e20
               = 3.4863012402e-06 s^-2
I u_hat        = (0, 20*0.70710678, 30*0.70710678) = (0, 14.14213562, 21.21320344)
u_hat x I u    = (0.70710678 * (21.21320344 - 14.14213562), 0, 0) = (5.0, 0, 0) kg m^2
T              = 3.4863012402e-06 * 5.0 = 1.7431506201e-05 N m about +x
```

| Quantity | Hand value | Code value | Relative difference |
|---|---:|---:|---:|
| `3 mu / R^3` | 3.4863012402e-06 s^-2 | 3.4863012402e-06 | 9.5e-12 |
| `T_x` at 45 deg | 1.7431506201e-05 N m | 1.7431506201e-05 | 9.5e-12 |
| `T_y`, `T_z` | 0 | 0 | exactly 0 |
| `gravity_gradient_max_magnitude` | 1.7431506201e-05 N m | 1.7431506201e-05 | 9.5e-12 |

(The 9.5e-12 is the rounding of the hand value to eleven significant figures, not a code
error: the full-tensor and analytic-maximum routes agree with each other to 0.)

**The analytic maximum at 45 deg is reproduced.** Sweeping the nadir offset over 90 001
samples in `[0, 90] deg`, the argmax of `|T|` falls at **45.000000 deg** (absolute error
7.1e-15 deg) and the maximum equals `3 mu / (2 R^3) |Izz - Iyy|` to 0 relative
difference. The full-tensor form `3 mu/R^3 u x (I u)` and the planar form
`3 mu/(2R^3)(Izz - Iyy) sin 2 theta` agree over the whole sweep to 1e-14 of the torque
scale. The torque is exactly zero when nadir lies along any principal axis.

### 1.2 Aerodynamic — PASS

`rho = 1e-12 kg m^-3`, `v = (7500, 0, 0) m s^-1`, `Cd = 2.2`, `A = 1.5 m^2`,
`cp offset = (0, 0, 0.1) m`.

```
0.5 * 2.2 * 1.5 * 1e-12 = 1.65e-12 ;  |v|^2 = 5.625e7 m^2 s^-2
|F| = 1.65e-12 * 5.625e7 = 9.28125e-05 N along -x
T   = (0,0,0.1) x (-9.28125e-05, 0, 0) = (0, -9.28125e-06, 0) N m
```

Code: `T_y = -9.2812500000e-06 N m`, relative difference **1.8e-16**; `T_x = T_z = 0`
exactly. The torque is exactly zero when the offset is parallel to the flow.

### 1.3 Solar radiation pressure — PASS

Sun along `+z_body`, `A = 2.0 m^2`, `q = 0.6`, `cp offset = (0.3, 0, 0) m`, `d = 1 AU`.

```
P   = 1361 / 299792458 = 4.5398073356e-06 N m^-2
|F| = P * 2.0 * 1.6    = 1.4527383474e-05 N along -z
T   = (0.3,0,0) x (0,0,-|F|) = (0, 4.3582150422e-06, 0) N m
```

Code reproduces `P` to 0 and `T_y = 4.3582150422e-06 N m` to 0 relative difference. The
`1/d^2` law is exact (at `d = 2 AU` the torque is exactly one quarter), and the torque is
identically zero when `illuminated=False`.

### 1.4 Residual magnetic dipole — PASS

`m = (0.1, 0, 0) A m^2`, `B = (0, 3e-05, 0) T`, so `T = m x B = (0, 0, 3e-06) N m`;
`1 A m^2 * 1 T = 1 N m` exactly, so no unit conversion appears. Code: `T_z = 3.0e-06`,
relative difference 0. Zero for `m` parallel to `B`.

Centred-dipole field at the mean Earth radius:

```
k / Re^3 = 7.96e15 / 6371200^3 = 3.0778634807e-05 T  (equatorial, northward)
pole     = 2 k / Re^3          = 6.1557269614e-05 T  (directed into the Earth)
```

Both reproduced to 1e-14. The vector field and the closed-form magnitude
`(k/r^3) sqrt(1 + 3 sin^2 dec)` agree over 401 declinations to **1.1e-15**.

---

## 2. The atmosphere table's own consistency

`atmosphere_table_continuity.py`. The 28-band piecewise-exponential table was
transcribed rather than copied from a machine-readable file, so it is checked against
itself: evaluating band *k* at the base altitude of band *k+1* must return band *k+1*'s
own base density, because the published table is constructed to be continuous.

| Check | Result | Tolerance | Outcome |
|---|---:|---:|---|
| Worst relative mismatch over all 27 boundaries | 1.3597e-03 | 2e-3 | PASS |
| Worst relative mismatch above 25 km (26 boundaries) | 9.5877e-05 | 1e-4 | PASS |

The single 1.36e-3 mismatch is the 0-25 km band, a coarse single fit across the
troposphere and lower stratosphere. Every band above 25 km closes to better than 1e-4,
which is what a mistyped digit in any of the 84 numbers would break.

**Reported finding, not smoothed away.** Because of that same 0-25 km mismatch, the
density function is *not* strictly monotonic: on a 5 m grid over 0-1000 km there is
exactly one rising step, at the 25 km boundary, where density increases by **0.0671 %**
(3.896385e-02 to 3.899000e-02 kg m^-3). It is a property of the published table. Above
25 km the function is strictly decreasing, and the aerodynamic torque model is only
valid in free-molecular flow above roughly 150 km, so the step is outside the model's
validity range — but it is real and it is documented here and in the README Limitations.

Spot values used elsewhere in this document:

| altitude | 300 km | 400 km | 500 km | 600 km | 700 km | 800 km |
|---|---:|---:|---:|---:|---:|---:|
| rho [kg m^-3] | 2.418e-11 | 3.725e-12 | 6.967e-13 | 1.454e-13 | 3.614e-14 | 1.170e-14 |

---

## 3. Orbit geometry, Sun model, eclipse and the orbit-averaged field

`orbit_geometry.py`, 29 checks, all PASS.

### 3.1 Kinematics and frames

At 500 km (`R = 6878137 m`): `v = sqrt(mu/R) = 7612.608173 m s^-1`,
`T = 5676.9780 s = 94.6163 min`, and `T = 2 pi R / v` to 1.6e-16. Over 2001 samples,
`|r|` and `|v|` are constant to 2.2e-16, `r . v = 0` to 2.2e-16, and the maximum
declination equals the inclination (51.6 deg) to 7.1e-15 deg. The LVLH DCM is orthonormal
to 4.4e-16, has determinant 1 to 4.4e-16, and maps `r_hat` to `(0, 0, -1)` to 2.2e-16.

### 3.2 Sun model against externally checkable astronomical facts

`julian_date(2000-01-01 12:00 UTC) = 2451545.0` exactly. Sampling every day of calendar
2026:

| Quantity | Computed | Expected | Tolerance | Outcome |
|---|---:|---:|---:|---|
| Minimum Earth-Sun distance | 0.983293 AU (day 4) | 0.9833 AU, early January | 5e-4 AU | PASS |
| Maximum Earth-Sun distance | 1.016709 AU (day 186) | 1.0167 AU, early July | 5e-4 AU | PASS |
| Maximum solar declination | +23.4354 deg | +23.44 deg | 0.05 deg | PASS |
| Minimum solar declination | -23.4357 deg | -23.44 deg | 0.05 deg | PASS |
| Unit-vector norm | max deviation 1.1e-16 | 1 | 1e-15 | PASS |

`sun_direction_for_beta` round-trips through `beta_angle` with a worst error of
**2.8e-14 deg** over beta from -80 to +80 deg.

### 3.3 Cylindrical eclipse fraction against its closed form

`f_ecl(beta) = (1/pi) arccos( sqrt(R^2 - Re^2) / (R cos beta) )`. At 500 km,
`Re/R = 0.92730590`, `f_ecl(0) = arcsin(Re/R)/pi = 0.37788152`, and the critical beta
above which the orbit never enters the shadow cylinder is **68.0187 deg**.

| beta | sampled (200 000 pts) | closed form | absolute error |
|---:|---:|---:|---:|
| 0 deg | 0.37788500 | 0.37788152 | 3.5e-06 |
| 20 deg | 0.36959500 | 0.36959113 | 3.9e-06 |
| 40 deg | 0.33750500 | 0.33750057 | 4.4e-06 |
| 60 deg | 0.23072500 | 0.23072217 | 2.8e-06 |
| 67 deg | 0.09262500 | 0.09262649 | 1.5e-06 |
| 69.0 deg (above critical) | 0.0 | 0.0 | exactly 0 |

### 3.4 Orbit-averaged geomagnetic field against a closed form derived here

With `r_hat(u) = cos u P_hat + sin u Q_hat` and `sin(dec) = sin(i) sin(u)`, the average
of the centred-dipole field over one revolution is exactly
`<B> = (k/R^3) [ z_hat - (3/2) sin(i) Q_hat ]`, because `<sin u cos u> = 0` and
`<sin^2 u> = 1/2`. Against a 100 001-point trapezoidal average at 500 km:

| inclination | max relative difference |
|---:|---:|
| 0 deg | 1.7e-12 |
| 28.5 deg | 5.2e-15 |
| 51.6 deg | 5.7e-15 |
| 90 deg | 7.8e-15 |
| 97.8 deg | 6.6e-15 |

The tolerance is 1e-10 relative, set by the accumulated float64 roundoff of a
100 001-term sum (`N * eps = 2.2e-11`), not by a modelling argument. Field magnitude at
500 km: **24 462 nT** equatorial, **48 925 nT** polar.

---

## 4. Momentum accumulated per orbit vs direct integration

`momentum_integration.py`, 20 checks, all PASS. Reference case: 500 km, i = 51.6 deg,
beta = 20 deg, nadir-pointing with 5 deg pitch and 5 deg roll, the reference smallsat,
`P = 5676.9780 s`.

### 4.1 Constant body-frame torques — exact closed forms

For a nadir-pointing spacecraft on a circular orbit the gravity-gradient torque is
*constant in the body frame*; so is the aerodynamic torque when the co-rotating
correction is off, because the relative wind is then fixed in LVLH. Both are verified
constant to 1e-14, and the momentum integral must therefore equal `T * P` exactly.

| Source | `T_body` [N m] | `T*P` [N m s] | `int T dt` [N m s] | Relative difference |
|---|---|---|---|---:|
| gravity gradient | (6.332938e-07, 1.907139e-06, -1.112353e-07) | (3.595195e-03, 1.082678e-02, -6.314806e-04) | identical to 11 digits | **1.13e-14** |
| aerodynamic | (-3.615203e-08, -1.281033e-06, 5.268739e-07) | (-2.052343e-04, -7.272394e-03, 2.991051e-03) | identical to 11 digits | **7.99e-15** |

### 4.2 The same torques in ECI — a second closed form

The LVLH y axis is the fixed vector `-h_hat` while the x and z axes turn through a full
revolution, so for any torque constant in LVLH the ECI orbit average is exactly
`<T_eci> = -(T_lvlh)_y h_hat`.

| Source | `<T_eci>` closed form [N m] | `<T_eci>` numeric [N m] | Relative difference |
|---|---|---|---:|
| gravity gradient | (0, 1.496522e-06, -1.186129e-06) | (1.4e-22, 1.496522e-06, -1.186129e-06) | **9.2e-16** |
| aerodynamic | (0, -1.036104e-06, 8.212059e-07) | (7.7e-23, -1.036104e-06, 8.212059e-07) | **4.8e-16** |

`dh_eci` over one orbit matches `<T>_ana * P` to 8.2e-16 and 4.7e-16 respectively.

### 4.3 Full profile against grid-independent references

**Three continuous sources** (gravity gradient, aerodynamic with co-rotation, magnetic).
Their ECI histories are smooth and exactly periodic, so the closed-period trapezoidal
rule is spectrally accurate and every grid must agree. Reference `|dh| =
4.573122077267e-03 N m s`:

| N | 181 | 361 | 721 | 1441 | 2881 |
|---|---:|---:|---:|---:|---:|
| relative error | 9.1e-15 | 8.6e-15 | 8.8e-15 | 1.2e-14 | 9.1e-15 |

**Solar source**, against a reference that never touches the sample grid: the cylindrical
shadow boundaries are solved in closed form (entry `u = 113.473596 deg`, exit
`u = 246.526404 deg`, analytic eclipse fraction 0.3695911346 against 0.3694991754
sampled at N = 11521) and the torque is integrated over the sunlit arc with adaptive
Gauss-Kronrod, component by component.

```
QUADPACK reference dh_solar = (-1.2065216122e-04, -1.2863853089e-04, 3.7137042242e-04) N m s
|dh_solar| = 4.1112140090e-04 N m s      (worst reported quad absolute error 6.4e-18)
```

| N | 181 | 361 | 721 | 1441 | 2881 | 11521 |
|---|---:|---:|---:|---:|---:|---:|
| relative error | 7.46e-03 | 4.04e-04 | 3.51e-03 | 1.55e-03 | 5.67e-04 | 1.69e-04 |
| derived edge bound | 7.68e-02 | 3.84e-02 | 1.92e-02 | 9.60e-03 | 4.80e-03 | 1.20e-03 |

The error does not fall like a power of N and cannot: the torque steps to and from zero
between two adjacent samples, and the error depends on where the two eclipse edges land
inside their sample intervals. What is guaranteed is the edge bound
`2 T_peak dt / |dh_solar|` with `T_peak = 5.007208e-07 N m`; the observed error is
inside it at every N, with a worst observed/bound ratio of **0.183**.

**Total**, referenced against the sum of the two independent references above
(`|dh| = 4.3541712112e-03 N m s`):

| N | 181 | 361 | 721 | 1441 | 2881 |
|---|---:|---:|---:|---:|---:|
| relative error | 7.04e-04 | 3.81e-05 | **3.31e-04** | 1.46e-04 | 5.35e-05 |

At the package default of 721 samples the total momentum accumulated over one orbit is
accurate to **0.033 %**, and that error is entirely the eclipse edges.

Per-source momentum over one orbit at N = 11521 (ECI frame):

| Source | `|dh|` [N m s] | `|<T>|` [N m] |
|---|---:|---:|
| gravity gradient | 1.084062e-02 | 1.909576e-06 |
| aerodynamic | 6.911776e-03 | 1.217510e-06 |
| solar | 4.111292e-04 | 7.242044e-08 |
| magnetic | 2.236218e-03 | 3.939099e-07 |
| **total** | **4.354188e-03** | 7.669905e-07 |

The total is smaller than three of its parts because the contributions point in different
directions in inertial space and partly cancel — which is the whole reason the secular
momentum has to be computed as a vector and not as a sum of magnitudes.

---

## 5. Representative LEO smallsat magnitudes — **one REPORTED FAILURE**

`leo_smallsat_magnitudes.py`. Vehicle (this package's own definition, in
`disturbtorque.presets`, not taken from any published spacecraft): 100 kg,
`I = diag(4.0, 8.0, 10.0) kg m^2`, drag area 0.6 m^2 with Cd = 2.2, sunlit area 1.2 m^2
with q = 0.6, both centres of pressure offset `(0.02, 0.02, 0.05) m`
(`|offset| = 0.057446 m`), residual dipole `(0.05, 0.05, 0.10) A m^2`
(`|m| = 0.122474 A m^2`), nadir-pointing with 5 deg pitch and 5 deg roll, i = 51.6 deg,
beta = 0 deg.

Peak torque magnitude over one orbit, body frame [N m]:

| alt [km] | gravity gradient | aerodynamic | solar | magnetic | total |
|---:|---:|---:|---:|---:|---:|
| 300 | 2.1989e-06 | 4.6186e-05 | 5.0072e-07 | 4.5123e-06 | 4.6802e-05 |
| 400 | 2.1030e-06 | 6.9993e-06 | 5.0072e-07 | 4.3155e-06 | 8.7054e-06 |
| 500 | 2.0126e-06 | 1.2881e-06 | 5.0072e-07 | 4.1300e-06 | 4.2051e-06 |
| 600 | 1.9273e-06 | 2.6455e-07 | 5.0072e-07 | 3.9550e-06 | 4.3051e-06 |
| 700 | 1.8468e-06 | 6.4726e-08 | 5.0072e-07 | 3.7897e-06 | 4.2919e-06 |
| 800 | 1.7707e-06 | 2.0630e-08 | 5.0072e-07 | 3.6335e-06 | 4.1406e-06 |
| 1000 | 1.6305e-06 | 5.1625e-09 | 5.0072e-07 | 3.3460e-06 | 3.7995e-06 |

**Comparison band:** 1e-07 to 1e-04 N m, the order-of-magnitude envelope inside which
disturbance-torque estimates for small Earth-orbiting spacecraft fall in Wertz,
*Spacecraft Attitude Determination and Control*, and Larson & Wertz, *Space Mission
Analysis and Design*. It is a band rather than a value because no page reference was
verified — see the citation policy above. That makes it a weak check and it is labelled
as one.

### Result: **FAILED**, reported not adjusted

Three values fall outside the band, all aerodynamic:

| Source | Altitude | Value | Position |
|---|---:|---:|---|
| aerodynamic | 700 km | 6.4726e-08 N m | below the band's floor by a factor **1.54** |
| aerodynamic | 800 km | 2.0630e-08 N m | below by a factor **4.85** |
| aerodynamic | 1000 km | 5.1625e-09 N m | below by a factor **19.37** |

Nothing was tuned to bring them inside: the drag coefficient stays at 2.2, the areas and
offsets stay as defined above, and the density comes from the unmodified table. The
physical reading is that the band describes the *drag regime* of LEO, and this vehicle
above 700 km is no longer in it: the density at 800 km is 1.17e-14 kg m^-3, four orders
of magnitude below the 300 km value, so aerodynamic torque is genuinely negligible there
and the band's lower edge is not a meaningful floor for it. That is a limitation of the
comparison rather than evidence of a modelling error — but it is a failed check as
stated, and it is counted as one. At 300-600 km, the regime the band describes, all four
sources are inside the band: **PASS**.

### Sharper checks that do not depend on any quoted band — all PASS

| Check | Result | Tolerance | Outcome |
|---|---|---:|---|
| 45 deg roll gravity gradient equals `3 mu/(2R^3) |Izz - Iyy| = 3.6749087912e-06 N m` | relative difference 0 | 1e-12 | PASS |
| Aerodynamic/solar peak-torque crossover altitude | **559.7 km** | 400-700 km | PASS |
| `T_aero(600)/T_aero(500)` equals `rho ratio (0.2086981484) x v^2 ratio (0.9856695276) = 0.2057074053` | relative difference 5.4e-16 | 1e-12 | PASS |
| Co-rotating atmosphere reduces the aerodynamic rms torque | ratio **0.921252** (-7.87 %), against a kinematic bound of about twice the 4.09 % speed reduction | 1-12 % | PASS |

The crossover altitude moves with the area ratio, the offsets and above all with solar
activity, which the density model does not represent at all. Read it as "somewhere in
the 500-600 km decade for this configuration at mean activity", not as a constant.

---

## 6. Inter-source discrepancies in the constants — reported, not reconciled

**Solar constant.** Modern total solar irradiance is 1361 W m^-2; the older textbook
value used by Wertz and by Larson & Wertz is 1367 W m^-2.

```
P = 1361 / 299792458 = 4.5398073356e-06 N m^-2   (this package's default)
P = 1367 / 299792458 = 4.5598211814e-06 N m^-2   (textbook)
difference 0.441 %, propagating linearly into every SRP torque.
```

`SOLAR_IRRADIANCE_1AU_SMAD` is exported so a user can reproduce textbook worked examples
exactly.

**Earth dipole moment.** The package defaults to the value used in the textbook
disturbance-torque estimate, `k = 7.960e+15 T m^3`, implying an equatorial surface field

```
B0 = k / Re^3 = 7.960e15 / 6371200^3 = 3.077863e-05 T = 30 779 nT.
```

A centred-dipole reduction of a recent IGRF epoch gives an equatorial surface field near
3.0e-05 T, i.e. `k` near 7.759e+15 T m^3. The two differ by **2.6 %**, and the magnetic
torque is linear in `k`. Nothing is averaged or split between them: the textbook value
is used, and the spread is stated here, in `constants.py`, and in the README
Limitations. It sits on top of the 20-30 % pointwise error of the centred-dipole
approximation itself, which is by far the larger term.

---

## 7. Test-suite evidence

`python -m pytest tests/ -q` from `products/P027/`: **81 passed, 0 failed, 0 skipped**
(run in this session from cleared caches, 13.1 s). `ruff check src/ tests/` — clean. The suite comprises:

* **Known-answer tests** (15) carrying the section-1 hand arithmetic in their comments,
  plus the period, speed, density spot values, Julian date, Sun extremes and the eclipse
  closed form.
* **Input-validation tests** (29): non-positive-definite and triangle-inequality-violating
  inertia tensors, asymmetric tensors, negative areas, reflectance outside [0, 1],
  non-unit direction vectors, non-finite inputs, negative radii and gravitational
  parameters, altitudes outside the density model's range, Julian dates outside
  1900-2100, unknown source and frame strings, and too-small sample counts.
* **Hypothesis property tests** (11) of the algebraic identities: `T . u = 0` for the
  gravity gradient, its `1/R^3` scaling, the aerodynamic torque's exact linearity in
  density, area and Cd and its quadratic dependence on speed, perpendicularity of every
  torque to its own offset and forcing direction, antisymmetry of `m x B`, orthonormality
  and right-handedness of the LVLH and body DCMs, the dipole magnitude bracket
  `k/r^3 <= |B| <= 2k/r^3`, Kepler's third law, and unit norm of the Sun vector.
* **Integration tests** (13): the full pipeline from `Spacecraft` and `Orbit` through
  `compute_profile`, `budget` and `momentum_accumulation`; frame-change invariance of
  torque magnitude; secular-plus-cyclic reconstruction; the ECI orbit-normal closed form
  on an independent 600 km sun-synchronous case; and the CLI driven in a child process
  with its JSON output compared against the library.
* **Regression and benchmark tests** (13): the reference-case budget pinned to nine
  significant figures for all five sources, the total momentum pinned to 4.3538152233e-03
  N m s, the altitude sweep pinned at 300/500/800 km, the atmosphere table's worst
  boundary mismatch pinned to 9.5877e-05, the orbit-averaged-field closed form, the
  body-from-LVLH sign convention, a wall-time benchmark on the 721-sample sweep, and
  equatorial, polar, sun-synchronous and retrograde orbits.

---

## 8. What is NOT validated

Stated plainly, because a Level-2 claim is only as good as its scope.

* **No comparison against flight data, wind-tunnel data or a higher-fidelity simulator.**
  Every number here is a closed form, an internal consistency check, an astronomical
  fact, or a comparison against an order-of-magnitude band.
* **No validation of the models themselves.** The four torque expressions are the
  textbook first-order forms and are implemented as published. Whether they describe any
  particular spacecraft is outside this package.
* **No validation of the atmosphere against measurement.** The exponential table is
  checked only for internal consistency. It has no solar-activity dependence, which is
  the dominant error above 400 km — larger than every other uncertainty in this package
  combined.
* **No validation of the magnetic field against IGRF.** The centred non-tilted dipole is
  a sizing model; its pointwise error against IGRF is not measured here.
* **No penumbra.** The eclipse model is cylindrical umbra only.
* **No Earth albedo, no Earth infrared, no thermal re-radiation, no aerodynamic lift, no
  attitude-dependent projected area, no flexible-body or slosh contribution, no
  eddy-current or hysteresis damping torque, no J2.** Each is an absent term, not a
  validated approximation.
* **No closed-loop dynamics.** This package computes torques and their integrals for a
  *prescribed* attitude history. It does not propagate attitude, and the momentum it
  reports is the external-torque integral, not a wheel state.
