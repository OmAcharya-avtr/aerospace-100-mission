# KeepOut — Validation evidence (Level 1, Educational)

**Product:** P030 KeepOut · **Version:** 0.1.0 · **Date of run:** 2026-08-31
**Environment:** Python 3.11, NumPy, SciPy, 2 CPU cores. Every number in this
file was produced by running the scripts in this directory in the session that
wrote it; each script's raw stdout is committed beside it.

| # | Check | Script | Raw output | Result |
|---|---|---|---|---|
| 1 | Cone/cap geometry vs closed-form spherical geometry | `validate_cone_geometry.py` | `cone_geometry_output.txt` | PASS |
| 2 | Earth angular radius vs `arcsin(R_E/(R_E+h))` and vs numerical maximisation | `validate_earth_angular_radius.py` | `earth_angular_radius_output.txt` | PASS |
| 3 | Hand-computed two-overlapping-cone case | `validate_hand_case.py` | `hand_case_output.txt` | PASS |
| 4 | Rotational invariance and window-search consistency | `validate_invariance.py` | `invariance_output.txt` | PASS |
| 5 | Low-precision Sun/Moon series vs published orbital constants | `validate_ephemeris.py` | `ephemeris_output.txt` | PASS |

Reproduce, from `products/P030/`:

```bash
python validation/validate_cone_geometry.py
python validation/validate_earth_angular_radius.py
python validation/validate_hand_case.py
python validation/validate_invariance.py
python validation/validate_ephemeris.py
```

Wall-clock on the 2-core build machine: 43.7 s, 0.8 s, 3.3 s, 65.0 s and
0.9 s respectively, 113.6 s in total. The 2 000 000-sample Monte Carlo
cross-checks and the 20 000-rotation sweep dominate.

---

## 1. Cone geometry against closed-form spherical geometry

### 1a. Cap solid angle

**Reference.** `Omega = 2 pi (1 - cos alpha)`, the elementary integral
`int_0^alpha 2 pi sin(theta) d(theta)` over the unit sphere. Compared against
SciPy's adaptive `quad` of that integrand.

Twelve angular radii from 0.5 deg to 180 deg. **Worst `|closed form −
quadrature|` = 1.776e-15 sr**, tolerance 1e-12 → **PASS**. Ten of the twelve
agree to 0 or 1 ulp.

### 1b. Two-cap intersection, exactly known cases

The closed form in `cap_intersection_solid_angle` is the Gauss-Bonnet lens area

```
A = 2 (pi - gamma) - 2 alpha_1 cos(r1) - 2 alpha_2 cos(r2)
```

with `alpha_1`, `alpha_2`, `gamma` the angles of the spherical triangle
(axis 1, axis 2, boundary crossing), evaluated by the half-angle form
(Todhunter, *Spherical Trigonometry*, 5th ed., 1886, Art. 45).

Twelve cases with an independently known answer:

| Case | Exact value | Worst deviation |
|---|---|---|
| Coincident caps, `r = 10, 45, 80 deg` | the cap's own area | 0.000e+00 |
| Disjoint caps, `d > r1 + r2` | 0 | 0.000e+00 |
| Two hemispheres, `d = 5, 30, 90, 150, 175 deg` | lune of area `2 (pi − d)` | 4.441e-16 |
| Caps whose complements are disjoint (`r1 + r2 + d ≥ 2 pi`) | `A1 + A2 − 4 pi` | 0.000e+00 |

**Worst `|library − exact|` = 4.441e-16 sr**, tolerance 1e-13 → **PASS**.

The hemisphere case is the sharpest of these: two hemispheres separated by `d`
intersect in a lune of dihedral angle `pi − d`, and a spherical lune of angle
`A` has area `2 A`. That value is independent of the lens derivation entirely.

### 1c. Lens formula against band quadrature, 300 random configurations

Two algorithms with nothing in common: the Gauss-Bonnet lens area above, and
the band quadrature in `keepout.regions`, which computes the blocked azimuth arc
on each ring of constant colatitude in closed form and integrates it over
`cos(theta)` by Gauss-Legendre.

Random axes (isotropic) and random half-angles on `[0.05, pi − 0.05]` rad, seed
`20260831`. **Worst `|difference|` = 2.655e-11 sr** at `r1 = 84.1298 deg,
r2 = 93.5324 deg, separation = 38.2916 deg`, where the two routes give
5.035653126579215 sr and 5.035653126605770 sr. Tolerance 1e-10 → **PASS**.

### 1d. Band-quadrature convergence

Same 300 configurations, varying `nodes_per_band`:

| `nodes_per_band` | worst `|difference|` [sr] |
|---:|---:|
| 12 | 1.157e-04 |
| 24 | 1.910e-06 |
| 48 | 2.165e-08 |
| 64 | 2.407e-09 |
| **96 (default)** | **2.655e-11** |
| 128 | 1.108e-12 |

The blocked azimuth measure behaves like `sqrt(|u − u_edge|)` at each band
edge, because `arccos` has an infinite derivative where its argument reaches
`±1`. Plain Gauss-Legendre would converge algebraically against that; the
implementation substitutes `u = mid + half·sin(s)`, whose `cos(s)` Jacobian
cancels the square root. The table above is with that substitution in place.

### 1e. Monte Carlo cross-check

Uniform directions from normalised isotropic Gaussians, 2 000 000 samples,
binomial error bar `4 pi sqrt(p(1−p)/n)`:

| Configuration | Band quadrature [sr] | Monte Carlo [sr] | Discrepancy |
|---|---|---|---|
| Two caps, 30/25 deg, 40 deg apart | 11.255653505 | 11.259254442 ± 0.002712670 | 1.33 σ |
| Three caps, 50/45/40 deg, mutually overlapping | 8.518227685 | 8.518516445 ± 0.004152211 | 0.07 σ |
| Sun 45 + Earth 77 + Moon 15 deg, LEO-like | 5.952500686 | 5.952564096 ± 0.004436728 | 0.01 σ |

All within 4 σ → **PASS**.

---

## 2. Earth angular radius

**Reference.** `alpha = arcsin(R_E / (R_E + h))`, the tangent-cone half-angle
from a point at distance `R_E + h` to a sphere of radius `R_E`; the radius to
the tangent point is perpendicular to the line of sight (Wertz, *Spacecraft
Attitude Determination and Control*, Reidel 1978, Sec. 5.2; Vallado,
*Fundamentals of Astrodynamics and Applications*, 4th ed., 2013, Sec. 11.7).
`R_E = 6378137.0 m`, the WGS-84 equatorial radius (NIMA TR8350.2, 3rd ed.).

### 2a. Against the analytic expression

Nineteen altitudes from the surface to lunar distance. **Worst `|library −
arcsin(R_E/(R_E+h))|` = 0.000e+00 rad** — bit-identical at every altitude →
**PASS**.

Selected values:

| `h` [km] | angular radius [deg] |
|---:|---:|
| 0 | 90.0000000000 |
| 400 | 70.2179312813 |
| 550 | 67.0159484695 |
| 800 | 62.6916609107 |
| 20200 (MEO/GNSS) | 13.8851727117 |
| 35786 (GEO) | 8.7004880880 |
| 384400 (lunar distance) | 0.9352021618 |

### 2b. Against a route that never uses the tangency identity

The apparent angular radius is by definition the maximum, over the body's
surface, of the angle subtended at the observer between the body centre and a
surface point. `validate_earth_angular_radius.py` maximises that angle
numerically with a bounded scalar optimiser, using no tangency relation
anywhere. **Worst `|arcsin form − numerical maximum|` = 1.399e-14 rad**,
tolerance 1e-9 → **PASS**.

### 2c. Limits and monotonicity

- `h = 0` gives exactly 90 deg (`|difference|` = 0.000e+00 rad).
- Strictly decreasing over `h = 0` to 5e7 m at 20 001 samples: **True**.
- Small-angle limit `alpha → R_E/(R_E+h)`: relative difference falls from
  6.695e-06 at `h = 1e9 m` to 6.780e-12 at `h = 1e12 m`, i.e. as `h^-2`, which
  is the leading correction of the arcsine series.

### 2d. Limb-referenced cone construction

`body_exclusion_cone(..., reference="limb")` must return
`arcsin(R_E/(R_E+h)) + instrument angle`. Four cases (400 km/10 deg,
550 km/10 deg, 700 km/20 deg, 35786 km/5 deg): **worst `|difference|` =
0.000e+00 rad** → **PASS**. At 550 km with a 10 deg instrument angle the cone
half-angle is **77.0159484695 deg**.

---

## 3. The hand-computed two-overlapping-cone case

**Geometry.** Cone A: axis `+x`, half-angle 30 deg. Cone B: axis at 40 deg from
`+x` in the `x-y` plane, half-angle 25 deg. Since
`|30 − 25| = 5 < 40 < 55 = 30 + 25`, the caps properly overlap and neither
contains the other.

Every test boresight lies in the `x-y` plane, so its angle to axis A is its own
azimuth and its angle to axis B is `|azimuth − 40 deg|`. Cone A covers azimuths
`(−30, +30)`, cone B covers `(+15, +65)`.

### 3a. The arithmetic, as printed by `validate_hand_case.py`

```
cos r1 = 0.8660254037844387    sin r1 = 0.49999999999999994
cos r2 = 0.9063077870366499    sin r2 = 0.42261826174069944
cos d  = 0.766044443118978     sin d  = 0.6427876096865393

cos a1 = (cos r2 - cos r1 cos d) / (sin r1 sin d)
       cos r1 cos d = 0.6634139481689384
       numerator    = 0.9063077870366499 - 0.6634139481689384 = 0.24289383886771154
       denominator  = 0.49999999999999994 * 0.6427876096865393 = 0.32139380484326957
       cos a1 = 0.7557514650481854  ->  a1 = 0.713995443187023 rad = 40.90892548618917 deg

cos a2 = (cos r1 - cos r2 cos d) / (sin r2 sin d)
       cos r2 cos d = 0.6942720440148838
       numerator    = 0.8660254037844387 - 0.6942720440148838 = 0.17175335976955486
       denominator  = 0.42261826174069944 * 0.6427876096865393 = 0.2716537822741844
       cos a2 = 0.6322509421061604  ->  a2 = 0.8863412197824059 rad = 50.78361110200917 deg

cos g  = (cos d - cos r1 cos r2) / (sin r1 sin r2)
       cos r1 cos r2 = 0.7848855672213958
       numerator     = 0.766044443118978 - 0.7848855672213958 = -0.018841124102417783
       denominator   = 0.49999999999999994 * 0.42261826174069944 = 0.2113091308703497
       cos g  = -0.08916379535902733  ->  g  = 1.6600786915769774 rad = 95.1155026869607 deg

A = 2 (pi - g) - 2 a1 cos r1 - 2 a2 cos r2
       2 (pi - g)  = 2.9630279240256314
       2 a1 cos r1 = 1.236676383972582
       2 a2 cos r2 = 1.6065958989207145
       A           = 0.11975564113233483 sr
```

**Library `cap_intersection_solid_angle` = 0.11975564113233417 sr**,
`|difference| = 6.661e-16 sr`, tolerance 1e-15 → **PASS**. The library reaches
the same value through the half-angle form of the spherical law of cosines, so
the two do not share a code path; the 6.7e-16 gap is the difference between the
two algebraically equivalent routes at double precision, and the half-angle form
is the one that stays accurate for tiny triangles (see §3d).

### 3b. Union and allowed sky

```
A1    = 2 pi (1 - cos 30 deg) = 0.8417872144769325 sr
A2    = 2 pi (1 - cos 25 deg) = 0.5886855358884618 sr
union = A1 + A2 - A           = 1.3107171092330594 sr
allowed = 4 pi - union        = 11.255653505126112 sr
```

Band quadrature (672 nodes) gives **11.255653505126142 sr**,
`|difference| = 3.020e-14 sr`, tolerance 1e-10 → **PASS**. As a fraction of the
sky, **0.8956964465352217**, i.e. 89.569645 %.

Monte Carlo with 2 000 000 samples: **11.254498071 ± 0.002717027 sr**, 0.43 σ
from the hand value.

### 3c. Violation verdicts

Eleven boresights, every one decided by hand first:

| azimuth [deg] | hand reasoning | expected | library |
|---:|---|---|---|
| 0 | 0 < 30 inside A; 40 > 25 outside B | {A} | {A} |
| 5 | 5 < 30 inside A; 35 > 25 outside B | {A} | {A} |
| 20 | 20 < 30 inside A; 20 < 25 inside B | {A,B} | {A,B} |
| 25 | 25 < 30 inside A; 15 < 25 inside B | {A,B} | {A,B} |
| 35 | 35 > 30 outside A; 5 < 25 inside B | {B} | {B} |
| 64 | 64 > 30 outside A; 24 < 25 inside B | {B} | {B} |
| 66 | 66 > 30 outside A; 26 > 25 outside B | {} | {} |
| 70 | 70 > 30 outside A; 30 > 25 outside B | {} | {} |
| −29 | 29 < 30 inside A; 69 > 25 outside B | {A} | {A} |
| −31 | 31 > 30 outside A; 71 > 25 outside B | {} | {} |
| 180 | opposite the union | {} | {} |

11 / 11 exact → **PASS**.

At azimuth 25 deg the margins are **−5.0000000000 deg** (cone A) and
**−10.0000000000 deg** (cone B), and `violations()` returns `('B', 'A')` —
deepest violation first, as documented.

### 3d. Why the half-angle form

The direct law-of-cosines form `cos A = (cos a − cos b cos c)/(sin b sin c)`
cancels catastrophically when all three sides are near `sqrt(eps) ≈ 1e-8` rad.
A Hypothesis run found `r1 = r2 = d = 1.64e-07` rad, for which the direct form
returned an intersection of **−0.003683278942092194 sr** — negative, and larger
in magnitude than either cap. The half-angle form
`tan²(A/2) = sin(s−b) sin(s−c) / (sin s sin(s−a))` takes sines of positive
quantities only and does not cancel; that failure is now covered by
`tests/test_properties.py::TestCapAreaIdentities`.

---

## 4. Rotational invariance

A keep-out verdict is a statement about relative geometry: rotating the
boresight and every cone axis by the same element of SO(3) must change nothing.
Rotations are drawn from the Haar measure on SO(3) via uniform unit quaternions
(Shoemake, *Graphics Gems III*, 1992).

| Check | Trials | Result | Tolerance |
|---|---:|---|---|
| Per-cone margin change, 3 random cones each trial | 20 000 | worst **1.110e-15 rad** (2.290e-07 mas) | 1e-12 rad |
| Discrete verdict changes, boresights ≥ 1e-7 rad clear of every boundary | 20 000 of 20 000 qualified | **0** | 0 |
| Discrete verdict changes, no clearance restriction | 20 000 | **0** | reported |
| Allowed solid angle of a fixed three-cone set under rotation | 200 | worst **3.360e-11 sr** on a base of 5.952500685656604 sr | 1e-9 sr |

All **PASS**. The rotated solid angles span
`[5.952500685623002, 5.952500685656723] sr`, a spread of 3.4e-11 sr, which is
the quadrature's own node-placement sensitivity rather than a geometric effect.

**A limitation this measured, not a pass.** Hypothesis found a configuration
where the *discrete* verdict does flip under rotation: a cone of half-angle
1.14e-128 rad with the boresight 2.76e-65 rad from its axis. Rotating a unit
vector perturbs it by of order 1e-16 rad, so a verdict taken closer to a cone
boundary than that is not decidable in double precision, and the property test
is therefore stated for boresights at least 1e-7 rad (0.02 arcsec) clear of
every boundary. The continuous statement — margins are invariant — holds
everywhere and is the one measured in the first row above.

### 4b. Window search

550 km SSO, `i = 97.6 deg`, epoch JD 2461119.5, Sun 45 deg / Earth 10 deg /
Moon 15 deg to the limb, target `unit([0.2, −0.9, 0.3])`. Scan 0 to 11478 s at
20 s (574 samples), boundaries refined by Brent's method.

- Orbital period **5738.993 s** (95.650 min); 3 windows found.
- **Worst `|margin|` at a refined boundary = 8.523e-11 rad**, tolerance 1e-7 →
  **PASS**. Individual boundaries: 3.593e-11, 8.523e-11, 7.993e-11, 4.001e-11
  rad.
- Re-testing the margin directly at 1 s resolution over the scan span (11 461
  samples): **0 disagreements** with the window list → **PASS**.

---

## 5. Low-precision Sun and Moon series

These two routines are a convenience so the package can be exercised end to end;
the geometry, not the ephemeris, is the product. **No numerical ephemeris
(DE440 or equivalent) is available in this environment, so no comparison against
one was made and no absolute-accuracy claim is stated anywhere in this
repository.** What follows is what can be checked against published orbital
constants.

Sun: Vallado (2013) Algorithm 29. Moon: Vallado (2013) Algorithm 31. Sampling:
three years from JD 2461041.5 (2026-01-01), at 0.01 d for the Sun and 0.005 d
for the Moon.

| # | Check | Reference value | Computed | Deviation | Tolerance | Result |
|---|---|---|---|---|---|---|
| 5.1 | Sun max/min declination | ±23.4393 deg (obliquity at J2000) | +23.435849 / −23.435784 deg | 0.003451 / 0.003516 deg | 0.01 deg | PASS |
| 5.2 | Sun ecliptic latitude (zero by construction) | 0 | max 3.181e-15 deg | — | 1e-9 deg | PASS |
| 5.3 | Sun distance at perihelion | 0.98329 AU | 0.983292 AU | 2e-06 AU | 5e-4 AU | PASS |
| 5.3 | Sun distance at aphelion | 1.01671 AU | 1.016710 AU | 0.000000 AU | 5e-4 AU | PASS |
| 5.3 | Perihelion date, 2026 | first week of January | 2.760 days into the year | — | 0–7 d | PASS |
| 5.4 | Tropical year from northward equinox crossings | 365.2422 d | 365.242320 d | 0.000120 d | 0.01 d | PASS |
| 5.5 | Moon mean geocentric distance | 384 400 km | 385 012.805 km | 1.594e-03 relative | 1 % | PASS |
| 5.6 | Moon sidereal period | 27.321582 d | 27.328765 d | 0.007183 d (10.3 min) | 0.01 d | PASS |
| 5.7 | Moon max ecliptic latitude | 5.145 deg orbital inclination | 5.324119 deg | +0.179 deg | [4.945, 5.945] deg | PASS |

Notes on the two that are not tight:

- The Moon's computed mean distance sits **0.16 % above** the published mean.
  The distance comes from the parallax series as `r = R_E / sin(P)`, which is a
  four-term truncation; the perigee/apogee spread it produces, 357 377 km to
  406 019 km, brackets the published extremes but its mean is biased high by
  the truncation.
- The recovered sidereal month is **0.0072 d long**. The series' mean
  longitude rate, 481267.8813 deg per Julian century, corresponds to
  27.321582 d exactly, so this is not a coefficient error: it is the residue of
  the periodic terms over a span that is not an integer number of revolutions,
  plus the truncation.
- The maximum ecliptic latitude exceeds the 5.145 deg mean orbital inclination
  because the latitude series carries three further terms of amplitude 0.28,
  0.28 and 0.17 deg; their partial coincidence with the leading 5.13 deg term
  is what produces the 5.32 deg peak. The published extreme lunar ecliptic
  latitude is about 5.3 deg, which is what the check bounds.

---

## Test suite

`python -m pytest tests/ -q` from `products/P030/`: **159 passed, 0 failed,
0 skipped** in 13.55 s. `python -m ruff check src/ tests/`: **All checks
passed**.

Hypothesis property tests cover: symmetry, boundedness, rotation invariance and
scale invariance of `angular_separation`; rotation invariance of the violation
set and of the per-cone margins; `contains` ⇔ negative margin; cap-area bounds
and symmetry; rotation invariance of the allowed solid angle; and agreement
between the band quadrature and the two-cap closed form.

## What Level 1 means here

All evidence above is analytic, hand-calculated, or a cross-check between two
independent implementations inside this repository. Nothing has been compared
against flight data, an operational scheduling tool, a numerical ephemeris, or a
third-party astrodynamics library. The obvious next step for a Level 2 product
is to reproduce a published instrument's keep-out envelope and to replace the
low-precision series with Skyfield or SPICE directions.
