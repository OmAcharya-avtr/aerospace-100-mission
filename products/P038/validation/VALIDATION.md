# Validation evidence — dopplerkit 0.1.0

Validation level **1** (hand calculations, closed-form known answers and
internal consistency). Nothing here has been compared against measured
tracking data, against a flight-dynamics system, or against a professional
tool. Research-grade and educational.

Every number below was produced by running the scripts in this directory on
2026-10-04 with Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, sgp4 2.27. Raw
stdout is committed beside each script as `<script>_output.txt`, so any drift
is visible in a diff.

```bash
python validation/validate_range_rate.py            # checks 1a, 1b, 1c
python validation/validate_doppler_closedform.py    # checks 2a, 2b, 2c, 3a, 3b
python validation/validate_two_way_relativistic.py  # checks 4a, 4b, 4c
python validation/validate_lighttime.py             # checks 5a, 5b, 5c, 5d, 5e
python validation/validate_tle_pass.py              # supplementary, regression only
python -m pytest tests/ -q                          # 146 tests
```

---

## The sign convention under test

Stated once, identical at every interface in the package, and asserted in
`tests/test_signs.py` (23 assertions) and `tests/test_properties.py`
(12 Hypothesis properties):

| Quantity | Definition | Positive means |
|---|---|---|
| Range-rate `rho_dot` | `d\|rho\|/dt`, `rho = r_target - r_observer` | **receding** (range increasing) |
| One-way Doppler `Delta_f` | `-f_c * rho_dot / c` | **approaching** (up-shift) |
| Two-way Doppler | `-2 G f_up * rho_dot / c` | **approaching** (up-shift) |
| Doppler rate | `-ways * G * f_c * rho_ddot / c` | range-acceleration negative |
| Pre-compensation offset | `-Delta_f` | receding |

Range-rate is **invariant** under reversing the link direction. Reversing
uplink and downlink does not flip any sign; only the carrier changes.

Order retained: `O(beta^1)` with `beta = rho_dot/c`. The `O(beta^2)` terms are
reported (check 4b), never applied.

---

## Derivations used as known answers

### Circular overhead pass geometry

Observer inertially fixed at radius `r_o` on the +x axis; satellite in a
circular orbit of radius `r_s` in the x-y plane, at the zenith at `t = 0`,
mean motion `n = sqrt(mu/r_s^3)` (Vallado, 4th ed., two-body circular orbit):

```
r_obs = (r_o, 0, 0)
r_sat = r_s (cos n t, sin n t, 0)
v_sat = r_s n (-sin n t, cos n t, 0)
```

**Range.** `rho^2 = (r_s cos n t - r_o)^2 + (r_s sin n t)^2 = r_s^2 + r_o^2 - 2 r_s r_o cos n t`.

**Range-rate.** `rho_dot = rho . v_sat / rho`. The numerator is

```
(r_s cos n t - r_o)(-r_s n sin n t) + (r_s sin n t)(r_s n cos n t)
  = -r_s^2 n sin cos + r_o r_s n sin + r_s^2 n sin cos
  = r_o r_s n sin n t
```

so `rho_dot = r_s r_o n sin(n t) / rho`. At `t = 0` this is exactly zero:
closest approach and the Doppler zero crossing coincide **by construction**
for this geometry, which is why check 3 uses two independent numerical
methods rather than relying on the formula.

**Range-acceleration.** Differentiating:
`rho_ddot = r_s r_o n^2 cos(n t) / rho - rho_dot^2 / rho`.
At `t = 0`, `rho = r_s - r_o` and `rho_dot = 0`, so
`rho_ddot(0) = r_s r_o n^2 / (r_s - r_o)`, the maximum over the pass.

**At the geometric horizon.** Zero elevation requires the line of sight
perpendicular to `r_obs`, i.e. `cos n t = r_o / r_s`. Then
`sin n t = sqrt(1 - r_o^2/r_s^2) = rho / r_s` with `rho = sqrt(r_s^2 - r_o^2)`,
and the range-rate collapses:

```
rho_dot(t_h) = r_s r_o n (rho/r_s) / rho = r_o n     EXACTLY
rho_ddot(t_h) = (r_o^2 n^2 - rho_dot^2)/rho = 0      EXACTLY
```

Both are used as zero-tolerance known answers (check 1c). The second one is
why the Doppler rate is zero, not negative, at the two ends of the pass in
`screenshots/pass_doppler_profile.png`.

### Classical Doppler

`f_rx = f_tx (1 - rho_dot/c)`, hence `Delta_f = -f_c rho_dot / c`, first order
in v/c. Peak one-way shift over an overhead pass is therefore
`-f_c r_o n / c` and the Doppler rate at closest approach is
`-f_c r_s r_o n^2 / (c (r_s - r_o))`.

### Two-way coherent transponder

```
f_ground_rx = G f_up (1 - rho_dot_up/c)(1 - rho_dot_down/c)
Delta_f_2way = -G f_up [(rho_dot_up + rho_dot_down)/c - rho_dot_up rho_dot_down/c^2]
```

At one epoch (`rho_dot_up = rho_dot_down`) with `G = 1`, the first-order term
is **exactly** twice the one-way shift. Reference for the two-way
formulation: Moyer, *Formulation for Observed and Computed Values of Deep
Space Network Data Types for Navigation*, JPL Deep Space Communications and
Navigation Series Monograph 2, 2000.

### Relativistic expansion

```
f_rx/f_tx = sqrt(1 - v^2/c^2) / (1 + rho_dot/c)
          = 1 - rho_dot/c + (rho_dot/c)^2 - v^2/(2c^2) + O(beta^3)
```

The package keeps the `-rho_dot/c` term. The `O(beta^2)` remainder
`(rho_dot/c)^2 - v^2/(2c^2)` is reported by
`relativistic_fraction_second_order`. `v` is the **transmitter** inertial
speed, so the `-v^2/(2c^2)` part is transmitter proper-time dilation and does
not vanish at closest approach.

### Light time

Down-leg: `c tau = |r_sat(t_r - tau) - r_sta(t_r)|`, solved by fixed-point
iteration from `tau = 0`. The iteration map has derivative `-rho_dot/c`, so
the contraction factor is `|rho_dot|/c`; check 5c measures it.

---

## Evidence table

Baseline throughout, unless stated: circular overhead pass at 500 km altitude
above the WGS-84 equatorial radius, `mu` = 3.986004418e14 m^3/s^2, observer at
6378137 m, orbit radius 6878137 m, mean motion 1.106783446335e-03 rad/s,
orbital period 94.616300 min, orbital speed 7612.608173 m/s (beta = 2.539293e-05),
horizon-to-horizon 693.2639 s, carrier 2.2 GHz unless a sweep is named.

| # | Check | Reference for the equation | Computed | Expected | Tolerance | Result |
|---|---|---|---|---|---|---|
| 1a | Central-difference range-rate vs analytic dot product, worst over 4 epochs at h = 0.0390625 s | Vallado 4th ed. range-rate observation equation; 2nd-order central difference | 8.4354e-05 m/s error | 0 | 1e-4 m/s | **PASS** |
| 1a | Observed convergence order, h = 20 s down to h = 0.039 s | truncation `(h^2/24) rho'''` | 2.000 (every halving quarters the error) | 2 | ±0.01 | **PASS** |
| 1a | Round-off floor | `eps*rho/h`, eps = 2.22e-16, rho ~ 1e6 m | error stops falling near h ≈ 0.01 s, best ≈ 1.3e-07 m/s; rises to 7.9e-07 m/s at h = 1e-3 s | — | reported | **reported, not a pass/fail** |
| 1b | Vector dot-product code vs closed form, worst over 11 epochs — range | same | 9.0804e-09 m | 0 | 1e-6 m | **PASS** |
| 1b | — range-rate | same | 6.6393e-11 m/s | 0 | 1e-8 m/s | **PASS** |
| 1b | — range-acceleration | same | 4.9027e-13 m/s² | 0 | 1e-9 m/s² | **PASS** |
| 1c | `rho(0) = r_s - r_o` | derivation above | 500000.000000 m | 500000.000000 m | 1e-6 m | **PASS, diff 0** |
| 1c | `rho_dot(0) = 0` | derivation above | 0.000000 m/s | 0 | exact | **PASS, diff 0** |
| 1c | `rho_dot(t_h) = r_o n` | derivation above | 7059.216450 m/s | 7059.216450 m/s | 1e-6 m/s | **PASS, 2.73e-12** |
| 1c | `rho_ddot(t_h) = 0` | derivation above | -1.42e-14 m/s² | 0 | 1e-9 m/s² | **PASS** |
| 1c | `rho_ddot(0) = r_s r_o n²/(r_s-r_o)` | derivation above | 107.478098 m/s² | 107.478098 m/s² | 1e-9 m/s² | **PASS, diff 0** |
| 1c | `rho(t_h) = sqrt(r_s²-r_o²)` | tangent line of sight | 2574516.847877 m | 2574516.847877 m | 1e-6 m | **PASS, diff 0** |
| 2a | Library Doppler vs the closed form, 4 carriers × 9 epochs | classical Doppler relation | worst relative difference 3.827e-16 | 0 | 1e-9 relative | **PASS** |
| 2b | Peak one-way Doppler = `f_c r_o n / c`, 2.2 GHz | classical Doppler + geometry | 51803.425255 Hz | 51803.425255 Hz | 1e-6 relative | **PASS, 1.46e-11 Hz** |
| 2b | Same, 401 MHz / 8.4 GHz / 26 GHz | — | 9442.351603 / 197794.896430 / 612222.298472 Hz | identical to hand value | 1e-6 relative | **PASS** |
| 2c | Doppler rate at closest approach = `-f_c r_s r_o n²/(c(r_s-r_o))`, 2.2 GHz | derivation above | -788.718357 Hz/s | -788.718357 Hz/s | 1e-9 relative | **PASS, diff 0** |
| 3a | Doppler zero crossing vs time of closest approach, independent methods (brentq on `rho_dot`; bounded Brent minimisation of `rho`) | — | crossing +0.000000000 s, TCA -0.000000333 s, difference 3.333e-07 s | 0 | 1e-6 s (the minimiser's own `xatol`) | **PASS** |
| 3a | Same, against a sampled profile's integration step | — | within one step at 30 / 10 / 1 / 0.1 s sampling | ≤ 1 step | 1 step | **PASS** |
| 3b | Zero crossing is carrier-independent, 401 MHz to 26 GHz | the carrier cancels in `Delta_f = 0` | spread 0.000e+00 s | 0 | exact | **PASS** |
| 4a | Two-way is exactly twice one-way, G = 1, 5 range-rates × 4 carriers | Moyer 2000, composed one-way legs | bit-exact equality in all 20 cases | exact | **zero** | **PASS** |
| 4a | Same, Hypothesis sweep over 150 examples per property | — | bit-exact | exact | **zero** | **PASS** |
| 4b | O(beta²) special-relativistic term at closest approach, 2.2 GHz | expansion above | **-0.709281 Hz** (fraction -3.224004e-10) | — | reported | **reported** |
| 4b | Same at 401 MHz / 8.4 GHz / 26 GHz | — | -0.129283 / -2.708163 / -8.382410 Hz | — | reported | **reported** |
| 4b | O(beta²) SR term at the geometric horizon, 2.2 GHz | — | +0.510535 Hz, i.e. 9.86e-06 of the 51803 Hz classical shift | — | reported | **reported** |
| 4b | Two-way classical cascade term (dropped), 2.2 GHz at the horizon | `+f rho_dot_up rho_dot_down / c²` | **+1.219816 Hz** | — | reported | **reported** |
| 4b | Static gravitational term (order of magnitude), 2.2 GHz downlink | `(mu/c²)(1/r_rx - 1/r_tx)` | **+0.111205 Hz** (fraction +5.054774e-11) | — | reported | **reported** |
| 4c | Two-way with light-time-separated legs vs the exactly-2x instantaneous form, 2.2 GHz | Moyer 2000 | worst departure **2.6309 Hz**, 2.539e-05 of the peak two-way excursion | — | reported | **reported** |
| 4c | Hand explanation of that departure: two-way Doppler rate × light time | -1577.4367 Hz/s × 1.667820 ms | -2.6309 Hz | -2.6309 Hz measured | 1e-4 Hz | **PASS, diff 0.0000 Hz** |
| 5a | Static pair light time = `rho/c` | definition | 1 iteration, residual exactly 0 | 1 iteration, 0 | exact | **PASS** |
| 5b | Light-time iteration over the pass, 9 epochs × 2 legs | Moyer 2000 fixed-point solution | **max 3 iterations**, worst residual **3.353e-08 m**, all converged | — | 1e-6 m | **PASS** |
| 5c | Measured contraction rate vs predicted `\|rho_dot\|/c` at peak range-rate | derivative of the iteration map | measured 2.355e-05, predicted 2.354701e-05, ratio **1.000** | 1 | 1 % | **PASS** |
| 5d | Two-way round trip equals the sum of its legs, 5 epochs | — | consistent, 2–3 iterations per leg | — | 1e-6 m of range | **PASS** |
| 5e | Light-time range minus instantaneous range | — | up to **60.62 m** over the pass; 0.000149 m at closest approach | — | reported | **reported** |
| S1 | Station rotation velocity, equator | `omega_E x r`, WGS-84 | 465.1 m/s | 465.1 m/s | 1e-9 relative | **PASS** |
| S2 | Doppler error from forcing the station velocity to zero, ISS pass, 2.2 GHz | — | **2086.97 Hz** worst at 100 s sampling (`validate_tle_pass.py`); **2088.40 Hz** at 5 s sampling (`example_tle_pass.py`) | — | reported | **reported** |
| S3 | Pre-compensation with the sign flipped, 2.0255 GHz uplink | — | 95390.07 Hz residual vs 47694.47 Hz uncompensated — **exactly 2.0000x worse than doing nothing** | 2x | reported | **reported** |
| S4 | Pre-compensation with the stated sign, same case | — | 1.1231 Hz residual (the O(beta²) term) | `f beta²` | reported | **reported** |

---

## The checks that are uncomfortable

These are the credible ones.

**The first-order pre-compensation does not cancel exactly.** It leaves a
residual of `-f beta^2`, which is 1.12 Hz at 2.0255 GHz for a 500 km LEO
(S4 above). `precompensation_offset_hz(..., exact=True)` inverts the classical
relation and cancels to machine precision, and the test suite asserts both.
A reader who needs the residual to be zero must use the exact form.

**The exactly-twice identity is a single-epoch statement.** With finite light
time the two legs are evaluated at different epochs and two-way Doppler is
*not* exactly twice the instantaneous one-way shift — it departs by 2.63 Hz at
closest approach for this pass (check 4c). Both facts are in the package:
`two_way_doppler_hz` is the single-epoch form and
`two_way_doppler_two_leg_hz` is the separated form. Choosing the wrong one is
a 2.6 Hz error at S-band.

**Bit-exactness fails in the subnormal float range.** Hypothesis found that at
`rho_dot = 2.2250738585072014e-308 m/s` and `f_c = 1 MHz` the one-way shift
underflows to a subnormal, where `2*x` is not exactly representable, so the
two-way result differs from twice the one-way result in the last bit. This is
IEEE-754 gradual underflow, not a defect in the formulation, and it is
recorded as its own test
(`tests/test_properties.py::test_doubling_identity_degrades_in_the_subnormal_range`)
rather than papered over. No physical range-rate is 1e-308 m/s.

**The relativistic terms are not the largest thing being neglected.** At
2.2 GHz the whole `O(beta^2)` budget is under 1.3 Hz. The station rotation
velocity, if dropped, costs about 2087 Hz (S2); the frame reduction's own UT1
neglect is worth about 0.22 Hz (computed in `frames.py`, not measured); and
the tropospheric delay rate, which this package does not model at all, is
larger than any of the relativistic terms at L- and S-band. Reporting the
relativistic magnitudes is useful precisely because it shows they are not
where the error is.

**The TLE path's Doppler rate is zero.** `satellite_state_teme` leaves the
acceleration at zero, so a profile built from `sgp4` states carries only the
kinematic part of the range acceleration. The CLI prints that caveat in the
`tle` output. The closed-form model is the reference for Doppler rate.

**The TLE numbers are regression values, not an external reference.** Check
S2 and the values pinned in `tests/test_frames.py` were produced by this
repository. They pin the frame reduction so a change is visible; they do not
validate it against an independent ephemeris, and no claim is made that they
do. Validating the frame reduction against an external source is out of scope
at Level 1 and is listed as a limitation in the README.

---

## Test suite

`python -m pytest tests/ -q` → **146 passed, 0 failed, 0 skipped, 0 errors**
(9.20 s, counts taken from the JUnit XML).

| File | Tests | What it covers |
|---|---|---|
| `test_geometry.py` | 13 | hand-calculated range/range-rate/range-acceleration, validation, finite differences |
| `test_signs.py` | 23 | every sign in the package, including link reversal and pre-compensation direction |
| `test_doppler.py` | 19 | hand-calculated Doppler, linearity, input validation, relativistic magnitudes |
| `test_analytic.py` | 18 | closed-form known answers, vector/closed-form cross-check, convergence order |
| `test_lighttime.py` | 13 | convergence, residual reporting, leg ordering, non-convergence reporting |
| `test_passprofile.py` | 12 | profile arrays, event locators, carrier independence of the zero crossing |
| `test_frames.py` | 21 | geodetic conversion, GMST, station velocity, sgp4 integration, elevation |
| `test_properties.py` | 12 | Hypothesis properties, 150 examples each, plus the subnormal exception |
| `test_cli.py` | 15 | exit codes, JSON payloads, every argument-validation path |
