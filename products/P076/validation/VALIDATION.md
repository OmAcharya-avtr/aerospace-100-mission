# Validation evidence — pointahead 0.1.0

Validation level 2. Every number below was produced by a script in this
directory, executed in this build, and is reproducible by running that script.
Each script writes its raw output beside itself as
`validation/<script>_output.txt`. Where a check failed, it is recorded as
failed; no tolerance in this repository was widened to make a check pass, and
the one place a derived constant was corrected is stated with both values.

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use. No output of this package is suitable for an
operational pointing decision.

## Environment the numbers were measured in

Recorded by `validation/_bootstrap.py:environment_lines` in the header of every
output file, read from `importlib.metadata`, not declared.

| component | version |
|---|---|
| python | 3.13.17 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| pyyaml | 6.0.3 |
| sgp4 | 2.27 |
| matplotlib | 3.10.8 |
| pytest | 8.4.2 |
| hypothesis | 6.140.3 |
| ruff | 0.14.2 |
| **astropy** | **not installed** |
| cpu cores | 2 |

## Deviations from the batch specification

### 1. `astropy` cross-check replaced, and nothing was taken from `astropy`

The Batch 08 specification for P076 requires the aberration terms to be
"cross-checked against `astropy`'s independent implementation to a stated
tolerance".

**`astropy` is not installed in the environment this build was measured in and
installing it was not permitted.** The pinned interpreter for this build is
frozen at `shared/pinned_env_2026-10-24.txt` and `astropy` is not in it;
concurrent `pip install` in this container mutated the shared interpreter
mid-build on 2026-10-16, which is why installing is forbidden. The absence is
recorded as a measurement, not a claim: `importlib.metadata.version("astropy")`
raises `PackageNotFoundError`, and the header of every output file in this
directory prints `astropy : not installed`.

**No number anywhere in this repository came from `astropy`.** `astropy` is
named in the README alternatives table, where it belongs — it is the correct
tool for aberration itself, and this package's contribution is the
crosslink-specific geometry and the attributed budget, not the aberration
transformation.

The substitute is **three independent exact implementations** of the
relativistic aberration transformation, written from three different
derivations, plus the iterated light-time reference:

| route | derivation |
|---|---|
| `angle_relation` | the trigonometric relation `cos(theta') = (cos theta - beta) / (1 - beta cos theta)` |
| `momentum_boost` | a Lorentz boost of the photon four-momentum `k = (1, p)` |
| `event_worldline` | a Lorentz transformation of the spacetime coordinates of two events on the photon worldline |

Agreement between them is an implementation check and is stated as such. The
*physics* is checked by the closed-form properties of the transformation
(section 2 below), by the geometric invariants, and by the iterated light-time
reference. This is weaker than an independent third-party implementation would
be and that is the honest cost of the substitution.

### 2. Only one point in each declared dependency range was executed

`templates/PRODUCT_BUILD_GUIDE.md` requires the suite to be run at both ends of
every declared dependency range. This build could not do that: installing
anything into the shared container was forbidden for the reason above, so every
number here comes from exactly the versions in the table at the top. The ranges
in `pyproject.toml` bound the minor version the numbers were measured against;
they are not evidence that both ends were exercised. This is recorded in the
README under Limitations as well as here.

## Test suite

Counted from junit XML, never from pytest stdout, because a suite that collects
zero tests prints success and exits zero.

```
python -m pytest tests/ -q --junitxml=junit.xml
```

| quantity | value |
|---|---|
| `testsuite` declared tests | 272 |
| `<testcase>` elements counted | 272 |
| failures | 0 |
| errors | 0 |
| skipped | 0 |

`ruff check src/ tests/ examples/ validation/` — clean, no findings.

## Validation checks: 105 passed, 1 failed

The failure is real, is diagnosed below, and is left in place.

### 1. Aberration: three exact routes cross-checked (`validate_aberration.py`)

| check | reference | result | tolerance |
|---|---|---|---|
| pairwise disagreement, 2000 random directions, `\|beta\|` in [1e-09, 0.95] | the other two exact routes | worst 1.054369e-15 rad | 1e-13 rad |
| pairwise disagreement, 2000 draws, `\|beta\|` in [1e-06, 1e-04] | the other two exact routes | worst 4.126163e-16 rad | 1e-15 rad |
| collinear direction unchanged | the transformation itself | worst 0 rad | 1e-16 rad |
| transverse displacement equals `arcsin(beta)` | closed form, six values of `beta` | worst 2.710505e-16 relative | 1e-14 |
| inverse boost round-trip | identity | worst 9.854756e-16 rad | 1e-14 rad |
| two collinear boosts compose by `(b1+b2)/(1+b1 b2)` | relativistic velocity addition | worst 2.775558e-16 rad | 1e-14 rad |
| every route refuses `\|beta\| = 1` | — | 4 of 4 refused | — |
| first-order aberration error at orbital speed | exact route, oblique direction | 1.588957e-10 rad | 1e-09 rad |

The first-order form's error scaling was measured, not assumed: `beta^3 / 2`
for a direction exactly transverse to the boost (ratio to `beta^3/2` of 1.000000
at `beta = 1e-05`, 0.997537 at `beta = 0.1`) and `O(beta^2)` for an oblique one.
The module docstring originally claimed `O(beta^2)` for both; the transverse
case is cubic because the second-order term vanishes there, and a test written
for a quadratic ratio measured 1000 per decade instead of 100 and failed.
Pinned as `test_regression_first_order_aberration_error_is_cubic_when_transverse`.

### 2. Two-body propagation against an independent integration (`validate_kepler.py`)

| check | reference | result | tolerance |
|---|---|---|---|
| universal variables vs `scipy.integrate.solve_ivp` DOP853, rtol 1e-13, 40 cases | independent numerical integration of the same equation of motion | worst relative position 1.013e-13, velocity 1.254e-13 | 1e-10 relative |
| specific orbital energy | exact two-body invariant | worst drift 1.796569e-15 relative | 1e-12 relative |
| specific angular momentum | exact two-body invariant | worst drift 5.192154e-16 relative | 1e-12 relative |
| `\|f gdot - fdot g - 1\|` | exact identity | worst 4.440892e-16 | 1e-12 |
| forward-and-back round trip | identity | worst 1.808407e-15 relative | 1e-12 relative |
| one full period at four altitudes | identity | worst 1.979e-08 m | 1e-06 m |

Altitudes 400 to 20000 km, speeds 0.85 to 1.25 of circular, intervals 60 to
3600 s.

### 3. Light-time solution against an exact closed form (`validate_lighttime.py`)

| check | reference | result | tolerance |
|---|---|---|---|
| transmit light time, remote 1 light-second away at 0.6 c transverse | `tau = 1 / sqrt(1 - 0.36) = 1.25` s exactly | 1.250000000000000 s | 1e-14 relative |
| iterated vs the exact quadratic root, 400 straight-line cases, both directions | closed-form positive root | worst 0.000000e+00 relative | 1e-14 relative |
| transmit residual at the converged solution, three named geometries | the light-time equation | 0.000e+00 s on all three | 1e-17 s |
| geometric light time bracketed by the two solutions, 60 seeded crosslinks | — | 0 violations | 0 |
| contraction factor per iteration | `\|v_remote . n\| / c` | within 1 per cent on all three geometries | 1 per cent |
| gap `tau_tx - tau_rx` | `2 (v_remote . n) tau / c` | worst 7.557e-10 relative | 1 per cent |

At `beta = 0.6` the fixed-point map contracts by only about 0.6 per step and the
30-iteration default cap does **not** reach the 1e-15 s default tolerance. The
solver reports `converged = False` rather than claiming otherwise, and that is
asserted by test. An orbital crosslink has `beta` near 2.521032469246712e-05 and converges in
three iterations.

### 4. The closed-form baseline scored against the iterated reference (`validate_pointahead.py`)

This is the comparison the specification asks for. Three error metrics are
reported because none alone is honest.

| geometry | range / km | closed form / urad | reference / urad | magnitude diff / rad | closed-loop offset diff / rad | open-loop transmit diff / rad |
|---|---|---|---|---|---|---|
| leo_crossing | 2085.600 | 45.104546 | 45.104546 | 1.608483e-14 | 1.855254e-14 | 2.044581e-10 |
| leo_trailing | 2085.600 | 15.069535 | 15.069535 | 1.429412e-14 | 1.477483e-14 | 2.347832e-10 |
| geo_relay_to_leo | 35840.014 | 35.912474 | 35.912474 | 1.190121e-13 | 1.370754e-13 | 7.385140e-10 |

Over a seeded family of 300 crosslinks, ranges 494.5 to 8268.0 km, point-ahead
angles 0.705636 to 97.774864 urad:

| quantity | median | p95 | max |
|---|---|---|---|
| closed-loop offset difference / rad | 2.463345e-14 | 5.901257e-14 | 7.339728e-14 |
| open-loop transmit difference / rad | 3.791155e-10 | 6.488676e-10 | 6.979398e-10 |
| closed-loop difference / the angle | 8.032747e-10 | 1.014754e-09 | 4.826409e-09 |
| float64 direction resolution / rad | 4.394229e-15 | 1.574783e-14 | 4.564142e-14 |

The open-loop error exceeded the closed-loop error on 300 of 300, as it must:
the open-loop metric also contains the common-mode line-of-sight error that a
received-beam tracker removes.

**Baseline verdict.** The closed-form first-order baseline wins. Its worst
disagreement with the iterated reference on the point-ahead offset is
7.339728e-14 rad over 300 crosslinks, which is 4.826409e-09 of the angle and is
of the same order as the float64 direction resolution of the geometry (median
4.394229e-15 rad, p95 1.574783e-14 rad, max 4.564142e-14 rad, with 37 of the 300
closed-loop errors at or below their own floor). It needs no trajectory model
and no
iteration, and it is the cheaper of the two by the factor in the
unstable-timings table below. The reference is not thereby useless: it is the only way to obtain the light-time budget lines at all, it is
what makes the numerical sensitivity checks possible, and it is required for the
absolute transmit direction when a terminal cannot close the loop on a received
beam.

**A result worth stating because it is not obvious.** The closed form does not
degrade appreciably with range, even though the trajectory curvature over the
light time grows with it. Median closed-loop error by range band, 50 crosslinks
per band: 1.157432e-14 rad at 494 to 2909 km rising only to 3.687112e-14 rad at
6657 to 8268 km, while the error *as a fraction of the angle* is flat at
6.690292e-10 to 8.747972e-10. The reason is a cancellation: the transverse velocity at the
advanced epoch exceeds the nominal one by `a_perp tau` and at the retarded epoch
falls short by the same amount, and the point-ahead offset is their sum, so the
first-order curvature terms cancel exactly.

### 5. The attributed budget, every analytic line against a propagation (`validate_budget.py`)

Each analytic line is checked by perturbing exactly the input it is declared
for, re-running the whole reference pipeline, and differencing the point-ahead
offsets. Four lines x 33 geometries x 4 iteration caps x 3 ephemeris ages =
1584 combinations.

| line | pass | fail | median tightness | worst |
|---|---|---|---|---|
| `relative_velocity_knowledge` (equality, 1 per cent) | 396 | 0 | 1.315466e-07 | 2.700381e-06 |
| `ephemeris_staleness_velocity` (upper bound) | 396 | 0 | 2.409995e-02 | 6.783312e-01 |
| `ephemeris_staleness_geometry` (upper bound) | 396 | 0 | 7.978914e-02 | 9.462824e-01 |
| `light_time_iteration_residual` (upper bound) | 360 | **36** | 9.923261e-01 | 1.000000e+00 |
| **total** | **1548** | **36** | | |

#### FAILED CHECK: `every analytic line verifies on every geometry and configuration measured`

36 of 1584 combinations failed. Every one is the `light_time_iteration_residual`
line at **iteration cap zero**, on 12 of the 33 geometries, at all three
ephemeris ages. Worst `numeric / analytic` ratio 1.7564. Both sides of every
failing comparison are of order 1e-14 rad.

Diagnosis. At iteration cap zero the receive and transmit light-time truncation
errors have opposite signs and nearly cancel in the point-ahead offset, so the
*predicted* error is smaller than the arithmetic noise of the four unit
directions it is measured from. Both sides are then quantisation patterns at the
float64 direction resolution (1.029436e-14 rad for the `leo_trailing` geometry)
and the bound holds or fails at random.

The honest statement: **the light-time iteration budget line cannot be verified
at iteration cap zero in float64 for these geometries**, and at one or more
iterations it verifies on every combination measured. The engineering
consequence is the opposite of alarming — at cap zero the line is 1e-14 rad,
which is nine orders of magnitude below the dominant term of every realistic
budget. The recommendation in the README is to iterate at least once, and the
default in every shipped case is one.

The float64 floor constant was corrected once, from `4 eps R / rho` to
`8 sqrt(3) eps R / rho`, when the first derivation was found to ignore the three
components of each direction and the outer subtraction of two offsets:
2.9717261752352168e-15 rad against a measured 9.46444779611722e-15 rad on
`leo_trailing` at cap zero. **It has not been widened again to make these 36
combinations pass.** Pinned as
`test_regression_noise_floor_counts_components_and_the_outer_subtraction`.

#### Two analytic lines were wrong when first written

Both were found by this check and by nothing else. Both are pinned as named
regression tests with literal inputs.

| line | first formulation | predicted | measured | factor |
|---|---|---|---|---|
| `ephemeris_staleness_velocity` | `2 \|a_rel_perp\| dt / c` | 2.087363736690764e-22 rad | 2.3807507009717773e-08 rad | 1.140554e+14 |
| `ephemeris_staleness_geometry` | cubic coefficient `dt^3 / 6` | half the bound | — | 2 |

The first is the more interesting of the two. For two terminals at **equal
radius** under two-body gravity, `a_rel = -(mu / r^3) d` with `d` the separation
vector, so the relative acceleration is **exactly along the line of sight** and
its transverse component is zero to rounding — measured transverse fraction
4.263e-16 for `leo_crossing` and 4.084e-16 for `leo_trailing`. A
first-order-only staleness model therefore predicts **zero** staleness error for
the commonest crosslink geometry there is, a same-altitude pair. The jerk term
`|adot_rel_perp| dt^2 / 2` supplies the whole line there.

The second is a plain arithmetic slip with a specific trap in it. The stale
position error is `r(t - dt) + v(t - dt) dt - r(t) = -a dt^2 / 2 + adot dt^3 / 3`:
the `dt^3 / 3` is `-1/6` from the position expansion plus `+1/2` from carrying
the stale velocity forward, and is twice the `dt^3 / 6` a plain Taylor series
suggests.

#### The dominant term changes with ephemeris age

`leo_crossing`, 5 cm/s per-axis velocity knowledge, one light-time iteration,
1550 nm / 25 mm beam:

| age / s | total / urad | dominant term | share | pointing loss / dB |
|---|---|---|---|---|
| 0 | 4.717309e-04 | relative_velocity_knowledge | 1.0000 | 4.962714e-09 |
| 1 | 4.724721e-04 | relative_velocity_knowledge | 0.9984 | 4.978322e-09 |
| 10 | 2.688034e-03 | ephemeris_staleness_velocity | 0.9842 | 1.611388e-07 |
| 30 | 2.387950e-02 | ephemeris_staleness_velocity | 0.9971 | 1.271690e-05 |
| 60 | 9.627913e-02 | ephemeris_staleness_velocity | 0.9892 | 2.067259e-04 |
| 300 | 2.963074e+00 | ephemeris_staleness_velocity | 0.8035 | 1.958013e-01 |
| 900 | 5.222055e+01 | ephemeris_staleness_geometry | 0.9119 | 6.081540e+01 |
| 3600 | 3.066995e+03 | ephemeris_staleness_geometry | 0.9937 | 2.097765e+05 |
The two rows at 900 and 3600 s are past the validity of the linearisation they
are computed from; they are shown because the budget prints them and a reader
needs to see where the model stops meaning anything, not because the numbers
are trustworthy. The bound tightness degrades with age for the same reason, and the
worst tightness in the 1584-combination sweep is in the table above.

#### The beam is what makes the total a result

`leo_crossing` at 300 s, total 2.963074 urad, identical in every row:

| beam | divergence half-angle / urad | fraction of beam | pointing loss / dB | headroom |
|---|---|---|---|---|
| 1550 nm, 10 mm waist | 49.338032 | 6.005659e-02 | 3.132821e-02 | 5.6498e+00 |
| 1550 nm, 25 mm waist | 19.735213 | 1.501415e-01 | 1.958013e-01 | 2.2599e+00 |
| 1064 nm, 75 mm waist | 4.515756 | 6.561634e-01 | 3.739713e+00 | 5.1711e-01 |
| 1064 nm, 150 mm waist | 2.257878 | 1.312327e+00 | 1.495885e+01 | 2.5855e-01 |

The pointing loss spans more than two orders of magnitude for one unchanged
error.

### 6. Geometric invariants (`validate_invariants.py`)

| invariant | result | criterion |
|---|---|---|
| angle vanishes for purely radial relative motion, 120 crosslinks | closed form worst 0.000000e+00 rad, reference worst 6.729973e-16 rad | 1e-18 / 1e-15 rad |
| negating the relative velocity leaves the magnitude unchanged | worst 0.000000e+00 relative | 1e-14 relative |
| negating the relative velocity negates the offset | worst 1.681746e-15 relative | 1e-14 relative |
| rotational equivariance, 240 comparisons | worst absolute angle residual 7.532847e-15 rad, worst offset residual 7.740676e-15 rad, 0 of 240 above their own float64 floor | each geometry's own `direction_noise_floor_rad` |
| reverse-link magnitude symmetry, 120 crosslinks | worst 2.535478e-08 relative | 1e-04 relative |
| linearity in transverse relative speed over seven decades | 0 of 7 points outside `max(1e-09 relative, the floor)` | — |
| range independence of the angle, factor of 8 in range | constant to nine significant figures (45.104545702 urad throughout) | — |

**Two acceptance criteria in this script were wrong when first written**, and
both are recorded because the failure mode is instructive. The rotational
equivariance check used a pure 1e-12 relative tolerance and failed at a worst
relative residual of 3.7e-09 on a crosslink whose point-ahead angle is
2.045329e-07 rad — a residual of 7.7e-16 rad, three hundred times *below* the
2.3e-13 rad float64 direction resolution of that geometry. The linearity check
used a pure 1e-09 relative tolerance and failed at `v_perp = 1e-03` m/s, where
the angle is 6.7e-12 rad and the residual is 3.3e-17 rad. A relative criterion
on a quantity whose scale varies by two orders across a family is the wrong
criterion. Both now carry an absolute floor derived from the arithmetic and both
residuals are printed.

### 7. Frame dependence of the attribution (`validate_frames.py`)

This is the claim the attribution rests on and it cuts both ways, so both halves
are measured. One physical configuration is viewed from six inertial frames,
reached by adding a constant velocity to both terminals.

| geometry | spread of the total / rad | spread of the light-time lead / rad | ratio | spread of the total / the angle |
|---|---|---|---|---|
| leo_crossing | 3.858295e-14 | 4.510455e-05 | 1.169e+09 | 8.554115e-10 |
| leo_trailing | 1.249001e-14 | 6.713697e-05 | 5.375e+09 | 8.288251e-10 |
| geo_relay_to_leo | 6.821943e-14 | 1.054346e-04 | 1.546e+09 | 1.899603e-09 |

In the local terminal's rest frame both aberration lines are **exactly** zero
and the light-time lead carries the whole 45.104546 urad. In the remote
terminal's rest frame the lead is zero and the two aberration lines carry it.
The invariance of the total is first order in `beta`, and the second-order
residual was measured by boosting along a fixed axis: `change / beta^2` is
2.782630e-05 at a 10 km/s boost and 2.717512e-05 at a 10000 km/s boost.

**Consequence for a reader.** A light-time-lead or aberration number taken out
of this package without the frame it was computed in is meaningless. Every
decomposition this package returns carries its frame label, and
`Crosslink.boosted` forces the linear trajectory model because a boosted
Keplerian state is not a Keplerian state.

### 8. Working directory and absolute paths (`probe_working_directories.py`)

| check | result |
|---|---|
| 17 scripts x 5 working directories (filesystem root, temporary directory, product root, `validation/`, `src/`) | 85 of 85 invocations exited 0 |
| absolute-path prefixes in any tracked text file | 74 files scanned, 0 matches |

`python -I` is deliberately **not** used for the probe: `-I` strips the script's
own directory from `sys.path`, makes `import _bootstrap` fail, and produces a
false failure indistinguishable from the defect being probed for.

### 9. Command-line interface (`validate_cli.py`)

Eight success invocations all exit 0, with their full output recorded. Three
error paths — a missing case file, an unknown scenario, no arguments — all exit
non-zero, name the problem in `stderr`, and print no traceback.

### 10. Case files round-trip (`regenerate_cases.py`)

The three shipped YAML cases are generated from `pointahead.scenario` and
serialised by `pointahead.cases.dump_case`. Regenerating leaves all three byte
for byte unchanged, and all three reload through the loader with a range
difference of 0.000000e+00 m.

## Defects found in this build

Eleven, all pinned in `tests/test_regressions.py` with literal inputs, because
`.hypothesis/` is gitignored and a counterexample that is not pinned is lost on
a fresh clone. Every number in the table is re-measured from those literal
inputs by `validation/validate_regressions.py`, so it traces to an output file
rather than to a test assertion.

| # | found by | nature | defect |
|---|---|---|---|
| 1 | Hypothesis | **code** | universal-variable Kepler iteration did not converge over 88 revolutions of a 37.9 s orbit (`radius=400000.0, factor=0.6, dt=3354.9375`). Fixed by exact reduction modulo the orbital period before iterating. |
| 2 | numerical sensitivity check | **code** | `ephemeris_staleness_velocity` omitted the jerk term and predicted 2.087363736690764e-22 rad against a measured 2.3807507009717773e-08 rad, because the relative acceleration of an equal-radius pair is exactly radial. |
| 3 | numerical sensitivity check | **code** | `ephemeris_staleness_geometry` used `dt^3 / 6` where the correct coefficient is `dt^3 / 3`; the bound sat a factor of two below the measurement. |
| 4 | validation script | criterion | float64 direction-resolution constant derived as 4 instead of `8 sqrt(3)`, giving 2.9717261752352168e-15 rad against a measured 9.46444779611722e-15 rad. |
| 5 | validation script | prediction | light-time contraction factor and light-time gap predicted from the *relative* radial velocity; the light-time equation holds the local position fixed, so it is the *remote* radial velocity. Wrong by a factor of two on one geometry and by twelve orders of magnitude on another. |
| 6 | pytest | documentation | hand-written literal `5.003461712e-05` for `15000 / 299792458`, which is `5.003461427972281e-05`. |
| 7 | pytest | documentation | aberration module docstring claimed `O(beta^2)` for the first-order error; it is `O(beta^3)` for a transverse direction. |
| 8 | Hypothesis | criterion | `arcsin(beta)` aberration property at `beta = 1e-12`: residual 8.113064303823435e-18 rad, 8.113064e-06 relative, below the 2.220446049250313e-16 float64 unit-vector resolution. |
| 9 | Hypothesis | criterion | three further relative-only tolerances falsified at angles below 1e-08 rad with absolute residuals below 2e-17 rad (`test_regression_closed_form_at_four_hundred_kilometre_radius`, `..._radial_motion_at_seven_hundred_thirty_eight_kilometres`, `..._rotation_invariance_at_four_hundred_kilometre_radius`). |
| 10 | Hypothesis | criterion | `reference_agrees_with_the_closed_form` at a 12.2 km range: 8.29631502385908e-15 rad, 31.646581 times below that geometry's floor while being 1.243608e-06 of the angle. |
| 11 | re-audit of the suite | criterion | `pytest.approx(expected, rel=r)` carries a **default absolute tolerance of 1e-12** and accepts a value within either bound, so on an angle of order 1e-05 rad every relative tolerance tighter than 2e-08 is silently replaced by it. The known-answer test asserted the closed-form `total_rad` equal to `2 v_perp / c` at `rel=1e-12` and passed, while the two differ by a relative 2.086218195129917e-10: `total_rad` is the angle between two unit vectors offset either side of the line of sight, so it is `2 arctan(theta/2) = theta - theta^3/12`, not `theta`. Found by re-running the whole suite with `abs=0.0` substituted into every `rel=` call; exactly 3 assertions of 271 depended on the default. All three are now explicit and the suite passes with `abs=0.0` everywhere. |

Hypothesis found six of the eleven. The hand-written suite found none of those
six, and the eleventh was found by a deliberate audit rather than by any test.

## Timings — UNSTABLE

Declared unstable. These move between runs on shared hardware and appear in no
prose conclusion anywhere in this repository. 2 CPU cores. The table below is a
snapshot from one run; the authoritative values are the ones in
`validation/validate_pointahead_output.txt` and
`validation/validate_kepler_output.txt`, which are re-measured every time those
scripts execute, so a reader who re-runs them will see different numbers and
that is the point of the heading.

| operation | calls | mean / ms |
|---|---|---|
| `closed_form_point_ahead` | 300 | 0.10532 |
| `reference_point_ahead` (Kepler trajectory, converged) | 300 | 0.61501 |
| `propagate_two_body` | 40 | 0.0461 |
| `scipy.integrate.solve_ivp` DOP853, rtol 1e-13 | 40 | 1.6587 |

## What is not validated

- **An assertion tolerance is not a measurement.** The suite was re-audited
  with `pytest.approx`'s default absolute tolerance removed and passes, but
  that audit covers the 95 `approx` calls that existed when it was run and is
  not a standing guard. `test_regression_pytest_approx_default_absolute_floor`
  asserts the default is still 1e-12, so a pytest release that changes it is a
  visible failure rather than a silent loosening; it does not catch a new
  assertion written with a relative-only tolerance.
- **No comparison against an independent third-party aberration or light-time
  implementation.** See deviation 1. The three internal routes agree to
  1e-15 rad, which demonstrates implementation consistency and not correctness
  against the field.
- **No flight or laboratory data of any kind.** Every geometry is generated by
  `pointahead.scenario` or loaded from a committed YAML case, and every number
  is a statement about those generators.
- **No force model beyond two-body.** Oblateness, drag, solar radiation
  pressure and third bodies are absent, and the ephemeris staleness lines are
  therefore statements about two-body propagation, not about a real ephemeris.
- **No gravitational light bending**, no refractive medium, no atmosphere.
- **No jitter spectrum.** The beam context is the static diffraction-limited
  pointing loss; a real terminal's error is a distribution and the loss has to
  be integrated over it.
- **Both ends of the dependency ranges.** See deviation 2.

## Reproducing every number above

From the product root, with `src` on `PYTHONPATH` or the package installed:

```bash
python -m pytest tests/ -q --junitxml=junit.xml
ruff check src/ tests/ examples/ validation/
python validation/validate_aberration.py
python validation/validate_kepler.py
python validation/validate_lighttime.py
python validation/validate_pointahead.py
python validation/validate_budget.py
python validation/validate_invariants.py
python validation/validate_frames.py
python validation/validate_cli.py
python validation/regenerate_cases.py
python validation/validate_regressions.py
python validation/worked_example.py
python validation/probe_working_directories.py
```

Every script writes its raw output to `validation/<script>_output.txt`. The
exit code answers "did the measurement complete"; a failed check is reported in
that file's `FAILED CHECKS` block and does not change the exit code. See
`validation/_bootstrap.py` for why.
