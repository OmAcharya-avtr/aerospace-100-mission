# DetumbleSim 0.1.0 — Validation evidence (Level 2, Research)

Every number in this file was produced by running the scripts in this
directory in the build session, on Python 3.11 with numpy, scipy and
scikit-learn on **2 CPU cores**.  Each script writes its raw stdout to
`<script>_output.txt` next to itself, and those captures are committed.
Rerun any of them from the product root with `python validation/<script>.py`.

| Script | Raw output | Wall time |
|---|---|---|
| `field_model_check.py` | `field_model_check_output.txt` | 1.7 s |
| `momentum_monotonicity.py` | `momentum_monotonicity_output.txt` | 4.8 s |
| `gain_scaling.py` | `gain_scaling_output.txt` | 21.6 s |
| `controllability_gap.py` | `controllability_gap_output.txt` | 6.7 s |
| `learned_vs_fixed_ci.py` | `learned_vs_fixed_ci_output.txt` | 98.3 s |

Total validation wall time **≈ 133 s**, inside the 3-minute-per-run,
2-core compute budget.  The 305-test suite adds 15.9 s.

**All spacecraft, orbit and telemetry data in this package is SIMULATED.**
No flight telemetry, no measured magnetometer record and no on-orbit
detumble log is used anywhere.  The only external measured-model data are the
twelve IGRF-14 field magnitudes in V1, which come from the British Geological
Survey web service and are used as the *reference* the package is checked
against.

---

## Summary

| ID | Check | Reference | Result | Criterion |
|---|---|---|---|---|
| V1-A1 | Dipole truncation error vs IGRF-14, 12 points | BGS IGRF-14 web service, 2025-01-01 | median \|rel. error\| **8.967 %** | ≤ 15 % — **PASS** |
| V1-A2 | Same, worst point | same | max \|rel. error\| **71.860 %** (45° S, 0° E, 500 km) | ≤ 25 % — **FAIL** |
| V1-A3 | Geomagnetic north pole | WDC Kyoto IGRF-14 2025: 80.8° N, 72.8° W | 80.7894° N, 72.7628° W; Δ = 0.0106°, 0.0372° | ≤ 0.05° — **PASS** |
| V1-A4 | \|B\| = 2·B0 at the dipole pole, B0 at the dipole equator | closed form | rel. error 2.999e-13 both | rtol 1e-9 — **PASS** |
| V2-B1 | Kinetic energy non-increasing under ideal B-dot, any inertia | eq. (1), `control.ideal_bdot_torque` | 0 of 16964 draws violate | **PASS** |
| V2-B2 | \|H\| non-increasing, isotropic inertia | eq. (2) with J = jI | 0 of 16964 draws violate | **PASS** |
| V2-B3 | \|H\| non-increasing, **asymmetric** inertia | the product spec's stated property | 94 of 16964 draws (0.55 %) violate it | **FALSIFIED**, not a defect |
| V2-B3a | Analytic maximum of the momentum-rise bracket | closed form derived in `test_control.py` | analytic 0.415475947 vs numerical 0.415475947, rel. diff 2.6e-11 | **PASS** |
| V3-C1 | Detumble time ∝ 1/gain, unsaturated | `analytic.py` eq. (4) | fitted log-log slope **−0.994903**, k·t spread 1.270 % | \|slope+1\| ≤ 0.05 — **PASS** |
| V3-C2 | Measured time inside the analytic modal bracket | `analytic.modal_time_constants` | **8 of 8** points inside | **PASS** |
| V3-C3 | Where the averaged model breaks | same | 0 of 5 inside once runs last < 1 orbit; whole-range slope −0.577 | measurement |
| V4-D1 | Worst-axis geometry factor, equatorial vs sun-synchronous | isotropic value 2/3 | 0.06050 vs 0.45882 → **11.02×** and 1.45× slower than isotropic | measurement |
| V4-D3 | Detumble time, same vehicle and gain, two orbits | — | equatorial 31887.7 s vs sun-synchronous 2676.5 s (**11.91×**) | measurement |
| V4-D5 | `geometry_factors()` vs `controllability_report()` | two code paths | max difference 5.551e-17 | < 1e-12 — **PASS** |
| V5-E6 | Learned scheduler vs tuned fixed gain, 40 paired held-out scenarios, w = 0 | paired 95 % CI | **−0.206 [−0.308, −0.103]** orbits — learned wins | interval excludes 0 |
| V5-E6 | Same at w = 2 (energy-weighted) | paired 95 % CI | **+0.181 [+0.038, +0.324]** — the **fixed gain wins** | interval excludes 0 |
| V5-E6 | Learned scheduler vs 3-coefficient power-law fit, w = 0 | paired 95 % CI | **+0.078 [+0.010, +0.146]** — the **power law wins** | interval excludes 0 |

---

## V1 — Does the tilted dipole reproduce published field magnitudes?

`field_model_check.py`.  The field model is the **degree-1 truncation of
IGRF-14** at main-field epoch 2025.0 (`g(1,0) = −29350.0`,
`g(1,1) = −1410.3`, `h(1,1) = 4545.5` nT), not a full IGRF evaluation, and the
README says so everywhere it is mentioned.

Reference data: twelve total-intensity values from the British Geological
Survey IGRF-14 web service
(`https://geomag.bgs.ac.uk/web_service/GMModels/igrf/14/`) queried at
2025-01-01.  Two caveats are recorded rather than hidden: BGS takes *geodetic*
latitude while this package uses geocentric spherical latitude (a difference
up to ≈ 0.19°, far smaller than the errors below), and a query against the
reference model is not an independent implementation of it.

Criteria were fixed before the run: A1 median ≤ 15 %, A2 max ≤ 25 %,
A3 pole within 0.05°, A4 rtol 1e-9.

| lat [°N] | lon [°E] | alt [km] | IGRF-14 [nT] | dipole [nT] | rel. error |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 31835 | 29833.5 | −6.29 % |
| 0 | 0 | 500 | 24172 | 23788.9 | −1.59 % |
| 0 | 180 | 500 | 26925 | 23788.9 | −11.65 % |
| 45 | 0 | 500 | 37356 | 38269.3 | +2.44 % |
| −45 | 0 | 500 | 21033 | 36147.3 | **+71.86 %** |
| 80 | 0 | 500 | 45087 | 46720.8 | +3.62 % |
| −80 | 0 | 500 | 37254 | 46139.2 | +23.85 % |
| 0 | 90 | 400 | 34445 | 25627.7 | −25.60 % |
| 30 | 270 | 400 | 38097 | 36539.9 | −4.09 % |
| 60 | 120 | 800 | 41591 | 34967.5 | −15.93 % |
| −30 | 315 | 400 | 19625 | 29445.3 | **+50.04 %** |
| 90 | 0 | 0 | 56879 | 58892.6 | +3.54 % |

Median 8.967 %, mean 18.374 %, RMS 27.946 %, max 71.860 %.

**A2 FAILED.** No coefficient was adjusted and no tolerance was widened.  The
two worst points sit in and beside the South Atlantic Anomaly, and the third
worst is the Indian Ocean low: exactly where the non-dipole terms of the
geomagnetic field are largest.  A centred dipole is a first-order model and
this is what first order costs.  The consequence for the rest of the package
is stated in the README under Limitations: detumble rates scale with |B|², so
a 25 % field error is a ~56 % error in instantaneous damping rate at that
point, though the errors partially average out over an orbit.

A3 and A4 pass cleanly and are genuine known-answer checks: the geomagnetic
pole derived from these three coefficients lands within 0.04° of the value
WDC Kyoto publishes for IGRF-14 at epoch 2025, and the closed-form pole and
equator field strengths are reproduced to 3e-13.

**Overall V1: A1 PASS, A2 FAIL, A3 PASS, A4 PASS.**

---

## V2 — Is angular momentum monotone under B-dot?

`momentum_monotonicity.py`.  The product specification states that "angular
momentum decreases monotonically under B-dot in the absence of saturation".
That is **true for kinetic energy and for an isotropic inertia, and false in
general**, and this script settles both halves with numbers.

With the ideal B-dot torque `L = −k|B|²ω⊥`:

    dT/dt = ω·L = −k|B|²|ω⊥|² ≤ 0                    for ANY inertia   (1)
    H·L   = −k|B|² [ ωᵀJω − (ω·B̂)(B̂ᵀJω) ]                            (2)

The bracket in (2) is positive for every field direction only when `J` is
isotropic.

| Check | Draws | Violations | Verdict |
|---|---:|---:|---|
| B1 energy rate ≤ 0, random asymmetric J | 16964 | 0 | PASS |
| B2 `H·L ≤ 0`, isotropic J = 1.7 I | 16964 | 0 | PASS |
| B3 `H·L ≤ 0`, random asymmetric J | 16964 | **94 (0.55 %)** | **property FALSIFIED** |

Worst violating draw: `J = diag(1.5893, 1.3904, 0.6541)` (J_max/J_min = 2.430),
`H·L = +6.931e-04`, normalised `+9.121e-02`.

**B3a, the closed-form counterexample.** For `J = diag(1, 1, J₃)` and
`ω = (a, 0, c)`, maximising the negated bracket over unit `B̂` in the `(ω, Jω)`
plane gives

    max = ( −(J₃c² + a²) + sqrt( (J₃c² − a²)² + a²c²(1+J₃)² ) ) / 2

and the discriminant identity

    (J₃c² − a²)² + a²c²(1+J₃)² − (J₃c² + a²)² = a²c²(1 − J₃)²

makes it strictly positive whenever `J₃ ≠ 1` and `a, c ≠ 0`.  For
`J₃ = 4, a = c = 1`: analytic maximum **0.415475947**, numerical maximum over
200001 field directions **0.415475947**, relative difference **2.619e-11**,
attained at 60.4818° from the x axis.  The identity check returns
**9.000000000** on both sides.  The same counterexample is a unit test
(`tests/test_control.py::TestMomentumIsNotMonotoneWhenAsymmetric`).

**B4, in the full simulator, unsaturated (500 A m² limit).**  Note that the
simulator uses the flight-realistic backward-difference law
`m = −k (B[i]−B[i−1])/Δt`, which also picks up the field change caused by
orbital motion, so (1) and (2) hold only approximately step by step.

| Inertia | saturated | energy steps rising | \|H\| steps rising | largest \|H\| rise |
|---|---:|---:|---:|---:|
| `diag(0.05, 0.05, 0.05)` | 0.000 % | 0 of 10000 | 0 of 10000 | — |
| `diag(0.02, 0.03, 0.045)` | 0.000 % | 85 of 10000 | 156 of 10000 (1.560 %) | 1.0165e-07 N m s = 0.0018 % of \|H₀\| |

Energy over the asymmetric run falls from 4.4658e-04 J to 1.0666e-06 J;
`|H|` from 5.5930e-03 to 3.0977e-04 N m s.  The excursions are real but tiny.

**B5, with saturation (0.2 A m², asymmetric inertia).** 3.150 % of control
steps saturated; 51.060 % of `|H|` steps and 49.610 % of energy steps rise,
with the largest single energy rise 4.965e-10 J against a starting energy of
4.466e-04 J.  Once the command is clipped the applied torque is no longer
`−k|B|²ω⊥` and neither inequality applies step by step; the run still ends at
2.025e-07 J.

**Overall V2: B1 PASS, B2 PASS, B3 the specification's momentum claim is
FALSIFIED for asymmetric inertia** and replaced in this package by the energy
statement, which is the one that is actually true in general.

---

## V3 — Does detumble time scale as 1/gain?

`gain_scaling.py`.  Isotropic `j = 0.05 kg m²`, 500 km sun-synchronous orbit,
`|ω₀| = 9.6690 deg/s` to a 1.0 deg/s target, RK4 at a 4 s control step.  The
dipole limit is set to a deliberately unphysical 50 A m² because the analytic
model assumes no saturation; the measured saturated fraction is **0.000 % on
every point**, so the assumption held.

| gain k [A m² s/T] | t_sim [s] | orbits | t_isotropic [s] | bracket [fast, slow] | inside |
|---:|---:|---:|---:|---|---|
| 3.0000e+03 | 56200.5 | 9.90 | 41301.8 | [27938.3, 60011.5] | yes |
| 4.4580e+03 | 38598.1 | 6.80 | 27794.0 | [18801.1, 40384.8] | yes |
| 6.6245e+03 | 25693.6 | 4.53 | 18704.0 | [12652.2, 27176.9] | yes |
| 9.8440e+03 | 17281.5 | 3.04 | 12586.9 | [8514.3, 18288.7] | yes |
| 1.4628e+04 | 11692.3 | 2.06 | 8470.3 | [5729.7, 12307.4] | yes |
| 2.1737e+04 | 7770.2 | 1.37 | 5700.1 | [3855.8, 8282.3] | yes |
| 3.2302e+04 | 5225.4 | 0.92 | 3835.9 | [2594.8, 5573.6] | yes |
| 4.8000e+04 | 3657.7 | 0.64 | 2581.4 | [1746.1, 3750.7] | yes |

- **C1 PASS**: fitted log-log slope **−0.994903** against the predicted −1
  (`|slope + 1| = 0.005097`); the product `k·t` varies by only **1.270 %**
  across a 16× gain range.
- **C2 PASS**: **8 of 8** measured times lie inside the analytic modal
  bracket.  The isotropic estimate is systematically low (measured/isotropic
  ≈ 1.32–1.42) because this orbit's geometry factors are
  `(0.45882, 0.55563, 0.98555)`, not `(2/3, 2/3, 2/3)`.

**C3, where the model breaks (reported, not hidden).** Extending to gains
6.4e4–1.0e6, where the detumble takes 0.13–0.94 orbits, **0 of 5** points lie
inside the bracket and the log-log slope over the whole range degrades to
**−0.577111** against −0.994903 over the multi-orbit range alone.  Orbit
averaging needs a separation of timescales, and once the detumble is faster
than an orbit there is none.

**C4, near-equatorial orbit (i = 5°).** Geometry factors
`(0.09229, 0.95242, 0.95530)`.  Three of eight gains never reach the target
inside 250000 s; over the five that finish the log-log slope is **−1.010127**
and detumble times are **12.540×** the sun-synchronous case at the same gains.
Only 2 of 8 points fall inside the analytic bracket, because at this
inclination the slow mode is so slow that the target rate approaches the
B-dot floor.

**Overall V3: C1 PASS, C2 PASS**; C3 and C4 are measurements.

---

## V4 — The controllability gap along **B**

`controllability_gap.py`.  A magnetorquer produces `L = m × B`, which is
identically perpendicular to `B`, so the rate component along `B` receives no
torque at that instant.  `rank(I − B̂B̂ᵀ) = 2` and `trace = 2` exactly, for
every field direction tested.  Detumbling works only because `B̂` moves.

**D1, orbit-averaged geometry factors** — eigenvalues of
`(⟨|B|²⟩I − ⟨BBᵀ⟩)/⟨|B|²⟩`, which sum to exactly 2 and equal `(2/3, 2/3, 2/3)`
only for a perfectly isotropic field-direction history.  500 km, 10-orbit
span, 8000 samples:

| inclination [°] | B_rms [µT] | λ_min | λ_mid | λ_max | anisotropy | λ_min/(2/3) | mean uncontrollable fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24.160 | **0.06050** | 0.96836 | 0.97114 | 16.053 | 0.0907 | 0.9695 |
| 5 | 24.437 | 0.09230 | 0.95242 | 0.95529 | 10.350 | 0.1384 | 0.9533 |
| 20 | 26.562 | 0.29975 | 0.84493 | 0.85532 | 2.853 | 0.4496 | 0.8325 |
| 45 | 32.119 | 0.51393 | 0.65693 | 0.82914 | 1.613 | 0.7709 | 0.5237 |
| 63.4 | 35.593 | 0.47859 | 0.58133 | 0.94008 | 1.964 | 0.7179 | 0.5990 |
| 80 | 37.237 | 0.45642 | 0.55271 | 0.99087 | 2.171 | 0.6846 | 0.6310 |
| 97.4 | 37.059 | 0.45882 | 0.55563 | 0.98554 | 2.148 | 0.6882 | 0.6277 |

The equatorial orbit's worst axis damps **11.02× slower** than the isotropic
ideal; the sun-synchronous orbit's, only 1.45× slower.  The weakest inertial
direction for the equatorial orbit is `[−0.0222, −0.02805, 0.99936]` — the
polar axis, i.e. the direction the tilted dipole stays closest to along an
equatorial track.

**D2, instantaneous uncontrollable fraction** `|ω·B̂|/|ω|` for a rate along
the weakest direction, over 10 orbits:

| inclination | mean | min | max | fraction of time above 0.9 |
|---:|---:|---:|---:|---:|
| 0° | 0.9695 | 0.9443 | 0.9907 | **1.0000** |
| 97.4° | 0.6278 | 0.0007 | 0.9999 | 0.2650 |

On an equatorial orbit, more than 90 % of a weak-axis rate is beyond control
authority **100 % of the time**.

**D3, direct simulation.** Identical vehicle (`J = 0.05 I` kg m², 0.2 A m²),
identical gain 1.5e5, `ω₀ = (5, 5, 5) deg/s`, 40000 s:

| inclination | detumble to 1 deg/s | \|ω\| at 40000 s | weak-axis part | weak-axis share | saturated |
|---:|---:|---:|---:|---:|---:|
| 0° | **31887.7 s** | 0.66915 deg/s | 0.66621 deg/s | **0.9956** | 1.01 % |
| 97.4° | **2676.5 s** | 0.15703 deg/s | 0.02876 deg/s | 0.1832 | 1.32 % |

An 11.91× difference in detumble time between two orbits that differ only in
inclination, and on the equatorial orbit **99.56 % of what is left is spin
about the axis the magnetorquers cannot reach**.

**D5**: the two independent code paths that compute the geometry factors agree
to **5.551e-17** (tolerance 1e-12) — PASS.

---

## V5 — The learned gain scheduler against classical gain rules

`learned_vs_fixed_ci.py`.  Training seeds 1000–1019 (20 scenarios), held-out
seeds 5000–5039 (40 scenarios), disjoint.  Nothing tuned on the training set
is retuned on the held-out set, and the held-out set is simulated exactly once
per policy.  All scenarios are synthetic
(`detumblesim.scenarios.sample_scenario`).

Cost `= t_detumble/T_orbit + w · ∫|m|²dt/(m_max² T_orbit)`.  Because the two
terms are recorded separately, every comparison is re-scored at `w = 0`
(time only), `w = 0.5` (default) and `w = 2` **without re-simulating**, so no
conclusion rests on one arbitrary weight.

**E1, fixed-gain tuning.** Mean training cost per grid gain: 2.6183, 1.7350,
0.8536, **0.6525**, 0.6905, 1.0712, 1.3372, 2.1202 → tuned
`k_fixed = 7.196857e+04 A m² s/T`.  The per-scenario oracle reaches 0.4926, so
the headroom a perfect constant-gain scheduler could capture is **24.51 %**,
and the oracle uses 6 of the 8 grid gains — the best gain genuinely varies by
vehicle.

**E2, sized-gain baseline** (`k = c·m_max/(⟨|B|⟩·ω_est)`), estimator and
coefficient jointly tuned on training: best is `estimator = max, c = 2.3784`,
training mean cost 0.5689.

**E3, power-law baseline** fitted by least squares to the training oracle
gains:

    log10 k = 6.2816 + 0.6978 log10(m_max) + 0.4397 log10(j)

RMS residual **0.3050 dex** over 20 scenarios.  The naive sizing rule would
give `b = 1`; the fitted exponent is 0.6978 and is reported as measured.

**E4, learned scheduler.** 1068 feature rows × 8 features, labels spanning
`[−0.2857, +1.1429]` dex, `RandomForestRegressor(n_estimators=200,
max_depth=6, min_samples_leaf=4, random_state=0)`.  Impurity importances:

| feature | importance |
|---|---:|
| `log10_max_dipole` | **0.7715** |
| `log10_inertia_scale` | **0.2284** |
| `log10_rate_proxy` | 0.0000 |
| `log10_mean_field_t` | 0.0000 |
| `rate_trend_per_1000s` | 0.0000 |
| `saturation_duty` | 0.0000 |
| `field_variability` | 0.0000 |
| `log10_elapsed_s` | 0.0000 |

**100.0 % of the importance sits on the two static vehicle parameters.**  The
six time-varying magnetometer features carry none of it.  The "scheduler" has
learned a per-vehicle gain lookup, not a schedule.

**E5, held-out summary (40 scenarios, default `w = 0.5`).**

| policy | cost (95 % CI) | time [orbits] (95 % CI) | energy term (95 % CI) | failures | saturated |
|---|---|---|---|---:|---:|
| fixed | 1.170 [0.788, 1.552] | 0.885 [0.637, 1.133] | 0.285 [0.130, 0.441] | 0 | 16.97 % |
| sized | 1.422 [0.795, 2.050] | 1.131 [0.585, 1.677] | 0.292 [0.176, 0.408] | **2** | 25.43 % |
| powerlaw | 0.999 [0.622, 1.376] | 0.601 [0.410, 0.793] | 0.398 [0.205, 0.590] | 0 | 42.60 % |
| learned | 1.061 [0.690, 1.433] | 0.680 [0.488, 0.871] | 0.382 [0.193, 0.571] | 0 | 32.46 % |

Mean detumble time over runs that finished: fixed 5162.7 s (n = 40), sized
4579.4 s (n = 38), powerlaw 3521.6 s (n = 40), learned 3967.9 s (n = 40).
Mean gain actually used: 7.1969e+04, 2.3799e+05, 1.8506e+05, 1.7415e+05.

Scheduler confidence over 2586 gain updates: mean 0.9799, min 0.6482, max
1.0000.  This is an ensemble-spread heuristic, not a calibrated interval.

**These marginal intervals overlap heavily and resolve almost nothing** — the
scenarios differ enormously in difficulty.  The paired differences are the
comparison that has power.

**E6, paired differences** (negative favours the first policy):

| w | comparison | mean difference | 95 % CI | verdict |
|---:|---|---:|---|---|
| 0.0 | learned − fixed | −0.206 | [−0.308, −0.103] | **learned wins** |
| 0.0 | powerlaw − fixed | −0.284 | [−0.402, −0.165] | **powerlaw wins** |
| 0.0 | sized − fixed | +0.246 | [−0.208, +0.699] | not resolved |
| 0.0 | learned − powerlaw | +0.078 | [+0.010, +0.146] | **powerlaw wins** |
| 0.0 | learned − sized | −0.451 | [−0.898, −0.004] | learned wins |
| 0.5 | learned − fixed | −0.109 | [−0.196, −0.021] | **learned wins** |
| 0.5 | powerlaw − fixed | −0.171 | [−0.273, −0.069] | **powerlaw wins** |
| 0.5 | sized − fixed | +0.252 | [−0.184, +0.688] | not resolved |
| 0.5 | learned − powerlaw | +0.063 | [−0.010, +0.135] | not resolved |
| 0.5 | learned − sized | −0.361 | [−0.783, +0.061] | not resolved |
| 2.0 | learned − fixed | +0.181 | [+0.038, +0.324] | **fixed wins** |
| 2.0 | powerlaw − fixed | +0.166 | [+0.010, +0.322] | **fixed wins** |
| 2.0 | sized − fixed | +0.272 | [−0.150, +0.694] | not resolved |
| 2.0 | learned − powerlaw | +0.016 | [−0.094, +0.125] | not resolved |
| 2.0 | learned − sized | −0.091 | [−0.530, +0.349] | not resolved |

### The honest reading

1. **On detumble time alone the learned scheduler beats the hand-tuned fixed
   gain**, by 0.206 orbits [0.103, 0.308] — about 23 % of the fixed gain's
   mean detumble time.  That is a real, resolved win.
2. **Weight the coil energy heavily and the fixed gain wins instead**, by
   0.181 [0.038, 0.324].  The learned scheduler buys speed by commanding a
   ~2.4× larger gain, which saturates the torquers on 32.5 % of steps against
   17.0 % for the fixed gain.  Neither result is more valid than the other;
   which one matters depends on a mission's power budget.
3. **The learned scheduler never beats a three-coefficient log-linear fit of
   the same training data.**  At `w = 0` the power law wins outright
   (+0.078 [+0.010, +0.146]); at `w = 0.5` and `w = 2` the two are
   indistinguishable.  A 200-tree RandomForest with eight features is being
   matched or beaten by three numbers fitted with `numpy.linalg.lstsq`.
4. **Feature importances explain why.** All of the model's discriminating
   power is on `m_max` and `j`, the two inputs the power law also uses.  The
   magnetometer-derived features — the ones that would make it a *scheduler*
   rather than a lookup — contribute nothing measurable.
5. **The sized-gain rule, the one an ADCS engineer would actually write, is
   not shown to beat the fixed gain on held-out data** at any weight, and it
   failed to detumble 2 of 40 scenarios inside the simulated span.  It was
   tuned on training data where it looked better (0.5689 vs 0.6525); that
   advantage did not transfer.

**Overall V5: the AI element earns a resolved win over the naive fixed gain on
detumble time, loses to it when energy is weighted, and is not shown to beat a
three-parameter classical regression on anything.  The classical baseline is
the recommendation.**

---

## Known limitations recorded here

1. **V1-A2 failed.**  The centred dipole is wrong by up to 71.9 % in the South
   Atlantic Anomaly.  Detumble rates scale with `|B|²`, so every detumble time
   in this package inherits a field-model error that is not quantified
   end-to-end (only the field error itself is).
2. The spec's angular-momentum monotonicity claim is false for asymmetric
   inertia (V2-B3).  This package states the energy version instead.
3. The 1/k scaling law is a multi-orbit-average result and degrades to a
   slope of −0.577 once detumbles take less than an orbit (V3-C3).
4. V5 uses 20 training and 40 held-out scenarios, sized by the 3-minute
   compute budget.  The marginal per-policy intervals are far too wide to
   resolve anything (e.g. fixed 1.170 [0.788, 1.552]); only the paired
   differences have power, and five of the fifteen paired comparisons are
   still unresolved at this sample size and are reported as such.
5. The scheduler's confidence output is an ensemble-spread heuristic with **no
   coverage calibration**.  Its mean over 2586 updates is 0.9799, which says
   the trees usually agree, not that the predictions are usually right.
6. The learned scheduler's training target is the best *constant* gain per
   scenario, so the constant-gain oracle (24.51 % headroom on training) is an
   upper bound on what this target can teach.  A genuinely time-varying
   optimal gain was never computed.
7. No aerodynamic, gravity-gradient or residual-dipole disturbance torque is
   modelled, and the orbit is circular and unperturbed.  Both would lengthen
   real detumbles relative to these numbers.
