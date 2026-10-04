# ConstelLink — validation evidence

Validation level 3. Research-grade software. Not flight-qualified, not
certified, not approved for operational aerospace use.

Every number in this file was produced by running the script named beside it,
in this repository, in the session that wrote this file. Raw stdout is saved
next to each script as `<script>_output.txt`. Nothing here is quoted from a
paper, a datasheet or memory.

Reproduce everything:

```bash
for s in validation/validate_*.py; do python "$s"; done
```

| script | what it checks | result |
|---|---|---|
| `validate_sgp4_vector.py` | SGP4 against the verification data shipped with `sgp4` | PASS |
| `validate_contact_windows.py` | contact windows against closed-form central angles | PASS |
| `validate_walker.py` | Walker pattern geometry and the realised orbit radius | PASS |
| `validate_routing.py` | Dijkstra against exhaustive route enumeration | PASS |
| `validate_ilp.py` | integer max flow against brute force; the `pulp` model | PASS (pulp solve NOT RUN — see below) |
| `validate_capacity.py` | RF and optical models against closed forms and quadrature | PASS |
| `validate_failure_modes.py` | the four required failure modes plus input validation | PASS |
| `validate_uncertainty.py` | four separate uncertainty sources | REPORTED (measurements, no pass/fail) |
| `validate_calibration.py` | predictor calibration on grouped splits | PASS (criteria are the identity and probability range, not the ranking) |
| `validate_benchmark.py` | performance with its measurement method | REPORTED (container numbers) |

---

## 1. SGP4 against the shipped verification data

**Reference.** `SGP4-VER.TLE` and `tcppver.out`, both installed by the `sgp4`
package (version 2.27 in this environment). They are the verification suite of
Vallado, Crawford, Hujsak & Kelso 2006, "Revisiting Spacetrack Report #3",
AIAA 2006-6753. The script reads them from the installed package directory, so
the reference is whatever the reader's own install ships.

**Case.** Satellite 00005, the TEME example (eccentricity 0.186), over the
verification span of 0 to 4320 minutes from epoch in 360 minute steps —
13 reference rows.

| quantity | measured | declared tolerance |
|---|---|---|
| max position residual | **6.819e-09 km** | 1e-6 km |
| max velocity residual | **7.705e-10 km/s** | 1e-7 km/s |
| wrapper vs `sgp4_tsince` at +4320 min | 1.693e-05 km | 1 km |

The tolerance is two decades above the print precision of `tcppver.out`
(1e-8 km). The wrapper cross-check is looser because the element epoch is
reconstructed from the TLE's own 1e-8 day field, which is about 0.9 ms, and
this orbit moves at roughly 7 km/s.

**What this does not establish.** This checks that `constellink` drives `sgp4`
correctly. It is not a validation of SGP4 against truth orbits and says
nothing about accuracy far from epoch.

Script: `validate_sgp4_vector.py` · Output: `validate_sgp4_vector_output.txt`

---

## 2. Contact windows against closed-form central angles

Four checks, all on a two-body circular orbit so the only approximation left
is the window finder itself.

### 2.1 Ground access, central angle at the window edges

Closed form (Wertz & Larson (eds.) 1999, *Space Mission Analysis and Design*,
3rd ed., Ch. 5): `lambda_max = arccos((R_e / r) cos eps) - eps`.

Equatorial circular orbit at 550 km, equatorial station at zero altitude,
10 deg mask, 6 h horizon, 30 s scan, 1e-4 s bisection.

| quantity | value |
|---|---|
| `lambda_max`, closed form | **14.956635125 deg** |
| worst residual at a reported rise or set time | **1.30e-06 deg** |
| declared tolerance | 1e-4 deg |

### 2.2 Ground access, pass duration

`T = 2 lambda_max / (n - omega_E)` with `n = sqrt(mu / r^3)`.

| quantity | value |
|---|---|
| two-body mean motion `n` | 1.094823692886e-03 rad/s |
| `omega_E` (WGS-84 sidereal) | 7.292115000000e-05 rad/s |
| `omega_E` (IAU 1982 GMST linear term) | 7.292115855307e-05 rad/s |
| predicted duration (GMST rate) | **510.895157 s** |
| measured durations | 510.895156, 510.895157, 510.895157 s |
| worst residual | **1e-06 s** |
| declared tolerance | 0.5 s |

Both Earth-rotation rates are printed because the difference between them is
the dominant term in the residual; neither is quietly preferred.

### 2.3 Inter-satellite clearance against `gamma_max`

`gamma_max = 2 arccos(r_block / r)` with `r_block = R_e + 100 km`.

| quantity | value |
|---|---|
| `gamma_max`, closed form | **41.528367944 deg** |
| bisected flip angle of `isl_clear` | 41.528367944 deg |
| residual | **1.908e-14 deg** |
| declared tolerance | 1e-5 deg |

### 2.4 Earth-limb distance at a real ISL window edge

Two co-planar circular satellites at 6928.137 and 7048.137 km, 20 h horizon,
30 s scan, 1e-4 s bisection.

| edge | minimum geocentric radius of the segment | residual against `r_block` = 6478.137 km |
|---|---|---|
| open | 6478.137000 km | **2.06e-07 km** |
| close | 6478.136999 km | **−9.87e-07 km** |

Declared tolerance 1e-3 km.

Script: `validate_contact_windows.py` · Output:
`validate_contact_windows_output.txt`

---

## 3. Walker generator: realised vs requested geometry

The generator sets the SGP4 mean motion from the two-body relation
`n = sqrt(mu / a^3)`, while SGP4 consumes a Kozai mean motion. The realised
orbit is therefore not exactly the requested one, and this is measured rather
than assumed.

| requested altitude | requested radius | mean realised radius | offset | peak-to-peak variation over one orbit |
|---|---|---|---|---|
| 400 km | 6778.137 km | 6778.010 km | **−0.127 km** | **11.945 km** |
| 550 km | 6928.137 km | 6928.013 km | **−0.124 km** | 11.945 km |
| 800 km | 7178.137 km | 7178.017 km | **−0.120 km** | 11.945 km |
| 1200 km | 7578.137 km | 7578.024 km | **−0.113 km** | 11.945 km |

The peak-to-peak variation is the J2 short-periodic term at 53 deg
inclination; the mean offset is the Kozai/Kepler difference. Declared bounds:
offset ≤ 5 km, variation ≤ 15 km.

Pattern angles (inclination, RAAN, mean anomaly) reproduce the Walker
`53:24/4/1` definition with a worst error of **2.842e-14 deg** over all 24
satellites, against a 1e-6 deg tolerance.

Reported, not pass/fail: the intra-plane chord for `W00-00` to `W00-01` in a
24/4/1 shell measures 6928.818 km mean with 10.345 km peak-to-peak, against
6928.137 km for a perfect circle at the requested radius — the same offset,
scaled by the chord geometry.

Script: `validate_walker.py` · Output: `validate_walker_output.txt`

---

## 4. Routing: Dijkstra against exhaustive enumeration

Two independent implementations sharing only the graph object: Dijkstra with a
binary heap, and depth-first enumeration of every feasible route.

### 4.1 Hand-built instances

| instance | Dijkstra | enumeration | expected (written out in the script) |
|---|---|---|---|
| two hops, no alternative | 120.006671282 s | 120.006671282 s | 120.006671282 s |
| direct hop wins | 60.010006923 s | 60.010006923 s | 60.010006923 s |
| store and forward | 180.006671282 s | 180.006671282 s | 180.006671282 s |

### 4.2 Random instances

40 instances, alternating dense (0.55 link occupancy) and sparse (0.12), 4-5
nodes, 5 slots of 60 s.

| quantity | value |
|---|---|
| instances with a route | 30 |
| instances with no route (both methods agreed on absence) | 10 |
| routes enumerated in total | 513 |
| **worst latency gap** | **0.000e+00 s** |

Declared tolerance 1e-9 s.

Script: `validate_routing.py` · Output: `validate_routing_output.txt`

---

## 5. Integer maximum flow: ILP against brute force

### 5.1 Hand-built instances

| instance | ILP | brute force | expected |
|---|---|---|---|
| single chain, 1 unit | 1 | 1 | 1 |
| two parallel paths, 2 units | 2 | 2 | 2 |

Flow quantum 60 Mbit; backend `scipy-highs`.

### 5.2 Random instances

60 attempted, two families (unit capacities, and mixed capacities of 1-2
quanta).

| quantity | value |
|---|---|
| instances compared | **27** |
| instances skipped (capacity box above 1e6 points) | 33 |
| instances with nonzero flow | 23 |
| largest optimum seen | 4 units = 240 Mbit |
| **mismatches** | **0** |

### 5.3 The `pulp` model — what was and was not run

The product specification names `pulp` as the ILP front end. **No `pulp`
objective value was produced in this session.** `pulp` 4.0.0 ships no bundled
CBC binary and this build container has no external MILP solver, so
`pulp.listSolvers(onlyAvailable=True)` returns an empty list and the `"pulp"`
backend raises `NoSolverError`.

What *was* verified is that the `pulp` model and the program the HiGHS backend
solves are the same program:

| structural check | result |
|---|---|
| variable count | PASS |
| constraint count | PASS |
| variable bounds and integrality | PASS |
| objective coefficients | PASS |
| constraint coefficients (every conservation row) | PASS |

Program size for the checked instance: 24 variables, 15 conservation rows.

This is weaker than an objective comparison and is labelled as such. A reader
with CBC, HiGHS or GLPK installed gets the `pulp` solve by passing
`backend="pulp"`, and `validate_ilp.py` will then compare the two objectives
automatically.

Script: `validate_ilp.py` · Output: `validate_ilp_output.txt`

---

## 6. Link capacity against closed forms

### 6.1 The dB chain reduces to Friis when losses are zeroed

Friis 1946, Proc. IRE 34(5), 254-256. The library works in dB; the reference
is evaluated in watts. Receive system temperature set to 1 K so that G/T is
numerically the receive gain.

| range | frequency | `P_rx` from the dB chain | `P_rx` from Friis | relative error |
|---|---|---|---|---|
| 500 km | 2.2 GHz | 1.872562437718e-11 W | 1.8725624377e-11 W | 6.9e-16 |
| 1500 km | 26.0 GHz | 2.366586917596e-11 W | 2.3665869176e-11 W | 7.9e-15 |
| 5000 km | 8.4 GHz | 3.226436313573e-10 W | 3.2264363136e-10 W | 3.7e-15 |
| 36000 km | 20.0 GHz | 1.735907938688e-10 W | 1.7359079387e-10 W | 3.9e-15 |

Declared tolerance 1e-12 relative.

### 6.2 Free-space path loss against the engineering form

Computed constant `K = 20 log10(4 pi x 1e3 x 1e9 / c)` = **92.447783222 dB**
(the commonly quoted rounded value is 92.45 dB). Over 20 combinations of
1-400 000 km and 0.4-193.4 GHz the largest difference between
`free_space_path_loss_db` and `K + 20 log10(d_km) + 20 log10(f_GHz)` is
**0.00e+00 dB**, against a 1e-10 dB tolerance.

### 6.3 Infinite-bandwidth Shannon limit

Minimum energy per bit `10 log10(ln 2)` = −1.591745 dB.

| bandwidth | Eb/N0 at capacity | gap to the limit |
|---|---|---|
| 1 MHz | 22.019779 dB | 2.36e+01 dB |
| 10 GHz | −1.244443 dB | 3.47e-01 dB |
| 1 THz | −1.588035 dB | 3.71e-03 dB |
| 1e14 Hz | −1.591708 dB | 3.71e-05 dB |
| 1e16 Hz | −1.591745 dB | **3.73e-07 dB** |

Monotone convergence at the expected first order in `1/B`. The widest
bandwidths are physically absurd and are swept only to show the limit being
approached at the predicted rate.

### 6.4 Gaussian aperture capture against numerical quadrature

`f_geo = 1 - exp(-2 a^2 / w^2)` against `scipy.integrate.quad` of the Gaussian
intensity over the aperture.

| `a/w` | library | quadrature | relative error |
|---|---|---|---|
| 0.01 | 0.0001999800 | 0.0001999800 | 5.7e-14 |
| 0.20 | 0.0768836536 | 0.0768836536 | 7.2e-16 |
| 1.00 | 0.8646647168 | 0.8646647168 | 2.6e-16 |
| 3.00 | 0.9999999848 | 0.9999999848 | 3.3e-16 |

Declared tolerance 1e-8 relative.

### 6.5 Pointing loss algebra

`-10 log10(exp(-2 x^2))` against `(20/ln 10) x^2` over `x` in {0, 0.05, 0.1,
0.25, 0.5, 1.0}: worst difference **1.78e-15 dB**, against a 1e-12 dB
tolerance.

Script: `validate_capacity.py` · Output: `validate_capacity_output.txt`

---

## 7. Failure modes

All five cases PASS. Full transcript in `validate_failure_modes_output.txt`.

| case | required behaviour | observed |
|---|---|---|
| empty contact graph | degrade cleanly, no exception | `is_empty` true, hold-edge skeleton built, routing returns `None`, max flow 0, 3 singleton components |
| partitioned constellation | detect and report, route within but not across | 2 components found, route across is `None`, route within exists, unreachable set is exactly the other component |
| satellite lost mid-horizon | truncate, invalidate, keep the node known | straddling window truncated to exactly 90.000000 s, later window removed, route gone, node still in the node set |
| TLE epoch far from the window | **raise**, never extrapolate | raises at ±30 d and +365 d, raises when any sample of the window is outside the limit, raises at constellation level, explicit `check_epoch=False` escape hatch works |
| input validation sample | raise `ValueError`/`KeyError` with an actionable message | 8 of 8 raised the expected type |

The raised epoch message, verbatim from the run:

```
satellite 'W00-00': requested window reaches 30.000 days from the element
epoch 2026-04-01T00:00:00+00:00, exceeding max_epoch_age_days=7.0. SGP4
degrades silently far from epoch; supply a fresher element set, or raise
max_epoch_age_days deliberately and record the resulting accuracy loss.
```

Script: `validate_failure_modes.py` · Output:
`validate_failure_modes_output.txt`

---

## 8. Uncertainty analysis

Four error sources, kept separate. These are measurements, not pass/fail.

### 8.1 Along-track propagation error to window-edge shift

Mean anomaly perturbed by `dM = ds / a`; ground windows recomputed.

| along-track error | `dM` | max edge shift | mean duration change |
|---|---|---|---|
| 0.1 km | 1.443e-05 rad | 0.0293 s | −0.0000 s |
| 1.0 km | 1.443e-04 rad | **0.1465 s** | −0.0391 s |
| 5.0 km | 7.217e-04 rad | 0.7031 s | −0.2246 s |
| 20.0 km | 2.887e-03 rad | 2.7832 s | −0.9082 s |

Edge shift per km: 0.2930, 0.1465, 0.1406, 0.1392 s/km. The smallest
perturbation sits at the 0.05 s default bisection-tolerance floor, which is
why its apparent slope is larger; the three larger perturbations are the ones
to read. The along-track error is a **user input**: this package asserts no
TLE error magnitude of its own.

### 8.2 Contact-scan grid step

One ISL over 6 h, recomputed at seven grid steps.

| step | windows | total duration | delta vs the 5 s reference |
|---|---|---|---|
| 5 s | 8 | 5859.473 s | — |
| 30 s | 8 | 5859.478 s | +0.005 s |
| 120 s | 8 | 5859.478 s | +0.005 s |
| 600 s | 8 | 5859.467 s | −0.006 s |

No window is missed at any step for this link. A window shorter than the step
*can* be missed, which is why this table exists: a user picks a step from
their own shortest window of interest.

### 8.3 Bisection refinement tolerance

| tolerance | max \|dt_open\| vs the 1e-6 s reference | max \|dt_close\| |
|---|---|---|
| 1e-3 s | 0.000409 s | 0.000438 s |
| 0.05 s (default) | 0.005542 s | 0.005228 s |
| 1 s | 0.453885 s | 0.455467 s |
| 10 s | 3.555318 s | 3.562223 s |

Edge error is bounded by the tolerance and is independent of the grid step,
because bisection re-propagates rather than interpolating the sampled values.

### 8.4 Link-budget parameter uncertainty

Monte Carlo, 4000 draws, independent Gaussian inputs.

RF terminal at 1500 km, 1-sigma inputs: `tx_power_dbw` 0.3,
`tx_gain_dbi` 0.5, `rx_g_over_t_db_per_k` 0.7, `other_loss_db` 0.3.

| statistic | value |
|---|---|
| mean | 156.700 Mbit/s (sem 0.563) |
| standard deviation | 35.599 Mbit/s |
| p5 / p50 / p95 | 105.890 / 152.338 / 220.439 Mbit/s |
| **p95/p5 spread** | **3.184 dB** |
| analytic quadrature prediction | 3.155 dB |

The agreement between the measured spread and the quadrature sum of the input
sigmas is an independent check on the Monte Carlo itself.

Optical terminal at 1500 km, 1-sigma inputs: `tx_power_w` 0.05,
`beam_divergence_full_rad` 2e-6, `pointing_error_rad` 1e-6,
`rx_optics_loss_db` 0.3: mean 11085.163 Mbit/s (sem 23.340), p5/p95
8811.432 / 13681.781 Mbit/s.

Inputs are drawn independently; real terminal parameters are correlated, so
this is a sensitivity measurement and not a calibrated predictive interval.

Script: `validate_uncertainty.py` · Output: `validate_uncertainty_output.txt`

---

## 9. Calibration — the headline result

**The learned model is not the best-calibrated predictor on this dataset. The
logistic-regression baseline is.** That is the measured result and it is kept.

Dataset: 1974 contacts from a 24/4/1 Walker shell at 550 km over 18 h, with 5
ground stations; 328 ground rows, the rest inter-satellite. Base rate 0.7523.
Grouped split by link, 30 % test, five split seeds. Mean ± sample standard
deviation over the five seeds:

| predictor | Brier (lower better) | **reliability REL (lower better)** | resolution RES (higher better) | ECE | accuracy @0.5 |
|---|---|---|---|---|---|
| climatology baseline | 0.15562 ± 0.02887 | 0.00645 ± 0.00487 | 0.03131 ± 0.00722 | 0.07025 ± 0.02991 | 0.8067 |
| **logistic baseline** | **0.07112 ± 0.01866** | **0.00260 ± 0.00151** | 0.11246 ± 0.01727 | **0.02583 ± 0.00777** | 0.8993 |
| learned (bagged GBM, Platt) | 0.07330 ± 0.01964 | 0.00516 ± 0.00284 | 0.11224 ± 0.01684 | 0.03929 ± 0.01064 | 0.8950 |

The learned model loses to the logistic baseline on every calibration measure
and on the proper score, and ties it on resolution. The climatology baseline
has by far the worst Brier score — it has almost no resolution, which is what
a climatology is — but its reliability term is within a factor of 2.5 of the
learned model's, which is the honest way to read "a constant forecast is hard
to beat on calibration alone".

On the single pinned split in `tests/test_regression.py` (a 6 h, seed 7
dataset) the climatology baseline has the *lowest* reliability term of the
three. Both orderings are recorded; neither is the one the project would have
preferred.

### 9.1 Brier decomposition, split seed 0

| predictor | bins | Brier | REL | RES | UNC | residual | occupied bins |
|---|---|---|---|---|---|---|---|
| climatology | equal-10 | 0.18479 | 0.00545 | 0.02908 | 0.20836 | 5.35e-05 | 5 |
| climatology | distinct | 0.18479 | 0.00597 | 0.02954 | 0.20836 | **0.00e+00** | 7 |
| logistic | equal-10 | 0.08818 | 0.00476 | 0.12517 | 0.20836 | 2.30e-04 | 8 |
| logistic | distinct | 0.08818 | 0.08818 | 0.20836 | 0.20836 | **−3.82e-15** | 517 |
| learned | equal-10 | 0.09220 | 0.00979 | 0.12501 | 0.20836 | −9.44e-04 | 10 |
| learned | distinct | 0.09220 | 0.09220 | 0.20836 | 0.20836 | **−3.86e-15** | 517 |

The equal-width residual is **not** rounding error: it is the within-bin
component of Stephenson, Coelho & Jolliffe 2008, *Weather and Forecasting*
23(4), 752-757. On distinct-value bins the three-term identity is exact, which
is the check. The distinct-value rows also expose the degeneracy that makes
them useless as a calibration measurement for a continuous predictor — one row
per bin, so REL collapses onto the Brier score and RES onto UNC.

### 9.2 Bootstrap intervals, split seed 0, 2000 resamples

| predictor | Brier | 95 % CI |
|---|---|---|
| climatology | 0.18479 | 0.16261 - 0.20827 |
| logistic | 0.08818 | 0.07448 - 0.10249 |
| learned | 0.09220 | 0.07781 - 0.10702 |

The logistic and learned intervals overlap substantially, so the logistic
baseline's win on Brier score is **not** significant at this sample size. Its
win on the reliability term across five seeds (0.00260 ± 0.00151 against
0.00516 ± 0.00284) is the more robust part of the result.

### 9.3 Leakage measurement

Row-wise splitting puts the same link on both sides. Measured optimism
(grouped Brier minus row-wise Brier, positive means the row-wise split
flatters the predictor):

| predictor | grouped | row-wise | optimism |
|---|---|---|---|
| climatology | 0.18479 | 0.17044 | +0.01435 |
| logistic | 0.08818 | 0.08568 | +0.00249 |
| learned | 0.09220 | 0.08546 | +0.00674 |

The learned model gains 2.7x more from leakage than the logistic baseline,
which is what a higher-capacity model is expected to do. All results reported
elsewhere use the grouped split.

### 9.4 Uncertainty output

Learned-model ensemble standard deviation on the held-out split: mean 0.0411,
max 0.2599. This is epistemic spread between bootstrap members only. It does
not capture the irreducible log-normal scintillation fade in the
data-generating process; the probability itself expresses that.

### 9.5 What this does not establish

The labels come from the capacity models this package ships, driven by a
latent weather and scintillation state. Agreement here is self-consistency.
**No claim about real link availability follows from these numbers.** See
`DATASET_CARD.md`.

Script: `validate_calibration.py` · Output: `validate_calibration_output.txt`
· Figure: `reliability_diagram.png`

---

## 10. Performance

Container measurement, one CPU core shared with other workloads. Not a
measurement on flight or edge hardware. Environment block and measurement
method are in the output file.

| stage | repeats | min | median | max | peak Python allocation |
|---|---|---|---|---|---|
| ephemeris, 24 sat × 181 epochs | 3 | 11.176 ms | 11.468 ms | 11.489 ms | 238 KiB |
| contact scan, 276 ISL + 120 ground pairs | 3 | 1048.968 ms | 1165.419 ms | 1192.547 ms | 115 KiB |
| time-expanded unroll, 180 slots | 3 | 66.784 ms | 68.281 ms | 75.537 ms | 3938 KiB |
| Dijkstra, one pair | 3 | 0.021 ms | 0.026 ms | 0.075 ms | 2.9 KiB |
| Dijkstra, all 27 nodes to one station | 3 | 2.351 ms | 2.391 ms | 2.603 ms | 28 KiB |
| ILP max flow (HiGHS), one pair | 1 | 1237.314 ms | — | — | 6100 KiB |
| dataset generation, 18 h horizon | 1 | 6466.146 ms | — | — | 2475 KiB |
| logistic baseline fit | 3 | 5.295 ms | 5.332 ms | 18.666 ms | 308 KiB |
| learned model fit, 5 members | 1 | 953.766 ms | — | — | 3811 KiB |
| learned model inference, 517 rows | 3 | 51.078 ms | 52.035 ms | 52.579 ms | 111 KiB |

Run-to-run spread on a shared single core is substantial: an earlier run of
the same script gave 23.9 ms for the ephemeris stage and 1581 ms for the ILP.
That spread is the reason the minimum, median and maximum are all printed and
the mean is not.

Graph size for the timed scenario: 27 nodes, 338 contact windows, 4887
time-expanded nodes, 12 354 edges.

Measured contact-scan scaling in satellite count: fitted exponent
**1.955**, against the expected 2 for the pair loop (48.992, 186.245 and
421.120 ms minimum at 8, 16 and 24 satellites).

Measured `time.perf_counter` resolution in this container: 6.1e-08 s.

Peak memory is `tracemalloc`, which counts Python-level allocations only, so
it is a lower bound: memory held inside the SGP4 C++ propagator and inside
HiGHS is not counted.

Script: `validate_benchmark.py` · Output: `validate_benchmark_output.txt`

---

## 11. Test suite

From an actual run of `python -m pytest tests/ -q --junitxml=...` in this
repository, read from the junit XML and not from stdout:

| metric | value |
|---|---|
| tests collected | **387** |
| failures | **0** |
| errors | **0** |
| skipped | **0** |
| wall time | 22.3 s |

`ruff check src/ tests/ examples/ validation/` reports no findings at
`line-length = 100`.

---

## 12. Things that are NOT validated

Stated so they are not mistaken for omissions.

1. **The `pulp` solve.** Structural equality only; see section 5.3.
2. **Real link availability.** The learned predictor is validated against a
   dataset this package generated. There is no comparison with measured link
   outages anywhere in this repository.
3. **SGP4 accuracy against truth orbits.** Section 1 checks the wrapper, not
   the theory.
4. **Atmospheric attenuation against measurement.** The Kim-Kruse and cosecant
   models are implemented and their algebra is checked; neither is compared
   with a measured attenuation time series.
5. **The queueing models against a simulation.** The closed forms are checked
   against each other and against hand values; no discrete-event simulation is
   run to confirm the stationary means on a real contact pattern, and the
   module docstring states that a contact window is not a stationary M/M/1
   server in the first place.
6. **Any number on flight or edge hardware.** Every timing figure is a
   container measurement.
