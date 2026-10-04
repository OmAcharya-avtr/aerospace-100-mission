# ConstelLink — requirements and verification matrix

Validation level 3. Research-grade. Not flight-qualified, not certified, not
approved for operational aerospace use.

Every requirement below is verified by something that runs in this repository.
Where a requirement is verified only partially, the matrix says so and names
what is missing, rather than marking it closed.

Shorthand used in the matrix:

| tag | meaning |
|---|---|
| `T` | `tests/<file>::<test>` — runs under `python -m pytest tests/ -q` |
| `V` | `validation/<script>.py` — raw output in `validation/<script>_output.txt` |
| `E` | `examples/<script>.py` — produces a PNG in `screenshots/` |

---

## Functional requirements

### REQ-01 — Propagation is delegated, not reimplemented

The package SHALL propagate orbits through the `sgp4` package and SHALL NOT
contain an independent SGP4 implementation. The wrapper SHALL reproduce the
verification output shipped with `sgp4` to a stated tolerance.

**Rationale.** A second SGP4 would be a second source of error with no second
source of truth.

**Acceptance.** Position residual against `tcppver.out` for satellite 00005
below 1e-6 km and velocity residual below 1e-7 km/s over the full 4320 min
verification span.

---

### REQ-02 — Inter-satellite visibility uses an exact segment test

Inter-satellite line-of-sight SHALL be decided by the exact minimum distance
from the Earth's centre to the segment joining the two satellites, compared
against a blocking sphere of radius `R_e + h_grazing`. The grazing altitude
SHALL be a caller-settable parameter with a documented default.

**Acceptance.** For equal radii, the test flips at the closed-form
`gamma_max = 2 arccos(r_block / r)` to within 1e-5 deg.

---

### REQ-03 — Ground access uses the elevation mask and a range limit

Satellite-to-ground contact SHALL be decided jointly by a station elevation
mask and a terminal maximum slant range, and the window edges SHALL be refined
by bisection to a caller-settable tolerance.

**Acceptance.** For a two-body circular equatorial orbit and an equatorial
station, the Earth central angle at the reported rise and set times matches
the closed form `lambda_max = arccos((R_e/r) cos eps) - eps` to within
1e-4 deg, and the pass duration matches `2 lambda_max / (n - omega_E)` to
within 0.5 s.

---

### REQ-04 — Walker generation states its own mean-motion caveat

`walker_delta` SHALL produce the requested `i: T/P/F` pattern angles exactly
and SHALL document that the realised orbit radius differs from the requested
one because SGP4 consumes a Kozai mean motion while the generator sets a
Kepler mean motion. The realised radius SHALL be measured, not assumed.

**Acceptance.** Pattern angles reproduce the definition to within 1e-6 deg;
the mean realised radius is within 5 km of the requested radius and the
measured offset is reported in `validation/validate_walker_output.txt`.

---

### REQ-05 — The contact graph is a first-class object

The package SHALL expose a time-varying contact graph that answers: which
links are open in a given interval, what the connected components are, and
whether the graph is partitioned — both per slot and over the union of the
horizon.

**Acceptance.** Components and partition detection agree with hand-built
instances; a slot-wise partition is detectable even when the union graph is
connected.

---

### REQ-06 — The time-expanded graph is a layered DAG with explicit costs

The unrolling SHALL produce hold edges (one slot of waiting) and transmit
edges (one slot plus the one-way propagation delay), with per-edge capacity
derived from a caller-supplied rate function. Partial slot coverage SHALL
scale the capacity by the realised overlap fraction, and the default SHALL
require full coverage.

**Acceptance.** Edge counts, costs and capacities match hand calculations;
capacity scales with the overlap fraction as specified.

---

### REQ-07 — Minimum-latency routing agrees with exhaustive enumeration

Routing SHALL use Dijkstra over the time-expanded graph and SHALL agree with
exhaustive route enumeration on instances small enough to enumerate, both on
the minimum latency and on the existence of a route.

**Acceptance.** Zero latency gap over 3 hand instances and 40 random
instances, including 10 with no route.

---

### REQ-08 — Maximum-flow scheduling agrees with brute force

Scheduling SHALL solve an integer maximum-flow program on the time-expanded
network and SHALL agree with exhaustive enumeration of the integer capacity
box on instances small enough to enumerate.

**Acceptance.** Zero mismatches over 2 hand instances and the random instances
whose capacity box fits the enumeration budget.

---

### REQ-09 — The `pulp` model and the solved program are the same program

The package SHALL build the maximum-flow program as a `pulp` model for export
and SHALL verify that model is structurally identical to the program actually
solved — variable count, bounds, integrality, objective and every conservation
row.

**Known limitation, stated here rather than in a footnote.** `pulp` 4.0.0
ships no bundled CBC binary and the build environment has no external MILP
solver, so no `pulp` objective value was produced. The solved objective comes
from `scipy.optimize.milp` (HiGHS). This requirement is therefore **partially
verified**: the structural equality is checked, the `pulp` solve is not.

---

### REQ-10 — Link capacity reduces to Friis when losses are zeroed

The RF chain SHALL reduce exactly to the Friis transmission formula when every
loss term is zero, and the free-space path loss SHALL match the standard
engineering form to floating-point precision.

**Acceptance.** Received power from the dB chain agrees with a linear-domain
Friis evaluation to a relative 1e-12 across four range/frequency cases.

---

### REQ-11 — The optical model matches its closed forms and a quadrature

The Gaussian geometric capture fraction SHALL match a direct numerical
integration of the beam intensity over the aperture, and the pointing loss in
dB SHALL match `(20/ln 10) (theta_err/theta_half)^2` identically.

**Acceptance.** Relative agreement with quadrature below 1e-8 over
`a/w` from 0.01 to 3; pointing-loss agreement below 1e-12 dB.

---

### REQ-12 — Atmospheric models refuse to extrapolate out of validity

The ground-leg slant-path scaling SHALL raise rather than return a number
below the elevation at which the plane-parallel assumption fails, and the
visibility model SHALL raise on non-physical inputs.

**Acceptance.** `slant_path_attenuation_db` raises below 10 deg by default and
names the model and its limitation in the message.

---

### REQ-13 — Queueing models refuse unstable regimes

Queueing delay SHALL raise rather than return infinity or a negative number
when the offered utilisation reaches or exceeds one, and the M/D/1 waiting
time SHALL be exactly half the M/M/1 waiting time at matched utilisation.

**Acceptance.** Both properties verified over a utilisation sweep.

---

### REQ-14 — Two deterministic baselines precede the learned predictor

A climatology baseline and a logistic-regression baseline SHALL be implemented
and evaluated on identical grouped splits before and alongside the learned
link-availability predictor. The headline metric SHALL be calibration, and the
result SHALL be reported whichever predictor wins.

**Acceptance.** Three predictors scored on five grouped splits with reliability,
resolution, ECE, log loss and accuracy reported together. **Measured outcome:
the logistic baseline wins on Brier score and on the reliability term; the
learned model wins neither.** That is recorded in README.md, MODEL_CARD.md and
`tests/test_regression.py::test_the_pinned_ranking_is_the_measured_one`.

---

### REQ-15 — The learned predictor exposes an uncertainty output

The learned model SHALL expose a per-sample uncertainty alongside its
probability, and the documentation SHALL state what that uncertainty does and
does not capture.

**Acceptance.** Ensemble standard deviation is non-zero and documented as
epistemic spread only, not as the irreducible scintillation noise in the
data-generating process.

---

### REQ-16 — Verification statistics carry their own uncertainty

Reported scores SHALL be accompanied by bootstrap intervals, and the Brier
decomposition SHALL report its identity residual rather than asserting an
exactness it does not have for equal-width bins.

**Acceptance.** Percentile bootstrap intervals on every predictor; the
equal-width residual is reported as the Stephenson-Coelho-Jolliffe within-bin
component and the distinct-value decomposition is exact to 1e-12.

---

### REQ-17 — Failure modes refuse rather than guess

The package SHALL behave correctly on: an empty contact graph, a partitioned
constellation, a satellite lost mid-horizon, and a TLE epoch far from the
requested window. The last SHALL **raise**, never silently extrapolate.

**Acceptance.** Each case verified by a dedicated test class and by a
validation script; the epoch guard raises in both time directions, is applied
to the whole requested window rather than only its first sample, and its
message names the epoch, the limit and the reason.

---

### REQ-18 — Every reported number is reproducible and pinned

Every number in README.md and VALIDATION.md SHALL come from a script in this
repository, and a regression suite SHALL pin seeded outputs so an unintended
change in geometry, routing, capacity or model training fails a test.

**Acceptance.** Regression suite pins window counts and durations, graph size,
route latency, flow optimum, capacity values, dataset statistics and predictor
scores; reproduction commands are listed in README.md.

---

## Verification matrix

| Req | Verified by | Evidence |
|---|---|---|
| REQ-01 | `V validate_sgp4_vector`, `T test_constellation.py::test_satellite_from_tle_and_epoch_reconstruction` | max \|dr\| 6.819e-09 km, max \|dv\| 7.705e-10 km/s, 13 reference rows |
| REQ-02 | `V validate_contact_windows` (checks 3a, 3b), `T test_geometry.py::test_isl_clear_flips_at_gamma_max` | flip residual 1.908e-14 deg; limb at window edge within 9.87e-07 km |
| REQ-03 | `V validate_contact_windows` (checks 1, 2), `T test_contacts.py::test_two_body_ground_window_matches_closed_form_duration` | edge central-angle residual ≤ 1.30e-06 deg; duration residual ≤ 1e-06 s |
| REQ-04 | `V validate_walker`, `T test_constellation.py::test_walker_mean_motion_matches_two_body_relation` | radius offset −0.113 to −0.127 km; worst pattern angle error 2.842e-14 deg |
| REQ-05 | `T test_graph.py` (components, partition, adjacency), `V validate_failure_modes` case 2, `E example_contact_graph` | 5 to 12 components per 60 s slot over a 3 h horizon; 3 in the union graph |
| REQ-06 | `T test_graph.py::test_time_expanded_edge_counts_and_costs`, `::test_time_expanded_partial_overlap_scales_capacity` | hand-checked edge counts, costs and capacities |
| REQ-07 | `V validate_routing`, `T test_routing.py::test_dijkstra_equals_enumeration_on_random_small_graphs` | worst latency gap 0.000e+00 s over 40 instances, 513 routes enumerated |
| REQ-08 | `V validate_ilp` parts 1-2, `T test_flow.py::test_ilp_matches_brute_force_on_random_small_instances` | 0 mismatches over 2 hand + 27 random instances |
| REQ-09 | `V validate_ilp` part 3, `T test_flow.py::test_pulp_model_matches_the_matrix_program` | 5 of 5 structural checks pass; **pulp objective NOT RUN — no solver available** |
| REQ-10 | `V validate_capacity` checks 1-2, `T test_capacity.py::test_rf_link_reduces_to_friis_with_losses_zeroed` | relative error ≤ 7.9e-15; FSPL constant K = 92.447783222 dB, diff 0.00e+00 dB |
| REQ-11 | `V validate_capacity` checks 4-5, `T test_capacity.py::test_optical_geometric_capture_closed_form` | quadrature agreement ≤ 5.7e-14 relative; pointing loss ≤ 1.78e-15 dB |
| REQ-12 | `T test_capacity.py::test_slant_path_refuses_below_validity_elevation`, `V validate_failure_modes` case 5 | raises below 10 deg; message names Ippolito 2008 Ch. 4 and the `itur` package |
| REQ-13 | `T test_queueing.py::test_unstable_queue_raises`, `::test_md1_waiting_time_is_half_of_mm1` | exact factor of 2 over rho in {0.1 … 0.95} |
| REQ-14 | `V validate_calibration`, `E example_calibration`, `T test_regression.py::test_the_pinned_ranking_is_the_measured_one` | logistic REL 0.00260 ± 0.00151, learned 0.00516 ± 0.00284, climatology 0.00645 ± 0.00487 over 5 seeds |
| REQ-15 | `T test_availability.py::test_learned_model_outputs_probabilities_and_spread` | ensemble std mean 0.0411, max 0.2599 on the held-out split |
| REQ-16 | `V validate_calibration` part 3, `T test_metrics.py::test_equal_width_residual_is_the_within_bin_component` | distinct-value identity residual ≤ 3.9e-15; equal-width residual 5.4e-05 to 9.4e-04 |
| REQ-17 | `T test_failure_modes.py` (4 classes, 33 tests), `V validate_failure_modes` | all 5 validation cases PASS |
| REQ-18 | `T test_regression.py` (12 pinned tests), README "Reproducing every number" | 387 tests collected, 0 failed, 0 errors, 0 skipped |

---

## Uncertainty analysis

Reported in full in `validation/validate_uncertainty_output.txt`. Four
independent error sources, kept separate because they do not combine:

| source | method | measured |
|---|---|---|
| propagation input | perturb mean anomaly by `ds / a`, recompute windows | 0.1465 s max edge shift per 1 km of along-track error; 0.139-0.147 s/km over 1-20 km |
| discretisation (grid step) | recompute one ISL's windows at 5-600 s steps | window count unchanged; total duration varies by ≤ 0.006 s |
| discretisation (bisection) | recompute at refinement tolerances 1e-6 to 10 s | edge error bounded by the tolerance: 0.000409 s at 1e-3 s, 3.56 s at 10 s |
| link-budget parameters | Monte Carlo, 4000 draws, stated 1-sigma inputs | RF p95/p5 spread 3.184 dB against an analytic quadrature prediction of 3.155 dB |

The ILP and routing results carry no stochastic uncertainty: they are exact
integer and shortest-path optima on the graph they are given. Their
uncertainty is entirely inherited from the contact graph and the capacity
model above.

---

## Performance benchmark

Reported in full in `validation/validate_benchmark_output.txt`, with the
environment block and the measurement method. One CPU core, shared with other
workloads, so the minimum is the cleanest estimate and the maximum shows
contention. These are container numbers and are not measurements on flight or
edge hardware.

| stage | min | median | notes |
|---|---|---|---|
| ephemeris, 24 sat × 181 epochs | 11.2 ms | 11.5 ms | |
| contact scan, 276 ISL + 120 ground pairs | 1049 ms | 1165 ms | `O(n_sat^2 n_t)` |
| time-expanded unroll, 180 slots | 66.8 ms | 68.3 ms | 12 354 edges |
| Dijkstra, one pair | 0.021 ms | 0.026 ms | |
| ILP max flow (HiGHS), one pair | 1237 ms | — | one repeat |
| dataset generation, 18 h horizon | 6466 ms | — | 1974 rows |
| learned model fit, 5 members | 954 ms | — | |

Measured contact-scan scaling exponent in satellite count: 1.955 against the
expected 2. Run-to-run spread on a shared core is large; an earlier run of the
same script gave 23.9 ms for the ephemeris stage and 1581 ms for the ILP.

---

## Out of scope — stated so it is not mistaken for an omission

* Attitude, pointing control and acquisition dynamics. Pointing error enters
  the optical model as a static angle, not as a tracking loop.
* Protocol behaviour: no CCSDS framing, no bundle protocol, no congestion
  control. The routing here is a graph computation, not a protocol.
* Interference, spectrum coordination and regulatory constraints.
* Relativistic corrections to propagation delay.
* ITU-R P-series atmospheric models. The ground leg uses a textbook cosecant
  scaling of a user-supplied zenith attenuation; readers who need the ITU-R
  models should use the `itur` package.
* Hardware-in-the-loop or on-target timing. Every number here is a
  workstation/container measurement.
