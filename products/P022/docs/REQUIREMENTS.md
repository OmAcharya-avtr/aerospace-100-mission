# CMGSteer — Requirements and Verification Matrix

**Product:** P022 CMGSteer · **Version:** 0.1.0 · **Status:** TESTING
**Validation level:** 3 · **Licence:** AGPL-3.0-or-later
**Date of this revision:** 2026-09-02

This document states what `cmgsteer` is required to do, in numbered,
individually verifiable terms, and names for each requirement the test or
validation script that verifies it. Requirements are written so that a
verification can *fail*: "shall reproduce X to within Y" rather than "shall be
accurate".

Paths are relative to `products/P022/`. Test identifiers are pytest node ids;
validation identifiers are scripts in `validation/` whose captured output is
stored alongside them as `*_output.txt`.

---

## 1. Scope and definitions

CMGSteer models single-gimbal control-moment-gyro arrays: their geometry and
momentum map, the Jacobian every steering law inverts, the structure of the
singular set, four steering laws, null-motion reconfiguration, and the torque
error that accumulates through a manoeuvre. It contains no vehicle dynamics, no
attitude controller and no orbit.

| Term | Meaning in this document |
|---|---|
| *array* | a set of single-gimbal CMGs with fixed gimbal axes and constant rotor speed |
| *momentum map* | `h(delta) = sum_i h0_i h_hat_i(delta_i)` [N·m·s] |
| *Jacobian* | `A(delta) = dh/ddelta` [N·m·s/rad] |
| *torque* | `tau = -A ddelta/dt` [N·m], the torque delivered **to the vehicle** |
| *singularity measure* | `m = sqrt(det(A A^T)) = prod_k sigma_k` [(N·m·s/rad)³] |
| *external singularity* | a singular configuration at which every `eps_i = sign(u . h_hat_i)` has the same sign; the momentum is on the envelope |
| *internal singularity* | any other singular configuration |
| *null motion* | a gimbal rate in `null(A)`, which delivers no torque |
| *path momentum error* | `sum_k |(-tau_k dt) - (h_{k+1} - h_k)|` [N·m·s] over a run |
| *net momentum error* | the norm of the same per-step errors summed as vectors |
| *published value* | a number from a cited, externally published source |

---

## 2. Functional requirements

### R-01 — Array geometry
The package shall construct pyramid and roof CMG array geometries from a skew
angle and a rotor momentum, shall accept an arbitrary set of gimbal axes, and
shall reject a reference axis parallel to its gimbal axis, a non-positive rotor
momentum, and a skew angle outside `(0, 90)` degrees with a diagnostic naming
the offending input.

### R-02 — Momentum map against the published closed form
The four-CMG pyramid momentum map shall agree with the closed form quoted
throughout the SGCMG literature to within **1e-13 N·m·s** over 2000 random
configurations, using a construction that does not contain that closed form.

### R-03 — Jacobian against numerical differentiation
The analytic Jacobian shall agree with a central-difference derivative of the
momentum map to within **1e-8 N·m·s/rad** at the optimal step, for both
geometries, over 400 random configurations, and the worst deviation and the
step at which it occurs shall be reported rather than asserted.

*Rationale:* a Jacobian that is not the derivative of the momentum map the
package itself integrates makes every downstream number meaningless.

### R-04 — Torque sign convention
`array.torque(delta, ddelta)` shall equal `-A(delta) ddelta` exactly, and the
convention shall be stated in the package docstring, in every steering-law
docstring and in the README.

### R-05 — Singularity measure and singular direction
The package shall compute `m`, `sigma_min`, the condition number and the
singular direction `u`, and `m` shall equal `sqrt(det(A A^T))` to within 1e-9
relative wherever the determinant is computable.

### R-06 — Analytic singular configurations
Given a body direction and a sign vector, the package shall return the gimbal
angles of the corresponding analytically singular configuration, and the
numerical singularity measure there shall be below **1e-13** for both
geometries over 8000 constructed points. A direction parallel to a gimbal axis
shall raise a diagnostic rather than return a wrong configuration.

### R-07 — Singularity classification
The package shall classify a singular configuration as external or internal
from the sign vector `eps_i = sign(u . h_hat_i)`, and as elliptic, hyperbolic
or degenerate from the signature of the second-order form restricted to
`null(A)`. **Every** external singularity shall be classified elliptic, and
both passabilities shall be observed among internal singularities.

### R-08 — Singular-surface mapping
The package shall map any singular surface in momentum space by sweeping the
singular direction over the sphere at a fixed sign vector, shall provide the
outer momentum envelope as the all-positive case, and every mapped point shall
have `m < 1e-13`.

### R-09 — Pseudo-inverse exactness
Away from a singularity (`m >= 0.1` for a unit-momentum four-CMG array) the
Moore-Penrose pseudo-inverse shall reproduce the commanded torque to within
**1e-12 N·m**, and the degradation inside the near-singular bands shall be
reported as a table rather than excluded.

### R-10 — SR-inverse closed-form torque error
The singularity-robust inverse's torque error shall equal
`sum_k [lam/(sigma_k^2+lam)] (u_k . tau) u_k` to within **1e-13 N·m**
componentwise, over ten values of `lam` spanning eleven decades and 200 random
states, for both geometries, and the result shall be presented as a table of
error against `lam`.

### R-11 — Generalised SR inverse
The generalised SR inverse shall reduce exactly to the SR inverse when its
dither amplitude is zero (deviation **0** rad/s), and its dither shall be
deterministic in the supplied time so that a run is reproducible.

### R-12 — Null motion
The package shall provide the null-space basis, the orthogonal projector onto
it, a signed unit null vector with a documented sign convention, and classical
gradient and preferred-angle null-motion policies. Adding any null motion shall
leave the delivered torque unchanged to within **1e-10 N·m**, and a request for
a scalar null coefficient at a configuration whose null space is not
one-dimensional shall raise a diagnostic.

### R-13 — Gimbal-rate saturation
The package shall apply a symmetric gimbal-rate limit in either a
component-wise (`clip`) or a direction-preserving (`scale`) mode, shall record
how many components were limited on every step, and `scale` shall preserve the
delivered torque direction to within **1e-9** in cosine while `clip` shall not.

### R-14 — Torque error accounting
A steering run shall report both the instantaneous torque error, evaluated at
the step's own gimbal angles, and the momentum error, which additionally
contains the integration error of the explicit Euler gimbal update. The
momentum error shall be first order in the step size, with the measured ratio
between successive halvings inside **[1.8, 2.2]**.

### R-15 — Failure modes
The package shall model a locked (failed) gimbal, keeping its rotor momentum in
the momentum map and removing its column from the Jacobian; shall raise a
diagnostic naming the shortfall when fewer than three gimbals remain free; and
shall behave correctly and finitely at external singularities, internal
singularities, under gimbal-rate saturation, and after a CMG failure.

### R-16 — Learned null-motion policy with uncertainty
The learned policy shall predict a null-motion coefficient from the current
state alone, shall expose a per-prediction ensemble spread and a scalar
confidence rather than only a point estimate, and shall be benchmarked against
(a) plain SR-inverse steering, (b) classical gradient null motion and (c) the
plain pseudo-inverse, over a seeded held-out manoeuvre suite, with paired
bootstrap confidence intervals.

### R-17 — Honest reporting of the AI result
Whichever method wins each benchmark metric shall be reported as measured,
including every regime in which the classical method wins or in which no method
is distinguishable from another; a difference whose bootstrap interval contains
zero shall be reported as indistinguishable, never as a win.

### R-18 — Determinism, regeneration and compute budget
Every dataset, manoeuvre suite and simulation result shall be reproducible
bit-for-bit from an integer seed; no data or model artefact shall be stored as a
file; and the complete training-plus-benchmark validation shall finish in under
**180 s** on two CPU cores.

### R-19 — Uncertainty analysis
The package's sensitivity to gimbal-angle measurement error, rotor momentum
error and their effect on the singularity measure shall be quantified by Monte
Carlo and compared with first-order analytic predictions, with the mean ratio
of the two within **5%** at the smallest sigma tested.

### R-20 — Command-line interface
`python -m cmgsteer` shall expose array description, singularity
classification, single-step steering and a seeded manoeuvre run as subcommands,
shall exit 1 when its own acceptance check fails and 2 on invalid input, and
shall emit a one-line diagnostic rather than a traceback for invalid input.

---

## 3. Verification matrix

| Req | Verified by | Kind |
|---|---|---|
| R-01 | `tests/test_arrays.py::TestPyramidGeometry` · `::TestRoofArray` · `::TestGeneralArrayAndValidation` | test |
| R-02 | `validation/validate_jacobian.py` §3 · `tests/test_arrays.py::TestMomentumMap::test_matches_the_published_closed_form` | validation + test |
| R-03 | `validation/validate_jacobian.py` §1 · `tests/test_arrays.py::TestJacobian::test_matches_central_differences` · `tests/test_properties.py::TestGeometryProperties` | validation + test |
| R-04 | `validation/validate_jacobian.py` §4 · `tests/test_arrays.py::TestTorqueConvention` · `tests/test_properties.py::TestSteeringProperties::test_roof_array_torque_is_minus_jacobian_times_rates` | validation + test |
| R-05 | `tests/test_singularity.py::TestMeasureKnownAnswers` · `validation/validate_singularity.py` §2b | validation + test |
| R-06 | `validation/validate_singularity.py` §1 · `tests/test_singularity.py::TestAnalyticSingularConfigurations` · `tests/test_properties.py::TestGeometryProperties::test_analytic_singular_configurations_have_zero_measure` | validation + test |
| R-07 | `validation/validate_singularity.py` §2c · `tests/test_singularity.py::TestClassification` · `tests/test_failure_modes.py::TestExternalSingularity` · `::TestInternalSingularity` | validation + test |
| R-08 | `validation/validate_singularity.py` §2d · `tests/test_singularity.py::TestSurfaces` | validation + test |
| R-09 | `validation/validate_steering.py` §1 · `tests/test_steering.py::TestPseudoInverse` · `tests/test_properties.py::TestSteeringProperties::test_pseudo_inverse_is_exact_away_from_singularity` | validation + test |
| R-10 | `validation/validate_steering.py` §2, §3 · `tests/test_steering.py::TestSRInverse::test_matches_the_closed_form_error` · `tests/test_properties.py::TestSteeringProperties::test_sr_error_matches_the_closed_form` | validation + test |
| R-11 | `validation/validate_steering.py` §5 · `tests/test_steering.py::TestGSR` | validation + test |
| R-12 | `tests/test_nullmotion.py` (all classes) · `tests/test_properties.py::TestSteeringProperties::test_null_motion_never_changes_the_delivered_torque` | test |
| R-13 | `tests/test_steering.py::TestRateLimit` · `tests/test_failure_modes.py::TestGimbalRateSaturation` · `tests/test_properties.py::TestSteeringProperties::test_rate_limit_is_respected` | test |
| R-14 | `tests/test_simulate.py::TestRunSteering::test_momentum_error_is_first_order_in_dt` · `::test_pseudo_inverse_torque_error_is_numerically_zero` · `validation/VALIDATION.md` §"What failed" | validation + test |
| R-15 | `tests/test_failure_modes.py` (all four classes) · `tests/test_arrays.py::TestLocking` | test |
| R-16 | `validation/validate_nullmotion_ml.py` §2-§5 · `tests/test_ml.py::TestLearnedPolicy` · `tests/test_integration.py::TestFullPipeline` | validation + test |
| R-17 | `validation/validate_nullmotion_ml.py` §5d, §7 (recorded in `validation/VALIDATION.md` §5 and `MODEL_CARD.md` §7-§9, and in the README) · `tests/test_regression.py::TestPinnedDataset::test_oracle_beats_zero_and_gradient_loses` | validation + test |
| R-18 | `tests/test_regression.py` (all pinned outputs) · `tests/test_dataset.py::TestManoeuvreSuite::test_is_deterministic_in_the_seed` · `::TestPolicyDataset::test_is_deterministic` · `tests/test_ml.py::TestLearnedPolicy::test_training_is_deterministic` · `validation/validate_nullmotion_ml.py` §2 (40.47 s of dataset + training, 125 s total) | validation + test |
| R-19 | `validation/validate_uncertainty.py` §1-§4 | validation |
| R-20 | `tests/test_cli.py` (all classes) | test |

Performance is reported by `validation/validate_performance.py` and bounded by
`tests/test_performance.py`; it is not a numbered requirement because no
throughput target was specified for 0.1.0.

---

## 4. Requirements deliberately **not** met in 0.1.0

Recorded here so the gap is visible rather than implied.

| Ref | Not implemented | Consequence |
|---|---|---|
| N-01 | Double-gimbal and variable-speed CMGs | single-gimbal, constant rotor speed only; the momentum map and every steering law assume it |
| N-02 | Gimbal dynamics — rate-servo lag, gimbal inertia, friction, backlash, rate quantisation | the commanded rate is assumed to be achieved instantly, so the reported torque error is a lower bound |
| N-03 | Vehicle dynamics, attitude control loop, orbit, disturbances | open loop: the torque command is an input, and nothing here says whether the closed loop is stable |
| N-04 | Higher-order integration of the gimbal angles | explicit Euler only, so the momentum error is first order in `dt`; see R-14 |
| N-05 | Exact steering laws — inverse kinematics, singularity-avoidance path planning, global steering | only local, instantaneous laws are implemented |
| N-06 | Gimbal-angle limits | gimbals are assumed to rotate continuously through 360 degrees; a limited-travel gimbal is not modelled |
| N-07 | Momentum-envelope volume and containment tests | the envelope is mapped as a point cloud, not as a polytope with a membership test |
| N-08 | Null motion of dimension greater than one | the learned policy and `unit_null_vector` need a one-dimensional null space, so arrays with more than four free gimbals fall back to `null_space_basis` and the classical projector |
| N-09 | Reinforcement learning for the null-motion policy | behaviour cloning of a lookahead oracle only; no PyTorch in the target environment |
| N-10 | Rotor spin-up and spin-down transients, bearing drag, thermal effects | rotor momentum is a constant |

---

## 5. Verification status summary

Run in the 0.1.0 build session on 2026-09-02 (Python 3.11.15, numpy 2.4.4,
scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.9, 2 CPU cores):

* `python -m pytest tests/ -q` → **285 passed** in 60 s
* `ruff check src/ tests/ examples/ validation/` → **All checks passed**
* six validation scripts executed, raw output stored in `validation/`
* three examples executed, PNGs stored in `screenshots/`

Two verifications record a **measured negative result rather than a pass**;
both are described in `validation/VALIDATION.md` §5 and in `MODEL_CARD.md`:

1. R-16/R-17: the learned null-motion policy's held-out label regression has
   **R² = −0.0939**, worse than predicting the training mean, and its
   closed-loop difference from plain SR-inverse steering is indistinguishable
   from zero on every metric (path momentum error, net momentum error, minimum
   singularity measure) over 16 manoeuvres. So is the classical gradient null
   motion's. The only distinguishable difference in the benchmark is that the
   plain pseudo-inverse is **worse** than the SR inverse.
2. R-16: the confidence output correlates with the label error at r = −0.166
   and its deciles are non-monotone, so it ranks trustworthiness too weakly to
   gate anything. The ensemble spread is 0.287 of the rms error and must not be
   used as a variance.
