# Changelog

All notable changes to CMGSteer are documented in this file.

## [0.1.0] - 2026-09-02

Initial release.

### Added

- `cmgsteer.arrays`: `CMGArray` around gimbal axes, reference axes and rotor
  momenta, with the momentum map `h(delta) = sum_i h0_i h_hat_i(delta_i)` and
  its Jacobian `A = dh/ddelta`; builders for the standard pyramid
  (`pyramid_array`, skew angle `arctan(4/3)` = 53.13 deg by default), roof
  arrays with parallel gimbal pairs (`roof_array`) and arbitrary geometries
  (`general_array`); failure modelling through `with_locked`, which keeps a
  failed CMG's rotor momentum in the momentum map and removes its column from
  the Jacobian. The torque convention `tau = -A ddelta` is stated in every
  docstring.
- `cmgsteer.singularity`: the singularity measure `m = sqrt(det(A A^T))`
  computed as a product of singular values; `sigma_min`, condition number and
  singular direction; an analytic `manipulability_gradient` derived from the
  singular value decomposition so that it stays finite at `m = 0`;
  `singular_configuration`, which builds an analytically singular configuration
  from a direction and a sign vector; `classify_singularity`, which separates
  external (saturation) from internal singularities and elliptic from
  hyperbolic ones by the signature of the second-order form on `null(A)`; and
  `singular_surface` / `momentum_envelope` for mapping the singular set in
  momentum space.
- `cmgsteer.steering`: four laws over one `SteeringResult` —
  `pseudo_inverse_steer`, `sr_inverse_steer` (Nakamura & Hanafusa 1986;
  Bedrossian et al. 1990) with an adaptive robustness parameter,
  `gsr_inverse_steer` (Wie, Bailey & Heiberg 2001) with deterministic
  off-diagonal dither, and a null-motion term accepted by all of them.
  `sr_torque_error_closed_form` evaluates the exact SVD expression the SR
  inverse's error must satisfy. `apply_rate_limit` offers component-wise
  clipping or direction-preserving scaling and records what it changed.
- `cmgsteer.nullmotion`: `null_space_basis`, `null_projector`, a signed
  `unit_null_vector`, and the classical `GradientNullMotion` (Yoshikawa 1985)
  and `PreferredAngleNullMotion` (Vadali, Walker & Oh 1990) policies over one
  `NullMotionPolicy` interface.
- `cmgsteer.simulate`: `TorqueProfile`, `rest_to_rest_profile`,
  `constant_profile` and `run_steering`, which records the instantaneous torque
  error and the momentum error separately so that the steering error and the
  explicit-Euler integration error are never conflated.
- `cmgsteer.dataset`: seeded `manoeuvre_suite` generation and
  `generate_policy_dataset`, which labels visited states with the null-motion
  coefficient a 25-step lookahead oracle would choose.
- `cmgsteer.ml.LearnedNullMotion`: an ensemble of five scikit-learn MLPs trained
  to imitate that oracle, with per-prediction ensemble spread and a scalar
  confidence.
- `cmgsteer._fast.FastStepper`: a fused single-SVD steering step used only by
  dataset generation, 5.35x faster than the public path and verified identical
  to 1e-15.
- `python -m cmgsteer` CLI: `array`, `singularity`, `steer` and `manoeuvre`,
  exiting 1 when a computed result fails the subcommand's acceptance check and
  2 on invalid input.
- Validation suite (`validation/`), six scripts with saved raw stdout, and
  `docs/REQUIREMENTS.md` with 20 numbered requirements and a verification
  matrix.
- 285 tests including Hypothesis property tests, failure-mode tests for internal
  and external singularities, gimbal-rate saturation and array degeneracy after
  a CMG failure, a regression suite with pinned seeded outputs, and a
  performance benchmark.

### Measured results reported as they came out

- The learned null-motion policy's held-out label regression has
  **R² = −0.0939**, worse than predicting the training mean.
- Over 16 held-out manoeuvres, neither the learned policy nor the classical
  gradient null motion differs from plain SR-inverse steering by an amount whose
  bootstrap 95% interval excludes zero, on path momentum error, net momentum
  error or minimum singularity measure. The learned policy costs 16.4x the
  runtime of the law it does not distinguishably improve on.
- The only distinguishable difference in that benchmark is that the plain
  pseudo-inverse is **worse** than the SR inverse (+49.8% path momentum error,
  interval excluding zero).
- The confidence output correlates with the label error at r = −0.166 with
  non-monotone deciles; it is not usable as a gate.

### Known limitations

Single-gimbal CMGs at constant rotor speed only; no gimbal dynamics or
rate-servo lag; no vehicle dynamics, controller or orbit; explicit Euler
integration of the gimbal angles, so the momentum error is first order in `dt`;
no gimbal-angle travel limits; scalar null-motion coefficients only, so the
learned policy needs a one-dimensional null space. See `README.md` and
`docs/REQUIREMENTS.md` §4.
