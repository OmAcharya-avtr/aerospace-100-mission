# Changelog

## 0.1.0 — 2026-08-01

Initial release (status: TESTING, validation Level 1 / Educational).

- `Quaternion` class: scalar-first `[w, x, y, z]`, Hamilton product, active
  rotation; normalization, conjugate/inverse, vector rotation, exp/log maps,
  SLERP. Normalize-or-raise unit-norm policy (`NORM_TOL = 1e-6`).
- Vectorized array API in `quatkit.core` (multiply, rotate, exp/log, slerp).
- Conversions: quaternion ↔ DCM (Shepperd max-component extraction),
  quaternion ↔ ZYX aerospace Euler angles (GimbalLockWarning near ±90° pitch),
  quaternion ↔ axis-angle, quaternion ↔ Rodrigues/Gibbs vector and MRP.
- Attitude error: multiplicative error quaternion, 2×vector-part small-angle
  error vector, exact angle between attitudes.
- Kinematics: q̇ = ½ q ⊗ [0, ω], RK4 propagation with per-step
  renormalization, closed-form solution for constant ω.
- Tests: known-answer, input-validation, gimbal-lock edge cases, Hypothesis
  property tests. Level-1 validation vs scipy Rotation and analytic solutions.
