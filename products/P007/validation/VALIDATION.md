# QuatKit — Validation Evidence (Level 1, Educational)

All numbers below were produced by running the scripts in this directory in the
build session on 2026-08-01 (Python 3.11.15, numpy 2.4.4, scipy 1.17.1). Raw
console output is saved alongside each script:

| Script | Output file |
|---|---|
| `check_known_rotations.py` | `known_rotations_output.txt` |
| `check_scipy_cross.py` | `scipy_cross_output.txt` |
| `check_rk4_vs_analytic.py` | `rk4_vs_analytic_output.txt` |

Rerun from `products/P007/` with `python validation/<script>.py` (deterministic;
the scipy check uses fixed seed 20260801).

Convention under validation: scalar-first `[w, x, y, z]`, Hamilton product,
active rotation `v' = q ⊗ [0, v] ⊗ q*`.

## 1. Known rotation test vectors (hand-checkable)

Reference: right-hand-rule action of 90°/180° rotations about the principal
axes (elementary result; e.g. Markley & Crassidis 2014, Sec. 2.6). Nine
vector-rotation cases (x̂→ŷ under Rz(90°), ẑ→x̂ under Ry(90°), …) and three
hand-written 90° DCMs.

- **Result: 12/12 PASS.**
- Worst-case absolute component error: **2.220e-16** (tolerance 1e-15, i.e.
  machine epsilon level).

## 2. Cross-check against `scipy.spatial.transform.Rotation`

N = **1000** uniformly random unit quaternions (Gaussian-normalized, seed
20260801; scipy stores scalar-last, converted with `np.roll`).

| Check | Max deviation over N = 1000 | Tolerance |
|---|---|---|
| `quat_to_dcm` vs `Rotation.as_matrix()` | **9.992e-16** (matrix element) | 1e-12 |
| `dcm_to_quat` (Shepperd) vs original quaternion | **5.011e-16 rad** (rotation-angle error) | 1e-12 |
| `quat_rotate` vs `Rotation.apply()` | **2.665e-15** (vector component) | 1e-12 |
| `quat_to_euler_zyx` vs `as_euler('ZYX')` | **1.998e-15 rad** (986/1000 samples outside the gimbal margin \|sin θ\| < 0.99) | 1e-12 |

- **Result: ALL PASS.** Agreement with scipy is at rounding level.

## 3. RK4 propagation vs closed-form solution (constant ω)

Analytic reference: q(t) = q₀ ⊗ exp_q(ω t) for constant body rate
(Markley & Crassidis 2014, Eq. 3.25). Case: ω = [0.10, 0.20, −0.15] rad/s
(|ω| = 0.2693 rad/s), t ∈ [0, 60] s, q₀ = identity.

| RK4 step dt [s] | Max attitude angle error [rad] |
|---|---|
| 0.40 | 1.131e-06 |
| 0.20 | 7.075e-08 |
| 0.10 | 4.422e-09 |
| 0.05 | **2.764e-10** |

- Observed convergence order from successive error ratios: **4.00, 4.00, 4.00**
  (theory: global order 4 for classical RK4).
- Unit-norm drift over 60 s at dt = 0.05 s: **7.735e-13** with
  `renormalize=False`, **2.220e-16** with the default per-step renormalization
  (documents the renormalization strategy's effect).
- **Result: ALL PASS** (criteria: error < 1e-9 rad at dt = 0.05 s, order > 3.5,
  renormalized drift < 1e-14).

## 4. Property-based tests (pytest + Hypothesis)

`tests/test_properties.py` verifies, with 100–200 random examples per property:
unit norm closed under the Hamilton product; q ⊗ q⁻¹ = identity (< 1e-12);
rotation preserves vector norms (< 1e-10) and inner products; DCM
orthogonality RᵀR = I and det R = +1 (< 1e-12); round trips for DCM, ZYX Euler
(away from the gimbal margin), MRP, Gibbs vector, and exp/log; SLERP output
unit and angle-bounded. Full suite: **89 passed** (`python -m pytest tests/ -q`).

## 5. Supporting evidence from examples

- `examples/tumbling_body.py` (torque-free asymmetric body, 120 s, dt = 0.02 s):
  angular-momentum magnitude drift 5.4e-12 kg·m²/s, kinetic-energy drift
  3.0e-12 J (constants of motion of the ω-integration feeding quatkit),
  max ||q| − 1| = 2.2e-16 during quaternion propagation.
- `examples/slerp_demo.py`: SLERP angle-vs-t deviation from a perfect linear
  ramp: 4.4e-16 rad over a 157.5° reorientation.

## Scope and honesty notes

- Level 1 (Educational): validation is against analytic/hand-checkable results
  and an independent library (scipy), not against flight data or an
  independent professional GNC tool.
- The scipy Euler cross-check masks the 1.4 % of random samples inside the
  gimbal margin, because inside it quatkit deliberately returns the documented
  roll = 0 fallback rather than scipy's convention; the attitude reconstructed
  from the fallback triple is still verified correct in
  `tests/test_conversions.py::TestGimbalLock`.
- No failures were observed; no tolerances were loosened during the build.
