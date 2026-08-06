# QuatKit

**Status:** TESTING · **Class:** compact · **Validation level:** 1 (Educational) · **AI:** no

> **Convention — read this first.** QuatKit quaternions are **scalar-first**
> `q = [w, x, y, z]`, use the **Hamilton product** (i j = k), and act as
> **active rotations**: `v' = q ⊗ [0, v] ⊗ q* = R(q) v`. Euler angles are the
> aerospace **ZYX (yaw–pitch–roll)** intrinsic sequence in radians.
> `scipy.spatial.transform.Rotation` stores quaternions scalar-LAST — convert
> with `np.roll(q, -1)` / `np.roll(q, 1)`.

## Executive overview

QuatKit is a compact, numpy-vectorized toolbox for quaternion attitude
representation in aerospace guidance, navigation and control (GNC) work. It
provides a strict unit-norm `Quaternion` class plus a flat array API:
Hamilton algebra, exp/log maps, SLERP, conversions between quaternion / DCM /
ZYX Euler / axis-angle / Rodrigues (Gibbs) / MRP, multiplicative attitude
error metrics, and RK4 attitude propagation `q̇ = ½ q ⊗ [0, ω]` with a
documented renormalization strategy. Dependencies: numpy and scipy only
(scipy is used only for validation cross-checks).

## Aerospace problem

Attitude in GNC is stored and propagated in many parameterizations, each with
sharp edges: Euler angles gimbal-lock at ±90° pitch, Gibbs vectors blow up at
180°, MRPs at 360°, and quaternions carry a unit-norm constraint and a sign
double cover. Convention mix-ups (scalar-first vs scalar-last, Hamilton vs
JPL, active vs passive) are a classic source of sign errors in flight
software. QuatKit implements one clearly stated convention end-to-end, makes
the singular cases explicit (warn or raise), and validates every conversion
against hand-checkable rotations and scipy.

## Intended users

Students and engineers learning or prototyping attitude dynamics, estimation
(MEKF-style error states), and control; instructors building GNC exercises.
Not intended for flight software (see Safety statement).

## Engineering theory

All equations below use the convention stated at the top. Sources: Markley &
Crassidis, *Fundamentals of Spacecraft Attitude Determination and Control*,
Springer, 2014 (Chs. 2–3, 6); Shuster, "A Survey of Attitude Representations",
*J. Astronautical Sciences* 41(4), 1993; Shepperd, "Quaternion from Rotation
Matrix", *J. Guidance and Control* 1(3), 1978; Shoemake, "Animating Rotation
with Quaternion Curves", *SIGGRAPH* 19(3), 1985.

- **Hamilton product** (dimensionless): for `q = (w, 𝐯)`,
  `q₁ ⊗ q₂ = (w₁w₂ − 𝐯₁·𝐯₂, w₁𝐯₂ + w₂𝐯₁ + 𝐯₁×𝐯₂)`. `q2 * q1` composes
  "rotate by q1, then q2". Valid for all quaternions.
- **Rotation** of a vector (units of v preserved): `v' = q ⊗ [0, v] ⊗ q*`,
  evaluated in the expanded form `v' = v + 2w(𝐯×v) + 2𝐯×(𝐯×v)`. Assumes
  |q| = 1 (enforced by the class; see normalization policy).
- **Exp/log maps** (rotation vector in radians):
  `exp(φ) = [cos(|φ|/2), sin(|φ|/2) φ̂]`, log is its inverse restricted to the
  principal branch (angle in [0, π]). Small angles handled by series limit
  (no division by zero).
- **DCM**: `R(q)` per Markley & Crassidis 2014 Eq. (2.125), transposed to the
  active-rotation-matrix convention so it matches scipy's `as_matrix()`. The
  classical spacecraft *attitude matrix* (reference→body) is `A = R(q)ᵀ`.
  Inverse conversion via Shepperd's max-component method (Shepperd 1978),
  stable for all attitudes; input checked orthogonal with det = +1.
- **ZYX Euler** (rad): `R = Rz(ψ) Ry(θ) Rx(φ)`. Singular at θ = ±90°
  (gimbal lock): within `GIMBAL_LOCK_MARGIN_RAD = 1e-6` of |sin θ| = 1 a
  `GimbalLockWarning` is emitted, roll is set to 0 and the unobservable
  degree of freedom is absorbed into yaw; the returned triple still
  reconstructs the correct attitude.
- **Axis-angle / Gibbs / MRP** (dimensionless): `g = 𝐯/w = â tan(θ/2)`
  (raises ValueError at 180°); `p = 𝐯/(1+w) = â tan(θ/4)` (principal set
  |p| ≤ 1 chosen via the double cover, finite at 180°). Shuster 1993.
- **Attitude error** (Markley, *J. Guidance Control Dyn.* 26(2), 2003):
  multiplicative error `δq = q_ref⁻¹ ⊗ q` (so `q = q_ref ⊗ δq`); small-angle
  error vector **δθ = 2·vec(δq)** [rad to O(θ³)] — the documented
  2×vector-part MEKF convention; exact angle `2·atan2(|vec δq|, |w δq|)`.
- **Kinematics** (Markley & Crassidis 2014, Eq. 3.21): `q̇ = ½ q ⊗ [0, ω]`,
  ω = body angular velocity [rad/s] of body w.r.t. reference, expressed in
  body axes. Classical RK4 (global order 4); closed form
  `q(t) = q₀ ⊗ exp(ω t)` for constant ω (Eq. 3.25).
- **Renormalization strategy**: RK4 drifts |q| by truncation error
  (~8e-13 over 1200 steps in validation), so `propagate` renormalizes
  `q ← q/|q|` after every step — the standard projection remedy; it does not
  bias the attitude. Disable with `renormalize=False` to observe the drift.

## Architecture

```
src/quatkit/
├── core.py            # vectorized array ops: multiply, rotate, exp/log, slerp, normalize
├── quaternion.py      # unit-norm Quaternion class (normalize-or-raise policy)
├── conversions.py     # DCM, ZYX Euler (+GimbalLockWarning), axis-angle, Gibbs, MRP
├── attitude_error.py  # error quaternion, 2*vec error vector, angle between
└── kinematics.py      # q̇ = ½ q⊗[0,ω], RK4 propagate, constant-ω closed form
```

Array functions accept `(..., 4)` / `(..., 3)` and broadcast; the class wraps
a single quaternion and enforces invariants.

## Installation

```bash
pip install .            # from products/P007/ (or: pip install -e .)
```

Requires Python ≥ 3.10, numpy ≥ 1.24, scipy ≥ 1.10. Tests/examples also run
without installation (they insert `src/` on `sys.path`).

## Quick start

```python
import numpy as np
from quatkit import Quaternion, propagate, angle_between

# scalar-first [w, x, y, z]; 90° about +z
q = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
q.rotate([1, 0, 0])                      # -> [0, 1, 0]
q.to_euler_zyx()                         # -> [1.5708, 0, 0] (yaw, pitch, roll)

q2 = Quaternion.from_euler_zyx(0.3, -0.2, 0.1)
(q2 * q).rotate([1, 0, 0])               # rotate by q first, then q2
q.slerp(q2, 0.5)                         # geodesic midpoint

qs = propagate(q.as_array(), lambda t: np.array([0.1, 0.2, -0.15]),
               np.linspace(0, 60, 1201))  # RK4, renormalized each step
```

Normalization policy: `Quaternion(...)` **raises ValueError** if the input is
more than `NORM_TOL = 1e-6` from unit norm unless you pass `normalize=True`;
inputs within tolerance are silently renormalized to machine precision. So a
non-unit quaternion can never reach `rotate()` unnoticed.

## Configuration

No config files. Tunables are function arguments: `normalize=` (constructor),
`renormalize=` (propagation), `atol=` (`dcm_to_quat` orthogonality check);
`quatkit.GIMBAL_LOCK_MARGIN_RAD` documents the warning margin.

## Examples

Run from `products/P007/` (outputs land in `screenshots/`):

- `python examples/tumbling_body.py` — torque-free asymmetric rigid body
  (intermediate-axis tumble), ω from Euler's equations, attitude via quatkit
  RK4; plots ZYX Euler angles, body rates, and conservation drift
  (`screenshots/tumbling_euler_angles.png`). Euler-angle jumps at chart
  boundaries/gimbal passages are expected and annotated.
- `python examples/slerp_demo.py` — SLERP vs nlerp angle linearity and the
  great-circle trace of a body axis (`screenshots/slerp_interpolation.png`).

## Validation

Level 1 (Educational). Full evidence with raw outputs in
[`validation/VALIDATION.md`](validation/VALIDATION.md); all scripts rerunnable
and seeded. Highlights (actually measured in this build):

- 12/12 hand-checkable 90°/180° principal-axis cases, worst error 2.2e-16.
- vs scipy `Rotation`, N = 1000 random attitudes (seed 20260801): max DCM
  element deviation 1.0e-15; `dcm_to_quat` round-trip angle error 5.0e-16 rad;
  vector rotation 2.7e-15; ZYX Euler 2.0e-15 rad.
- RK4 vs analytic constant-ω solution (|ω| = 0.269 rad/s, 60 s): max angle
  error 2.8e-10 rad at dt = 0.05 s; observed convergence order 4.00.
- Hypothesis property tests: unit-norm closure, q ⊗ q⁻¹ = 1, norm/inner-product
  preservation under rotation, DCM orthogonality, representation round trips.

## Benchmark results

Not a performance-focused product; indicative timings on the 2-core build
container: full test suite (89 tests incl. ~2000 Hypothesis examples) ≈ 3 s;
RK4 propagation 1200 steps ≈ 0.1 s; vectorized `quat_rotate` on 10⁶
quaternion-vector pairs runs as a handful of numpy kernels.

## AI model details

Not applicable — this product contains no AI/ML components.

## Hardware requirements

Any machine running Python 3.10+; < 100 MB RAM for typical use; no GPU. All
validation and examples complete in well under 3 minutes on 2 CPU cores.

## Limitations

- Educational (Level 1): validated against analytic results and scipy only —
  no flight data, no independent professional GNC tool comparison.
- Fixed-step RK4 only; no adaptive step-size control, no angular-velocity
  dynamics (Euler's equations are demonstrated in the example, not part of
  the library API).
- `dcm_to_quat` and `quat_to_axis_angle` process one attitude at a time
  (batched input loops internally); other operations are vectorized.
- Euler angles: only the aerospace ZYX sequence is provided (no other
  sequences). Inside the gimbal margin the roll = 0 fallback differs from
  scipy's choice (attitude is still reconstructed correctly).
- Single (Hamilton, scalar-first, active) convention; no JPL-convention
  interoperability helpers beyond the documented scipy `np.roll` conversion.
- SLERP is implemented for single quaternion pairs (array of t values is
  supported; arrays of quaternion pairs are not).

## Safety statement

This software is educational / research-grade. It is not flight-qualified,
not certified, and not approved for operational aerospace use.

## Roadmap

- Batched `dcm_to_quat` (vectorized Shepperd) and batched SLERP.
- Additional Euler sequences (ZXZ, XYZ) with per-sequence singularity handling.
- SQUAD (C¹ quaternion spline) interpolation.
- Optional angular-dynamics module (Euler's equations, gyrostat).

## License

MIT — see [LICENSE](LICENSE). © 2026 OPTIMA Organisation.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

## Citation

```
OPTIMA Organisation (2026). QuatKit v0.1.0: quaternion and attitude-
representation toolbox for aerospace GNC (scalar-first Hamilton convention).
Educational software, MIT license.
```

Key references: Markley & Crassidis (2014), *Fundamentals of Spacecraft
Attitude Determination and Control*, Springer; Shuster (1993), *J. Astronaut.
Sci.* 41(4); Shepperd (1978), *J. Guidance and Control* 1(3); Shoemake (1985),
*SIGGRAPH* 19(3).
