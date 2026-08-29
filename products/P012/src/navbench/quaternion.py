r"""Unit-quaternion attitude algebra for NavBench.

Convention (stated once, used everywhere)
-----------------------------------------
* **Storage:** scalar-first, ``q = [w, x, y, z]``, ``w`` the scalar part.
* **Product:** Hamilton, ``i j = k`` (NOT the JPL/Shuster convention).
* **Action:** *active* rotation of a vector inside one frame, equivalently the
  body-to-inertial map::

      v_N = R(q) v_B = vec(q ⊗ [0, v_B] ⊗ q*)

  so ``q`` is the attitude of the body frame **relative to** the inertial
  (reference) frame, and the *attitude matrix* used by the estimation
  literature is ``A(q) = R(q)ᵀ`` (inertial → body).
* **Kinematics:** with ``ω`` the angular velocity of body w.r.t. inertial
  expressed **in body axes** [rad/s]::

      q̇ = ½ q ⊗ [0, ω]

* **Error:** *local* (body-frame) multiplicative error, ``q = q̂ ⊗ δq(a)``,
  with ``δq(a) ≈ [1, a/2]`` and ``a`` the three-component attitude error in
  radians. This is the error definition used by :mod:`navbench.mekf`.

The same convention is used by product **P007 QuatKit**; NavBench implements it
independently (no cross-product imports) and cross-checks it against
``scipy.spatial.transform.Rotation`` in ``validation/v4_attitude_mekf.py``.
SciPy stores quaternions **scalar-last**: convert with ``np.roll(q, -1)``.

References
----------
* Markley, F. L. and Crassidis, J. L. (2014), *Fundamentals of Spacecraft
  Attitude Determination and Control*, Springer. §2.9 covers the quaternion
  parameterisation, the attitude matrix and the kinematic equation.
* Shuster, M. D. (1993), "A Survey of Attitude Representations",
  *Journal of the Astronautical Sciences* **41**(4), 439–517.
* Lefferts, E. J., Markley, F. L. and Shuster, M. D. (1982), "Kalman Filtering
  for Spacecraft Attitude Estimation", *Journal of Guidance, Control, and
  Dynamics* **5**(5), 417–429.

Units: quaternion components dimensionless; angular rate rad/s; rotation
vectors rad. Validity: all functions require ``‖q‖ ≈ 1``; the small-angle
approximations flagged below are first order in the rotation angle.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "as_quaternion",
    "quat_multiply",
    "quat_conjugate",
    "quat_normalize",
    "quat_rotate",
    "quat_to_dcm",
    "dcm_to_quat",
    "quat_from_rotvec",
    "quat_to_rotvec",
    "small_angle_quat",
    "attitude_matrix",
    "skew",
    "quat_angle_between",
    "quat_canonical",
]


def as_quaternion(q: ArrayLike, name: str = "q") -> NDArray[np.float64]:
    """Validate and return a length-4 float array ``[w, x, y, z]``.

    Raises
    ------
    ValueError
        If the input does not have exactly four components or is not finite.
    """
    arr = np.asarray(q, dtype=float).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(f"{name} must have 4 components [w, x, y, z], got shape {np.shape(q)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values: {arr}")
    return arr


def skew(v: ArrayLike) -> NDArray[np.float64]:
    r"""Skew-symmetric cross-product matrix ``[v×]`` with ``[v×] u = v × u``.

    Parameters
    ----------
    v : array_like, shape (3,)
        Any 3-vector; units are carried through unchanged.

    Returns
    -------
    ndarray, shape (3, 3)
    """
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.shape != (3,):
        raise ValueError(f"skew() needs a 3-vector, got shape {np.shape(v)}")
    return np.array(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], dtype=float
    )


def quat_multiply(p: ArrayLike, q: ArrayLike) -> NDArray[np.float64]:
    r"""Hamilton product ``p ⊗ q`` (scalar-first).

    ``(p ⊗ q)`` composes rotations so that ``R(p ⊗ q) = R(p) R(q)``.

    Source: Markley & Crassidis (2014) §2.9.3 (quaternion composition), with
    the Hamilton sign convention. Dimensionless.
    """
    a = as_quaternion(p, "p")
    b = as_quaternion(q, "q")
    w1, v1 = a[0], a[1:]
    w2, v2 = b[0], b[1:]
    return np.concatenate(
        ([w1 * w2 - float(np.dot(v1, v2))], w1 * v2 + w2 * v1 + np.cross(v1, v2))
    )


def quat_conjugate(q: ArrayLike) -> NDArray[np.float64]:
    """Quaternion conjugate ``q* = [w, -x, -y, -z]`` (the inverse for unit q)."""
    a = as_quaternion(q)
    return np.concatenate(([a[0]], -a[1:]))


def quat_normalize(q: ArrayLike, tol: float = 1e-12) -> NDArray[np.float64]:
    """Return ``q/‖q‖``.

    Raises
    ------
    ValueError
        If ``‖q‖ <= tol`` (the zero quaternion has no attitude meaning).
    """
    a = as_quaternion(q)
    n = float(np.linalg.norm(a))
    if n <= tol:
        raise ValueError(f"cannot normalize a quaternion of norm {n:.3e} (tol={tol:.1e})")
    return a / n


def quat_canonical(q: ArrayLike) -> NDArray[np.float64]:
    """Resolve the sign double cover by forcing ``w >= 0``.

    ``q`` and ``-q`` are the same rotation; comparisons and error metrics must
    account for that. Source: Shuster (1993) §2.
    """
    a = as_quaternion(q)
    return -a if a[0] < 0.0 else a.copy()


def quat_rotate(q: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
    r"""Active rotation ``v_N = R(q) v_B`` (body → inertial).

    Implemented as ``vec(q ⊗ [0, v] ⊗ q*)``. Units of ``v`` are preserved.
    """
    a = quat_normalize(q)
    u = np.asarray(v, dtype=float).reshape(-1)
    if u.shape != (3,):
        raise ValueError(f"quat_rotate() needs a 3-vector, got shape {np.shape(v)}")
    w, vec = a[0], a[1:]
    t = 2.0 * np.cross(vec, u)
    return u + w * t + np.cross(vec, t)


def quat_to_dcm(q: ArrayLike) -> NDArray[np.float64]:
    r"""Rotation matrix ``R(q)`` (body → inertial), i.e. ``v_N = R(q) v_B``.

    ``R = (w² − ‖v‖²) I + 2 v vᵀ + 2 w [v×]`` — Markley & Crassidis (2014)
    §2.9.2, transposed to the active/Hamilton convention used here.
    """
    a = quat_normalize(q)
    w, v = a[0], a[1:]
    return (w * w - float(np.dot(v, v))) * np.eye(3) + 2.0 * np.outer(v, v) + 2.0 * w * skew(v)


def attitude_matrix(q: ArrayLike) -> NDArray[np.float64]:
    r"""Attitude matrix ``A(q) = R(q)ᵀ`` (inertial → body): ``v_B = A(q) v_N``.

    This is the matrix called ``A(q)`` throughout Markley & Crassidis (2014).
    """
    return quat_to_dcm(q).T


def dcm_to_quat(dcm: ArrayLike) -> NDArray[np.float64]:
    r"""Invert :func:`quat_to_dcm` using Shepperd's numerically stable branch.

    Source: Shepperd, S. W. (1978), "Quaternion from Rotation Matrix",
    *Journal of Guidance and Control* **1**(3), 223–224. The branch with the
    largest pivot is used, which bounds the relative error for every input
    rotation including the 180° cases where the naive trace formula fails.

    Raises
    ------
    ValueError
        If ``dcm`` is not 3×3 or departs from orthonormality by more than 1e-6.
    """
    m = np.asarray(dcm, dtype=float)
    if m.shape != (3, 3):
        raise ValueError(f"dcm_to_quat() needs a 3x3 matrix, got shape {m.shape}")
    orth = float(np.max(np.abs(m @ m.T - np.eye(3))))
    if orth > 1e-6:
        raise ValueError(f"dcm is not orthonormal: max|R Rᵀ − I| = {orth:.3e}")
    tr = float(np.trace(m))
    pivots = np.array([tr, m[0, 0], m[1, 1], m[2, 2]], dtype=float)
    k = int(np.argmax(pivots))
    if k == 0:
        s = np.sqrt(1.0 + tr) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s])
    elif k == 1:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s,
                      (m[0, 2] + m[2, 0]) / s])
    elif k == 2:
        s = np.sqrt(1.0 - m[0, 0] + m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s,
                      (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 - m[0, 0] - m[1, 1] + m[2, 2]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return quat_canonical(quat_normalize(q))


def quat_from_rotvec(rotvec: ArrayLike) -> NDArray[np.float64]:
    r"""Exponential map ``exp([0, θ n /2])`` from a rotation vector [rad].

    ``q = [cos(θ/2), n sin(θ/2)]`` with ``θ = ‖rotvec‖`` and ``n`` its unit
    direction. A Taylor series is used below ``θ = 1e-8`` rad, where
    ``sin(θ/2)/θ → 1/2`` to machine precision, to avoid 0/0.

    Source: Markley & Crassidis (2014) §2.9.3. Valid for all θ; the map is
    2π-periodic in θ up to the quaternion sign double cover.
    """
    r = np.asarray(rotvec, dtype=float).reshape(-1)
    if r.shape != (3,):
        raise ValueError(f"quat_from_rotvec() needs a 3-vector, got shape {np.shape(rotvec)}")
    theta = float(np.linalg.norm(r))
    if theta < 1e-8:
        # cos(θ/2) ≈ 1 − θ²/8, sin(θ/2)/θ ≈ 1/2 − θ²/48
        return quat_normalize(np.concatenate(([1.0 - theta * theta / 8.0], 0.5 * r)))
    return np.concatenate(([np.cos(0.5 * theta)], np.sin(0.5 * theta) * r / theta))


def quat_to_rotvec(q: ArrayLike) -> NDArray[np.float64]:
    """Logarithm map: rotation vector [rad] with angle in ``[0, π]``.

    The quaternion is first put in canonical form (``w >= 0``) so that the
    returned angle is the *shortest* rotation, which is what an attitude error
    metric must report.
    """
    a = quat_canonical(quat_normalize(q))
    w = float(np.clip(a[0], -1.0, 1.0))
    v = a[1:]
    sn = float(np.linalg.norm(v))
    if sn < 1e-12:
        return 2.0 * v  # θ → 0: rotvec ≈ 2 · vec
    theta = 2.0 * np.arctan2(sn, w)
    return theta * v / sn


def small_angle_quat(a: ArrayLike) -> NDArray[np.float64]:
    r"""First-order error quaternion ``δq(a) ≈ [1, a/2]``, renormalized.

    This is the MEKF parameterisation of the three-component attitude error
    ``a`` [rad] (Lefferts, Markley & Shuster 1982, §II). The scalar part is
    recovered by normalisation rather than being set to ``sqrt(1 − ‖a/2‖²)``, so
    the function never produces a NaN for ``‖a‖ > 2``; the *first-order*
    interpretation is only valid for ``‖a‖ ≲ 0.1 rad`` (≈ 6°), where the
    difference from the exact exponential map is below ``‖a‖³/48``.
    """
    v = np.asarray(a, dtype=float).reshape(-1)
    if v.shape != (3,):
        raise ValueError(f"small_angle_quat() needs a 3-vector, got shape {np.shape(a)}")
    return quat_normalize(np.concatenate(([1.0], 0.5 * v)))


def quat_angle_between(p: ArrayLike, q: ArrayLike) -> float:
    """Shortest rotation angle [rad] between two attitudes, in ``[0, π]``.

    Computed from the canonical error quaternion ``p* ⊗ q``; robust to the sign
    double cover.
    """
    dq = quat_canonical(quat_multiply(quat_conjugate(quat_normalize(p)), quat_normalize(q)))
    return float(2.0 * np.arctan2(np.linalg.norm(dq[1:]), np.clip(dq[0], -1.0, 1.0)))
