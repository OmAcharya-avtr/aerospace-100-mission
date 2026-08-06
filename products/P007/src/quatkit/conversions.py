"""Conversions between attitude representations.

Convention: quaternions are **scalar-first** ``[w, x, y, z]``, Hamilton product,
**active** rotation ``v' = q ⊗ [0, v] ⊗ q* = R(q) v``. Euler angles use the
aerospace **ZYX (yaw-pitch-roll)** intrinsic sequence in radians.

The direction cosine matrix (DCM) returned here is the *rotation matrix* R(q)
that actively rotates vectors, matching ``scipy.spatial.transform.Rotation``'s
``as_matrix()``. The classical spacecraft *attitude matrix* A (which transforms
reference-frame coordinates into body-frame coordinates, Markley & Crassidis
2014 Sec. 2.9) is its transpose: ``A = R.T``.

Gimbal lock: the ZYX sequence is singular at pitch = ±90°, where yaw and roll
become indistinguishable (only their sum/difference is observable).
:func:`quat_to_euler_zyx` emits a :class:`GimbalLockWarning` within
``GIMBAL_LOCK_MARGIN_RAD`` of the singularity and returns the standard
fallback roll = 0 with all rotation about the lost degree of freedom
absorbed into yaw.

References
----------
- Markley, F. L. and Crassidis, J. L., *Fundamentals of Spacecraft Attitude
  Determination and Control*, Springer, 2014, Chapters 2-3.
- Shuster, M. D., "A Survey of Attitude Representations", *Journal of the
  Astronautical Sciences*, Vol. 41, No. 4, 1993.
- Shepperd, S. W., "Quaternion from Rotation Matrix", *Journal of Guidance
  and Control*, Vol. 1, No. 3, 1978 (max-component method used in
  :func:`dcm_to_quat`).
"""

from __future__ import annotations

import warnings

import numpy as np

from .core import _asq, _asv, quat_normalize

__all__ = [
    "GIMBAL_LOCK_MARGIN_RAD",
    "GimbalLockWarning",
    "axis_angle_to_quat",
    "dcm_to_quat",
    "euler_zyx_to_quat",
    "mrp_to_quat",
    "quat_to_axis_angle",
    "quat_to_dcm",
    "quat_to_euler_zyx",
    "quat_to_mrp",
    "quat_to_rodrigues",
    "rodrigues_to_quat",
]

#: Pitch margin (radians) around ±90° inside which a GimbalLockWarning is emitted.
#: 1e-6 rad of sin(pitch) margin ≈ within 0.08° of the singularity.
GIMBAL_LOCK_MARGIN_RAD = 1e-6


class GimbalLockWarning(UserWarning):
    """Emitted when a ZYX Euler extraction is within GIMBAL_LOCK_MARGIN_RAD of ±90° pitch."""


# ---------------------------------------------------------------------------
# DCM
# ---------------------------------------------------------------------------

def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Unit quaternion -> 3x3 rotation matrix R with v' = R v (active rotation).

    Vectorized: input (..., 4) gives output (..., 3, 3).

    Source: Markley & Crassidis 2014, Eq. (2.125) (transposed to the active
    rotation-matrix convention; matches scipy Rotation.as_matrix()).
    Assumes |q| = 1; a non-unit q scales R by |q|².
    """
    q = _asq(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty(q.shape[:-1] + (3, 3))
    r[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    r[..., 0, 1] = 2.0 * (x * y - w * z)
    r[..., 0, 2] = 2.0 * (x * z + w * y)
    r[..., 1, 0] = 2.0 * (x * y + w * z)
    r[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    r[..., 1, 2] = 2.0 * (y * z - w * x)
    r[..., 2, 0] = 2.0 * (x * z - w * y)
    r[..., 2, 1] = 2.0 * (y * z + w * x)
    r[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return r


def dcm_to_quat(dcm: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion [w, x, y, z] with w >= 0.

    Uses Shepperd's max-component method (Shepperd 1978), numerically stable
    for all attitudes. Input is checked for orthogonality and det = +1 within
    ``atol`` (dimensionless); ValueError otherwise. Batched input (..., 3, 3)
    is processed row by row.
    """
    dcm = np.asarray(dcm, dtype=float)
    if dcm.shape[-2:] != (3, 3):
        raise ValueError(f"DCM must have shape (..., 3, 3), got {dcm.shape}")
    if dcm.ndim > 2:
        flat = dcm.reshape(-1, 3, 3)
        return np.stack([dcm_to_quat(m, atol=atol) for m in flat]).reshape(
            dcm.shape[:-2] + (4,)
        )
    err = np.max(np.abs(dcm.T @ dcm - np.eye(3)))
    if err > atol or abs(np.linalg.det(dcm) - 1.0) > atol:
        raise ValueError(
            f"input is not a rotation matrix (orthogonality error {err:.2e}, "
            f"det {np.linalg.det(dcm):.6f}); tolerance atol={atol}"
        )
    tr = np.trace(dcm)
    # Squared quadruple magnitudes (each >= 0); pick the largest for stability.
    qq = np.array([
        1.0 + tr,
        1.0 + 2.0 * dcm[0, 0] - tr,
        1.0 + 2.0 * dcm[1, 1] - tr,
        1.0 + 2.0 * dcm[2, 2] - tr,
    ])
    i = int(np.argmax(qq))
    s = 0.5 / np.sqrt(qq[i])
    if i == 0:
        q = np.array([
            qq[0] * s,
            (dcm[2, 1] - dcm[1, 2]) * s,
            (dcm[0, 2] - dcm[2, 0]) * s,
            (dcm[1, 0] - dcm[0, 1]) * s,
        ])
    elif i == 1:
        q = np.array([
            (dcm[2, 1] - dcm[1, 2]) * s,
            qq[1] * s,
            (dcm[0, 1] + dcm[1, 0]) * s,
            (dcm[0, 2] + dcm[2, 0]) * s,
        ])
    elif i == 2:
        q = np.array([
            (dcm[0, 2] - dcm[2, 0]) * s,
            (dcm[0, 1] + dcm[1, 0]) * s,
            qq[2] * s,
            (dcm[1, 2] + dcm[2, 1]) * s,
        ])
    else:
        q = np.array([
            (dcm[1, 0] - dcm[0, 1]) * s,
            (dcm[0, 2] + dcm[2, 0]) * s,
            (dcm[1, 2] + dcm[2, 1]) * s,
            qq[3] * s,
        ])
    if q[0] < 0.0:
        q = -q
    return quat_normalize(q)


# ---------------------------------------------------------------------------
# Euler ZYX (aerospace yaw-pitch-roll)
# ---------------------------------------------------------------------------

def euler_zyx_to_quat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """ZYX (intrinsic yaw-pitch-roll) Euler angles [rad] -> unit quaternion.

    Rotation composition: R = Rz(yaw) · Ry(pitch) · Rx(roll), i.e. yaw about
    the inertial z axis, then pitch about the intermediate y, then roll about
    the final body x. Standard aerospace 3-2-1 sequence (Markley & Crassidis
    2014, Sec. 2.8; angles in radians, any real values accepted).
    """
    hy, hp, hr = 0.5 * float(yaw), 0.5 * float(pitch), 0.5 * float(roll)
    cy, sy = np.cos(hy), np.sin(hy)
    cp, sp = np.cos(hp), np.sin(hp)
    cr, sr = np.cos(hr), np.sin(hr)
    # q = qz(yaw) ⊗ qy(pitch) ⊗ qx(roll), expanded Hamilton products.
    return np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ])


def quat_to_euler_zyx(q: np.ndarray) -> np.ndarray:
    """Unit quaternion -> ZYX Euler angles [yaw, pitch, roll] in radians.

    Ranges: yaw, roll in (-π, π]; pitch in [-π/2, π/2].

    Gimbal lock: at pitch = ±90° yaw and roll are not separately observable.
    Within ``GIMBAL_LOCK_MARGIN_RAD`` of the singularity this function emits a
    :class:`GimbalLockWarning`, returns roll = 0 and puts the whole remaining
    rotation into yaw (fallback yaw = atan2(-R01, R11)). The returned triple
    still reconstructs the correct attitude via :func:`euler_zyx_to_quat`.

    Vectorized: input (..., 4) gives output (..., 3).
    """
    q = _asq(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sin_pitch = np.clip(2.0 * (w * y - x * z), -1.0, 1.0)
    locked = np.abs(sin_pitch) >= 1.0 - GIMBAL_LOCK_MARGIN_RAD
    if np.any(locked):
        warnings.warn(
            "pitch within GIMBAL_LOCK_MARGIN_RAD of ±90° (ZYX singularity): "
            "roll set to 0 and absorbed into yaw",
            GimbalLockWarning,
            stacklevel=2,
        )
    pitch = np.arcsin(sin_pitch)
    yaw_reg = np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
    roll_reg = np.arctan2(2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))
    # Fallback at lock: R01 = 2(xy - wz), R11 = 1 - 2(x² + z²); roll := 0.
    yaw_lock = np.arctan2(-2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z))
    yaw = np.where(locked, yaw_lock, yaw_reg)
    roll = np.where(locked, 0.0, roll_reg)
    return np.stack([yaw, pitch, roll], axis=-1)


# ---------------------------------------------------------------------------
# Axis-angle
# ---------------------------------------------------------------------------

def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation axis (any nonzero 3-vector) and angle [rad] -> unit quaternion.

    q = [cos(θ/2), sin(θ/2) â]. Euler's rotation theorem representation
    (Markley & Crassidis 2014, Eq. 2.122). Raises ValueError on a zero axis.
    """
    axis = _asv(axis)
    n = np.linalg.norm(axis, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("rotation axis must be a nonzero vector")
    axis = axis / n
    half = 0.5 * np.asarray(angle, dtype=float)[..., np.newaxis]
    return np.concatenate([np.cos(half), np.sin(half) * axis], axis=-1)


def quat_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit quaternion -> (unit axis, angle in [0, π] radians).

    The double cover is resolved to the minimal-angle representation
    (scalar part taken non-negative). For the identity quaternion the axis is
    arbitrary and [1, 0, 0] is returned with angle 0. Single quaternion only.
    """
    q = _asq(q)
    if q.ndim != 1:
        raise ValueError("quat_to_axis_angle expects a single quaternion of shape (4,)")
    if q[0] < 0.0:
        q = -q
    n = float(np.linalg.norm(q[1:]))
    angle = 2.0 * float(np.arctan2(n, q[0]))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0]), angle
    return q[1:] / n, angle


# ---------------------------------------------------------------------------
# Rodrigues (Gibbs) vector and Modified Rodrigues Parameters (MRP)
# ---------------------------------------------------------------------------

def quat_to_rodrigues(q: np.ndarray) -> np.ndarray:
    """Unit quaternion -> Rodrigues (Gibbs) vector g = qv / qw = â tan(θ/2).

    Dimensionless. Singular at θ = 180° (qw = 0): raises ValueError when
    |qw| < 1e-8. Source: Shuster 1993, Sec. "Gibbs vector"; Markley &
    Crassidis 2014, Sec. 2.9.4.
    """
    q = _asq(q)
    w = q[..., :1]
    if np.any(np.abs(w) < 1e-8):
        raise ValueError(
            "Rodrigues (Gibbs) vector is singular at 180° rotations (qw = 0); "
            "use MRPs or the quaternion itself near 180°"
        )
    return q[..., 1:] / w


def rodrigues_to_quat(g: np.ndarray) -> np.ndarray:
    """Rodrigues (Gibbs) vector -> unit quaternion (w > 0 branch).

    q = [1, g] / sqrt(1 + |g|²). Valid for any finite g.
    """
    g = _asv(g)
    n2 = np.sum(g * g, axis=-1, keepdims=True)
    scale = 1.0 / np.sqrt(1.0 + n2)
    return np.concatenate([scale, scale * g], axis=-1)


def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    """Unit quaternion -> Modified Rodrigues Parameters p = qv / (1 + qw) = â tan(θ/4).

    Dimensionless. The double cover is resolved by flipping to qw >= 0 first,
    which selects the principal MRP set with |p| <= 1 (rotation angle <= 180°)
    and avoids the p -> ∞ singularity at θ = 360°.
    Source: Markley & Crassidis 2014, Sec. 2.9.5; Shuster 1993.
    """
    q = _asq(q)
    q = q * np.where(q[..., :1] < 0.0, -1.0, 1.0)
    return q[..., 1:] / (1.0 + q[..., :1])


def mrp_to_quat(p: np.ndarray) -> np.ndarray:
    """Modified Rodrigues Parameters -> unit quaternion.

    qw = (1 - |p|²) / (1 + |p|²), qv = 2 p / (1 + |p|²). Valid for finite p;
    |p| <= 1 maps to rotations of at most 180°.
    """
    p = _asv(p)
    n2 = np.sum(p * p, axis=-1, keepdims=True)
    denom = 1.0 + n2
    return np.concatenate([(1.0 - n2) / denom, 2.0 * p / denom], axis=-1)
