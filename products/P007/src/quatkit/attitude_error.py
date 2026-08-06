"""Attitude error metrics between two attitude quaternions.

Convention: scalar-first ``[w, x, y, z]``, Hamilton product, active rotation.

Error definition (multiplicative, body-frame): the error quaternion δq
satisfies ``q = q_ref ⊗ δq``, i.e. ``δq = q_ref⁻¹ ⊗ q``. δq is the extra
rotation, expressed in the reference attitude's body frame, that takes the
reference attitude to the actual attitude. This is the standard multiplicative
error used in attitude estimation (Markley & Crassidis 2014, Sec. 6.1;
Markley, F. L., "Attitude Error Representations for Kalman Filtering",
Journal of Guidance, Control, and Dynamics, Vol. 26, No. 2, 2003).

Small-angle error vector convention: ``δθ = 2 · vec(δq)`` (dimensionless →
radians for small angles), since vec(δq) = sin(θ/2) â ≈ (θ/2) â. This is the
first-order "2×vector-part" convention common in MEKF formulations; it is
accurate to O(θ³) and exact in direction.
"""

from __future__ import annotations

import numpy as np

from .core import _asq, quat_conjugate, quat_multiply

__all__ = [
    "angle_between",
    "attitude_error_vector",
    "error_quaternion",
]


def error_quaternion(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Multiplicative error quaternion δq = q_ref⁻¹ ⊗ q (so that q = q_ref ⊗ δq).

    Both inputs must be unit quaternions, scalar-first (..., 4). The double
    cover is resolved so the returned scalar part is non-negative (minimal
    rotation angle). Broadcasts over leading axes.
    """
    q, q_ref = _asq(q), _asq(q_ref)
    dq = quat_multiply(quat_conjugate(q_ref), q)
    return dq * np.where(dq[..., :1] < 0.0, -1.0, 1.0)


def attitude_error_vector(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Small-angle attitude error vector δθ = 2 · vec(δq) [rad, to first order].

    Uses the documented 2×vector-part convention: for a true error angle θ the
    returned magnitude is 2 sin(θ/2) = θ − θ³/24 + …, so it is accurate to
    O(θ³) for small errors and saturates at 2 for θ = 180°. Direction is the
    exact error rotation axis in the reference body frame.
    """
    return 2.0 * error_quaternion(q, q_ref)[..., 1:]


def angle_between(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Exact rotation angle [rad, in [0, π]] between two attitudes.

    θ = 2 · atan2(|vec(δq)|, |w(δq)|) — numerically stable for both tiny and
    near-180° separations (better conditioned than 2·acos near θ = 0).
    """
    dq = error_quaternion(q, q_ref)
    return 2.0 * np.arctan2(np.linalg.norm(dq[..., 1:], axis=-1), np.abs(dq[..., 0]))
