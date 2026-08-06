"""quatkit — quaternion and attitude-representation toolbox for aerospace GNC.

CONVENTION (repeated deliberately, it governs every function in this package):

* Quaternions are **scalar-first**: ``q = [w, x, y, z]``.
* **Hamilton product** (i j = k); ``q2 * q1`` means "rotate by q1, then q2".
* **Active rotation**: ``v' = q ⊗ [0, v] ⊗ q* = R(q) v``; the DCM returned by
  ``quat_to_dcm`` matches ``scipy.spatial.transform.Rotation.as_matrix()``
  (note scipy stores quaternions scalar-LAST — convert with ``np.roll``).
* Euler angles: aerospace **ZYX (yaw-pitch-roll)** intrinsic sequence, radians.

Educational / research-grade software (validation Level 1). Not flight
qualified. See README.md and validation/VALIDATION.md.
"""

from .attitude_error import angle_between, attitude_error_vector, error_quaternion
from .conversions import (
    GIMBAL_LOCK_MARGIN_RAD,
    GimbalLockWarning,
    axis_angle_to_quat,
    dcm_to_quat,
    euler_zyx_to_quat,
    mrp_to_quat,
    quat_to_axis_angle,
    quat_to_dcm,
    quat_to_euler_zyx,
    quat_to_mrp,
    quat_to_rodrigues,
    rodrigues_to_quat,
)
from .core import (
    quat_conjugate,
    quat_exp,
    quat_identity,
    quat_inverse,
    quat_log,
    quat_multiply,
    quat_norm,
    quat_normalize,
    quat_rotate,
    quat_slerp,
)
from .kinematics import (
    closed_form_constant_omega,
    propagate,
    quat_derivative,
    rk4_step,
)
from .quaternion import NORM_TOL, Quaternion

__version__ = "0.1.0"

__all__ = [
    "GIMBAL_LOCK_MARGIN_RAD",
    "NORM_TOL",
    "GimbalLockWarning",
    "Quaternion",
    "__version__",
    "angle_between",
    "attitude_error_vector",
    "axis_angle_to_quat",
    "closed_form_constant_omega",
    "dcm_to_quat",
    "error_quaternion",
    "euler_zyx_to_quat",
    "mrp_to_quat",
    "propagate",
    "quat_conjugate",
    "quat_derivative",
    "quat_exp",
    "quat_identity",
    "quat_inverse",
    "quat_log",
    "quat_multiply",
    "quat_norm",
    "quat_normalize",
    "quat_rotate",
    "quat_slerp",
    "quat_to_axis_angle",
    "quat_to_dcm",
    "quat_to_euler_zyx",
    "quat_to_mrp",
    "quat_to_rodrigues",
    "rk4_step",
    "rodrigues_to_quat",
]
