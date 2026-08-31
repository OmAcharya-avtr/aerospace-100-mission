"""wahbakit -- static attitude determination from vector observations.

Wahba's problem, solved four ways with every convention stated at the call site.

Conventions (read this before trusting any output)
--------------------------------------------------
* An observation is a pair ``(b_i, r_i)``: ``b_i`` measured in the **body**
  frame, ``r_i`` known in the **reference** frame.  The argument order is
  always ``(body, reference)``.
* The attitude matrix ``A`` satisfies ``b_i ~= A r_i``; it is the
  reference-to-body DCM, orthogonal with ``det(A) = +1``.
* Quaternions are **scalar first**, ``q = [w, x, y, z]``, Hamilton product,
  with ``A = dcm_from_quat(q)`` identical to
  ``scipy.spatial.transform.Rotation.from_quat([x, y, z, w]).as_matrix()``.
  Every returned quaternion has ``w >= 0``.  This matches P007 QuatKit; it is
  the transpose of the attitude matrix Shuster's papers write, and the
  conversion is done inside :mod:`wahbakit.davenport`.
* Attitude covariance is ``E[delta_theta delta_theta^T]`` in **rad^2** with
  ``delta_theta = log(A_est A_true^T)`` in the **body** frame.
* Angles are radians unless the name ends in ``_deg``.  Sigmas are transverse
  angular standard deviations in radians.

Near-parallel observations
--------------------------
Every solver runs an observability gate (Eq. O4) before it does any algebra and
raises :class:`DegenerateObservationsError` when the geometry does not
determine three axes.  Pass ``check_degeneracy=False`` to disable it; that does
not make the answer valid.  What each method does without the gate is stated in
its module docstring.

Educational / research-grade software, validation Level 1.  Not flight
qualified.  See ``README.md`` and ``validation/VALIDATION.md``.
"""

from __future__ import annotations

from .conventions import (
    ORTHOGONALITY_TOL,
    UNIT_TOL,
    angle_between_dcm,
    attitude_error_vector,
    dcm_from_quat,
    is_rotation,
    quat_canonical,
    quat_conjugate,
    quat_from_dcm,
    quat_multiply,
    quat_normalize,
    rotation_vector_from_dcm,
    skew,
    unit_vectors,
)
from .covariance import (
    COVARIANCE_METHODS,
    attitude_covariance,
    covariance_axis_sigmas_deg,
    optimal_covariance,
    triad_covariance,
)
from .davenport import davenport_matrix, profile_parts, q_method
from .observations import (
    DEFAULT_DEGENERACY_TOL,
    DegenerateObservationsError,
    Observability,
    VectorObservations,
)
from .olae import olae, olae_normal_equations
from .quest import (
    SEQUENTIAL_ROTATION_QUATS,
    characteristic_coefficients,
    characteristic_polynomial,
    quest,
    quest_lambda_max,
)
from .solution import AttitudeSolution, wahba_gain, wahba_loss
from .solve import METHODS, solve_wahba
from .triad import triad, triad_frame

__version__ = "0.1.0"

__all__ = [
    "COVARIANCE_METHODS",
    "DEFAULT_DEGENERACY_TOL",
    "METHODS",
    "ORTHOGONALITY_TOL",
    "SEQUENTIAL_ROTATION_QUATS",
    "UNIT_TOL",
    "AttitudeSolution",
    "DegenerateObservationsError",
    "Observability",
    "VectorObservations",
    "__version__",
    "angle_between_dcm",
    "attitude_covariance",
    "attitude_error_vector",
    "characteristic_coefficients",
    "characteristic_polynomial",
    "covariance_axis_sigmas_deg",
    "davenport_matrix",
    "dcm_from_quat",
    "is_rotation",
    "olae",
    "olae_normal_equations",
    "optimal_covariance",
    "profile_parts",
    "q_method",
    "quat_canonical",
    "quat_conjugate",
    "quat_from_dcm",
    "quat_multiply",
    "quat_normalize",
    "quest",
    "quest_lambda_max",
    "rotation_vector_from_dcm",
    "skew",
    "solve_wahba",
    "triad",
    "triad_covariance",
    "triad_frame",
    "unit_vectors",
    "wahba_gain",
    "wahba_loss",
]
