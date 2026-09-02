"""cmgsteer -- control-moment-gyro array geometry, singularity analysis and steering laws.

CONVENTIONS (they govern every function in this package):

* Vectors are in the spacecraft **body frame**, right-handed and orthonormal,
  with the active-rotation convention of QuatKit (P007).  No quaternion is used
  internally; the convention matters only when a caller rotates an array.
* A single-gimbal CMG has gimbal axis ``g``, reference axis ``c`` (the rotor
  momentum direction at gimbal angle zero, perpendicular to ``g``) and
  transverse axis ``s = g x c``.  Gimbal angles increase in the right-handed
  sense about ``g``.
* ``A(delta) = dh/ddelta`` [N*m*s/rad] is the momentum-map Jacobian, and the
  torque delivered **to the vehicle** is ``tau = -A(delta) ddelta/dt`` [N*m].
  Every ``torque`` argument and return value in this package is that ``tau``.
* Angles are radians, momenta N*m*s, torques N*m, rates rad/s, time seconds.
* The singularity measure is ``m = sqrt(det(A A^T))`` [(N*m*s/rad)^3].

Research-grade software (validation Level 3).  Not flight-qualified, not
certified, not approved for operational aerospace use.  See README.md,
docs/REQUIREMENTS.md and validation/VALIDATION.md.
"""

from .arrays import (
    STANDARD_PYRAMID_SKEW_DEG,
    CMGArray,
    general_array,
    pyramid_array,
    roof_array,
)
from .dataset import (
    ManoeuvreSuite,
    PolicyDataset,
    generate_policy_dataset,
    manoeuvre_suite,
    rollout_score,
)
from .ml import (
    LearnedNullMotion,
    NullMotionAction,
    feature_names,
    policy_features,
)
from .nullmotion import (
    GradientNullMotion,
    NoNullMotion,
    NullMotionPolicy,
    PreferredAngleNullMotion,
    null_motion_from_coefficients,
    null_projector,
    unit_null_vector,
)
from .simulate import (
    SteeringHistory,
    TorqueProfile,
    constant_profile,
    rest_to_rest_profile,
    run_steering,
)
from .singularity import (
    SingularityInfo,
    classify_singularity,
    condition_number,
    fibonacci_directions,
    manipulability_gradient,
    min_singular_value,
    momentum_envelope,
    null_space_basis,
    singular_configuration,
    singular_direction,
    singular_surface,
    singularity_measure,
)
from .steering import (
    METHODS,
    SteeringResult,
    apply_rate_limit,
    gsr_inverse_steer,
    pseudo_inverse_steer,
    robustness_parameter,
    sr_inverse_steer,
    sr_torque_error_closed_form,
    steer,
)

__version__ = "0.1.0"

__all__ = [
    "CMGArray",
    "GradientNullMotion",
    "LearnedNullMotion",
    "METHODS",
    "ManoeuvreSuite",
    "NoNullMotion",
    "NullMotionAction",
    "NullMotionPolicy",
    "PolicyDataset",
    "PreferredAngleNullMotion",
    "STANDARD_PYRAMID_SKEW_DEG",
    "SingularityInfo",
    "SteeringHistory",
    "SteeringResult",
    "TorqueProfile",
    "__version__",
    "apply_rate_limit",
    "classify_singularity",
    "condition_number",
    "constant_profile",
    "feature_names",
    "fibonacci_directions",
    "general_array",
    "generate_policy_dataset",
    "gsr_inverse_steer",
    "manipulability_gradient",
    "manoeuvre_suite",
    "min_singular_value",
    "momentum_envelope",
    "null_motion_from_coefficients",
    "null_projector",
    "null_space_basis",
    "policy_features",
    "pseudo_inverse_steer",
    "pyramid_array",
    "rest_to_rest_profile",
    "robustness_parameter",
    "rollout_score",
    "roof_array",
    "run_steering",
    "singular_configuration",
    "singular_direction",
    "singular_surface",
    "singularity_measure",
    "sr_inverse_steer",
    "sr_torque_error_closed_form",
    "steer",
    "unit_null_vector",
]
