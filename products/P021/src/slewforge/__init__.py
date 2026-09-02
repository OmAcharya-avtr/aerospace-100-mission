"""SlewForge: constrained rest-to-rest slew planning for spacecraft.

Rigid-body attitude dynamics with reaction-wheel actuation and saturation;
eigenaxis, bang-bang, trapezoidal and smoothed rest-to-rest profiles; hard
keep-out cones for the Sun, Earth and Moon with closed-form violation
detection along the path; a constrained planner that returns either a feasible
path or a named reason it could not; time and momentum cost accounting; and a
learned warm start benchmarked against the cold-started optimiser.

Conventions: quaternions are scalar-first, products are Hamilton, rotations are
active, and an attitude quaternion maps body vectors to the inertial frame --
the conventions of QuatKit (P007). Angles are radians, times seconds, torques
N*m, momenta N*m*s, inertias kg m^2.

This software is research-grade. It is not flight-qualified, not certified,
and not approved for operational aerospace use.
"""

from __future__ import annotations

from .attitude import (
    axis_angle_from_quat,
    cross3,
    quat_angle,
    quat_conjugate,
    quat_from_axis_angle,
    quat_from_rotvec,
    quat_identity,
    quat_multiply,
    quat_normalize,
    quat_relative,
    quat_rotate,
    quat_slerp,
    quat_to_dcm,
    rotate_about_axis,
    unit_vector,
)
from .dataset import (
    FEATURE_NAMES,
    PlanningDataset,
    generate_dataset,
    generate_problems,
    problem_features,
    reference_spacecraft,
)
from .dynamics import (
    RigidBody,
    SimulationResult,
    eigenaxis_torque,
    inertial_momentum,
    propagate,
    simulate_profile,
)
from .keepout import (
    ASTRONOMICAL_UNIT_M,
    EARTH_RADIUS_M,
    MOON_RADIUS_M,
    SUN_RADIUS_M,
    ArcViolation,
    KeepOutCone,
    KeepOutSet,
    angular_radius,
    arc_coefficients_raw,
    body_keepout_cone,
    earth_angular_radius,
    min_margin_on_arc_raw,
)
from .ml import LearnedWarmStart, WarmStartPrediction
from .planner import (
    INFEASIBILITY_REASONS,
    ActuatorCheck,
    Instrument,
    PlanResult,
    SlewPath,
    SlewProblem,
    SlewSegment,
    canonical_frame,
    canonical_rotvec,
    cold_start_points,
    direct_violations,
    path_margins,
    path_min_margin,
    path_violations,
    plan,
    verify_actuators,
)
from .profiles import (
    PROFILE_NAMES,
    SlewProfile,
    bang_bang_profile,
    make_profile,
    smoothed_profile,
)
from .wheels import WheelArray, orthogonal_wheels, pyramid_wheels

__version__ = "0.1.0"

__all__ = [
    "ASTRONOMICAL_UNIT_M",
    "EARTH_RADIUS_M",
    "FEATURE_NAMES",
    "INFEASIBILITY_REASONS",
    "MOON_RADIUS_M",
    "PROFILE_NAMES",
    "SUN_RADIUS_M",
    "ActuatorCheck",
    "ArcViolation",
    "Instrument",
    "KeepOutCone",
    "KeepOutSet",
    "LearnedWarmStart",
    "PlanResult",
    "PlanningDataset",
    "RigidBody",
    "SimulationResult",
    "SlewPath",
    "SlewProblem",
    "SlewProfile",
    "SlewSegment",
    "WarmStartPrediction",
    "WheelArray",
    "__version__",
    "angular_radius",
    "arc_coefficients_raw",
    "axis_angle_from_quat",
    "bang_bang_profile",
    "body_keepout_cone",
    "canonical_frame",
    "canonical_rotvec",
    "cold_start_points",
    "cross3",
    "direct_violations",
    "earth_angular_radius",
    "eigenaxis_torque",
    "generate_dataset",
    "generate_problems",
    "inertial_momentum",
    "make_profile",
    "min_margin_on_arc_raw",
    "orthogonal_wheels",
    "path_margins",
    "path_min_margin",
    "path_violations",
    "plan",
    "problem_features",
    "propagate",
    "pyramid_wheels",
    "quat_angle",
    "quat_conjugate",
    "quat_from_axis_angle",
    "quat_from_rotvec",
    "quat_identity",
    "quat_multiply",
    "quat_normalize",
    "quat_relative",
    "quat_rotate",
    "quat_slerp",
    "quat_to_dcm",
    "reference_spacecraft",
    "rotate_about_axis",
    "simulate_profile",
    "smoothed_profile",
    "unit_vector",
    "verify_actuators",
]
