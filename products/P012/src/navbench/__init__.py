"""navbench — a controlled bench for aerospace attitude and navigation filters.

Runs one truth trajectory and one sensor suite through a linear Kalman filter,
an EKF, a UKF and a multiplicative EKF, and scores them on **consistency**
(NEES and NIS against chi-squared bounds) as well as on error.  Adds a learned
adaptive process-noise tuner benchmarked honestly against a fixed hand-tuned
``Q`` and against classical innovation-based adaptive estimation.

Research-grade software (validation Level 3).  Not flight-qualified, not
certified, not approved for operational aerospace use.  See README.md and
validation/VALIDATION.md.
"""

from .adaptive import (
    FEATURE_NAMES,
    N_FEATURES,
    AdaptiveQPrediction,
    AdaptiveRunResult,
    LearnedAdaptiveQ,
    MehraAdaptiveQ,
    generate_adaptive_dataset,
    innovation_features,
    run_adaptive_kf,
)
from .attitude import (
    axis_angle_from_quat,
    dcm_from_quat,
    euler_moment_derivative,
    euler_zyx_from_quat,
    quat_angle_between,
    quat_canonical,
    quat_conjugate,
    quat_derivative,
    quat_from_axis_angle,
    quat_from_dcm,
    quat_from_euler_zyx,
    quat_from_small_angle,
    quat_identity,
    quat_multiply,
    quat_norm,
    quat_normalize,
    quat_propagate,
    quat_rotate,
    skew,
    small_angle_from_quat,
)
from .bench import DIVERGENCE_QUANTILE, EstimatorScore, compare_scores, score_run
from .consistency import (
    ConsistencyResult,
    WhitenessResult,
    chi2_bounds,
    consistency_test,
    ensemble_consistency,
    innovation_whiteness,
    nees,
    nis,
)
from .ekf import ExtendedKalmanFilter, numerical_jacobian
from .kf import (
    CovarianceCollapseError,
    FilterResult,
    KalmanFilter,
    covariance_health,
    joseph_update,
    steady_state_riccati,
    symmetrize,
)
from .mekf import (
    MekfResult,
    MultiplicativeEKF,
    attitude_state_transition,
    gyro_process_noise,
)
from .models import (
    constant_velocity_2d,
    constant_velocity_cwna,
    constant_velocity_dwna,
    radar_jacobian,
    radar_measurement,
    random_walk,
    simulate_linear_system,
    simulate_radar_scenario,
)
from .sensors import (
    AccelerometerModel,
    GpsModel,
    GpsOutput,
    GyroModel,
    GyroOutput,
    StarTrackerModel,
    SunSensorModel,
    VectorOutput,
    arw_deg_per_sqrt_hour_to_si,
    rrw_deg_per_hour_1p5_to_si,
)
from .truth import (
    MU_EARTH,
    R_EARTH,
    AirborneTruth,
    AttitudeTruth,
    OrbitTruth,
    Trajectory,
    airborne_trajectory,
    attitude_trajectory,
    circular_orbit_state,
    orbit_trajectory,
)
from .ukf import MerweSigmaPoints, UnscentedKalmanFilter, unscented_transform

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # attitude
    "skew",
    "quat_identity",
    "quat_multiply",
    "quat_conjugate",
    "quat_normalize",
    "quat_norm",
    "quat_canonical",
    "dcm_from_quat",
    "quat_from_dcm",
    "quat_from_axis_angle",
    "axis_angle_from_quat",
    "quat_from_small_angle",
    "small_angle_from_quat",
    "quat_rotate",
    "quat_derivative",
    "quat_propagate",
    "quat_angle_between",
    "quat_from_euler_zyx",
    "euler_zyx_from_quat",
    "euler_moment_derivative",
    # truth
    "MU_EARTH",
    "R_EARTH",
    "AttitudeTruth",
    "OrbitTruth",
    "AirborneTruth",
    "Trajectory",
    "attitude_trajectory",
    "orbit_trajectory",
    "airborne_trajectory",
    "circular_orbit_state",
    # sensors
    "GyroModel",
    "GyroOutput",
    "StarTrackerModel",
    "SunSensorModel",
    "AccelerometerModel",
    "GpsModel",
    "GpsOutput",
    "VectorOutput",
    "arw_deg_per_sqrt_hour_to_si",
    "rrw_deg_per_hour_1p5_to_si",
    # filters
    "KalmanFilter",
    "FilterResult",
    "CovarianceCollapseError",
    "joseph_update",
    "symmetrize",
    "covariance_health",
    "steady_state_riccati",
    "ExtendedKalmanFilter",
    "numerical_jacobian",
    "UnscentedKalmanFilter",
    "MerweSigmaPoints",
    "unscented_transform",
    "MultiplicativeEKF",
    "MekfResult",
    "attitude_state_transition",
    "gyro_process_noise",
    # consistency
    "chi2_bounds",
    "nees",
    "nis",
    "ConsistencyResult",
    "consistency_test",
    "ensemble_consistency",
    "innovation_whiteness",
    "WhitenessResult",
    # models
    "random_walk",
    "constant_velocity_cwna",
    "constant_velocity_dwna",
    "constant_velocity_2d",
    "radar_measurement",
    "radar_jacobian",
    "simulate_linear_system",
    "simulate_radar_scenario",
    # adaptive
    "N_FEATURES",
    "FEATURE_NAMES",
    "innovation_features",
    "MehraAdaptiveQ",
    "LearnedAdaptiveQ",
    "AdaptiveQPrediction",
    "AdaptiveRunResult",
    "run_adaptive_kf",
    "generate_adaptive_dataset",
    # bench
    "EstimatorScore",
    "score_run",
    "compare_scores",
    "DIVERGENCE_QUANTILE",
]
