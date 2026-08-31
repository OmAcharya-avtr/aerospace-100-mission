"""DetumbleSim: magnetorquer detumbling simulation, sizing and controllability.

Post-deployment detumbling of a rigid spacecraft with magnetorquers, using a
tilted centred-dipole magnetic field along a circular orbit; B-dot and
cross-product control laws with per-axis dipole saturation; the orbit-averaged
first-order detumble-time model; and quantification of the controllability gap
along the instantaneous field direction.  A learned B-dot gain scheduler is
included and benchmarked against classical gain rules.

Research-grade software.  Not flight-qualified, not certified, not approved
for operational aerospace use.
"""

from __future__ import annotations

from .analytic import (
    FieldMoments,
    damping_matrix,
    detumble_time_first_order,
    geometry_factors,
    max_torque_nm,
    modal_time_constants,
    orbit_field_moments,
    saturation_time_bound_s,
)
from .attitude import (
    angular_momentum,
    dcm_to_quat,
    kinetic_energy,
    quat_kinematics,
    quat_multiply,
    quat_normalize,
    quat_to_dcm,
    rigid_body_derivative,
    skew,
)
from .constants import (
    B0_NT,
    IGRF14_2025_G10_NT,
    IGRF14_2025_G11_NT,
    IGRF14_2025_H11_NT,
    MU_EARTH,
    OMEGA_EARTH_RAD_S,
    R_EARTH_M,
)
from .control import (
    BDotController,
    CrossProductController,
    ideal_bdot_torque,
    magnetic_torque,
    perpendicular_component,
)
from .controllability import (
    ControllabilityReport,
    controllability_report,
    instantaneous_projector,
    residual_rate_along,
    uncontrollable_fraction,
)
from .evaluate import (
    ENERGY_WEIGHT,
    RunScore,
    fit_power_law_gain,
    oracle_gain,
    run_policy,
    score_run,
    training_rows,
)
from .features import FEATURE_NAMES, N_FEATURES, TelemetryWindow, rate_proxy
from .magfield import (
    B0_T,
    DIPOLE_G_NT,
    dipole_field_ecef,
    dipole_field_eci,
    dipole_tilt_deg,
    field_magnitude_nt,
    geomagnetic_north_pole_deg,
    spherical_position_ecef,
)
from .metrics import Interval, format_interval, mean_ci, paired_difference_ci
from .orbit import CircularOrbit
from .policies import (
    FixedGainPolicy,
    PowerLawGainPolicy,
    ScheduledGainPolicy,
    SizedGainPolicy,
    wrap_with_saturation_feedback,
)
from .scenarios import DEFAULT_TARGET_RATE_RAD_S, Scenario, sample_scenario, sample_scenarios
from .scheduler import GainScheduler
from .simulate import (
    DetumbleConfig,
    DetumbleResult,
    crossing_time,
    field_history_eci,
    simulate_detumble,
)
from .spacecraft import Magnetorquer, inertia_from_diagonal, validate_inertia

__version__ = "0.1.0"

__all__ = [
    "B0_NT",
    "B0_T",
    "BDotController",
    "CircularOrbit",
    "ControllabilityReport",
    "CrossProductController",
    "DEFAULT_TARGET_RATE_RAD_S",
    "DIPOLE_G_NT",
    "DetumbleConfig",
    "DetumbleResult",
    "ENERGY_WEIGHT",
    "FEATURE_NAMES",
    "FieldMoments",
    "FixedGainPolicy",
    "GainScheduler",
    "IGRF14_2025_G10_NT",
    "IGRF14_2025_G11_NT",
    "IGRF14_2025_H11_NT",
    "Interval",
    "MU_EARTH",
    "Magnetorquer",
    "N_FEATURES",
    "OMEGA_EARTH_RAD_S",
    "PowerLawGainPolicy",
    "R_EARTH_M",
    "RunScore",
    "Scenario",
    "ScheduledGainPolicy",
    "SizedGainPolicy",
    "TelemetryWindow",
    "__version__",
    "angular_momentum",
    "controllability_report",
    "crossing_time",
    "damping_matrix",
    "dcm_to_quat",
    "detumble_time_first_order",
    "dipole_field_ecef",
    "dipole_field_eci",
    "dipole_tilt_deg",
    "field_history_eci",
    "field_magnitude_nt",
    "fit_power_law_gain",
    "format_interval",
    "geomagnetic_north_pole_deg",
    "geometry_factors",
    "ideal_bdot_torque",
    "inertia_from_diagonal",
    "instantaneous_projector",
    "kinetic_energy",
    "magnetic_torque",
    "max_torque_nm",
    "mean_ci",
    "modal_time_constants",
    "oracle_gain",
    "orbit_field_moments",
    "paired_difference_ci",
    "perpendicular_component",
    "quat_kinematics",
    "quat_multiply",
    "quat_normalize",
    "quat_to_dcm",
    "rate_proxy",
    "residual_rate_along",
    "rigid_body_derivative",
    "run_policy",
    "sample_scenario",
    "sample_scenarios",
    "saturation_time_bound_s",
    "score_run",
    "simulate_detumble",
    "skew",
    "spherical_position_ecef",
    "training_rows",
    "uncontrollable_fraction",
    "validate_inertia",
    "wrap_with_saturation_feedback",
]
