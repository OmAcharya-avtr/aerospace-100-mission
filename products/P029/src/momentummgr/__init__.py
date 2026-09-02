"""momentummgr — reaction-wheel momentum management for Earth-orbiting spacecraft.

What is here
------------
* An independent implementation of the four environmental disturbance torques and of the
  momentum they accumulate over a circular orbit, with a Gauss-Legendre reference
  quadrature that splits the solar term at the analytic eclipse boundaries
  (:mod:`momentummgr.torques`, :mod:`momentummgr.accumulation`). It exists so that this
  package's momentum budget can be cross-checked against P027 ``disturbtorque`` without
  either code sharing a line with the other; the result is in ``validation/VALIDATION.md``.
* Reaction-wheel array algebra: redundant allocation, the conservative body-momentum
  envelope, and exact null-space biasing to keep wheels off zero speed
  (:mod:`momentummgr.wheels`).
* Magnetic and thruster desaturation, including the instantaneous field-direction
  controllability limit and the time-averaged controllability Gramian
  (:mod:`momentummgr.desaturation`).
* A desaturation scheduling problem, a tuned classical fixed-threshold scheduler and a
  learned scheduler with a confidence output, benchmarked against each other on held-out
  episodes with bootstrap confidence intervals (:mod:`momentummgr.episodes`,
  :mod:`momentummgr.policies`, :mod:`momentummgr.learned`).

Research-grade software. Not flight-qualified, not certified, not approved for
operational aerospace use. The learned scheduler is not certified for operational flight
use.
"""

from __future__ import annotations

from .accumulation import (
    SOURCES,
    OrbitSweep,
    momentum_budget,
    momentum_history_eci,
    momentum_per_orbit_eci,
    secular_torque_eci,
    sweep_orbit,
)
from .constants import (
    ASTRONOMICAL_UNIT,
    DEFAULT_DRAG_COEFFICIENT,
    EARTH_REDUCED_DIPOLE,
    MU_EARTH,
    OMEGA_EARTH,
    R_EARTH_EQUATORIAL,
    R_EARTH_MEAN,
    SOLAR_IRRADIANCE_1AU,
    SPEED_OF_LIGHT,
    SRP_PRESSURE_1AU,
    STANDARD_GRAVITY,
)
from .desaturation import (
    MagneticCommand,
    ThrusterDump,
    averaged_controllability,
    dipole_cost,
    magnetic_dump_command,
    thruster_dump,
    uncontrollable_fraction,
)
from .environment import (
    CircularOrbit,
    SpacecraftProperties,
    beta_angle,
    body_dcm_from_lvlh,
    circular_state,
    density,
    dipole_field_eci,
    eclipse_boundaries,
    eclipse_fraction,
    is_illuminated,
    lvlh_dcm,
    node_axes,
    orbital_period,
    reference_orbit,
    reference_smallsat,
    sun_direction_for_beta,
)
from .episodes import (
    FEATURE_NAMES,
    N_FEATURES,
    Episode,
    EpisodeMetrics,
    Rollout,
    build_episode,
    episode_cost,
    rollout,
    sample_episode,
    simulate_masks,
)
from .learned import (
    LearnedScheduler,
    MaskSearchResult,
    harvest_training_rows,
    search_best_mask,
    train_scheduler,
)
from .policies import (
    AlwaysOnScheduler,
    FixedThresholdScheduler,
    NeverScheduler,
    evaluate_policy,
    tune_fixed_threshold,
)
from .torques import (
    aerodynamic_force,
    aerodynamic_torque,
    gravity_gradient_torque,
    gravity_gradient_worst_case,
    residual_dipole_torque,
    srp_force,
    srp_torque,
)
from .wheels import (
    Allocation,
    WheelArray,
    count_zero_crossings,
    orthogonal_three,
    pyramid_four,
    tetrahedral_four,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # constants
    "MU_EARTH",
    "R_EARTH_EQUATORIAL",
    "R_EARTH_MEAN",
    "OMEGA_EARTH",
    "SPEED_OF_LIGHT",
    "ASTRONOMICAL_UNIT",
    "SOLAR_IRRADIANCE_1AU",
    "SRP_PRESSURE_1AU",
    "EARTH_REDUCED_DIPOLE",
    "STANDARD_GRAVITY",
    "DEFAULT_DRAG_COEFFICIENT",
    # environment
    "CircularOrbit",
    "SpacecraftProperties",
    "node_axes",
    "orbital_period",
    "circular_state",
    "lvlh_dcm",
    "body_dcm_from_lvlh",
    "sun_direction_for_beta",
    "beta_angle",
    "eclipse_boundaries",
    "eclipse_fraction",
    "is_illuminated",
    "density",
    "dipole_field_eci",
    "reference_smallsat",
    "reference_orbit",
    # torques
    "gravity_gradient_torque",
    "gravity_gradient_worst_case",
    "aerodynamic_force",
    "aerodynamic_torque",
    "srp_force",
    "srp_torque",
    "residual_dipole_torque",
    # accumulation
    "SOURCES",
    "OrbitSweep",
    "sweep_orbit",
    "momentum_per_orbit_eci",
    "momentum_history_eci",
    "secular_torque_eci",
    "momentum_budget",
    # wheels
    "WheelArray",
    "Allocation",
    "pyramid_four",
    "tetrahedral_four",
    "orthogonal_three",
    "count_zero_crossings",
    # desaturation
    "MagneticCommand",
    "magnetic_dump_command",
    "uncontrollable_fraction",
    "averaged_controllability",
    "dipole_cost",
    "ThrusterDump",
    "thruster_dump",
    # scheduling
    "Episode",
    "EpisodeMetrics",
    "Rollout",
    "N_FEATURES",
    "FEATURE_NAMES",
    "sample_episode",
    "build_episode",
    "simulate_masks",
    "rollout",
    "episode_cost",
    "FixedThresholdScheduler",
    "AlwaysOnScheduler",
    "NeverScheduler",
    "tune_fixed_threshold",
    "evaluate_policy",
    "LearnedScheduler",
    "MaskSearchResult",
    "search_best_mask",
    "harvest_training_rows",
    "train_scheduler",
]
