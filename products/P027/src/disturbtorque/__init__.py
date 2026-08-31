"""disturbtorque - environmental disturbance torques for spacecraft attitude sizing.

Four textbook models (gravity gradient, aerodynamic, solar radiation pressure, residual
magnetic dipole), each stated with its source, units, assumptions and validity range;
swept over a circular orbit; split into secular and cyclic parts; and integrated into
the angular momentum a control system has to store and dump.

Nothing here is new physics. The contribution is a small, dependency-light, individually
hand-checked implementation with the validity ranges written down. See
``validation/VALIDATION.md`` for what was checked, against what, and what failed.
"""

from __future__ import annotations

from .constants import (
    DEFAULT_DRAG_COEFFICIENT,
    EARTH_DIPOLE_MOMENT,
    MU_EARTH,
    OMEGA_EARTH,
    R_EARTH_EQUATORIAL,
    R_EARTH_MEAN,
    SOLAR_IRRADIANCE_1AU,
    SOLAR_IRRADIANCE_1AU_SMAD,
    SPEED_OF_LIGHT,
    SRP_PRESSURE_1AU,
)
from .atmosphere import density
from .frames import (
    beta_angle,
    body_from_lvlh,
    circular_orbit_state,
    eclipse_fraction_cylindrical,
    in_eclipse_cylindrical,
    julian_date,
    lvlh_from_eci,
    node_axes,
    orbit_normal,
    orbital_period,
    sun_direction_for_beta,
    sun_distance_au,
    sun_unit_vector_eci,
)
from .magnetic import dipole_field_eci, dipole_field_magnitude, mean_dipole_field_over_orbit
from .presets import REFERENCE_ORBIT, REFERENCE_SMALLSAT, reference_orbit, reference_smallsat
from .profile import SOURCES, TorqueProfile, budget, compute_profile, momentum_accumulation
from .spacecraft import Orbit, Spacecraft
from .torques import (
    aerodynamic_force,
    aerodynamic_torque,
    gravity_gradient_max_magnitude,
    gravity_gradient_planar,
    gravity_gradient_torque,
    magnetic_torque,
    solar_radiation_force,
    solar_radiation_torque,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEFAULT_DRAG_COEFFICIENT",
    "EARTH_DIPOLE_MOMENT",
    "MU_EARTH",
    "OMEGA_EARTH",
    "R_EARTH_EQUATORIAL",
    "R_EARTH_MEAN",
    "SOLAR_IRRADIANCE_1AU",
    "SOLAR_IRRADIANCE_1AU_SMAD",
    "SOURCES",
    "SPEED_OF_LIGHT",
    "SRP_PRESSURE_1AU",
    "Orbit",
    "REFERENCE_ORBIT",
    "REFERENCE_SMALLSAT",
    "Spacecraft",
    "TorqueProfile",
    "aerodynamic_force",
    "aerodynamic_torque",
    "beta_angle",
    "body_from_lvlh",
    "budget",
    "circular_orbit_state",
    "compute_profile",
    "density",
    "dipole_field_eci",
    "dipole_field_magnitude",
    "eclipse_fraction_cylindrical",
    "gravity_gradient_max_magnitude",
    "gravity_gradient_planar",
    "gravity_gradient_torque",
    "in_eclipse_cylindrical",
    "julian_date",
    "lvlh_from_eci",
    "magnetic_torque",
    "mean_dipole_field_over_orbit",
    "momentum_accumulation",
    "node_axes",
    "orbit_normal",
    "orbital_period",
    "reference_orbit",
    "reference_smallsat",
    "solar_radiation_force",
    "solar_radiation_torque",
    "sun_direction_for_beta",
    "sun_distance_au",
    "sun_unit_vector_eci",
]
