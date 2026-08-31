"""keepout -- celestial keep-out geometry for spacecraft pointing.

Sun, Earth and Moon exclusion cones for sensitive instruments; a violation test
for a given boresight direction; allowed-attitude region computation; and
keep-out-aware pointing windows over an orbit.

Angles are radians, lengths metres, times seconds, solid angles steradians.
The package is deterministic apart from the explicitly seeded Monte Carlo
estimator; there are no machine-learning components.

Educational and research-grade. Not flight-qualified, not certified, not
approved for operational aerospace use.
"""

from .bodies import (
    ASTRONOMICAL_UNIT_M,
    EARTH_MU,
    EARTH_RADIUS_M,
    J2000_JD,
    MOON_RADIUS_M,
    SUN_RADIUS_M,
    angular_radius,
    earth_angular_radius,
    earth_direction_from_position,
    julian_date,
    moon_direction_mod,
    sun_direction_mod,
)
from .cones import ExclusionCone, KeepOutSet, body_exclusion_cone
from .geometry import (
    angular_separation,
    cap_intersection_solid_angle,
    cap_solid_angle,
    cap_union_solid_angle,
    fibonacci_sphere,
    random_rotations,
    rotation_matrix,
    spherical_to_unit,
    unit,
    unit_to_spherical,
)
from .regions import (
    SolidAngleEstimate,
    allowed_directions,
    allowed_fraction,
    allowed_mask,
    allowed_solid_angle,
    allowed_solid_angle_monte_carlo,
)
from .windows import (
    OrbitPointingProblem,
    Window,
    circular_orbit_positions,
    orbital_period,
    windows_from_margin,
)

__version__ = "0.1.0"

__all__ = [
    "ASTRONOMICAL_UNIT_M",
    "EARTH_MU",
    "EARTH_RADIUS_M",
    "ExclusionCone",
    "J2000_JD",
    "KeepOutSet",
    "MOON_RADIUS_M",
    "OrbitPointingProblem",
    "SUN_RADIUS_M",
    "SolidAngleEstimate",
    "Window",
    "allowed_directions",
    "allowed_fraction",
    "allowed_mask",
    "allowed_solid_angle",
    "allowed_solid_angle_monte_carlo",
    "angular_radius",
    "angular_separation",
    "body_exclusion_cone",
    "cap_intersection_solid_angle",
    "cap_solid_angle",
    "cap_union_solid_angle",
    "circular_orbit_positions",
    "earth_angular_radius",
    "earth_direction_from_position",
    "fibonacci_sphere",
    "julian_date",
    "moon_direction_mod",
    "orbital_period",
    "random_rotations",
    "rotation_matrix",
    "spherical_to_unit",
    "sun_direction_mod",
    "unit",
    "unit_to_spherical",
    "windows_from_margin",
    "__version__",
]
