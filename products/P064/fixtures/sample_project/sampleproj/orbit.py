"""Two-body circular-orbit arithmetic. Bottom of the import graph."""

import math

MU_EARTH = 3.986004418e14  # m^3 s^-2


def circular_velocity(radius):
    """Circular orbital speed at radius (m) in m/s."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    return math.sqrt(MU_EARTH / radius)


def period(radius):
    """Keplerian period of a circular orbit of radius (m), in seconds."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    return 2.0 * math.pi * math.sqrt(radius**3 / MU_EARTH)
