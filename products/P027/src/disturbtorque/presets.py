"""Reference vehicle and orbit used by the examples, the validation scripts and the CLI.

These numbers are *this package's own definition* of a representative case, chosen to be
typical of an ESPA-class LEO microsatellite. They are not taken from any published
spacecraft and are not attributed to one. Every validation number that mentions a
"representative LEO smallsat" comes from exactly this definition, so the comparisons in
``validation/VALIDATION.md`` are reproducible.
"""

from __future__ import annotations

import numpy as np

from .spacecraft import Orbit, Spacecraft

__all__ = ["REFERENCE_SMALLSAT", "REFERENCE_ORBIT", "reference_smallsat", "reference_orbit"]


def reference_smallsat() -> Spacecraft:
    """Representative 100 kg LEO microsatellite, deployed-panel configuration.

    Inertia diag(4.0, 8.0, 10.0) kg m^2; drag area 0.6 m^2 with Cd = 2.2 and a
    (0.02, 0.02, 0.05) m centre-of-pressure offset; sunlit area 1.2 m^2 with q = 0.6 and
    the same offset; residual dipole (0.05, 0.05, 0.10) A m^2, magnitude 0.1225 A m^2.
    """
    return Spacecraft(
        inertia=np.diag([4.0, 8.0, 10.0]),
        drag_area_m2=0.6,
        drag_coefficient=2.2,
        cp_aero_offset_m=np.array([0.02, 0.02, 0.05]),
        srp_area_m2=1.2,
        srp_reflectance=0.6,
        cp_srp_offset_m=np.array([0.02, 0.02, 0.05]),
        residual_dipole_am2=np.array([0.05, 0.05, 0.10]),
        mass_kg=100.0,
    )


def reference_orbit(altitude_km: float = 500.0) -> Orbit:
    """Circular orbit at ``altitude_km``, inclination 51.6 deg, RAAN 0, nadir-pointing
    with 5 deg pitch and 5 deg roll offsets so the gravity-gradient torque is non-zero.
    """
    return Orbit(
        altitude_m=float(altitude_km) * 1000.0,
        inclination_rad=np.radians(51.6),
        raan_rad=0.0,
        pitch_rad=np.radians(5.0),
        roll_rad=np.radians(5.0),
    )


REFERENCE_SMALLSAT: Spacecraft = reference_smallsat()
REFERENCE_ORBIT: Orbit = reference_orbit()
