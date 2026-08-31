"""Spacecraft and orbit description objects.

Both are frozen dataclasses that validate on construction, so an invalid inertia tensor
or a negative area fails at the point of definition rather than inside a torque model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _validate as _v
from .constants import DEFAULT_DRAG_COEFFICIENT, MU_EARTH, R_EARTH_EQUATORIAL
from .frames import orbital_period

__all__ = ["Spacecraft", "Orbit"]


@dataclass(frozen=True)
class Spacecraft:
    """Rigid-body properties needed for the four environmental disturbance torques.

    All offsets are the vector from the centre of mass to the relevant centre of
    pressure, expressed in the **body** frame, in metres. The sign convention matters:
    the torque is ``offset x force``.

    Attributes
    ----------
    inertia : ndarray (3, 3)
        Inertia tensor about the centre of mass, body axes [kg m^2]. A length-3
        sequence is accepted and read as principal moments.
    drag_area_m2 : float
        Projected area normal to the relative wind [m^2]. Held constant over the orbit
        by this package; see :mod:`disturbtorque.torques` for the consequence.
    drag_coefficient : float
        Free-molecular drag coefficient, dimensionless.
    cp_aero_offset_m : ndarray (3,)
        Centre-of-pressure minus centre-of-mass, body frame [m].
    srp_area_m2 : float
        Projected sunlit area [m^2].
    srp_reflectance : float
        Reflectance factor q, dimensionless, 0 for a perfect absorber and 1 for a
        perfect specular reflector normal to the Sun; the force carries (1 + q).
    cp_srp_offset_m : ndarray (3,)
        Optical centre-of-pressure minus centre-of-mass, body frame [m].
    residual_dipole_am2 : ndarray (3,)
        Residual magnetic dipole of the vehicle, body frame [A m^2].
    mass_kg : float or None
        Mass [m]; carried for reporting only, no torque depends on it.
    """

    inertia: NDArray[np.float64]
    drag_area_m2: float = 0.0
    drag_coefficient: float = DEFAULT_DRAG_COEFFICIENT
    cp_aero_offset_m: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )
    srp_area_m2: float = 0.0
    srp_reflectance: float = 0.6
    cp_srp_offset_m: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    residual_dipole_am2: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    mass_kg: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "inertia", _v.as_inertia_matrix(self.inertia))
        object.__setattr__(
            self, "drag_area_m2", _v.non_negative(self.drag_area_m2, "drag_area_m2")
        )
        object.__setattr__(
            self, "drag_coefficient", _v.non_negative(self.drag_coefficient, "drag_coefficient")
        )
        object.__setattr__(
            self, "cp_aero_offset_m", _v.as_vector3(self.cp_aero_offset_m, "cp_aero_offset_m")
        )
        object.__setattr__(self, "srp_area_m2", _v.non_negative(self.srp_area_m2, "srp_area_m2"))
        object.__setattr__(
            self, "srp_reflectance", _v.in_range(self.srp_reflectance, "srp_reflectance", 0.0, 1.0)
        )
        object.__setattr__(
            self, "cp_srp_offset_m", _v.as_vector3(self.cp_srp_offset_m, "cp_srp_offset_m")
        )
        object.__setattr__(
            self,
            "residual_dipole_am2",
            _v.as_vector3(self.residual_dipole_am2, "residual_dipole_am2"),
        )
        if self.mass_kg is not None:
            object.__setattr__(self, "mass_kg", _v.positive(self.mass_kg, "mass_kg"))

    @property
    def principal_moments(self) -> NDArray[np.float64]:
        """Principal moments of inertia [kg m^2], ascending."""
        return np.linalg.eigvalsh(self.inertia)


@dataclass(frozen=True)
class Orbit:
    """Circular Earth orbit with a fixed pointing offset from nadir.

    Attributes
    ----------
    altitude_m : float
        Altitude above the WGS-84 equatorial radius [m].
    inclination_rad, raan_rad : float
        Inclination and right ascension of the ascending node [rad].
    yaw_rad, pitch_rad, roll_rad : float
        Fixed 3-2-1 attitude offsets of the body frame from LVLH [rad]. All zero is
        perfect nadir pointing.
    mu : float
        Gravitational parameter [m^3 s^-2].
    """

    altitude_m: float
    inclination_rad: float = 0.0
    raan_rad: float = 0.0
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    mu: float = MU_EARTH

    def __post_init__(self) -> None:
        alt = _v.as_finite_float(self.altitude_m, "altitude_m")
        if alt <= 0.0:
            raise ValueError(f"altitude_m must be > 0 (an orbit, not a hole), got {alt!r}")
        object.__setattr__(self, "altitude_m", alt)
        object.__setattr__(
            self,
            "inclination_rad",
            _v.in_range(self.inclination_rad, "inclination_rad", -np.pi, np.pi),
        )
        for name in ("raan_rad", "yaw_rad", "pitch_rad", "roll_rad"):
            object.__setattr__(self, name, _v.as_finite_float(getattr(self, name), name))
        object.__setattr__(self, "mu", _v.positive(self.mu, "mu"))

    @property
    def radius_m(self) -> float:
        """Orbit radius from the Earth centre [m]."""
        return R_EARTH_EQUATORIAL + self.altitude_m

    @property
    def period_s(self) -> float:
        """Keplerian orbital period [s]."""
        return orbital_period(self.radius_m, self.mu)

    @property
    def speed_ms(self) -> float:
        """Inertial circular speed sqrt(mu/r) [m s^-1]."""
        return float(np.sqrt(self.mu / self.radius_m))

    @property
    def mean_motion_rad_s(self) -> float:
        """Mean motion sqrt(mu/r^3) [rad s^-1]."""
        return float(np.sqrt(self.mu / self.radius_m**3))


def _asarray3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    return _v.as_vector3(value, name)
