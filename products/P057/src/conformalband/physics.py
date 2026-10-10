"""Steady level-flight energy model for a small fixed-wing electric UAV.

Equations
---------
Drag polar (Anderson, J.D., *Aircraft Performance and Design*, McGraw-Hill,
1999, ch. 2-3; standard result):

    C_D = C_D0 + C_L^2 / (pi * AR * e),        AR = b^2 / S

Steady level flight, lift equals weight, so the induced drag power is

    P_induced = 2 * (m g)^2 / (rho * V * pi * b^2 * e)              [W]

and the parasite drag power is

    P_parasite = 0.5 * rho * V^3 * S * C_D0                          [W]

Electrical power at the battery terminals, with a propulsive efficiency and a
constant avionics draw (the battery-powered-aircraft bookkeeping of Traub,
L.W., "Range and Endurance Estimates for Battery-Powered Aircraft", *Journal
of Aircraft*, Vol. 48, No. 2, 2011):

    P_electrical = (P_parasite + P_induced) / eta(V) + P_avionics    [W]

Energy for a straight leg of ground distance ``d`` flown at true airspeed
``V`` into a headwind component ``w``:

    V_ground = V - w,   t = d / V_ground,   E = P_electrical * t / 3600  [Wh]

Propulsive efficiency is modelled as a quadratic fall-off away from the design
airspeed, which is the term the analytic baseline regressor in
:mod:`conformalband.baseline` cannot represent:

    eta(V) = eta_0 * (1 - c * ((V - V_design) / V_design)^2)

Validity and assumptions
------------------------
- Steady, level, coordinated flight. No climb, no acceleration, no turn.
- Incompressible, low Reynolds number regime; ``C_D0`` and ``e`` constant.
- Headwind component is constant over the leg and strictly less than the true
  airspeed, so ground speed is positive.
- Units are SI throughout except energy, which is watt-hours.
- The model is an illustrative airframe with chosen coefficients. It is not a
  measurement of any vehicle and nothing here is validated against flight
  test data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.80665
"""Standard gravity [m/s^2], CGPM 1901 / ISO 80000-3."""

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class Airframe:
    """Fixed geometry and propulsion coefficients of the illustrative airframe.

    Attributes
    ----------
    wing_area:
        Reference wing area ``S`` [m^2].
    wingspan:
        Wingspan ``b`` [m].
    cd0:
        Zero-lift drag coefficient ``C_D0`` [-].
    oswald:
        Oswald span efficiency ``e`` [-], in (0, 1].
    eta_prop:
        Propulsive efficiency ``eta_0`` at the design airspeed [-], in (0, 1].
    avionics_power:
        Constant non-propulsive electrical draw [W].
    design_airspeed:
        Airspeed at which ``eta(V) = eta_0`` [m/s].
    eta_curvature:
        Curvature ``c`` [-] of the efficiency fall-off. Zero reproduces a
        constant-efficiency model.
    """

    wing_area: float = 0.30
    wingspan: float = 1.20
    cd0: float = 0.035
    oswald: float = 0.85
    eta_prop: float = 0.62
    avionics_power: float = 12.0
    design_airspeed: float = 20.0
    eta_curvature: float = 2.50

    def __post_init__(self) -> None:
        if self.wing_area <= 0.0:
            raise ValueError(f"wing_area must be > 0 m^2, got {self.wing_area}")
        if self.wingspan <= 0.0:
            raise ValueError(f"wingspan must be > 0 m, got {self.wingspan}")
        if self.cd0 <= 0.0:
            raise ValueError(f"cd0 must be > 0, got {self.cd0}")
        if not 0.0 < self.oswald <= 1.0:
            raise ValueError(f"oswald must be in (0, 1], got {self.oswald}")
        if not 0.0 < self.eta_prop <= 1.0:
            raise ValueError(f"eta_prop must be in (0, 1], got {self.eta_prop}")
        if self.avionics_power < 0.0:
            raise ValueError(f"avionics_power must be >= 0 W, got {self.avionics_power}")
        if self.design_airspeed <= 0.0:
            raise ValueError(f"design_airspeed must be > 0 m/s, got {self.design_airspeed}")
        if self.eta_curvature < 0.0:
            raise ValueError(f"eta_curvature must be >= 0, got {self.eta_curvature}")

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio ``b^2 / S`` [-]."""
        return self.wingspan**2 / self.wing_area


DEFAULT_AIRFRAME = Airframe()
"""The airframe every shipped dataset and validation script uses."""


def propulsive_efficiency(
    airspeed: np.ndarray, airframe: Airframe = DEFAULT_AIRFRAME
) -> np.ndarray:
    """Propulsive efficiency ``eta(V)`` [-].

    Parameters
    ----------
    airspeed:
        True airspeed [m/s], positive.
    airframe:
        Geometry and propulsion coefficients.

    Returns
    -------
    ndarray
        Efficiency in (0, 1], same shape as ``airspeed``.
    """
    v = np.asarray(airspeed, dtype=float)
    if np.any(v <= 0.0):
        raise ValueError("airspeed must be > 0 m/s everywhere")
    rel = (v - airframe.design_airspeed) / airframe.design_airspeed
    eta = airframe.eta_prop * (1.0 - airframe.eta_curvature * rel**2)
    if np.any(eta <= 0.0):
        raise ValueError(
            "propulsive efficiency fell to zero or below; airspeed is far outside the "
            "validity range of the quadratic efficiency model"
        )
    return eta


def level_flight_power(
    airspeed: np.ndarray,
    mass: np.ndarray,
    air_density: np.ndarray,
    airframe: Airframe = DEFAULT_AIRFRAME,
    *,
    constant_efficiency: bool = False,
) -> np.ndarray:
    """Electrical power for steady level flight [W].

    Parameters
    ----------
    airspeed:
        True airspeed [m/s].
    mass:
        All-up mass [kg].
    air_density:
        Ambient air density [kg/m^3].
    airframe:
        Geometry and propulsion coefficients.
    constant_efficiency:
        If ``True``, use ``eta_0`` for all airspeeds. This is the form the
        analytic baseline regressor fits, and the difference between the two
        is the model misspecification the audit measures.

    Returns
    -------
    ndarray
        Electrical power at the battery terminals [W].
    """
    v = np.asarray(airspeed, dtype=float)
    m = np.asarray(mass, dtype=float)
    rho = np.asarray(air_density, dtype=float)
    if np.any(v <= 0.0):
        raise ValueError("airspeed must be > 0 m/s everywhere")
    if np.any(m <= 0.0):
        raise ValueError("mass must be > 0 kg everywhere")
    if np.any(rho <= 0.0):
        raise ValueError("air_density must be > 0 kg/m^3 everywhere")

    parasite = 0.5 * rho * v**3 * airframe.wing_area * airframe.cd0
    weight = m * GRAVITY
    induced = 2.0 * weight**2 / (rho * v * np.pi * airframe.wingspan**2 * airframe.oswald)
    eta = (
        np.full(np.broadcast(v, m, rho).shape, airframe.eta_prop)
        if constant_efficiency
        else propulsive_efficiency(v, airframe)
    )
    return (parasite + induced) / eta + airframe.avionics_power


def leg_energy(
    airspeed: np.ndarray,
    mass: np.ndarray,
    air_density: np.ndarray,
    headwind: np.ndarray,
    distance: np.ndarray,
    airframe: Airframe = DEFAULT_AIRFRAME,
    *,
    constant_efficiency: bool = False,
) -> np.ndarray:
    """Battery energy for one straight leg [Wh].

    Parameters
    ----------
    airspeed:
        True airspeed [m/s].
    mass:
        All-up mass [kg].
    air_density:
        Ambient air density [kg/m^3].
    headwind:
        Headwind component along the leg [m/s], positive into the wind.
        Must be strictly less than ``airspeed``.
    distance:
        Ground distance of the leg [m], positive.
    airframe:
        Geometry and propulsion coefficients.
    constant_efficiency:
        See :func:`level_flight_power`.

    Returns
    -------
    ndarray
        Energy consumed over the leg [Wh].
    """
    v = np.asarray(airspeed, dtype=float)
    w = np.asarray(headwind, dtype=float)
    d = np.asarray(distance, dtype=float)
    if np.any(d <= 0.0):
        raise ValueError("distance must be > 0 m everywhere")
    ground_speed = v - w
    if np.any(ground_speed <= 0.0):
        raise ValueError(
            "headwind must be strictly less than airspeed; ground speed reached zero, "
            "so leg time is undefined"
        )
    power = level_flight_power(
        v, mass, air_density, airframe, constant_efficiency=constant_efficiency
    )
    return power * (d / ground_speed) / SECONDS_PER_HOUR
