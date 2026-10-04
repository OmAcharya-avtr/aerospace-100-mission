"""dopplerkit -- Doppler and range-rate geometry for a satellite link.

The sign convention, stated once here and identically at every interface in
the package:

* **Range-rate** ``rho_dot = d|rho|/dt`` is **positive when receding**
  (the range is increasing), negative when approaching, zero at closest
  approach.  It is **invariant** under reversing the link direction.
* **Doppler shift** ``Delta_f = -f_carrier * rho_dot / c`` is therefore
  **positive (up-shifted) when approaching**.
* **Two-way coherent** ``Delta_f = -2 G f_uplink * rho_dot / c``, exactly
  twice the one-way shift when ``G = 1`` and both legs are at one epoch.
* **Pre-compensation offset** is the negative of the Doppler shift.

Order retained throughout: O(beta^1) with ``beta = rho_dot/c``.  The
O(beta^2) special-relativistic and two-way cascade terms, and an
order-of-magnitude static gravitational term, are *reported* by
:mod:`dopplerkit.doppler` and the ``relativistic`` CLI subcommand, and are
never applied.

Research-grade and educational.  Not flight-qualified, not certified, not
approved for operational aerospace use.
"""

from __future__ import annotations

from .analytic import CircularOverheadPass
from .constants import (
    C_M_S,
    MU_EARTH_M3_S2,
    OMEGA_EARTH_RAD_S,
    WGS84_A_M,
    WGS84_E2,
    WGS84_F,
)
from .doppler import (
    DOPPLER_CONVENTION,
    DopplerObservable,
    LinkDirection,
    doppler_observable,
    doppler_rate_hz_per_s,
    gravitational_shift_fraction,
    one_way_doppler_hz,
    precompensation_offset_hz,
    relativistic_correction_hz,
    relativistic_fraction_second_order,
    two_way_doppler_hz,
    two_way_doppler_two_leg_hz,
)
from .frames import (
    ecef_to_teme_m,
    elevation_deg,
    geodetic_to_ecef_m,
    gmst_rad,
    julian_date,
    satellite_state_teme,
    station_state_teme,
)
from .geometry import (
    RANGE_RATE_CONVENTION,
    State,
    range_acceleration_mps2,
    range_rate_finite_difference_mps,
    range_rate_mps,
    relative_position_m,
    slant_range_m,
)
from .lighttime import (
    LightTimeSolution,
    TwoWayLightTime,
    down_leg_light_time,
    two_way_light_time,
    up_leg_light_time,
)
from .passprofile import (
    DopplerProfile,
    compute_profile,
    doppler_zero_crossing_s,
    time_of_closest_approach_s,
)

__version__ = "0.1.0"

__all__ = [
    "C_M_S",
    "DOPPLER_CONVENTION",
    "MU_EARTH_M3_S2",
    "OMEGA_EARTH_RAD_S",
    "RANGE_RATE_CONVENTION",
    "WGS84_A_M",
    "WGS84_E2",
    "WGS84_F",
    "CircularOverheadPass",
    "DopplerObservable",
    "DopplerProfile",
    "LightTimeSolution",
    "LinkDirection",
    "State",
    "TwoWayLightTime",
    "__version__",
    "compute_profile",
    "doppler_observable",
    "doppler_rate_hz_per_s",
    "doppler_zero_crossing_s",
    "down_leg_light_time",
    "ecef_to_teme_m",
    "elevation_deg",
    "geodetic_to_ecef_m",
    "gmst_rad",
    "gravitational_shift_fraction",
    "julian_date",
    "one_way_doppler_hz",
    "precompensation_offset_hz",
    "range_acceleration_mps2",
    "range_rate_finite_difference_mps",
    "range_rate_mps",
    "relative_position_m",
    "relativistic_correction_hz",
    "relativistic_fraction_second_order",
    "satellite_state_teme",
    "slant_range_m",
    "station_state_teme",
    "time_of_closest_approach_s",
    "two_way_doppler_hz",
    "two_way_doppler_two_leg_hz",
    "two_way_light_time",
    "up_leg_light_time",
]
