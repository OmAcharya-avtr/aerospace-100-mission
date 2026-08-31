"""Orbital sweep: torque time histories, secular/cyclic split and momentum accumulation.

The sweep is deterministic. One orbit is sampled uniformly in argument of latitude,
which for a circular orbit is uniform in time, so the trapezoidal rule on the closed
period is the natural quadrature and converges rapidly for the smooth contributions.
The solar term is *not* smooth (it switches at eclipse entry and exit), which is why
the eclipse-driven quadrature error is reported separately in
``validation/momentum_integration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _validate as _v
from .atmosphere import density as atmospheric_density
from .constants import OMEGA_EARTH, R_EARTH_EQUATORIAL, SRP_PRESSURE_1AU
from .frames import body_from_lvlh, circular_orbit_state, in_eclipse_cylindrical, lvlh_from_eci
from .magnetic import dipole_field_eci
from .spacecraft import Orbit, Spacecraft
from .torques import (
    aerodynamic_torque,
    gravity_gradient_torque,
    magnetic_torque,
    solar_radiation_torque,
)

__all__ = ["SOURCES", "TorqueProfile", "compute_profile", "momentum_accumulation", "budget"]

SOURCES: tuple[str, ...] = ("gravity_gradient", "aerodynamic", "solar", "magnetic")

Frame = Literal["body", "eci"]


@dataclass(frozen=True)
class TorqueProfile:
    """One orbit of disturbance-torque history and the geometry that produced it.

    Attributes
    ----------
    u_rad : ndarray (N,)
        Argument of latitude samples, 0 to 2*pi inclusive [rad].
    time_s : ndarray (N,)
        Time from the ascending node [s].
    period_s : float
        Orbital period [s].
    r_eci, v_eci : ndarray (N, 3)
        Position [m] and inertial velocity [m s^-1].
    illuminated : ndarray (N,) of bool
        True where the vehicle is in sunlight (cylindrical shadow model).
    density_kg_m3 : ndarray (N,)
        Atmospheric density at each sample [kg m^-3].
    b_field_eci_t : ndarray (N, 3)
        Geomagnetic field [T].
    torques_body, torques_eci : dict[str, ndarray (N, 3)]
        Per-source torque [N m] in the body and ECI frames.
    """

    u_rad: NDArray[np.float64]
    time_s: NDArray[np.float64]
    period_s: float
    r_eci: NDArray[np.float64]
    v_eci: NDArray[np.float64]
    illuminated: NDArray[np.bool_]
    density_kg_m3: NDArray[np.float64]
    b_field_eci_t: NDArray[np.float64]
    torques_body: dict[str, NDArray[np.float64]]
    torques_eci: dict[str, NDArray[np.float64]]

    def _store(self, frame: Frame) -> dict[str, NDArray[np.float64]]:
        if frame == "body":
            return self.torques_body
        if frame == "eci":
            return self.torques_eci
        raise ValueError(f"frame must be 'body' or 'eci', got {frame!r}")

    def torque(self, source: str = "total", frame: Frame = "body") -> NDArray[np.float64]:
        """Torque history [N m], shape (N, 3), for one source or their sum."""
        store = self._store(frame)
        if source == "total":
            return np.sum(np.stack([store[s] for s in SOURCES]), axis=0)
        if source not in store:
            raise ValueError(f"source must be 'total' or one of {SOURCES}, got {source!r}")
        return store[source]

    def secular(self, source: str = "total", frame: Frame = "body") -> NDArray[np.float64]:
        """Orbit-averaged (secular) torque vector [N m], shape (3,).

        The average is the trapezoidal integral over the closed period divided by the
        period, which for a periodic sampled function on a uniform grid is the natural
        estimator.
        """
        t = self.torque(source, frame)
        return np.trapezoid(t, self.time_s, axis=0) / self.period_s

    def cyclic(self, source: str = "total", frame: Frame = "body") -> NDArray[np.float64]:
        """Cyclic (zero-mean) part of the torque history [N m], shape (N, 3)."""
        return self.torque(source, frame) - self.secular(source, frame)

    def cyclic_peak(self, source: str = "total", frame: Frame = "body") -> float:
        """Largest instantaneous magnitude of the cyclic torque over the orbit [N m]."""
        return float(np.max(np.linalg.norm(self.cyclic(source, frame), axis=1)))

    def peak_magnitude(self, source: str = "total", frame: Frame = "body") -> float:
        """Largest instantaneous torque magnitude over the orbit [N m]."""
        return float(np.max(np.linalg.norm(self.torque(source, frame), axis=1)))

    def rms_magnitude(self, source: str = "total", frame: Frame = "body") -> float:
        """Root-mean-square torque magnitude over the orbit [N m]."""
        t = self.torque(source, frame)
        return float(np.sqrt(np.mean(np.sum(t**2, axis=1))))

    @property
    def eclipse_fraction(self) -> float:
        """Fraction of the orbit spent in shadow, from the sampled illumination flag."""
        return float(1.0 - np.mean(self.illuminated))


def compute_profile(
    spacecraft: Spacecraft,
    orbit: Orbit,
    sun_unit_eci: ArrayLike,
    n_samples: int = 721,
    distance_au: float = 1.0,
    co_rotating_atmosphere: bool = True,
    pressure_1au: float = SRP_PRESSURE_1AU,
) -> TorqueProfile:
    """Evaluate all four disturbance torques over one orbit.

    Parameters
    ----------
    spacecraft : Spacecraft
        Vehicle properties.
    orbit : Orbit
        Circular orbit plus the fixed pointing offset from nadir.
    sun_unit_eci : array_like (3,)
        Unit vector from the Earth to the Sun in ECI. Build one with
        :func:`disturbtorque.frames.sun_unit_vector_eci` for a calendar date or
        :func:`disturbtorque.frames.sun_direction_for_beta` for a chosen beta angle.
    n_samples : int
        Samples over one orbit, endpoints included. Must be >= 9.
    distance_au : float
        Earth-Sun distance [AU]; scales the solar pressure by 1/d^2.
    co_rotating_atmosphere : bool
        If True the relative wind is ``v - omega_E x r``, i.e. the atmosphere rotates
        with the Earth. This reduces the relative speed by up to about 6 % at the
        equator on a prograde orbit and therefore the aerodynamic torque by up to about
        12 %. If False the inertial velocity is used.
    pressure_1au : float
        Solar radiation pressure at 1 AU [N m^-2].

    Returns
    -------
    TorqueProfile

    Notes
    -----
    Altitude for the density lookup is taken as ``orbit.radius_m - R_EARTH_EQUATORIAL``,
    i.e. spherical and constant over the circular orbit. Against a WGS-84 geodetic altitude this differs by up to 21 km at the
    poles, which at 500 km altitude changes the density by a factor of about
    ``exp(21/63.8) = 1.39``. That is a real and documented limitation of the sweep, not
    of the torque expressions.
    """
    if not isinstance(spacecraft, Spacecraft):
        raise TypeError(f"spacecraft must be a Spacecraft, got {type(spacecraft).__name__}")
    if not isinstance(orbit, Orbit):
        raise TypeError(f"orbit must be an Orbit, got {type(orbit).__name__}")
    n = int(n_samples)
    if n < 9:
        raise ValueError(f"n_samples must be >= 9, got {n_samples!r}")
    s_hat = _v.as_unit_vector(sun_unit_eci, "sun_unit_eci")
    d_au = _v.positive(distance_au, "distance_au")

    period = orbit.period_s
    u = np.linspace(0.0, 2.0 * np.pi, n)
    t = u / (2.0 * np.pi) * period
    r, v = circular_orbit_state(
        orbit.radius_m, orbit.inclination_rad, orbit.raan_rad, u, orbit.mu
    )

    c_bl = body_from_lvlh(orbit.yaw_rad, orbit.pitch_rad, orbit.roll_rad)
    nadir_body = c_bl @ np.array([0.0, 0.0, 1.0])

    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH]) if co_rotating_atmosphere else np.zeros(3)
    v_rel_eci = v - np.cross(omega_vec, r)

    b_eci = dipole_field_eci(r)
    lit = np.asarray(in_eclipse_cylindrical(r, s_hat), dtype=bool)
    illuminated = ~lit
    # The orbit is circular, so the spherical altitude is a single number; taking it from
    # orbit.radius_m rather than from the per-sample |r| avoids float noise straddling a
    # density-table band boundary, where the table itself steps by up to 5e-6 relative.
    altitude = orbit.radius_m - R_EARTH_EQUATORIAL
    rho = np.full(n, float(atmospheric_density(max(altitude, 0.0))))

    tb = {name: np.zeros((n, 3)) for name in SOURCES}
    te = {name: np.zeros((n, 3)) for name in SOURCES}

    gg_body = gravity_gradient_torque(
        spacecraft.inertia, nadir_body, orbit.radius_m, orbit.mu
    )

    for i in range(n):
        c_be = c_bl @ lvlh_from_eci(r[i], v[i])
        tb["gravity_gradient"][i] = gg_body
        tb["aerodynamic"][i] = aerodynamic_torque(
            float(rho[i]),
            c_be @ v_rel_eci[i],
            spacecraft.drag_coefficient,
            spacecraft.drag_area_m2,
            spacecraft.cp_aero_offset_m,
        )
        tb["solar"][i] = solar_radiation_torque(
            c_be @ s_hat,
            spacecraft.srp_area_m2,
            spacecraft.srp_reflectance,
            spacecraft.cp_srp_offset_m,
            d_au,
            bool(illuminated[i]),
            pressure_1au,
        )
        tb["magnetic"][i] = magnetic_torque(spacecraft.residual_dipole_am2, c_be @ b_eci[i])
        for name in SOURCES:
            te[name][i] = c_be.T @ tb[name][i]

    return TorqueProfile(
        u_rad=u,
        time_s=t,
        period_s=period,
        r_eci=r,
        v_eci=v,
        illuminated=illuminated,
        density_kg_m3=rho,
        b_field_eci_t=b_eci,
        torques_body=tb,
        torques_eci=te,
    )


def momentum_accumulation(
    profile: TorqueProfile, source: str = "total", frame: Frame = "eci"
) -> NDArray[np.float64]:
    """Cumulative time integral of the torque over one orbit [N m s], shape (N, 3).

    ``h(t) = int_0^t T dt'``, evaluated by the cumulative trapezoidal rule on the
    profile's own time grid, starting from zero.

    Frame caveat, stated because it is easy to get wrong: in the ``eci`` frame this is
    the change in the vehicle's inertial angular momentum caused by the external torque,
    which is what sizes a desaturation budget. In the ``body`` frame it is only the time
    integral of the body-frame components; because that frame rotates once per orbit it
    is *not* an inertial momentum change, and it is the quantity relevant to a wheel
    whose axes are fixed in the body.
    """
    t = profile.torque(source, frame)
    time = profile.time_s
    dt = np.diff(time)
    increments = 0.5 * (t[1:] + t[:-1]) * dt[:, None]
    out = np.zeros_like(t)
    out[1:] = np.cumsum(increments, axis=0)
    return out


def budget(
    profile: TorqueProfile, frame: Frame = "body"
) -> dict[str, dict[str, float | NDArray[np.float64]]]:
    """Per-source torque and momentum summary over one orbit.

    Returns a dict keyed by source name (plus ``"total"``) with, for each:

    ``peak_nm``
        Largest instantaneous torque magnitude [N m].
    ``rms_nm``
        RMS torque magnitude [N m].
    ``secular_nm``
        Orbit-averaged torque vector [N m], shape (3,).
    ``secular_magnitude_nm``
        Its magnitude [N m].
    ``cyclic_peak_nm``
        Peak magnitude of the torque with the secular part removed [N m].
    ``secular_momentum_per_orbit_nms``
        ``|secular| * period`` [N m s], the momentum that must be dumped each orbit.
    ``cyclic_momentum_peak_nms``
        Peak magnitude of the cyclic momentum excursion within the orbit [N m s], the
        momentum a wheel must be able to store without desaturating.
    """
    out: dict[str, dict[str, float | NDArray[np.float64]]] = {}
    for source in (*SOURCES, "total"):
        secular = profile.secular(source, frame)
        cyc = profile.cyclic(source, frame)
        dt = np.diff(profile.time_s)
        inc = 0.5 * (cyc[1:] + cyc[:-1]) * dt[:, None]
        h_cyc = np.zeros_like(cyc)
        h_cyc[1:] = np.cumsum(inc, axis=0)
        out[source] = {
            "peak_nm": profile.peak_magnitude(source, frame),
            "rms_nm": profile.rms_magnitude(source, frame),
            "secular_nm": secular,
            "secular_magnitude_nm": float(np.linalg.norm(secular)),
            "cyclic_peak_nm": profile.cyclic_peak(source, frame),
            "secular_momentum_per_orbit_nms": float(np.linalg.norm(secular)) * profile.period_s,
            "cyclic_momentum_peak_nms": float(np.max(np.linalg.norm(h_cyc, axis=1))),
        }
    return out
