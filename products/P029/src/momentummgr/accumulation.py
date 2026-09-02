"""Momentum accumulation over a circular orbit.

Two independent quadratures are provided on purpose.

``momentum_per_orbit_eci``
    Gauss-Legendre quadrature in argument of latitude, with the solar term integrated
    only over the analytically located sunlit arcs. The three continuous sources are
    low-order trigonometric polynomials in ``u``, so a modest Gauss-Legendre rule is
    exact to roundoff; the solar term is smooth *inside* each arc, so splitting at the
    closed-form eclipse boundaries removes the only discontinuity. This is the reference.
``momentum_history_eci``
    Cumulative trapezoid on a uniform sample grid, which is what a time-stepped
    simulation actually does, and what is needed to plot the momentum inside the orbit.

Reporting both is the point: the difference between them at a given grid size is the
discretisation error a scheduler inherits, and for the solar term it does not fall like
a power of the sample count, because the torque steps to and from zero at the eclipse
edges.

Frame caveat, stated because it is easy to get wrong. In ECI the time integral of the
external torque *is* the change in the vehicle's total inertial angular momentum, which
is what sizes a desaturation budget. In the body frame the same integral is not a
momentum change, because the frame rotates once per orbit; the body-frame wheel
momentum obeys the Euler equation in :mod:`momentummgr.wheels`, not a bare integral.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _validate as _v
from .constants import OMEGA_EARTH, R_EARTH_EQUATORIAL, SRP_PRESSURE_1AU
from .environment import (
    CircularOrbit,
    SpacecraftProperties,
    body_dcm_from_lvlh,
    density,
    dipole_field_eci,
    eclipse_boundaries,
    is_illuminated,
    node_axes,
)
from .torques import gravity_gradient_torque

__all__ = [
    "SOURCES",
    "OrbitSweep",
    "sweep_orbit",
    "momentum_per_orbit_eci",
    "momentum_history_eci",
    "secular_torque_eci",
    "momentum_budget",
]

SOURCES: tuple[str, ...] = ("gravity_gradient", "aerodynamic", "solar", "magnetic")


def _attitude_stack(
    orbit: CircularOrbit, u: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(r_eci, v_eci, c_be, c_bl)`` for a vector of arguments of latitude."""
    p_hat, q_hat, h_hat = node_axes(orbit.inclination_rad, orbit.raan_rad)
    cu, su = np.cos(u)[:, None], np.sin(u)[:, None]
    r_hat = cu * p_hat + su * q_hat
    v_hat = -su * p_hat + cu * q_hat
    r = orbit.radius_m * r_hat
    v = orbit.speed_ms * v_hat
    n = u.size
    c_le = np.empty((n, 3, 3))
    c_le[:, 0, :] = v_hat
    c_le[:, 1, :] = -h_hat
    c_le[:, 2, :] = -r_hat
    c_bl = body_dcm_from_lvlh(orbit.yaw_rad, orbit.pitch_rad, orbit.roll_rad)
    c_be = np.einsum("ij,njk->nik", c_bl, c_le)
    return r, v, c_be, c_bl


def _torques_body(
    spacecraft: SpacecraftProperties,
    orbit: CircularOrbit,
    sun_hat: NDArray[np.float64],
    u: NDArray[np.float64],
    co_rotating_atmosphere: bool = True,
    distance_au: float = 1.0,
    pressure_1au: float = SRP_PRESSURE_1AU,
    force_illuminated: bool | None = None,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64], NDArray[np.float64]]:
    """Body-frame torques at each ``u``; returns ``(torques, c_be, r, b_eci, lit)``."""
    r, v, c_be, c_bl = _attitude_stack(orbit, u)
    omega = np.array([0.0, 0.0, OMEGA_EARTH]) if co_rotating_atmosphere else np.zeros(3)
    v_rel_eci = v - np.cross(omega, r)
    b_eci = dipole_field_eci(r)
    lit = (
        np.ones(u.size, dtype=bool)
        if force_illuminated is True
        else np.asarray(is_illuminated(r, sun_hat), dtype=bool)
    )
    # Circular orbit: one spherical altitude for the whole sweep. Taking it from the
    # orbit radius rather than per-sample |r| keeps float noise from straddling a
    # density-band boundary, where the table itself steps by up to 1e-4 relative.
    rho = float(density(max(orbit.radius_m - R_EARTH_EQUATORIAL, 0.0)))

    nadir_body = c_bl @ np.array([0.0, 0.0, 1.0])
    gg = gravity_gradient_torque(spacecraft.inertia, nadir_body, orbit.radius_m, orbit.mu)

    v_rel_body = np.einsum("nij,nj->ni", c_be, v_rel_eci)
    speed = np.linalg.norm(v_rel_body, axis=1)
    f_aero = (
        -0.5
        * rho
        * spacecraft.drag_coefficient
        * spacecraft.drag_area_m2
        * speed[:, None]
        * v_rel_body
    )

    s_body = np.einsum("nij,j->ni", c_be, sun_hat)
    f_srp = (
        -(pressure_1au / distance_au**2)
        * spacecraft.srp_area_m2
        * (1.0 + spacecraft.srp_reflectance)
        * s_body
        * lit[:, None]
    )

    b_body = np.einsum("nij,nj->ni", c_be, b_eci)

    torques = {
        "gravity_gradient": np.repeat(gg[None, :], u.size, axis=0),
        "aerodynamic": np.cross(spacecraft.cp_aero_offset_m, f_aero),
        "solar": np.cross(spacecraft.cp_srp_offset_m, f_srp),
        "magnetic": np.cross(spacecraft.residual_dipole_am2, b_body),
    }
    return torques, c_be, r, b_eci, lit


@dataclass(frozen=True)
class OrbitSweep:
    """One orbit of environment and disturbance-torque history on a uniform grid.

    Attributes
    ----------
    u_rad, time_s : (N,)
        Argument of latitude [rad] and time from the ascending node [s], endpoints
        inclusive.
    period_s : float
        Orbital period [s].
    r_eci : (N, 3)
        Position [m].
    c_be : (N, 3, 3)
        DCM mapping ECI components into the body frame at each sample.
    b_eci_t : (N, 3)
        Geomagnetic field [T].
    illuminated : (N,) bool
        Cylindrical-shadow sunlight flag.
    torques_body, torques_eci : dict[str, (N, 3)]
        Per-source disturbance torque [N m].
    """

    u_rad: NDArray[np.float64]
    time_s: NDArray[np.float64]
    period_s: float
    r_eci: NDArray[np.float64]
    c_be: NDArray[np.float64]
    b_eci_t: NDArray[np.float64]
    illuminated: NDArray[np.bool_]
    torques_body: dict[str, NDArray[np.float64]]
    torques_eci: dict[str, NDArray[np.float64]]

    def torque(self, source: str = "total", frame: str = "body") -> NDArray[np.float64]:
        """Torque history [N m], shape ``(N, 3)``, for one source or their sum."""
        if frame == "body":
            store = self.torques_body
        elif frame == "eci":
            store = self.torques_eci
        else:
            raise ValueError(f"frame must be 'body' or 'eci', got {frame!r}")
        if source == "total":
            return np.sum(np.stack([store[s] for s in SOURCES]), axis=0)
        if source not in store:
            raise ValueError(f"source must be 'total' or one of {SOURCES}, got {source!r}")
        return store[source]

    @property
    def eclipse_fraction_sampled(self) -> float:
        """Fraction of the orbit flagged as in shadow by the sample grid, dimensionless."""
        return float(1.0 - np.mean(self.illuminated))

    def b_body_t(self) -> NDArray[np.float64]:
        """Geomagnetic field in body components [T], shape ``(N, 3)``."""
        return np.einsum("nij,nj->ni", self.c_be, self.b_eci_t)


def sweep_orbit(
    spacecraft: SpacecraftProperties,
    orbit: CircularOrbit,
    sun_hat: ArrayLike,
    n_samples: int = 721,
    co_rotating_atmosphere: bool = True,
    distance_au: float = 1.0,
    pressure_1au: float = SRP_PRESSURE_1AU,
) -> OrbitSweep:
    """Evaluate the environment and all four disturbance torques over one orbit.

    ``n_samples`` points spaced uniformly in argument of latitude, endpoints included,
    which for a circular orbit is uniform in time. Must be at least 9.
    """
    if not isinstance(spacecraft, SpacecraftProperties):
        raise TypeError(
            f"spacecraft must be a SpacecraftProperties, got {type(spacecraft).__name__}"
        )
    if not isinstance(orbit, CircularOrbit):
        raise TypeError(f"orbit must be a CircularOrbit, got {type(orbit).__name__}")
    n = _v.as_int_at_least(n_samples, "n_samples", 9)
    s_hat = _v.as_unit_vector(sun_hat, "sun_hat")
    d_au = _v.positive(distance_au, "distance_au")
    u = np.linspace(0.0, 2.0 * np.pi, n)
    period = orbit.period_s
    tb, c_be, r, b_eci, lit = _torques_body(
        spacecraft, orbit, s_hat, u, co_rotating_atmosphere, d_au, pressure_1au
    )
    te = {k: np.einsum("nji,nj->ni", c_be, v) for k, v in tb.items()}
    return OrbitSweep(
        u_rad=u,
        time_s=u / (2.0 * np.pi) * period,
        period_s=period,
        r_eci=r,
        c_be=c_be,
        b_eci_t=b_eci,
        illuminated=lit,
        torques_body=tb,
        torques_eci=te,
    )


def _gauss_legendre_eci(
    spacecraft: SpacecraftProperties,
    orbit: CircularOrbit,
    sun_hat: NDArray[np.float64],
    source: str,
    lo: float,
    hi: float,
    n_nodes: int,
    co_rotating_atmosphere: bool,
    distance_au: float,
    pressure_1au: float,
    force_illuminated: bool | None,
) -> NDArray[np.float64]:
    """Gauss-Legendre integral of one ECI torque component over ``u`` in ``[lo, hi]``."""
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    half = 0.5 * (hi - lo)
    u = 0.5 * (hi + lo) + half * x
    tb, c_be, _, _, _ = _torques_body(
        spacecraft,
        orbit,
        sun_hat,
        u,
        co_rotating_atmosphere,
        distance_au,
        pressure_1au,
        force_illuminated,
    )
    t_eci = np.einsum("nji,nj->ni", c_be, tb[source])
    return half * (w[None, :] @ t_eci)[0]


def momentum_per_orbit_eci(
    spacecraft: SpacecraftProperties,
    orbit: CircularOrbit,
    sun_hat: ArrayLike,
    source: str = "total",
    n_nodes: int = 96,
    co_rotating_atmosphere: bool = True,
    distance_au: float = 1.0,
    pressure_1au: float = SRP_PRESSURE_1AU,
) -> NDArray[np.float64]:
    r"""Inertial angular momentum accumulated over exactly one orbit [N m s], ECI.

    .. math:: \Delta \mathbf{h} = \int_0^P \mathbf{T}_{ECI}\, dt
              = \frac{P}{2\pi}\int_0^{2\pi} \mathbf{T}_{ECI}(u)\, du

    evaluated by an ``n_nodes``-point Gauss-Legendre rule. The gravity-gradient,
    aerodynamic and magnetic integrands are trigonometric polynomials of degree at most
    three in ``u``, so the rule is exact to roundoff for any ``n_nodes >= 4``; 96 is used
    because the same routine also serves the solar term. The solar integrand is
    discontinuous at the eclipse edges, so it is integrated only over the sunlit arcs,
    whose limits come from the closed form in
    :func:`momentummgr.environment.eclipse_boundaries` and never from a sample grid.

    Returns a shape-``(3,)`` vector in N m s. Use ``source='total'`` for the sum.
    """
    if source not in (*SOURCES, "total"):
        raise ValueError(f"source must be 'total' or one of {SOURCES}, got {source!r}")
    s_hat = _v.as_unit_vector(sun_hat, "sun_hat")
    n_nodes = _v.as_int_at_least(n_nodes, "n_nodes", 4)
    d_au = _v.positive(distance_au, "distance_au")
    scale = orbit.period_s / (2.0 * np.pi)

    def one(src: str) -> NDArray[np.float64]:
        if src != "solar":
            return scale * _gauss_legendre_eci(
                spacecraft, orbit, s_hat, src, 0.0, 2.0 * np.pi, n_nodes,
                co_rotating_atmosphere, d_au, pressure_1au, None,
            )
        bounds = eclipse_boundaries(
            orbit.radius_m, orbit.inclination_rad, orbit.raan_rad, s_hat
        )
        if bounds is None:
            arcs = [(0.0, 2.0 * np.pi)]
        else:
            u_in, u_out = bounds
            arcs = (
                [(u_out, u_in)] if u_out < u_in else [(u_out, 2.0 * np.pi), (0.0, u_in)]
            )
        total = np.zeros(3)
        for lo, hi in arcs:
            total += _gauss_legendre_eci(
                spacecraft, orbit, s_hat, "solar", lo, hi, n_nodes,
                co_rotating_atmosphere, d_au, pressure_1au, True,
            )
        return scale * total

    if source == "total":
        return sum((one(s) for s in SOURCES), start=np.zeros(3))
    return one(source)


def secular_torque_eci(
    spacecraft: SpacecraftProperties,
    orbit: CircularOrbit,
    sun_hat: ArrayLike,
    source: str = "total",
    **kwargs: object,
) -> NDArray[np.float64]:
    """Orbit-averaged ECI torque [N m], ``momentum_per_orbit_eci / period``."""
    return momentum_per_orbit_eci(spacecraft, orbit, sun_hat, source, **kwargs) / orbit.period_s  # type: ignore[arg-type]


def momentum_history_eci(sweep: OrbitSweep, source: str = "total") -> NDArray[np.float64]:
    """Cumulative trapezoidal momentum integral on the sweep grid [N m s], shape ``(N, 3)``.

    ``h(t) = int_0^t T_ECI dt'`` starting from zero. This is the sampled counterpart of
    :func:`momentum_per_orbit_eci`; comparing the two at ``t = P`` measures the
    discretisation error of the grid, which is what
    ``validation/p027_cross_check.py`` reports.
    """
    t = sweep.torque(source, "eci")
    dt = np.diff(sweep.time_s)
    inc = 0.5 * (t[1:] + t[:-1]) * dt[:, None]
    out = np.zeros_like(t)
    out[1:] = np.cumsum(inc, axis=0)
    return out


def momentum_budget(
    spacecraft: SpacecraftProperties,
    orbit: CircularOrbit,
    sun_hat: ArrayLike,
    n_samples: int = 721,
    **kwargs: object,
) -> dict[str, dict[str, float]]:
    """Per-source momentum summary over one orbit.

    For each source and for ``'total'``:

    ``secular_per_orbit_nms``
        ``|Gauss-Legendre dh over one orbit|`` [N m s] — the momentum that has to be
        dumped once per orbit.
    ``cyclic_peak_nms``
        Largest excursion of the momentum integral about the straight secular ramp
        within the orbit [N m s] — the momentum the wheels must be able to *store*
        without desaturating, which is a different and often larger number.
    ``peak_torque_nm``, ``rms_torque_nm``
        Instantaneous ECI torque magnitude statistics over the sample grid [N m].
    """
    sweep = sweep_orbit(spacecraft, orbit, sun_hat, n_samples, **kwargs)  # type: ignore[arg-type]
    out: dict[str, dict[str, float]] = {}
    for source in (*SOURCES, "total"):
        dh = momentum_per_orbit_eci(spacecraft, orbit, sun_hat, source, **kwargs)  # type: ignore[arg-type]
        hist = momentum_history_eci(sweep, source)
        ramp = np.outer(sweep.time_s / sweep.period_s, dh)
        t_eci = sweep.torque(source, "eci")
        out[source] = {
            "secular_per_orbit_nms": float(np.linalg.norm(dh)),
            "cyclic_peak_nms": float(np.max(np.linalg.norm(hist - ramp, axis=1))),
            "peak_torque_nm": float(np.max(np.linalg.norm(t_eci, axis=1))),
            "rms_torque_nm": float(np.sqrt(np.mean(np.sum(t_eci**2, axis=1)))),
        }
    return out
