"""Fixed-step Runge-Kutta propagation, vectorised over satellites.

One integrator, classical fourth-order Runge-Kutta, is implemented here. It is
vectorised across satellites rather than across time: a single call advances an
``(n, 6)`` state array, so the Python per-step overhead is paid once for the
whole population instead of once per object. That is what makes the dataset in
:mod:`orbitdrift.dataset` affordable on two cores, and the measured cost is
recorded in ``validation/validate_step_refinement.py``.

Why a fixed step and not an adaptive one. The product's output is a divergence
*curve*, and a curve assembled from an adaptive integrator's internal steps
has a numerical error that varies along it in a way the curve's own shape then
inherits. A fixed step makes the truncation error a declared, measurable
constant of the run, and the step-refinement study measures it. The price is
that the step must be chosen for the shortest period in the population, which
is wasteful for the high orbits; that cost is real and is recorded in
README.md under Limitations.

The accuracy claim this module makes is exactly this and nothing more: the
measured convergence order under step halving, and the measured difference
against the closed-form two-body solution of :mod:`orbitdrift.kepler`, both
published with numbers in ``validation/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import SECONDS_PER_DAY, WGS72, GravityModel
from .errors import PropagationError
from .forces import ForceModel, zonal_potential
from .frames import moon_position_teme_km, sun_position_teme_km

EPHEMERIS_GRID_S = 60.0
"""Spacing of the tabulated luni-solar ephemeris, seconds.

The Sun and Moon positions needed by the third-body term are tabulated once
per call on a grid of this spacing and linearly interpolated at each
Runge-Kutta stage, rather than evaluated from the trigonometric series four
times per step. The series in :mod:`orbitdrift.frames` are numpy expressions
over about thirty scalar terms, and at a step of order ten seconds they
dominated the whole integrator: tabulating them is what brings the dataset in
:mod:`orbitdrift.dataset` inside the compute budget, and the measured
speed-up is reported in ``validation/validate_step_refinement.py``.

The cost is an interpolation error. The Moon moves at 1.53e-4 degree per
second, so over a 60 s interval the chord departs from the arc by
``r omega^2 dt^2 / 8``, about 1.2e-3 km out of 384400 km, a relative
direction error of 3e-9. Against a lunar third-body acceleration of order
1e-9 km/s^2 that is 3e-18 km/s^2. ``tests/test_integrate.py`` measures the
position difference between a tabulated run and a run that evaluates the
series directly, and requires it to be far below the truncation floor.
"""


class _BodyEphemeris:
    """Tabulated Sun and Moon positions with linear interpolation in time."""

    def __init__(self, jd0: float, span_s: float, grid_s: float = EPHEMERIS_GRID_S) -> None:
        n = max(2, int(np.ceil(span_s / grid_s)) + 2)
        self._jd0 = float(jd0)
        self._grid_days = float(grid_s) / SECONDS_PER_DAY
        offsets = np.arange(n) * self._grid_days
        jd = self._jd0 + offsets
        self._sun = sun_position_teme_km(jd)
        self._moon = moon_position_teme_km(jd)
        self._n = n

    def at(self, jd: float) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated (sun, moon) geocentric positions in km at ``jd``."""
        x = (jd - self._jd0) / self._grid_days
        i = int(x)
        if i < 0:
            i = 0
        elif i > self._n - 2:
            i = self._n - 2
        w = x - i
        sun = self._sun[i] + w * (self._sun[i + 1] - self._sun[i])
        moon = self._moon[i] + w * (self._moon[i + 1] - self._moon[i])
        return sun, moon


@dataclass(frozen=True)
class Trajectory:
    """Sampled states of ``n`` objects at ``m`` times.

    Attributes
    ----------
    times_s
        Seconds from the common initial epoch, shape ``(m,)``.
    positions_km
        Shape ``(n, m, 3)``.
    velocities_km_s
        Shape ``(n, m, 3)``.
    step_s
        The integration step actually used.
    force_label
        :meth:`orbitdrift.forces.ForceModel.label` of the model used.
    """

    times_s: np.ndarray
    positions_km: np.ndarray
    velocities_km_s: np.ndarray
    step_s: float
    force_label: str

    @property
    def n_objects(self) -> int:
        return int(self.positions_km.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.positions_km.shape[1])


def _derivative(
    state: np.ndarray,
    jd: float,
    force: ForceModel,
    cd_area_over_mass: np.ndarray,
    ephemeris: _BodyEphemeris | None,
) -> np.ndarray:
    accel = force.acceleration_km_s2(
        state[:, :3],
        state[:, 3:],
        jd,
        cd_area_over_mass,
        bodies=ephemeris.at(jd) if ephemeris else None,
    )
    return np.concatenate([state[:, 3:], accel], axis=1)


def propagate(
    r0_km: np.ndarray,
    v0_km_s: np.ndarray,
    jd0: float,
    cd_area_over_mass_m2_kg: np.ndarray,
    *,
    n_steps: int,
    step_s: float,
    force: ForceModel | None = None,
    sample_every: int = 1,
) -> Trajectory:
    """Propagate ``n`` objects with classical RK4 and return sampled states.

    Parameters
    ----------
    r0_km, v0_km_s
        Initial states, shape ``(n, 3)``, in TEME.
    jd0
        Julian date of the epoch **shared by every object in the call**, as a
        scalar. The drag and third-body terms are functions of absolute time,
        so objects with different epochs belong in different calls. Making the
        epoch scalar rather than per-object is what lets the luni-solar series
        in :mod:`orbitdrift.frames` be evaluated once per Runge-Kutta stage
        for the whole batch instead of once per object: for a batch of sixty
        objects that is the difference between an affordable dataset and an
        unaffordable one, and the cost is measured in
        ``validation/validate_step_refinement.py``.
    cd_area_over_mass_m2_kg
        ``C_D A / m`` per object, shape ``(n,)``.
    n_steps
        Number of fixed steps to take. The span is ``n_steps * step_s``. The
        step count is given rather than a duration so that no call can land on
        a fractional step and have it silently rounded: the caller states the
        grid it wants and the sampling must divide it.
    step_s
        Fixed step, seconds.
    force
        Force model. Defaults to :class:`orbitdrift.forces.ForceModel` with
        every term on.
    sample_every
        Record the state every this many steps; must divide ``n_steps``, so
        that the returned sample times are exactly evenly spaced.

    Raises
    ------
    PropagationError
        On a non-finite state, a state inside the Earth, a non-positive step
        count, or ``sample_every`` not dividing the step count.
    """
    force = force or ForceModel()
    r0 = np.atleast_2d(np.asarray(r0_km, dtype=float))
    v0 = np.atleast_2d(np.asarray(v0_km_s, dtype=float))
    jd_epoch = float(jd0)
    coefficient = np.atleast_1d(np.asarray(cd_area_over_mass_m2_kg, dtype=float))
    if r0.shape != v0.shape or r0.shape[1] != 3:
        raise PropagationError(f"r0 and v0 must both be (n, 3); got {r0.shape}, {v0.shape}")
    if coefficient.shape[0] != r0.shape[0]:
        raise PropagationError(
            f"cd_area_over_mass must have length {r0.shape[0]}; "
            f"got {coefficient.shape[0]}"
        )
    if step_s <= 0.0:
        raise PropagationError(f"step_s must be positive; got {step_s}")
    n_steps = int(n_steps)
    if n_steps < 1:
        raise PropagationError(f"n_steps must be at least 1; got {n_steps}")
    if sample_every < 1 or n_steps % sample_every != 0:
        raise PropagationError(
            f"sample_every={sample_every} must be >= 1 and divide the {n_steps} steps"
        )

    ephemeris = (
        _BodyEphemeris(jd_epoch, n_steps * float(step_s) + 2.0 * float(step_s))
        if force.third_body
        else None
    )
    state = np.concatenate([r0, v0], axis=1)
    n_samples = n_steps // sample_every + 1
    positions = np.empty((r0.shape[0], n_samples, 3))
    velocities = np.empty((r0.shape[0], n_samples, 3))
    times = np.empty(n_samples)
    positions[:, 0] = r0
    velocities[:, 0] = v0
    times[0] = 0.0

    h = float(step_s)
    half = 0.5 * h
    day = h / SECONDS_PER_DAY
    for index in range(1, n_steps + 1):
        t0 = (index - 1) * h
        jd_a = jd_epoch + t0 / SECONDS_PER_DAY
        jd_b = jd_a + 0.5 * day
        jd_c = jd_a + day
        k1 = _derivative(state, jd_a, force, coefficient, ephemeris)
        k2 = _derivative(state + half * k1, jd_b, force, coefficient, ephemeris)
        k3 = _derivative(state + half * k2, jd_b, force, coefficient, ephemeris)
        k4 = _derivative(state + h * k3, jd_c, force, coefficient, ephemeris)
        state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(state)):
            raise PropagationError(f"state became non-finite at step {index} of {n_steps}")
        if index % sample_every == 0:
            slot = index // sample_every
            positions[:, slot] = state[:, :3]
            velocities[:, slot] = state[:, 3:]
            times[slot] = index * h

    radius = np.linalg.norm(positions, axis=-1)
    if np.any(radius < force.gravity.radius_km):
        worst = float(np.min(radius))
        raise PropagationError(
            f"propagated radius {worst:.3f} km is below the Earth radius "
            f"{force.gravity.radius_km:.3f} km; the object re-entered within the span"
        )
    return Trajectory(times, positions, velocities, h, force.label())


def specific_energy_km2_s2(
    r_km: np.ndarray, v_km_s: np.ndarray, gravity: GravityModel = WGS72, *, zonal: bool = True
) -> np.ndarray:
    """Specific mechanical energy ``v^2/2 + V(r)``, km^2/s^2.

    With ``zonal=True`` the potential is the full zonal one from
    :func:`orbitdrift.forces.zonal_potential`; with ``zonal=False`` it is
    ``-mu/r``. Passing the wrong one is the single easiest way to measure a
    spurious energy drift, because the ``J2`` term is 1e-3 of the two-body
    term and a conservation residual of 1e-9 would be swamped by it, so the
    flag is explicit and has no default that could be silently wrong for the
    force model in use.
    """
    v = np.asarray(v_km_s, dtype=float)
    kinetic = 0.5 * np.sum(v * v, axis=-1)
    if zonal:
        return kinetic + zonal_potential(r_km, gravity)
    r = np.asarray(r_km, dtype=float)
    return kinetic - gravity.mu_km3_s2 / np.linalg.norm(r, axis=-1)


def angular_momentum_km2_s(r_km: np.ndarray, v_km_s: np.ndarray) -> np.ndarray:
    """Specific angular momentum vector ``r x v``, km^2/s, shape ``(..., 3)``."""
    return np.cross(np.asarray(r_km, dtype=float), np.asarray(v_km_s, dtype=float))


def invariant_residuals(trajectory: Trajectory, *, zonal: bool) -> dict[str, np.ndarray]:
    """Relative drift of energy, ``|h|`` and ``h_z`` along a trajectory.

    Returns, for each object, the maximum over the sampled times of
    ``|q(t) - q(0)| / |q(0)|`` for each of the three quantities.

    The three behave differently and the difference is physics, not numerics:

    - ``energy`` and ``h_z`` are **exact** constants of motion of the two-body
      plus zonal field, because that potential is time-independent and
      axially symmetric about z.
    - ``|h|`` is **not** a constant of that field. ``J2`` exerts a torque
      about any axis in the equatorial plane, so the angular momentum vector
      precesses about z at constant ``h_z`` while its magnitude oscillates.
      A measured ``|h|`` drift is therefore the expected answer and its
      magnitude is a property of the orbit, not of the integrator.

    ``validation/validate_conservation.py`` reports all three and says which
    of them is a numerical tolerance and which is a physical amplitude.
    """
    energy = specific_energy_km2_s2(
        trajectory.positions_km, trajectory.velocities_km_s, zonal=zonal
    )
    h_vector = angular_momentum_km2_s(trajectory.positions_km, trajectory.velocities_km_s)
    h_magnitude = np.linalg.norm(h_vector, axis=-1)
    h_z = h_vector[..., 2]
    out: dict[str, np.ndarray] = {}
    for name, series in (("energy", energy), ("h_magnitude", h_magnitude), ("h_z", h_z)):
        initial = series[:, :1]
        out[name] = np.max(np.abs(series - initial) / np.abs(initial), axis=1)
    return out
