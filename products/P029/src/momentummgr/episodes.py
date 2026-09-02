"""Desaturation episodes: the scheduling problem, its simulator and its cost.

An *episode* is one vehicle on one orbit for a few orbits, discretised into equal
decision windows. In each window the scheduler decides whether to run the magnetorquers.
Running them costs magnetorquer duty; not running them lets the wheels fill up. The two
metrics reported everywhere in this package are exactly that trade:

``duty_fraction``
    ``int |m| dt / (m_max T)``, dimensionless, the magnetorquer's share of the episode.
    Proportional to :func:`momentummgr.desaturation.dipole_cost`.
``near_saturation_fraction``
    Fraction of the episode with ``|h_wheel_body| > 0.8 h_env``, dimensionless, where
    ``h_env`` is the conservative array envelope of
    :attr:`momentummgr.wheels.WheelArray.guaranteed_body_envelope_nms`.

Wheel dynamics
--------------
With the attitude held in LVLH the total angular momentum is ``H = I omega + h_w`` and
the body-frame Euler equation gives

.. math:: \\dot{\\mathbf{h}}_w = \\mathbf{T}_{dist} + \\mathbf{T}_{mag}
          - \\boldsymbol{\\omega}\\times(\\mathbf{I}\\boldsymbol{\\omega} + \\mathbf{h}_w)

(Wertz, *Spacecraft Attitude Determination and Control*; Sidi, *Spacecraft Dynamics and
Control*; Markley and Crassidis, *Fundamentals of Spacecraft Attitude Determination and
Control*). The constant part ``-omega x (I omega)`` is the gyroscopic term of a
nadir-pointing vehicle: it is of the same order as the gravity-gradient torque and it is
kept, not dropped. Integration is Heun's method (explicit trapezoidal, second order) on a fixed
substep. Plain Euler was tried first and rejected: at the default five substeps per
600 s window it overstated the magnetorquer duty of the baseline scheduler by 49 %
against a converged reference, because inside an active window the dipole command
shrinks as the momentum is removed and a first-order step does not see that. The step
sensitivity of the second-order scheme is measured in
``validation/learned_vs_fixed_ci.py``, not assumed.

Everything in the episode is **simulated**. No flight telemetry is used anywhere in this
package. See ``DATASET_CARD.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import _validate as _v
from .accumulation import _torques_body
from .constants import OMEGA_EARTH
from .environment import (
    CircularOrbit,
    SpacecraftProperties,
    dipole_field_eci,
    sun_direction_for_beta,
)
from .wheels import WheelArray, pyramid_four

__all__ = [
    "Episode",
    "EpisodeMetrics",
    "Rollout",
    "N_FEATURES",
    "FEATURE_NAMES",
    "DEFAULT_DIPOLE_TILT_RAD",
    "sample_episode",
    "build_episode",
    "simulate_masks",
    "rollout",
    "episode_cost",
]

N_FEATURES: int = 11
FEATURE_NAMES: tuple[str, ...] = (
    "h_fraction",                 # |h| / h_env, from wheel tachometers
    "sin_theta_now",              # |h x B_hat| / |h|, dumpable fraction now
    "b_norm_now",                 # |B| / 3e-5 T, from the magnetometer
    "dh_last_window",             # |h(k) - h(k-1)| / h_env, measured growth
    "sin_theta_next",             # same as 1, one window ahead, from the onboard model
    "b_norm_next",                # same as 2, one window ahead
    "best_merit_next_3",          # max over the next 3 windows of |B| sin(theta) / 3e-5
    "merit_now",                  # |B| sin(theta) / 3e-5 at this window
    "windows_since_dump",         # capped at 20 and scaled to [0, 1]
    "coast_h_fraction_3",         # |h| / h_env predicted 3 windows ahead with no dumping
    "coast_h_fraction_6",         # |h| / h_env predicted 6 windows ahead with no dumping
)
"""The eleven scheduler features. Every one is computable onboard from wheel
tachometers, a magnetometer, an onboard field model and an orbit propagator, plus — for
the last two only — a modelled disturbance torque, which any momentum-management system
already carries in order to size its own wheels.

The two ``coast_h_fraction`` features propagate the wheel equation forward with the
magnetorquers off, one Euler step per window, and report the resulting momentum
fraction. They are the causal substitute for what the offline label search gets for
free: knowing how much margin is left before the wheels fill. They are the two most
important features the model uses.

Nothing here uses the future wheel state under the policy's own future actions, the
realised disturbance, or any other quantity a spacecraft would not have at the moment of
the decision."""

DEFAULT_DIPOLE_TILT_RAD: float = float(np.radians(9.4))
"""Geomagnetic dipole tilt used by the episode generator [rad]; see
:func:`momentummgr.environment.dipole_field_eci`. Representative of recent IGRF epochs,
configurable, and deliberately non-zero so the field geometry does not repeat exactly
each orbit."""

_B_SCALE_T: float = 3.0e-5
"""Field magnitude used to non-dimensionalise the magnetometer features [T]; roughly the
equatorial surface field of the centred dipole. A scaling constant, not a physical claim."""


@dataclass(frozen=True)
class Episode:
    """A precomputed desaturation scheduling problem.

    Attributes
    ----------
    seed : int
        Generator seed; the episode is fully determined by it.
    orbit, spacecraft, wheels
        The vehicle and its orbit.
    max_dipole_am2 : float
        Magnetorquer dipole limit [A m^2].
    window_s : float
        Decision window length [s].
    n_windows : int
        Number of decision windows.
    substeps : int
        Integration substeps per window.
    torque_body_nm : (K, 3)
        Disturbance torque at each window midpoint, body frame [N m], with the constant
        gyroscopic term ``-omega x (I omega)`` already added.
    b_body_t : (K, 3)
        Geomagnetic field at each window midpoint, body frame [T].
    b_norm_t : (K,)
        Field magnitude at each window midpoint [T]; cached because the feature code is
        called once per window in every rollout.
    omega_body_rad_s : (3,)
        Body rate w.r.t. inertial [rad s^-1].
    initial_momentum_nms : (3,)
        Wheel momentum at the start of the episode, body frame [N m s].
    gain : float
        Cross-product law gain [s^-1]; see :func:`build_episode`.
    envelope_nms : float
        Conservative body-momentum envelope of the wheel array [N m s].
    """

    seed: int
    orbit: CircularOrbit
    spacecraft: SpacecraftProperties
    wheels: WheelArray
    max_dipole_am2: float
    window_s: float
    n_windows: int
    substeps: int
    torque_body_nm: NDArray[np.float64]
    b_body_t: NDArray[np.float64]
    b_norm_t: NDArray[np.float64]
    omega_body_rad_s: NDArray[np.float64]
    initial_momentum_nms: NDArray[np.float64]
    gain: float
    envelope_nms: float

    @property
    def duration_s(self) -> float:
        """Episode length [s]."""
        return self.window_s * self.n_windows


@dataclass(frozen=True)
class EpisodeMetrics:
    """Outcome of one simulated episode.

    ``duty_fraction`` and ``near_saturation_fraction`` are the two reported metrics;
    ``cost`` is the scalar the offline search minimises and is defined in
    :func:`episode_cost`.
    """

    dipole_cost_am2s: float
    duty_fraction: float
    near_saturation_fraction: float
    max_h_fraction: float
    violated: bool
    cost: float


@dataclass(frozen=True)
class Rollout:
    """A single simulated episode with its per-window record.

    Attributes
    ----------
    actions : (K,) bool
        Whether the magnetorquers ran in each window, after the safety override.
    confidences : (K,)
        Policy confidence in each action, in [0, 1]; 1.0 wherever the safety override
        fired or the policy does not produce a confidence.
    features : (K, N_FEATURES)
        Feature vector seen at the start of each window.
    h_history_nms : (K * substeps + 1, 3)
        Wheel momentum, body frame [N m s].
    dipole_history_am2 : (K * substeps, 3)
        Commanded dipole [A m^2].
    time_s : (K * substeps + 1,)
        Sample times [s].
    metrics : EpisodeMetrics
    """

    actions: NDArray[np.bool_]
    confidences: NDArray[np.float64]
    features: NDArray[np.float64]
    h_history_nms: NDArray[np.float64]
    dipole_history_am2: NDArray[np.float64]
    time_s: NDArray[np.float64]
    metrics: EpisodeMetrics


NEAR_SATURATION_FRACTION: float = 0.8
"""``|h| / h_env`` above which the array counts as near saturation, dimensionless."""

SAFETY_OVERRIDE_FRACTION: float = 0.95
"""``|h| / h_env`` above which every policy in this package dumps regardless of what it
would otherwise have decided. Applied identically to the baseline and to the learned
scheduler so the comparison is of scheduling, not of who has a safety net."""


def episode_cost(
    duty_fraction: float,
    near_saturation_fraction: float,
    max_h_fraction: float,
    saturation_weight: float = 1.0,
    violation_weight: float = 10.0,
) -> float:
    """Scalar objective for the offline search and for tuning the baseline.

    ``J = duty_fraction + saturation_weight * near_saturation_fraction
    + violation_weight * max(0, max_h_fraction - 1)``

    Both leading terms are already dimensionless fractions of the episode, so the weight
    of 1.0 states that one second near saturation is worth one second of magnetorquer
    duty. That is a choice, not a derivation; it is exposed as an argument and its effect
    is swept in ``validation/scheduler_benchmark.py``.
    """
    return float(
        duty_fraction
        + saturation_weight * near_saturation_fraction
        + violation_weight * max(0.0, max_h_fraction - 1.0)
    )


def build_episode(
    seed: int,
    orbit: CircularOrbit,
    spacecraft: SpacecraftProperties,
    wheels: WheelArray,
    sun_hat: NDArray[np.float64],
    max_dipole_am2: float,
    initial_momentum_nms: NDArray[np.float64],
    n_orbits: float = 6.0,
    window_s: float = 600.0,
    substeps: int = 5,
    dipole_tilt_rad: float = DEFAULT_DIPOLE_TILT_RAD,
) -> Episode:
    """Precompute the disturbance torque and field history of one scheduling episode.

    The gain of the cross-product law is set so that the commanded dipole reaches its
    limit when a quarter of the envelope is available perpendicular to the field:
    ``gain = m_max * <|B|> / (0.25 h_env)`` [s^-1]. Above that the command saturates in
    magnitude and only its direction matters, which is the regime a smallsat
    magnetorquer normally works in.
    """
    n_orb = _v.positive(n_orbits, "n_orbits")
    win = _v.positive(window_s, "window_s")
    sub = _v.as_int_at_least(substeps, "substeps", 1)
    m_max = _v.positive(max_dipole_am2, "max_dipole_am2")
    period = orbit.period_s
    n_windows = max(int(round(n_orb * period / win)), 4)
    t_mid = (np.arange(n_windows) + 0.5) * win
    u = 2.0 * np.pi * t_mid / period

    torques, c_be, r, _, _ = _torques_body(spacecraft, orbit, sun_hat, u)
    t_body = sum(torques.values())
    b_eci = dipole_field_eci(r, tilt_rad=dipole_tilt_rad, rotation_angle_rad=OMEGA_EARTH * t_mid)
    b_body = np.einsum("nij,nj->ni", c_be, b_eci)

    omega = orbit.body_rate_body_rad_s
    gyro = -np.cross(omega, spacecraft.inertia @ omega)
    envelope = wheels.guaranteed_body_envelope_nms
    gain = m_max * float(np.mean(np.linalg.norm(b_body, axis=1))) / (0.25 * envelope)
    return Episode(
        seed=int(seed),
        orbit=orbit,
        spacecraft=spacecraft,
        wheels=wheels,
        max_dipole_am2=m_max,
        window_s=win,
        n_windows=n_windows,
        substeps=sub,
        torque_body_nm=t_body + gyro,
        b_body_t=b_body,
        b_norm_t=np.linalg.norm(b_body, axis=1),
        omega_body_rad_s=omega,
        initial_momentum_nms=np.asarray(initial_momentum_nms, dtype=float),
        gain=gain,
        envelope_nms=envelope,
    )


def sample_episode(
    seed: int,
    n_orbits: float = 6.0,
    window_s: float = 600.0,
    substeps: int = 5,
    require_feasible: bool = True,
    max_attempts: int = 12,
) -> Episode:
    """Draw one episode from the documented parameter distributions.

    The ranges are listed in ``DATASET_CARD.md`` and are chosen so that a smallsat-class
    wheel set fills within a few orbits: badly balanced vehicles, small wheels and modest
    magnetorquers. They are **not** a survey of flown spacecraft; they are a sampling
    envelope. Deterministic in ``seed``.

    With ``require_feasible`` a draw is rejected when even running the magnetorquers in
    every window fails to keep the wheels inside their envelope, and the next draw for the
    same seed is taken from ``default_rng([seed, attempt])``. Such vehicles exist and are
    a real design outcome, but they are a *sizing* failure, not a scheduling problem, and
    including them would compare two schedulers on episodes neither can win. The
    rejection rate is measured and reported in ``DATASET_CARD.md``; set the flag to False
    to sample the unfiltered distribution. Raises ``RuntimeError`` if ``max_attempts``
    draws are all infeasible.
    """
    attempts = _v.as_int_at_least(max_attempts, "max_attempts", 1)
    for attempt in range(attempts):
        rng = np.random.default_rng([int(seed), attempt])
        episode = _draw_episode(rng, seed, n_orbits, window_s, substeps)
        if not require_feasible:
            return episode
        always = np.ones((1, episode.n_windows), dtype=bool)
        if not simulate_masks(episode, always)[0].violated:
            return episode
    raise RuntimeError(
        f"no feasible episode found for seed {seed} in {attempts} attempts; every draw "
        "saturated the wheels even with the magnetorquers on in every window"
    )


def _draw_episode(
    rng: np.random.Generator,
    seed: int,
    n_orbits: float,
    window_s: float,
    substeps: int,
) -> Episode:
    """One unconditioned draw from the episode distribution."""
    altitude_km = float(rng.uniform(400.0, 650.0))
    inclination = float(np.radians(rng.uniform(30.0, 98.0)))
    raan = float(rng.uniform(0.0, 2.0 * np.pi))
    orbit = CircularOrbit(
        altitude_m=altitude_km * 1000.0,
        inclination_rad=inclination,
        raan_rad=raan,
        yaw_rad=float(np.radians(rng.uniform(-10.0, 10.0))),
        pitch_rad=float(np.radians(rng.uniform(-12.0, 12.0))),
        roll_rad=float(np.radians(rng.uniform(-12.0, 12.0))),
    )
    moments = np.sort(rng.uniform(2.0, 14.0, size=3))
    if moments[0] + moments[1] < moments[2]:
        moments[2] = moments[0] + moments[1]
    spacecraft = SpacecraftProperties(
        inertia=np.diag(rng.permutation(moments)),
        drag_area_m2=float(rng.uniform(0.3, 1.5)),
        drag_coefficient=2.2,
        cp_aero_offset_m=rng.uniform(-0.06, 0.06, size=3),
        srp_area_m2=float(rng.uniform(0.5, 2.5)),
        srp_reflectance=float(rng.uniform(0.2, 0.9)),
        cp_srp_offset_m=rng.uniform(-0.06, 0.06, size=3),
        residual_dipole_am2=rng.uniform(-0.35, 0.35, size=3),
        mass_kg=float(rng.uniform(50.0, 180.0)),
    )
    wheels = pyramid_four(
        wheel_inertia_kg_m2=float(rng.uniform(5e-4, 2e-3)),
        max_momentum_nms=float(rng.uniform(0.02, 0.08)),
    )
    sun_hat = sun_direction_for_beta(
        inclination,
        raan,
        float(np.radians(rng.uniform(-70.0, 70.0))),
        float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    envelope = wheels.guaranteed_body_envelope_nms
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    h0 = direction * envelope * float(rng.uniform(0.0, 0.35))
    return build_episode(
        seed=seed,
        orbit=orbit,
        spacecraft=spacecraft,
        wheels=wheels,
        sun_hat=sun_hat,
        max_dipole_am2=float(rng.uniform(0.5, 3.0)),
        initial_momentum_nms=h0,
        n_orbits=n_orbits,
        window_s=window_s,
        substeps=substeps,
    )


def _limited_dipole(
    h: NDArray[np.float64], b: NDArray[np.float64], gain: float, m_max: float
) -> NDArray[np.float64]:
    """Cross-product dipole command for a batch of momentum states, magnitude-limited."""
    b_sq = float(b @ b)
    m = -(gain / b_sq) * np.cross(b, h)
    norm = np.linalg.norm(m, axis=-1, keepdims=True)
    scale = np.where(norm > m_max, m_max / np.maximum(norm, 1e-300), 1.0)
    return m * scale


def simulate_masks(episode: Episode, masks: NDArray[np.bool_]) -> list[EpisodeMetrics]:
    """Simulate many fixed on/off schedules at once.

    ``masks`` has shape ``(M, n_windows)``; entry ``(i, k)`` says whether schedule ``i``
    runs the magnetorquers in window ``k``. The safety override of
    :data:`SAFETY_OVERRIDE_FRACTION` is applied to every schedule, so a mask that would
    otherwise saturate still gets the same protection a policy would. Returns one
    :class:`EpisodeMetrics` per row. Vectorised over ``M``; this is what makes the
    offline label search affordable.
    """
    m = np.asarray(masks, dtype=bool)
    if m.ndim != 2 or m.shape[1] != episode.n_windows:
        raise ValueError(
            f"masks must have shape (M, {episode.n_windows}), got {m.shape}"
        )
    n_masks = m.shape[0]
    dt = episode.window_s / episode.substeps
    h = np.repeat(episode.initial_momentum_nms[None, :], n_masks, axis=0)
    omega = episode.omega_body_rad_s
    env = episode.envelope_nms
    cost = np.zeros(n_masks)
    near = np.zeros(n_masks)
    peak = np.linalg.norm(h, axis=1) / env
    n_steps = episode.n_windows * episode.substeps
    for k in range(episode.n_windows):
        td = episode.torque_body_nm[k]
        b = episode.b_body_t[k]
        for _ in range(episode.substeps):
            frac = np.linalg.norm(h, axis=1) / env
            active = m[:, k] | (frac >= SAFETY_OVERRIDE_FRACTION)
            dip1 = _limited_dipole(h, b, episode.gain, episode.max_dipole_am2) * active[:, None]
            k1 = td + np.cross(dip1, b) - np.cross(omega, h)
            h_pred = h + dt * k1
            dip2 = (
                _limited_dipole(h_pred, b, episode.gain, episode.max_dipole_am2)
                * active[:, None]
            )
            k2 = td + np.cross(dip2, b) - np.cross(omega, h_pred)
            cost += 0.5 * (np.linalg.norm(dip1, axis=1) + np.linalg.norm(dip2, axis=1)) * dt
            h = h + 0.5 * dt * (k1 + k2)
            frac = np.linalg.norm(h, axis=1) / env
            near += (frac > NEAR_SATURATION_FRACTION).astype(float)
            peak = np.maximum(peak, frac)
    duty = cost / (episode.max_dipole_am2 * episode.duration_s)
    near_frac = near / n_steps
    return [
        EpisodeMetrics(
            dipole_cost_am2s=float(cost[i]),
            duty_fraction=float(duty[i]),
            near_saturation_fraction=float(near_frac[i]),
            max_h_fraction=float(peak[i]),
            violated=bool(peak[i] > 1.0),
            cost=episode_cost(float(duty[i]), float(near_frac[i]), float(peak[i])),
        )
        for i in range(n_masks)
    ]


def _features(
    episode: Episode,
    k: int,
    hx: float,
    hy: float,
    hz: float,
    dh_last: float,
    windows_since_dump: int,
) -> NDArray[np.float64]:
    """Feature vector at the start of window ``k``; see :data:`FEATURE_NAMES`.

    Written in scalar arithmetic for the same reason as the integrator in
    :func:`rollout`: it runs once per window in every one of thousands of rollouts.
    """
    env = episode.envelope_nms
    h_norm = np.sqrt(hx * hx + hy * hy + hz * hz)
    last = episode.n_windows - 1
    b_all = episode.b_body_t
    b_norms = episode.b_norm_t

    def merit(idx: int) -> tuple[float, float]:
        i = idx if idx < last else last
        b_norm = float(b_norms[i])
        if h_norm == 0.0 or b_norm == 0.0:
            return 0.0, b_norm
        bx, by, bz = b_all[i, 0], b_all[i, 1], b_all[i, 2]
        cx = hy * bz - hz * by
        cy = hz * bx - hx * bz
        cz = hx * by - hy * bx
        cross = np.sqrt(cx * cx + cy * cy + cz * cz)
        return float(cross / (h_norm * b_norm)), b_norm

    sin_now, b_now = merit(k)
    sin_next, b_next = merit(k + 1)
    best = 0.0
    for j in (1, 2, 3):
        sin_j, b_j = merit(k + j)
        best = max(best, sin_j * b_j)

    # Coast prediction: propagate h forward with the magnetorquers off, one Euler step
    # per window, and record the momentum fraction after 3 and after 6 windows.
    ox, oy, oz = (float(v) for v in episode.omega_body_rad_s)
    w = episode.window_s
    cx, cy, cz = hx, hy, hz
    coast3 = coast6 = 0.0
    for j in range(6):
        td = episode.torque_body_nm[k + j if k + j < last else last]
        gx = oy * cz - oz * cy
        gy = oz * cx - ox * cz
        gz = ox * cy - oy * cx
        cx += w * (td[0] - gx)
        cy += w * (td[1] - gy)
        cz += w * (td[2] - gz)
        if j == 2:
            coast3 = np.sqrt(cx * cx + cy * cy + cz * cz) / env
        elif j == 5:
            coast6 = np.sqrt(cx * cx + cy * cy + cz * cz) / env

    return np.array(
        [
            h_norm / env,
            sin_now,
            b_now / _B_SCALE_T,
            dh_last / env,
            sin_next,
            b_next / _B_SCALE_T,
            best / _B_SCALE_T,
            sin_now * b_now / _B_SCALE_T,
            min(windows_since_dump, 20) / 20.0,
            coast3,
            coast6,
        ]
    )


def _scalar_dipole(
    hx: float, hy: float, hz: float,
    bx: float, by: float, bz: float,
    b_sq: float, gain: float, m_max: float, active: bool,
) -> tuple[float, float, float, float, float, float, float]:
    """Limited cross-product dipole and its torque, in scalars.

    Returns ``(mx, my, mz, |m|, Tx, Ty, Tz)``. All zero when ``active`` is False.
    """
    if not active:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    cx = by * hz - bz * hy
    cy = bz * hx - bx * hz
    cz = bx * hy - by * hx
    scale = -gain / b_sq
    mx, my, mz = scale * cx, scale * cy, scale * cz
    m_norm = np.sqrt(mx * mx + my * my + mz * mz)
    if m_norm > m_max:
        trim = m_max / m_norm
        mx, my, mz = mx * trim, my * trim, mz * trim
        m_norm = m_max
    return mx, my, mz, m_norm, my * bz - mz * by, mz * bx - mx * bz, mx * by - my * bx


def rollout(
    episode: Episode,
    decide: Callable[[int, NDArray[np.float64]], tuple[bool, float]],
    record_history: bool = True,
) -> Rollout:
    """Simulate one episode closed-loop under a decision function.

    ``decide(k, features)`` is called once per window with the feature vector of
    :data:`FEATURE_NAMES` and must return ``(actuate, confidence)``. The safety override
    of :data:`SAFETY_OVERRIDE_FRACTION` is applied afterwards and is recorded in
    ``actions`` with confidence 1.0.

    The substep integration is written out in scalar arithmetic rather than in numpy
    calls. That is not premature optimisation: tuning the baseline and the learned
    scheduler both require thousands of closed-loop rollouts, and the per-call overhead of
    three-element numpy operations dominated them by an order of magnitude.

    With ``record_history=False`` the momentum and dipole time histories are not stored
    and are returned as zero-length arrays; the metrics, actions, confidences and
    features are unaffected. Used by the tuning loops, which discard the histories.
    """
    dt = episode.window_s / episode.substeps
    ox, oy, oz = (float(v) for v in episode.omega_body_rad_s)
    env = episode.envelope_nms
    m_max = episode.max_dipole_am2
    gain = episode.gain
    n_steps = episode.n_windows * episode.substeps
    hx, hy, hz = (float(v) for v in episode.initial_momentum_nms)
    if record_history:
        h_hist = np.zeros((n_steps + 1, 3))
        dip_hist = np.zeros((n_steps, 3))
        time_s = np.arange(n_steps + 1) * dt
        h_hist[0] = (hx, hy, hz)
    else:
        h_hist = np.zeros((0, 3))
        dip_hist = np.zeros((0, 3))
        time_s = np.zeros(0)
    actions = np.zeros(episode.n_windows, dtype=bool)
    confidences = np.zeros(episode.n_windows)
    feats = np.zeros((episode.n_windows, N_FEATURES))
    near_limit = NEAR_SATURATION_FRACTION * env
    dh_last = 0.0
    since = 99
    step = 0
    near = 0
    peak = np.sqrt(hx * hx + hy * hy + hz * hz) / env
    cost = 0.0
    for k in range(episode.n_windows):
        h0x, h0y, h0z = hx, hy, hz
        f = _features(episode, k, hx, hy, hz, dh_last, since)
        feats[k] = f
        want, conf = decide(k, f)
        if f[0] >= SAFETY_OVERRIDE_FRACTION:
            want, conf = True, 1.0
        want = bool(want)
        actions[k] = want
        confidences[k] = float(conf)
        since = 0 if want else since + 1
        tdx, tdy, tdz = (float(v) for v in episode.torque_body_nm[k])
        bx, by, bz = (float(v) for v in episode.b_body_t[k])
        b_sq = bx * bx + by * by + bz * bz
        for _ in range(episode.substeps):
            mx1, my1, mz1, n1, t1x, t1y, t1z = _scalar_dipole(
                hx, hy, hz, bx, by, bz, b_sq, gain, m_max, want
            )
            k1x = tdx + t1x - (oy * hz - oz * hy)
            k1y = tdy + t1y - (oz * hx - ox * hz)
            k1z = tdz + t1z - (ox * hy - oy * hx)
            px, py, pz = hx + dt * k1x, hy + dt * k1y, hz + dt * k1z
            mx2, my2, mz2, n2, t2x, t2y, t2z = _scalar_dipole(
                px, py, pz, bx, by, bz, b_sq, gain, m_max, want
            )
            k2x = tdx + t2x - (oy * pz - oz * py)
            k2y = tdy + t2y - (oz * px - ox * pz)
            k2z = tdz + t2z - (ox * py - oy * px)
            mx, my, mz = 0.5 * (mx1 + mx2), 0.5 * (my1 + my2), 0.5 * (mz1 + mz2)
            cost += 0.5 * (n1 + n2) * dt
            hx += 0.5 * dt * (k1x + k2x)
            hy += 0.5 * dt * (k1y + k2y)
            hz += 0.5 * dt * (k1z + k2z)
            step += 1
            mag = np.sqrt(hx * hx + hy * hy + hz * hz)
            if mag > near_limit:
                near += 1
            frac = mag / env
            if frac > peak:
                peak = frac
            if record_history:
                h_hist[step] = (hx, hy, hz)
                dip_hist[step - 1] = (mx, my, mz)
        dh_last = float(
            np.sqrt((hx - h0x) ** 2 + (hy - h0y) ** 2 + (hz - h0z) ** 2)
        )
    duty = cost / (m_max * episode.duration_s)
    near_frac = near / n_steps
    return Rollout(
        actions=actions,
        confidences=confidences,
        features=feats,
        h_history_nms=h_hist,
        dipole_history_am2=dip_hist,
        time_s=time_s,
        metrics=EpisodeMetrics(
            dipole_cost_am2s=float(cost),
            duty_fraction=float(duty),
            near_saturation_fraction=float(near_frac),
            max_h_fraction=float(peak),
            violated=bool(peak > 1.0),
            cost=episode_cost(float(duty), float(near_frac), float(peak)),
        ),
    )
