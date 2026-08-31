"""Coupled attitude + orbit + magnetic-control simulation.

The state is ``(q, omega)``: the inertial-to-body quaternion and the body rate.
Position comes from the closed-form circular orbit (``orbit.py``) rather than
being integrated, and the field from the tilted dipole (``magfield.py``); the
whole inertial field history is therefore precomputed in one vectorised call
before the attitude loop starts.

Integration
-----------
Classical fixed-step RK4 on ``(q, omega)``.  The commanded dipole is held
constant over each **control step** (zero-order hold), which is how a real
ADCS runs it, and the dynamics are advanced with ``substeps`` RK4 steps inside
each control step.  The quaternion is renormalised once per control step; the
largest norm drift seen before renormalisation is reported in
``DetumbleResult.max_quat_norm_error``.

The B-dot derivative is estimated the way flight software does it, by a
backward difference of successive magnetometer samples,

    dB/dt[i] = (B[i] - B[i-1]) / dt_control

with optional zero-mean Gaussian magnetometer noise.  On the first control
step there is no previous sample, so the commanded dipole is zero.

The RK4 right-hand side and the quaternion rotation are written out in scalar
form rather than with numpy array calls.  On 3- and 4-vectors the numpy
per-call overhead dominates the arithmetic, and the scalar form measured about
20x faster in this package's own benchmark
(``tests/test_benchmark_regression.py``).

Units: SI throughout (rad/s, T, A m^2, N m, kg m^2, s).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .attitude import quat_normalize
from .constants import OMEGA_EARTH_RAD_S
from .magfield import dipole_field_ecef
from .orbit import CircularOrbit
from .spacecraft import Magnetorquer, validate_inertia


class Controller(Protocol):
    """Anything that maps ``(B_body, dB/dt_body, omega_body)`` to a dipole."""

    def command(
        self,
        b_body_t: ArrayLike,
        b_dot_body_t_s: ArrayLike | None = ...,
        omega_body: ArrayLike | None = ...,
    ) -> NDArray[np.float64]:
        ...


@dataclass
class DetumbleConfig:
    """Everything needed to run one detumble simulation.

    Parameters
    ----------
    inertia : ndarray (3, 3)
        Body inertia tensor [kg m^2], symmetric positive definite.
    orbit : CircularOrbit
        Orbit driving the field history.
    magnetorquer : Magnetorquer
        Per-axis dipole limits [A m^2].
    omega0_rad_s : ndarray (3,)
        Initial body rate [rad/s].
    q0 : ndarray (4,)
        Initial scalar-first inertial-to-body quaternion.
    duration_s : float
        Maximum simulated time [s].
    control_dt_s : float
        Control and magnetometer sampling interval [s].
    substeps : int
        RK4 steps per control interval.
    target_rate_rad_s : float
        Detumble threshold on ``|omega|`` [rad/s]; default 0.1 deg/s.
    mag_noise_t : float
        Standard deviation of per-axis magnetometer noise [T]; 0 disables it.
    seed : int
        Seed for the magnetometer-noise stream.
    stop_when_detumbled : bool
        Stop as soon as ``|omega|`` first falls to ``target_rate_rad_s``.
        Sweeps use this because only the crossing time is needed.
    """

    inertia: NDArray[np.float64]
    orbit: CircularOrbit = field(default_factory=CircularOrbit)
    magnetorquer: Magnetorquer = field(default_factory=Magnetorquer)
    omega0_rad_s: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.05, -0.03, 0.04])
    )
    q0: NDArray[np.float64] = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    duration_s: float = 20000.0
    control_dt_s: float = 1.0
    substeps: int = 2
    target_rate_rad_s: float = 0.0017453292519943296  # 0.1 deg/s
    mag_noise_t: float = 0.0
    seed: int = 0
    stop_when_detumbled: bool = False

    def __post_init__(self) -> None:
        self.inertia = validate_inertia(self.inertia)
        w = np.asarray(self.omega0_rad_s, dtype=float)
        if w.shape != (3,):
            raise ValueError(f"omega0_rad_s must have shape (3,), got {w.shape}")
        if not np.all(np.isfinite(w)):
            raise ValueError("omega0_rad_s contains non-finite entries")
        self.omega0_rad_s = w
        self.q0 = quat_normalize(self.q0)
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")
        if not np.isfinite(self.control_dt_s) or self.control_dt_s <= 0.0:
            raise ValueError(f"control_dt_s must be positive, got {self.control_dt_s}")
        if self.control_dt_s > self.duration_s:
            raise ValueError("control_dt_s must not exceed duration_s")
        if int(self.substeps) < 1:
            raise ValueError(f"substeps must be >= 1, got {self.substeps}")
        self.substeps = int(self.substeps)
        if self.target_rate_rad_s < 0.0:
            raise ValueError("target_rate_rad_s must be non-negative")
        if self.mag_noise_t < 0.0:
            raise ValueError("mag_noise_t must be non-negative")


@dataclass
class DetumbleResult:
    """Time histories and summary metrics of one detumble run.

    Attributes
    ----------
    t_s : ndarray (N,)
        Sample times at control-step boundaries [s].
    omega_rad_s : ndarray (N, 3)
        Body rate [rad/s].
    quat : ndarray (N, 4)
        Inertial-to-body quaternion, scalar first.
    b_body_t : ndarray (N, 3)
        True body-frame field [T] (noise-free).
    dipole_am2 : ndarray (N, 3)
        Applied post-saturation dipole [A m^2], held over the following step.
    torque_nm : ndarray (N, 3)
        Applied torque [N m].
    saturated : ndarray (N,) of bool
        True where the commanded dipole hit a per-axis limit.
    rate_norm_rad_s, energy_j, h_norm_nms : ndarray (N,)
        ``|omega|``, rotational kinetic energy [J], ``|J omega|`` [N m s].
    detumble_time_s : float
        Linearly interpolated first crossing of ``target_rate_rad_s`` [s];
        NaN if the threshold is never reached inside ``duration_s``.
    actuation_cost_a2m4s : float
        ``integral |m|^2 dt`` [A^2 m^4 s], proportional to coil ohmic energy
        for a fixed coil resistance and turns-area product.
    saturated_fraction : float
        Fraction of control steps with at least one saturated axis.
    max_quat_norm_error : float
        Largest ``| |q| - 1 |`` observed before renormalisation.
    """

    t_s: NDArray[np.float64]
    omega_rad_s: NDArray[np.float64]
    quat: NDArray[np.float64]
    b_body_t: NDArray[np.float64]
    dipole_am2: NDArray[np.float64]
    torque_nm: NDArray[np.float64]
    saturated: NDArray[np.bool_]
    rate_norm_rad_s: NDArray[np.float64]
    energy_j: NDArray[np.float64]
    h_norm_nms: NDArray[np.float64]
    detumble_time_s: float
    actuation_cost_a2m4s: float
    saturated_fraction: float
    max_quat_norm_error: float

    @property
    def detumbled(self) -> bool:
        """True if the rate threshold was reached inside the simulated span."""
        return bool(np.isfinite(self.detumble_time_s))


def field_history_eci(
    orbit: CircularOrbit, t_s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Inertial dipole field [T] at each time in ``t_s``, shape ``(N, 3)``.

    Vectorised equivalent of calling ``magfield.dipole_field_eci`` in a loop.
    """
    t = np.asarray(t_s, dtype=float)
    r_eci = orbit.position_eci(t)
    th = orbit.gmst0_rad + OMEGA_EARTH_RAD_S * t
    c, s = np.cos(th), np.sin(th)
    # ECEF components of the inertial position (rotation by +th about z).
    x_e = c * r_eci[:, 0] + s * r_eci[:, 1]
    y_e = -s * r_eci[:, 0] + c * r_eci[:, 1]
    r_ecef = np.stack([x_e, y_e, r_eci[:, 2]], axis=-1)
    b_ecef = dipole_field_ecef(r_ecef)
    # Rotate the field back to ECI (rotation by -th about z).
    bx = c * b_ecef[:, 0] - s * b_ecef[:, 1]
    by = s * b_ecef[:, 0] + c * b_ecef[:, 1]
    return np.stack([bx, by, b_ecef[:, 2]], axis=-1)


def _rotate_by_quat(q, v):
    """``A(q) v`` in scalar form: inertial vector ``v`` into body axes."""
    q0, q1, q2, q3 = q
    v0, v1, v2 = v
    dot = q1 * v0 + q2 * v1 + q3 * v2
    a = q0 * q0 - (q1 * q1 + q2 * q2 + q3 * q3)
    cx = q2 * v2 - q3 * v1
    cy = q3 * v0 - q1 * v2
    cz = q1 * v1 - q2 * v0
    return (
        a * v0 + 2.0 * q1 * dot - 2.0 * q0 * cx,
        a * v1 + 2.0 * q2 * dot - 2.0 * q0 * cy,
        a * v2 + 2.0 * q3 * dot - 2.0 * q0 * cz,
    )


def _rk4_substeps(q, w, jm, jinv, torque, dt, n):
    """``n`` RK4 steps of ``(q, omega)`` under a constant body torque.

    ``jm`` and ``jinv`` are row-major 9-tuples of the inertia tensor and its
    inverse; ``torque`` is a 3-tuple [N m]; ``q`` a scalar-first 4-tuple and
    ``w`` a 3-tuple [rad/s].  Returns the advanced ``(q, w)`` tuples.

    Written out in scalars and fully unrolled: this is the hot loop of the
    whole package, and on 3- and 4-vectors the numpy per-call overhead is an
    order of magnitude larger than the arithmetic itself.
    """
    (j00, j01, j02, j10, j11, j12, j20, j21, j22) = jm
    (i00, i01, i02, i10, i11, i12, i20, i21, i22) = jinv
    lx, ly, lz = torque
    h = dt * 0.5
    sixth = dt / 6.0
    q0, q1, q2, q3 = q
    w0, w1, w2 = w
    for _ in range(n):
        a0, a1, a2, a3, b0, b1, b2 = q0, q1, q2, q3, w0, w1, w2
        jw0 = j00 * b0 + j01 * b1 + j02 * b2
        jw1 = j10 * b0 + j11 * b1 + j12 * b2
        jw2 = j20 * b0 + j21 * b1 + j22 * b2
        r0 = lx - (b1 * jw2 - b2 * jw1)
        r1 = ly - (b2 * jw0 - b0 * jw2)
        r2 = lz - (b0 * jw1 - b1 * jw0)
        c4 = i00 * r0 + i01 * r1 + i02 * r2
        c5 = i10 * r0 + i11 * r1 + i12 * r2
        c6 = i20 * r0 + i21 * r1 + i22 * r2
        c0 = -0.5 * (a1 * b0 + a2 * b1 + a3 * b2)
        c1 = 0.5 * (a0 * b0 + a2 * b2 - a3 * b1)
        c2 = 0.5 * (a0 * b1 + a3 * b0 - a1 * b2)
        c3 = 0.5 * (a0 * b2 + a1 * b1 - a2 * b0)
        a0 = q0 + h * c0
        a1 = q1 + h * c1
        a2 = q2 + h * c2
        a3 = q3 + h * c3
        b0 = w0 + h * c4
        b1 = w1 + h * c5
        b2 = w2 + h * c6
        jw0 = j00 * b0 + j01 * b1 + j02 * b2
        jw1 = j10 * b0 + j11 * b1 + j12 * b2
        jw2 = j20 * b0 + j21 * b1 + j22 * b2
        r0 = lx - (b1 * jw2 - b2 * jw1)
        r1 = ly - (b2 * jw0 - b0 * jw2)
        r2 = lz - (b0 * jw1 - b1 * jw0)
        d4 = i00 * r0 + i01 * r1 + i02 * r2
        d5 = i10 * r0 + i11 * r1 + i12 * r2
        d6 = i20 * r0 + i21 * r1 + i22 * r2
        d0 = -0.5 * (a1 * b0 + a2 * b1 + a3 * b2)
        d1 = 0.5 * (a0 * b0 + a2 * b2 - a3 * b1)
        d2 = 0.5 * (a0 * b1 + a3 * b0 - a1 * b2)
        d3 = 0.5 * (a0 * b2 + a1 * b1 - a2 * b0)
        a0 = q0 + h * d0
        a1 = q1 + h * d1
        a2 = q2 + h * d2
        a3 = q3 + h * d3
        b0 = w0 + h * d4
        b1 = w1 + h * d5
        b2 = w2 + h * d6
        jw0 = j00 * b0 + j01 * b1 + j02 * b2
        jw1 = j10 * b0 + j11 * b1 + j12 * b2
        jw2 = j20 * b0 + j21 * b1 + j22 * b2
        r0 = lx - (b1 * jw2 - b2 * jw1)
        r1 = ly - (b2 * jw0 - b0 * jw2)
        r2 = lz - (b0 * jw1 - b1 * jw0)
        e4 = i00 * r0 + i01 * r1 + i02 * r2
        e5 = i10 * r0 + i11 * r1 + i12 * r2
        e6 = i20 * r0 + i21 * r1 + i22 * r2
        e0 = -0.5 * (a1 * b0 + a2 * b1 + a3 * b2)
        e1 = 0.5 * (a0 * b0 + a2 * b2 - a3 * b1)
        e2 = 0.5 * (a0 * b1 + a3 * b0 - a1 * b2)
        e3 = 0.5 * (a0 * b2 + a1 * b1 - a2 * b0)
        a0 = q0 + dt * e0
        a1 = q1 + dt * e1
        a2 = q2 + dt * e2
        a3 = q3 + dt * e3
        b0 = w0 + dt * e4
        b1 = w1 + dt * e5
        b2 = w2 + dt * e6
        jw0 = j00 * b0 + j01 * b1 + j02 * b2
        jw1 = j10 * b0 + j11 * b1 + j12 * b2
        jw2 = j20 * b0 + j21 * b1 + j22 * b2
        r0 = lx - (b1 * jw2 - b2 * jw1)
        r1 = ly - (b2 * jw0 - b0 * jw2)
        r2 = lz - (b0 * jw1 - b1 * jw0)
        f4 = i00 * r0 + i01 * r1 + i02 * r2
        f5 = i10 * r0 + i11 * r1 + i12 * r2
        f6 = i20 * r0 + i21 * r1 + i22 * r2
        f0 = -0.5 * (a1 * b0 + a2 * b1 + a3 * b2)
        f1 = 0.5 * (a0 * b0 + a2 * b2 - a3 * b1)
        f2 = 0.5 * (a0 * b1 + a3 * b0 - a1 * b2)
        f3 = 0.5 * (a0 * b2 + a1 * b1 - a2 * b0)
        q0 += sixth * (c0 + 2.0 * d0 + 2.0 * e0 + f0)
        q1 += sixth * (c1 + 2.0 * d1 + 2.0 * e1 + f1)
        q2 += sixth * (c2 + 2.0 * d2 + 2.0 * e2 + f2)
        q3 += sixth * (c3 + 2.0 * d3 + 2.0 * e3 + f3)
        w0 += sixth * (c4 + 2.0 * d4 + 2.0 * e4 + f4)
        w1 += sixth * (c5 + 2.0 * d5 + 2.0 * e5 + f5)
        w2 += sixth * (c6 + 2.0 * d6 + 2.0 * e6 + f6)
    return (q0, q1, q2, q3), (w0, w1, w2)


def simulate_detumble(config: DetumbleConfig, controller: Controller) -> DetumbleResult:
    """Run one detumble simulation.

    Parameters
    ----------
    config : DetumbleConfig
    controller : Controller
        Object exposing ``command(b_body_t, b_dot_body_t_s, omega_body)``
        returning a commanded dipole [A m^2].

    Returns
    -------
    DetumbleResult
    """
    cfg = config
    n_steps = int(np.floor(cfg.duration_s / cfg.control_dt_s))
    if n_steps < 1:
        raise ValueError("duration_s / control_dt_s must be at least 1")
    dt_c = float(cfg.control_dt_s)
    dt_sub = dt_c / cfg.substeps
    j = cfg.inertia
    j_inv = np.linalg.inv(j)
    jm = tuple(float(x) for x in j.ravel())
    jinv = tuple(float(x) for x in j_inv.ravel())
    rng = np.random.default_rng(cfg.seed)
    noise = float(cfg.mag_noise_t)
    lim = cfg.magnetorquer.max_dipole_am2
    lim0, lim1, lim2 = float(lim[0]), float(lim[1]), float(lim[2])
    target = float(cfg.target_rate_rad_s)

    n_out = n_steps + 1
    t_grid = np.arange(n_out, dtype=float) * dt_c
    b_eci_all = field_history_eci(cfg.orbit, t_grid)
    noise_all = (
        rng.normal(0.0, noise, size=(n_out, 3)) if noise > 0.0 else np.zeros((n_out, 3))
    )

    q = tuple(float(x) for x in cfg.q0)
    w = tuple(float(x) for x in cfg.omega0_rad_s)

    ts: list[float] = []
    ws: list[tuple[float, float, float]] = []
    qs: list[tuple[float, float, float, float]] = []
    bs: list[tuple[float, float, float]] = []
    ms: list[tuple[float, float, float]] = []
    ls: list[tuple[float, float, float]] = []
    sats: list[bool] = []

    prev_meas: tuple[float, float, float] | None = None
    max_qerr = 0.0
    cost = 0.0
    zero3 = (0.0, 0.0, 0.0)

    for i in range(n_out):
        b_b = _rotate_by_quat(q, b_eci_all[i])
        ts.append(t_grid[i])
        ws.append(w)
        qs.append(q)
        bs.append(b_b)
        if i == n_out - 1:
            ms.append(zero3)
            ls.append(zero3)
            sats.append(False)
            break

        nz = noise_all[i]
        meas = (b_b[0] + nz[0], b_b[1] + nz[1], b_b[2] + nz[2])
        if prev_meas is None:
            m_cmd = zero3
        else:
            b_dot = (
                (meas[0] - prev_meas[0]) / dt_c,
                (meas[1] - prev_meas[1]) / dt_c,
                (meas[2] - prev_meas[2]) / dt_c,
            )
            raw = controller.command(
                np.array(meas), np.array(b_dot), np.array(w)
            )
            m_cmd = (float(raw[0]), float(raw[1]), float(raw[2]))
        prev_meas = meas

        sat = (
            abs(m_cmd[0]) > lim0 or abs(m_cmd[1]) > lim1 or abs(m_cmd[2]) > lim2
        )
        m0 = lim0 if m_cmd[0] > lim0 else (-lim0 if m_cmd[0] < -lim0 else m_cmd[0])
        m1 = lim1 if m_cmd[1] > lim1 else (-lim1 if m_cmd[1] < -lim1 else m_cmd[1])
        m2 = lim2 if m_cmd[2] > lim2 else (-lim2 if m_cmd[2] < -lim2 else m_cmd[2])
        torque = (
            m1 * b_b[2] - m2 * b_b[1],
            m2 * b_b[0] - m0 * b_b[2],
            m0 * b_b[1] - m1 * b_b[0],
        )
        ms.append((m0, m1, m2))
        ls.append(torque)
        sats.append(sat)
        cost += (m0 * m0 + m1 * m1 + m2 * m2) * dt_c

        q, w = _rk4_substeps(q, w, jm, jinv, torque, dt_sub, cfg.substeps)
        qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) ** 0.5
        if abs(qn - 1.0) > max_qerr:
            max_qerr = abs(qn - 1.0)
        q = (q[0] / qn, q[1] / qn, q[2] / qn, q[3] / qn)

        if cfg.stop_when_detumbled:
            if (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) ** 0.5 <= target:
                b_next = _rotate_by_quat(q, b_eci_all[i + 1])
                ts.append(t_grid[i + 1])
                ws.append(w)
                qs.append(q)
                bs.append(b_next)
                ms.append(zero3)
                ls.append(zero3)
                sats.append(False)
                break

    t_s = np.asarray(ts, dtype=float)
    omega = np.asarray(ws, dtype=float)
    quat = np.asarray(qs, dtype=float)
    b_body = np.asarray(bs, dtype=float)
    dipole = np.asarray(ms, dtype=float)
    torque_h = np.asarray(ls, dtype=float)
    sat_arr = np.asarray(sats, dtype=bool)

    rate = np.linalg.norm(omega, axis=1)
    energy = 0.5 * np.einsum("ij,jk,ik->i", omega, j, omega)
    h_norm = np.linalg.norm(omega @ j.T, axis=1)
    n_ctrl = max(len(ts) - 1, 1)

    return DetumbleResult(
        t_s=t_s,
        omega_rad_s=omega,
        quat=quat,
        b_body_t=b_body,
        dipole_am2=dipole,
        torque_nm=torque_h,
        saturated=sat_arr,
        rate_norm_rad_s=rate,
        energy_j=energy,
        h_norm_nms=h_norm,
        detumble_time_s=crossing_time(t_s, rate, target),
        actuation_cost_a2m4s=cost,
        saturated_fraction=float(sat_arr[:n_ctrl].mean()),
        max_quat_norm_error=max_qerr,
    )


def crossing_time(
    t_s: NDArray[np.float64], values: NDArray[np.float64], threshold: float
) -> float:
    """First time ``values`` falls to ``threshold``, linearly interpolated [s].

    Returns NaN if the series never reaches the threshold.
    """
    v = np.asarray(values, dtype=float)
    t = np.asarray(t_s, dtype=float)
    below = np.flatnonzero(v <= threshold)
    if below.size == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(t[0])
    v0, v1 = float(v[i - 1]), float(v[i])
    if v0 == v1:
        return float(t[i])
    frac = (v0 - threshold) / (v0 - v1)
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))
