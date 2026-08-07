"""Pointing-loop controllers (PID, LQR) and a benchmark harness.

Controllers act on a single gimbal axis modelled by
``trackbench.dynamics.GimbalAxis`` (eq. 1-2 there):

    J theta_ddot + b theta_dot = tau

PID
---
Discrete-time PID with derivative-on-measurement and conditional-integration
anti-windup (the standard textbook structure, Astrom & Hagglund 2006,
"Advanced PID Control", ch. 3):

    e[k]   = r[k] - y[k]
    P      = Kp e[k]
    I[k]   = I[k-1] + Ki e[k] dt          (frozen when the output is
                                           saturated and the update would
                                           push it further into saturation)
    D      = -Kd (y[k] - y[k-1]) / dt     (derivative on measurement, so a
                                           setpoint step gives no kick)
    u      = clip(P + I + D, -u_max, u_max)

Tuning used in the shipped scenarios (documented, hand-derived): treating
the plant as a double integrator theta/tau = 1/(J s^2) (valid when
b/J << closed-loop bandwidth), PD control gives the closed loop

    J s^2 + Kd s + Kp = 0  =>  wn = sqrt(Kp/J),  zeta = Kd / (2 sqrt(Kp J))

so for a target natural frequency wn [rad/s] and damping ratio zeta:

    Kp = J wn^2,   Kd = 2 zeta J wn,   Ki = alpha wn Kp   (alpha ~ 0.1)   (5)

``pid_gains_from_bandwidth`` implements eq. (5). The second-order step
response then has the textbook metrics (Ogata 2010, "Modern Control
Engineering", 5th ed., ch. 5):

    overshoot  Mp = exp(-pi zeta / sqrt(1 - zeta^2))                     (6)
    peak time  tp = pi / (wn sqrt(1 - zeta^2))                           (7)

which are used as the hand-checkable validation case (VALIDATION.md sec. 3).

LQR
---
Infinite-horizon continuous-time LQR on the exact linear model (2). With
state x = [theta - r, theta_dot], cost

    J_cost = int_0^inf (x^T Q x + u^T R u) dt                            (8)

the optimal gain is K = R^-1 B^T P where P solves the continuous-time
algebraic Riccati equation (Anderson & Moore 1990, "Optimal Control:
Linear Quadratic Methods")

    A^T P + P A - P B R^-1 B^T P + Q = 0                                 (9)

solved with ``scipy.linalg.solve_continuous_are``. Linearisation note: the
plant (2) is already linear and time-invariant; the only nonlinearities in
the simulated system are the torque/rate/acceleration saturations, which
the LQR design ignores (it is applied with output clipping). This is the
documented modelling deviation.

Units: rad, rad/s, N m, s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are

from trackbench.dynamics import GimbalAxis

__all__ = [
    "PIDController",
    "LQRController",
    "pid_gains_from_bandwidth",
    "lqr_weights_from_bandwidth",
    "zoh_discretize",
    "StepMetrics",
    "step_response",
    "disturbance_rejection_rms",
    "bandwidth_estimate",
    "benchmark_controllers",
]


def pid_gains_from_bandwidth(
    inertia: float, wn: float, zeta: float = 0.707, integral_alpha: float = 0.1
) -> tuple[float, float, float]:
    """PID gains from eq. (5) for a target bandwidth and damping.

    Parameters
    ----------
    inertia : float
        J [kg m^2], > 0.
    wn : float
        Target closed-loop natural frequency [rad/s], > 0.
    zeta : float
        Target damping ratio, > 0 (0.707 -> ~4.3 % overshoot by eq. 6).
    integral_alpha : float
        Ki = integral_alpha * wn * Kp; >= 0. Keep small (<= 0.2) so the
        integrator does not move the dominant poles.

    Returns
    -------
    (Kp [N m/rad], Ki [N m/(rad s)], Kd [N m s/rad])
    """
    if inertia <= 0 or not math.isfinite(inertia):
        raise ValueError(f"inertia must be > 0, got {inertia!r}")
    if wn <= 0 or not math.isfinite(wn):
        raise ValueError(f"wn must be > 0 [rad/s], got {wn!r}")
    if zeta <= 0 or not math.isfinite(zeta):
        raise ValueError(f"zeta must be > 0, got {zeta!r}")
    if integral_alpha < 0:
        raise ValueError(f"integral_alpha must be >= 0, got {integral_alpha!r}")
    kp = inertia * wn**2
    kd = 2.0 * zeta * inertia * wn
    ki = integral_alpha * wn * kp
    return kp, ki, kd


def lqr_weights_from_bandwidth(
    inertia: float, wn: float, q_angle: float = 1.0
) -> tuple[float, float, float]:
    """LQR weights giving a target closed-loop natural frequency.

    For the double integrator theta/tau = 1/(J s^2) with cost weights
    Q = diag(q_angle, 0) and R = r, the LQR closed loop is the
    second-order Butterworth pattern (Anderson & Moore 1990, sec. 3.4:
    "cheap-control"/root-square-locus result) with

        wn = (q_angle / (r J^2))^(1/4),   zeta = sqrt(2)/2

    so the weight that realises a target ``wn`` [rad/s] is

        r = q_angle / (J^2 wn^4)                                     (10)

    The residual plant damping b shifts the achieved poles slightly; the
    shipped scenarios verify the realised bandwidth by sine sweep rather
    than trusting eq. (10) exactly.

    Returns
    -------
    (q_angle, q_rate, r_torque)
    """
    if inertia <= 0 or not math.isfinite(inertia):
        raise ValueError(f"inertia must be > 0, got {inertia!r}")
    if wn <= 0 or not math.isfinite(wn):
        raise ValueError(f"wn must be > 0 [rad/s], got {wn!r}")
    if q_angle <= 0 or not math.isfinite(q_angle):
        raise ValueError(f"q_angle must be > 0, got {q_angle!r}")
    return q_angle, 0.0, q_angle / (inertia**2 * wn**4)


@dataclass
class PIDController:
    """Discrete PID with derivative-on-measurement and anti-windup.

    Parameters
    ----------
    kp, ki, kd : float
        Gains [N m/rad], [N m/(rad s)], [N m s/rad]; all >= 0.
    u_max : float
        Output (torque) limit [N m], > 0.
    """

    kp: float
    ki: float
    kd: float
    u_max: float
    name: str = "PID"
    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_y: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for nm in ("kp", "ki", "kd"):
            v = float(getattr(self, nm))
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{nm} must be finite and >= 0, got {v!r}")
            setattr(self, nm, v)
        u = float(self.u_max)
        if not math.isfinite(u) or u <= 0:
            raise ValueError(f"u_max must be finite and > 0, got {self.u_max!r}")
        self.u_max = u

    def reset(self) -> None:
        """Clear the integrator and derivative memory."""
        self._integral = 0.0
        self._prev_y = None

    @property
    def integral(self) -> float:
        """Current integrator state [N m]."""
        return self._integral

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        """Compute the torque command [N m] for one sample of length ``dt`` [s]."""
        if dt <= 0 or not math.isfinite(dt):
            raise ValueError(f"dt must be > 0 [s], got {dt!r}")
        e = float(setpoint) - float(measurement)
        if self._prev_y is None:
            self._prev_y = float(measurement)
        d_term = -self.kd * (float(measurement) - self._prev_y) / dt
        self._prev_y = float(measurement)

        trial_i = self._integral + self.ki * e * dt
        u_unsat = self.kp * e + trial_i + d_term
        u = float(np.clip(u_unsat, -self.u_max, self.u_max))
        # conditional integration: only accept the integral update if it does
        # not drive the (already saturated) output further into saturation
        if u == u_unsat or (u_unsat - u) * e < 0:
            self._integral = trial_i
        return u


@dataclass
class LQRController:
    """Infinite-horizon LQR for the second-order gimbal axis, eq. (8)-(9).

    Parameters
    ----------
    axis : GimbalAxis
        Plant providing (A, B).
    q_angle : float
        Weight on squared angle error [1/rad^2], > 0.
    q_rate : float
        Weight on squared rate [s^2/rad^2], >= 0.
    r_torque : float
        Weight on squared torque [1/(N m)^2], > 0.
    discrete_dt : float or None
        If given, solve the discrete-time ARE for this sample period [s]
        using a zero-order-hold discretisation instead of the
        continuous-time ARE.
    """

    axis: GimbalAxis
    q_angle: float = 1.0
    q_rate: float = 0.0
    r_torque: float = 1.0
    discrete_dt: float | None = None
    name: str = "LQR"

    def __post_init__(self) -> None:
        if not isinstance(self.axis, GimbalAxis):
            raise TypeError("axis must be a GimbalAxis")
        if self.q_angle <= 0 or not math.isfinite(self.q_angle):
            raise ValueError(f"q_angle must be > 0, got {self.q_angle!r}")
        if self.q_rate < 0 or not math.isfinite(self.q_rate):
            raise ValueError(f"q_rate must be >= 0, got {self.q_rate!r}")
        if self.r_torque <= 0 or not math.isfinite(self.r_torque):
            raise ValueError(f"r_torque must be > 0, got {self.r_torque!r}")
        a, b = self.axis.state_space()
        q = np.diag([self.q_angle, self.q_rate])
        r = np.array([[self.r_torque]])
        if self.discrete_dt is None:
            p = solve_continuous_are(a, b, q, r)
            self.gain = np.linalg.solve(r, b.T @ p).ravel()
        else:
            dt = float(self.discrete_dt)
            if dt <= 0 or not math.isfinite(dt):
                raise ValueError(f"discrete_dt must be > 0 [s], got {self.discrete_dt!r}")
            ad, bd = zoh_discretize(a, b, dt)
            p = solve_discrete_are(ad, bd, q * dt, r * dt)
            self.gain = np.linalg.solve(r * dt + bd.T @ p @ bd, bd.T @ p @ ad).ravel()
        self.riccati_p = p
        self.u_max = self.axis.torque_max
        self._prev_y: float | None = None
        self._rate_est = 0.0

    @property
    def closed_loop_poles(self) -> np.ndarray:
        """Eigenvalues of A - B K [rad/s] (continuous design)."""
        a, b = self.axis.state_space()
        return np.linalg.eigvals(a - b @ self.gain.reshape(1, -1))

    def reset(self) -> None:
        """Clear the internal rate estimator."""
        self._prev_y = None
        self._rate_est = 0.0

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        """Torque command [N m]; rate is estimated by backward difference."""
        if dt <= 0 or not math.isfinite(dt):
            raise ValueError(f"dt must be > 0 [s], got {dt!r}")
        y = float(measurement)
        if self._prev_y is None:
            self._rate_est = 0.0
        else:
            self._rate_est = (y - self._prev_y) / dt
        self._prev_y = y
        x = np.array([y - float(setpoint), self._rate_est])
        u = float(-self.gain @ x)
        return float(np.clip(u, -self.u_max, self.u_max))


def zoh_discretize(a: np.ndarray, b: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold discretisation via the matrix exponential.

    [[Ad, Bd], [0, I]] = expm([[A, B], [0, 0]] * dt)   (standard result,
    e.g. Franklin, Powell & Workman 1998, "Digital Control of Dynamic
    Systems", ch. 4).
    """
    from scipy.linalg import expm

    n = a.shape[0]
    m = b.shape[1]
    big = np.zeros((n + m, n + m))
    big[:n, :n] = a
    big[:n, n:] = b
    e = expm(big * dt)
    return e[:n, :n], e[:n, n:]


@dataclass
class StepMetrics:
    """Step-response metrics.

    Attributes (all [s] except overshoot [-] and steady_state_error [rad])
    ---------------------------------------------------------------------
    rise_time : 10 %-to-90 % rise time.
    overshoot : (peak - final) / final, dimensionless.
    settling_time : time to enter and stay within 2 % of the final value.
    peak_time : time of the maximum response.
    steady_state_error : setpoint - mean of the last 10 % of the response.
    """

    rise_time: float
    overshoot: float
    settling_time: float
    peak_time: float
    steady_state_error: float


def _metrics(t: np.ndarray, y: np.ndarray, setpoint: float) -> StepMetrics:
    final = float(np.mean(y[int(0.9 * len(y)) :]))
    ref = setpoint if setpoint != 0 else 1.0
    lo, hi = 0.1 * ref, 0.9 * ref
    idx_lo = np.flatnonzero(y >= lo)
    idx_hi = np.flatnonzero(y >= hi)
    rise = (
        float(t[idx_hi[0]] - t[idx_lo[0]]) if idx_lo.size and idx_hi.size else float("nan")
    )
    peak_i = int(np.argmax(y))
    overshoot = float((y[peak_i] - ref) / abs(ref))
    band = 0.02 * abs(ref)
    outside = np.flatnonzero(np.abs(y - ref) > band)
    settle = float(t[outside[-1] + 1]) if outside.size and outside[-1] + 1 < len(t) else 0.0
    return StepMetrics(
        rise_time=rise,
        overshoot=max(overshoot, 0.0),
        settling_time=settle,
        peak_time=float(t[peak_i]),
        steady_state_error=float(ref - final),
    )


def _simulate_axis(
    axis: GimbalAxis,
    controller,
    setpoint_fn,
    disturbance: np.ndarray | None,
    dt: float,
    n_steps: int,
    sensor_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-loop simulation of one axis. Returns (t, measured LOS error, torque).

    ``disturbance`` is an additive line-of-sight angular disturbance [rad]
    (platform jitter) of length ``n_steps``; the controlled quantity is
    ``axis.angle + disturbance[k]``.
    """
    axis.reset()
    controller.reset()
    if rng is None:
        rng = np.random.default_rng(0)
    t = np.arange(n_steps) * dt
    y = np.zeros(n_steps)
    u = np.zeros(n_steps)
    for k in range(n_steps):
        d = 0.0 if disturbance is None else float(disturbance[k])
        los = axis.angle + d
        meas = los + (rng.normal(0.0, sensor_noise) if sensor_noise > 0 else 0.0)
        sp = setpoint_fn(t[k])
        torque = controller.update(sp, meas, dt)
        axis.step(torque, dt)
        y[k] = los
        u[k] = torque
    return t, y, u


def step_response(
    axis: GimbalAxis,
    controller,
    setpoint: float,
    dt: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, StepMetrics]:
    """Closed-loop unit-step response and its metrics.

    Returns (t [s], theta [rad], StepMetrics).
    """
    if duration <= 0 or dt <= 0:
        raise ValueError("dt and duration must be > 0 [s]")
    n = int(round(duration / dt))
    t, y, _ = _simulate_axis(axis, controller, lambda _t: setpoint, None, dt, n)
    return t, y, _metrics(t, y, setpoint)


def disturbance_rejection_rms(
    axis: GimbalAxis,
    controller,
    disturbance: np.ndarray,
    dt: float,
    sensor_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> float:
    """RMS line-of-sight error [rad] under an additive angular disturbance.

    The open-loop RMS is ``np.std(disturbance)``; the ratio of the two is
    the disturbance-rejection factor reported by the benchmark.
    """
    disturbance = np.asarray(disturbance, dtype=float)
    if disturbance.ndim != 1 or disturbance.size < 2:
        raise ValueError("disturbance must be a 1-D array with >= 2 samples")
    _, y, _ = _simulate_axis(
        axis,
        controller,
        lambda _t: 0.0,
        disturbance,
        dt,
        disturbance.size,
        sensor_noise=sensor_noise,
        rng=rng,
    )
    burn = disturbance.size // 10
    return float(np.sqrt(np.mean(y[burn:] ** 2)))


def bandwidth_estimate(
    axis: GimbalAxis,
    controller,
    dt: float,
    freqs: np.ndarray | None = None,
    cycles: int = 6,
) -> float:
    """-3 dB closed-loop tracking bandwidth [Hz] by sine sweep.

    For each probe frequency a sinusoidal setpoint is tracked and the
    output/input amplitude ratio is measured by least-squares projection
    onto sin/cos at that frequency (single-bin DFT). The bandwidth is the
    lowest frequency where the gain first falls below 1/sqrt(2) of the
    low-frequency gain, found by linear interpolation in log-frequency.
    """
    if freqs is None:
        freqs = np.logspace(math.log10(0.5), 2.0, 18)
    freqs = np.asarray(freqs, dtype=float)
    amp = 1e-5  # small-signal probe [rad], keeps the loop out of saturation
    gains = np.zeros(freqs.size)
    for i, f in enumerate(freqs):
        n = max(int(round(cycles / f / dt)), 64)
        t, y, _ = _simulate_axis(
            axis, controller, lambda tt, f=f: amp * math.sin(2 * math.pi * f * tt), None, dt, n
        )
        burn = n // 3
        tt, yy = t[burn:], y[burn:]
        s = np.sin(2 * math.pi * f * tt)
        c = np.cos(2 * math.pi * f * tt)
        a1 = 2.0 * np.mean(yy * s)
        b1 = 2.0 * np.mean(yy * c)
        gains[i] = math.hypot(a1, b1) / amp
    g0 = gains[0]
    thr = g0 / math.sqrt(2.0)
    below = np.flatnonzero(gains < thr)
    if below.size == 0:
        return float(freqs[-1])
    i = int(below[0])
    if i == 0:
        return float(freqs[0])
    lf0, lf1 = math.log(freqs[i - 1]), math.log(freqs[i])
    g_prev, g_cur = gains[i - 1], gains[i]
    frac = (g_prev - thr) / (g_prev - g_cur) if g_prev != g_cur else 0.0
    return float(math.exp(lf0 + frac * (lf1 - lf0)))


def benchmark_controllers(
    axis_factory,
    controllers: dict,
    dt: float = 1e-3,
    step_setpoint: float = 1e-4,
    step_duration: float = 2.0,
    disturbance: np.ndarray | None = None,
) -> list[dict]:
    """Run step / disturbance / bandwidth benchmarks for several controllers.

    Parameters
    ----------
    axis_factory : callable
        Zero-argument callable returning a fresh ``GimbalAxis`` (each
        benchmark gets an independent plant instance).
    controllers : dict[str, callable]
        Name -> callable(axis) returning a controller.
    disturbance : np.ndarray, optional
        Angular disturbance series [rad] for the rejection test.

    Returns
    -------
    list of dict
        One row per controller with keys: name, rise_time_s, overshoot,
        settling_time_s, steady_state_error_rad, dist_rms_rad,
        rejection_factor, bandwidth_hz.
    """
    rows = []
    for name, factory in controllers.items():
        axis = axis_factory()
        ctrl = factory(axis)
        _, _, m = step_response(axis, ctrl, step_setpoint, dt, step_duration)
        row = {
            "name": name,
            "rise_time_s": m.rise_time,
            "overshoot": m.overshoot,
            "settling_time_s": m.settling_time,
            "steady_state_error_rad": m.steady_state_error,
        }
        if disturbance is not None:
            axis_d = axis_factory()
            ctrl_d = factory(axis_d)
            rms = disturbance_rejection_rms(axis_d, ctrl_d, disturbance, dt)
            row["dist_rms_rad"] = rms
            ol = float(np.std(disturbance))
            row["rejection_factor"] = ol / rms if rms > 0 else float("inf")
        axis_b = axis_factory()
        ctrl_b = factory(axis_b)
        row["bandwidth_hz"] = bandwidth_estimate(axis_b, ctrl_b, dt)
        rows.append(row)
    return rows
