"""Closed-loop simulation of the GNC loop with fault injection.

One step of the loop, in the order the flight software would execute it:

1. form the true measurement ``z = H x + v``;
2. inject the sensor fault, if any (:func:`fdiscope.faults.apply_sensor_fault`);
3. run the Kalman update on the *faulted* measurement, yielding the innovation
   ``y_k`` and its covariance ``S``;
4. compute the PD command from the posterior estimate and the reference;
5. clip to the wheel torque limit;
6. inject the actuator fault, if any, and clip again -- the plant receives
   ``u_actual``, the filter is told ``u_cmd``;
7. propagate the plant with ``u_actual`` and the process noise;
8. propagate the filter with ``u_cmd``.

Step 6 is the point of the whole exercise: the filter's model of the actuator
is wrong exactly when the actuator is faulted, and the innovation is the only
place that shows.

The reference is a sinusoidal attitude command, ``theta_ref(t) = A sin(2 pi t
/ T_ref)``, so the commanded torque is persistently non-zero.  Without a
persistently exciting command a loss of effectiveness is unobservable --
``(1 - l) * 0 == 0`` -- and this package would be measuring nothing.

Normalised residual
-------------------
With ``S = L L^T`` (Cholesky), the normalised residual is ``r_k = L^-1 y_k``,
which is ``N(0, I_m)`` under the fault-free hypothesis for a consistent
filter, so ``|r_k|^2`` is the normalised innovation squared.  The simulation
is started from the steady-state filter covariance
(:func:`fdiscope.kalman.steady_state_covariance`) so that no start-up
transient contaminates the false-alarm measurement.

Units: [rad], [rad/s], [N m], [s].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .faults import FaultSpec, FaultType, apply_actuator_fault, apply_sensor_fault
from .kalman import KalmanFilter, KalmanState, steady_state_covariance
from .plant import ControllerGains, LoopMatrices, PlantConfig, loop_matrices

__all__ = ["LoopConfig", "LoopRun", "build_filter", "simulate_loop"]


@dataclass(frozen=True)
class LoopConfig:
    """Everything needed to run one closed-loop case.

    Parameters
    ----------
    plant : PlantConfig
        Plant, sensors and wheel limit.
    gains : ControllerGains
        PD gains.
    n_steps : int
        Number of samples to simulate.  Must be >= 2.
    ref_amplitude_rad : float
        Amplitude ``A`` of the sinusoidal attitude reference [rad].
    ref_period_s : float
        Period ``T_ref`` of the reference [s].  Must be positive.
    seed : int
        Seed for the process- and measurement-noise streams.
    noise : bool
        If False, both noise streams are identically zero.  Used to compute
        deterministic fault signatures.
    x0 : ndarray, shape (2,), optional
        True initial state.  Defaults to zeros.
    """

    plant: PlantConfig = field(default_factory=PlantConfig)
    gains: ControllerGains = field(default_factory=ControllerGains)
    n_steps: int = 3000
    ref_amplitude_rad: float = 0.02
    ref_period_s: float = 60.0
    seed: int = 0
    noise: bool = True
    x0: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if int(self.n_steps) < 2:
            raise ValueError(f"n_steps must be >= 2, got {self.n_steps}")
        if not np.isfinite(self.ref_period_s) or self.ref_period_s <= 0.0:
            raise ValueError(f"ref_period_s must be positive, got {self.ref_period_s}")
        if not np.isfinite(self.ref_amplitude_rad):
            raise ValueError("ref_amplitude_rad must be finite")
        if len(self.x0) != 2:
            raise ValueError(f"x0 must have 2 elements, got {len(self.x0)}")


@dataclass(frozen=True)
class LoopRun:
    """Histories produced by :func:`simulate_loop`.

    Attributes
    ----------
    t_s : ndarray, shape (N,)
        Sample times [s].
    x_true : ndarray, shape (N, 2)
        True state before the update of that step [rad, rad/s].
    x_est : ndarray, shape (N, 2)
        Posterior estimate [rad, rad/s].
    innovation : ndarray, shape (N, 2)
        ``y_k`` [rad, rad/s].
    residual : ndarray, shape (N, 2)
        Normalised innovation ``L^-1 y_k``, dimensionless, ``N(0, I)`` under
        ``H0``.
    nis : ndarray, shape (N,)
        ``|r_k|^2``, dimensionless.
    u_cmd_nm, u_actual_nm : ndarray, shape (N,)
        Commanded and delivered torque [N m].
    innovation_cov : ndarray, shape (2, 2)
        Steady-state ``S`` [rad^2, (rad/s)^2].
    chol_s : ndarray, shape (2, 2)
        Lower Cholesky factor of ``S``.
    onset_step : int
        First faulted sample, or ``-1`` for a fault-free run.
    fault : FaultSpec
        The injected fault.
    """

    t_s: NDArray[np.float64]
    x_true: NDArray[np.float64]
    x_est: NDArray[np.float64]
    innovation: NDArray[np.float64]
    residual: NDArray[np.float64]
    nis: NDArray[np.float64]
    u_cmd_nm: NDArray[np.float64]
    u_actual_nm: NDArray[np.float64]
    innovation_cov: NDArray[np.float64] = field(repr=False)
    chol_s: NDArray[np.float64] = field(repr=False)
    onset_step: int = -1
    fault: FaultSpec = field(default_factory=FaultSpec)


def build_filter(matrices: LoopMatrices) -> KalmanFilter:
    """Kalman filter matched to the healthy plant.

    Parameters
    ----------
    matrices : LoopMatrices
        From :func:`fdiscope.plant.loop_matrices`.

    Returns
    -------
    KalmanFilter
        Filter with the same ``F``, ``G``, ``H``, ``Q``, ``R`` as the plant.
        Model mismatch is *not* simulated: every residual excursion in this
        package comes from an injected fault, not from a wrong model.  That
        is a deliberate simplification and it flatters every detector here
        (see README Limitations).
    """
    return KalmanFilter(f=matrices.f, h=matrices.h, q=matrices.q, r=matrices.r, g=matrices.g)


def simulate_loop(
    config: LoopConfig,
    fault: FaultSpec | None = None,
    steady_state_gain: bool = True,
) -> LoopRun:
    """Run the closed loop for ``config.n_steps`` samples.

    Parameters
    ----------
    config : LoopConfig
        Loop configuration.
    fault : FaultSpec, optional
        Fault to inject.  ``None`` or ``FaultType.NONE`` gives a healthy run.
    steady_state_gain : bool
        If True (the default) the update uses the constant steady-state gain
        ``K`` instead of propagating ``P`` every step.  The filter is
        time-invariant and is *started* at the steady-state covariance, so
        ``P`` never moves and the two paths are algebraically identical; the
        fast path is about 20x quicker, which is what makes the benchmark fit
        the compute budget.  Setting False runs the full
        :class:`fdiscope.kalman.KalmanFilter` recursion instead, and
        ``tests/test_simulate.py`` asserts the two agree to 1e-12.

    Returns
    -------
    LoopRun
        Full histories, including the normalised residual sequence the
        detectors consume.
    """
    if not isinstance(config, LoopConfig):
        raise TypeError(f"config must be a LoopConfig, got {type(config).__name__}")
    spec = fault if fault is not None else FaultSpec()
    if not isinstance(spec, FaultSpec):
        raise TypeError(f"fault must be a FaultSpec, got {type(spec).__name__}")

    mats = loop_matrices(config.plant)
    kf = build_filter(mats)
    p_prior, s = steady_state_covariance(kf)
    chol = np.linalg.cholesky(s)
    n = int(config.n_steps)
    dt = mats.dt_s
    j = config.plant.inertia_kgm2
    u_max = config.plant.max_torque_nm

    rng = np.random.default_rng(int(config.seed))
    if config.noise:
        q_chol = np.linalg.cholesky(mats.q + 1e-30 * np.eye(2))
        w_hist = (q_chol @ rng.standard_normal((2, n))).T
        r_chol = np.linalg.cholesky(mats.r)
        v_hist = (r_chol @ rng.standard_normal((2, n))).T
    else:
        w_hist = np.zeros((n, 2))
        v_hist = np.zeros((n, 2))

    t = np.arange(n, dtype=float) * dt
    theta_ref = config.ref_amplitude_rad * np.sin(2.0 * np.pi * t / config.ref_period_s)
    omega_ref = (
        config.ref_amplitude_rad
        * (2.0 * np.pi / config.ref_period_s)
        * np.cos(2.0 * np.pi * t / config.ref_period_s)
    )

    x_true = np.array(config.x0, dtype=float)
    state = KalmanState(x=x_true.copy(), p=p_prior.copy())
    gain = p_prior @ kf.h.T @ np.linalg.inv(s)
    s_inv = np.linalg.inv(s)
    chol_inv = np.linalg.inv(chol)
    f_mat = mats.f
    g_vec = mats.g[:, 0]
    kp = config.gains.kp
    kd = config.gains.kd
    x_prior = x_true.copy()

    x_true_hist = np.zeros((n, 2))
    x_est_hist = np.zeros((n, 2))
    innov = np.zeros((n, 2))
    resid = np.zeros((n, 2))
    nis = np.zeros(n)
    u_cmd_hist = np.zeros(n)
    u_act_hist = np.zeros(n)

    sensor_latch: float | None = None
    actuator_latch: float | None = None

    # The inner loop is written out in scalars rather than 2-vectors.  The
    # state is two-dimensional and fixed, and 480 runs of 2000 steps sit on
    # the critical path of every benchmark; the scalar form is about four
    # times faster than the equivalent NumPy expressions, whose per-call
    # overhead dominates at this size.  `steady_state_gain=False` runs the
    # readable matrix path through KalmanFilter and the two are asserted equal
    # in tests/test_simulate.py.
    f00, f01, f10, f11 = (float(v) for v in (f_mat[0, 0], f_mat[0, 1], f_mat[1, 0], f_mat[1, 1]))
    g0, g1 = float(g_vec[0]), float(g_vec[1])
    k00, k01, k10, k11 = (float(v) for v in (gain[0, 0], gain[0, 1], gain[1, 0], gain[1, 1]))
    c00, c10, c11 = float(chol_inv[0, 0]), float(chol_inv[1, 0]), float(chol_inv[1, 1])
    is00, is01, is11 = float(s_inv[0, 0]), float(s_inv[0, 1]), float(s_inv[1, 1])
    xt0, xt1 = float(x_true[0]), float(x_true[1])
    xp0, xp1 = float(x_prior[0]), float(x_prior[1])
    simple = steady_state_gain and spec.kind is FaultType.NONE
    # Python lists, not NumPy views: scalar arithmetic on np.float64 is several
    # times slower than on float, and this loop is the benchmark's critical path.
    w0_l, w1_l = w_hist[:, 0].tolist(), w_hist[:, 1].tolist()
    v0_l, v1_l = v_hist[:, 0].tolist(), v_hist[:, 1].tolist()
    ref0_l, ref1_l = theta_ref.tolist(), omega_ref.tolist()
    out_xt0, out_xt1 = [0.0] * n, [0.0] * n
    out_xq0, out_xq1 = [0.0] * n, [0.0] * n
    out_y0, out_y1 = [0.0] * n, [0.0] * n
    out_r0, out_r1 = [0.0] * n, [0.0] * n
    out_nis, out_uc, out_ua = [0.0] * n, [0.0] * n, [0.0] * n

    for k in range(n):
        out_xt0[k] = xt0
        out_xt1[k] = xt1
        z0 = xt0 + v0_l[k]
        z1 = xt1 + v1_l[k]
        if not simple:
            z_meas, sensor_latch = apply_sensor_fault(
                np.array([z0, z1]), spec, k, dt, sensor_latch
            )
            z0, z1 = float(z_meas[0]), float(z_meas[1])

        if steady_state_gain:
            y0 = z0 - xp0
            y1 = z1 - xp1
            xq0 = xp0 + k00 * y0 + k01 * y1
            xq1 = xp1 + k10 * y0 + k11 * y1
            out_nis[k] = y0 * (is00 * y0 + is01 * y1) + y1 * (is01 * y0 + is11 * y1)
        else:
            upd = kf.update(state, np.array([z0, z1]))
            y0, y1 = float(upd.innovation[0]), float(upd.innovation[1])
            state = upd.state
            xq0, xq1 = float(state.x[0]), float(state.x[1])
            out_nis[k] = upd.nis
        out_y0[k] = y0
        out_y1[k] = y1
        out_r0[k] = c00 * y0
        out_r1[k] = c10 * y0 + c11 * y1
        out_xq0[k] = xq0
        out_xq1[k] = xq1

        u_cmd = -j * (kp * (xq0 - ref0_l[k]) + kd * (xq1 - ref1_l[k]))
        u_cmd = u_max if u_cmd > u_max else (-u_max if u_cmd < -u_max else u_cmd)
        if simple:
            u_act = u_cmd
        else:
            u_act, actuator_latch = apply_actuator_fault(u_cmd, spec, k, dt, actuator_latch)
            u_act = u_max if u_act > u_max else (-u_max if u_act < -u_max else u_act)
        out_uc[k] = u_cmd
        out_ua[k] = u_act

        xt0, xt1 = (
            f00 * xt0 + f01 * xt1 + g0 * u_act + w0_l[k],
            f10 * xt0 + f11 * xt1 + g1 * u_act + w1_l[k],
        )
        if steady_state_gain:
            xp0, xp1 = f00 * xq0 + f01 * xq1 + g0 * u_cmd, f10 * xq0 + f11 * xq1 + g1 * u_cmd
        else:
            state = kf.predict(state, [u_cmd])
            xp0, xp1 = float(state.x[0]), float(state.x[1])

    x_true_hist[:, 0], x_true_hist[:, 1] = out_xt0, out_xt1
    x_est_hist[:, 0], x_est_hist[:, 1] = out_xq0, out_xq1
    innov[:, 0], innov[:, 1] = out_y0, out_y1
    resid[:, 0], resid[:, 1] = out_r0, out_r1
    nis[:] = out_nis
    u_cmd_hist[:] = out_uc
    u_act_hist[:] = out_ua

    onset = int(spec.onset_step) if spec.kind is not FaultType.NONE else -1
    return LoopRun(
        t_s=t,
        x_true=x_true_hist,
        x_est=x_est_hist,
        innovation=innov,
        residual=resid,
        nis=nis,
        u_cmd_nm=u_cmd_hist,
        u_actual_nm=u_act_hist,
        innovation_cov=s,
        chol_s=chol,
        onset_step=onset,
        fault=spec,
    )
