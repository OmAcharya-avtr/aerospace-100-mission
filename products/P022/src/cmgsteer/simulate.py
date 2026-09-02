"""Manoeuvre profiles and torque-error accounting for a CMG steering run.

The simulation is deliberately of the steering law, not of the vehicle: the
commanded torque history is an input, and what is measured is how faithfully
the array delivers it and what the gimbals do on the way.  There is no attitude
state, no controller and no external disturbance.

Two error accountings are kept, because they answer different questions.

* **Instantaneous torque error** ``tau_cmd - tau_achieved`` [N*m], with
  ``tau_achieved = -A(delta_k) ddelta_k`` evaluated at the step's own gimbal
  angles.  This is the steering law's own error and is what the closed-form SR
  expression predicts.
* **Momentum error** ``(-tau_cmd dt) - (h(delta_{k+1}) - h(delta_k))``
  [N*m*s].  This is what the vehicle actually feels, and it additionally
  contains the first-order integration error of the explicit Euler gimbal
  update, which grows as ``O(dt^2)`` per step.  Reporting only the first would
  understate the error near a singularity, where the gimbal rates are largest
  and the Euler step is therefore worst.

Assumptions: ideal gimbal-rate servo, explicit Euler integration of the gimbal
angles, constant commanded torque across a step, and no gimbal-angle
measurement error (:mod:`cmgsteer` has a separate uncertainty study for that).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .arrays import CMGArray
from .nullmotion import NoNullMotion, NullMotionPolicy
from .steering import METHODS, steer

__all__ = [
    "SteeringHistory",
    "TorqueProfile",
    "constant_profile",
    "rest_to_rest_profile",
    "run_steering",
]

PROFILE_SHAPES: tuple[str, ...] = ("bang-bang", "sine")


@dataclass(frozen=True)
class TorqueProfile:
    """A commanded body-torque history on a uniform time grid.

    Attributes
    ----------
    torques
        ``(n_steps, 3)`` commanded body torque [N*m], one row per step.
    dt
        Step length [s].
    name
        Label used in benchmark tables.
    """

    torques: NDArray[np.float64]
    dt: float
    name: str = "profile"

    def __post_init__(self) -> None:
        t = np.atleast_2d(np.asarray(self.torques, dtype=float))
        if t.ndim != 2 or t.shape[1] != 3:
            raise ValueError(f"torques must have shape (n_steps, 3), got {t.shape}")
        if t.shape[0] < 1:
            raise ValueError("a torque profile needs at least one step")
        if not np.all(np.isfinite(t)):
            raise ValueError("torques must be finite [N*m]")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"dt must be positive and finite [s], got {self.dt}")
        t.flags.writeable = False
        object.__setattr__(self, "torques", t)

    @property
    def n_steps(self) -> int:
        """Number of steps in the profile."""
        return int(self.torques.shape[0])

    @property
    def duration(self) -> float:
        """Total profile duration [s]."""
        return float(self.n_steps * self.dt)

    @property
    def momentum_change(self) -> NDArray[np.float64]:
        """Net array momentum change the profile demands, ``-sum(tau) dt`` [N*m*s]."""
        return -self.torques.sum(axis=0) * self.dt

    @property
    def peak_momentum(self) -> float:
        """Largest momentum magnitude the array must hold during the profile [N*m*s]."""
        running = -np.cumsum(self.torques, axis=0) * self.dt
        return float(np.max(np.linalg.norm(running, axis=1)))


def constant_profile(
    torque: ArrayLike, duration: float, dt: float, name: str = "constant"
) -> TorqueProfile:
    """A constant commanded torque held for ``duration`` seconds."""
    t = np.asarray(torque, dtype=float).reshape(-1)
    if t.shape != (3,):
        raise ValueError(f"torque must have shape (3,), got {t.shape}")
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive [s]")
    n = max(1, int(round(duration / dt)))
    return TorqueProfile(np.tile(t, (n, 1)), dt, name)


def rest_to_rest_profile(
    axis: ArrayLike,
    momentum_change: float,
    duration: float,
    dt: float,
    shape: str = "sine",
    name: str = "rest-to-rest",
) -> TorqueProfile:
    """A rest-to-rest torque profile about one axis.

    The profile integrates to zero net momentum change, so the array returns to
    its starting momentum, and reaches a peak stored momentum of
    ``momentum_change`` [N*m*s] half way through.  Two shapes:

    * ``"bang-bang"``: ``+tau_max`` then ``-tau_max``, with
      ``tau_max = 2 dh / T``.
    * ``"sine"``: ``tau_max sin(2 pi t / T)`` with ``tau_max = pi dh / T``.

    Parameters
    ----------
    axis
        Body-frame axis of the manoeuvre; normalised internally.
    momentum_change
        Peak stored momentum ``dh`` [N*m*s], > 0.
    duration
        Manoeuvre duration ``T`` [s], > 0.
    dt
        Step length [s], > 0.
    shape
        One of :data:`PROFILE_SHAPES`.
    """
    u = np.asarray(axis, dtype=float).reshape(-1)
    if u.shape != (3,):
        raise ValueError(f"axis must have shape (3,), got {u.shape}")
    norm = float(np.linalg.norm(u))
    if norm == 0.0:
        raise ValueError("axis must be a non-zero vector")
    u = u / norm
    if momentum_change <= 0.0:
        raise ValueError(f"momentum_change must be positive [N*m*s], got {momentum_change}")
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive [s]")
    n = max(2, int(round(duration / dt)))
    t_mid = (np.arange(n) + 0.5) * dt
    if shape == "bang-bang":
        tau_max = 2.0 * momentum_change / duration
        scalar = np.where(t_mid < 0.5 * duration, tau_max, -tau_max)
    elif shape == "sine":
        tau_max = np.pi * momentum_change / duration
        scalar = tau_max * np.sin(2.0 * np.pi * t_mid / duration)
    else:
        raise ValueError(f"shape must be one of {PROFILE_SHAPES}, got {shape!r}")
    # The commanded body torque is -dh/dt, so storing +dh along `axis` needs a
    # body torque along -axis first.  Sign is carried by the caller's axis.
    return TorqueProfile(scalar[:, None] * (-u)[None, :], dt, name)


@dataclass(frozen=True)
class SteeringHistory:
    """Everything recorded by :func:`run_steering`.

    Array shapes: ``n = n_steps``.  ``deltas``, ``momentum``, ``measure`` and
    ``min_singular_value`` have ``n + 1`` entries (the initial state plus one
    per step); the rest have ``n``.
    """

    times: NDArray[np.float64]
    deltas: NDArray[np.float64]
    momentum: NDArray[np.float64]
    gimbal_rates: NDArray[np.float64]
    commanded_torque: NDArray[np.float64]
    achieved_torque: NDArray[np.float64]
    torque_error: NDArray[np.float64]
    momentum_error: NDArray[np.float64]
    measure: NDArray[np.float64]
    min_singular_value: NDArray[np.float64]
    lam: NDArray[np.float64]
    rate_limited: NDArray[np.bool_]
    null_rates: NDArray[np.float64]
    method: str
    policy: str
    dt: float

    @property
    def torque_error_norm(self) -> NDArray[np.float64]:
        """Per-step instantaneous torque-error magnitude [N*m]."""
        return np.linalg.norm(self.torque_error, axis=1)

    @property
    def max_torque_error(self) -> float:
        """Largest instantaneous torque error over the run [N*m]."""
        return float(np.max(self.torque_error_norm))

    @property
    def rms_torque_error(self) -> float:
        """Root-mean-square instantaneous torque error [N*m]."""
        return float(np.sqrt(np.mean(self.torque_error_norm**2)))

    @property
    def accumulated_momentum_error(self) -> float:
        """Norm of the summed per-step momentum error [N*m*s].

        This is the momentum the vehicle ends up with that it did not ask for,
        so it is the honest single-number summary of a steering run.
        """
        return float(np.linalg.norm(self.momentum_error.sum(axis=0)))

    @property
    def total_momentum_error_path(self) -> float:
        """Path-length sum of per-step momentum-error magnitudes [N*m*s].

        Unlike :attr:`accumulated_momentum_error`, errors in opposite
        directions do not cancel, so this measures how hard the run was rather
        than only where it finished.
        """
        return float(np.sum(np.linalg.norm(self.momentum_error, axis=1)))

    @property
    def min_measure(self) -> float:
        """Smallest singularity measure visited [(N*m*s/rad)^3]."""
        return float(np.min(self.measure))

    @property
    def n_rate_limited(self) -> int:
        """Number of steps on which the gimbal-rate limit was active."""
        return int(np.count_nonzero(self.rate_limited))

    @property
    def peak_gimbal_rate(self) -> float:
        """Largest commanded gimbal rate magnitude over the run [rad/s]."""
        return float(np.max(np.abs(self.gimbal_rates))) if self.gimbal_rates.size else 0.0

    def steps_below_measure(self, threshold: float) -> int:
        """Number of recorded states whose singularity measure is below ``threshold``."""
        return int(np.count_nonzero(self.measure < threshold))


def run_steering(
    array: CMGArray,
    initial_deltas: ArrayLike,
    profile: TorqueProfile,
    method: str = "sr",
    null_policy: NullMotionPolicy | None = None,
    max_gimbal_rate: float | None = None,
    saturation_mode: str = "clip",
    **law_kwargs: object,
) -> SteeringHistory:
    """Run a steering law over a commanded torque profile.

    Parameters
    ----------
    array
        The CMG array.
    initial_deltas
        ``(n,)`` starting gimbal angles [rad].
    profile
        Commanded body-torque history.
    method
        One of :data:`cmgsteer.steering.METHODS`.
    null_policy
        Null-motion policy; ``None`` means :class:`~cmgsteer.nullmotion.NoNullMotion`.
    max_gimbal_rate
        Symmetric gimbal-rate limit [rad/s], or ``None``.
    saturation_mode
        ``"clip"`` or ``"scale"``; see :func:`cmgsteer.steering.apply_rate_limit`.
    **law_kwargs
        Passed to the steering law (``lam``, ``lam0``, ``mu``, ``eps0`` ...).

    Returns
    -------
    :class:`SteeringHistory`.
    """
    if method not in METHODS:
        raise ValueError(f"unknown steering method {method!r}; expected one of {METHODS}")
    d = np.asarray(initial_deltas, dtype=float).reshape(-1)
    if d.shape[0] != array.n_cmgs:
        raise ValueError(f"initial_deltas must have length {array.n_cmgs}, got {d.shape[0]}")
    policy = null_policy if null_policy is not None else NoNullMotion()
    policy.reset()

    n = profile.n_steps
    dt = profile.dt
    n_free = array.n_free
    deltas = np.empty((n + 1, array.n_cmgs))
    momentum = np.empty((n + 1, 3))
    measure = np.empty(n + 1)
    min_sv = np.empty(n + 1)
    rates = np.empty((n, n_free))
    achieved = np.empty((n, 3))
    torque_err = np.empty((n, 3))
    mom_err = np.empty((n, 3))
    lam = np.empty(n)
    limited = np.zeros(n, dtype=bool)
    null_rates = np.zeros((n, n_free))

    deltas[0] = d
    momentum[0] = array.momentum(d)
    for k in range(n):
        t_now = k * dt
        cmd = profile.torques[k]
        null = policy.rates(array, deltas[k], cmd, t_now)
        kwargs = dict(law_kwargs)
        if method == "gsr":
            kwargs.setdefault("time", t_now)
        result = steer(
            array,
            deltas[k],
            cmd,
            method=method,
            null_rates=null,
            max_gimbal_rate=max_gimbal_rate,
            saturation_mode=saturation_mode,
            **kwargs,
        )
        rates[k] = result.gimbal_rates
        achieved[k] = result.achieved_torque
        torque_err[k] = result.torque_error
        lam[k] = result.lam
        limited[k] = result.rate_limited
        null_rates[k] = null
        measure[k] = result.measure
        min_sv[k] = result.min_singular_value
        deltas[k + 1] = deltas[k] + array.expand_rates(result.gimbal_rates) * dt
        momentum[k + 1] = array.momentum(deltas[k + 1])
        mom_err[k] = (-cmd * dt) - (momentum[k + 1] - momentum[k])

    jac = array.jacobian(deltas[n])
    sv = np.linalg.svd(jac, compute_uv=False)
    measure[n] = float(np.prod(sv))
    min_sv[n] = float(sv[-1])

    return SteeringHistory(
        times=np.arange(n + 1) * dt,
        deltas=deltas,
        momentum=momentum,
        gimbal_rates=rates,
        commanded_torque=np.array(profile.torques),
        achieved_torque=achieved,
        torque_error=torque_err,
        momentum_error=mom_err,
        measure=measure,
        min_singular_value=min_sv,
        lam=lam,
        rate_limited=limited,
        null_rates=null_rates,
        method=method,
        policy=policy.name,
        dt=dt,
    )
