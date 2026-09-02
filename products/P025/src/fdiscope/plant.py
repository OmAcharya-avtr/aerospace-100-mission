"""Single-axis spacecraft attitude control loop used as the FDI test bed.

Model
-----
A rigid single-axis rotation with a reaction-wheel torque command.  The
continuous plant is the double integrator

.. math::
    J \\ddot{\\theta}(t) = u(t) + w(t)

with ``J`` the axis moment of inertia [kg m^2], ``theta`` the attitude angle
[rad], ``u`` the commanded wheel torque [N m] and ``w`` a zero-mean white
disturbance torque [N m].  This is Euler's equation for rotation about a
principal axis with the gyroscopic term dropped, valid for small rates about
one axis (Markley & Crassidis 2014, ch. 3; Wie 2008, ch. 6).

Exact zero-order-hold discretisation at step ``dt`` (Franklin, Powell &
Workman, *Digital Control of Dynamic Systems*, 3rd ed., 1998, sec. 4.3), with
state ``x = [theta, omega]`` in [rad, rad/s]::

    F = [[1, dt], [0, 1]]
    G = [[dt^2 / (2 J)], [dt / J]]

The process-noise covariance for a continuous white disturbance torque of
one-sided spectral density ``q_t`` [N^2 m^2 s] is the standard
continuous-to-discrete integral (Bar-Shalom, Rong Li & Kirubarajan 2001,
sec. 6.2, "continuous white-noise acceleration" model)::

    Q = (q_t / J^2) * [[dt^3 / 3, dt^2 / 2], [dt^2 / 2, dt]]

Measurements
------------
Two sensors, both sampled every step:

* a coarse attitude sensor,  ``z_0 = theta + v_0``, ``v_0 ~ N(0, r_a)``
* a rate gyro,               ``z_1 = omega + v_1``, ``v_1 ~ N(0, r_g)``

so ``H = I_2`` and ``R = diag(r_a, r_g)`` in [rad^2, (rad/s)^2].  The defaults
are a 0.05 deg (1-sigma) attitude sensor and a 0.02 deg/s (1-sigma) rate gyro,
i.e. a coarse-pointing sensor suite rather than a star tracker.  Both are
modelled as white; a real gyro's angle random walk and rate random walk and a
real attitude sensor's low-frequency errors are **not** modelled (see README
Limitations).

Structural note on observability of a constant angle bias
---------------------------------------------------------
Because both states are measured and ``F`` has a unit eigenvalue in the angle
direction, the steady-state innovation under a constant *angle*-sensor bias is
exactly zero: the estimate simply shifts by the bias.  The closed form is in
:func:`fdiscope.analytic.innovation_dc_gain`, whose second column alone is
non-zero.  A constant attitude-sensor bias is therefore detectable only during
its transient, and this package measures how long that transient lasts rather
than pretending otherwise.

Control law
-----------
Full-state-feedback PD on the *estimated* state,

.. math::
    u_k = -J (k_p \\hat{\\theta}_k + k_d \\hat{\\omega}_k) + u_{ff,k}

with ``k_p = omega_n^2`` [1/s^2] and ``k_d = 2 zeta omega_n`` [1/s], the
standard second-order form (Wie 2008, sec. 7.2).  The loop is therefore a
closed-loop estimator-plus-controller, which is what makes fault detection
non-trivial: a sensor fault is fed back into the actuator command and a wheel
fault is partly masked by the controller.

Units
-----
Angles [rad], rates [rad/s], torque [N m], inertia [kg m^2], time [s].

Validity
--------
Linear, single axis, small angle, no gyroscopic coupling, no flexible modes,
no wheel dynamics or quantisation, no time delay between measurement and
actuation beyond the one-step zero-order hold.

References
----------
Markley, F. L. and Crassidis, J. L., *Fundamentals of Spacecraft Attitude
    Determination and Control*, Springer, 2014.
Wie, B., *Space Vehicle Dynamics and Control*, 2nd ed., AIAA, 2008.
Franklin, G. F., Powell, J. D. and Workman, M. L., *Digital Control of Dynamic
    Systems*, 3rd ed., Addison-Wesley, 1998.
Bar-Shalom, Y., Rong Li, X. and Kirubarajan, T., *Estimation with Applications
    to Tracking and Navigation*, Wiley, 2001.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["PlantConfig", "ControllerGains", "LoopMatrices", "loop_matrices"]


def _positive(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return v


def _non_negative(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")
    return v


@dataclass(frozen=True)
class PlantConfig:
    """Single-axis attitude plant and sensor suite.

    Parameters
    ----------
    inertia_kgm2 : float
        Axis moment of inertia ``J`` [kg m^2].
    dt_s : float
        Sample and control period [s].
    torque_noise_psd : float
        One-sided spectral density of the disturbance torque [N^2 m^2 s].
    attitude_var_rad2 : float
        Attitude-sensor measurement variance [rad^2].  The default
        7.6154e-7 rad^2 is a 0.05 deg 1-sigma sensor.
    gyro_var_rad2_s2 : float
        Rate-gyro measurement variance [(rad/s)^2].  The default
        1.2185e-7 (rad/s)^2 is a 0.02 deg/s 1-sigma gyro.
    max_torque_nm : float
        Symmetric wheel torque limit [N m].  The commanded torque is clipped
        to ``+/- max_torque_nm`` before it reaches the plant.
    """

    inertia_kgm2: float = 12.0
    dt_s: float = 0.1
    torque_noise_psd: float = 4.0e-8
    attitude_var_rad2: float = 7.6154e-7
    gyro_var_rad2_s2: float = 1.2185e-7
    max_torque_nm: float = 0.05

    def __post_init__(self) -> None:
        _positive(self.inertia_kgm2, "inertia_kgm2")
        _positive(self.dt_s, "dt_s")
        _non_negative(self.torque_noise_psd, "torque_noise_psd")
        _positive(self.attitude_var_rad2, "attitude_var_rad2")
        _positive(self.gyro_var_rad2_s2, "gyro_var_rad2_s2")
        _positive(self.max_torque_nm, "max_torque_nm")


@dataclass(frozen=True)
class ControllerGains:
    """PD gains in second-order canonical form.

    Parameters
    ----------
    natural_freq_rad_s : float
        Closed-loop natural frequency ``omega_n`` [rad/s].
    damping : float
        Closed-loop damping ratio ``zeta`` [-].  Must be positive.
    """

    natural_freq_rad_s: float = 0.35
    damping: float = 0.707

    def __post_init__(self) -> None:
        _positive(self.natural_freq_rad_s, "natural_freq_rad_s")
        _positive(self.damping, "damping")

    @property
    def kp(self) -> float:
        """Proportional gain ``omega_n^2`` [1/s^2]."""
        return float(self.natural_freq_rad_s**2)

    @property
    def kd(self) -> float:
        """Rate gain ``2 zeta omega_n`` [1/s]."""
        return float(2.0 * self.damping * self.natural_freq_rad_s)

    def torque(self, state_estimate: NDArray[np.float64], inertia_kgm2: float) -> float:
        """Commanded torque [N m] for an estimated state ``[theta, omega]``."""
        x = np.asarray(state_estimate, dtype=float).reshape(-1)
        if x.size != 2:
            raise ValueError(f"state_estimate must have 2 elements, got {x.size}")
        return float(-inertia_kgm2 * (self.kp * x[0] + self.kd * x[1]))


@dataclass(frozen=True)
class LoopMatrices:
    """Discrete-time matrices of the single-axis loop.

    Attributes
    ----------
    f : ndarray, shape (2, 2)
        State transition, dimensionless / [s].
    g : ndarray, shape (2, 1)
        Torque input matrix [rad/(N m), (rad/s)/(N m)].
    h : ndarray, shape (2, 2)
        Measurement matrix (identity: angle and rate are both measured).
    q : ndarray, shape (2, 2)
        Discrete process-noise covariance [rad^2, rad^2/s, rad^2/s^2].
    r : ndarray, shape (2, 2)
        Measurement-noise covariance [rad^2, (rad/s)^2].
    dt_s : float
        Step [s].
    """

    f: NDArray[np.float64]
    g: NDArray[np.float64]
    h: NDArray[np.float64] = field(repr=False)
    q: NDArray[np.float64] = field(repr=False)
    r: NDArray[np.float64] = field(repr=False)
    dt_s: float = 0.1


def loop_matrices(config: PlantConfig) -> LoopMatrices:
    """Zero-order-hold discretisation of the double integrator.

    Parameters
    ----------
    config : PlantConfig
        Plant, sensor and actuator parameters.

    Returns
    -------
    LoopMatrices
        ``F``, ``G``, ``H``, ``Q``, ``R`` as documented in the module
        docstring.

    Notes
    -----
    ``F`` and ``G`` are the exact ZOH discretisation of the double integrator
    (Franklin, Powell & Workman 1998, sec. 4.3); ``Q`` is the continuous
    white-noise-acceleration discretisation of Bar-Shalom et al. 2001,
    sec. 6.2, scaled by ``1/J^2`` because the disturbance is a torque, not an
    angular acceleration.
    """
    if not isinstance(config, PlantConfig):
        raise TypeError(f"config must be a PlantConfig, got {type(config).__name__}")
    dt = config.dt_s
    j = config.inertia_kgm2
    f = np.array([[1.0, dt], [0.0, 1.0]])
    g = np.array([[0.5 * dt * dt / j], [dt / j]])
    h = np.eye(2)
    q_scale = config.torque_noise_psd / (j * j)
    q = q_scale * np.array([[dt**3 / 3.0, 0.5 * dt * dt], [0.5 * dt * dt, dt]])
    r = np.diag([config.attitude_var_rad2, config.gyro_var_rad2_s2])
    return LoopMatrices(f=f, g=g, h=h, q=q, r=r, dt_s=dt)
