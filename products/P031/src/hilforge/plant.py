"""The plant and sensor models both backends drive.

Why one model, used twice
-------------------------
Simulation/device parity is only meaningful if the two backends are not two
different simulations. :class:`AttitudePlant` is instantiated once per backend
and the device backend's loopback driver wraps *this same class*, so a parity
failure can only come from the HAL path, never from a modelling difference.

Model
-----
Single-axis rigid-body attitude, torque in, attitude and rate out:

    I * theta_ddot(t) = u(t)                                [N*m]

with ``I`` the moment of inertia about the axis [kg*m^2], ``theta`` the
attitude angle [rad] and ``u`` the commanded torque [N*m]. This is the
single-axis reduction of Euler's rigid-body equations with the gyroscopic
coupling dropped; see Wie, *Space Vehicle Dynamics and Control*, 2nd ed.,
AIAA 2008, Chapter 5 (rigid-body dynamics) and Chapter 7 (single-axis
manoeuvres). Validity: small rates, one axis, no flexible modes, no
disturbance torques, rigid body.

Under a zero-order-hold torque held constant over a step ``dt`` the state
transition is exact for the double integrator:

    theta[k+1] = theta[k] + omega[k]*dt + 0.5*(u[k]/I)*dt^2
    omega[k+1] = omega[k] + (u[k]/I)*dt

(Franklin, Powell & Workman, *Digital Control of Dynamic Systems*, 3rd ed.,
Addison-Wesley 1998, §4.3: the ZOH-equivalent discrete model of a continuous
plant; for a double integrator the matrix exponential terminates after two
terms, so the expressions above carry no discretisation error.) Validity:
``u`` genuinely constant across the step, which is what an actuator holding a
command between loop iterations does.

Gyro model
----------
    omega_meas = omega_true + b + n,   n ~ N(0, sigma_n^2),  sigma_n = N/sqrt(dt)

with ``b`` a constant bias [rad/s] and ``N`` the angle random walk
coefficient [rad/s/sqrt(Hz)], the white-noise-on-rate term of the
IEEE Std 952-2020 single-axis gyro error model (*IEEE Standard
Specification Format Guide and Test Procedure for Single-Axis
Interferometric Fiber Optic Gyros*, §12 error terms). The ``1/sqrt(dt)``
scaling is the standard conversion from a continuous-time PSD to a
discrete-sample standard deviation. Bias instability, rate random walk,
scale-factor error and misalignment are **not** modelled.

Determinism
-----------
Noise is drawn from ``numpy.random.Generator(PCG64(seed))``. NumPy documents
PCG64 streams as reproducible across platforms and versions for a given seed,
which is what lets a seeded run be compared bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ConfigurationError

__all__ = ["AttitudePlant", "PDController", "PDGains", "PlantConfig"]


@dataclass(frozen=True)
class PlantConfig:
    """Plant and sensor parameters, with units.

    Attributes
    ----------
    inertia_kgm2:
        Moment of inertia about the controlled axis [kg*m^2], > 0.
    gyro_bias_rad_s:
        Constant gyro bias [rad/s].
    gyro_arw_rad_s_sqrt_hz:
        Angle random walk coefficient ``N`` [rad/s/sqrt(Hz)], >= 0.
    encoder_noise_rad:
        Standard deviation of the attitude measurement noise [rad], >= 0.
    torque_limit_nm:
        Actuator saturation magnitude [N*m], > 0.
    theta0_rad, omega0_rad_s:
        Initial attitude [rad] and rate [rad/s].
    """

    inertia_kgm2: float = 12.0
    gyro_bias_rad_s: float = 1.0e-4
    gyro_arw_rad_s_sqrt_hz: float = 3.0e-4
    encoder_noise_rad: float = 5.0e-5
    torque_limit_nm: float = 1.5
    theta0_rad: float = 0.08
    omega0_rad_s: float = 0.0

    def __post_init__(self) -> None:
        if not (self.inertia_kgm2 > 0.0):
            raise ConfigurationError(
                f"inertia_kgm2 must be > 0, got {self.inertia_kgm2!r}"
            )
        if self.gyro_arw_rad_s_sqrt_hz < 0.0:
            raise ConfigurationError(
                f"gyro_arw_rad_s_sqrt_hz must be >= 0, got {self.gyro_arw_rad_s_sqrt_hz!r}"
            )
        if self.encoder_noise_rad < 0.0:
            raise ConfigurationError(
                f"encoder_noise_rad must be >= 0, got {self.encoder_noise_rad!r}"
            )
        if not (self.torque_limit_nm > 0.0):
            raise ConfigurationError(
                f"torque_limit_nm must be > 0, got {self.torque_limit_nm!r}"
            )


class AttitudePlant:
    """Single-axis attitude plant with a noisy gyro and encoder.

    The object is mutable state plus a seeded generator. :meth:`snapshot` and
    :meth:`restore` exist so the recovery procedure in
    :mod:`hilforge.deploy` can put the plant back exactly where it was,
    including the generator's internal state.

    Parameters
    ----------
    config:
        Plant and sensor parameters.
    seed:
        Seed for the PCG64 noise stream.
    """

    def __init__(self, config: PlantConfig | None = None, *, seed: int = 0) -> None:
        self.config = config or PlantConfig()
        self._seed = int(seed)
        self._rng = np.random.Generator(np.random.PCG64(self._seed))
        self._theta = float(self.config.theta0_rad)
        self._omega = float(self.config.omega0_rad_s)
        self._torque = 0.0
        self.step_count = 0

    @property
    def seed(self) -> int:
        """Seed of the noise stream."""
        return self._seed

    @property
    def state(self) -> np.ndarray:
        """True state ``[theta rad, omega rad/s]`` as a fresh array."""
        return np.array([self._theta, self._omega], dtype=np.float64)

    def step(self, dt_s: float) -> np.ndarray:
        """Advance the plant by ``dt_s`` [s] under the held torque.

        Returns the new true state. Uses the exact ZOH double-integrator
        transition (module docstring).
        """
        if dt_s < 0.0:
            raise ConfigurationError(f"dt_s must be >= 0, got {dt_s!r}")
        acc = self._torque / self.config.inertia_kgm2
        self._theta = self._theta + self._omega * dt_s + 0.5 * acc * dt_s * dt_s
        self._omega = self._omega + acc * dt_s
        self.step_count += 1
        return self.state

    def apply_torque(self, torque_nm: float) -> tuple[float, bool]:
        """Hold ``torque_nm`` [N*m] until the next call, saturating at the limit.

        Returns ``(applied_torque_nm, saturated)``.
        """
        lim = self.config.torque_limit_nm
        applied = float(np.clip(torque_nm, -lim, lim))
        saturated = bool(applied != float(torque_nm))
        self._torque = applied
        return applied, saturated

    def measure(self, dt_s: float) -> np.ndarray:
        """Noisy measurement ``[theta_meas rad, omega_meas rad/s]``.

        ``dt_s`` [s] is the sampling interval, used to convert the gyro ARW
        coefficient into a per-sample standard deviation (module docstring).
        """
        if not (dt_s > 0.0):
            raise ConfigurationError(f"dt_s must be > 0 for a measurement, got {dt_s!r}")
        sigma_w = self.config.gyro_arw_rad_s_sqrt_hz / np.sqrt(dt_s)
        noise = self._rng.standard_normal(2)
        theta_m = self._theta + self.config.encoder_noise_rad * noise[0]
        omega_m = self._omega + self.config.gyro_bias_rad_s + sigma_w * noise[1]
        return np.array([theta_m, omega_m], dtype=np.float64)

    def snapshot(self) -> dict[str, object]:
        """Complete restorable state, including the RNG bit state."""
        return {
            "theta": self._theta,
            "omega": self._omega,
            "torque": self._torque,
            "step_count": self.step_count,
            "rng_state": self._rng.bit_generator.state,
        }

    def restore(self, snap: dict[str, object]) -> None:
        """Restore a :meth:`snapshot`. Raises ``KeyError`` on a foreign dict."""
        self._theta = float(snap["theta"])  # type: ignore[arg-type]
        self._omega = float(snap["omega"])  # type: ignore[arg-type]
        self._torque = float(snap["torque"])  # type: ignore[arg-type]
        self.step_count = int(snap["step_count"])  # type: ignore[arg-type]
        self._rng.bit_generator.state = snap["rng_state"]


@dataclass(frozen=True)
class PDGains:
    """Proportional-derivative gains for the single-axis loop.

    ``u = -kp * (theta - theta_cmd) - kd * omega`` [N*m], the textbook
    single-axis PD attitude law (Wie 2008, §7.2). For the double integrator
    ``I theta_ddot = u`` the closed loop is
    ``theta_ddot + (kd/I) theta_dot + (kp/I) theta = (kp/I) theta_cmd``, i.e.
    ``wn = sqrt(kp/I)`` [rad/s] and ``zeta = kd / (2 sqrt(kp I))`` [-].

    Attributes
    ----------
    kp:
        Proportional gain [N*m/rad], > 0.
    kd:
        Derivative gain [N*m*s/rad], >= 0.
    """

    kp: float = 12.0
    kd: float = 16.8

    def __post_init__(self) -> None:
        if not (self.kp > 0.0):
            raise ConfigurationError(f"kp must be > 0, got {self.kp!r}")
        if self.kd < 0.0:
            raise ConfigurationError(f"kd must be >= 0, got {self.kd!r}")

    def natural_frequency_rad_s(self, inertia_kgm2: float) -> float:
        """``wn = sqrt(kp / I)`` [rad/s]."""
        if not (inertia_kgm2 > 0.0):
            raise ConfigurationError(f"inertia_kgm2 must be > 0, got {inertia_kgm2!r}")
        return float(np.sqrt(self.kp / inertia_kgm2))

    def damping_ratio(self, inertia_kgm2: float) -> float:
        """``zeta = kd / (2 sqrt(kp I))`` [-]."""
        if not (inertia_kgm2 > 0.0):
            raise ConfigurationError(f"inertia_kgm2 must be > 0, got {inertia_kgm2!r}")
        return float(self.kd / (2.0 * np.sqrt(self.kp * inertia_kgm2)))


class PDController:
    """Stateless single-axis PD attitude controller.

    Parameters
    ----------
    gains:
        See :class:`PDGains`.
    theta_cmd_rad:
        Commanded attitude [rad].
    """

    def __init__(self, gains: PDGains | None = None, *, theta_cmd_rad: float = 0.0) -> None:
        self.gains = gains or PDGains()
        self.theta_cmd_rad = float(theta_cmd_rad)

    def command(self, measurement: np.ndarray) -> np.ndarray:
        """Torque command [N*m] from ``[theta_meas rad, omega_meas rad/s]``.

        Returns a length-1 array so it matches the actuator channel spec.
        """
        m = np.asarray(measurement, dtype=np.float64)
        if m.shape != (2,):
            raise ValueError(f"measurement must have shape (2,), got {m.shape}")
        u = -self.gains.kp * (m[0] - self.theta_cmd_rad) - self.gains.kd * m[1]
        return np.array([u], dtype=np.float64)
