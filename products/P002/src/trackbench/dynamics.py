"""Gimbal dynamics, platform-jitter synthesis and angle-sensor model.

Scope
-----
Two-axis (azimuth / elevation) coarse pointing gimbal, modelled as two
decoupled rigid-body axes driven by motor torque, plus a base-motion
(platform) jitter disturbance and a noisy angle sensor.

Plant model (per axis)
----------------------
Newton's second law for rotation about a fixed axis with viscous damping
(standard rigid-body result, e.g. Meirovitch 2001, "Fundamentals of
Vibrations", ch. 1; the same second-order model is used for gimbal
pointing loops in Hemmati (ed.) 2006, "Deep Space Optical Communications"):

    J * theta_ddot + b * theta_dot = tau                       (1)

with
    J     [kg m^2]   axis inertia                (> 0)
    b     [N m s/rad] viscous damping            (>= 0)
    tau   [N m]      commanded motor torque, |tau| <= tau_max
    theta [rad]      axis angle

State-space form x = [theta, theta_dot]^T:

    xdot = A x + B tau,   A = [[0, 1], [0, -b/J]],  B = [0, 1/J]^T   (2)

Open-loop transfer function theta(s)/tau(s) = 1 / (s (J s + b)):
a free integrator plus a pole at s = -b/J. The corresponding mechanical
time constant is tau_m = J / b [s].

Limits (documented, applied in ``GimbalAxis.step``):
- torque limit  tau_max [N m]  -> acceleration limit |theta_ddot| <=
  tau_max / J (when theta_dot = 0);
- rate limit    rate_max [rad/s] -> theta_dot is clipped, and the
  acceleration is zeroed when it would push |theta_dot| further past the
  limit (a hard saturation, not a smooth model);
- acceleration limit accel_max [rad/s^2] -> optional extra clip applied to
  the achieved angular acceleration, representing a drive-electronics or
  structural constraint tighter than tau_max / J.

Integration uses fixed-step RK4 on eq. (2) with the torque held constant
over the step (zero-order hold), which is the standard digital-control
assumption.

Jitter disturbance
------------------
Platform angular jitter is synthesised from a target one-sided power
spectral density S(f) [rad^2/Hz] by spectral factorisation in the
frequency domain (the standard "random phase / amplitude spectrum" method,
see e.g. Percival & Walden 1993, "Spectral Analysis for Physical
Applications", or Shinozuka & Deodatis 1991, Appl. Mech. Rev. 44(4), for
the spectral representation of stationary Gaussian processes):

    X_k = sqrt(S(f_k) * fs * N / 2) * exp(i * phi_k),  phi_k ~ U[0, 2 pi)
    x[n] = irfft(X)[n]                                            (3)

so that the periodogram of x, P(f_k) = 2 |rfft(x)_k|^2 / (fs N), has
expectation S(f_k). DC and Nyquist bins are set to zero (zero-mean signal,
no aliasing energy at Nyquist).

Sensor
------
Quadrant-detector / star-tracker style angle sensor with additive white
Gaussian noise of standard deviation NEA (noise-equivalent angle) [rad],
optional quantisation, and optional dropout (fraction of samples returning
the previous valid measurement flagged invalid). NEA is the conventional
figure of merit for optical tracking sensors (Hemmati (ed.) 2006).

Units: rad, rad/s, rad/s^2, N m, s, Hz unless stated otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "GimbalAxis",
    "TwoAxisGimbal",
    "JitterPSD",
    "synthesize_jitter",
    "welch_psd",
    "AngleSensor",
]


def _pos(name: str, value: float) -> float:
    v = float(value)
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return v


def _nonneg(name: str, value: float) -> float:
    v = float(value)
    if not (math.isfinite(v) and v >= 0):
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return v


@dataclass
class GimbalAxis:
    """Single gimbal axis: second-order plant with rate/accel/torque limits.

    Parameters
    ----------
    inertia : float
        J [kg m^2], > 0.
    damping : float
        b [N m s/rad], >= 0.
    torque_max : float
        |tau| limit [N m], > 0.
    rate_max : float
        |theta_dot| limit [rad/s], > 0.
    accel_max : float or None
        Optional |theta_ddot| limit [rad/s^2]; None disables the extra clip.
    """

    inertia: float
    damping: float
    torque_max: float
    rate_max: float
    accel_max: float | None = None

    def __post_init__(self) -> None:
        self.inertia = _pos("inertia", self.inertia)
        self.damping = _nonneg("damping", self.damping)
        self.torque_max = _pos("torque_max", self.torque_max)
        self.rate_max = _pos("rate_max", self.rate_max)
        if self.accel_max is not None:
            self.accel_max = _pos("accel_max", self.accel_max)
        self.angle = 0.0
        self.rate = 0.0
        self.saturated_torque = False
        self.saturated_rate = False

    # --- linear model -------------------------------------------------
    def state_space(self) -> tuple[np.ndarray, np.ndarray]:
        """Continuous-time (A, B) of eq. (2); state [angle, rate], input torque."""
        a = np.array([[0.0, 1.0], [0.0, -self.damping / self.inertia]])
        b = np.array([[0.0], [1.0 / self.inertia]])
        return a, b

    @property
    def mechanical_time_constant(self) -> float:
        """J / b [s]; ``inf`` for an undamped axis."""
        return math.inf if self.damping == 0.0 else self.inertia / self.damping

    # --- simulation ---------------------------------------------------
    def reset(self, angle: float = 0.0, rate: float = 0.0) -> None:
        """Reset the axis state [rad], [rad/s]."""
        self.angle = float(angle)
        self.rate = float(rate)
        self.saturated_torque = False
        self.saturated_rate = False

    def _accel(self, rate: float, torque: float) -> float:
        acc = (torque - self.damping * rate) / self.inertia
        if self.accel_max is not None:
            acc = float(np.clip(acc, -self.accel_max, self.accel_max))
        return acc

    def step(self, torque: float, dt: float) -> tuple[float, float]:
        """Advance one step of ``dt`` [s] under constant ``torque`` [N m].

        RK4 on eq. (2) with zero-order-hold torque, followed by rate
        saturation. Returns the new (angle [rad], rate [rad/s]).
        """
        dt = _pos("dt", dt)
        t_cmd = float(torque)
        if not math.isfinite(t_cmd):
            raise ValueError(f"torque must be finite, got {torque!r}")
        t_sat = float(np.clip(t_cmd, -self.torque_max, self.torque_max))
        self.saturated_torque = t_sat != t_cmd

        x = np.array([self.angle, self.rate])

        def deriv(state: np.ndarray) -> np.ndarray:
            return np.array([state[1], self._accel(state[1], t_sat)])

        k1 = deriv(x)
        k2 = deriv(x + 0.5 * dt * k1)
        k3 = deriv(x + 0.5 * dt * k2)
        k4 = deriv(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        rate = float(x[1])
        clipped = float(np.clip(rate, -self.rate_max, self.rate_max))
        self.saturated_rate = clipped != rate
        self.angle = float(x[0])
        self.rate = clipped
        return self.angle, self.rate


class TwoAxisGimbal:
    """Two decoupled gimbal axes (azimuth, elevation).

    Cross-axis coupling (gimbal-lock geometry, Coriolis terms) is NOT
    modelled; see README Limitations. Valid for small excursions about
    boresight where the two axes are approximately orthogonal and
    kinematically decoupled.
    """

    def __init__(self, az: GimbalAxis, el: GimbalAxis) -> None:
        if not isinstance(az, GimbalAxis) or not isinstance(el, GimbalAxis):
            raise TypeError("az and el must be GimbalAxis instances")
        self.az = az
        self.el = el

    @classmethod
    def from_config(cls, cfg: dict) -> TwoAxisGimbal:
        """Build from a nested dict with keys ``az`` and ``el``."""
        if not isinstance(cfg, dict) or "az" not in cfg or "el" not in cfg:
            raise ValueError("gimbal config must be a dict with 'az' and 'el' keys")
        return cls(GimbalAxis(**cfg["az"]), GimbalAxis(**cfg["el"]))

    def reset(self, angles: tuple[float, float] = (0.0, 0.0)) -> None:
        """Reset both axes to ``angles`` [rad] with zero rate."""
        self.az.reset(angles[0])
        self.el.reset(angles[1])

    @property
    def angles(self) -> np.ndarray:
        """Current [az, el] angles [rad]."""
        return np.array([self.az.angle, self.el.angle])

    @property
    def rates(self) -> np.ndarray:
        """Current [az, el] rates [rad/s]."""
        return np.array([self.az.rate, self.el.rate])

    @property
    def saturated(self) -> bool:
        """True if either axis saturated torque or rate on the last step."""
        return (
            self.az.saturated_torque
            or self.az.saturated_rate
            or self.el.saturated_torque
            or self.el.saturated_rate
        )

    def step(self, torques: np.ndarray, dt: float) -> np.ndarray:
        """Advance both axes by ``dt`` [s] under ``torques`` = [tau_az, tau_el]."""
        torques = np.asarray(torques, dtype=float)
        if torques.shape != (2,):
            raise ValueError(f"torques must have shape (2,), got {torques.shape}")
        self.az.step(float(torques[0]), dt)
        self.el.step(float(torques[1]), dt)
        return self.angles


@dataclass
class JitterPSD:
    """Parametric one-sided platform-jitter PSD model.

    S(f) = S0 / (1 + (f / f_corner)^2)^(order/2)   [rad^2/Hz]           (4)

    A flat low-frequency plateau S0 rolling off as f^-order above
    f_corner. This shape (flat-then-roll-off) is the usual empirical
    description of spacecraft micro-vibration / platform jitter spectra;
    the specific parameters are user-supplied and NOT taken from any
    particular mission (see README Limitations).

    Parameters
    ----------
    s0 : float
        Low-frequency PSD plateau [rad^2/Hz], > 0.
    f_corner : float
        Corner frequency [Hz], > 0.
    order : float
        Roll-off order (2 -> f^-2 decay), > 0.
    """

    s0: float
    f_corner: float
    order: float = 2.0

    def __post_init__(self) -> None:
        self.s0 = _pos("s0", self.s0)
        self.f_corner = _pos("f_corner", self.f_corner)
        self.order = _pos("order", self.order)

    def __call__(self, f: np.ndarray) -> np.ndarray:
        """Evaluate S(f) [rad^2/Hz] for frequencies f [Hz] >= 0."""
        f = np.asarray(f, dtype=float)
        if np.any(f < 0):
            raise ValueError("frequencies must be >= 0 Hz")
        return self.s0 / (1.0 + (f / self.f_corner) ** 2) ** (self.order / 2.0)

    def variance(self, f_max: float, n: int = 200001) -> float:
        """Analytic-by-quadrature variance = integral_0^f_max S(f) df [rad^2]."""
        f_max = _pos("f_max", f_max)
        f = np.linspace(0.0, f_max, int(n))
        return float(np.trapezoid(self(f), f))


def synthesize_jitter(
    psd: Callable[[np.ndarray], np.ndarray],
    n: int,
    fs: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Synthesise a zero-mean stationary time series with one-sided PSD ``psd``.

    Implements eq. (3): random-phase spectral factorisation. The returned
    series has length ``n``, sample rate ``fs`` [Hz], and its expected
    periodogram equals ``psd(f)`` on the rfft grid.

    Parameters
    ----------
    psd : callable
        f [Hz] -> S(f) [rad^2/Hz]; must accept and return arrays.
    n : int
        Number of samples, >= 8.
    fs : float
        Sample rate [Hz], > 0.
    rng : np.random.Generator, optional
        Seeded generator for reproducibility.

    Returns
    -------
    np.ndarray, shape (n,)
        Jitter time series [rad].
    """
    if int(n) < 8:
        raise ValueError(f"n must be >= 8, got {n!r}")
    n = int(n)
    fs = _pos("fs", fs)
    if rng is None:
        rng = np.random.default_rng(0)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    s = np.asarray(psd(freqs), dtype=float)
    if s.shape != freqs.shape:
        raise ValueError("psd callable must return one value per frequency")
    if np.any(s < 0) or not np.all(np.isfinite(s)):
        raise ValueError("psd must return finite non-negative values")
    amp = np.sqrt(s * fs * n / 2.0)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=freqs.size)
    spec = amp * np.exp(1j * phase)
    spec[0] = 0.0  # zero mean
    if n % 2 == 0:
        spec[-1] = 0.0  # Nyquist bin real -> drop
    return np.fft.irfft(spec, n=n)


def welch_psd(x: np.ndarray, fs: float, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD estimate (Hann window, 50 % overlap).

    Thin wrapper over ``scipy.signal.welch`` kept here so validation
    scripts and tests share one estimator. Returns (freqs [Hz],
    PSD [units^2/Hz]).
    """
    from scipy.signal import welch

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    nperseg = min(int(nperseg), x.size)
    f, p = welch(x, fs=fs, nperseg=nperseg, window="hann", noverlap=nperseg // 2)
    return f, p


@dataclass
class AngleSensor:
    """Angle sensor with noise-equivalent angle (NEA), quantisation, dropout.

    Parameters
    ----------
    nea : float
        1-sigma additive white Gaussian noise on each measured axis [rad],
        >= 0.
    quantization : float or None
        LSB size [rad]; None disables quantisation.
    dropout_prob : float
        Per-sample probability of an invalid measurement, in [0, 1).
        On dropout the last valid measurement is returned and ``valid``
        is False.
    """

    nea: float
    quantization: float | None = None
    dropout_prob: float = 0.0

    def __post_init__(self) -> None:
        self.nea = _nonneg("nea", self.nea)
        if self.quantization is not None:
            self.quantization = _pos("quantization", self.quantization)
        if not 0.0 <= float(self.dropout_prob) < 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1), got {self.dropout_prob!r}")
        self.dropout_prob = float(self.dropout_prob)
        self._last = np.zeros(2)

    def reset(self) -> None:
        """Clear the held last-valid measurement."""
        self._last = np.zeros(2)

    def measure(
        self, truth: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, bool]:
        """Measure a 2-vector of true angles [rad].

        Returns
        -------
        (measurement [rad], valid)
        """
        truth = np.asarray(truth, dtype=float)
        if truth.shape != (2,):
            raise ValueError(f"truth must have shape (2,), got {truth.shape}")
        if self.dropout_prob > 0.0 and rng.random() < self.dropout_prob:
            return self._last.copy(), False
        m = truth + rng.normal(0.0, self.nea, size=2) if self.nea > 0 else truth.copy()
        if self.quantization is not None:
            m = np.round(m / self.quantization) * self.quantization
        self._last = m
        return m.copy(), True
