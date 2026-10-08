"""Declared linear-Gaussian digital twin and its fixed-gain residual generator.

Model
-----
The twin is a discrete-time linear-Gaussian state-space model

    x_{k+1} = A x_k + B u_k + w_k,     w_k ~ N(0, Q)                     (1)
    y_k     = C x_k + d + v_k,         v_k ~ N(0, R)                     (2)

with ``w`` and ``v`` mutually independent white sequences and ``d`` a declared
scalar measurement offset (units of the measurement). All matrices are declared
by the caller; nothing here is identified from data.

Residual generator
------------------
The monitor runs a **fixed-gain steady-state Kalman filter** built from the
declared model:

    x_hat_{k+1} = A x_hat_k + B u_k + K e_k                               (3)
    e_k         = y_k - (C x_hat_k + d)                                   (4)

where ``x_hat_k`` is the one-step prediction ``x_hat_{k|k-1}``, ``K`` is the
steady-state **predictor** gain

    K = A P C^T S^{-1},      S = C P C^T + R,

and ``P`` is the stabilising solution of the discrete-time filtering algebraic
Riccati equation

    P = A P A^T + Q - A P C^T (C P C^T + R)^{-1} C P A^T.                 (5)

Source: Anderson, B. D. O. & Moore, J. B. (1979), *Optimal Filtering*,
Prentice-Hall, chapters 3 and 4 (steady-state filter and the algebraic Riccati
equation). The fixed-gain form is the one actually deployed on a twin: the gain
is computed once offline, so a residual monitor never sees a transient gain.

The factor ``A`` in ``K = A P C^T S^{-1}`` is the difference between the
*filter* gain, which corrects ``x_hat_{k|k}``, and the *predictor* gain, which
is what equation (3) needs because it propagates ``x_hat_{k+1|k}`` in one step.
Dropping it leaves a sub-optimal filter whose innovations are no longer white.
This repository made exactly that mistake during the build, measured a lag-1
residual autocorrelation of +0.0105 instead of 0, and the episode is recorded
in ``validation/VALIDATION.md`` rather than quietly removed.

When the asset obeys the declared model exactly, the innovation sequence
``e_k`` is zero-mean, white and Gaussian with variance ``S`` -- the innovations
property. Source: Kailath, T. (1968), "An Innovations Approach to Least-Squares
Estimation, Part I: Linear Filtering in Additive Noise", *IEEE Transactions on
Automatic Control* 13(6), 646-655. The **normalised residual**

    z_k = e_k / sqrt(S)                                                   (6)

is therefore i.i.d. N(0, 1) under the in-control hypothesis, which is the only
property every detector in this package relies on. ``validation/
validate_twin.py`` measures it rather than assuming it.

Units
-----
The shipped reference twin is a single-axis attitude channel: state
``[theta (rad), theta_dot (rad/s)]``, input a commanded angular acceleration
(rad/s^2), measurement ``theta`` (rad), sample interval ``dt`` (s). The
normalised residual ``z`` is dimensionless.

Validity range
--------------
Equations (1)-(6) hold for a time-invariant, detectable, stabilisable model
with ``R > 0`` and ``Q`` positive semi-definite. The steady-state gain is only
the optimal gain if the declared ``Q`` and ``R`` are the true ones; the whole
point of this package is that they may not be.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_discrete_are


@dataclass(frozen=True)
class LinearGaussianTwin:
    """A declared discrete-time linear-Gaussian model, equations (1)-(2).

    Parameters
    ----------
    A:
        State transition matrix, shape ``(n, n)``, dimensionless.
    B:
        Input matrix, shape ``(n, m)``; maps input units to state units.
    C:
        Measurement matrix, shape ``(1, n)``. Only scalar measurements are
        supported, because the normalised residual of equation (6) is then a
        scalar and every analytic threshold in :mod:`twininvalidate.thresholds`
        is exact rather than approximate.
    Q:
        Process-noise covariance, shape ``(n, n)``, state units squared.
    R:
        Measurement-noise variance, shape ``(1, 1)``, measurement units squared.
    dt:
        Sample interval in seconds.
    offset:
        Declared constant measurement offset ``d`` of equation (2), in
        measurement units.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    dt: float
    offset: float = 0.0

    def __post_init__(self) -> None:
        for name in ("A", "B", "C", "Q", "R"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim != 2:
                raise ValueError(f"{name} must be a 2-D array, got ndim={value.ndim}")
            object.__setattr__(self, name, value)
        n = self.A.shape[0]
        if self.A.shape != (n, n):
            raise ValueError(f"A must be square, got shape {self.A.shape}")
        if self.B.shape[0] != n:
            raise ValueError(f"B must have {n} rows, got shape {self.B.shape}")
        if self.C.shape != (1, n):
            raise ValueError(f"C must have shape (1, {n}) (scalar output), got {self.C.shape}")
        if self.Q.shape != (n, n):
            raise ValueError(f"Q must have shape ({n}, {n}), got {self.Q.shape}")
        if self.R.shape != (1, 1):
            raise ValueError(f"R must have shape (1, 1), got {self.R.shape}")
        if float(self.R[0, 0]) <= 0.0:
            raise ValueError("R must be strictly positive; a zero-variance sensor is not a twin")
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if not np.allclose(self.Q, self.Q.T, atol=1e-15):
            raise ValueError("Q must be symmetric")
        eigs = np.linalg.eigvalsh(self.Q)
        if eigs.min() < -1e-12 * max(1.0, float(eigs.max())):
            raise ValueError(f"Q must be positive semi-definite; min eigenvalue {eigs.min():.3e}")

    @property
    def n_states(self) -> int:
        """Number of states."""
        return int(self.A.shape[0])

    @property
    def n_inputs(self) -> int:
        """Number of inputs."""
        return int(self.B.shape[1])

    def steady_state(self) -> SteadyStateFilter:
        """Solve equation (5) and return the fixed-gain residual generator."""
        p = solve_discrete_are(self.A.T, self.C.T, self.Q, self.R)
        s = float((self.C @ p @ self.C.T + self.R)[0, 0])
        if s <= 0.0:
            raise ValueError(f"innovation variance must be positive, got {s:.3e}")
        k = (self.A @ p @ self.C.T) / s
        return SteadyStateFilter(twin=self, P=p, S=s, K=k)

    def with_offset(self, offset: float) -> LinearGaussianTwin:
        """Return a copy whose declared measurement offset is ``offset``."""
        return LinearGaussianTwin(
            A=self.A, B=self.B, C=self.C, Q=self.Q, R=self.R, dt=self.dt, offset=float(offset)
        )


@dataclass(frozen=True)
class SteadyStateFilter:
    """Fixed-gain steady-state Kalman filter, equations (3)-(6).

    Attributes
    ----------
    twin:
        The declared model the gain was computed from.
    P:
        Steady-state one-step prediction covariance, the solution of (5).
    S:
        Innovation variance ``C P C^T + R``, measurement units squared.
    K:
        Steady-state **predictor** gain ``A P C^T / S``, shape ``(n, 1)``.
    """

    twin: LinearGaussianTwin
    P: np.ndarray
    S: float
    K: np.ndarray

    def riccati_residual(self) -> float:
        """Max-abs residual of equation (5) at the stored ``P``.

        A number near machine precision times ``max|P|`` confirms the solver
        returned a genuine solution; used as a known-answer check in the tests.
        """
        a, c, q, r = self.twin.A, self.twin.C, self.twin.Q, self.twin.R
        p = self.P
        s = c @ p @ c.T + r
        rhs = a @ p @ a.T + q - a @ p @ c.T @ np.linalg.inv(s) @ c @ p @ a.T
        return float(np.max(np.abs(rhs - p)))

    def closed_loop(self) -> np.ndarray:
        """``A - K C``, the matrix that governs the prediction-error dynamics.

        With the predictor gain, ``x_tilde_{k+1} = (A - K C) x_tilde_k + w_k -
        K v_k``, so its spectral radius must be below 1 for the monitor to have
        a bounded residual at all.
        """
        return self.twin.A - self.K @ self.twin.C

    def spectral_radius(self) -> float:
        """Spectral radius of ``A - K C``; must be < 1 for a usable monitor."""
        return float(np.max(np.abs(np.linalg.eigvals(self.closed_loop()))))


def zoh_discretise(
    a_c: np.ndarray, b_c: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold discretisation of ``xdot = a_c x + b_c u``.

    Uses the matrix-exponential block form

        expm([[a_c, b_c], [0, 0]] * dt) = [[A, B], [0, I]],

    which is exact for a piecewise-constant input. Source: Franklin, G. F.,
    Powell, J. D. & Workman, M. L. (1998), *Digital Control of Dynamic
    Systems*, 3rd ed., Addison-Wesley (zero-order-hold equivalent).

    Parameters
    ----------
    a_c, b_c:
        Continuous-time matrices, shapes ``(n, n)`` and ``(n, m)``.
    dt:
        Sample interval in seconds, strictly positive.

    Returns
    -------
    ``(A, B)`` discrete-time matrices.
    """
    a_c = np.asarray(a_c, dtype=float)
    b_c = np.asarray(b_c, dtype=float)
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    n, m = a_c.shape[0], b_c.shape[1]
    block = np.zeros((n + m, n + m))
    block[:n, :n] = a_c
    block[:n, n:] = b_c
    e = expm(block * dt)
    return e[:n, :n].copy(), e[:n, n:].copy()


def cwna_process_noise(dt: float, sigma_a: float) -> np.ndarray:
    """Discrete process-noise covariance for continuous white-noise acceleration.

    For a double-integrator channel driven by continuous white acceleration of
    one-sided spectral density ``sigma_a**2`` (units (rad/s^2)^2 / Hz), the
    exact discrete equivalent over one sample is

        Q = sigma_a**2 * [[dt**3/3, dt**2/2], [dt**2/2, dt]].

    Source: Bar-Shalom, Y., Li, X.-R. & Kirubarajan, T. (2001), *Estimation
    with Applications to Tracking and Navigation*, Wiley (ISBN
    978-0-471-41655-5), the continuous white-noise-acceleration model. No page
    number is quoted because none was verified in this environment.

    Here it is applied to the ``[theta, theta_dot]`` pair of the reference
    attitude channel, which is a double integrator plus a spring-damper term;
    the spring-damper term changes ``A`` but not the noise injection path, so
    the expression above is the correct ``Q`` for the shipped twin.

    Parameters
    ----------
    dt:
        Sample interval in seconds.
    sigma_a:
        Acceleration noise amplitude in rad/s^2 per sqrt(Hz).

    Returns
    -------
    ``(2, 2)`` covariance, units ``[rad^2, rad^2/s; rad^2/s, rad^2/s^2]``.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if sigma_a < 0.0:
        raise ValueError(f"sigma_a must be non-negative, got {sigma_a}")
    return (sigma_a**2) * np.array(
        [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]], dtype=float
    )


REFERENCE_OMEGA_N = 2.0
"""Declared natural frequency of the reference attitude channel, rad/s."""

REFERENCE_ZETA = 0.10
"""Declared damping ratio of the reference attitude channel, dimensionless."""

REFERENCE_DT = 0.05
"""Sample interval of the reference channel, s (20 Hz)."""

REFERENCE_SIGMA_A = 0.02
"""Declared acceleration noise amplitude, rad/s^2 per sqrt(Hz)."""

REFERENCE_SIGMA_MEAS = 0.01
"""Declared measurement noise standard deviation, rad."""


def attitude_channel(
    omega_n: float = REFERENCE_OMEGA_N,
    zeta: float = REFERENCE_ZETA,
    dt: float = REFERENCE_DT,
    sigma_a: float = REFERENCE_SIGMA_A,
    sigma_meas: float = REFERENCE_SIGMA_MEAS,
    offset: float = 0.0,
) -> LinearGaussianTwin:
    """Build a single-axis attitude channel as a :class:`LinearGaussianTwin`.

    Continuous dynamics, a damped second-order attitude loop:

        theta_ddot + 2 zeta omega_n theta_dot + omega_n^2 theta = omega_n^2 u,

    i.e. ``a_c = [[0, 1], [-omega_n^2, -2 zeta omega_n]]`` and
    ``b_c = [[0], [omega_n^2]]``, discretised by :func:`zoh_discretise`. The
    measurement is ``theta`` with noise variance ``sigma_meas**2``.

    This is an illustrative channel, not any flight vehicle. Validity range:
    ``omega_n dt`` well below the Nyquist limit (``omega_n dt = 0.1`` rad for
    the defaults, so about 63 samples per natural period) and ``0 < zeta < 1``.

    Parameters
    ----------
    omega_n:
        Natural frequency, rad/s, strictly positive.
    zeta:
        Damping ratio, dimensionless, in ``(0, 1)`` for an oscillatory channel.
    dt:
        Sample interval, s.
    sigma_a:
        Acceleration noise amplitude, rad/s^2 per sqrt(Hz).
    sigma_meas:
        Measurement noise standard deviation, rad.
    offset:
        Declared measurement offset, rad.
    """
    if omega_n <= 0.0:
        raise ValueError(f"omega_n must be positive, got {omega_n}")
    if not 0.0 <= zeta < 1.0:
        raise ValueError(f"zeta must lie in [0, 1) for this channel, got {zeta}")
    if sigma_meas <= 0.0:
        raise ValueError(f"sigma_meas must be positive, got {sigma_meas}")
    a_c = np.array([[0.0, 1.0], [-(omega_n**2), -2.0 * zeta * omega_n]])
    b_c = np.array([[0.0], [omega_n**2]])
    a_d, b_d = zoh_discretise(a_c, b_c, dt)
    return LinearGaussianTwin(
        A=a_d,
        B=b_d,
        C=np.array([[1.0, 0.0]]),
        Q=cwna_process_noise(dt, sigma_a),
        R=np.array([[sigma_meas**2]]),
        dt=dt,
        offset=offset,
    )


def reference_twin() -> LinearGaussianTwin:
    """The shipped reference twin: :func:`attitude_channel` at its defaults."""
    return attitude_channel()


def reference_excitation(n_samples: int, dt: float = REFERENCE_DT) -> np.ndarray:
    """Reference excitation, shape ``(n_samples, 1)``, rad/s^2.

    A change in an asset parameter is only visible in the residual if the asset
    is being driven: with ``u = 0`` and the state at the origin, an actuator
    gain error produces no innovation at all. This package therefore ships a
    persistently exciting input,

        u_k = u0 + a_d sin(2 pi f_d k dt),   u0 = 1.0, a_d = 0.2, f_d = 0.15 Hz,

    a constant command with a small dither. The constant part is what makes an
    actuator-gain change produce a *constant* residual mean rather than an
    oscillation, which is the regime in which the CUSUM and GLR statistics of
    :mod:`twininvalidate.detectors` are the well-specified tests for the job.
    The dither is below the channel's 0.318 Hz natural frequency.

    The excitation requirement is a genuine limitation, not an implementation
    detail: on a quiescent asset this monitor has nothing to work with. It is
    stated in the README under Limitations.
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    k = np.arange(n_samples, dtype=float)
    return (1.0 + 0.2 * np.sin(2.0 * np.pi * 0.15 * k * dt)).reshape(-1, 1)
