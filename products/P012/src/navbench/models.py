r"""Canonical navigation/tracking models used by the bench and the validation.

Three models, each chosen because it isolates one question.

1. :class:`ConstantVelocity` — linear time-invariant
---------------------------------------------------
State ``[p, v]`` per axis (``n = 2d``), discretised continuous white-noise
acceleration (the "DWNA"/"CWNA" model):

.. math::
    F = I_d \otimes \begin{bmatrix}1 & \Delta t\\ 0 & 1\end{bmatrix},\qquad
    Q = q\, I_d \otimes
        \begin{bmatrix}\Delta t^3/3 & \Delta t^2/2\\ \Delta t^2/2 & \Delta t\end{bmatrix}

with ``q`` the acceleration power spectral density [m²/s³]. Position-only
measurement ``H = [I_d ⊗ (1, 0)]``, ``R = σ_p² I_d`` [m²].

Source: Bar-Shalom, Rong Li & Kirubarajan (2001), *Estimation with Applications
to Tracking and Navigation*, §6.2 (continuous white noise acceleration model).
Assumption: the target's acceleration is a zero-mean white process; validity:
manoeuvres slower than the sampling interval, i.e. genuinely non-manoeuvring or
weakly manoeuvring targets. This is the model used for the analytic Riccati
check because it is time-invariant and observable.

2. :class:`RangeBearing` — mildly nonlinear measurement
-------------------------------------------------------
.. math::
    h(x) = \begin{bmatrix}\sqrt{(p_x-s_x)^2+(p_y-s_y)^2}\\
           \operatorname{atan2}(p_y-s_y,\ p_x-s_x)\end{bmatrix}

The measurement nonlinearity is governed by the ratio of the position
uncertainty to the range: the second-order term in the bearing expansion is
``O(σ_p²/r²)``. At ``r = 50 km`` with ``σ_p = 100 m`` that is ``4×10⁻⁶`` — the
"near-linear" regime. Bringing the sensor to ``r = 300 m`` with the same
uncertainty makes it ``O(0.1)`` — strongly nonlinear. Source: Bar-Shalom, Li &
Kirubarajan (2001), §10.3 (converted measurements and the EKF for
range/bearing).

3. :class:`UnivariateGrowth` — strongly nonlinear benchmark
------------------------------------------------------------
.. math::
    x_k &= 0.5x_{k-1} + \frac{25x_{k-1}}{1+x_{k-1}^2} + 8\cos(1.2(k-1)) + w_{k-1} \\
    z_k &= x_k^2/20 + v_k,\qquad w\sim N(0,10),\ v\sim N(0,1)

This is the standard "univariate non-stationary growth model" of Kitagawa, G.
(1987), "Non-Gaussian State-Space Modeling of Nonstationary Time Series",
*Journal of the American Statistical Association* **82**(400), 1032–1041, used
as the reference nonlinear benchmark by Gordon, N. J., Salmond, D. J. and
Smith, A. F. M. (1993), "Novel approach to nonlinear/non-Gaussian Bayesian
state estimation", *IEE Proceedings-F* **140**(2), 107–113.

It is deliberately brutal for Gaussian filters: the measurement is **even**, so
the exact posterior is bimodal, and ``∂h/∂x = x/10`` vanishes at ``x = 0``,
where the EKF's gain collapses to zero. NavBench uses it to show what
"degrades more gracefully" means quantitatively — **not** to claim that any
Gaussian filter is adequate here.

Units: SI throughout for models 1–2 (m, m/s, s, rad); model 3 is
dimensionless by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["ConstantVelocity", "RangeBearing", "UnivariateGrowth", "cv_transition", "cv_process_noise"]


def cv_transition(dt: float, dim: int = 2) -> NDArray[np.float64]:
    """``F`` for a ``dim``-axis constant-velocity model, state ordered ``[p, v]`` per axis."""
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    blk = np.array([[1.0, dt], [0.0, 1.0]])
    return np.kron(np.eye(dim), blk)


def cv_process_noise(dt: float, q: float, dim: int = 2) -> NDArray[np.float64]:
    """``Q`` for the continuous white-noise-acceleration model, PSD ``q`` [m²/s³]."""
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if q < 0.0:
        raise ValueError(f"q (acceleration PSD) must be >= 0, got {q}")
    blk = q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
    return np.kron(np.eye(dim), blk)


@dataclass(frozen=True)
class ConstantVelocity:
    """Linear time-invariant constant-velocity model with position measurements.

    Parameters
    ----------
    dt : float
        Sample interval [s].
    q_psd : float
        Acceleration power spectral density [m²/s³].
    sigma_pos : float
        Per-axis position measurement noise 1-σ [m].
    dim : int
        Number of spatial axes (1, 2 or 3).
    """

    dt: float = 1.0
    q_psd: float = 0.1
    sigma_pos: float = 5.0
    dim: int = 2

    def __post_init__(self) -> None:
        if self.dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2 or 3, got {self.dim}")
        if self.sigma_pos <= 0.0:
            raise ValueError(f"sigma_pos must be > 0, got {self.sigma_pos}")

    @property
    def n(self) -> int:
        """State dimension ``2·dim``."""
        return 2 * self.dim

    @property
    def m(self) -> int:
        """Measurement dimension ``dim``."""
        return self.dim

    def f(self) -> NDArray[np.float64]:
        """State transition matrix."""
        return cv_transition(self.dt, self.dim)

    def q(self, scale: float = 1.0) -> NDArray[np.float64]:
        """Process-noise covariance, optionally scaled (used by the adapters)."""
        if scale <= 0.0:
            raise ValueError(f"scale must be > 0, got {scale}")
        return cv_process_noise(self.dt, self.q_psd * scale, self.dim)

    def h(self) -> NDArray[np.float64]:
        """Measurement matrix (position only)."""
        return np.kron(np.eye(self.dim), np.array([[1.0, 0.0]]))

    def r(self) -> NDArray[np.float64]:
        """Measurement-noise covariance [m²]."""
        return self.sigma_pos ** 2 * np.eye(self.dim)

    def simulate(
        self, x0: ArrayLike, n_steps: int, rng: np.random.Generator, q_true_scale: float = 1.0
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Simulate truth and measurements.

        Returns ``(states, measurements)`` of shapes ``(n_steps+1, n)`` and
        ``(n_steps, m)``; measurement ``k`` corresponds to state ``k+1``.
        """
        f = self.f()
        qm = self.q(q_true_scale)
        hh = self.h()
        rr = self.r()
        lq = np.linalg.cholesky(qm + 1e-18 * np.eye(self.n))
        lr = np.linalg.cholesky(rr)
        x = np.asarray(x0, dtype=float).reshape(-1)
        if x.size != self.n:
            raise ValueError(f"x0 must have {self.n} components, got {x.size}")
        xs = np.zeros((n_steps + 1, self.n))
        zs = np.zeros((n_steps, self.m))
        xs[0] = x
        for k in range(n_steps):
            x = f @ x + lq @ rng.standard_normal(self.n)
            xs[k + 1] = x
            zs[k] = hh @ x + lr @ rng.standard_normal(self.m)
        return xs, zs


@dataclass(frozen=True)
class RangeBearing:
    """Range/bearing measurement of a 2-D constant-velocity target.

    Parameters
    ----------
    sensor : tuple of float
        Sensor position ``(s_x, s_y)`` [m].
    sigma_range : float
        Range noise 1-σ [m].
    sigma_bearing : float
        Bearing noise 1-σ [rad].
    """

    sensor: tuple[float, float] = (0.0, 0.0)
    sigma_range: float = 20.0
    sigma_bearing: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.sigma_range <= 0.0 or self.sigma_bearing <= 0.0:
            raise ValueError("sigma_range and sigma_bearing must both be > 0")

    @property
    def m(self) -> int:
        """Measurement dimension (2)."""
        return 2

    def r(self) -> NDArray[np.float64]:
        """Measurement-noise covariance ``diag(σ_r², σ_θ²)``."""
        return np.diag([self.sigma_range ** 2, self.sigma_bearing ** 2])

    def h(self, x: ArrayLike) -> NDArray[np.float64]:
        """Measurement function for state ordered ``[px, vx, py, vy]``."""
        s = np.asarray(x, dtype=float).reshape(-1)
        dx = s[0] - self.sensor[0]
        dy = s[2] - self.sensor[1]
        rng = float(np.hypot(dx, dy))
        if rng < 1e-9:
            raise ValueError("range/bearing measurement is singular at zero range")
        return np.array([rng, float(np.arctan2(dy, dx))])

    def h_jac(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""Analytic Jacobian ``∂h/∂x``.

        ``∂r/∂p = (dx, dy)/r`` and ``∂θ/∂p = (−dy, dx)/r²``.
        """
        s = np.asarray(x, dtype=float).reshape(-1)
        dx = s[0] - self.sensor[0]
        dy = s[2] - self.sensor[1]
        r2 = dx * dx + dy * dy
        r = float(np.sqrt(r2))
        if r < 1e-9:
            raise ValueError("range/bearing Jacobian is singular at zero range")
        j = np.zeros((2, s.size))
        j[0, 0] = dx / r
        j[0, 2] = dy / r
        j[1, 0] = -dy / r2
        j[1, 2] = dx / r2
        return j

    def simulate(
        self, states: ArrayLike, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Noisy range/bearing measurements for a state history, shape ``(K, 2)``."""
        xs = np.atleast_2d(np.asarray(states, dtype=float))
        lr = np.linalg.cholesky(self.r())
        return np.array([self.h(x) + lr @ rng.standard_normal(2) for x in xs])


@dataclass(frozen=True)
class UnivariateGrowth:
    """Kitagawa (1987) univariate non-stationary growth model.

    Parameters
    ----------
    q : float
        Process-noise variance (standard value 10).
    r : float
        Measurement-noise variance (standard value 1).
    """

    q: float = 10.0
    r: float = 1.0

    def __post_init__(self) -> None:
        if self.q <= 0.0 or self.r <= 0.0:
            raise ValueError("q and r must both be > 0")

    @property
    def n(self) -> int:
        """State dimension (1)."""
        return 1

    @property
    def m(self) -> int:
        """Measurement dimension (1)."""
        return 1

    def f(self, x: ArrayLike, k: int) -> NDArray[np.float64]:
        """One-step propagation from step ``k`` to ``k+1`` (deterministic part)."""
        s = np.asarray(x, dtype=float).reshape(-1)
        return 0.5 * s + 25.0 * s / (1.0 + s * s) + 8.0 * np.cos(1.2 * k)

    def f_jac(self, x: ArrayLike, k: int) -> NDArray[np.float64]:
        r"""``∂f/∂x = 0.5 + 25(1 − x²)/(1 + x²)²`` (independent of ``k``)."""
        del k
        s = float(np.asarray(x, dtype=float).reshape(-1)[0])
        d = 1.0 + s * s
        return np.array([[0.5 + 25.0 * (1.0 - s * s) / (d * d)]])

    def h(self, x: ArrayLike) -> NDArray[np.float64]:
        """``z = x²/20``."""
        s = np.asarray(x, dtype=float).reshape(-1)
        return s * s / 20.0

    def h_jac(self, x: ArrayLike) -> NDArray[np.float64]:
        """``∂h/∂x = x/10``; **zero at x = 0**, where the EKF gain vanishes."""
        s = float(np.asarray(x, dtype=float).reshape(-1)[0])
        return np.array([[s / 10.0]])

    def simulate(
        self, x0: float, n_steps: int, rng: np.random.Generator
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Simulate truth and measurements; shapes ``(n_steps+1, 1)`` and ``(n_steps, 1)``."""
        xs = np.zeros((n_steps + 1, 1))
        zs = np.zeros((n_steps, 1))
        xs[0, 0] = x0
        for k in range(n_steps):
            xs[k + 1] = self.f(xs[k], k) + np.sqrt(self.q) * rng.standard_normal(1)
            zs[k] = self.h(xs[k + 1]) + np.sqrt(self.r) * rng.standard_normal(1)
        return xs, zs
