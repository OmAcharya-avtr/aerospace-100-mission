r"""Linear Kalman filter with the Joseph-form covariance update.

Model (discrete time, Bar-Shalom, Rong Li & Kirubarajan 2001, Ch. 5):

.. math::
    x_k &= F x_{k-1} + B u_{k-1} + w_{k-1}, \quad w \sim N(0, Q) \\
    z_k &= H x_k + v_k, \quad v \sim N(0, R)

with ``w``, ``v`` zero mean, white, mutually uncorrelated.

Recursion::

    x⁻ = F x⁺ + B u                    P⁻ = F P⁺ Fᵀ + Q
    ν  = z − H x⁻                      S  = H P⁻ Hᵀ + R
    K  = P⁻ Hᵀ S⁻¹                     x⁺ = x⁻ + K ν
    P⁺ = (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ

Units: caller's, used consistently. ``P`` carries squared state units, ``R``
squared measurement units, ``K`` state per measurement unit.

Validity: minimum-mean-square-error optimal only when the model is linear and
the noise statistics are exactly as declared. Under non-Gaussian noise it
remains the best *linear* unbiased estimator. Correlated process/measurement
noise and time-correlated biases require state augmentation and are not handled
here.

Related work: product **P017 EstimKit** is a compact standalone filter library
covering the same linear recursion plus an RTS smoother. NavBench implements
its filters independently because the bench needs a uniform driver interface,
per-step innovation bookkeeping and a pluggable process-noise adapter that a
general-purpose library does not expose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import joseph_update, symmetrize

__all__ = ["UpdateInfo", "KalmanFilter", "steady_state_riccati"]


@dataclass(frozen=True)
class UpdateInfo:
    """Per-update diagnostics.

    Attributes
    ----------
    innovation : ndarray, shape (m,)
        ``ν = z − ẑ`` in measurement units.
    innovation_cov : ndarray, shape (m, m)
        ``S``, squared measurement units.
    gain : ndarray, shape (n, m)
        Kalman gain ``K``.
    nis : float
        Normalised innovation squared ``νᵀ S⁻¹ ν`` (dimensionless).
    """

    innovation: NDArray[np.float64]
    innovation_cov: NDArray[np.float64]
    gain: NDArray[np.float64]
    nis: float


def _as_square(a: ArrayLike, n: int | None, name: str) -> NDArray[np.float64]:
    arr = np.atleast_2d(np.asarray(a, dtype=float))
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square, got shape {arr.shape}")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} must be {n}x{n} to match the state, got {arr.shape}")
    return arr


@dataclass
class KalmanFilter:
    """Discrete-time linear Kalman filter.

    Parameters
    ----------
    f : array_like, shape (n, n)
        State transition matrix.
    q : array_like, shape (n, n)
        Process-noise covariance (squared state units).
    h : array_like, shape (m, n)
        Measurement matrix.
    r : array_like, shape (m, m)
        Measurement-noise covariance (squared measurement units).
    x : array_like, shape (n,)
        Initial state estimate.
    p : array_like, shape (n, n)
        Initial state covariance.
    b : array_like, shape (n, p) or None
        Optional control-input matrix.
    """

    f: NDArray[np.float64]
    q: NDArray[np.float64]
    h: NDArray[np.float64]
    r: NDArray[np.float64]
    x: NDArray[np.float64]
    p: NDArray[np.float64]
    b: NDArray[np.float64] | None = None
    history: list[UpdateInfo] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float).reshape(-1)
        n = self.x.size
        self.f = _as_square(self.f, n, "f")
        self.q = symmetrize(_as_square(self.q, n, "q"))
        self.p = symmetrize(_as_square(self.p, n, "p"))
        self.h = np.atleast_2d(np.asarray(self.h, dtype=float))
        if self.h.shape[1] != n:
            raise ValueError(f"h must have {n} columns to match the state, got {self.h.shape}")
        self.r = symmetrize(_as_square(self.r, self.h.shape[0], "r"))
        if self.b is not None:
            self.b = np.atleast_2d(np.asarray(self.b, dtype=float))
            if self.b.shape[0] != n:
                raise ValueError(f"b must have {n} rows, got {self.b.shape}")
        if np.linalg.eigvalsh(self.p).min() <= 0.0:
            raise ValueError("initial covariance p must be positive definite")

    @property
    def n(self) -> int:
        """State dimension."""
        return int(self.x.size)

    @property
    def m(self) -> int:
        """Measurement dimension."""
        return int(self.h.shape[0])

    def predict(self, u: ArrayLike | None = None, q: ArrayLike | None = None) -> None:
        """Time update. ``q`` overrides the stored process noise for this step."""
        qk = self.q if q is None else symmetrize(_as_square(q, self.n, "q"))
        self.x = self.f @ self.x
        if u is not None:
            if self.b is None:
                raise ValueError("a control input u was supplied but b is None")
            self.x = self.x + self.b @ np.asarray(u, dtype=float).reshape(-1)
        self.p = symmetrize(self.f @ self.p @ self.f.T + qk)

    def update(self, z: ArrayLike, r: ArrayLike | None = None) -> UpdateInfo:
        """Measurement update; returns the innovation diagnostics."""
        zz = np.asarray(z, dtype=float).reshape(-1)
        if zz.size != self.m:
            raise ValueError(f"measurement has size {zz.size}, expected {self.m}")
        rk = self.r if r is None else symmetrize(_as_square(r, self.m, "r"))
        nu = zz - self.h @ self.x
        s = symmetrize(self.h @ self.p @ self.h.T + rk)
        k = np.linalg.solve(s.T, (self.p @ self.h.T).T).T
        self.x = self.x + k @ nu
        self.p = joseph_update(self.p, k, self.h, rk)
        lo = np.linalg.cholesky(s)
        y = np.linalg.solve(lo, nu)
        info = UpdateInfo(innovation=nu, innovation_cov=s, gain=k, nis=float(np.dot(y, y)))
        self.history.append(info)
        return info


def steady_state_riccati(
    f: ArrayLike, q: ArrayLike, h: ArrayLike, r: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Analytic steady-state solution of the discrete filtering Riccati equation.

    Solves the *prediction* (a priori) algebraic Riccati equation

    .. math::
        P^- = F\left[P^- - P^-H^\mathsf{T}(HP^-H^\mathsf{T}+R)^{-1}HP^-\right]F^\mathsf{T} + Q

    using :func:`scipy.linalg.solve_discrete_are`, and returns
    ``(P_prior, P_post, K)`` with

    .. math::
        K = P^-H^\mathsf{T}(HP^-H^\mathsf{T}+R)^{-1}, \qquad P^+ = (I-KH)P^-

    Source: Bar-Shalom, Rong Li & Kirubarajan (2001), §5.2.6; Anderson, B. D. O.
    and Moore, J. B. (1979), *Optimal Filtering*, Prentice-Hall, Ch. 4.

    Note on the SciPy convention: ``solve_discrete_are(a, b, q, r)`` solves the
    *control* DARE. The filtering DARE is its dual, obtained with
    ``a = Fᵀ``, ``b = Hᵀ``.

    Validity: requires ``(F, H)`` detectable and ``(F, G)`` stabilisable with
    ``Q = GGᵀ``; otherwise no stabilising solution exists and SciPy raises.
    """
    from scipy.linalg import solve_discrete_are

    ff = np.atleast_2d(np.asarray(f, dtype=float))
    qq = symmetrize(np.atleast_2d(np.asarray(q, dtype=float)))
    hh = np.atleast_2d(np.asarray(h, dtype=float))
    rr = symmetrize(np.atleast_2d(np.asarray(r, dtype=float)))
    p_prior = symmetrize(solve_discrete_are(ff.T, hh.T, qq, rr))
    s = symmetrize(hh @ p_prior @ hh.T + rr)
    k = np.linalg.solve(s.T, (p_prior @ hh.T).T).T
    p_post = symmetrize((np.eye(ff.shape[0]) - k @ hh) @ p_prior)
    return p_prior, p_post, k
