r"""Extended Kalman filter.

The EKF linearises the nonlinear dynamics and measurement about the current
estimate:

.. math::
    x_k = f(x_{k-1}, u) + w,\quad z_k = h(x_k) + v, \\
    F_k = \left.\frac{\partial f}{\partial x}\right|_{\hat x^{+}_{k-1}},\qquad
    H_k = \left.\frac{\partial h}{\partial x}\right|_{\hat x^{-}_{k}}

and then runs the linear recursion with ``F_k``, ``H_k``.

Source: Bar-Shalom, Rong Li & Kirubarajan (2001), *Estimation with Applications
to Tracking and Navigation*, §10.3; Simon (2006), *Optimal State Estimation*,
Ch. 13.

Validity and honesty
--------------------
The EKF is a **first-order** approximation. It is consistent only while the
posterior is well approximated by a Gaussian over the region where ``f`` and
``h`` are close to affine — quantitatively, while the second-order term
``½ tr(∂²h/∂x² P)`` is small compared with the measurement noise standard
deviation. NavBench measures the resulting inconsistency directly (see
:mod:`navbench.consistency` and ``validation/v3_ukf_vs_ekf.py``) rather than
asserting it.

Jacobians may be supplied analytically or left to the central-difference
fallback. The fallback has truncation error ``O(h²)`` and round-off error
``O(ε/h)``; with the default relative step ``h = εₘ^{1/3}·max(|x_i|, 1)`` the
attainable accuracy is about ``εₘ^{2/3} ≈ 4e-11`` relative — enough for a
sanity check, **not** enough for a tightly tuned filter. Numerical Jacobians
are opt-in and reported in the filter's ``uses_numeric_jacobian`` flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .kf import UpdateInfo
from .linalg import joseph_update, symmetrize

__all__ = ["ExtendedKalmanFilter", "numerical_jacobian"]

_CBRT_EPS = float(np.finfo(float).eps) ** (1.0 / 3.0)


def numerical_jacobian(
    fun: Callable[[NDArray[np.float64]], NDArray[np.float64]], x: ArrayLike
) -> NDArray[np.float64]:
    r"""Central-difference Jacobian ``∂fun/∂x`` at ``x``.

    Step ``h_i = εₘ^{1/3} max(|x_i|, 1)``, which balances the ``O(h²)``
    truncation error against the ``O(ε/h)`` cancellation error; the resulting
    accuracy is ``O(εₘ^{2/3}) ≈ 4×10⁻¹¹`` relative (Press et al., *Numerical
    Recipes*, 3rd ed., §5.7).
    """
    x0 = np.asarray(x, dtype=float).reshape(-1)
    f0 = np.asarray(fun(x0), dtype=float).reshape(-1)
    jac = np.zeros((f0.size, x0.size))
    for i in range(x0.size):
        h = _CBRT_EPS * max(abs(x0[i]), 1.0)
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        jac[:, i] = (
            np.asarray(fun(xp), dtype=float).reshape(-1)
            - np.asarray(fun(xm), dtype=float).reshape(-1)
        ) / (2.0 * h)
    return jac


@dataclass
class ExtendedKalmanFilter:
    """First-order extended Kalman filter with Joseph-form update.

    Parameters
    ----------
    f : callable ``(x, dt) -> x``
        Nonlinear state propagation.
    h : callable ``(x) -> z``
        Nonlinear measurement function.
    q : array_like, shape (n, n)
        Process-noise covariance for one step of length ``dt``.
    r : array_like, shape (m, m)
        Measurement-noise covariance.
    x : array_like, shape (n,)
        Initial state.
    p : array_like, shape (n, n)
        Initial covariance.
    f_jac : callable ``(x, dt) -> (n, n)`` or None
        Analytic ``∂f/∂x``. ``None`` selects the numerical fallback.
    h_jac : callable ``(x) -> (m, n)`` or None
        Analytic ``∂h/∂x``. ``None`` selects the numerical fallback.
    """

    f: Callable[[NDArray[np.float64], float], NDArray[np.float64]]
    h: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    q: NDArray[np.float64]
    r: NDArray[np.float64]
    x: NDArray[np.float64]
    p: NDArray[np.float64]
    f_jac: Callable[[NDArray[np.float64], float], NDArray[np.float64]] | None = None
    h_jac: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None
    history: list[UpdateInfo] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float).reshape(-1)
        n = self.x.size
        self.q = symmetrize(np.atleast_2d(np.asarray(self.q, dtype=float)))
        self.p = symmetrize(np.atleast_2d(np.asarray(self.p, dtype=float)))
        self.r = symmetrize(np.atleast_2d(np.asarray(self.r, dtype=float)))
        if self.q.shape != (n, n) or self.p.shape != (n, n):
            raise ValueError(f"q and p must be {n}x{n}, got {self.q.shape} and {self.p.shape}")
        if not callable(self.f) or not callable(self.h):
            raise TypeError("f and h must be callables")

    @property
    def uses_numeric_jacobian(self) -> bool:
        """True if either Jacobian falls back to central differences."""
        return self.f_jac is None or self.h_jac is None

    @property
    def n(self) -> int:
        """State dimension."""
        return int(self.x.size)

    def predict(self, dt: float = 1.0, q: ArrayLike | None = None) -> None:
        """Time update over ``dt`` [s]; ``q`` overrides the stored process noise."""
        qk = self.q if q is None else symmetrize(np.atleast_2d(np.asarray(q, dtype=float)))
        fj = (
            self.f_jac(self.x, dt)
            if self.f_jac is not None
            else numerical_jacobian(lambda v: self.f(v, dt), self.x)
        )
        fj = np.atleast_2d(np.asarray(fj, dtype=float))
        self.x = np.asarray(self.f(self.x, dt), dtype=float).reshape(-1)
        self.p = symmetrize(fj @ self.p @ fj.T + qk)

    def update(self, z: ArrayLike, r: ArrayLike | None = None) -> UpdateInfo:
        """Measurement update; returns innovation diagnostics."""
        zz = np.asarray(z, dtype=float).reshape(-1)
        rk = self.r if r is None else symmetrize(np.atleast_2d(np.asarray(r, dtype=float)))
        hj = (
            self.h_jac(self.x) if self.h_jac is not None else numerical_jacobian(self.h, self.x)
        )
        hj = np.atleast_2d(np.asarray(hj, dtype=float))
        zhat = np.asarray(self.h(self.x), dtype=float).reshape(-1)
        if zhat.size != zz.size:
            raise ValueError(f"h(x) returns size {zhat.size} but z has size {zz.size}")
        nu = zz - zhat
        s = symmetrize(hj @ self.p @ hj.T + rk)
        k = np.linalg.solve(s.T, (self.p @ hj.T).T).T
        self.x = self.x + k @ nu
        self.p = joseph_update(self.p, k, hj, rk)
        lo = np.linalg.cholesky(s)
        y = np.linalg.solve(lo, nu)
        info = UpdateInfo(innovation=nu, innovation_cov=s, gain=k, nis=float(np.dot(y, y)))
        self.history.append(info)
        return info
