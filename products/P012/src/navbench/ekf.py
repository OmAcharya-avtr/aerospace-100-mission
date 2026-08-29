"""Extended Kalman filter with user-supplied or numerical Jacobians.

The EKF linearises the nonlinear model about the current estimate
(Bar-Shalom, Li & Kirubarajan 2001, *Estimation with Applications to Tracking
and Navigation*, §10.3; Jazwinski 1970, *Stochastic Processes and Filtering
Theory*, §8.3):

    predict   x⁻ = f(x⁺, u)                F = ∂f/∂x |_{x⁺}
              P⁻ = F P⁺ Fᵀ + Q
    update    ν  = z − h(x⁻)               H = ∂h/∂x |_{x⁻}
              S  = H P⁻ Hᵀ + R,  K = P⁻ Hᵀ S⁻¹
              P⁺ = Joseph(P⁻, K, H, R)

VALIDITY.  First-order accuracy only.  The linearisation error is O(|x−x̂|²)
weighted by the model curvature, so the EKF degrades when the prior
uncertainty is large compared with the scale over which ``f``/``h`` curve.
That is exactly the regime probed in ``validation/v3_ukf_vs_ekf.py``.

The numerical-Jacobian fallback uses central differences with per-component
step scaling ``ε = eps^{1/3} max(|x_i|, 1)``, giving roughly ``eps^{2/3} ≈
4e-11`` relative accuracy.  It is a convenience, not a substitute for an
analytic Jacobian, and that caveat is repeated in the README.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .kf import CovarianceCollapseError, FilterResult, joseph_update, symmetrize

__all__ = ["numerical_jacobian", "ExtendedKalmanFilter"]

_CBRT_EPS = float(np.finfo(float).eps) ** (1.0 / 3.0)


def numerical_jacobian(
    fun: Callable[[NDArray[np.float64]], NDArray[np.float64]], x: ArrayLike
) -> NDArray[np.float64]:
    """Central-difference Jacobian ``∂fun/∂x`` at ``x``.

    Parameters
    ----------
    fun : callable
        Maps a length-``n`` array to a length-``m`` array.
    x : array_like, shape (n,)

    Returns
    -------
    ndarray, shape (m, n)

    Notes
    -----
    Step ``ε_i = eps^{1/3} · max(|x_i|, 1)``.  Truncation and round-off
    balance at ``eps^{2/3} ≈ 4e-11`` relative error; do not rely on this for
    tight consistency work.
    """
    x0 = np.asarray(x, dtype=float).ravel()
    if not np.all(np.isfinite(x0)):
        raise ValueError("x must be finite")
    f0 = np.asarray(fun(x0), dtype=float).ravel()
    n, m = x0.size, f0.size
    jac = np.zeros((m, n))
    for i in range(n):
        step = _CBRT_EPS * max(abs(x0[i]), 1.0)
        xp, xm = x0.copy(), x0.copy()
        xp[i] += step
        xm[i] -= step
        jac[:, i] = (
            np.asarray(fun(xp), dtype=float).ravel() - np.asarray(fun(xm), dtype=float).ravel()
        ) / (2.0 * step)
    return jac


class ExtendedKalmanFilter:
    """First-order EKF.

    Parameters
    ----------
    f_fun : callable
        ``f_fun(x, u) -> (n,)`` state propagation.  ``u`` may be ``None``.
    h_fun : callable
        ``h_fun(x) -> (m,)`` measurement prediction.
    q, r : array_like
        Process and measurement noise covariances, shapes (n, n) and (m, m).
    x0, p0 : array_like
        Initial estimate and covariance.
    f_jac, h_jac : callable, optional
        Analytic Jacobians ``f_jac(x, u) -> (n, n)`` and ``h_jac(x) -> (m, n)``.
        When omitted, :func:`numerical_jacobian` is used and
        ``uses_numerical_jacobian`` is set True.
    """

    def __init__(
        self,
        f_fun: Callable[..., NDArray[np.float64]],
        h_fun: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        q: ArrayLike,
        r: ArrayLike,
        x0: ArrayLike,
        p0: ArrayLike,
        f_jac: Callable[..., NDArray[np.float64]] | None = None,
        h_jac: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    ) -> None:
        if not callable(f_fun) or not callable(h_fun):
            raise TypeError("f_fun and h_fun must be callable")
        x = np.asarray(x0, dtype=float).ravel()
        if x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("x0 must be a non-empty finite vector")
        n = x.size
        self.q = self._psd(np.atleast_2d(np.asarray(q, dtype=float)), (n, n), "q")
        self.p = self._psd(np.atleast_2d(np.asarray(p0, dtype=float)), (n, n), "p0")
        rm = np.atleast_2d(np.asarray(r, dtype=float))
        m = rm.shape[0]
        self.r = self._psd(rm, (m, m), "r", strict=True)
        self.f_fun, self.h_fun = f_fun, h_fun
        self.f_jac, self.h_jac = f_jac, h_jac
        self.x = x.copy()
        self.n, self.m = n, m
        self.uses_numerical_jacobian = f_jac is None or h_jac is None

    @staticmethod
    def _psd(
        a: NDArray[np.float64], shape: tuple[int, int], name: str, strict: bool = False
    ) -> NDArray[np.float64]:
        if a.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be finite")
        scale = max(1.0, float(np.max(np.abs(a))))
        if float(np.max(np.abs(a - a.T))) > 1e-9 * scale:
            raise ValueError(f"{name} must be symmetric")
        eig = float(np.linalg.eigvalsh(0.5 * (a + a.T)).min())
        if strict and eig <= 0.0:
            raise ValueError(f"{name} must be positive definite; min eigenvalue {eig:.3e}")
        if not strict and eig < -1e-9 * scale:
            raise ValueError(f"{name} must be positive semi-definite; min eigenvalue {eig:.3e}")
        return symmetrize(a)

    def _fx(self, x: NDArray[np.float64], u: ArrayLike | None) -> NDArray[np.float64]:
        out = np.asarray(self.f_fun(x) if u is None else self.f_fun(x, u), dtype=float).ravel()
        if out.size != self.n:
            raise ValueError(f"f_fun must return {self.n} elements, got {out.size}")
        return out

    def _hx(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.asarray(self.h_fun(x), dtype=float).ravel()
        if out.size != self.m:
            raise ValueError(f"h_fun must return {self.m} elements, got {out.size}")
        return out

    def predict(self, u: ArrayLike | None = None, q: ArrayLike | None = None):
        """Time update; returns ``(x⁻, P⁻)``."""
        if self.f_jac is not None:
            fm = np.atleast_2d(
                np.asarray(
                    self.f_jac(self.x) if u is None else self.f_jac(self.x, u), dtype=float
                )
            )
            if fm.shape != (self.n, self.n):
                raise ValueError(f"f_jac must return shape ({self.n}, {self.n}), got {fm.shape}")
        else:
            fm = numerical_jacobian(lambda xx: self._fx(xx, u), self.x)
        qm = self.q if q is None else self._psd(
            np.atleast_2d(np.asarray(q, dtype=float)), (self.n, self.n), "q"
        )
        self.x = self._fx(self.x, u)
        self.p = symmetrize(fm @ self.p @ fm.T + qm)
        return self.x.copy(), self.p.copy()

    def update(self, z: ArrayLike, r: ArrayLike | None = None) -> dict[str, object]:
        """Measurement update; returns the same dict keys as :meth:`KalmanFilter.update`."""
        zz = np.asarray(z, dtype=float).ravel()
        if zz.size != self.m:
            raise ValueError(f"z must have {self.m} elements, got {zz.size}")
        if not np.all(np.isfinite(zz)):
            raise ValueError("z must be finite")
        rm = self.r if r is None else self._psd(
            np.atleast_2d(np.asarray(r, dtype=float)), (self.m, self.m), "r", strict=True
        )
        if self.h_jac is not None:
            hm = np.atleast_2d(np.asarray(self.h_jac(self.x), dtype=float))
            if hm.shape != (self.m, self.n):
                raise ValueError(f"h_jac must return shape ({self.m}, {self.n}), got {hm.shape}")
        else:
            hm = numerical_jacobian(self._hx, self.x)
        nu = zz - self._hx(self.x)
        s = symmetrize(hm @ self.p @ hm.T + rm)
        try:
            np.linalg.cholesky(s)
        except np.linalg.LinAlgError as exc:
            raise CovarianceCollapseError(
                f"innovation covariance S is not positive definite "
                f"(min eigenvalue {float(np.linalg.eigvalsh(s).min()):.3e})"
            ) from exc
        gain = np.linalg.solve(s, (self.p @ hm.T).T).T
        self.x = self.x + gain @ nu
        self.p = joseph_update(self.p, gain, hm, rm)
        return {
            "x": self.x.copy(),
            "p": self.p.copy(),
            "innovation": nu,
            "innovation_cov": s,
            "gain": gain,
            "nis": float(nu @ np.linalg.solve(s, nu)),
        }

    def run(self, measurements: ArrayLike, controls: ArrayLike | None = None) -> FilterResult:
        """Batch run.  Rows of ``measurements`` containing NaN are skipped."""
        z = np.atleast_2d(np.asarray(measurements, dtype=float))
        if z.ndim != 2 or z.shape[1] != self.m:
            raise ValueError(f"measurements must have shape (N, {self.m}), got {z.shape}")
        n_steps = z.shape[0]
        avail = np.all(np.isfinite(z), axis=1)
        u_all = None if controls is None else np.atleast_2d(np.asarray(controls, dtype=float))
        xp = np.zeros((n_steps, self.n))
        xu = np.zeros((n_steps, self.n))
        pp = np.zeros((n_steps, self.n, self.n))
        pu = np.zeros((n_steps, self.n, self.n))
        nu = np.full((n_steps, self.m), np.nan)
        ss = np.zeros((n_steps, self.m, self.m))
        kk = np.zeros((n_steps, self.n, self.m))
        nis = np.full(n_steps, np.nan)
        for k in range(n_steps):
            u = None if u_all is None else u_all[k]
            xp[k], pp[k] = self.predict(u=u)
            if avail[k]:
                out = self.update(z[k])
                nu[k] = out["innovation"]  # type: ignore[assignment]
                ss[k] = out["innovation_cov"]  # type: ignore[assignment]
                kk[k] = out["gain"]  # type: ignore[assignment]
                nis[k] = out["nis"]  # type: ignore[assignment]
            else:
                hm = (
                    np.atleast_2d(np.asarray(self.h_jac(self.x), dtype=float))
                    if self.h_jac is not None
                    else numerical_jacobian(self._hx, self.x)
                )
                ss[k] = symmetrize(hm @ self.p @ hm.T + self.r)
            xu[k], pu[k] = self.x.copy(), self.p.copy()
        return FilterResult(xp, xu, pp, pu, nu, ss, kk, nis, avail)
