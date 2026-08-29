"""Unscented Kalman filter with scaled symmetric sigma points.

References
----------
* Julier, S. J. & Uhlmann, J. K. (1997), "A New Extension of the Kalman
  Filter to Nonlinear Systems", *Proc. SPIE 3068*, Signal Processing, Sensor
  Fusion and Target Recognition VI, 182-193 — the original unscented
  transform.
* Julier, S. J. & Uhlmann, J. K. (2004), "Unscented Filtering and Nonlinear
  Estimation", *Proceedings of the IEEE* 92(3), 401-422 — the review, and the
  scaled form.
* Wan, E. A. & van der Merwe, R. (2000), "The Unscented Kalman Filter for
  Nonlinear Estimation", *IEEE AS-SPCC*, 153-158 — the α/β/κ parameterisation
  used here.

SIGMA POINTS.  With ``n`` states and ``λ = α²(n+κ) − n``:

    χ₀ = x̄,  χ_i = x̄ ± [ sqrt((n+λ) P) ]_i        i = 1…n
    W₀ᵐ = λ/(n+λ)                W₀ᶜ = λ/(n+λ) + (1 − α² + β)
    W_iᵐ = W_iᶜ = 1/(2(n+λ))

``sqrt(·)`` is the lower Cholesky factor, taken **column-wise**.  The weights
sum to 1 for the mean; ``W₀ᶜ`` may be negative, which is admissible for the
transform but means the reconstructed covariance is not guaranteed positive
definite for pathological inputs — a documented failure mode, surfaced as
:class:`~navbench.kf.CovarianceCollapseError`.

ACCURACY.  The transform captures the posterior mean and covariance to second
order for any nonlinearity (third order for Gaussian inputs), versus first
order for the EKF; it requires no Jacobians.  On a **linear** map it is exact
in exact arithmetic — the reduction to the linear KF verified in
``tests/test_ukf.py`` and in P017's validation.  Small α costs significant
digits: cancellation entering the sigma points at ``eps·|x|`` is amplified by
roughly ``1/α²``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .kf import CovarianceCollapseError, FilterResult, symmetrize

__all__ = ["MerweSigmaPoints", "unscented_transform", "UnscentedKalmanFilter"]


@dataclass(frozen=True)
class MerweSigmaPoints:
    """Scaled symmetric sigma-point set (Wan & van der Merwe 2000).

    Parameters
    ----------
    n : int
        State dimension, ≥ 1.
    alpha : float
        Spread parameter in ``(0, 1]`` (typical 1e-3 … 1).
    beta : float
        Prior-distribution parameter; ``β = 2`` is optimal for Gaussians.
    kappa : float
        Secondary scaling; requires ``n + κ > 0``.  ``κ = 0`` or ``3 − n``.
    """

    n: int
    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0

    def __post_init__(self) -> None:
        if int(self.n) != self.n or self.n < 1:
            raise ValueError(f"n must be an integer >= 1, got {self.n!r}")
        for name, val in (("alpha", self.alpha), ("beta", self.beta), ("kappa", self.kappa)):
            if not np.isfinite(float(val)):
                raise ValueError(f"{name} must be finite, got {val!r}")
        if not (0.0 < float(self.alpha) <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha!r}")
        if self.n + float(self.kappa) <= 0.0:
            raise ValueError(f"n + kappa must be > 0, got {self.n} + {self.kappa}")

    @property
    def lambda_(self) -> float:
        """``λ = α²(n+κ) − n``."""
        return float(self.alpha) ** 2 * (self.n + float(self.kappa)) - self.n

    @property
    def num_points(self) -> int:
        """``2n + 1``."""
        return 2 * self.n + 1

    def weights(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(Wm, Wc)``, each of length ``2n+1``."""
        lam = self.lambda_
        denom = self.n + lam
        wm = np.full(self.num_points, 1.0 / (2.0 * denom))
        wc = wm.copy()
        wm[0] = lam / denom
        wc[0] = lam / denom + (1.0 - float(self.alpha) ** 2 + float(self.beta))
        return wm, wc

    def sigma_points(self, x: ArrayLike, p: ArrayLike) -> NDArray[np.float64]:
        """Sigma points for mean ``x`` and covariance ``p``; shape ``(2n+1, n)``."""
        xm = np.asarray(x, dtype=float).ravel()
        if xm.size != self.n:
            raise ValueError(f"x must have {self.n} elements, got {xm.size}")
        pm = np.atleast_2d(np.asarray(p, dtype=float))
        if pm.shape != (self.n, self.n):
            raise ValueError(f"p must have shape ({self.n}, {self.n}), got {pm.shape}")
        if not (np.all(np.isfinite(xm)) and np.all(np.isfinite(pm))):
            raise ValueError("x and p must be finite")
        scaled = (self.n + self.lambda_) * symmetrize(pm)
        try:
            root = np.linalg.cholesky(scaled)
        except np.linalg.LinAlgError as exc:
            raise CovarianceCollapseError(
                "Cholesky factorisation of (n+lambda) P failed: covariance has lost "
                f"positive definiteness (min eigenvalue "
                f"{float(np.linalg.eigvalsh(symmetrize(pm)).min()):.3e})"
            ) from exc
        pts = np.zeros((self.num_points, self.n))
        pts[0] = xm
        for i in range(self.n):
            pts[1 + i] = xm + root[:, i]
            pts[1 + self.n + i] = xm - root[:, i]
        return pts


def unscented_transform(
    points: ArrayLike,
    wm: ArrayLike,
    wc: ArrayLike,
    noise_cov: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Weighted mean and covariance of a transformed sigma-point set.

    Returns ``(mean, cov)`` with ``cov = Σ Wᶜ_i (y_i − ȳ)(y_i − ȳ)ᵀ + noise_cov``.
    """
    y = np.atleast_2d(np.asarray(points, dtype=float))
    w_m = np.asarray(wm, dtype=float).ravel()
    w_c = np.asarray(wc, dtype=float).ravel()
    if y.shape[0] != w_m.size or y.shape[0] != w_c.size:
        raise ValueError(
            f"points has {y.shape[0]} rows but wm/wc have {w_m.size}/{w_c.size} entries"
        )
    mean = w_m @ y
    d = y - mean
    cov = (d * w_c[:, None]).T @ d
    if noise_cov is not None:
        nc = np.atleast_2d(np.asarray(noise_cov, dtype=float))
        if nc.shape != cov.shape:
            raise ValueError(f"noise_cov must have shape {cov.shape}, got {nc.shape}")
        cov = cov + nc
    return mean, symmetrize(cov)


class UnscentedKalmanFilter:
    """Additive-noise UKF.

    Parameters
    ----------
    f_fun : callable
        ``f_fun(x, u) -> (n,)``; ``u`` may be ``None``.
    h_fun : callable
        ``h_fun(x) -> (m,)``.
    q, r, x0, p0 : array_like
        As for :class:`~navbench.ekf.ExtendedKalmanFilter`.
    alpha, beta, kappa : float
        Sigma-point parameters (see :class:`MerweSigmaPoints`).
    """

    def __init__(
        self,
        f_fun: Callable[..., NDArray[np.float64]],
        h_fun: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        q: ArrayLike,
        r: ArrayLike,
        x0: ArrayLike,
        p0: ArrayLike,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        if not callable(f_fun) or not callable(h_fun):
            raise TypeError("f_fun and h_fun must be callable")
        x = np.asarray(x0, dtype=float).ravel()
        if x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("x0 must be a non-empty finite vector")
        n = x.size
        qm = np.atleast_2d(np.asarray(q, dtype=float))
        pm = np.atleast_2d(np.asarray(p0, dtype=float))
        rm = np.atleast_2d(np.asarray(r, dtype=float))
        for name, mat, shape in (("q", qm, (n, n)), ("p0", pm, (n, n))):
            if mat.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {mat.shape}")
            if not np.all(np.isfinite(mat)):
                raise ValueError(f"{name} must be finite")
        m = rm.shape[0]
        if rm.shape != (m, m) or not np.all(np.isfinite(rm)):
            raise ValueError(f"r must be a finite square matrix, got {rm.shape}")
        if float(np.linalg.eigvalsh(symmetrize(rm)).min()) <= 0.0:
            raise ValueError("r must be positive definite")
        self.f_fun, self.h_fun = f_fun, h_fun
        self.q, self.r, self.p = symmetrize(qm), symmetrize(rm), symmetrize(pm)
        self.x = x.copy()
        self.n, self.m = n, m
        self.points = MerweSigmaPoints(n=n, alpha=alpha, beta=beta, kappa=kappa)
        self._wm, self._wc = self.points.weights()

    def predict(self, u: ArrayLike | None = None, q: ArrayLike | None = None):
        """Time update through the unscented transform; returns ``(x⁻, P⁻)``."""
        qm = self.q if q is None else symmetrize(np.atleast_2d(np.asarray(q, dtype=float)))
        if qm.shape != (self.n, self.n):
            raise ValueError(f"q must have shape ({self.n}, {self.n}), got {qm.shape}")
        pts = self.points.sigma_points(self.x, self.p)
        prop = np.array(
            [
                np.asarray(self.f_fun(p) if u is None else self.f_fun(p, u), dtype=float).ravel()
                for p in pts
            ]
        )
        if prop.shape != (self.points.num_points, self.n):
            raise ValueError(f"f_fun must return {self.n} elements per sigma point")
        self.x, self.p = unscented_transform(prop, self._wm, self._wc, qm)
        self._prop_points = prop
        return self.x.copy(), self.p.copy()

    def update(self, z: ArrayLike, r: ArrayLike | None = None) -> dict[str, object]:
        """Measurement update; returns the standard result dict."""
        zz = np.asarray(z, dtype=float).ravel()
        if zz.size != self.m:
            raise ValueError(f"z must have {self.m} elements, got {zz.size}")
        if not np.all(np.isfinite(zz)):
            raise ValueError("z must be finite")
        rm = self.r if r is None else symmetrize(np.atleast_2d(np.asarray(r, dtype=float)))
        if rm.shape != (self.m, self.m):
            raise ValueError(f"r must have shape ({self.m}, {self.m}), got {rm.shape}")
        pts = self.points.sigma_points(self.x, self.p)
        zs = np.array([np.asarray(self.h_fun(p), dtype=float).ravel() for p in pts])
        if zs.shape != (self.points.num_points, self.m):
            raise ValueError(f"h_fun must return {self.m} elements per sigma point")
        z_hat, s = unscented_transform(zs, self._wm, self._wc, rm)
        dx = pts - self.x
        dz = zs - z_hat
        cross = (dx * self._wc[:, None]).T @ dz
        try:
            np.linalg.cholesky(s)
        except np.linalg.LinAlgError as exc:
            raise CovarianceCollapseError(
                "innovation covariance S is not positive definite "
                f"(min eigenvalue {float(np.linalg.eigvalsh(s).min()):.3e}); "
                "a negative W0c with a near-singular P is the usual cause"
            ) from exc
        gain = np.linalg.solve(s, cross.T).T
        nu = zz - z_hat
        self.x = self.x + gain @ nu
        self.p = symmetrize(self.p - gain @ s @ gain.T)
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
                pts = self.points.sigma_points(self.x, self.p)
                zs = np.array([np.asarray(self.h_fun(p), dtype=float).ravel() for p in pts])
                _, ss[k] = unscented_transform(zs, self._wm, self._wc, self.r)
            xu[k], pu[k] = self.x.copy(), self.p.copy()
        return FilterResult(xp, xu, pp, pu, nu, ss, kk, nis, avail)
