r"""Unscented Kalman filter (scaled symmetric sigma-point set).

Sigma-point and weight convention — stated explicitly, because results are not
reproducible without it. For an ``n``-dimensional state with mean ``x̄`` and
covariance ``P``:

.. math::
    \lambda = \alpha^{2}(n+\kappa) - n, \qquad S S^{\mathsf T} = (n+\lambda) P

with ``S`` the **lower-triangular** Cholesky factor. The ``2n+1`` sigma points
are ``X₀ = x̄`` and ``X_i = x̄ ± S_{:,i}`` (the **columns** of ``S``), with

.. math::
    W_0^{m} = \frac{\lambda}{n+\lambda},\quad
    W_0^{c} = \frac{\lambda}{n+\lambda} + (1-\alpha^{2}+\beta),\quad
    W_i^{m}=W_i^{c}=\frac{1}{2(n+\lambda)}

The mean weights sum to exactly 1; the covariance weights do not unless
``β = α² − 1``, which is intentional.

Sources: Julier, S. J. and Uhlmann, J. K. (2004), "Unscented Filtering and
Nonlinear Estimation", *Proceedings of the IEEE* **92**(3), 401–422 (the scaled
unscented transform, Eqs. (12)–(15)); Wan, E. A. and van der Merwe, R. (2000),
"The Unscented Kalman Filter for Nonlinear Estimation", *IEEE AS-SPCC*, 153–158
(the ``α, β, κ`` parameterisation and the ``β = 2`` Gaussian choice).

Parameter meaning and validity
------------------------------
* ``alpha ∈ (0, 1]`` sets the spread. Small ``α`` keeps the points local, which
  limits the influence of higher-order terms but makes ``W₀`` large and
  negative; ``1e-3`` (Wan & van der Merwe) and ``1`` (original UT) are both
  standard.
* ``beta >= 0`` folds prior knowledge into ``W₀ᶜ``; ``β = 2`` is optimal for a
  Gaussian prior.
* ``kappa`` is a secondary scaling; ``0`` and ``3 − n`` are the usual choices.

Requires ``alpha > 0`` and ``n + kappa > 0`` so that ``n + λ > 0`` and the
Cholesky factor exists. The UKF captures the posterior mean and covariance to
**second** order for any nonlinearity (third for Gaussian priors with β = 2),
against the EKF's first — but it is still a Gaussian-assumed-density filter and
will be inconsistent for genuinely multimodal posteriors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .kf import UpdateInfo
from .linalg import safe_cholesky, symmetrize

__all__ = ["SigmaPointSpec", "sigma_points", "UnscentedKalmanFilter"]


@dataclass(frozen=True)
class SigmaPointSpec:
    """Scaled unscented transform parameters."""

    alpha: float = 1.0e-3
    beta: float = 2.0
    kappa: float = 0.0

    def lam(self, n: int) -> float:
        """``λ = α²(n+κ) − n``."""
        if self.alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if n + self.kappa <= 0.0:
            raise ValueError(f"need n + kappa > 0, got n={n}, kappa={self.kappa}")
        return self.alpha ** 2 * (n + self.kappa) - n

    def weights(self, n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(Wm, Wc)`` each of length ``2n+1``."""
        lam = self.lam(n)
        wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = lam / (n + lam) + (1.0 - self.alpha ** 2 + self.beta)
        return wm, wc


def sigma_points(
    x: ArrayLike, p: ArrayLike, spec: SigmaPointSpec = SigmaPointSpec()
) -> NDArray[np.float64]:
    """Return the ``(2n+1, n)`` array of scaled sigma points."""
    xm = np.asarray(x, dtype=float).reshape(-1)
    n = xm.size
    pp = symmetrize(np.atleast_2d(np.asarray(p, dtype=float)))
    if pp.shape != (n, n):
        raise ValueError(f"p must be {n}x{n}, got {pp.shape}")
    lam = spec.lam(n)
    s = safe_cholesky((n + lam) * pp)
    pts = np.zeros((2 * n + 1, n))
    pts[0] = xm
    for i in range(n):
        pts[1 + i] = xm + s[:, i]
        pts[1 + n + i] = xm - s[:, i]
    return pts


@dataclass
class UnscentedKalmanFilter:
    """Additive-noise UKF.

    Parameters
    ----------
    f : callable ``(x, dt) -> x``
    h : callable ``(x) -> z``
    q, r : array_like
        Additive process- and measurement-noise covariances.
    x, p : array_like
        Initial state and covariance.
    spec : SigmaPointSpec
        Sigma-point parameters (see module docstring).
    """

    f: Callable[[NDArray[np.float64], float], NDArray[np.float64]]
    h: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    q: NDArray[np.float64]
    r: NDArray[np.float64]
    x: NDArray[np.float64]
    p: NDArray[np.float64]
    spec: SigmaPointSpec = field(default_factory=SigmaPointSpec)
    history: list[UpdateInfo] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float).reshape(-1)
        n = self.x.size
        self.q = symmetrize(np.atleast_2d(np.asarray(self.q, dtype=float)))
        self.p = symmetrize(np.atleast_2d(np.asarray(self.p, dtype=float)))
        self.r = symmetrize(np.atleast_2d(np.asarray(self.r, dtype=float)))
        if self.q.shape != (n, n) or self.p.shape != (n, n):
            raise ValueError(f"q and p must be {n}x{n}, got {self.q.shape} and {self.p.shape}")

    @property
    def n(self) -> int:
        """State dimension."""
        return int(self.x.size)

    def predict(self, dt: float = 1.0, q: ArrayLike | None = None) -> None:
        """Unscented time update over ``dt`` [s]."""
        qk = self.q if q is None else symmetrize(np.atleast_2d(np.asarray(q, dtype=float)))
        wm, wc = self.spec.weights(self.n)
        pts = sigma_points(self.x, self.p, self.spec)
        prop = np.array([np.asarray(self.f(pt, dt), dtype=float).reshape(-1) for pt in pts])
        xm = wm @ prop
        d = prop - xm
        self.x = xm
        self.p = symmetrize(np.einsum("i,ij,ik->jk", wc, d, d) + qk)

    def update(self, z: ArrayLike, r: ArrayLike | None = None) -> UpdateInfo:
        """Unscented measurement update; returns innovation diagnostics."""
        zz = np.asarray(z, dtype=float).reshape(-1)
        rk = self.r if r is None else symmetrize(np.atleast_2d(np.asarray(r, dtype=float)))
        wm, wc = self.spec.weights(self.n)
        pts = sigma_points(self.x, self.p, self.spec)
        zs = np.array([np.asarray(self.h(pt), dtype=float).reshape(-1) for pt in pts])
        zhat = wm @ zs
        dz = zs - zhat
        dx = pts - self.x
        s = symmetrize(np.einsum("i,ij,ik->jk", wc, dz, dz) + rk)
        pxz = np.einsum("i,ij,ik->jk", wc, dx, dz)
        k = np.linalg.solve(s.T, pxz.T).T
        nu = zz - zhat
        self.x = self.x + k @ nu
        self.p = symmetrize(self.p - k @ s @ k.T)
        lo = np.linalg.cholesky(s)
        y = np.linalg.solve(lo, nu)
        info = UpdateInfo(innovation=nu, innovation_cov=s, gain=k, nis=float(np.dot(y, y)))
        self.history.append(info)
        return info
