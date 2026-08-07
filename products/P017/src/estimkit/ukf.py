r"""Unscented Kalman filter and the scaled unscented transform.

Weight and sigma-point convention (stated explicitly)
-----------------------------------------------------
This module implements the **scaled** symmetric sigma-point set of Julier
and Uhlmann. For an ``n``-dimensional state with mean :math:`\bar x` and
covariance :math:`P`, define

.. math::
    \lambda = \alpha^{2}(n + \kappa) - n

and take :math:`S` to be the **lower-triangular Cholesky factor** of
:math:`(n+\lambda)P`, i.e. :math:`SS^{\mathsf{T}} = (n+\lambda)P`. The
:math:`2n+1` sigma points are the **columns** of :math:`S`:

.. math::
    \mathcal{X}_0 &= \bar x \\
    \mathcal{X}_i &= \bar x + S_{:,i}, & i = 1 \dots n \\
    \mathcal{X}_{i+n} &= \bar x - S_{:,i}, & i = 1 \dots n

with separate mean and covariance weights

.. math::
    W_0^{m} &= \frac{\lambda}{n+\lambda} \\
    W_0^{c} &= \frac{\lambda}{n+\lambda} + (1 - \alpha^{2} + \beta) \\
    W_i^{m} = W_i^{c} &= \frac{1}{2(n+\lambda)}, \quad i = 1 \dots 2n

The mean weights sum to exactly 1. The covariance weights do **not** sum
to 1 unless :math:`\beta = \alpha^{2} - 1`; that is intentional and is the
reason :math:`W_0^{c}` may be (and usually is) negative. Because
:math:`\mathcal{X}_0 - \bar x = 0`, the value of :math:`W_0^{c}` does not
affect the reconstruction of :math:`P` from the sigma points, nor the
transform of an affine function.

Parameter meanings
------------------
- ``alpha`` (:math:`\alpha \in (0, 1]`) sets the spread of the sigma points
  about the mean. Small :math:`\alpha` keeps the points local, which limits
  the influence of higher-order nonlinearity but makes :math:`W_0^{m}`
  large and negative. Typical: ``1e-3`` (Wan & van der Merwe) or ``1``
  (original unscented transform).
- ``beta`` (:math:`\beta \ge 0`) folds prior knowledge of the distribution
  into :math:`W_0^{c}`; :math:`\beta = 2` is optimal for a Gaussian prior.
- ``kappa`` (:math:`\kappa`) is a secondary scaling parameter; ``0`` and
  ``3 - n`` are both standard choices. ``kappa = 3 - n`` matches the fourth
  moment of a Gaussian for scalar states but gives :math:`n + \kappa = 3`
  regardless of ``n``, and can make :math:`W_0` negative for ``n > 3``.

Requirement: ``alpha > 0`` and ``n + kappa > 0``, so that
:math:`n + \lambda = \alpha^{2}(n+\kappa) > 0` and the Cholesky factor
exists.

References
----------
- Julier, S. J. and Uhlmann, J. K., "Unscented filtering and nonlinear
  estimation", *Proceedings of the IEEE*, Vol. 92, No. 3, 2004 -- the
  consolidated statement of the unscented transform.
- Julier, S. J. and Uhlmann, J. K., "A New Extension of the Kalman Filter
  to Nonlinear Systems", *Proc. SPIE AeroSense*, 1997 -- the original
  presentation.
- Julier, S. J., "The scaled unscented transformation", *Proc. American
  Control Conference*, 2002 -- introduces :math:`\alpha` and the scaled
  weights used here.
- Wan, E. A. and van der Merwe, R., "The unscented Kalman filter for
  nonlinear estimation", *Proc. IEEE AS-SPCC*, 2000 -- the
  :math:`(\alpha, \beta, \kappa)` parameterisation as used in practice.
- Simon, D., *Optimal State Estimation*, Wiley 2006, Ch. 14.

Exactness property
------------------
For an affine map :math:`g(x) = Ax + b` the transform reproduces the exact
mean :math:`A\bar x + b` and covariance :math:`APA^{\mathsf{T}}` for any
admissible :math:`(\alpha, \beta, \kappa)`, because
:math:`\sum_i W_i^{m} = 1` and
:math:`\sum_{i=1}^{2n} W_i^{c}(\mathcal{X}_i - \bar x)
(\mathcal{X}_i - \bar x)^{\mathsf{T}} = SS^{\mathsf{T}}/(n+\lambda) = P`.
Consequently the UKF reduces to the linear Kalman filter on a
linear-Gaussian system up to floating-point round-off; this is checked in
``validation/`` and in the test suite.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .covariance import symmetrize
from .linear import FilterResult, UpdateResult

__all__ = ["SigmaPoints", "MerweSigmaPoints", "unscented_transform", "UnscentedKalmanFilter"]

VectorFunc = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class SigmaPoints:
    """A sigma-point set with its mean and covariance weights.

    Attributes
    ----------
    points : ndarray, shape (2n+1, n)
        Sigma points, row 0 being the mean point.
    wm : ndarray, shape (2n+1,)
        Mean weights; ``wm.sum() == 1`` to round-off.
    wc : ndarray, shape (2n+1,)
        Covariance weights.
    lambda_ : float
        The scaling parameter :math:`\\lambda`.
    """

    points: NDArray[np.float64]
    wm: NDArray[np.float64]
    wc: NDArray[np.float64]
    lambda_: float


class MerweSigmaPoints:
    """Scaled symmetric sigma-point generator (see the module docstring).

    Parameters
    ----------
    n : int
        State dimension, must be >= 1.
    alpha : float
        Spread parameter, must satisfy ``0 < alpha <= 1``. Values above 1
        are permitted numerically but are outside normal practice and are
        rejected here to catch parameter mix-ups.
    beta : float
        Prior-knowledge parameter, must be >= 0. ``2.0`` is optimal for a
        Gaussian.
    kappa : float
        Secondary scaling; must satisfy ``n + kappa > 0``.
    """

    def __init__(self, n: int, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0) -> None:
        if not isinstance(n, (int, np.integer)) or int(n) < 1:
            raise ValueError(f"n must be an integer >= 1, got {n!r}")
        n = int(n)
        alpha, beta, kappa = float(alpha), float(beta), float(kappa)
        if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
            raise ValueError(f"alpha must satisfy 0 < alpha <= 1, got {alpha}")
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError(f"beta must be finite and >= 0, got {beta}")
        if not np.isfinite(kappa) or n + kappa <= 0.0:
            raise ValueError(f"kappa must be finite with n + kappa > 0, got kappa={kappa}, n={n}")
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (n + kappa) - n
        self._scale = n + self.lambda_  # == alpha^2 (n + kappa) > 0

        wm = np.full(2 * n + 1, 1.0 / (2.0 * self._scale))
        wc = wm.copy()
        wm[0] = self.lambda_ / self._scale
        wc[0] = self.lambda_ / self._scale + (1.0 - alpha**2 + beta)
        self.wm = wm
        self.wc = wc

    @property
    def num_points(self) -> int:
        """Number of sigma points, ``2n + 1``."""
        return 2 * self.n + 1

    def generate(self, mean: ArrayLike, cov: ArrayLike) -> SigmaPoints:
        """Build the sigma-point set for ``(mean, cov)``.

        Parameters
        ----------
        mean : array_like, shape (n,)
        cov : array_like, shape (n, n)
            Must be symmetric positive definite; a Cholesky failure is
            re-raised as ``ValueError`` naming covariance collapse or a
            lost positive-definiteness as the likely cause.

        Returns
        -------
        SigmaPoints
        """
        x = np.asarray(mean, dtype=float).reshape(-1)
        if x.size != self.n:
            raise ValueError(f"mean must have {self.n} elements, got {x.size}")
        p = np.atleast_2d(np.asarray(cov, dtype=float))
        if p.shape != (self.n, self.n):
            raise ValueError(f"cov must be {self.n}x{self.n}, got {p.shape}")
        try:
            s = np.linalg.cholesky(symmetrize(self._scale * p))
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Cholesky factorisation of (n + lambda) P failed: P is not positive "
                "definite. This is the standard observable symptom of covariance "
                "collapse or of a covariance driven indefinite by round-off."
            ) from exc
        pts = np.empty((self.num_points, self.n))
        pts[0] = x
        for i in range(self.n):
            pts[1 + i] = x + s[:, i]
            pts[1 + self.n + i] = x - s[:, i]
        return SigmaPoints(pts, self.wm.copy(), self.wc.copy(), self.lambda_)


def unscented_transform(
    points: ArrayLike,
    wm: ArrayLike,
    wc: ArrayLike,
    noise_cov: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Weighted mean and covariance of a transformed sigma-point set.

    .. math::
        \bar y = \sum_i W_i^{m}\,\mathcal{Y}_i, \qquad
        P_y = \sum_i W_i^{c}(\mathcal{Y}_i - \bar y)
              (\mathcal{Y}_i - \bar y)^{\mathsf{T}} + N

    Parameters
    ----------
    points : array_like, shape (L, d)
        Transformed sigma points.
    wm, wc : array_like, shape (L,)
        Mean and covariance weights.
    noise_cov : array_like, shape (d, d), optional
        Additive noise covariance :math:`N` (``Q`` for the time update,
        ``R`` for the measurement update).

    Returns
    -------
    (mean, cov) : (ndarray (d,), ndarray (d, d))
        ``cov`` is symmetrised. It is **not** guaranteed positive definite
        when :math:`W_0^{c} < 0` and the map is strongly nonlinear -- that
        is the known cost of the scaled transform, and the reason
        ``noise_cov`` (positive definite ``R``/``Q``) matters in practice.
    """
    y = np.atleast_2d(np.asarray(points, dtype=float))
    wm_a = np.asarray(wm, dtype=float).reshape(-1)
    wc_a = np.asarray(wc, dtype=float).reshape(-1)
    if y.shape[0] != wm_a.size or y.shape[0] != wc_a.size:
        raise ValueError(
            f"weights must match the number of sigma points: got {y.shape[0]} points, "
            f"{wm_a.size} mean weights, {wc_a.size} covariance weights"
        )
    mean = wm_a @ y
    dev = y - mean
    cov = (dev.T * wc_a) @ dev
    if noise_cov is not None:
        nc = np.atleast_2d(np.asarray(noise_cov, dtype=float))
        if nc.shape != (y.shape[1], y.shape[1]):
            raise ValueError(
                f"noise_cov must be {y.shape[1]}x{y.shape[1]}, got {nc.shape}"
            )
        cov = cov + nc
    return mean, symmetrize(cov)


class UnscentedKalmanFilter:
    """Additive-noise unscented Kalman filter.

    Parameters
    ----------
    f : callable
        State transition ``f(x) -> x_next``, ``(n,) -> (n,)``.
    h : callable
        Measurement function ``h(x) -> z``, ``(n,) -> (m,)``.
    process_noise : array_like, shape (n, n)
        ``Q``, added after the time-update transform (additive-noise UKF).
    measurement_noise : array_like, shape (m, m)
        ``R``, added after the measurement transform.
    alpha, beta, kappa : float
        Sigma-point parameters; see :class:`MerweSigmaPoints` and the
        module docstring.

    Notes
    -----
    This is the *additive* noise formulation: noise covariances are added
    to the transformed covariances rather than the state being augmented
    with noise components. That is correct when the noise enters
    additively, which is the case for every model shipped in
    :mod:`estimkit.models`. For multiplicative or state-dependent noise the
    augmented-state UKF is required and is not implemented here.

    The covariance update uses :math:`P^{+} = P^{-} - K S K^{\\mathsf{T}}`,
    which is the standard UKF form because no measurement matrix ``H``
    exists; it is the Joseph form's algebraic equivalent given
    :math:`K = C_{xz}S^{-1}`. The result is re-symmetrised each step. See
    the README for why this is a weaker numerical guarantee than the linear
    filter's Joseph update.
    """

    def __init__(
        self,
        f: VectorFunc,
        h: VectorFunc,
        process_noise: ArrayLike,
        measurement_noise: ArrayLike,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        if not callable(f) or not callable(h):
            raise TypeError("f and h must be callables mapping a state vector to a vector")
        q = symmetrize(np.atleast_2d(np.asarray(process_noise, dtype=float)))
        r = symmetrize(np.atleast_2d(np.asarray(measurement_noise, dtype=float)))
        if np.min(np.linalg.eigvalsh(q)) < -1e-12:
            raise ValueError("Q must be positive semi-definite")
        if np.min(np.linalg.eigvalsh(r)) < 0.0:
            raise ValueError("R must be positive semi-definite")
        self.f = f
        self.h = h
        self.process_noise = q
        self.measurement_noise = r
        self.n = q.shape[0]
        self.m = r.shape[0]
        self.sigma = MerweSigmaPoints(self.n, alpha=alpha, beta=beta, kappa=kappa)

    def predict(
        self, x: ArrayLike, p: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Time update.

        Returns
        -------
        (x_pred, P_pred, cross_cov)
            ``cross_cov`` is :math:`\\mathrm{Cov}(x_k, x_{k+1})`, needed by
            the unscented RTS smoother; shape ``(n, n)``.
        """
        sp = self.sigma.generate(x, p)
        prop = np.array([np.asarray(self.f(pt), dtype=float).reshape(-1) for pt in sp.points])
        if prop.shape[1] != self.n:
            raise ValueError(f"f must return {self.n} elements, got {prop.shape[1]}")
        x_pred, p_pred = unscented_transform(prop, sp.wm, sp.wc, self.process_noise)
        dx = sp.points - np.asarray(x, dtype=float).reshape(-1)
        dy = prop - x_pred
        cross = (dx.T * sp.wc) @ dy
        return x_pred, p_pred, cross

    def update(self, x: ArrayLike, p: ArrayLike, z: ArrayLike) -> UpdateResult:
        """Measurement update via the unscented transform of ``h``."""
        z_arr = np.asarray(z, dtype=float).reshape(-1)
        if z_arr.size != self.m:
            raise ValueError(f"z must have {self.m} elements, got {z_arr.size}")
        sp = self.sigma.generate(x, p)
        zs = np.array([np.asarray(self.h(pt), dtype=float).reshape(-1) for pt in sp.points])
        if zs.shape[1] != self.m:
            raise ValueError(f"h must return {self.m} elements, got {zs.shape[1]}")
        z_pred, s = unscented_transform(zs, sp.wm, sp.wc, self.measurement_noise)
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        dx = sp.points - x_arr
        dz = zs - z_pred
        cross = (dx.T * sp.wc) @ dz
        gain = np.linalg.solve(s, cross.T).T
        innovation = z_arr - z_pred
        x_post = x_arr + gain @ innovation
        p_post = symmetrize(np.atleast_2d(np.asarray(p, dtype=float)) - gain @ s @ gain.T)
        nis = float(innovation @ np.linalg.solve(s, innovation))
        return UpdateResult(x_post, p_post, gain, innovation, s, nis)

    def filter(self, x0: ArrayLike, p0: ArrayLike, measurements: ArrayLike) -> FilterResult:
        """Run predict/update over a measurement sequence.

        ``FilterResult.transition`` holds, for each step, the *effective*
        transition matrix :math:`F^{\\mathrm{eff}}_k` defined by
        :math:`P^{+}_{k-1}(F^{\\mathrm{eff}}_k)^{\\mathsf{T}} = C_k`, where
        :math:`C_k = \\mathrm{Cov}(x_{k-1}, x_k)` comes from the sigma-point
        propagation. With that definition the linear RTS recursion in
        :func:`estimkit.smoother.rts_smooth` reproduces the unscented RTS
        smoother exactly, so no separate smoother implementation is needed.
        """
        x = np.asarray(x0, dtype=float).reshape(-1)
        p = np.atleast_2d(np.asarray(p0, dtype=float))
        zs = np.atleast_2d(np.asarray(measurements, dtype=float))
        if zs.shape[1] != self.m:
            raise ValueError(f"measurements must have shape (T, {self.m}), got {zs.shape}")
        t = zs.shape[0]
        if t == 0:
            raise ValueError("measurements must contain at least one time step")

        x_prior = np.empty((t, self.n))
        p_prior = np.empty((t, self.n, self.n))
        x_post = np.empty((t, self.n))
        p_post = np.empty((t, self.n, self.n))
        gains = np.empty((t, self.n, self.m))
        innov = np.empty((t, self.m))
        innov_cov = np.empty((t, self.m, self.m))
        nis = np.empty(t)
        transitions = np.empty((t, self.n, self.n))

        for k in range(t):
            p_prev = p
            x, p, cross = self.predict(x, p)
            transitions[k] = np.linalg.solve(symmetrize(p_prev), cross).T
            x_prior[k] = x
            p_prior[k] = p
            res = self.update(x, p, zs[k])
            x, p = res.x, res.p
            x_post[k] = x
            p_post[k] = p
            gains[k] = res.gain
            innov[k] = res.innovation
            innov_cov[k] = res.innovation_cov
            nis[k] = res.nis

        return FilterResult(
            x_prior=x_prior,
            p_prior=p_prior,
            x_post=x_post,
            p_post=p_post,
            gain=gains,
            innovation=innov,
            innovation_cov=innov_cov,
            nis=nis,
            transition=transitions,
        )
