"""Extended Kalman filter with user-supplied Jacobians.

Model
-----
.. math::
    x_k = f(x_{k-1}, u_{k-1}) + w_{k-1}, \\quad w \\sim N(0, Q) \\\\
    z_k = h(x_k) + v_k, \\quad v \\sim N(0, R)

The EKF propagates the mean through the nonlinear maps and the covariance
through their Jacobians evaluated at the current estimate:

.. math::
    F_k = \\left.\\frac{\\partial f}{\\partial x}\\right|_{\\hat x_{k-1}},
    \\qquad
    H_k = \\left.\\frac{\\partial h}{\\partial x}\\right|_{\\hat x^{-}_{k}}

References
----------
- Bar-Shalom, Rong Li & Kirubarajan, *Estimation with Applications to
  Tracking and Navigation*, Wiley 2001, Ch. 10 (nonlinear filtering).
- Simon, D., *Optimal State Estimation*, Wiley 2006, Ch. 13 (the extended
  Kalman filter), including the first-order linearisation error analysis.

Validity
--------
The EKF is a first-order method: it is accurate only while the second and
higher derivatives of ``f`` and ``h`` are negligible over the region
covered by the state uncertainty. Concretely, the neglected term in the
mean is ``(1/2) tr(Hess . P)``; when that is comparable to the state
standard deviation the estimate is biased and the covariance is optimistic.
Strongly nonlinear measurement geometry (e.g. bearing-only tracking at
short range, or range-only tracking near the sensor) is the classic failure
case; see ``examples/ukf_vs_ekf_nonlinear.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .covariance import joseph_update, symmetrize
from .linear import FilterResult, UpdateResult

__all__ = ["numerical_jacobian", "ExtendedKalmanFilter", "EKFCallables"]

VectorFunc = Callable[[NDArray[np.float64]], NDArray[np.float64]]
JacobianFunc = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def numerical_jacobian(
    func: VectorFunc,
    x: ArrayLike,
    epsilon: float | None = None,
) -> NDArray[np.float64]:
    r"""Central-difference Jacobian of ``func`` at ``x``.

    .. math::
        J_{ij} \approx \frac{f_i(x + h e_j) - f_i(x - h e_j)}{2h}

    Parameters
    ----------
    func : callable
        Maps ``(n,)`` to ``(m,)``. Must be deterministic and return a
        1-D array of fixed length.
    x : array_like, shape (n,)
        Evaluation point.
    epsilon : float, optional
        Step ``h``. Default: ``cbrt(eps) * max(|x_j|, 1)`` per component,
        with ``eps`` the double-precision machine epsilon
        (``cbrt(2.22e-16) ~= 6.06e-6``), which balances the
        :math:`O(h^2)` truncation error against the :math:`O(\epsilon/h)`
        round-off error for a central difference.

    Returns
    -------
    ndarray, shape (m, n)

    Warnings
    --------
    **Accuracy caveat -- read before using this in place of an analytic
    Jacobian.**

    1. *Truncation vs round-off.* The central difference is exact only to
       :math:`O(h^{2})`; the achievable accuracy is roughly
       :math:`\epsilon^{2/3} \approx 4\times10^{-11}` relative for
       well-scaled smooth functions, and degrades to nothing useful when
       ``func`` is evaluated to less than full double precision (e.g. it
       internally integrates an ODE with a loose tolerance, or reads a
       lookup table).
    2. *Scaling.* A single scalar ``epsilon`` across components with very
       different magnitudes (metres and radians in the same state vector)
       will be far too large for one and far too small for another. The
       default per-component scaling mitigates but does not remove this.
    3. *Non-smoothness.* Angle wrapping, ``abs``, ``min``/``max``,
       saturation and table interpolation produce Jacobians that are wrong
       (not merely inaccurate) at or near the kink, and the error is
       silent -- the filter simply becomes inconsistent.
    4. *Cost.* ``2n`` evaluations of ``func`` per Jacobian, per step.
    5. *Consequence in the filter.* A Jacobian error does not raise; it
       shows up as an inconsistent covariance (NIS outside its chi-squared
       bounds) or slow divergence. Prefer an analytic Jacobian for anything
       beyond exploration, and use this helper to *check* the analytic one.
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    f0 = np.asarray(func(x_arr), dtype=float).reshape(-1)
    n = x_arr.size
    m = f0.size
    jac = np.empty((m, n))
    base = np.cbrt(np.finfo(float).eps)
    for j in range(n):
        h = base * max(abs(x_arr[j]), 1.0) if epsilon is None else float(epsilon)
        if h <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        xp = x_arr.copy()
        xm = x_arr.copy()
        xp[j] += h
        xm[j] -= h
        fp = np.asarray(func(xp), dtype=float).reshape(-1)
        fm = np.asarray(func(xm), dtype=float).reshape(-1)
        if fp.size != m or fm.size != m:
            raise ValueError("func must return a vector of constant length")
        jac[:, j] = (fp - fm) / (2.0 * h)
    return jac


@dataclass(frozen=True)
class EKFCallables:
    """Container for the four callables an EKF needs (documentation aid)."""

    f: VectorFunc
    f_jac: JacobianFunc | None
    h: VectorFunc
    h_jac: JacobianFunc | None


class ExtendedKalmanFilter:
    """First-order extended Kalman filter, Joseph-form covariance update.

    Parameters
    ----------
    f : callable
        State transition ``f(x) -> x_next``; maps ``(n,)`` to ``(n,)``.
    h : callable
        Measurement function ``h(x) -> z``; maps ``(n,)`` to ``(m,)``.
    process_noise : array_like, shape (n, n)
        ``Q`` [squared state units per step].
    measurement_noise : array_like, shape (m, m)
        ``R`` [squared measurement units].
    f_jac : callable, optional
        Analytic ``df/dx`` returning ``(n, n)``. If ``None``, the
        central-difference :func:`numerical_jacobian` is used -- see its
        accuracy caveat; this is intended for exploration, not production.
    h_jac : callable, optional
        Analytic ``dh/dx`` returning ``(m, n)``; same fallback rule.

    Notes
    -----
    ``f`` and ``h`` are called with a plain 1-D float array. Bind any extra
    arguments (control input, time step) with ``functools.partial`` or a
    closure before constructing the filter; this keeps the Jacobian
    contract unambiguous.
    """

    def __init__(
        self,
        f: VectorFunc,
        h: VectorFunc,
        process_noise: ArrayLike,
        measurement_noise: ArrayLike,
        f_jac: JacobianFunc | None = None,
        h_jac: JacobianFunc | None = None,
    ) -> None:
        if not callable(f) or not callable(h):
            raise TypeError("f and h must be callables mapping a state vector to a vector")
        for name, jac in (("f_jac", f_jac), ("h_jac", h_jac)):
            if jac is not None and not callable(jac):
                raise TypeError(f"{name} must be callable or None")
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
        self.f_jac = f_jac
        self.h_jac = h_jac
        self.n = q.shape[0]
        self.m = r.shape[0]

    def transition_jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``df/dx`` at ``x``, analytic if supplied else numerical."""
        if self.f_jac is not None:
            jac = np.atleast_2d(np.asarray(self.f_jac(x), dtype=float))
        else:
            jac = numerical_jacobian(self.f, x)
        if jac.shape != (self.n, self.n):
            raise ValueError(f"f_jac must return shape ({self.n}, {self.n}), got {jac.shape}")
        return jac

    def measurement_jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``dh/dx`` at ``x``, analytic if supplied else numerical."""
        if self.h_jac is not None:
            jac = np.atleast_2d(np.asarray(self.h_jac(x), dtype=float))
        else:
            jac = numerical_jacobian(self.h, x)
        if jac.shape != (self.m, self.n):
            raise ValueError(f"h_jac must return shape ({self.m}, {self.n}), got {jac.shape}")
        return jac

    def predict(
        self, x: ArrayLike, p: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Propagate the mean through ``f`` and the covariance through ``F``.

        Returns
        -------
        (x_pred, P_pred) : (ndarray (n,), ndarray (n, n))
            ``P_pred = F P F^T + Q``, symmetrised.
        """
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        if x_arr.size != self.n:
            raise ValueError(f"x must have {self.n} elements, got {x_arr.size}")
        p_arr = np.atleast_2d(np.asarray(p, dtype=float))
        if p_arr.shape != (self.n, self.n):
            raise ValueError(f"P must be {self.n}x{self.n}, got {p_arr.shape}")
        fj = self.transition_jacobian(x_arr)
        x_pred = np.asarray(self.f(x_arr), dtype=float).reshape(-1)
        if x_pred.size != self.n:
            raise ValueError(f"f must return {self.n} elements, got {x_pred.size}")
        return x_pred, symmetrize(fj @ p_arr @ fj.T + self.process_noise)

    def update(self, x: ArrayLike, p: ArrayLike, z: ArrayLike) -> UpdateResult:
        """Measurement update with ``H`` evaluated at the predicted state.

        Returns
        -------
        UpdateResult
            ``gain``/``innovation_cov``/``nis`` have the same meaning as in
            the linear filter, but they are only as trustworthy as the
            linearisation.
        """
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        p_arr = np.atleast_2d(np.asarray(p, dtype=float))
        z_arr = np.asarray(z, dtype=float).reshape(-1)
        if z_arr.size != self.m:
            raise ValueError(f"z must have {self.m} elements, got {z_arr.size}")
        hj = self.measurement_jacobian(x_arr)
        z_pred = np.asarray(self.h(x_arr), dtype=float).reshape(-1)
        if z_pred.size != self.m:
            raise ValueError(f"h must return {self.m} elements, got {z_pred.size}")
        innovation = z_arr - z_pred
        s = symmetrize(hj @ p_arr @ hj.T + self.measurement_noise)
        gain = np.linalg.solve(s, hj @ p_arr).T
        x_post = x_arr + gain @ innovation
        p_post = joseph_update(p_arr, gain, hj, self.measurement_noise)
        nis = float(innovation @ np.linalg.solve(s, innovation))
        return UpdateResult(x_post, p_post, gain, innovation, s, nis)

    def filter(self, x0: ArrayLike, p0: ArrayLike, measurements: ArrayLike) -> FilterResult:
        """Run predict/update over a measurement sequence.

        ``FilterResult.transition`` stores the per-step transition Jacobian
        ``F_k`` actually used, so the RTS smoother can consume the result
        directly (an approximation -- see :func:`estimkit.smoother.rts_smooth`).
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
            transitions[k] = self.transition_jacobian(x)
            x, p = self.predict(x, p)
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
