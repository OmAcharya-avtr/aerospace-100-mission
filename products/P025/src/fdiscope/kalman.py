"""Discrete linear Kalman filter, written to supply an innovation sequence.

Conventions follow EstimKit (P017 of this portfolio), cited as related work
only -- **nothing is imported from it**, and this implementation is
independent.  The recursion is the textbook one (Bar-Shalom, Rong Li &
Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley
2001, ch. 5; Simon, *Optimal State Estimation*, Wiley 2006, ch. 5)::

    predict:  x^- = F x^+ + G u
              P^- = F P^+ F^T + Q

    update:   y   = z - H x^-                         (innovation)
              S   = H P^- H^T + R                     (innovation covariance)
              K   = P^- H^T S^-1                      (Kalman gain)
              x^+ = x^- + K y
              P^+ = (I - K H) P^- (I - K H)^T + K R K^T        (Joseph form)

The Joseph form is used because it stays symmetric positive semi-definite
under gain error, which matters here: the whole product is about what the
innovation does when the model is wrong.

Whiteness of the innovation
---------------------------
For a *consistent* filter the innovation sequence is zero-mean, white, with
covariance ``S``.  The normalised innovation squared

.. math::
    \\epsilon_k = y_k^T S_k^{-1} y_k

is then chi-squared distributed with ``m`` degrees of freedom, ``m`` the
measurement dimension.  Everything downstream in this package -- the
chi-squared test, the CUSUM, the residual features -- rests on that single
fact, which is why :mod:`fdiscope.residuals` provides an explicit whiteness
check rather than assuming it.

Units are whatever the caller uses consistently; ``P`` carries squared state
units, ``R`` squared measurement units, ``K`` state per measurement unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_discrete_are

__all__ = ["KalmanState", "UpdateResult", "KalmanFilter", "steady_state_covariance"]


def _square(a: ArrayLike, n: int | None, name: str) -> NDArray[np.float64]:
    arr = np.atleast_2d(np.asarray(a, dtype=float))
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-D array, got shape {arr.shape}")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} must be {n}x{n}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _vector(a: ArrayLike, n: int, name: str) -> NDArray[np.float64]:
    arr = np.asarray(a, dtype=float).reshape(-1)
    if arr.size != n:
        raise ValueError(f"{name} must have {n} elements, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def symmetrize(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``(P + P^T) / 2``, removing accumulated asymmetry."""
    return 0.5 * (p + p.T)


@dataclass
class KalmanState:
    """Filter state: mean and covariance.

    Attributes
    ----------
    x : ndarray, shape (n,)
        State mean.
    p : ndarray, shape (n, n)
        State covariance (squared state units).
    """

    x: NDArray[np.float64]
    p: NDArray[np.float64]

    def copy(self) -> KalmanState:
        """Deep copy of the state."""
        return KalmanState(x=self.x.copy(), p=self.p.copy())


@dataclass(frozen=True)
class UpdateResult:
    """Outcome of one measurement update.

    Attributes
    ----------
    state : KalmanState
        Posterior mean and covariance.
    innovation : ndarray, shape (m,)
        ``y = z - H x^-`` [measurement units].
    innovation_cov : ndarray, shape (m, m)
        ``S = H P^- H^T + R`` [squared measurement units].
    gain : ndarray, shape (n, m)
        Kalman gain ``K``.
    nis : float
        ``y^T S^-1 y``, dimensionless; chi-squared with ``m`` degrees of
        freedom for a consistent filter.
    """

    state: KalmanState
    innovation: NDArray[np.float64]
    innovation_cov: NDArray[np.float64] = field(repr=False)
    gain: NDArray[np.float64] = field(repr=False)
    nis: float = 0.0


class KalmanFilter:
    """Time-invariant discrete linear Kalman filter.

    Parameters
    ----------
    f : array_like, shape (n, n)
        State transition matrix.
    h : array_like, shape (m, n)
        Measurement matrix.
    q : array_like, shape (n, n)
        Process-noise covariance, symmetric positive semi-definite.
    r : array_like, shape (m, m)
        Measurement-noise covariance, symmetric positive definite.
    g : array_like, shape (n, p), optional
        Control input matrix.  Omit for an autonomous system.

    Raises
    ------
    ValueError
        On shape mismatch, non-finite entries, a non-symmetric ``Q`` or ``R``,
        a non-positive-definite ``R``, or a ``Q`` with a negative eigenvalue.
    """

    def __init__(
        self,
        f: ArrayLike,
        h: ArrayLike,
        q: ArrayLike,
        r: ArrayLike,
        g: ArrayLike | None = None,
    ) -> None:
        self.f = _square(f, None, "f")
        self.n = self.f.shape[0]
        hh = np.atleast_2d(np.asarray(h, dtype=float))
        if hh.ndim != 2 or hh.shape[1] != self.n:
            raise ValueError(f"h must be (m, {self.n}), got shape {hh.shape}")
        self.h = hh
        self.m = hh.shape[0]
        self.q = _square(q, self.n, "q")
        self.r = _square(r, self.m, "r")
        if not np.allclose(self.q, self.q.T, atol=1e-12):
            raise ValueError("q must be symmetric")
        if not np.allclose(self.r, self.r.T, atol=1e-12):
            raise ValueError("r must be symmetric")
        if float(np.min(np.linalg.eigvalsh(self.q))) < -1e-12:
            raise ValueError("q must be positive semi-definite")
        if float(np.min(np.linalg.eigvalsh(self.r))) <= 0.0:
            raise ValueError("r must be positive definite")
        if g is None:
            self.g: NDArray[np.float64] | None = None
            self.p_in = 0
        else:
            gg = np.atleast_2d(np.asarray(g, dtype=float))
            if gg.ndim != 2 or gg.shape[0] != self.n:
                raise ValueError(f"g must be ({self.n}, p), got shape {gg.shape}")
            self.g = gg
            self.p_in = gg.shape[1]

    def predict(self, state: KalmanState, u: ArrayLike | float | None = None) -> KalmanState:
        """One time update.

        Parameters
        ----------
        state : KalmanState
            Posterior of the previous step.
        u : array_like or float, optional
            Control input applied over the step.  Required if the filter was
            built with a ``g`` matrix.

        Returns
        -------
        KalmanState
            Prior for the next measurement.
        """
        x = _vector(state.x, self.n, "state.x")
        p = _square(state.p, self.n, "state.p")
        x_new = self.f @ x
        if self.g is not None:
            if u is None:
                raise ValueError("u is required when the filter has a control matrix g")
            uu = _vector(np.atleast_1d(u), self.p_in, "u")
            x_new = x_new + self.g @ uu
        elif u is not None:
            raise ValueError("u given but the filter has no control matrix g")
        p_new = symmetrize(self.f @ p @ self.f.T + self.q)
        return KalmanState(x=x_new, p=p_new)

    def update(self, state: KalmanState, z: ArrayLike) -> UpdateResult:
        """One measurement update in Joseph form.

        Parameters
        ----------
        state : KalmanState
            Prior from :meth:`predict`.
        z : array_like, shape (m,)
            Measurement.

        Returns
        -------
        UpdateResult
            Posterior plus the innovation, its covariance, the gain and the
            normalised innovation squared.
        """
        x = _vector(state.x, self.n, "state.x")
        p = _square(state.p, self.n, "state.p")
        zz = _vector(z, self.m, "z")
        y = zz - self.h @ x
        s = symmetrize(self.h @ p @ self.h.T + self.r)
        s_inv = np.linalg.inv(s)
        k = p @ self.h.T @ s_inv
        x_post = x + k @ y
        ikh = np.eye(self.n) - k @ self.h
        p_post = symmetrize(ikh @ p @ ikh.T + k @ self.r @ k.T)
        nis = float(y @ s_inv @ y)
        return UpdateResult(
            state=KalmanState(x=x_post, p=p_post),
            innovation=y,
            innovation_cov=s,
            gain=k,
            nis=nis,
        )


def steady_state_covariance(
    kf: KalmanFilter, max_iter: int = 5000, tol: float = 1e-14, method: str = "dare"
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Steady-state prior covariance of a time-invariant filter.

    Two routes to the same fixed point:

    ``"dare"`` (default)
        Solve the discrete algebraic Riccati equation directly with
        ``scipy.linalg.solve_discrete_are(F^T, H^T, Q, R)``, which is the
        standard filtering form of the DARE (Anderson & Moore, *Optimal
        Filtering*, Prentice-Hall, 1979, ch. 4).  About 250x faster than
        iterating, which matters because every simulated run needs it.
    ``"iterate"``
        Run the Riccati recursion to convergence.  Kept as the transparent
        reference; ``tests/test_kalman.py`` asserts the two agree to 1e-12.

    Parameters
    ----------
    kf : KalmanFilter
        Time-invariant filter.
    max_iter : int
        Iteration cap, ``"iterate"`` only.
    tol : float
        ``"iterate"`` only.  Relative convergence tolerance: the iteration
        stops when the maximum
        absolute change in ``P^-`` falls below ``tol * max|P^-|``.  It is
        relative because ``P`` here carries squared radians and is of order
        1e-7, so an absolute tolerance would stop far too early and leave a
        residual mismatch visible in the innovation.

    Returns
    -------
    (p_prior, s) : tuple of ndarray
        Steady-state prior covariance and the corresponding innovation
        covariance ``S = H P^- H^T + R``.

    Raises
    ------
    RuntimeError
        If the recursion has not converged within ``max_iter`` iterations.
    ValueError
        If ``method`` is neither ``"dare"`` nor ``"iterate"``.

    Notes
    -----
    Used to start every simulation at the filter's steady state so that no
    transient contaminates the measured false-alarm rate.
    """
    if method not in ("dare", "iterate"):
        raise ValueError(f"method must be 'dare' or 'iterate', got {method!r}")
    if method == "dare":
        p = symmetrize(np.asarray(solve_discrete_are(kf.f.T, kf.h.T, kf.q, kf.r), dtype=float))
        return p, symmetrize(kf.h @ p @ kf.h.T + kf.r)
    p = np.eye(kf.n)
    for _ in range(int(max_iter)):
        s = symmetrize(kf.h @ p @ kf.h.T + kf.r)
        k = p @ kf.h.T @ np.linalg.inv(s)
        ikh = np.eye(kf.n) - k @ kf.h
        p_post = symmetrize(ikh @ p @ ikh.T + k @ kf.r @ k.T)
        p_next = symmetrize(kf.f @ p_post @ kf.f.T + kf.q)
        scale = max(float(np.max(np.abs(p_next))), 1e-300)
        if float(np.max(np.abs(p_next - p))) < tol * scale:
            p = p_next
            break
        p = p_next
    else:
        raise RuntimeError(f"Riccati recursion did not converge in {max_iter} iterations")
    s = symmetrize(kf.h @ p @ kf.h.T + kf.r)
    return p, s
