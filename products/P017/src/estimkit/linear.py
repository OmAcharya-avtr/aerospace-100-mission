"""Linear (discrete-time) Kalman filter with Joseph-form covariance update.

Model
-----
Discrete-time linear-Gaussian system (Bar-Shalom, Rong Li & Kirubarajan,
*Estimation with Applications to Tracking and Navigation*, Wiley 2001,
Ch. 5; Simon, *Optimal State Estimation*, Wiley 2006, Ch. 5):

.. math::
    x_{k} &= F x_{k-1} + B u_{k-1} + w_{k-1}, \\quad w \\sim N(0, Q) \\\\
    z_{k} &= H x_{k} + v_{k}, \\quad v \\sim N(0, R)

with ``w`` and ``v`` zero-mean, white, and mutually uncorrelated.

Recursion
---------
Predict::

    x^- = F x^+ + B u
    P^- = F P^+ F^T + Q

Update::

    y  = z - H x^-                      (innovation)
    S  = H P^- H^T + R                  (innovation covariance)
    K  = P^- H^T S^-1                   (Kalman gain)
    x^+ = x^- + K y
    P^+ = (I - K H) P^- (I - K H)^T + K R K^T      (Joseph form)

Units are whatever the caller uses consistently; ``P`` carries squared
state units, ``R`` squared measurement units, ``K`` state per measurement
unit.

Validity
--------
Optimal (minimum-mean-square-error) only when the model is linear, the
noises are white, zero-mean, mutually uncorrelated and have the stated
covariances. Under non-Gaussian noise it remains the best *linear*
unbiased estimator. Correlated process/measurement noise, coloured noise
and time-correlated biases are not handled here (state augmentation is the
standard remedy; see README Limitations).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .covariance import joseph_update, symmetrize

__all__ = ["UpdateResult", "FilterResult", "KalmanFilter", "steady_state"]


def _square(a: ArrayLike, n: int | None, name: str) -> NDArray[np.float64]:
    arr = np.atleast_2d(np.asarray(a, dtype=float))
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-D array, got shape {arr.shape}")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} must be {n}x{n}, got {arr.shape}")
    return arr


def _vector(a: ArrayLike, n: int, name: str) -> NDArray[np.float64]:
    arr = np.asarray(a, dtype=float).reshape(-1)
    if arr.size != n:
        raise ValueError(f"{name} must have {n} elements, got {arr.size}")
    return arr


@dataclass(frozen=True)
class UpdateResult:
    """Outcome of one measurement update.

    Attributes
    ----------
    x : ndarray, shape (n,)
        Posterior state mean.
    p : ndarray, shape (n, n)
        Posterior covariance (Joseph form, symmetrised).
    gain : ndarray, shape (n, m)
        Kalman gain ``K``.
    innovation : ndarray, shape (m,)
        Measurement residual ``z - H x^-`` [measurement units].
    innovation_cov : ndarray, shape (m, m)
        ``S = H P^- H^T + R`` [squared measurement units].
    nis : float
        Normalised innovation squared ``y^T S^-1 y`` (dimensionless).
        For a consistent filter this is chi-squared distributed with ``m``
        degrees of freedom, so its expectation is ``m``.
    """

    x: NDArray[np.float64]
    p: NDArray[np.float64]
    gain: NDArray[np.float64]
    innovation: NDArray[np.float64]
    innovation_cov: NDArray[np.float64]
    nis: float


@dataclass
class FilterResult:
    """Histories produced by :meth:`KalmanFilter.filter`.

    All arrays are stacked over time, index 0 corresponding to the first
    measurement. ``x_prior``/``p_prior`` are the one-step predictions used
    by that update; the RTS smoother needs both prior and posterior
    sequences.
    """

    x_prior: NDArray[np.float64]
    p_prior: NDArray[np.float64]
    x_post: NDArray[np.float64]
    p_post: NDArray[np.float64]
    gain: NDArray[np.float64]
    innovation: NDArray[np.float64]
    innovation_cov: NDArray[np.float64]
    nis: NDArray[np.float64]
    transition: NDArray[np.float64] = field(repr=False)

    def __len__(self) -> int:
        return int(self.x_post.shape[0])


class KalmanFilter:
    """Time-invariant (or per-step overridden) discrete linear Kalman filter.

    Parameters
    ----------
    transition : array_like, shape (n, n)
        State-transition matrix ``F`` (dimensionless mapping of state to
        state over one step).
    measurement : array_like, shape (m, n)
        Measurement matrix ``H`` [measurement units / state units].
    process_noise : array_like, shape (n, n)
        Process-noise covariance ``Q`` [squared state units per step].
    measurement_noise : array_like, shape (m, m)
        Measurement-noise covariance ``R`` [squared measurement units].
    control : array_like, shape (n, l), optional
        Control matrix ``B``. If omitted, ``predict`` accepts no control.

    Raises
    ------
    ValueError
        On non-square ``F``/``Q``/``R``, shape mismatches, or ``Q``/``R``
        that are not symmetric positive semi-definite / positive definite.
    """

    def __init__(
        self,
        transition: ArrayLike,
        measurement: ArrayLike,
        process_noise: ArrayLike,
        measurement_noise: ArrayLike,
        control: ArrayLike | None = None,
    ) -> None:
        f = _square(transition, None, "F")
        n = f.shape[0]
        h = np.atleast_2d(np.asarray(measurement, dtype=float))
        if h.ndim != 2 or h.shape[1] != n:
            raise ValueError(f"H must have shape (m, {n}), got {h.shape}")
        m = h.shape[0]
        q = _square(process_noise, n, "Q")
        r = _square(measurement_noise, m, "R")

        if np.min(np.linalg.eigvalsh(symmetrize(q))) < -1e-12:
            raise ValueError("Q must be positive semi-definite")
        r_min = np.min(np.linalg.eigvalsh(symmetrize(r)))
        if r_min < 0.0:
            raise ValueError("R must be positive semi-definite")

        b = None
        if control is not None:
            b = np.atleast_2d(np.asarray(control, dtype=float))
            if b.ndim != 2 or b.shape[0] != n:
                raise ValueError(f"B must have shape ({n}, l), got {b.shape}")

        self.transition = f
        self.measurement = h
        self.process_noise = symmetrize(q)
        self.measurement_noise = symmetrize(r)
        self.control = b
        self.n = n
        self.m = m

    # ------------------------------------------------------------------ #
    # Single-step API
    # ------------------------------------------------------------------ #
    def predict(
        self,
        x: ArrayLike,
        p: ArrayLike,
        u: ArrayLike | None = None,
        transition: ArrayLike | None = None,
        process_noise: ArrayLike | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """One time-update step: ``x^- = F x + B u``, ``P^- = F P F^T + Q``.

        Parameters
        ----------
        x : array_like, shape (n,)
            Current state mean.
        p : array_like, shape (n, n)
            Current covariance.
        u : array_like, optional
            Control input; requires ``control`` to have been supplied.
        transition, process_noise : array_like, optional
            Per-step overrides of ``F`` and ``Q`` (time-varying systems).

        Returns
        -------
        (ndarray, ndarray)
            Predicted mean and covariance. The covariance is symmetrised.
        """
        f = self.transition if transition is None else _square(transition, self.n, "F")
        q = (
            self.process_noise
            if process_noise is None
            else symmetrize(_square(process_noise, self.n, "Q"))
        )
        x_arr = _vector(x, self.n, "x")
        p_arr = _square(p, self.n, "P")

        x_pred = f @ x_arr
        if u is not None:
            if self.control is None:
                raise ValueError("control input supplied but the filter has no B matrix")
            u_arr = np.asarray(u, dtype=float).reshape(-1)
            if u_arr.size != self.control.shape[1]:
                raise ValueError(
                    f"u must have {self.control.shape[1]} elements, got {u_arr.size}"
                )
            x_pred = x_pred + self.control @ u_arr
        p_pred = symmetrize(f @ p_arr @ f.T + q)
        return x_pred, p_pred

    def update(
        self,
        x: ArrayLike,
        p: ArrayLike,
        z: ArrayLike,
        measurement: ArrayLike | None = None,
        measurement_noise: ArrayLike | None = None,
    ) -> UpdateResult:
        """One measurement-update step using the Joseph-form covariance update.

        Parameters
        ----------
        x : array_like, shape (n,)
            Prior (predicted) state mean.
        p : array_like, shape (n, n)
            Prior covariance.
        z : array_like, shape (m,)
            Measurement [measurement units].
        measurement, measurement_noise : array_like, optional
            Per-step overrides of ``H`` and ``R``.

        Returns
        -------
        UpdateResult

        Notes
        -----
        ``S`` is inverted via :func:`numpy.linalg.solve` rather than an
        explicit inverse. With ``R = 0`` and an observable ``H``, ``S`` may
        be singular; a :class:`numpy.linalg.LinAlgError` is re-raised as a
        ``ValueError`` naming the likely cause.
        """
        h = self.measurement if measurement is None else np.atleast_2d(
            np.asarray(measurement, dtype=float)
        )
        if h.shape[1] != self.n:
            raise ValueError(f"H must have shape (m, {self.n}), got {h.shape}")
        m = h.shape[0]
        r = (
            self.measurement_noise
            if measurement_noise is None
            else symmetrize(_square(measurement_noise, m, "R"))
        )
        x_arr = _vector(x, self.n, "x")
        p_arr = _square(p, self.n, "P")
        z_arr = _vector(z, m, "z")

        innovation = z_arr - h @ x_arr
        s = symmetrize(h @ p_arr @ h.T + r)
        try:
            gain = np.linalg.solve(s, h @ p_arr).T
            s_inv_y = np.linalg.solve(s, innovation)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
            raise ValueError(
                "innovation covariance S is singular; check for R = 0 combined with "
                "a collapsed prior covariance, or duplicated rows in H"
            ) from exc

        x_post = x_arr + gain @ innovation
        p_post = joseph_update(p_arr, gain, h, r)
        nis = float(innovation @ s_inv_y)
        return UpdateResult(x_post, p_post, gain, innovation, s, nis)

    # ------------------------------------------------------------------ #
    # Batch API
    # ------------------------------------------------------------------ #
    def filter(
        self,
        x0: ArrayLike,
        p0: ArrayLike,
        measurements: ArrayLike,
        controls: ArrayLike | None = None,
    ) -> FilterResult:
        """Run predict/update over a measurement sequence.

        Parameters
        ----------
        x0 : array_like, shape (n,)
            Initial state mean (before the first prediction).
        p0 : array_like, shape (n, n)
            Initial covariance.
        measurements : array_like, shape (T, m)
            Measurement sequence. ``T`` predict/update pairs are run: the
            filter predicts from ``k-1`` to ``k`` and then updates with
            ``measurements[k]``.
        controls : array_like, shape (T, l), optional
            Control sequence; ``controls[k]`` is applied in the prediction
            that produces step ``k``.

        Returns
        -------
        FilterResult
        """
        x = _vector(x0, self.n, "x0")
        p = _square(p0, self.n, "P0")
        zs = np.atleast_2d(np.asarray(measurements, dtype=float))
        if zs.ndim != 2 or zs.shape[1] != self.m:
            raise ValueError(f"measurements must have shape (T, {self.m}), got {zs.shape}")
        t = zs.shape[0]
        if t == 0:
            raise ValueError("measurements must contain at least one time step")
        us = None
        if controls is not None:
            us = np.atleast_2d(np.asarray(controls, dtype=float))
            if us.shape[0] != t:
                raise ValueError("controls must have the same number of steps as measurements")

        x_prior = np.empty((t, self.n))
        p_prior = np.empty((t, self.n, self.n))
        x_post = np.empty((t, self.n))
        p_post = np.empty((t, self.n, self.n))
        gains = np.empty((t, self.n, self.m))
        innov = np.empty((t, self.m))
        innov_cov = np.empty((t, self.m, self.m))
        nis = np.empty(t)

        for k in range(t):
            u = None if us is None else us[k]
            x, p = self.predict(x, p, u=u)
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

        transitions = np.repeat(self.transition[None, :, :], t, axis=0)
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


def steady_state(
    transition: ArrayLike,
    measurement: ArrayLike,
    process_noise: ArrayLike,
    measurement_noise: ArrayLike,
    max_iter: int = 100_000,
    tol: float = 1e-14,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    r"""Iterate the Riccati recursion to its fixed point.

    Solves the discrete-time algebraic Riccati equation (DARE) of filtering

    .. math::
        P^{-} = F \\left[P^{-} - P^{-}H^{\\mathsf{T}}
                (H P^{-} H^{\\mathsf{T}} + R)^{-1} H P^{-}\\right]
                F^{\\mathsf{T}} + Q

    by simple fixed-point iteration of the filter's own predict/update
    covariance recursion, started from ``P = Q``.

    Parameters
    ----------
    transition, measurement, process_noise, measurement_noise : array_like
        ``F``, ``H``, ``Q``, ``R`` as in :class:`KalmanFilter`.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence threshold on ``max |P_k - P_{k-1}|`` (absolute).

    Returns
    -------
    p_prior : ndarray, shape (n, n)
        Steady-state predicted covariance :math:`P^{-}_{\\infty}`.
    p_post : ndarray, shape (n, n)
        Steady-state posterior covariance :math:`P^{+}_{\\infty}`.
    gain : ndarray, shape (n, m)
        Steady-state Kalman gain :math:`K_{\\infty}`.
    iterations : int
        Iterations used.

    Notes
    -----
    Convergence to a unique stabilising solution requires ``(F, H)``
    detectable and ``(F, Q^{1/2})`` stabilisable (Bar-Shalom, Rong Li &
    Kirubarajan 2001, Ch. 5; Simon 2006, Ch. 7). Neither condition is
    checked here: non-convergence is reported by raising ``RuntimeError``
    after ``max_iter``. Fixed-point iteration is used deliberately -- it is
    the same recursion the filter runs, so agreement with the filter's
    converged gain is a check of the *filter*, not of a separate solver.
    The independent hand solution is given in ``validation/VALIDATION.md``.
    """
    kf = KalmanFilter(transition, measurement, process_noise, measurement_noise)
    p_post = np.array(kf.process_noise, dtype=float)
    x = np.zeros(kf.n)
    z = np.zeros(kf.m)
    gain = np.zeros((kf.n, kf.m))
    p_prior = p_post
    for i in range(1, max_iter + 1):
        _, p_prior = kf.predict(x, p_post)
        res = kf.update(x, p_prior, z)
        delta = float(np.max(np.abs(res.p - p_post)))
        p_post = res.p
        gain = res.gain
        if delta < tol:
            return p_prior, p_post, gain, i
    raise RuntimeError(
        f"Riccati iteration did not converge in {max_iter} iterations "
        "(check detectability of (F, H) and stabilisability of (F, Q^1/2))"
    )
