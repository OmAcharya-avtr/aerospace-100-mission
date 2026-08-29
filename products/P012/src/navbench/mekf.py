r"""Multiplicative extended Kalman filter (MEKF) for attitude and gyro bias.

Why multiplicative
------------------
A unit quaternion has four components but only three degrees of freedom. An
additive EKF on the four components carries a singular covariance (the norm
constraint direction has zero variance) and drifts off the unit sphere. The
MEKF keeps a **reference quaternion** ``q̂`` outside the filter state and
estimates a *three*-component multiplicative error, so the covariance is a
genuine 3×3 (here 6×6 with the gyro bias) non-singular matrix.

State and error definition
--------------------------
Reference: ``q̂`` (attitude, body → inertial) and ``b̂`` (gyro bias [rad/s]).
Error state ``δx = [a, δb] ∈ R⁶`` with the **local / body-frame** convention

.. math:: q = \hat q \otimes \delta q(a),\qquad \delta q(a)\approx[1,\ a/2],
          \qquad \delta b = b - \hat b

``a`` is a small rotation vector [rad]; ``δb`` a bias error [rad/s].

Error dynamics (derived, not quoted)
------------------------------------
With the gyro model ``ω̃ = ω + b + n_v``, ``ḃ = n_u`` and the estimate
propagated by ``ω̂ = ω̃ − b̂``, differentiating ``δq = q̂^{*}\otimes q`` and
keeping first-order terms gives

.. math::
    \dot a = -[\hat\omega\times]\,a - \delta b - n_v, \qquad \dot{\delta b} = n_u

which is the standard result of Lefferts, E. J., Markley, F. L. and Shuster,
M. D. (1982), "Kalman Filtering for Spacecraft Attitude Estimation", *Journal
of Guidance, Control, and Dynamics* **5**(5), 417–429. See also Markley, F. L.
and Crassidis, J. L. (2014), *Fundamentals of Spacecraft Attitude Determination
and Control*, Springer, §6.2 (the MEKF), and Crassidis, J. L. and Junkins,
J. L. (2011), *Optimal Estimation of Dynamic Systems*, 2nd ed., CRC Press, §7.1.

Discrete transition (exact for a constant ``ω̂`` over the step, ``θ = ‖ω̂‖Δt``,
``n = ω̂/‖ω̂‖``):

.. math::
    \Phi_{aa} &= e^{-[\hat\omega\times]\Delta t}
        = I - \sin\theta\,[n\times] + (1-\cos\theta)[n\times]^2 \\
    \Phi_{ab} &= -\!\int_0^{\Delta t}\! e^{-[\hat\omega\times]s}\,ds
        = -\Big(I\Delta t - \tfrac{1-\cos\theta}{\|\hat\omega\|}[n\times]
          + \tfrac{\theta-\sin\theta}{\|\hat\omega\|}[n\times]^2\Big) \\
    \Phi_{ba} &= 0,\qquad \Phi_{bb} = I

Series expansions are used below ``θ = 1e-9`` rad to avoid 0/0.

Discrete process noise (van Loan integration of the above with white ``n_v``,
``n_u``; the standard MEKF form given in Crassidis & Junkins 2011 §7.1):

.. math::
    Q_d = \begin{bmatrix}
        (\sigma_v^2\Delta t + \tfrac13\sigma_u^2\Delta t^3) I &
        -\tfrac12\sigma_u^2\Delta t^2 I \\
        -\tfrac12\sigma_u^2\Delta t^2 I & \sigma_u^2\Delta t\, I
    \end{bmatrix}

with ``σ_v`` the angle-random-walk coefficient [rad/√s] and ``σ_u`` the
rate-random-walk coefficient [rad/s^{3/2}]. This form neglects the rotation of
the noise over the step, which is ``O(‖ω̂‖Δt)`` relative — negligible for
``‖ω̂‖Δt ≪ 1``, and the class raises if that is violated by more than 0.1 rad.

Reset
-----
After the measurement update the estimated error is folded into the reference
and the error state is zeroed::

    q̂ ← normalize(q̂ ⊗ δq(a⁺));   b̂ ← b̂ + δb⁺;   δx ← 0

The first-order reset leaves ``P`` unchanged. The optional second-order
covariance reset ``P_aa ← G P_aa Gᵀ`` with ``G = I − ½[a⁺×]`` follows Markley,
F. L. (2003), "Attitude Error Representations for Kalman Filtering", *Journal
of Guidance, Control, and Dynamics* **26**(2), 311–317; it is off by default
because the standard MEKF does not apply it and its effect is
``O(‖a⁺‖²)`` — for the star-tracker accuracies simulated here, below 1e-9
relative.

Units: ``a`` rad, ``δb`` rad/s, ``P`` mixed rad² / (rad/s)² blocks.
Validity: first order in ``‖a‖``; the small-angle error parameterisation is
accurate to better than 0.1 % for ``‖a‖ < 0.1`` rad (≈ 5.7°).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .kf import UpdateInfo
from .linalg import joseph_update, symmetrize
from .quaternion import (
    attitude_matrix,
    quat_conjugate,
    quat_from_rotvec,
    quat_multiply,
    quat_normalize,
    skew,
    small_angle_quat,
)

__all__ = ["MekfConfig", "MultiplicativeEKF", "error_state_transition", "attitude_process_noise"]


def error_state_transition(omega_hat: ArrayLike, dt: float) -> NDArray[np.float64]:
    """6×6 discrete error-state transition ``Φ`` (see module docstring).

    Parameters
    ----------
    omega_hat : array_like, shape (3,)
        Bias-corrected body rate [rad/s].
    dt : float
        Step [s].
    """
    w = np.asarray(omega_hat, dtype=float).reshape(3)
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    wn = float(np.linalg.norm(w))
    phi = np.zeros((6, 6))
    phi[3:, 3:] = np.eye(3)
    if wn * dt < 1e-9:
        phi[:3, :3] = np.eye(3) - skew(w) * dt
        phi[:3, 3:] = -(np.eye(3) * dt - 0.5 * dt * dt * skew(w))
        return phi
    n = w / wn
    kx = skew(n)
    kx2 = kx @ kx
    theta = wn * dt
    phi[:3, :3] = np.eye(3) - np.sin(theta) * kx + (1.0 - np.cos(theta)) * kx2
    integral = (
        np.eye(3) * dt
        - ((1.0 - np.cos(theta)) / wn) * kx
        + ((theta - np.sin(theta)) / wn) * kx2
    )
    phi[:3, 3:] = -integral
    return phi


def attitude_process_noise(sigma_v: float, sigma_u: float, dt: float) -> NDArray[np.float64]:
    """6×6 discrete process noise ``Q_d`` for the MEKF error state.

    Parameters
    ----------
    sigma_v : float
        Angle random walk [rad/√s].
    sigma_u : float
        Rate random walk [rad/s^{3/2}].
    dt : float
        Step [s].
    """
    if sigma_v < 0.0 or sigma_u < 0.0:
        raise ValueError(f"sigma_v and sigma_u must be >= 0, got {sigma_v}, {sigma_u}")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    i3 = np.eye(3)
    q = np.zeros((6, 6))
    q[:3, :3] = (sigma_v ** 2 * dt + sigma_u ** 2 * dt ** 3 / 3.0) * i3
    q[:3, 3:] = -0.5 * sigma_u ** 2 * dt ** 2 * i3
    q[3:, :3] = -0.5 * sigma_u ** 2 * dt ** 2 * i3
    q[3:, 3:] = sigma_u ** 2 * dt * i3
    return q


@dataclass(frozen=True)
class MekfConfig:
    """MEKF configuration.

    Parameters
    ----------
    sigma_v : float
        Gyro angle random walk [rad/√s].
    sigma_u : float
        Gyro rate random walk [rad/s^{3/2}].
    exact_reset : bool
        Use the exact exponential map for the reset rotation instead of the
        first-order ``[1, a/2]``. Both are correct to first order; the exact
        map removes the ``O(‖a‖³)`` term.
    covariance_reset : bool
        Apply Markley's (2003) second-order covariance reset.
    max_omega_dt : float
        Guard on ``‖ω̂‖Δt`` [rad]; the discrete ``Q_d`` above is derived under
        ``‖ω̂‖Δt ≪ 1`` and the class raises above this bound.
    """

    sigma_v: float = 3.0e-5
    sigma_u: float = 1.0e-7
    exact_reset: bool = True
    covariance_reset: bool = False
    max_omega_dt: float = 0.1


@dataclass
class MultiplicativeEKF:
    """MEKF with a 6-state error ``[a, δb]``.

    Parameters
    ----------
    q_ref : array_like, shape (4,)
        Initial reference attitude (scalar-first).
    bias : array_like, shape (3,)
        Initial gyro-bias estimate [rad/s].
    p : array_like, shape (6, 6)
        Initial error covariance (rad² and (rad/s)² blocks).
    config : MekfConfig
    """

    q_ref: NDArray[np.float64]
    bias: NDArray[np.float64]
    p: NDArray[np.float64]
    config: MekfConfig = field(default_factory=MekfConfig)
    history: list[UpdateInfo] = field(default_factory=list, repr=False)
    n_resets: int = 0

    def __post_init__(self) -> None:
        self.q_ref = quat_normalize(self.q_ref)
        self.bias = np.asarray(self.bias, dtype=float).reshape(3)
        self.p = symmetrize(np.atleast_2d(np.asarray(self.p, dtype=float)))
        if self.p.shape != (6, 6):
            raise ValueError(f"p must be 6x6 for the [a, db] error state, got {self.p.shape}")
        if np.linalg.eigvalsh(self.p).min() <= 0.0:
            raise ValueError("initial covariance p must be positive definite")

    def propagate(
        self,
        omega_meas: ArrayLike,
        dt: float,
        q_scale: float = 1.0,
        omega_prev: ArrayLike | None = None,
    ) -> None:
        r"""Propagate with a gyro sample.

        Parameters
        ----------
        omega_meas : array_like, shape (3,)
            Measured body rate ``ω̃`` at the **end** of the step [rad/s].
        dt : float
            Step [s].
        q_scale : float
            Multiplier on the process-noise covariance, used by the adaptive
            schemes in :mod:`navbench.adaptive` and :mod:`navbench.ai`.
        omega_prev : array_like, shape (3,) or None
            Measured rate at the **start** of the step. When supplied, the
            reference attitude is advanced with the trapezoidal average
            ``½(ω̃_prev + ω̃) − b̂`` instead of the single end-point sample.

        Notes
        -----
        **Integration order matters more than most texts admit.** Advancing the
        attitude with a single rate sample held constant across the step
        (rectangular integration) leaves a deterministic ``O(‖ω̇‖Δt²)`` error
        per step that accumulates coherently — it is not white, so no ``Q`` can
        absorb it, and it silently destroys filter consistency. Measured on the
        torque-free trajectory of ``validation/v4_attitude_mekf.py``
        (``J = diag(10, 7, 5) kg m²``, ``ω(0) = (0.02, −0.01, 0.03) rad/s``,
        200 s): rectangular integration at ``Δt = 0.1 s`` drifts **180.9
        arcsec**, trapezoidal **0.173 arcsec** — a factor of 1046. Always pass
        ``omega_prev`` unless you are deliberately studying the effect.

        The trapezoidal average halves the white-noise variance contributed by
        the step, but because consecutive steps share samples the accumulated
        angle variance over ``N`` steps is ``σ_v²Δt(N − ½)``, i.e. the same as
        rectangular integration to within half a step. ``Q_d`` is therefore
        unchanged.
        """
        wm = np.asarray(omega_meas, dtype=float).reshape(3)
        if omega_prev is not None:
            wm = 0.5 * (wm + np.asarray(omega_prev, dtype=float).reshape(3))
        w_hat = wm - self.bias
        wdt = float(np.linalg.norm(w_hat)) * dt
        if wdt > self.config.max_omega_dt:
            raise ValueError(
                f"‖ω̂‖·dt = {wdt:.4f} rad exceeds max_omega_dt = {self.config.max_omega_dt}; "
                "the discrete process-noise model is not valid there — reduce dt"
            )
        if q_scale <= 0.0:
            raise ValueError(f"q_scale must be > 0, got {q_scale}")
        phi = error_state_transition(w_hat, dt)
        qd = attitude_process_noise(self.config.sigma_v, self.config.sigma_u, dt) * q_scale
        self.p = symmetrize(phi @ self.p @ phi.T + qd)
        self.q_ref = quat_normalize(quat_multiply(self.q_ref, quat_from_rotvec(w_hat * dt)))

    def _apply_reset(self, dx: NDArray[np.float64]) -> None:
        a = dx[:3]
        dq = quat_from_rotvec(a) if self.config.exact_reset else small_angle_quat(a)
        self.q_ref = quat_normalize(quat_multiply(self.q_ref, dq))
        self.bias = self.bias + dx[3:]
        if self.config.covariance_reset:
            g = np.eye(6)
            g[:3, :3] = np.eye(3) - 0.5 * skew(a)
            self.p = symmetrize(g @ self.p @ g.T)
        self.n_resets += 1

    def update_quaternion(self, q_meas: ArrayLike, r: ArrayLike) -> UpdateInfo:
        r"""Update with a star-tracker quaternion measurement.

        The residual is ``z = 2\,\mathrm{vec}(\hat q^{*}\otimes\tilde q)``, which
        to first order equals ``a + n`` with ``n`` the body-axis tracker noise;
        hence ``H = [I₃ | 0₃]``. Source: Markley & Crassidis (2014) §6.2.
        """
        qm = quat_normalize(q_meas)
        rr = symmetrize(np.atleast_2d(np.asarray(r, dtype=float)))
        if rr.shape != (3, 3):
            raise ValueError(f"r must be 3x3 for a quaternion measurement, got {rr.shape}")
        dq = quat_multiply(quat_conjugate(self.q_ref), qm)
        if dq[0] < 0.0:
            dq = -dq
        z = 2.0 * dq[1:]
        h = np.hstack((np.eye(3), np.zeros((3, 3))))
        return self._linear_update(z, h, rr)

    def update_vector(self, b_meas: ArrayLike, ref_inertial: ArrayLike, r: ArrayLike) -> UpdateInfo:
        r"""Update with a body-frame unit-vector observation of a known direction.

        Predicted observation ``b̂ = A(q̂) r``. With ``q = q̂ ⊗ δq(a)`` and
        ``A(δq) ≈ I − [a×]``, the true observation is
        ``b = (I − [a×]) b̂ = b̂ + [b̂×] a``, so ``H = [[b̂×] | 0₃]``.
        The sign of this Jacobian is verified against a central difference in
        ``tests/test_mekf.py``.
        """
        bm = np.asarray(b_meas, dtype=float).reshape(3)
        ref = np.asarray(ref_inertial, dtype=float).reshape(3)
        ref = ref / np.linalg.norm(ref)
        rr = symmetrize(np.atleast_2d(np.asarray(r, dtype=float)))
        if rr.shape != (3, 3):
            raise ValueError(f"r must be 3x3 for a vector measurement, got {rr.shape}")
        b_hat = attitude_matrix(self.q_ref) @ ref
        h = np.hstack((skew(b_hat), np.zeros((3, 3))))
        return self._linear_update(bm - b_hat, h, rr)

    def _linear_update(
        self, z: NDArray[np.float64], h: NDArray[np.float64], r: NDArray[np.float64]
    ) -> UpdateInfo:
        # The error state is identically zero before the update (it is reset
        # after every update), so the innovation is z itself.
        s = symmetrize(h @ self.p @ h.T + r)
        k = np.linalg.solve(s.T, (self.p @ h.T).T).T
        dx = k @ z
        self.p = joseph_update(self.p, k, h, r)
        self._apply_reset(dx)
        lo = np.linalg.cholesky(s)
        y = np.linalg.solve(lo, z)
        info = UpdateInfo(innovation=z, innovation_cov=s, gain=k, nis=float(np.dot(y, y)))
        self.history.append(info)
        return info

    def attitude_error(self, q_true: ArrayLike) -> NDArray[np.float64]:
        r"""Three-component attitude error ``a`` [rad] w.r.t. a truth quaternion.

        ``a = rotvec(q̂^{*} ⊗ q_true)``, the same quantity the 3×3 attitude block
        of ``P`` describes — this is what NEES must be computed from.
        """
        from .quaternion import quat_to_rotvec

        return quat_to_rotvec(quat_multiply(quat_conjugate(self.q_ref), quat_normalize(q_true)))

    def error_state(self, q_true: ArrayLike, bias_true: ArrayLike) -> NDArray[np.float64]:
        """Full 6-vector error ``[a, b − b̂]`` for NEES scoring."""
        return np.concatenate(
            (self.attitude_error(q_true), np.asarray(bias_true, dtype=float).reshape(3) - self.bias)
        )
