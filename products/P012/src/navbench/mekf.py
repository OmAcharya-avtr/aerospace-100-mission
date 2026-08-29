"""Multiplicative extended Kalman filter (MEKF) for attitude with a quaternion state.

The MEKF carries the attitude as a **unit quaternion reference** ``q̂`` that is
never part of the filter state, plus a small-angle **error state**.  This
avoids the singular covariance that a 4-component quaternion state would have
(the unit-norm constraint makes any 4×4 attitude covariance rank 3) and
removes the need to renormalise a state vector inside the Kalman update.

STATE (6 elements)

    x = [ a (3) ; Δb (3) ]      a  : attitude error rotation vector [rad]
                                Δb : gyro bias error [rad/s]

with the multiplicative error definition (body-side, matching
:mod:`navbench.attitude`)

    q_true = q̂ ⊗ δq(a),   δq(a) = [cos(|a|/2), sin(|a|/2) a/|a|]

ERROR DYNAMICS (Lefferts, Markley & Shuster 1982, "Kalman Filtering for
Spacecraft Attitude Estimation", *J. Guidance, Control, and Dynamics* 5(5),
417-429, §III; Markley & Crassidis 2014, *Fundamentals of Spacecraft Attitude
Determination and Control*, §6.2.4):

    ȧ  = −[ω̂×] a − Δb − η_v            ω̂ = ω_gyro − b̂
    Δḃ = η_u

so the continuous system matrix is ``F_c = [[−[ω̂×], −I], [0, 0]]`` and the
discrete transition over ``Δt`` with ω̂ held constant is exactly

    Φ₁₁ = exp(−[ω̂×]Δt) = I − sinθ [u×] + (1−cosθ)[u×]²      θ = |ω̂|Δt, u = ω̂/|ω̂|
    Φ₁₂ = −( I Δt − [ω̂×](1−cosθ)/|ω̂|² + [ω̂×]²(θ−sinθ)/|ω̂|³ )
    Φ₂₁ = 0,  Φ₂₂ = I

(``Φ₁₂ = −∫₀^{Δt} exp(−[ω̂×]s) ds``.)  Below ``θ = 1e-2`` the three scalar
coefficients are evaluated by their Taylor series instead, because
``1 − cos θ`` and ``θ − sin θ`` lose roughly ``log₁₀(1/θ²)`` significant
digits to cancellation as ``θ → 0``; the series truncate at ``O(θ⁶)``
relative, and the two errors cross at about ``θ = 1e-2`` (both ≈1e-12).

PROCESS NOISE (Farrenkopf, R. L. (1978), "Analytic Steady-State Accuracy
Solutions for Two Common Spacecraft Attitude Estimators", *J. Guidance and
Control* 1(4), 282-284; Markley & Crassidis 2014 Eq. (6.93)):

    Q_d = [[ (σ_v² Δt + σ_u² Δt³/3) I ,  −(σ_u² Δt²/2) I ],
           [ −(σ_u² Δt²/2) I         ,   σ_u² Δt I        ]]

with σ_v the gyro angle random walk [rad/s^{1/2}] and σ_u the rate random
walk [rad/s^{3/2}] of :mod:`navbench.sensors`.

MEASUREMENTS
* **Unit vector** (star tracker line of sight, sun sensor): predicted body
  vector ``r̂_b = R(q̂)ᵀ r_i``; sensitivity ``H = [ [r̂_b×] , 0 ]``.  The QUEST
  measurement covariance ``σ²(I − r̂_b r̂_bᵀ)`` (Shuster & Oh 1981) is rank 2 —
  rotation about the line of sight is unobservable from one vector — so the
  filter uses the regularised ``R = σ² I``, the standard treatment.  With two
  or more non-parallel vectors the three attitude degrees of freedom are
  observable.
* **Full attitude quaternion**: innovation ``ν = rotvec(q̂* ⊗ q_meas)``
  [rad], ``H = [I, 0]``, ``R = σ² I``.

RESET.  After every update the estimated error is folded multiplicatively
into the reference and the error state is zeroed:

    q̂⁺ = normalize( q̂ ⊗ δq(â) ),   b̂⁺ = b̂ + Δb̂,   x⁺ = 0

The covariance is *not* transformed by the reset.  The exact reset would
apply the first-order Jacobian ``I − ½[â×]`` to the attitude block; that
correction is O(|â|) relative and is negligible for the sub-milliradian
errors the filter operates at (Markley 2003, "Attitude Error Representations
for Kalman Filtering", *J. Guidance, Control, and Dynamics* 26(2), 311-317,
§V).  The magnitude of the neglected term is measured and reported in
``validation/v4_mekf_quaternion.py``.

RELATED PRIOR ART.  Product P007 (QuatKit) provides the quaternion algebra
that this filter's reset step exercises.  NavBench re-implements that algebra
independently in :mod:`navbench.attitude` (nothing is imported across
products) and validates its own reset behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .attitude import (
    dcm_from_quat,
    quat_conjugate,
    quat_from_small_angle,
    quat_multiply,
    quat_normalize,
    skew,
    small_angle_from_quat,
)
from .kf import CovarianceCollapseError, joseph_update, symmetrize

__all__ = [
    "attitude_state_transition",
    "gyro_process_noise",
    "MekfResult",
    "MultiplicativeEKF",
]


def attitude_state_transition(omega_hat: ArrayLike, dt: float) -> NDArray[np.float64]:
    """Discrete 6×6 error-state transition ``Φ`` for a constant ``ω̂`` over ``dt``.

    Parameters
    ----------
    omega_hat : array_like, shape (3,)
        Bias-corrected body rate [rad/s].
    dt : float
        Step [s], must be finite (may be zero, giving ``Φ = I`` except for the
        ``−I Δt`` coupling, which then also vanishes).

    Returns
    -------
    ndarray, shape (6, 6)
    """
    w = np.asarray(omega_hat, dtype=float).reshape(3)
    if not np.all(np.isfinite(w)):
        raise ValueError("omega_hat must be finite")
    step = float(dt)
    if not np.isfinite(step):
        raise ValueError(f"dt must be finite, got {dt!r}")
    wx = skew(w)
    wx2 = wx @ wx
    norm = float(np.linalg.norm(w))
    theta = norm * step
    eye = np.eye(3)
    # Three scalar coefficients carry all the theta dependence:
    #   a = sin(theta)/|w|        -> dt (1 - t^2/6  + t^4/120)
    #   b = (1 - cos theta)/|w|^2 -> dt^2/2 (1 - t^2/12 + t^4/360)
    #   c = (theta - sin theta)/|w|^3 -> dt^3/6 (1 - t^2/20 + t^4/840)
    # The closed forms suffer catastrophic cancellation as theta -> 0
    # (1 - cos theta loses ~log10(1/theta^2) digits), while the series truncate
    # at O(theta^6) relative. Both errors are ~1e-12 relative at theta = 1e-2,
    # which is therefore the crossover. An earlier version switched at 1e-8 and
    # was caught by tests/test_properties.py::test_transition_composes_over_two_half_steps
    # at theta = 4.3e-7, where the closed form was wrong by 5e-10 absolute.
    t2 = theta * theta
    if theta < 1e-2:
        a = step * (1.0 - t2 / 6.0 + t2 * t2 / 120.0)
        b = 0.5 * step * step * (1.0 - t2 / 12.0 + t2 * t2 / 360.0)
        c = step**3 / 6.0 * (1.0 - t2 / 20.0 + t2 * t2 / 840.0)
    else:
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        a = sin_t / norm
        b = (1.0 - cos_t) / (norm * norm)
        c = (theta - sin_t) / norm**3
    phi11 = eye - a * wx + b * wx2
    phi12 = -(eye * step - b * wx + c * wx2)
    phi = np.zeros((6, 6))
    phi[:3, :3] = phi11
    phi[:3, 3:] = phi12
    phi[3:, 3:] = eye
    return phi


def gyro_process_noise(sigma_v: float, sigma_u: float, dt: float) -> NDArray[np.float64]:
    """Discrete 6×6 process-noise covariance ``Q_d`` (Farrenkopf 1978).

    Parameters
    ----------
    sigma_v : float
        Angle random walk [rad/s^{1/2}], ≥ 0.
    sigma_u : float
        Rate random walk [rad/s^{3/2}], ≥ 0.
    dt : float
        Step [s], > 0.

    Units of the blocks: attitude rad², cross rad²/s, bias rad²/s².
    """
    sv, su = float(sigma_v), float(sigma_u)
    step = float(dt)
    for name, val in (("sigma_v", sv), ("sigma_u", su)):
        if not np.isfinite(val) or val < 0.0:
            raise ValueError(f"{name} must be finite and >= 0, got {val!r}")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError(f"dt must be finite and > 0, got {dt!r}")
    eye = np.eye(3)
    q = np.zeros((6, 6))
    q[:3, :3] = (sv * sv * step + su * su * step**3 / 3.0) * eye
    q[:3, 3:] = -(su * su * step * step / 2.0) * eye
    q[3:, :3] = q[:3, 3:]
    q[3:, 3:] = (su * su * step) * eye
    return symmetrize(q)


@dataclass
class MekfResult:
    """Per-step MEKF histories.

    Attributes
    ----------
    t : ndarray, shape (N,)  [s]
    quat : ndarray, shape (N, 4)
        Reference quaternion after the step's update+reset.
    bias : ndarray, shape (N, 3)  [rad/s]
    covariance : ndarray, shape (N, 6, 6)
        Posterior error-state covariance (attitude rad², bias rad²/s²).
    innovation : ndarray, shape (N, m); NaN on steps with no measurement.
    innovation_cov : ndarray, shape (N, m, m)
    nis : ndarray, shape (N,); NaN on steps with no measurement.
    updated : ndarray of bool, shape (N,)
    reset_angle : ndarray, shape (N,)
        ``|â|`` folded into the reference at each reset [rad]; 0 where no
        update occurred.
    """

    t: NDArray[np.float64]
    quat: NDArray[np.float64]
    bias: NDArray[np.float64]
    covariance: NDArray[np.float64]
    innovation: NDArray[np.float64]
    innovation_cov: NDArray[np.float64]
    nis: NDArray[np.float64]
    updated: NDArray[np.bool_]
    reset_angle: NDArray[np.float64]

    def attitude_error(self, quat_true: ArrayLike) -> NDArray[np.float64]:
        """Attitude error rotation vectors ``rotvec(q̂* ⊗ q_true)`` [rad], shape (N, 3)."""
        qt = np.atleast_2d(np.asarray(quat_true, dtype=float))
        if qt.shape != self.quat.shape:
            raise ValueError(f"quat_true must have shape {self.quat.shape}, got {qt.shape}")
        return np.array(
            [
                small_angle_from_quat(quat_multiply(quat_conjugate(qh), quat_normalize(q)))
                for qh, q in zip(self.quat, qt, strict=True)
            ]
        )

    def error_state(self, quat_true: ArrayLike, bias_true: ArrayLike) -> NDArray[np.float64]:
        """Full 6-element error ``[a; b_true − b̂]``, shape (N, 6)."""
        bt = np.atleast_2d(np.asarray(bias_true, dtype=float))
        if bt.shape != self.bias.shape:
            raise ValueError(f"bias_true must have shape {self.bias.shape}, got {bt.shape}")
        return np.hstack([self.attitude_error(quat_true), bt - self.bias])


@dataclass
class MultiplicativeEKF:
    """Six-state MEKF: attitude error (3) + gyro bias error (3).

    Parameters
    ----------
    sigma_v, sigma_u : float
        Gyro ARW [rad/s^{1/2}] and RRW [rad/s^{3/2}] used to build ``Q_d``.
    dt : float
        Nominal propagation step [s], > 0.
    quat0 : array_like, shape (4,)
        Initial reference attitude (normalised on entry).
    bias0 : array_like, shape (3,), optional
        Initial gyro bias estimate [rad/s].  Default zero.
    p0 : array_like, shape (6, 6), optional
        Initial error covariance.  Default ``diag((0.1 rad)² ×3, (1e-4 rad/s)² ×3)``.
    q_scale : float, optional
        Multiplier applied to ``Q_d`` — the single knob the adaptive schemes in
        :mod:`navbench.adaptive` tune.  Default 1.0.

    Raises
    ------
    ValueError
        On invalid shapes, non-finite values, or a non-positive-semi-definite
        ``p0``.
    """

    sigma_v: float
    sigma_u: float
    dt: float
    quat0: ArrayLike
    bias0: ArrayLike = (0.0, 0.0, 0.0)
    p0: ArrayLike | None = None
    q_scale: float = 1.0
    quat: NDArray[np.float64] = field(init=False)
    bias: NDArray[np.float64] = field(init=False)
    p: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        self.sigma_v = float(self.sigma_v)
        self.sigma_u = float(self.sigma_u)
        self.dt = float(self.dt)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"dt must be finite and > 0, got {self.dt!r}")
        for name, val in (("sigma_v", self.sigma_v), ("sigma_u", self.sigma_u)):
            if not np.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {val!r}")
        qs = float(self.q_scale)
        if not np.isfinite(qs) or qs <= 0.0:
            raise ValueError(f"q_scale must be finite and > 0, got {self.q_scale!r}")
        self.q_scale = qs
        self.quat = quat_normalize(self.quat0)
        b = np.asarray(self.bias0, dtype=float).reshape(3)
        if not np.all(np.isfinite(b)):
            raise ValueError("bias0 must be finite")
        self.bias = b.copy()
        if self.p0 is None:
            p = np.diag([0.1**2] * 3 + [1e-4**2] * 3)
        else:
            p = np.atleast_2d(np.asarray(self.p0, dtype=float))
            if p.shape != (6, 6):
                raise ValueError(f"p0 must have shape (6, 6), got {p.shape}")
            if not np.all(np.isfinite(p)):
                raise ValueError("p0 must be finite")
            scale = max(1.0, float(np.max(np.abs(p))))
            if float(np.max(np.abs(p - p.T))) > 1e-9 * scale:
                raise ValueError("p0 must be symmetric")
            if float(np.linalg.eigvalsh(0.5 * (p + p.T)).min()) < -1e-12 * scale:
                raise ValueError("p0 must be positive semi-definite")
        self.p = symmetrize(p)

    @property
    def process_noise(self) -> NDArray[np.float64]:
        """Current ``q_scale · Q_d(σ_v, σ_u, dt)``, shape (6, 6)."""
        return self.q_scale * gyro_process_noise(self.sigma_v, self.sigma_u, self.dt)

    def predict(self, omega_meas: ArrayLike, dt: float | None = None) -> None:
        """Propagate the reference attitude and the error covariance one step.

        Parameters
        ----------
        omega_meas : array_like, shape (3,)
            Raw gyro measurement [rad/s]; the current bias estimate is removed
            internally.
        dt : float, optional
            Step override [s].
        """
        step = self.dt if dt is None else float(dt)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError(f"dt must be finite and > 0, got {dt!r}")
        w = np.asarray(omega_meas, dtype=float).reshape(3)
        if not np.all(np.isfinite(w)):
            raise ValueError("omega_meas must be finite")
        omega_hat = w - self.bias
        self.quat = quat_normalize(
            quat_multiply(self.quat, quat_from_small_angle(omega_hat * step))
        )
        phi = attitude_state_transition(omega_hat, step)
        qd = self.q_scale * gyro_process_noise(self.sigma_v, self.sigma_u, step)
        self.p = symmetrize(phi @ self.p @ phi.T + qd)

    def _update_linear(
        self, nu: NDArray[np.float64], h: NDArray[np.float64], r: NDArray[np.float64]
    ) -> dict[str, object]:
        s = symmetrize(h @ self.p @ h.T + r)
        try:
            np.linalg.cholesky(s)
        except np.linalg.LinAlgError as exc:
            raise CovarianceCollapseError(
                "MEKF innovation covariance is not positive definite "
                f"(min eigenvalue {float(np.linalg.eigvalsh(s).min()):.3e})"
            ) from exc
        gain = np.linalg.solve(s, (self.p @ h.T).T).T
        dx = gain @ nu
        self.p = joseph_update(self.p, gain, h, r)
        reset_angle = float(np.linalg.norm(dx[:3]))
        self.quat = quat_normalize(quat_multiply(self.quat, quat_from_small_angle(dx[:3])))
        self.bias = self.bias + dx[3:]
        return {
            "innovation": nu,
            "innovation_cov": s,
            "gain": gain,
            "nis": float(nu @ np.linalg.solve(s, nu)),
            "reset_angle": reset_angle,
            "dx": dx,
        }

    def update_vectors(
        self,
        body_vectors: ArrayLike,
        reference_vectors: ArrayLike,
        sigma_rad: float,
    ) -> dict[str, object]:
        """Update from one or more unit-vector observations.

        Parameters
        ----------
        body_vectors : array_like, shape (M, 3)
            Measured unit vectors in body axes.
        reference_vectors : array_like, shape (M, 3)
            Corresponding known inertial unit vectors.
        sigma_rad : float
            Per-axis measurement noise [rad], > 0.  ``R = σ² I_{3M}`` — the
            regularised form of the rank-2 QUEST covariance (see module
            docstring).
        """
        b = np.atleast_2d(np.asarray(body_vectors, dtype=float))
        r_i = np.atleast_2d(np.asarray(reference_vectors, dtype=float))
        if b.ndim != 2 or b.shape[1] != 3:
            raise ValueError(f"body_vectors must have shape (M, 3), got {b.shape}")
        if r_i.shape != b.shape:
            raise ValueError(
                f"reference_vectors must have shape {b.shape}, got {r_i.shape}"
            )
        if b.shape[0] == 0:
            raise ValueError("at least one vector observation is required")
        if not (np.all(np.isfinite(b)) and np.all(np.isfinite(r_i))):
            raise ValueError("body_vectors and reference_vectors must be finite")
        sig = float(sigma_rad)
        if not np.isfinite(sig) or sig <= 0.0:
            raise ValueError(f"sigma_rad must be finite and > 0, got {sigma_rad!r}")

        rot = dcm_from_quat(self.quat).T  # inertial -> body
        m = b.shape[0]
        h = np.zeros((3 * m, 6))
        nu = np.zeros(3 * m)
        for i in range(m):
            pred = rot @ (r_i[i] / np.linalg.norm(r_i[i]))
            h[3 * i : 3 * i + 3, :3] = skew(pred)
            nu[3 * i : 3 * i + 3] = b[i] / np.linalg.norm(b[i]) - pred
        return self._update_linear(nu, h, sig * sig * np.eye(3 * m))

    def update_quaternion(self, quat_meas: ArrayLike, sigma_rad: float) -> dict[str, object]:
        """Update from a full attitude measurement.

        ``ν = rotvec(q̂* ⊗ q_meas)`` [rad], ``H = [I₃, 0₃]``, ``R = σ² I₃``.
        """
        sig = float(sigma_rad)
        if not np.isfinite(sig) or sig <= 0.0:
            raise ValueError(f"sigma_rad must be finite and > 0, got {sigma_rad!r}")
        dq = quat_multiply(quat_conjugate(self.quat), quat_normalize(quat_meas))
        nu = small_angle_from_quat(dq)
        h = np.hstack([np.eye(3), np.zeros((3, 3))])
        return self._update_linear(nu, h, sig * sig * np.eye(3))

    def run(
        self,
        omega_meas: ArrayLike,
        *,
        quat_meas: ArrayLike | None = None,
        body_vectors: ArrayLike | None = None,
        reference_vectors: ArrayLike | None = None,
        sigma_rad: float = 1e-4,
        measurement_every: int = 1,
        dt: float | None = None,
    ) -> MekfResult:
        """Batch run over a gyro series with periodic attitude updates.

        Parameters
        ----------
        omega_meas : array_like, shape (N, 3)
            Gyro measurements [rad/s], one per step.
        quat_meas : array_like, shape (N, 4), optional
            Attitude measurements; rows of NaN mark a dropout.  Mutually
            exclusive with ``body_vectors``.
        body_vectors : array_like, shape (N, M, 3), optional
            Unit-vector observations per step; NaN marks a dropout.
        reference_vectors : array_like, shape (M, 3), optional
            Required with ``body_vectors``.
        sigma_rad : float
            Measurement noise [rad].
        measurement_every : int
            Apply an update every ``measurement_every`` steps (≥ 1).
        dt : float, optional
            Step override [s].

        Returns
        -------
        MekfResult
        """
        w = np.atleast_2d(np.asarray(omega_meas, dtype=float))
        if w.ndim != 2 or w.shape[1] != 3:
            raise ValueError(f"omega_meas must have shape (N, 3), got {w.shape}")
        n = w.shape[0]
        every = int(measurement_every)
        if every < 1:
            raise ValueError(f"measurement_every must be >= 1, got {measurement_every!r}")
        if (quat_meas is None) == (body_vectors is None):
            raise ValueError("supply exactly one of quat_meas or body_vectors")
        step = self.dt if dt is None else float(dt)

        if quat_meas is not None:
            qm = np.atleast_2d(np.asarray(quat_meas, dtype=float))
            if qm.shape != (n, 4):
                raise ValueError(f"quat_meas must have shape ({n}, 4), got {qm.shape}")
            m_dim = 3
            bv = None
            refs = None
        else:
            bv = np.asarray(body_vectors, dtype=float)
            if bv.ndim != 3 or bv.shape[0] != n or bv.shape[2] != 3:
                raise ValueError(f"body_vectors must have shape ({n}, M, 3), got {bv.shape}")
            if reference_vectors is None:
                raise ValueError("reference_vectors is required with body_vectors")
            refs = np.atleast_2d(np.asarray(reference_vectors, dtype=float))
            if refs.shape != (bv.shape[1], 3):
                raise ValueError(
                    f"reference_vectors must have shape ({bv.shape[1]}, 3), got {refs.shape}"
                )
            m_dim = 3 * bv.shape[1]
            qm = None

        ts = np.arange(1, n + 1) * step
        quats = np.zeros((n, 4))
        biases = np.zeros((n, 3))
        covs = np.zeros((n, 6, 6))
        nus = np.full((n, m_dim), np.nan)
        s_hist = np.zeros((n, m_dim, m_dim))
        nis = np.full(n, np.nan)
        upd = np.zeros(n, dtype=bool)
        reset = np.zeros(n)

        for k in range(n):
            self.predict(w[k], dt=step)
            do_update = (k + 1) % every == 0
            if do_update and qm is not None:
                if np.all(np.isfinite(qm[k])):
                    out = self.update_quaternion(qm[k], sigma_rad)
                else:
                    do_update = False
            elif do_update and bv is not None:
                if np.all(np.isfinite(bv[k])):
                    out = self.update_vectors(bv[k], refs, sigma_rad)
                else:
                    do_update = False
            if do_update:
                nus[k] = out["innovation"]  # type: ignore[assignment]
                s_hist[k] = out["innovation_cov"]  # type: ignore[assignment]
                nis[k] = out["nis"]  # type: ignore[assignment]
                reset[k] = out["reset_angle"]  # type: ignore[assignment]
                upd[k] = True
            quats[k] = self.quat
            biases[k] = self.bias
            covs[k] = self.p
        return MekfResult(ts, quats, biases, covs, nus, s_hist, nis, upd, reset)
