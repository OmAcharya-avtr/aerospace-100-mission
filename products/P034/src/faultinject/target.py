"""Reference target: a GNC loop used as the system under test.

The target is deliberately small and fully analytic, because the point of this
package is the *injection and campaign machinery*, and a target with a closed
form is the only kind against which an injected fault's effect can be predicted
on paper (see ``validation/validate_analytic_bias.py``).

Plant
-----
Double integrator (point mass commanded in acceleration), discretised exactly
under zero-order hold with step ``dt``:

    x_{k+1} = A x_k + B u_k + w_k,
    A = [[1, dt], [0, 1]],   B = [[dt^2/2], [dt]]

Units: position ``x[0]`` in metres, velocity ``x[1]`` in metres per second,
``u`` in metres per second squared.  This is the standard constant-acceleration
kinematic model; see Y. Bar-Shalom, X.-R. Li and T. Kirubarajan, *Estimation
with Applications to Tracking and Navigation*, Wiley (2001), Sec. 6.2.2, Eq.
(6.2.2-3) for ``A`` and the matching discrete process-noise covariance

    Q = sigma_a^2 * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]].

Estimator
---------
Fixed-gain (steady-state) Kalman filter with ``H = I`` (position and velocity
are both measured).  The gain is the converged solution of the discrete-time
Riccati recursion

    P^-_{k}  = A P_{k-1} A^T + Q
    K_k      = P^-_k H^T (H P^-_k H^T + R)^{-1}
    P_k      = (I - K_k H) P^-_k

iterated to a fixed point (Bar-Shalom et al. 2001, Sec. 5.2).  The gain is
frozen at that fixed point so that the filter is a *linear time-invariant*
operator, which is what makes the analytic bias/innovation result exact rather
than asymptotic.

Innovation
----------
``e_k = z_k - H x^-_k``, in channel units.  Recorded for every step on which a
measurement update actually occurred.

Controller
----------
PD on the estimate, tracking a sinusoidal position reference, with symmetric
command saturation:

    u_k = clip(kp (r_k - p^_k) - kd v^_k, -u_max, +u_max)

The control law is irrelevant to the analytic innovation result: ``u_k`` is
applied to the plant and used by the predictor, so it cancels in the
prediction-error recursion.  That is stated and checked in the validation
script.

Not flight software.  This is a textbook loop chosen for analysability.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

DT = 0.02
"""Loop step, seconds."""

SIGMA_POS = 0.05
"""Position measurement noise standard deviation, metres."""

SIGMA_VEL = 0.02
"""Velocity measurement noise standard deviation, metres per second."""

SIGMA_ACCEL = 0.5
"""Process (acceleration) noise standard deviation, metres per second squared."""

KP = 36.0
"""Proportional gain, per second squared (omega_n = 6 rad/s)."""

KD = 8.4
"""Derivative gain, per second (zeta = 0.70)."""

U_MAX = 30.0
"""Command saturation limit, metres per second squared."""

REF_AMPLITUDE = 1.0
"""Position reference amplitude, metres."""

REF_FREQ_HZ = 0.2
"""Position reference frequency, hertz."""

N_STEPS = 150
"""Default campaign run length in steps (3.0 s at DT = 0.02 s)."""


def plant_matrices(dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
    """``(A, B)`` of the zero-order-hold double integrator. Units: m, m/s, m/s^2."""
    a = np.array([[1.0, dt], [0.0, 1.0]])
    b = np.array([[0.5 * dt * dt], [dt]])
    return a, b


def process_covariance(dt: float = DT, sigma_a: float = SIGMA_ACCEL) -> np.ndarray:
    """Discrete process-noise covariance (Bar-Shalom et al. 2001, Sec. 6.2.2)."""
    return (sigma_a**2) * np.array(
        [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]]
    )


def measurement_covariance(
    sigma_p: float = SIGMA_POS, sigma_v: float = SIGMA_VEL
) -> np.ndarray:
    """Measurement noise covariance ``R = diag(sigma_p^2, sigma_v^2)``."""
    return np.diag([sigma_p**2, sigma_v**2])


def steady_state_gain(
    dt: float = DT,
    sigma_a: float = SIGMA_ACCEL,
    sigma_p: float = SIGMA_POS,
    sigma_v: float = SIGMA_VEL,
    tol: float = 1e-15,
    max_iter: int = 2000,
) -> tuple[np.ndarray, int, float]:
    """Converged Kalman gain of the discrete Riccati recursion.

    Returns ``(K, iterations, final_change)`` where ``final_change`` is the
    max-norm change of ``K`` on the last iteration.  Raises ``RuntimeError`` if
    the recursion has not converged to ``tol`` within ``max_iter`` steps, rather
    than returning an unconverged gain.
    """
    a, _ = plant_matrices(dt)
    q = process_covariance(dt, sigma_a)
    r = measurement_covariance(sigma_p, sigma_v)
    h = np.eye(2)
    p = q.copy()
    k_prev = np.zeros((2, 2))
    for i in range(1, max_iter + 1):
        p_pred = a @ p @ a.T + q
        k = p_pred @ h.T @ np.linalg.inv(h @ p_pred @ h.T + r)
        p = (np.eye(2) - k @ h) @ p_pred
        change = float(np.max(np.abs(k - k_prev)))
        if change < tol:
            return k, i, change
        k_prev = k
    raise RuntimeError(
        f"Riccati recursion did not converge to {tol} in {max_iter} iterations "
        f"(last change {change:.3e})"
    )


_K_SS, _K_ITERS, _K_CHANGE = steady_state_gain()


def kalman_gain() -> np.ndarray:
    """The frozen steady-state gain actually used by :class:`GncController`."""
    return _K_SS.copy()


def gain_convergence() -> tuple[int, float]:
    """``(iterations, final max-norm change)`` of the Riccati recursion."""
    return _K_ITERS, _K_CHANGE


def reference(step: int, dt: float = DT) -> float:
    """Position reference at ``step``, metres."""
    return REF_AMPLITUDE * math.sin(2.0 * math.pi * REF_FREQ_HZ * step * dt)


@dataclass
class GncController:
    """Fixed-gain Kalman filter plus PD controller: the software under test.

    The public interface is a single :meth:`step` taking a measurement mapping
    and returning a command mapping.  Fault injection happens entirely outside
    this class (see :class:`faultinject.wrapper.InjectionWrapper`); nothing in
    this file knows that injection exists.
    """

    dt: float = DT
    kp: float = KP
    kd: float = KD
    u_max: float = U_MAX
    p_hat: float = 0.0
    v_hat: float = 0.0
    u_prev: float = 0.0
    innovations: list[tuple[int, float, float]] = field(default_factory=list)
    updates: int = 0
    skipped: int = 0

    def __post_init__(self) -> None:
        k = _K_SS
        self.k11, self.k12 = float(k[0, 0]), float(k[0, 1])
        self.k21, self.k22 = float(k[1, 0]), float(k[1, 1])

    def reset(self) -> None:
        """Return the controller to its initial state."""
        self.p_hat = 0.0
        self.v_hat = 0.0
        self.u_prev = 0.0
        self.innovations = []
        self.updates = 0
        self.skipped = 0

    def _sanitise(self, value: float) -> float:
        """Hook for subclasses. The plain controller passes values through."""
        return value

    def step(self, k: int, meas: Mapping[str, float]) -> dict[str, float]:
        """One control step.

        Parameters
        ----------
        k
            Step index, used only for the reference and for innovation logging.
        meas
            Mapping with keys ``pos`` (m) and ``vel`` (m/s), and optional
            ``valid`` (0 or 1).  ``valid == 0`` means no measurement arrived, in
            which case the filter propagates without an update.

        Returns
        -------
        dict
            ``{"u": commanded acceleration in m/s^2}``.
        """
        dt = self.dt
        p_pred = self.p_hat + dt * self.v_hat + 0.5 * dt * dt * self.u_prev
        v_pred = self.v_hat + dt * self.u_prev
        valid = bool(meas.get("valid", 1.0))
        if valid:
            e_p = self._sanitise(float(meas["pos"]) - p_pred)
            e_v = self._sanitise(float(meas["vel"]) - v_pred)
            self.innovations.append((k, e_p, e_v))
            self.p_hat = p_pred + self.k11 * e_p + self.k12 * e_v
            self.v_hat = v_pred + self.k21 * e_p + self.k22 * e_v
            self.updates += 1
        else:
            self.p_hat = p_pred
            self.v_hat = v_pred
            self.skipped += 1
        r = reference(k, dt)
        u = self.kp * (r - self.p_hat) - self.kd * self.v_hat
        if u > self.u_max:
            u = self.u_max
        elif u < -self.u_max:
            u = -self.u_max
        elif u != u:  # NaN survives both comparisons and is passed through
            u = self._sanitise(u)
        self.u_prev = u
        return {"u": u}


class SanitisingGncController(GncController):
    """Variant that replaces non-finite innovations with zero.

    This exists only so that ``validation/validate_nan_detection.py`` can show
    the difference between a fault that *propagates* (plain controller) and one
    that is *silently absorbed* by the target, and that the monitor reports the
    second case as absorbed rather than as no fault at all.  It is not a
    recommended mitigation: swallowing a NaN destroys the evidence.
    """

    def _sanitise(self, value: float) -> float:
        return 0.0 if not math.isfinite(value) else value


@dataclass
class DoubleIntegratorPlant:
    """Discrete double integrator with additive process noise. Units m, m/s."""

    dt: float = DT
    p: float = 0.0
    v: float = 0.0

    def reset(self) -> None:
        self.p = 0.0
        self.v = 0.0

    def measure(self, noise_p: float, noise_v: float) -> dict[str, float]:
        """True state plus measurement noise. Units: m, m/s."""
        return {"pos": self.p + noise_p, "vel": self.v + noise_v, "valid": 1.0}

    def advance(self, u: float, w_p: float, w_v: float) -> None:
        """Propagate one step under acceleration ``u`` (m/s^2) plus process noise."""
        dt = self.dt
        p_next = self.p + dt * self.v + 0.5 * dt * dt * u + w_p
        v_next = self.v + dt * u + w_v
        self.p = p_next
        self.v = v_next
