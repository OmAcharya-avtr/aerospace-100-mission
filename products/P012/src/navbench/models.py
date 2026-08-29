"""Canonical dynamic and measurement models used by the bench.

All discrete process-noise matrices follow Bar-Shalom, Li & Kirubarajan
(2001), *Estimation with Applications to Tracking and Navigation*, §6.2-6.3.

CONTINUOUS WHITE-NOISE ACCELERATION (CWNA), per axis, state ``[p, v]``:

    F = [[1, T], [0, 1]]
    Q = q̃ [[T³/3, T²/2], [T²/2, T]]          q̃ = acceleration PSD [m²/s³]

(Bar-Shalom et al. Eq. (6.2.3-4).)  Assumptions: the acceleration is a
zero-mean continuous white noise, exactly integrated over the sampling
interval.  Validity: manoeuvre bandwidth well below 1/T.

DISCRETE WHITE-NOISE ACCELERATION (DWNA), per axis:

    F = [[1, T], [0, 1]]
    Q = σ_a² [[T⁴/4, T³/2], [T³/2, T²]]      σ_a in m/s²

(Bar-Shalom et al. Eq. (6.3.2-4).)  The acceleration is constant across a
sampling interval and independent between intervals; ``Q`` is singular
(rank 1 per axis), which is physically meaningful and numerically benign in
the Joseph form.

RADAR (range/bearing) MEASUREMENT of a 2-D Cartesian state ``[x, vx, y, vy]``:

    h(x) = [ sqrt(x² + y²) , atan2(y, x) ]        units [m, rad]

This is the standard nonlinearity used to separate EKF and UKF behaviour
(Julier & Uhlmann 2004 §V).  It is *weakly* nonlinear when the target is far
from the sensor relative to the position uncertainty, and *strongly*
nonlinear when the range is comparable with the uncertainty or the target
passes near the origin, where the bearing derivative ``∂θ/∂x = −y/r²``
blows up.  Validity of the Jacobian: ``r`` must be bounded away from 0; the
code raises rather than dividing by a vanishing range.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "random_walk",
    "constant_velocity_cwna",
    "constant_velocity_dwna",
    "constant_velocity_2d",
    "radar_measurement",
    "radar_jacobian",
    "simulate_linear_system",
    "simulate_radar_scenario",
]


def _check_dt(dt: float) -> float:
    t = float(dt)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError(f"dt must be finite and > 0 s, got {dt!r}")
    return t


def random_walk(q: float, r: float) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Scalar random walk ``x_k = x_{k-1} + w``, ``z = x + v``.

    Returns ``(F, H, Q, R)`` as 1×1 matrices.  ``q`` and ``r`` are variances in
    the state's own squared units.  This is the model with a hand-solvable
    steady-state Riccati equation (validation v1).
    """
    for name, val in (("q", q), ("r", r)):
        v = float(val)
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {val!r}")
    return (
        np.array([[1.0]]),
        np.array([[1.0]]),
        np.array([[float(q)]]),
        np.array([[float(r)]]),
    )


def constant_velocity_cwna(dt: float, q_psd: float) -> tuple[NDArray, NDArray]:
    """Single-axis CWNA constant-velocity model.

    Parameters
    ----------
    dt : float
        Sampling interval [s], > 0.
    q_psd : float
        Acceleration power spectral density q̃ [m²/s³], ≥ 0.

    Returns
    -------
    (F, Q) each 2×2, state ``[position m, velocity m/s]``.
    """
    t = _check_dt(dt)
    q = float(q_psd)
    if not np.isfinite(q) or q < 0.0:
        raise ValueError(f"q_psd must be finite and >= 0 m^2/s^3, got {q_psd!r}")
    f = np.array([[1.0, t], [0.0, 1.0]])
    qm = q * np.array([[t**3 / 3.0, t**2 / 2.0], [t**2 / 2.0, t]])
    return f, qm


def constant_velocity_dwna(dt: float, sigma_a: float) -> tuple[NDArray, NDArray]:
    """Single-axis DWNA constant-velocity model.

    Parameters
    ----------
    dt : float
        Sampling interval [s], > 0.
    sigma_a : float
        Per-interval acceleration standard deviation [m/s²], ≥ 0.

    Returns
    -------
    (F, Q) each 2×2, state ``[position m, velocity m/s]``.
    """
    t = _check_dt(dt)
    s = float(sigma_a)
    if not np.isfinite(s) or s < 0.0:
        raise ValueError(f"sigma_a must be finite and >= 0 m/s^2, got {sigma_a!r}")
    f = np.array([[1.0, t], [0.0, 1.0]])
    qm = s * s * np.array([[t**4 / 4.0, t**3 / 2.0], [t**3 / 2.0, t**2]])
    return f, qm


def constant_velocity_2d(dt: float, q_psd: float) -> tuple[NDArray, NDArray]:
    """Two-axis CWNA model with state ``[x, vx, y, vy]`` (m, m/s).

    The two axes are independent, so ``F`` and ``Q`` are block diagonal with
    the single-axis blocks of :func:`constant_velocity_cwna`.
    """
    f1, q1 = constant_velocity_cwna(dt, q_psd)
    f = np.zeros((4, 4))
    q = np.zeros((4, 4))
    f[:2, :2] = f1
    f[2:, 2:] = f1
    q[:2, :2] = q1
    q[2:, 2:] = q1
    return f, q


def radar_measurement(x: ArrayLike) -> NDArray[np.float64]:
    """``h(x) = [range m, bearing rad]`` for state ``[x, vx, y, vy]``.

    Raises
    ------
    ValueError
        If the range is below 1e-9 m, where the bearing is undefined.
    """
    s = np.asarray(x, dtype=float).ravel()
    if s.size != 4:
        raise ValueError(f"state must have 4 elements [x, vx, y, vy], got {s.size}")
    if not np.all(np.isfinite(s)):
        raise ValueError("state must be finite")
    r = float(np.hypot(s[0], s[2]))
    if r < 1e-9:
        raise ValueError(f"range {r:.3e} m is too small for the bearing to be defined")
    return np.array([r, float(np.arctan2(s[2], s[0]))])


def radar_jacobian(x: ArrayLike) -> NDArray[np.float64]:
    """Analytic ``∂h/∂x`` of :func:`radar_measurement`, shape (2, 4).

    ``∂r/∂x = x/r``, ``∂r/∂y = y/r``, ``∂θ/∂x = −y/r²``, ``∂θ/∂y = x/r²``.
    Velocity columns are zero (position-only measurement).
    """
    s = np.asarray(x, dtype=float).ravel()
    if s.size != 4:
        raise ValueError(f"state must have 4 elements [x, vx, y, vy], got {s.size}")
    if not np.all(np.isfinite(s)):
        raise ValueError("state must be finite")
    px, py = s[0], s[2]
    r2 = px * px + py * py
    r = np.sqrt(r2)
    if r < 1e-9:
        raise ValueError(f"range {r:.3e} m is too small for the Jacobian to be defined")
    return np.array(
        [
            [px / r, 0.0, py / r, 0.0],
            [-py / r2, 0.0, px / r2, 0.0],
        ]
    )


def simulate_linear_system(
    f: ArrayLike,
    h: ArrayLike,
    q: ArrayLike,
    r: ArrayLike,
    x0: ArrayLike,
    n_steps: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate ``x_k = F x_{k-1} + w``, ``z_k = H x_k + v``.

    Process and measurement noise are drawn from ``N(0, Q)`` and ``N(0, R)``
    using the symmetric eigenvalue square root, so singular ``Q`` (DWNA) is
    handled correctly.

    Returns
    -------
    (states, measurements) with shapes (N, n) and (N, m); ``states[k]`` is the
    truth at the same instant as ``measurements[k]``.
    """
    fm = np.atleast_2d(np.asarray(f, dtype=float))
    hm = np.atleast_2d(np.asarray(h, dtype=float))
    qm = np.atleast_2d(np.asarray(q, dtype=float))
    rm = np.atleast_2d(np.asarray(r, dtype=float))
    x = np.asarray(x0, dtype=float).ravel()
    n = x.size
    if fm.shape != (n, n):
        raise ValueError(f"f must have shape ({n}, {n}), got {fm.shape}")
    if hm.shape[1] != n:
        raise ValueError(f"h must have {n} columns, got {hm.shape}")
    m = hm.shape[0]
    if qm.shape != (n, n):
        raise ValueError(f"q must have shape ({n}, {n}), got {qm.shape}")
    if rm.shape != (m, m):
        raise ValueError(f"r must have shape ({m}, {m}), got {rm.shape}")
    steps = int(n_steps)
    if steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps!r}")

    lq = _psd_sqrt(qm, "q")
    lr = _psd_sqrt(rm, "r")
    states = np.zeros((steps, n))
    meas = np.zeros((steps, m))
    for k in range(steps):
        x = fm @ x + lq @ rng.standard_normal(n)
        states[k] = x
        meas[k] = hm @ x + lr @ rng.standard_normal(m)
    return states, meas


def _psd_sqrt(a: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Symmetric square root of a PSD matrix via its eigendecomposition."""
    sym = 0.5 * (a + a.T)
    w, v = np.linalg.eigh(sym)
    scale = max(1.0, float(np.max(np.abs(sym))))
    if float(w.min()) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semi-definite; min eigenvalue {w.min():.3e}")
    return v @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ v.T


def simulate_radar_scenario(
    *,
    dt: float,
    n_steps: int,
    q_psd: float,
    sigma_range: float,
    sigma_bearing: float,
    x0: ArrayLike,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """2-D constant-velocity truth with range/bearing measurements.

    Parameters
    ----------
    dt : float
        Sampling interval [s].
    n_steps : int
        Number of steps.
    q_psd : float
        Acceleration PSD per axis [m²/s³].
    sigma_range : float
        Range noise standard deviation [m], > 0.
    sigma_bearing : float
        Bearing noise standard deviation [rad], > 0.
    x0 : array_like, shape (4,)
        Initial truth ``[x, vx, y, vy]`` in m, m/s.
    rng : numpy.random.Generator

    Returns
    -------
    (states, measurements) with shapes (N, 4) and (N, 2).
    """
    f, q = constant_velocity_2d(dt, q_psd)
    sr, sb = float(sigma_range), float(sigma_bearing)
    for name, val in (("sigma_range", sr), ("sigma_bearing", sb)):
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {val!r}")
    steps = int(n_steps)
    if steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps!r}")
    x = np.asarray(x0, dtype=float).ravel()
    if x.size != 4:
        raise ValueError(f"x0 must have 4 elements, got {x.size}")
    lq = _psd_sqrt(q, "q")
    states = np.zeros((steps, 4))
    meas = np.zeros((steps, 2))
    for k in range(steps):
        x = f @ x + lq @ rng.standard_normal(4)
        states[k] = x
        z = radar_measurement(x)
        meas[k] = z + np.array([sr, sb]) * rng.standard_normal(2)
    return states, meas
