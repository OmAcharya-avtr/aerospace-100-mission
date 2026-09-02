"""SGCMG steering laws: pseudo-inverse, singularity-robust, generalised SR, null motion.

Every law solves the same under-determined problem: find gimbal rates
``ddelta`` such that the delivered body torque ``tau_body = -A(delta) ddelta``
matches a command ``tau_cmd`` [N*m].  Writing ``b = -tau_cmd`` for the
required momentum rate, the laws implemented here are

* **Moore-Penrose pseudo-inverse** (minimum-norm exact solution away from a
  singularity)::

      ddelta = A^+ b = A^T (A A^T)^{-1} b                                  (1)

* **Singularity-robust (SR) inverse** (Nakamura & Hanafusa 1986, brought to
  CMG steering by Bedrossian et al. 1990)::

      ddelta = A^T (A A^T + lam I)^{-1} b                                  (2)

  which trades exactness for a bounded gimbal rate near a singularity.  Its
  torque error has an exact closed form: with the singular value decomposition
  ``A = U S V^T``,

      tau_err = sum_k [lam / (sigma_k^2 + lam)] (u_k . tau_cmd) u_k        (3)

  so ``|tau_err|^2 = sum_k [lam/(sigma_k^2+lam)]^2 (u_k . tau_cmd)^2``.
  ``validation/validate_steering.py`` checks the implementation against (3).

* **Generalised singularity-robust (GSR) inverse** (Wie, Bailey & Heiberg
  2001)::

      ddelta = A^T (A A^T + lam E)^{-1} b,
      E = [[1, e3, e2], [e3, 1, e1], [e2, e1, 1]],
      e_i = eps0 sin(omega t + phi_i)                                      (4)

  The off-diagonal dither makes the error direction time-varying so that the
  array is pushed off, rather than held on, a singular surface.

* **Null motion**: any ``ddelta_null`` with ``A ddelta_null = 0`` can be added
  without changing the delivered torque; see :mod:`cmgsteer.nullmotion`.

Adaptive robustness parameter (used when ``lam`` is not given explicitly)::

      lam = lam0 * h0_mean^2 * exp(-mu * m / m_scale),   m_scale = h0_mean^3  (5)

which is the standard exponential form of Nakamura & Hanafusa and Wie et al.,
non-dimensionalised by the mean rotor momentum so that ``lam0`` and ``mu`` are
dimensionless.

Units: gimbal rates [rad/s], torques [N*m], ``A`` [N*m*s/rad], ``lam``
[(N*m*s/rad)^2].  Assumptions and validity are those of
:mod:`cmgsteer.arrays`; in addition every law here is instantaneous, i.e. it
assumes an ideal gimbal-rate servo with no lag and no rate quantisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .arrays import CMGArray

__all__ = [
    "METHODS",
    "SteeringResult",
    "apply_rate_limit",
    "gsr_inverse_steer",
    "pseudo_inverse_steer",
    "robustness_parameter",
    "sr_inverse_steer",
    "sr_torque_error_closed_form",
    "steer",
]

METHODS: tuple[str, ...] = ("pinv", "sr", "gsr")

DEFAULT_LAM0 = 0.01
DEFAULT_MU = 10.0
DEFAULT_EPS0 = 0.01
DEFAULT_OMEGA = 0.5 * np.pi
DEFAULT_PHASES: tuple[float, float, float] = (0.0, 0.5 * np.pi, np.pi)


@dataclass(frozen=True)
class SteeringResult:
    """Outcome of one steering-law evaluation.

    Attributes
    ----------
    gimbal_rates
        ``(n_free,)`` commanded gimbal rates [rad/s], after any rate limit.
    unlimited_rates
        ``(n_free,)`` rates before the rate limit was applied [rad/s].
    commanded_torque, achieved_torque, torque_error
        ``(3,)`` body torques [N*m]; ``torque_error = commanded - achieved``.
    torque_error_norm
        ``|torque_error|`` [N*m].
    measure
        Singularity measure at ``delta`` [(N*m*s/rad)^3].
    min_singular_value
        ``sigma_min`` of ``A`` [N*m*s/rad].
    lam
        Robustness parameter actually used [(N*m*s/rad)^2]; 0 for the
        pseudo-inverse.
    method
        ``"pinv"``, ``"sr"`` or ``"gsr"``, with ``"+null"`` appended when a
        null-motion term was added.
    rate_limited
        Whether the rate limit changed any component.
    n_rate_limited
        How many components the rate limit changed.
    null_rates
        The null-motion component of ``unlimited_rates`` [rad/s].
    extras
        Method-specific diagnostics.
    """

    gimbal_rates: NDArray[np.float64]
    unlimited_rates: NDArray[np.float64]
    commanded_torque: NDArray[np.float64]
    achieved_torque: NDArray[np.float64]
    torque_error: NDArray[np.float64]
    torque_error_norm: float
    measure: float
    min_singular_value: float
    lam: float
    method: str
    rate_limited: bool = False
    n_rate_limited: int = 0
    null_rates: NDArray[np.float64] | None = None
    extras: dict[str, float] = field(default_factory=dict)


def _check_torque(torque: ArrayLike) -> NDArray[np.float64]:
    t = np.asarray(torque, dtype=float).reshape(-1)
    if t.shape != (3,):
        raise ValueError(f"torque must have shape (3,), got {t.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError("torque must be finite [N*m]")
    return t


def _momentum_scale(array: CMGArray) -> float:
    return float(np.mean(array.rotor_momenta))


def _check_array(array: CMGArray) -> None:
    """Reject an array that cannot span three axes at all."""
    if array.n_free < 3:
        raise ValueError(
            f"steering a 3-axis command needs at least 3 free gimbals, this array has "
            f"{array.n_free}; it is under-actuated and no steering law can meet a general "
            "torque command"
        )


def robustness_parameter(
    array: CMGArray,
    measure: float,
    lam0: float = DEFAULT_LAM0,
    mu: float = DEFAULT_MU,
) -> float:
    """Adaptive SR robustness parameter, equation (5).

    Parameters
    ----------
    array
        Used only for its mean rotor momentum, which non-dimensionalises the
        expression.
    measure
        Singularity measure ``m`` [(N*m*s/rad)^3], must be >= 0.
    lam0
        Dimensionless robustness parameter at ``m = 0``; must be >= 0.
    mu
        Dimensionless decay rate; must be >= 0.  Larger ``mu`` confines the
        robustness term to a thinner shell around the singular surfaces.

    Returns
    -------
    ``lam`` [(N*m*s/rad)^2].
    """
    if measure < 0.0:
        raise ValueError(f"measure must be non-negative, got {measure}")
    if lam0 < 0.0:
        raise ValueError(f"lam0 must be non-negative, got {lam0}")
    if mu < 0.0:
        raise ValueError(f"mu must be non-negative, got {mu}")
    h0 = _momentum_scale(array)
    return float(lam0 * h0**2 * np.exp(-mu * measure / h0**3))


def apply_rate_limit(
    rates: ArrayLike, max_rate: float | None, mode: str = "clip"
) -> tuple[NDArray[np.float64], int]:
    """Apply a symmetric gimbal-rate limit.

    Parameters
    ----------
    rates
        Commanded gimbal rates [rad/s].
    max_rate
        Symmetric limit [rad/s], or ``None`` for no limit.
    mode
        ``"clip"`` limits each gimbal independently, which changes the
        direction of the delivered torque as well as its magnitude.
        ``"scale"`` divides the whole vector by the largest overshoot factor,
        which preserves the torque direction and loses only magnitude.

    Returns
    -------
    ``(limited_rates, n_components_changed)``.
    """
    r = np.asarray(rates, dtype=float).reshape(-1)
    if max_rate is None:
        return r.copy(), 0
    if max_rate <= 0.0:
        raise ValueError(f"max_rate must be positive [rad/s], got {max_rate}")
    over = np.abs(r) > max_rate
    n_over = int(np.count_nonzero(over))
    if n_over == 0:
        return r.copy(), 0
    if mode == "clip":
        return np.clip(r, -max_rate, max_rate), n_over
    if mode == "scale":
        return r * (max_rate / float(np.max(np.abs(r)))), n_over
    raise ValueError(f"mode must be 'clip' or 'scale', got {mode!r}")


def _finish(
    array: CMGArray,
    deltas: NDArray[np.float64],
    torque: NDArray[np.float64],
    rates: NDArray[np.float64],
    sv: NDArray[np.float64],
    lam: float,
    method: str,
    max_gimbal_rate: float | None,
    saturation_mode: str,
    null_rates: NDArray[np.float64] | None,
    extras: dict[str, float],
) -> SteeringResult:
    limited, n_over = apply_rate_limit(rates, max_gimbal_rate, saturation_mode)
    achieved = array.torque(deltas, limited)
    err = torque - achieved
    return SteeringResult(
        gimbal_rates=limited,
        unlimited_rates=rates,
        commanded_torque=torque,
        achieved_torque=achieved,
        torque_error=err,
        torque_error_norm=float(np.linalg.norm(err)),
        measure=float(np.prod(sv)),
        min_singular_value=float(sv[-1]),
        lam=float(lam),
        method=method,
        rate_limited=n_over > 0,
        n_rate_limited=n_over,
        null_rates=null_rates,
        extras=extras,
    )


def pseudo_inverse_steer(
    array: CMGArray,
    deltas: ArrayLike,
    torque: ArrayLike,
    null_rates: ArrayLike | None = None,
    max_gimbal_rate: float | None = None,
    saturation_mode: str = "clip",
) -> SteeringResult:
    """Moore-Penrose pseudo-inverse steering, equation (1).

    Exact away from a singularity and unbounded at one: the gimbal rate grows
    like ``1 / sigma_min``.  ``null_rates`` is an optional length-``n_free``
    null-motion term added to the particular solution.
    """
    _check_array(array)
    d = np.asarray(deltas, dtype=float).reshape(-1)
    t = _check_torque(torque)
    a = array.jacobian(d)
    sv = np.linalg.svd(a, compute_uv=False)
    rates = np.linalg.pinv(a) @ (-t)
    method = "pinv"
    null = None
    if null_rates is not None:
        null = np.asarray(null_rates, dtype=float).reshape(-1)
        if null.shape[0] != rates.shape[0]:
            raise ValueError(f"null_rates must have length {rates.shape[0]}, got {null.shape[0]}")
        rates = rates + null
        method = "pinv+null"
    return _finish(
        array, d, t, rates, sv, 0.0, method, max_gimbal_rate, saturation_mode, null, {}
    )


def sr_inverse_steer(
    array: CMGArray,
    deltas: ArrayLike,
    torque: ArrayLike,
    lam: float | None = None,
    lam0: float = DEFAULT_LAM0,
    mu: float = DEFAULT_MU,
    null_rates: ArrayLike | None = None,
    max_gimbal_rate: float | None = None,
    saturation_mode: str = "clip",
) -> SteeringResult:
    """Singularity-robust inverse steering, equation (2).

    Parameters
    ----------
    lam
        Absolute robustness parameter [(N*m*s/rad)^2].  ``None`` uses the
        adaptive form (5) with ``lam0`` and ``mu``.
    null_rates
        Optional null-motion term, length ``n_free``.

    Notes
    -----
    With ``lam > 0`` the delivered torque is never exact; the error is
    equation (3), which :func:`sr_torque_error_closed_form` evaluates.
    """
    _check_array(array)
    d = np.asarray(deltas, dtype=float).reshape(-1)
    t = _check_torque(torque)
    a = array.jacobian(d)
    sv = np.linalg.svd(a, compute_uv=False)
    measure = float(np.prod(sv))
    if lam is None:
        lam_used = robustness_parameter(array, measure, lam0=lam0, mu=mu)
    else:
        if lam < 0.0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        lam_used = float(lam)
    gram = a @ a.T + lam_used * np.eye(3)
    rates = a.T @ np.linalg.solve(gram, -t)
    method = "sr"
    null = None
    if null_rates is not None:
        null = np.asarray(null_rates, dtype=float).reshape(-1)
        if null.shape[0] != rates.shape[0]:
            raise ValueError(f"null_rates must have length {rates.shape[0]}, got {null.shape[0]}")
        rates = rates + null
        method = "sr+null"
    return _finish(
        array, d, t, rates, sv, lam_used, method, max_gimbal_rate, saturation_mode, null, {}
    )


def gsr_inverse_steer(
    array: CMGArray,
    deltas: ArrayLike,
    torque: ArrayLike,
    time: float = 0.0,
    lam: float | None = None,
    lam0: float = DEFAULT_LAM0,
    mu: float = DEFAULT_MU,
    eps0: float = DEFAULT_EPS0,
    omega: float = DEFAULT_OMEGA,
    phases: tuple[float, float, float] = DEFAULT_PHASES,
    null_rates: ArrayLike | None = None,
    max_gimbal_rate: float | None = None,
    saturation_mode: str = "clip",
) -> SteeringResult:
    """Generalised singularity-robust inverse steering, equation (4).

    Parameters
    ----------
    time
        Time [s] used by the deterministic off-diagonal dither.
    eps0, omega, phases
        Dither amplitude (dimensionless), angular frequency [rad/s] and the
        three phases [rad].  ``eps0 = 0`` reduces the law exactly to the SR
        inverse.
    """
    _check_array(array)
    d = np.asarray(deltas, dtype=float).reshape(-1)
    t = _check_torque(torque)
    if eps0 < 0.0:
        raise ValueError(f"eps0 must be non-negative, got {eps0}")
    if len(phases) != 3:
        raise ValueError(f"phases must have length 3, got {len(phases)}")
    a = array.jacobian(d)
    sv = np.linalg.svd(a, compute_uv=False)
    measure = float(np.prod(sv))
    if lam is None:
        lam_used = robustness_parameter(array, measure, lam0=lam0, mu=mu)
    else:
        if lam < 0.0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        lam_used = float(lam)
    e1, e2, e3 = (eps0 * np.sin(omega * time + p) for p in phases)
    e_mat = np.array([[1.0, e3, e2], [e3, 1.0, e1], [e2, e1, 1.0]])
    gram = a @ a.T + lam_used * e_mat
    rates = a.T @ np.linalg.solve(gram, -t)
    method = "gsr"
    null = None
    if null_rates is not None:
        null = np.asarray(null_rates, dtype=float).reshape(-1)
        if null.shape[0] != rates.shape[0]:
            raise ValueError(f"null_rates must have length {rates.shape[0]}, got {null.shape[0]}")
        rates = rates + null
        method = "gsr+null"
    return _finish(
        array,
        d,
        t,
        rates,
        sv,
        lam_used,
        method,
        max_gimbal_rate,
        saturation_mode,
        null,
        {"e1": float(e1), "e2": float(e2), "e3": float(e3)},
    )


def sr_torque_error_closed_form(
    jacobian: ArrayLike, torque: ArrayLike, lam: float
) -> NDArray[np.float64]:
    """Closed-form SR-inverse torque error, equation (3).

    Parameters
    ----------
    jacobian
        ``(3, n)`` Jacobian ``A`` [N*m*s/rad].
    torque
        Commanded body torque [N*m].
    lam
        Robustness parameter [(N*m*s/rad)^2], >= 0.

    Returns
    -------
    ``(3,)`` predicted torque error ``tau_cmd - tau_achieved`` [N*m].
    """
    a = np.asarray(jacobian, dtype=float)
    if a.ndim != 2 or a.shape[0] != 3:
        raise ValueError(f"jacobian must have shape (3, n), got {a.shape}")
    t = _check_torque(torque)
    if lam < 0.0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    u, sv, _ = np.linalg.svd(a)
    weights = lam / (sv**2 + lam)
    return u @ (weights * (u.T @ t))


def steer(
    array: CMGArray,
    deltas: ArrayLike,
    torque: ArrayLike,
    method: str = "sr",
    **kwargs: object,
) -> SteeringResult:
    """Dispatch to a steering law by name.

    ``method`` is one of :data:`METHODS`.  Remaining keyword arguments are
    passed through to the chosen law.
    """
    if method == "pinv":
        return pseudo_inverse_steer(array, deltas, torque, **kwargs)  # type: ignore[arg-type]
    if method == "sr":
        return sr_inverse_steer(array, deltas, torque, **kwargs)  # type: ignore[arg-type]
    if method == "gsr":
        return gsr_inverse_steer(array, deltas, torque, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown steering method {method!r}; expected one of {METHODS}")
