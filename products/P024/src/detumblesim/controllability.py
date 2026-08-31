"""Quantifying the magnetorquer controllability gap.

A magnetic dipole ``m`` in a field ``B`` produces the torque ``L = m x B``,
which is identically perpendicular to ``B``.  **No magnetorquer can ever
torque about the instantaneous field direction.**  Instantaneously the system
is therefore rank-2: one of the three body-rate degrees of freedom is
uncontrollable at every moment.

Detumbling works anyway because ``B_hat`` moves in inertial space as the
spacecraft goes round the orbit and the tilted dipole rotates with the Earth,
so the *time-averaged* damping matrix

    D_hat = < I - B_hat B_hat^T >          (unweighted)
    D     = k ( <|B|^2> I - <B B^T> )      (field-strength weighted, N m s)

is generally full rank.  How full is the question this module answers.  Both
matrices have trace identities: ``trace(D_hat) = 2`` exactly, and the
eigenvalues of ``D / (k <|B|^2>)`` also sum to 2.  A perfectly isotropic field
history gives eigenvalues ``(2/3, 2/3, 2/3)``; a field confined to a single
inertial direction gives ``(0, 1, 1)`` and the ``0`` direction never
detumbles.

For a near-equatorial orbit the field along the track stays close to the
dipole axis, so the damping about the axis direction is weak and a residual
spin persists there long after the other two axes are damped.  Near-polar
orbits sample the field direction far more uniformly.
``validation/controllability_gap.py`` measures both.

References
----------
Wertz, J. R. (ed.), "Spacecraft Attitude Determination and Control",
    D. Reidel, 1978, sec. 19.4 (magnetic control: the torque is always
    perpendicular to B).
Markley, F. L. and Crassidis, J. L., "Fundamentals of Spacecraft Attitude
    Determination and Control", Springer, 2014, sec. 7.5.
Avanzini, G. and Giulietti, F., "Magnetic Detumbling of a Rigid Spacecraft",
    J. Guidance, Control, and Dynamics, 35(4), 2012, pp. 1326-1334.
    doi:10.2514/1.53074
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .orbit import CircularOrbit


def uncontrollable_fraction(omega: ArrayLike, b_field: ArrayLike) -> float:
    """Fraction ``|omega . B_hat| / |omega|`` of the rate along ``B``.

    This component receives no control torque at that instant.  Returns 0.0
    for a zero rate (nothing to damp) and raises for a zero field.
    """
    w = np.asarray(omega, dtype=float)
    b = np.asarray(b_field, dtype=float)
    if w.shape != (3,) or b.shape != (3,):
        raise ValueError("omega and b_field must have shape (3,)")
    bn = float(np.linalg.norm(b))
    if bn <= 0.0:
        raise ValueError("b_field must be non-zero")
    wn = float(np.linalg.norm(w))
    if wn <= 0.0:
        return 0.0
    return float(abs(w @ b) / (bn * wn))


def instantaneous_projector(b_field: ArrayLike) -> NDArray[np.float64]:
    """``I - B_hat B_hat^T``: the rank-2 achievable-torque projector."""
    b = np.asarray(b_field, dtype=float)
    if b.shape != (3,):
        raise ValueError(f"b_field must have shape (3,), got {b.shape}")
    bn = float(np.linalg.norm(b))
    if bn <= 0.0:
        raise ValueError("b_field must be non-zero")
    bh = b / bn
    return np.eye(3) - np.outer(bh, bh)


@dataclass(frozen=True)
class ControllabilityReport:
    """Orbit-averaged controllability of a magnetorquer-only spacecraft.

    Attributes
    ----------
    span_s : float
        Averaging span [s].
    n_samples : int
        Samples used.
    direction_eigenvalues : ndarray (3,)
        Eigenvalues of ``< I - B_hat B_hat^T >``, ascending; sum = 2.
    weighted_eigenvalues : ndarray (3,)
        Eigenvalues of ``(<|B|^2> I - <B B^T>) / <|B|^2>``, ascending; sum = 2.
        These are the geometry factors of ``analytic.py`` and set the modal
        time constants.
    weakest_direction_eci : ndarray (3,)
        Unit eigenvector of the smallest weighted eigenvalue: the inertial
        direction about which the spacecraft is hardest to detumble.
    anisotropy : float
        ``lambda_max / lambda_min`` of the weighted eigenvalues; 1.0 is
        perfectly isotropic, large values mean one axis lags badly.
    isotropic_reference : float
        2/3, the value every eigenvalue takes for an isotropic field history.
    rms_field_t : float
        ``sqrt(<|B|^2>)`` [T].
    mean_uncontrollable_fraction : float
        Time-average of ``|omega . B_hat| / |omega|`` for a rate drawn along
        ``weakest_direction_eci``; how much of that rate is instantaneously
        beyond control authority.
    """

    span_s: float
    n_samples: int
    direction_eigenvalues: NDArray[np.float64]
    weighted_eigenvalues: NDArray[np.float64]
    weakest_direction_eci: NDArray[np.float64]
    anisotropy: float
    isotropic_reference: float
    rms_field_t: float
    mean_uncontrollable_fraction: float


def controllability_report(
    orbit: CircularOrbit, n_samples: int = 4000, span_s: float | None = None
) -> ControllabilityReport:
    """Measure the orbit-averaged controllability gap for one orbit geometry.

    Parameters
    ----------
    orbit : CircularOrbit
    n_samples : int
        Uniform samples over ``span_s``; must be >= 2.
    span_s : float, optional
        Averaging span [s], default one orbital period.  Longer spans let the
        Earth rotate the tilted dipole under the orbit, which is what
        eventually makes a near-equatorial orbit controllable at all.
    """
    if int(n_samples) < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    span = float(orbit.period_s if span_s is None else span_s)
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError(f"span_s must be positive, got {span_s}")
    from .simulate import field_history_eci

    t = np.linspace(0.0, span, int(n_samples), endpoint=False)
    b = field_history_eci(orbit, t)
    bn = np.linalg.norm(b, axis=1)
    bh = b / bn[:, None]
    dir_mat = np.eye(3) - (bh.T @ bh) / bh.shape[0]
    dir_eig = np.linalg.eigvalsh(dir_mat)
    mean_b2 = float(np.mean(bn**2))
    w_mat = np.eye(3) - (b.T @ b) / (b.shape[0] * mean_b2)
    w_eig, w_vec = np.linalg.eigh(w_mat)
    weakest = w_vec[:, 0] / np.linalg.norm(w_vec[:, 0])
    frac = float(np.mean(np.abs(bh @ weakest)))
    lam_min = float(w_eig[0])
    aniso = float(w_eig[-1] / lam_min) if lam_min > 0.0 else float("inf")
    return ControllabilityReport(
        span_s=span,
        n_samples=int(n_samples),
        direction_eigenvalues=dir_eig,
        weighted_eigenvalues=w_eig,
        weakest_direction_eci=weakest,
        anisotropy=aniso,
        isotropic_reference=2.0 / 3.0,
        rms_field_t=float(np.sqrt(mean_b2)),
        mean_uncontrollable_fraction=frac,
    )


def residual_rate_along(
    omega_history: ArrayLike, direction_eci: ArrayLike, quat_history: ArrayLike
) -> NDArray[np.float64]:
    """Body-rate component along a fixed *inertial* direction, per sample [rad/s].

    Parameters
    ----------
    omega_history : array_like (N, 3)
        Body rates [rad/s].
    direction_eci : array_like (3,)
        Unit inertial direction.
    quat_history : array_like (N, 4)
        Scalar-first inertial-to-body quaternions matching ``omega_history``.
    """
    from .attitude import quat_to_dcm

    w = np.asarray(omega_history, dtype=float)
    q = np.asarray(quat_history, dtype=float)
    d = np.asarray(direction_eci, dtype=float)
    if w.ndim != 2 or w.shape[1] != 3:
        raise ValueError(f"omega_history must have shape (N, 3), got {w.shape}")
    if q.shape != (w.shape[0], 4):
        raise ValueError("quat_history must have shape (N, 4) matching omega_history")
    if d.shape != (3,):
        raise ValueError(f"direction_eci must have shape (3,), got {d.shape}")
    dn = float(np.linalg.norm(d))
    if dn <= 0.0:
        raise ValueError("direction_eci must be non-zero")
    d = d / dn
    out = np.empty(w.shape[0])
    for i in range(w.shape[0]):
        out[i] = float(w[i] @ (quat_to_dcm(q[i]) @ d))
    return out
