"""Closed-form first-order detumble model.

Derivation
----------
For the ideal (unsaturated, slowly-varying-field) B-dot law the applied torque
is ``L = -k |B|^2 omega_perp`` (see ``control.ideal_bdot_torque``).  Writing
``omega_perp = (I - B_hat B_hat^T) omega`` and, for an isotropic inertia
``J = j I``, dropping the gyroscopic term (which vanishes identically when the
inertia is isotropic), the rate obeys the linear time-varying system

    j d(omega)/dt = -k (|B|^2 I - B B^T) omega                       (1)

If the body rate changes slowly compared with one orbit, (1) may be averaged
over an orbit to give the constant **damping matrix**

    D = k ( <|B|^2> I - <B B^T> )                          [N m s]   (2)

whose eigenvalues ``lambda_i`` set three modal time constants

    tau_i = j / lambda_i                                   [s]       (3)

and the time to fall from ``omega_0`` to ``omega_f`` in mode ``i`` is

    t_i = tau_i ln(omega_0 / omega_f)                       [s]      (4)

If the field direction were uniformly distributed over the sphere as seen from
the body, ``<B B^T> = (1/3) <|B|^2> I`` exactly, so ``D = (2/3) k <|B|^2> I``
and (3) collapses to the familiar

    tau = 3 j / (2 k <|B|^2>)                                        (5)

The factor 2/3 is therefore **derived**, not assumed; ``geometry_factors``
returns the actual eigenvalues of ``D / (k <|B|^2>)``, whose sum is exactly 2
for any field history and which equal (2/3, 2/3, 2/3) only in the isotropic
case.  This is the same quantity that ``controllability.py`` reports as the
controllability gap.

Equation (4) predicts ``t proportional to 1/k`` at fixed inertia and orbit;
``validation/gain_scaling.py`` tests that scaling against the simulator.

Validity
--------
(1)-(5) assume: no magnetorquer saturation, isotropic inertia (or a
gyroscopic term small compared with the control torque), the ideal
``dB/dt = -omega x B`` approximation (body rate much faster than the orbital
rate of change of the inertial field), and a body rate that varies slowly over
one orbit.  Saturation is the first of these to break in practice; the
simulator honours it and the analytic model does not.

References
----------
Stickler, A. C. and Alfriend, K. T., "Elementary Magnetic Attitude Control
    System", J. Spacecraft and Rockets, 13(5), 1976, pp. 282-287.
    doi:10.2514/3.57089
Avanzini, G. and Giulietti, F., "Magnetic Detumbling of a Rigid Spacecraft",
    J. Guidance, Control, and Dynamics, 35(4), 2012, pp. 1326-1334.
    doi:10.2514/1.53074
Wertz, J. R. (ed.), "Spacecraft Attitude Determination and Control",
    D. Reidel, 1978, sec. 19.4.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .orbit import CircularOrbit
from .spacecraft import Magnetorquer


@dataclass(frozen=True)
class FieldMoments:
    """Orbit-averaged magnetic-field moments in the inertial frame.

    Attributes
    ----------
    mean_b2_t2 : float
        ``<|B|^2>`` [T^2].
    outer_t2 : ndarray (3, 3)
        ``<B B^T>`` [T^2].
    rms_b_t : float
        ``sqrt(<|B|^2>)`` [T].
    n_samples : int
        Number of uniformly spaced samples used.
    span_s : float
        Averaging interval [s].
    """

    mean_b2_t2: float
    outer_t2: NDArray[np.float64]
    rms_b_t: float
    n_samples: int
    span_s: float


def orbit_field_moments(
    orbit: CircularOrbit, n_samples: int = 2000, span_s: float | None = None
) -> FieldMoments:
    """Average ``|B|^2`` and ``B B^T`` over ``span_s`` of the orbit.

    Parameters
    ----------
    orbit : CircularOrbit
    n_samples : int
        Uniformly spaced samples; must be >= 2.
    span_s : float, optional
        Averaging span [s].  Defaults to one orbital period.  Because the
        tilted dipole is fixed in the rotating Earth frame, the field seen
        along the orbit is only periodic over the (much longer) beat period of
        the orbital and Earth-rotation rates, so a longer span gives a
        different, and for detumbling more representative, average.
    """
    if int(n_samples) < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    span = float(orbit.period_s if span_s is None else span_s)
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError(f"span_s must be positive, got {span_s}")
    from .simulate import field_history_eci

    t = np.linspace(0.0, span, int(n_samples), endpoint=False)
    b = field_history_eci(orbit, t)
    b2 = float(np.mean(np.sum(b * b, axis=1)))
    outer = (b.T @ b) / b.shape[0]
    return FieldMoments(
        mean_b2_t2=b2,
        outer_t2=outer,
        rms_b_t=float(np.sqrt(b2)),
        n_samples=int(n_samples),
        span_s=span,
    )


def damping_matrix(moments: FieldMoments, gain: float) -> NDArray[np.float64]:
    """Orbit-averaged B-dot damping matrix ``D`` [N m s], equation (2)."""
    k = float(gain)
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError(f"gain must be positive, got {gain}")
    return k * (moments.mean_b2_t2 * np.eye(3) - moments.outer_t2)


def geometry_factors(moments: FieldMoments) -> NDArray[np.float64]:
    """Eigenvalues of ``D / (k <|B|^2>)``, ascending, dimensionless.

    Their sum is exactly 2 for any field history (trace identity).  The
    isotropic value is ``(2/3, 2/3, 2/3)``; the smallest entry is the
    controllability gap along the least-torqueable direction.
    """
    d_hat = np.eye(3) - moments.outer_t2 / moments.mean_b2_t2
    return np.linalg.eigvalsh(d_hat)


def modal_time_constants(
    moments: FieldMoments, gain: float, inertia_scalar_kgm2: float
) -> NDArray[np.float64]:
    """Modal detumble time constants ``tau_i = j / lambda_i`` [s], ascending.

    ``inertia_scalar_kgm2`` is the isotropic moment of inertia ``j``.  For a
    non-isotropic body use the largest principal moment for a conservative
    (slowest) estimate; the derivation assumes isotropy.
    """
    j = float(inertia_scalar_kgm2)
    if not np.isfinite(j) or j <= 0.0:
        raise ValueError(f"inertia_scalar_kgm2 must be positive, got {j}")
    lam = np.linalg.eigvalsh(damping_matrix(moments, gain))
    if np.any(lam <= 0.0):
        raise ValueError(
            "damping matrix is singular or indefinite: the field direction is "
            "confined to a plane or line over this span, so at least one axis "
            f"is uncontrollable (eigenvalues {lam})"
        )
    return np.sort(j / lam)


def detumble_time_first_order(
    inertia_scalar_kgm2: float,
    gain: float,
    moments: FieldMoments,
    omega0_rad_s: float,
    omega_target_rad_s: float,
    mode: str = "isotropic",
) -> float:
    """First-order detumble time [s], equation (4).

    Parameters
    ----------
    inertia_scalar_kgm2 : float
        Isotropic moment of inertia ``j`` [kg m^2].
    gain : float
        B-dot gain ``k`` [A m^2 s T^-1].
    moments : FieldMoments
        From ``orbit_field_moments``.
    omega0_rad_s, omega_target_rad_s : float
        Initial and target rate magnitudes [rad/s]; both positive with
        ``omega0 > omega_target``.
    mode : {"isotropic", "slowest", "fastest"}
        ``"isotropic"`` uses equation (5) with the exact 2/3 factor;
        ``"slowest"`` and ``"fastest"`` use the extreme modal time constants
        from equation (3), which bracket the true answer.
    """
    w0 = float(omega0_rad_s)
    wf = float(omega_target_rad_s)
    if not np.isfinite(w0) or w0 <= 0.0:
        raise ValueError(f"omega0_rad_s must be positive, got {omega0_rad_s}")
    if not np.isfinite(wf) or wf <= 0.0:
        raise ValueError(f"omega_target_rad_s must be positive, got {omega_target_rad_s}")
    if wf >= w0:
        raise ValueError("omega_target_rad_s must be smaller than omega0_rad_s")
    if mode == "isotropic":
        j = float(inertia_scalar_kgm2)
        if not np.isfinite(j) or j <= 0.0:
            raise ValueError("inertia_scalar_kgm2 must be positive")
        tau = 3.0 * j / (2.0 * float(gain) * moments.mean_b2_t2)
    elif mode in ("slowest", "fastest"):
        taus = modal_time_constants(moments, gain, inertia_scalar_kgm2)
        tau = float(taus[-1] if mode == "slowest" else taus[0])
    else:
        raise ValueError(
            f"mode must be 'isotropic', 'slowest' or 'fastest', got {mode!r}"
        )
    return float(tau * np.log(w0 / wf))


def max_torque_nm(magnetorquer: Magnetorquer, b_body_t: ArrayLike) -> float:
    """Largest torque magnitude [N m] the dipole box can produce in field ``B``.

    ``|m x B|`` is convex in ``m``, so its maximum over the box
    ``|m_i| <= m_max_i`` is attained at one of the eight vertices; all eight
    are enumerated.
    """
    b = np.asarray(b_body_t, dtype=float)
    if b.shape != (3,):
        raise ValueError(f"b_body_t must have shape (3,), got {b.shape}")
    lim = magnetorquer.max_dipole_am2
    best = 0.0
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        m = lim * np.array(signs)
        best = max(best, float(np.linalg.norm(np.cross(m, b))))
    return best


def saturation_time_bound_s(
    magnetorquer: Magnetorquer,
    moments: FieldMoments,
    inertia: ArrayLike,
    omega0_rad_s: ArrayLike,
    omega_target_rad_s: float,
) -> float:
    """Lower bound on detumble time [s] set by the dipole limit.

    No magnetic torque can exceed ``max_i |m x B|`` for the instantaneous
    field, so the angular-momentum magnitude cannot fall faster than that
    bound.  Using the RMS field over the orbit as a stand-in for ``|B|`` (the
    bound is evaluated with ``B = B_rms z_hat``, which is the isotropic
    average magnitude), the shortest possible time to shed
    ``|H_0| - |H_target|`` is

        t >= ( |J omega_0| - |J| omega_target ) / L_max

    This is a **bound, not a prediction**: it ignores the direction of the
    field, the controllability gap, and the fact that B-dot is not
    time-optimal.  Real B-dot runs take several times longer.
    """
    j = np.asarray(inertia, dtype=float)
    w0 = np.asarray(omega0_rad_s, dtype=float)
    h0 = float(np.linalg.norm(j @ w0))
    hf = float(np.linalg.norm(j, ord=2)) * float(omega_target_rad_s)
    if hf >= h0:
        raise ValueError("target momentum is not below the initial momentum")
    b_ref = np.array([0.0, 0.0, moments.rms_b_t])
    lmax = max_torque_nm(magnetorquer, b_ref)
    if lmax <= 0.0:
        raise ValueError("magnetorquer produces no torque in the reference field")
    return (h0 - hf) / lmax
