r"""Desaturation: magnetic dipole commands, their controllability limit, and thrusters.

Magnetic desaturation and the direction you cannot dump
-------------------------------------------------------
A magnetorquer produces :math:`\mathbf{T} = \mathbf{m}\times\mathbf{B}`. Every
achievable torque is therefore perpendicular to the instantaneous field, so the torque
lives in a **two-dimensional plane**, not in three-space, and the component of the wheel
momentum along :math:`\hat{\mathbf{B}}` cannot be changed at that instant no matter what
dipole is commanded. This is not a limitation of a particular control law; it is the
rank of the cross-product map, whose null space is spanned by :math:`\mathbf{B}` itself.

The standard cross-product dumping law

.. math:: \mathbf{m} = -\frac{k}{|\mathbf{B}|^2}\,\mathbf{B}\times\mathbf{h}
          \quad\Longrightarrow\quad
          \mathbf{T} = \mathbf{m}\times\mathbf{B}
          = -k\left[\mathbf{h} - (\mathbf{h}\cdot\hat{\mathbf{B}})\hat{\mathbf{B}}\right]

removes exactly the perpendicular part of the momentum and leaves the parallel part
untouched. Sources: Wertz, *Spacecraft Attitude Determination and Control*; Sidi,
*Spacecraft Dynamics and Control*; Markley and Crassidis, *Fundamentals of Spacecraft
Attitude Determination and Control*.

Controllability is recovered only *on average*, because the field direction sweeps as the
vehicle moves along the orbit. The relevant object is the averaged projector

.. math:: \mathbf{G} = \frac{1}{T}\int_0^T \left(\mathbf{I} -
          \hat{\mathbf{B}}\hat{\mathbf{B}}^{\!\top}\right) dt,

whose eigenvalues are the fraction of the interval for which each principal direction is
dumpable. All three strictly positive means every direction can be dumped given enough
time; the smallest eigenvalue is how much longer the worst direction takes. This is the
standard averaged-controllability argument for magnetic-only control. It is computed
here, reported, and never hidden.

Units: dipole A m^2, field T, torque N m, momentum N m s, dipole cost A m^2 s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _validate as _v
from .constants import STANDARD_GRAVITY

__all__ = [
    "MagneticCommand",
    "magnetic_dump_command",
    "uncontrollable_fraction",
    "averaged_controllability",
    "dipole_cost",
    "ThrusterDump",
    "thruster_dump",
]


@dataclass(frozen=True)
class MagneticCommand:
    """A magnetorquer command and what it can and cannot do at this instant.

    Attributes
    ----------
    dipole_am2 : (3,)
        Commanded dipole in body components [A m^2], already limited.
    torque_nm : (3,)
        Torque it produces, ``m x B`` [N m]. Always perpendicular to ``B``.
    saturated : bool
        True when the unlimited command exceeded ``max_dipole_am2`` and was scaled down.
    uncontrollable_nms : (3,)
        The part of the momentum error along ``B``, which this or any other dipole leaves
        untouched [N m s].
    uncontrollable_fraction : float
        ``|h . B_hat| / |h|``, dimensionless in [0, 1]. Zero means the whole momentum
        error is dumpable now; one means none of it is.
    """

    dipole_am2: NDArray[np.float64]
    torque_nm: NDArray[np.float64]
    saturated: bool
    uncontrollable_nms: NDArray[np.float64]
    uncontrollable_fraction: float


def magnetic_dump_command(
    momentum_error_nms: ArrayLike,
    b_body_t: ArrayLike,
    gain: float = 1.0,
    max_dipole_am2: float | None = None,
) -> MagneticCommand:
    r"""Cross-product magnetic desaturation command.

    ``m = -(gain / |B|^2) B x h``, scaled down in magnitude if it exceeds
    ``max_dipole_am2``. Scaling the whole vector preserves the command direction, which
    is the direction that removes momentum fastest; clipping component-wise would not.

    Parameters
    ----------
    momentum_error_nms : (3,)
        Momentum to be removed, body frame [N m s]. Usually the wheel momentum minus its
        bias target.
    b_body_t : (3,)
        Geomagnetic field in body components [T]. Must be non-zero.
    gain : float
        ``k`` in the law above [s^-1]. With no dipole limit the perpendicular momentum
        then decays with time constant ``1/k``.
    max_dipole_am2 : float or None
        Magnitude limit on the commanded dipole [A m^2]; ``None`` for no limit.

    Returns
    -------
    MagneticCommand

    Notes
    -----
    Assumptions: the field is known (a magnetometer measurement or an onboard model);
    the coil dipole responds instantaneously; there is no interaction with the vehicle's
    own residual dipole, which is treated as part of the disturbance, not of the command.
    Validity: any field model. With the centred non-tilted dipole used elsewhere in this
    package the field *direction* carries errors of tens of degrees against IGRF, and the
    quantity this function is about is a direction, so numbers produced with that field
    model are a geometry study, not a prediction for a specific vehicle on a specific
    day.
    """
    h = _v.as_vector3(momentum_error_nms, "momentum_error_nms")
    b = _v.as_vector3(b_body_t, "b_body_t")
    k = _v.as_finite_float(gain, "gain")
    b_norm = float(np.linalg.norm(b))
    if b_norm == 0.0:
        raise ValueError("b_body_t must be non-zero; a null field gives no magnetic torque")
    b_hat = b / b_norm
    parallel = float(h @ b_hat) * b_hat
    m = -(k / b_norm**2) * np.cross(b, h)
    saturated = False
    if max_dipole_am2 is not None:
        m_max = _v.positive(max_dipole_am2, "max_dipole_am2")
        m_norm = float(np.linalg.norm(m))
        if m_norm > m_max:
            m = m * (m_max / m_norm)
            saturated = True
    h_norm = float(np.linalg.norm(h))
    frac = float(abs(h @ b_hat) / h_norm) if h_norm > 0.0 else 0.0
    return MagneticCommand(
        dipole_am2=m,
        torque_nm=np.cross(m, b),
        saturated=saturated,
        uncontrollable_nms=parallel,
        uncontrollable_fraction=frac,
    )


def uncontrollable_fraction(
    momentum_error_nms: ArrayLike, b_body_t: ArrayLike
) -> NDArray[np.float64]:
    """``|h . B_hat| / |h|`` at each sample, dimensionless in [0, 1].

    Accepts ``(3,)`` or ``(N, 3)`` for either argument (broadcast). The fraction of the
    momentum error that no dipole can remove at that instant. Returns 0 where ``|h|`` is
    zero, since there is then nothing to remove.
    """
    h = np.atleast_2d(np.asarray(momentum_error_nms, dtype=float))
    b = np.atleast_2d(np.asarray(b_body_t, dtype=float))
    if h.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError("both arguments must have trailing dimension 3")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(b))):
        raise ValueError("both arguments must be finite")
    b_norm = np.linalg.norm(b, axis=-1)
    if np.any(b_norm == 0.0):
        raise ValueError("b_body_t must be non-zero everywhere")
    h_norm = np.linalg.norm(h, axis=-1)
    b_hat = b / b_norm[:, None]
    dot = np.abs(np.sum(h * b_hat, axis=-1))
    out = np.where(h_norm > 0.0, dot / np.where(h_norm > 0.0, h_norm, 1.0), 0.0)
    return out


def averaged_controllability(
    b_history_t: ArrayLike, time_s: ArrayLike | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Time-averaged magnetic controllability Gramian and its eigen-decomposition.

    .. math:: \mathbf{G} = \frac{1}{T}\int_0^T\left(\mathbf{I} - \hat{\mathbf{B}}
              \hat{\mathbf{B}}^{\!\top}\right)dt

    Parameters
    ----------
    b_history_t : (N, 3)
        Field history in a *fixed* frame [T] — normally ECI, or the body frame of an
        inertially fixed vehicle. Averaging body-frame samples of a rotating vehicle
        answers a different question and the caller must decide which they want.
    time_s : (N,) or None
        Sample times [s] for the trapezoidal average; ``None`` means uniform spacing.

    Returns
    -------
    (gramian, eigenvalues, eigenvectors)
        ``gramian`` is 3-by-3, dimensionless, with trace exactly 2 (each instantaneous
        projector has rank 2). ``eigenvalues`` are ascending, each in [0, 1]: the
        fraction of the interval over which the corresponding direction is dumpable. All
        three strictly positive means every direction is dumpable given enough time; the
        smallest is the bottleneck.
    """
    b = np.asarray(b_history_t, dtype=float)
    if b.ndim != 2 or b.shape[1] != 3:
        raise ValueError(f"b_history_t must have shape (N, 3), got {b.shape}")
    if b.shape[0] < 2:
        raise ValueError("at least two field samples are required")
    if not np.all(np.isfinite(b)):
        raise ValueError("b_history_t must be finite")
    norm = np.linalg.norm(b, axis=1)
    if np.any(norm == 0.0):
        raise ValueError("b_history_t must be non-zero at every sample")
    b_hat = b / norm[:, None]
    proj = np.eye(3)[None, :, :] - b_hat[:, :, None] * b_hat[:, None, :]
    if time_s is None:
        gram = proj.mean(axis=0)
    else:
        t = np.asarray(time_s, dtype=float)
        if t.shape != (b.shape[0],):
            raise ValueError(f"time_s must have shape ({b.shape[0]},), got {t.shape}")
        span = float(t[-1] - t[0])
        if span <= 0.0:
            raise ValueError("time_s must be increasing")
        gram = np.trapezoid(proj, t, axis=0) / span
    eigvals, eigvecs = np.linalg.eigh(gram)
    return gram, eigvals, eigvecs


def dipole_cost(dipole_history_am2: ArrayLike, time_s: ArrayLike) -> float:
    """Magnetorquer duty cost ``int |m| dt`` [A m^2 s], trapezoidal.

    This is the natural cost for a magnetic system: coil power is roughly proportional to
    ``m^2`` for a fixed coil, and coil-on time to ``|m|`` at a fixed commanded level, so
    ``int |m| dt`` is what a duty-cycle or thermal budget is written against. It is the
    magnetic analogue of propellant mass and is what the schedulers in
    :mod:`momentummgr.policies` are compared on.
    """
    m = np.asarray(dipole_history_am2, dtype=float)
    t = np.asarray(time_s, dtype=float)
    if m.ndim != 2 or m.shape[1] != 3:
        raise ValueError(f"dipole_history_am2 must have shape (N, 3), got {m.shape}")
    if t.shape != (m.shape[0],):
        raise ValueError(f"time_s must have shape ({m.shape[0]},), got {t.shape}")
    return float(np.trapezoid(np.linalg.norm(m, axis=1), t))


@dataclass(frozen=True)
class ThrusterDump:
    """Cost of dumping a momentum increment with a thruster pair.

    Attributes
    ----------
    momentum_nms : float
        Momentum magnitude removed [N m s].
    impulse_ns : float
        Total impulse required [N s].
    propellant_kg : float
        Propellant mass [kg].
    """

    momentum_nms: float
    impulse_ns: float
    propellant_kg: float


def thruster_dump(
    momentum_nms: ArrayLike | float,
    moment_arm_m: float,
    specific_impulse_s: float,
    couple: bool = True,
    efficiency: float = 1.0,
) -> ThrusterDump:
    r"""Impulse and propellant to dump a momentum increment with thrusters.

    .. math:: I = \frac{|\Delta \mathbf{h}|}{L\,\eta}\,N_{jet}, \qquad
              m_p = \frac{I}{I_{sp} g_0}

    with ``L`` the moment arm from the centre of mass to the thruster line of action and
    ``N_jet = 2`` for a pure couple (two opposed jets, no net force on the vehicle) or 1
    for a single jet, which also imparts a delta-v of ``I / m_sc``.

    Source: the rocket equation in impulse form (any astrodynamics text; Sidi,
    *Spacecraft Dynamics and Control*, gives the momentum-dumping application). Units: m,
    s, dimensionless efficiency; returns N s and kg.

    ``efficiency`` in ``(0, 1]`` lumps cosine losses, plume impingement and minimum
    impulse-bit quantisation into one number. It is not a physical constant and defaults
    to 1, i.e. the optimistic bound; a real system is worse.
    """
    dh = np.asarray(momentum_nms, dtype=float)
    mag = float(np.linalg.norm(dh)) if dh.ndim else float(abs(dh))
    if not np.isfinite(mag) or mag < 0.0:
        raise ValueError(f"momentum_nms must be finite and non-negative in magnitude, got {dh!r}")
    arm = _v.positive(moment_arm_m, "moment_arm_m")
    isp = _v.positive(specific_impulse_s, "specific_impulse_s")
    eta = _v.in_range(efficiency, "efficiency", 1e-6, 1.0)
    n_jet = 2.0 if couple else 1.0
    impulse = n_jet * mag / (arm * eta)
    return ThrusterDump(
        momentum_nms=mag,
        impulse_ns=impulse,
        propellant_kg=impulse / (isp * STANDARD_GRAVITY),
    )
