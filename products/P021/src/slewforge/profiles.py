"""Rest-to-rest eigenaxis slew profiles and their closed-form timing.

A rest-to-rest eigenaxis slew rotates through ``Delta`` [rad] about a single
axis ``e`` fixed in both frames, starting and ending at zero rate. Every
profile here supplies ``psi(t)``, ``psi_dot(t)`` and ``psi_ddot(t)`` with
``psi(0) = 0``, ``psi(T) = Delta`` and ``psi_dot(0) = psi_dot(T) = 0``.

Sizing model (stated because it is the rule of thumb this package exists to
check)
--------------------------------------------------------------------------
The scalar model treats the slew as a single-axis rotation of effective
inertia ``J_e = e^T J e`` [kg m^2] driven by an available torque ``tau_a``
[N*m], giving the peak angular acceleration ``alpha = tau_a / J_e``
[rad/s^2]. It drops the gyroscopic term ``omega x J omega``, which vanishes
only when ``e`` is a principal axis of ``J``. :mod:`slewforge.dynamics`
computes the exact torque an eigenaxis slew needs,

    tau(t) = psi_ddot J e + psi_dot^2 (e x J e),

and `validation/validate_eigenaxis_time.py` measures how much the scalar model
under-sizes the torque for a non-principal eigenaxis.

Closed forms (all exact, all derived in `validation/VALIDATION.md` sec. 1)
-------------------------------------------------------------------------
Bang-bang, no rate limit: accelerate at ``alpha`` for ``T/2``, decelerate for
``T/2``. Then ``Delta = alpha (T/2)^2`` and

    T = 2 sqrt(Delta / alpha),   omega_peak = sqrt(Delta alpha)

Bang-bang with a rate limit ``omega_max`` (trapezoidal): if
``sqrt(Delta alpha) > omega_max`` a coast phase appears and

    T = Delta / omega_max + omega_max / alpha

Smoothed, ``psi_ddot(t) = alpha sin(2 pi t / T)``: acceleration and therefore
commanded torque start and end at zero, so the profile excites no step in
torque. Integrating, ``Delta = alpha T^2 / (2 pi)`` and

    T = sqrt(2 pi Delta / alpha),   omega_peak = alpha T / pi

so a smoothed slew is ``sqrt(2 pi) / 2 = 1.2533`` times slower than bang-bang
for the same torque -- a 25.33 % time penalty bought with continuous torque.
Under a rate limit the smoothed profile stretches to ``T = 2 Delta / omega_max``
(peak rate exactly at the limit, no coast phase).

Momentum accounting is common to all three: for any rest-to-rest single-axis
profile the wheel momentum is ``h(t) = J_e psi_dot(t)``, so

    h_peak = J_e omega_peak,   integral |tau| dt = 2 J_e omega_peak

the second because the profile must build the peak momentum and then remove
all of it. That identity is a property test.

References
----------
B. Wie, *Space Vehicle Dynamics and Control*, 2nd ed., AIAA (2008), Sec. 5.3
    and 7.4 -- eigenaxis rest-to-rest slews, bang-bang time-optimal control of
    a single axis, and the momentum a slew demands of a wheel array.
J. R. Wertz (ed.), *Spacecraft Attitude Determination and Control*, Reidel
    (1978), Sec. 18.3 -- manoeuvre sizing.
J.-J. E. Slotine and W. Li, *Applied Nonlinear Control*, Prentice Hall (1991),
    Sec. 7.1 -- why a discontinuous acceleration command excites structural
    modes, the reason smoothed profiles are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "PROFILE_NAMES",
    "SlewProfile",
    "bang_bang_profile",
    "make_profile",
    "smoothed_profile",
]

PROFILE_NAMES = ("bang_bang", "trapezoidal", "smoothed")
"""Profile keys accepted by :func:`make_profile`. ``trapezoidal`` is
``bang_bang`` with the rate limit active; both are produced by
:func:`bang_bang_profile`, which selects between them automatically."""


@dataclass(frozen=True)
class SlewProfile:
    """A rest-to-rest single-axis timing law.

    Attributes
    ----------
    kind : str
        ``"bang_bang"``, ``"trapezoidal"`` or ``"smoothed"``.
    angle : float
        Total rotation ``Delta`` [rad], ``>= 0``.
    duration : float
        Manoeuvre time ``T`` [s].
    peak_accel : float
        Peak angular acceleration [rad/s^2].
    peak_rate : float
        Peak angular rate [rad/s].
    coast_time : float
        Duration of the constant-rate phase [s]; 0 unless rate-limited.
    inertia : float
        Effective inertia [kg m^2] used for the momentum figures. The planner
        passes ``|J e|``, the magnitude of the inertia tensor applied to the
        eigenaxis, because that is what sets the momentum the wheels must
        store; it equals ``e^T J e`` exactly when ``e`` is a principal axis.
    """

    kind: str
    angle: float
    duration: float
    peak_accel: float
    peak_rate: float
    coast_time: float
    inertia: float

    @property
    def peak_momentum(self) -> float:
        """Peak stored momentum ``J_e omega_peak`` [N*m*s]."""
        return self.inertia * self.peak_rate

    @property
    def momentum_throughput(self) -> float:
        """``integral |tau| dt = 2 J_e omega_peak`` [N*m*s].

        The total momentum the wheels must take in and give back. Equal to
        twice the peak for every rest-to-rest profile here, which is the
        cheapest available consistency check on a profile implementation.
        """
        return 2.0 * self.inertia * self.peak_rate

    @property
    def peak_torque(self) -> float:
        """Peak scalar torque ``J_e alpha`` [N*m] in the scalar sizing model."""
        return self.inertia * self.peak_accel

    @property
    def switch_times(self) -> tuple[float, ...]:
        """Instants [s] where the commanded acceleration jumps.

        Empty for the smoothed profile, which is continuous. ``(T/2,)`` for
        bang-bang; ``(t_acc, t_acc + t_coast)`` for trapezoidal. A fixed-step
        integrator that steps across one of these loses its order of accuracy,
        so :func:`slewforge.dynamics.simulate_profile` integrates the phases
        separately -- see `validation/validate_eigenaxis_time.py` PART D for
        the measured difference.
        """
        if self.duration <= 0.0 or self.kind == "smoothed":
            return ()
        t_acc = self.peak_rate / self.peak_accel if self.peak_accel > 0.0 else 0.0
        if self.coast_time > 0.0:
            return (t_acc, t_acc + self.coast_time)
        return (t_acc,)

    def _t(self, t: ArrayLike) -> NDArray[np.float64]:
        a = np.atleast_1d(np.asarray(t, dtype=float))
        if np.any(a < -1e-12) or np.any(a > self.duration + 1e-9):
            raise ValueError(
                f"time must lie in [0, {self.duration}] s, got range "
                f"[{float(np.min(a))}, {float(np.max(a))}]"
            )
        return np.clip(a, 0.0, self.duration)

    def angle_at(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Swept angle ``psi(t)`` [rad]."""
        a = self._t(t)
        out = _eval(self, a, 0)
        return float(out[0]) if np.ndim(t) == 0 else out

    def rate_at(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Angular rate ``psi_dot(t)`` [rad/s]."""
        a = self._t(t)
        out = _eval(self, a, 1)
        return float(out[0]) if np.ndim(t) == 0 else out

    def accel_at(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Angular acceleration ``psi_ddot(t)`` [rad/s^2]."""
        a = self._t(t)
        out = _eval(self, a, 2)
        return float(out[0]) if np.ndim(t) == 0 else out

    def momentum_at(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Stored momentum ``J_e psi_dot(t)`` [N*m*s]."""
        r = self.rate_at(t)
        return self.inertia * r if np.ndim(t) else float(self.inertia * float(r))


def _eval(p: SlewProfile, t: NDArray[np.float64], order: int) -> NDArray[np.float64]:
    """Evaluate psi (order 0), psi_dot (1) or psi_ddot (2) at times ``t``."""
    if p.duration <= 0.0:
        return np.zeros_like(t)
    if p.kind == "smoothed":
        w = 2.0 * math.pi / p.duration
        a = p.peak_accel
        if order == 2:
            return a * np.sin(w * t)
        if order == 1:
            return (a / w) * (1.0 - np.cos(w * t))
        return (a / w) * (t - np.sin(w * t) / w)
    # bang-bang / trapezoidal
    ta = p.peak_rate / p.peak_accel if p.peak_accel > 0.0 else 0.0
    tc = p.coast_time
    t1, t2 = ta, ta + tc
    out = np.zeros_like(t)
    ph1 = t <= t1
    ph2 = (t > t1) & (t <= t2)
    ph3 = t > t2
    if order == 2:
        out[ph1] = p.peak_accel
        out[ph2] = 0.0
        out[ph3] = -p.peak_accel
        return out
    if order == 1:
        out[ph1] = p.peak_accel * t[ph1]
        out[ph2] = p.peak_rate
        out[ph3] = p.peak_rate - p.peak_accel * (t[ph3] - t2)
        return np.maximum(out, 0.0)
    s1 = 0.5 * p.peak_accel * t1 * t1
    s2 = s1 + p.peak_rate * tc
    out[ph1] = 0.5 * p.peak_accel * t[ph1] ** 2
    out[ph2] = s1 + p.peak_rate * (t[ph2] - t1)
    dt = t[ph3] - t2
    out[ph3] = s2 + p.peak_rate * dt - 0.5 * p.peak_accel * dt**2
    return out


def _check(angle: float, peak_accel: float, inertia: float, rate_limit: float | None) -> None:
    if not math.isfinite(angle) or angle < 0.0:
        raise ValueError(f"angle must be finite and >= 0 rad, got {angle}")
    if not math.isfinite(peak_accel) or peak_accel <= 0.0:
        raise ValueError(f"peak_accel must be finite and > 0 rad/s^2, got {peak_accel}")
    if not math.isfinite(inertia) or inertia <= 0.0:
        raise ValueError(f"inertia must be finite and > 0 kg m^2, got {inertia}")
    if rate_limit is not None and (not math.isfinite(rate_limit) or rate_limit <= 0.0):
        raise ValueError(f"rate_limit must be finite and > 0 rad/s, got {rate_limit}")


def bang_bang_profile(
    angle: float,
    peak_accel: float,
    inertia: float,
    rate_limit: float | None = None,
) -> SlewProfile:
    """Time-optimal single-axis rest-to-rest profile, rate-limited if needed.

    Parameters
    ----------
    angle : float
        Rotation ``Delta`` [rad], ``>= 0``.
    peak_accel : float
        Available angular acceleration ``alpha = tau_a / J_e`` [rad/s^2].
    inertia : float
        Effective inertia ``J_e`` [kg m^2], used only for momentum figures.
    rate_limit : float or None
        Maximum permitted angular rate [rad/s]. ``None`` means unlimited, and
        the profile is then pure bang-bang.

    Returns
    -------
    SlewProfile
        ``kind == "trapezoidal"`` when the rate limit binds, otherwise
        ``"bang_bang"``.

    Notes
    -----
    A zero-angle slew returns a zero-duration profile rather than raising: the
    planner legitimately builds one when a via point coincides with an endpoint.
    """
    _check(angle, peak_accel, inertia, rate_limit)
    if angle == 0.0:
        return SlewProfile("bang_bang", 0.0, 0.0, peak_accel, 0.0, 0.0, inertia)
    omega_bb = math.sqrt(angle * peak_accel)
    if rate_limit is None or omega_bb <= rate_limit:
        t = 2.0 * math.sqrt(angle / peak_accel)
        return SlewProfile("bang_bang", angle, t, peak_accel, omega_bb, 0.0, inertia)
    t_acc = rate_limit / peak_accel
    s_acc = 0.5 * rate_limit * t_acc
    t_coast = (angle - 2.0 * s_acc) / rate_limit
    t = 2.0 * t_acc + t_coast
    return SlewProfile("trapezoidal", angle, t, peak_accel, rate_limit, t_coast, inertia)


def smoothed_profile(
    angle: float,
    peak_accel: float,
    inertia: float,
    rate_limit: float | None = None,
) -> SlewProfile:
    """Sinusoidal-acceleration rest-to-rest profile with zero end torque.

    ``psi_ddot(t) = a sin(2 pi t / T)``. Torque starts and ends at zero and is
    continuous throughout, which is why flight software uses profiles of this
    family instead of bang-bang on a spacecraft with flexible appendages.

    Under no rate limit ``T = sqrt(2 pi Delta / alpha)`` with ``a = alpha``.
    When the rate limit binds, ``T = 2 Delta / omega_max`` and the amplitude
    drops to ``a = 2 pi Delta / T^2 <= alpha``, so the acceleration limit is
    still respected.

    Parameters and units as :func:`bang_bang_profile`.
    """
    _check(angle, peak_accel, inertia, rate_limit)
    if angle == 0.0:
        return SlewProfile("smoothed", 0.0, 0.0, peak_accel, 0.0, 0.0, inertia)
    t_free = math.sqrt(2.0 * math.pi * angle / peak_accel)
    omega_free = peak_accel * t_free / math.pi
    if rate_limit is None or omega_free <= rate_limit:
        return SlewProfile("smoothed", angle, t_free, peak_accel, omega_free, 0.0, inertia)
    t = 2.0 * angle / rate_limit
    a = 2.0 * math.pi * angle / (t * t)
    return SlewProfile("smoothed", angle, t, a, rate_limit, 0.0, inertia)


def make_profile(
    kind: str,
    angle: float,
    peak_accel: float,
    inertia: float,
    rate_limit: float | None = None,
) -> SlewProfile:
    """Dispatch on a profile name from :data:`PROFILE_NAMES`.

    ``"bang_bang"`` and ``"trapezoidal"`` both call :func:`bang_bang_profile`;
    which of the two comes back depends on whether the rate limit binds, so
    asking for ``"trapezoidal"`` without a rate limit returns a bang-bang
    profile rather than raising.
    """
    if kind in ("bang_bang", "trapezoidal"):
        return bang_bang_profile(angle, peak_accel, inertia, rate_limit)
    if kind == "smoothed":
        return smoothed_profile(angle, peak_accel, inertia, rate_limit)
    raise ValueError(f"unknown profile {kind!r}; expected one of {PROFILE_NAMES}")
