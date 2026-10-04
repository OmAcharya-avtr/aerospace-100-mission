"""Pass profiles: range, range-rate, Doppler, Doppler rate and pre-compensation.

What this module adds on top of :mod:`dopplerkit.geometry` and
:mod:`dopplerkit.doppler` is tabulation over a pass and the two event
locators a carrier plan needs: the time of closest approach and the Doppler
zero crossing.

Sign convention, carried on the result object
---------------------------------------------
Unchanged from the rest of the package: ``range_rate_mps`` positive when
receding, ``doppler_hz`` positive when approaching (up-shift),
``precomp_offset_hz`` the negative of ``doppler_hz``, ``doppler_rate_hz_per_s``
negative when the Doppler shift is falling.  Over an overhead pass the Doppler
goes positive -> zero -> negative and the Doppler rate stays negative.

Events
------
:func:`time_of_closest_approach_s` minimises the range with
``scipy.optimize.minimize_scalar`` (bounded Brent).
:func:`doppler_zero_crossing_s` brackets the sign change of the range-rate and
refines it with ``scipy.optimize.brentq``.  These are computed by *independent*
methods on purpose: the L1 validation check "the Doppler zero crossing occurs
at closest approach" is only evidence if the two are not the same calculation.

Assumptions and validity
------------------------
* Geometry only, first order in v/c.  No light-time correction is applied to
  the tabulated quantities -- the light-time machinery is
  :mod:`dopplerkit.lighttime`, and the size of the effect on two-way Doppler
  is quantified in ``validation/validate_two_way_relativistic.py``.
* The two event locators assume the range is **unimodal** over the requested
  window, which holds for a single pass of a near-circular orbit but not
  across multiple passes or for a window that spans a whole orbit.  Both
  raise if the window does not bracket a range-rate sign change.
* Both endpoint state callables must return states in the same inertial frame
  at the same epoch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .doppler import (
    DOPPLER_CONVENTION,
    LinkDirection,
    doppler_rate_hz_per_s,
    one_way_doppler_hz,
    precompensation_offset_hz,
    two_way_doppler_hz,
)
from .geometry import (
    RANGE_RATE_CONVENTION,
    State,
    range_acceleration_mps2,
    range_rate_mps,
    slant_range_m,
)

StateFn = Callable[[float], State]
"""Callable mapping time [s] to a :class:`~dopplerkit.geometry.State`."""


@dataclass(frozen=True)
class DopplerProfile:
    """Tabulated link geometry and Doppler over a pass, SI units.

    Arrays are all the same length and aligned with ``times_s``.

    Fields
    ------
    times_s
        Sample epochs [s], in the same timebase as the state callables.
    range_m
        Slant range [m].
    range_rate_mps
        Range-rate [m/s], **positive when receding**.
    range_acceleration_mps2
        Range-acceleration [m/s^2].
    doppler_hz
        Doppler shift [Hz], **positive when approaching**.  One-way or two-way
        according to ``direction``/``ways``.
    doppler_rate_hz_per_s
        Doppler rate [Hz/s].
    precomp_offset_hz
        Transmit-frequency offset [Hz] that lands the carrier on nominal at
        the far end: the negative of the one-way Doppler shift on this leg.
        Present for one-way profiles only; ``None`` for two-way, because
        pre-compensating a coherent two-way link means pre-compensating its
        uplink leg.
    carrier_hz
        Reference carrier [Hz]; for two-way, ``turnaround_ratio * f_uplink``.
    direction, ways, turnaround_ratio
        Link bookkeeping.
    range_rate_convention, doppler_convention
        Verbatim :data:`~dopplerkit.geometry.RANGE_RATE_CONVENTION` and
        :data:`~dopplerkit.doppler.DOPPLER_CONVENTION`, so a profile written
        to disk carries its own sign definition.
    """

    times_s: np.ndarray
    range_m: np.ndarray
    range_rate_mps: np.ndarray
    range_acceleration_mps2: np.ndarray
    doppler_hz: np.ndarray
    doppler_rate_hz_per_s: np.ndarray
    precomp_offset_hz: np.ndarray | None
    carrier_hz: float
    direction: LinkDirection
    ways: int
    turnaround_ratio: float
    range_rate_convention: str = RANGE_RATE_CONVENTION
    doppler_convention: str = DOPPLER_CONVENTION

    def __len__(self) -> int:
        return int(self.times_s.size)

    def peak_doppler_hz(self) -> float:
        """Largest absolute Doppler shift in the profile [Hz], signed."""
        idx = int(np.argmax(np.abs(self.doppler_hz)))
        return float(self.doppler_hz[idx])

    def doppler_span_hz(self) -> float:
        """Peak-to-peak Doppler excursion over the profile [Hz], non-negative.

        This is the number a receiver's acquisition search window has to
        cover, which is why it is separate from :meth:`peak_doppler_hz`.
        """
        return float(np.max(self.doppler_hz) - np.min(self.doppler_hz))


def compute_profile(
    observer_state_fn: StateFn,
    target_state_fn: StateFn,
    times_s: Sequence[float] | np.ndarray,
    carrier_hz: float,
    direction: LinkDirection = LinkDirection.DOWNLINK,
    turnaround_ratio: float = 1.0,
) -> DopplerProfile:
    """Tabulate range, range-rate, Doppler, Doppler rate and pre-compensation.

    Parameters
    ----------
    observer_state_fn, target_state_fn
        Callables ``t [s] -> State``, both in the same inertial frame.  For a
        downlink the observer is the ground station and the target the
        spacecraft; for an uplink the roles of *transmitter* and *receiver*
        swap but the geometry does not, because range-rate is invariant under
        that swap (see :mod:`dopplerkit.geometry`).
    times_s
        Sample epochs [s], at least two, strictly increasing.
    carrier_hz
        Carrier [Hz].  For ``direction=LinkDirection.TWO_WAY`` this is the
        **uplink** carrier and the reference frequency recorded on the result
        is ``turnaround_ratio * carrier_hz``.
    direction
        :class:`~dopplerkit.doppler.LinkDirection`.
    turnaround_ratio
        Coherent transponder ratio G, used only for two-way.

    Returns
    -------
    DopplerProfile
        With the sign conventions attached verbatim.

    Notes
    -----
    Units: [s], [Hz]; outputs [m], [m/s], [m/s^2], [Hz], [Hz/s].  First order
    in v/c, no light-time correction, and the Doppler rate is only as good as
    the accelerations carried on the states (see
    :func:`dopplerkit.geometry.range_acceleration_mps2`).
    """
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError(f"times_s must be a 1-D sequence of length >= 2, got shape {times.shape}")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times_s must be strictly increasing")
    direction = LinkDirection(direction)
    ways = 2 if direction is LinkDirection.TWO_WAY else 1

    rng = np.empty(times.size)
    rate = np.empty(times.size)
    accel = np.empty(times.size)
    for i, t in enumerate(times):
        obs = observer_state_fn(float(t))
        tgt = target_state_fn(float(t))
        rng[i] = slant_range_m(obs, tgt)
        rate[i] = range_rate_mps(obs, tgt)
        accel[i] = range_acceleration_mps2(obs, tgt)

    if ways == 2:
        doppler = np.array([two_way_doppler_hz(r, carrier_hz, turnaround_ratio) for r in rate])
        precomp = None
        reference_hz = float(carrier_hz) * float(turnaround_ratio)
    else:
        doppler = np.array([one_way_doppler_hz(r, carrier_hz) for r in rate])
        precomp = np.array([precompensation_offset_hz(r, carrier_hz) for r in rate])
        reference_hz = float(carrier_hz)

    d_rate = np.array(
        [
            doppler_rate_hz_per_s(a, carrier_hz, ways=ways, turnaround_ratio=turnaround_ratio)
            for a in accel
        ]
    )
    return DopplerProfile(
        times_s=times,
        range_m=rng,
        range_rate_mps=rate,
        range_acceleration_mps2=accel,
        doppler_hz=doppler,
        doppler_rate_hz_per_s=d_rate,
        precomp_offset_hz=precomp,
        carrier_hz=reference_hz,
        direction=direction,
        ways=ways,
        turnaround_ratio=float(turnaround_ratio),
    )


def time_of_closest_approach_s(
    observer_state_fn: StateFn,
    target_state_fn: StateFn,
    t_lo_s: float,
    t_hi_s: float,
    xatol_s: float = 1e-6,
) -> float:
    """Epoch of minimum slant range in ``[t_lo_s, t_hi_s]`` [s].

    Bounded Brent minimisation of the range (``scipy.optimize.minimize_scalar``
    with ``method="bounded"``).  Assumes the range is unimodal over the
    window, which holds within one pass of a near-circular orbit.  Raises
    ``ValueError`` on an empty or reversed window.
    """
    if not t_hi_s > t_lo_s:
        raise ValueError(f"require t_hi_s > t_lo_s, got {t_lo_s} and {t_hi_s}")

    def _range(t: float) -> float:
        return slant_range_m(observer_state_fn(float(t)), target_state_fn(float(t)))

    result = minimize_scalar(
        _range, bounds=(float(t_lo_s), float(t_hi_s)), method="bounded",
        options={"xatol": float(xatol_s)},
    )
    if not result.success:
        raise RuntimeError(f"closest-approach minimisation failed: {result.message}")
    return float(result.x)


def doppler_zero_crossing_s(
    observer_state_fn: StateFn,
    target_state_fn: StateFn,
    t_lo_s: float,
    t_hi_s: float,
    xtol_s: float = 1e-9,
) -> float:
    """Epoch at which the Doppler shift crosses zero in ``[t_lo_s, t_hi_s]`` [s].

    Equivalent to the root of the range-rate, found with
    ``scipy.optimize.brentq``.  The carrier cancels out: the Doppler zero and
    the range-rate zero are the same instant at any frequency, which is why
    this function takes no carrier argument.

    The crossing direction is fixed by the convention: the Doppler goes from
    positive (approaching) to negative (receding), so the range-rate goes from
    negative to positive.  Raises ``ValueError`` if the window does not
    bracket a sign change, which is the honest failure for a window that does
    not contain a closest approach.
    """
    if not t_hi_s > t_lo_s:
        raise ValueError(f"require t_hi_s > t_lo_s, got {t_lo_s} and {t_hi_s}")

    def _rate(t: float) -> float:
        return range_rate_mps(observer_state_fn(float(t)), target_state_fn(float(t)))

    lo, hi = _rate(t_lo_s), _rate(t_hi_s)
    if lo * hi > 0.0:
        raise ValueError(
            f"range-rate does not change sign over [{t_lo_s}, {t_hi_s}] "
            f"(rho_dot = {lo:.6g} and {hi:.6g} m/s); the window contains no closest approach"
        )
    return float(brentq(_rate, float(t_lo_s), float(t_hi_s), xtol=float(xtol_s)))


__all__ = [
    "DopplerProfile",
    "StateFn",
    "compute_profile",
    "doppler_zero_crossing_s",
    "time_of_closest_approach_s",
]
