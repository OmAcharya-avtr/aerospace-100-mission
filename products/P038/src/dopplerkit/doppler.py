"""One-way and two-way Doppler, kept explicitly separate.

THE SIGN CONVENTION, inherited from :mod:`dopplerkit.geometry`
--------------------------------------------------------------
Range-rate ``rho_dot`` is positive when the endpoints are **separating**.
Therefore:

    Doppler shift is POSITIVE (up-shifted, received > transmitted) when the
    endpoints are APPROACHING (rho_dot < 0), and NEGATIVE when receding.

One-way, classical (non-relativistic, first order in v/c):

    f_received = f_transmitted * (1 - rho_dot / c)
    Delta_f_1way = f_received - f_transmitted = -f_c * rho_dot / c     [Hz]

This is the classical Doppler relation for a line-of-sight relative speed,
with the minus sign fixed by the receding-positive range-rate convention.

LINK DIRECTION
--------------
The first-order one-way shift depends on the link direction **only through
which carrier frequency f_c is used**.  The range-rate itself is invariant
under reversing the link (see :mod:`dopplerkit.geometry`), so

    Delta_f_up   = -f_up   * rho_dot / c      (ground -> spacecraft)
    Delta_f_down = -f_down * rho_dot / c      (spacecraft -> ground)

have the *same sign* and differ only by the ratio f_up / f_down.  Every
public function here therefore takes the carrier explicitly and records the
direction on the returned object rather than silently assuming one.

TWO-WAY, COHERENT TRANSPONDER
-----------------------------
Ground transmits ``f_up``.  The spacecraft transponder multiplies the received
frequency by the turnaround ratio ``G`` and retransmits.  The ground receiver
compares against the reference ``G * f_up``.  Composing the two one-way legs:

    f_ground_rx = G * f_up * (1 - rho_dot_up/c) * (1 - rho_dot_down/c)

so, exactly,

    Delta_f_2way = -G * f_up * [ (rho_dot_up + rho_dot_down)/c
                                 - rho_dot_up * rho_dot_down / c^2 ]

:func:`two_way_doppler_two_leg_hz` keeps only the first-order bracket term and
returns the dropped cross term separately, because it is the same order as the
relativistic correction (both are O(beta^2)) and a reader comparing the two
needs both numbers.  When the two legs are evaluated at the same epoch
(``rho_dot_up == rho_dot_down``) and ``G == 1``, the first-order two-way shift
is **exactly twice** the one-way shift; that identity is asserted in
``tests/test_doppler.py`` and measured in
``validation/validate_two_way_relativistic.py``.

WHAT ORDER IS KEPT AND WHAT IS DROPPED
--------------------------------------
Kept in the main functions: O(beta^1), beta = rho_dot/c.

Reported but **not** applied by default:

* ``relativistic_fraction_second_order`` -- the special-relativistic O(beta^2)
  term ``(rho_dot/c)^2 - v^2/(2 c^2)``, from expanding the exact relativistic
  one-way Doppler ratio ``sqrt(1 - v^2/c^2) / (1 + rho_dot/c)`` (standard
  special-relativity result; the same expansion underlies the Doppler
  observable formulation in Moyer, *Formulation for Observed and Computed
  Values of Deep Space Network Data Types for Navigation*, JPL Deep Space
  Communications and Navigation Series Monograph 2, 2000).  The second part is
  transmitter proper-time dilation and does not vanish at closest approach.
* ``gravitational_shift_fraction`` -- an order-of-magnitude Newtonian
  potential-difference estimate ``(U_rx - U_tx)/c^2`` with a point-mass Earth.

Dropped entirely, never modelled here: higher relativistic orders; the
Shapiro / gravitational light-time delay; tropospheric and ionospheric
delay and their time derivatives (which at L- and S-band commonly exceed the
O(beta^2) terms above); antenna phase-centre motion; transponder group delay
and its drift; oscillator instability and drift; and any troposphere-induced
bending of the line of sight.  None of these are in this package, and a
sub-Hz carrier prediction needs most of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .constants import C_M_S, MU_EARTH_M3_S2

DOPPLER_CONVENTION = (
    "Doppler positive (up-shift) when approaching, i.e. "
    "Delta_f = -f_carrier * rho_dot / c with rho_dot positive when receding"
)
"""One-line statement of the Doppler sign convention, carried on every result."""


class LinkDirection(str, Enum):
    """Which way the radio signal travels.

    The direction does not change the sign of the first-order Doppler shift
    (see module docstring); it is recorded so that a result can never be
    misread, and so that the carrier used is unambiguous.
    """

    UPLINK = "uplink"
    """Ground station transmits, spacecraft receives."""

    DOWNLINK = "downlink"
    """Spacecraft transmits, ground station receives."""

    TWO_WAY = "two-way"
    """Ground transmits, coherent transponder returns, ground receives."""


def _check_carrier(carrier_hz: float) -> float:
    if not isinstance(carrier_hz, (int, float)) or isinstance(carrier_hz, bool):
        raise TypeError(f"carrier_hz must be a real number, got {type(carrier_hz).__name__}")
    carrier_hz = float(carrier_hz)
    if not carrier_hz > 0.0:
        raise ValueError(f"carrier_hz must be > 0 Hz, got {carrier_hz}")
    return carrier_hz


def _check_range_rate(range_rate_mps: float, name: str = "range_rate_mps") -> float:
    value = float(range_rate_mps)
    if abs(value) >= C_M_S:
        raise ValueError(
            f"{name} must satisfy |rho_dot| < c = {C_M_S} m/s, got {value}; "
            "this package is first-order in v/c and is not valid near c"
        )
    return value


@dataclass(frozen=True)
class DopplerObservable:
    """A Doppler result with its convention, direction and order attached.

    Fields
    ------
    doppler_hz
        Frequency shift Delta_f = f_received - f_reference [Hz].  Positive
        means up-shifted, which by this package's convention means
        **approaching**.
    carrier_hz
        Reference carrier frequency [Hz].  For two-way this is
        ``turnaround_ratio * f_uplink``, the frequency the ground receiver
        compares against.
    range_rate_mps
        Range-rate used [m/s], positive when receding.  For a two-leg two-way
        result this is the *sum* of the two legs divided by two, i.e. the mean
        leg range-rate, so that ``doppler_hz`` is still
        ``-2 * carrier_hz * range_rate_mps / c``.
    direction
        :class:`LinkDirection`.
    ways
        1 for one-way, 2 for two-way.
    order_kept
        Text naming the order in v/c retained, e.g. ``"O(beta^1)"``.
    convention
        Verbatim :data:`DOPPLER_CONVENTION`.
    """

    doppler_hz: float
    carrier_hz: float
    range_rate_mps: float
    direction: LinkDirection
    ways: int
    order_kept: str = "O(beta^1), beta = rho_dot/c"
    convention: str = DOPPLER_CONVENTION

    def as_dict(self) -> dict[str, object]:
        """Plain-dict view, JSON-serialisable, units in the key names."""
        return {
            "doppler_hz": self.doppler_hz,
            "carrier_hz": self.carrier_hz,
            "range_rate_mps": self.range_rate_mps,
            "direction": self.direction.value,
            "ways": self.ways,
            "order_kept": self.order_kept,
            "convention": self.convention,
        }


def one_way_doppler_hz(range_rate_mps: float, carrier_hz: float) -> float:
    """One-way classical Doppler shift [Hz].

    ``Delta_f = -carrier_hz * rho_dot / c``.

    Convention: **positive return value means up-shift, which means
    approaching** (rho_dot < 0).  Negative means receding.

    Direction: one-way, either uplink or downlink -- the value is identical
    for both at a single epoch provided ``carrier_hz`` is the carrier of the
    leg in question.  Use :func:`doppler_observable` if you want the direction
    recorded on the result.

    Ways: one.

    Units: ``range_rate_mps`` [m/s], ``carrier_hz`` [Hz], return [Hz].

    Reference and validity: the classical (non-relativistic) Doppler relation
    ``f_rx = f_tx (1 - rho_dot/c)``, first order in v/c.  Fractional error of
    the neglected terms is O(beta^2) ~ 6e-10 for a 7.6 km/s LEO, quantified in
    ``validation/validate_two_way_relativistic.py``.  Raises ``ValueError``
    for |rho_dot| >= c.
    """
    carrier_hz = _check_carrier(carrier_hz)
    rho_dot = _check_range_rate(range_rate_mps)
    return -carrier_hz * rho_dot / C_M_S


def two_way_doppler_hz(
    range_rate_mps: float,
    uplink_carrier_hz: float,
    turnaround_ratio: float = 1.0,
) -> float:
    """Two-way coherent Doppler shift [Hz], both legs at one epoch.

    ``Delta_f = -2 * G * f_up * rho_dot / c`` with ``G = turnaround_ratio``.

    Convention: **positive means up-shift, which means approaching**, exactly
    as one-way.

    Direction: two-way -- ground transmits ``uplink_carrier_hz``, a coherent
    transponder returns ``G`` times what it received, ground receives.  The
    shift is referenced to ``G * uplink_carrier_hz``.

    Ways: two.  With ``turnaround_ratio = 1`` this is **exactly twice**
    ``one_way_doppler_hz(range_rate_mps, uplink_carrier_hz)``; that is the
    non-relativistic-limit identity the spec for this product calls out, and
    it is asserted exactly (zero tolerance) in the test suite.

    Units: [m/s], [Hz], dimensionless ratio; return [Hz].

    Validity: assumes both legs see the same range-rate, i.e. the light time
    is neglected.  The light-time-separated form is
    :func:`two_way_doppler_two_leg_hz`; for a 500 km LEO the two differ by
    about 1e-4 relative, quantified in
    ``validation/validate_two_way_relativistic.py``.  Reference: the composed
    one-way legs of a coherent transponder, as formulated for DSN two-way
    Doppler in Moyer (2000), Monograph 2.
    """
    carrier_hz = _check_carrier(uplink_carrier_hz)
    ratio = float(turnaround_ratio)
    if not ratio > 0.0:
        raise ValueError(f"turnaround_ratio must be > 0, got {ratio}")
    rho_dot = _check_range_rate(range_rate_mps)
    return -2.0 * ratio * carrier_hz * rho_dot / C_M_S


def two_way_doppler_two_leg_hz(
    range_rate_up_mps: float,
    range_rate_down_mps: float,
    uplink_carrier_hz: float,
    turnaround_ratio: float = 1.0,
) -> tuple[float, float]:
    """Two-way Doppler with the two legs evaluated separately [Hz].

    Returns ``(first_order_hz, dropped_cross_term_hz)`` where

        first_order_hz = -G f_up (rho_dot_up + rho_dot_down) / c
        dropped_cross_term_hz = +G f_up rho_dot_up rho_dot_down / c^2

    The sum of the two is the exact composition of the two classical one-way
    legs; the cross term is O(beta^2) and is returned rather than applied so
    that it can be compared with the relativistic O(beta^2) terms, which are
    the same size.

    Convention: **positive means up-shift, which means approaching** -- both
    legs use range-rate positive when receding.

    Direction: two-way, referenced to ``G * uplink_carrier_hz``.

    Ways: two.

    Units: both range-rates [m/s], carrier [Hz], returns both terms in [Hz].

    Validity: the two legs must be the up-leg and down-leg range-rates of the
    *same* two-way measurement, i.e. evaluated at the transmit and receive
    light-time-corrected epochs from :mod:`dopplerkit.lighttime`.  Passing the
    same value twice reproduces :func:`two_way_doppler_hz` in the first
    element.  Reference: Moyer (2000), Monograph 2, two-way Doppler
    formulation.
    """
    carrier_hz = _check_carrier(uplink_carrier_hz)
    ratio = float(turnaround_ratio)
    if not ratio > 0.0:
        raise ValueError(f"turnaround_ratio must be > 0, got {ratio}")
    up = _check_range_rate(range_rate_up_mps, "range_rate_up_mps")
    down = _check_range_rate(range_rate_down_mps, "range_rate_down_mps")
    first_order = -ratio * carrier_hz * (up + down) / C_M_S
    cross = ratio * carrier_hz * up * down / C_M_S**2
    return first_order, cross


def doppler_rate_hz_per_s(
    range_acceleration_mps2: float,
    carrier_hz: float,
    ways: int = 1,
    turnaround_ratio: float = 1.0,
) -> float:
    """Doppler rate d(Delta_f)/dt [Hz/s].

    ``d(Delta_f)/dt = -ways * G * carrier_hz * rho_ddot / c``.

    Convention: differentiating the Doppler convention, a **positive**
    range-acceleration (recession rate increasing) gives a **negative**
    Doppler rate.  Over an overhead pass the Doppler rate is negative
    throughout and most negative at closest approach, where the Doppler shift
    crosses zero from positive (approaching) to negative (receding).

    Direction: whichever leg ``carrier_hz`` belongs to; for ``ways=2`` the
    carrier is the uplink carrier and the result is referenced to
    ``G * carrier_hz``.

    Ways: ``ways`` must be 1 or 2.

    Units: [m/s^2], [Hz], return [Hz/s].

    Validity: first order in v/c, and only as good as ``rho_ddot``.  Note that
    :func:`dopplerkit.geometry.range_acceleration_mps2` returns the kinematic
    term only unless accelerations are supplied on the states; the analytic
    circular model in :mod:`dopplerkit.analytic` supplies the exact value for
    its own geometry.
    """
    carrier_hz = _check_carrier(carrier_hz)
    if ways not in (1, 2):
        raise ValueError(f"ways must be 1 or 2, got {ways}")
    ratio = float(turnaround_ratio) if ways == 2 else 1.0
    if not ratio > 0.0:
        raise ValueError(f"turnaround_ratio must be > 0, got {ratio}")
    return -float(ways) * ratio * carrier_hz * float(range_acceleration_mps2) / C_M_S


def precompensation_offset_hz(
    range_rate_mps: float,
    nominal_carrier_hz: float,
    exact: bool = False,
) -> float:
    """Transmit-frequency offset that lands the carrier on nominal at the far end [Hz].

    The transmitter sends ``nominal_carrier_hz + offset`` so that the receiver
    sees ``nominal_carrier_hz``.

    First order (``exact=False``, the default)::

        offset = +nominal_carrier_hz * rho_dot / c = -one_way_doppler_hz(...)

    Exact inversion of the classical relation (``exact=True``)::

        offset = nominal_carrier_hz * (1/(1 - rho_dot/c) - 1)

    Convention: the offset is **the negative of the Doppler shift**.  When
    approaching (rho_dot < 0) the link up-shifts, so the transmitter must send
    **low**: the offset is negative.  When receding it is positive.  Getting
    this backwards doubles the residual carrier error instead of cancelling
    it, which is the failure mode this function exists to prevent, and it is
    tested with an explicit sign assertion.

    Direction: applies to either leg; ``nominal_carrier_hz`` is the frequency
    the far-end receiver is tuned to.

    Ways: one.  Pre-compensating a two-way link means pre-compensating the
    uplink leg only -- the transponder turnaround carries the rest -- so there
    is deliberately no two-way variant of this function.

    Units: [m/s], [Hz], return [Hz].

    Validity: open-loop pre-compensation with a perfectly known range-rate.
    Residual carrier error is then set by range-rate prediction error, not by
    this arithmetic; see the Limitations section of the README.
    """
    carrier_hz = _check_carrier(nominal_carrier_hz)
    rho_dot = _check_range_rate(range_rate_mps)
    if exact:
        return carrier_hz * (1.0 / (1.0 - rho_dot / C_M_S) - 1.0)
    return carrier_hz * rho_dot / C_M_S


def relativistic_fraction_second_order(range_rate_mps: float, speed_mps: float) -> float:
    """Second-order (O(beta^2)) special-relativistic correction, as a fraction [-].

    ``(rho_dot/c)^2 - speed^2 / (2 c^2)``

    From expanding the exact relativistic one-way Doppler ratio

        f_rx / f_tx = sqrt(1 - v^2/c^2) / (1 + rho_dot/c)
                    = 1 - rho_dot/c + (rho_dot/c)^2 - v^2/(2 c^2) + O(beta^3)

    where ``v`` is the inertial speed of the **transmitter** (the clock whose
    proper time is dilated).  The first term is the second-order longitudinal
    term; the second is transmitter time dilation, which does **not** vanish
    at closest approach and is therefore the dominant relativistic term there.

    Convention: this is a fractional frequency offset to be **added** to the
    classical ``-rho_dot/c`` fraction, with the same sign convention
    (positive = up-shift).  Multiply by the carrier to get Hz, or use
    :func:`relativistic_correction_hz`.

    Ways: one-way.  For a two-way link the two legs each carry their own term
    and the transponder's own proper time enters as well; that composition is
    not implemented here and is listed as a limitation.

    Units: both inputs [m/s], return dimensionless.

    Validity and what is dropped: special relativity only, to O(beta^2).  No
    gravitational redshift (see :func:`gravitational_shift_fraction`), no
    Shapiro delay, no higher orders.  Standard special-relativity result; the
    same expansion underlies the Doppler observable in Moyer (2000),
    Monograph 2.  Magnitude for a representative 500 km LEO pass is reported
    in ``validation/validate_two_way_relativistic.py`` -- about -3.2e-10
    fractional at closest approach, which is sub-Hz at S-band.
    """
    rho_dot = _check_range_rate(range_rate_mps)
    speed = float(speed_mps)
    if speed < 0.0:
        raise ValueError(f"speed_mps must be >= 0, got {speed}")
    if speed >= C_M_S:
        raise ValueError(f"speed_mps must be < c, got {speed}")
    return (rho_dot / C_M_S) ** 2 - speed**2 / (2.0 * C_M_S**2)


def relativistic_correction_hz(
    range_rate_mps: float,
    speed_mps: float,
    carrier_hz: float,
) -> float:
    """The O(beta^2) special-relativistic correction expressed in Hz.

    ``carrier_hz * relativistic_fraction_second_order(rho_dot, speed)``.

    Convention, direction, ways, units and validity: exactly those of
    :func:`relativistic_fraction_second_order`, times a carrier in Hz, giving
    Hz.  **Not** added to the one-way or two-way results by this package; it is
    reported so a reader can decide whether it matters at their carrier.
    """
    carrier_hz = _check_carrier(carrier_hz)
    return carrier_hz * relativistic_fraction_second_order(range_rate_mps, speed_mps)


def gravitational_shift_fraction(
    r_transmitter_m: float,
    r_receiver_m: float,
    mu_m3_s2: float = MU_EARTH_M3_S2,
) -> float:
    """Order-of-magnitude static gravitational frequency shift, as a fraction [-].

    ``(U_tx - U_rx) / c^2`` with a point-mass Newtonian potential
    ``U(r) = -mu/r``, i.e.

        fraction = (mu/c^2) * (1/r_rx - 1/r_tx)

    from the static-field ratio ``f_rx/f_tx = 1 + (U_tx - U_rx)/c^2``.

    Convention: positive means the received frequency is **up-shifted**
    relative to the transmitted one, consistent with the rest of this module.
    A downlink (``r_tx > r_rx``) therefore gives a **positive** fraction: the
    photon falls into the well and blueshifts.  An uplink gives a negative
    fraction, the classic redshift.  Sign-checked in ``tests/test_signs.py``.

    Direction and ways: one leg, from the transmitter radius to the receiver
    radius.  Both radii are geocentric distances [m].

    Units: [m], [m], [m^3/s^2]; return dimensionless.

    Validity: this is an **order-of-magnitude indication only**.  It uses a
    spherical point-mass potential and omits the Earth's rotational
    (centrifugal) potential, the J2 term, the distinction between coordinate
    and proper time at each end, and the Shapiro delay.  Treat it as "is this
    term worth worrying about at my carrier", not as a correction to apply.
    The standard treatment is in Moyer (2000), Monograph 2; what is
    implemented here is only the leading static potential difference.
    """
    r_tx = float(r_transmitter_m)
    r_rx = float(r_receiver_m)
    mu = float(mu_m3_s2)
    if r_tx <= 0.0 or r_rx <= 0.0:
        raise ValueError(f"radii must be > 0 m, got r_tx={r_tx}, r_rx={r_rx}")
    if mu <= 0.0:
        raise ValueError(f"mu_m3_s2 must be > 0, got {mu}")
    return (mu / C_M_S**2) * (1.0 / r_rx - 1.0 / r_tx)


def doppler_observable(
    range_rate_mps: float,
    carrier_hz: float,
    direction: LinkDirection = LinkDirection.DOWNLINK,
    turnaround_ratio: float = 1.0,
) -> DopplerObservable:
    """Build a :class:`DopplerObservable` with the convention attached.

    ``direction=LinkDirection.TWO_WAY`` uses :func:`two_way_doppler_hz` with
    ``carrier_hz`` read as the **uplink** carrier and records
    ``carrier_hz = turnaround_ratio * uplink`` as the reference frequency;
    the uplink and downlink directions use :func:`one_way_doppler_hz`.

    Convention: as :data:`DOPPLER_CONVENTION`, carried verbatim on the result.
    Units: [m/s], [Hz]; ``DopplerObservable.doppler_hz`` is in Hz.
    """
    direction = LinkDirection(direction)
    if direction is LinkDirection.TWO_WAY:
        shift = two_way_doppler_hz(range_rate_mps, carrier_hz, turnaround_ratio)
        reference = _check_carrier(carrier_hz) * float(turnaround_ratio)
        return DopplerObservable(
            doppler_hz=shift,
            carrier_hz=reference,
            range_rate_mps=float(range_rate_mps),
            direction=direction,
            ways=2,
        )
    return DopplerObservable(
        doppler_hz=one_way_doppler_hz(range_rate_mps, carrier_hz),
        carrier_hz=_check_carrier(carrier_hz),
        range_rate_mps=float(range_rate_mps),
        direction=direction,
        ways=1,
    )


__all__ = [
    "DOPPLER_CONVENTION",
    "DopplerObservable",
    "LinkDirection",
    "doppler_observable",
    "doppler_rate_hz_per_s",
    "gravitational_shift_fraction",
    "one_way_doppler_hz",
    "precompensation_offset_hz",
    "relativistic_correction_hz",
    "relativistic_fraction_second_order",
    "two_way_doppler_hz",
    "two_way_doppler_two_leg_hz",
]
