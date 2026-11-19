"""The declared handover sequence: the four-phase timeline every model works from.

A ground-station handover on an optical downlink is declared here as an ordered
sequence of phases, because every one of this package's three baselines and its
learned component is a different way of answering the same question about the
same timeline. Writing the timeline down once, with units, is what makes the
three answers comparable.

THE TIMELINE
------------

::

    t = 0            link with the outgoing station breaks
    [0, T_acq)       ACQUISITION   egress 0.  Beacon detection, spatial
                                   acquisition, coarse then fine pointing lock
                                   at the incoming station.
    [T_acq, T_o)     RESTART       egress 0.  The optical link carries photons
                                   but no payload: the link-layer and transport
                                   session must be re-established before user
                                   data is accepted again.  T_o = T_acq + T_rst.
    [T_o, T_o+T_ramp) DRAIN(ramp)  egress ramps linearly 0 -> R_d.  The link is
                                   up and the backlog begins to drain, but
                                   while the instantaneous egress is below the
                                   source rate the buffer is still filling.
    [T_o+T_ramp, H)   DRAIN(full)  egress R_d.  Backlog drains if R_d > R_s.

Throughout, an on-board buffer of capacity ``B`` bits accepts data from a source
of rate ``R_s`` bit/s and is served at the egress rate above. The buffer is
tail-drop: data arriving at a full buffer is lost and is never recovered.

WHY THE THIRD PHASE IS THE POINT OF THE PACKAGE
-----------------------------------------------

A link budget gives an acquisition time, so the natural assumption is that a
handover costs the data that cannot be stored during acquisition. Two further
mechanisms are invisible to that assumption:

* the protocol restart extends the zero-egress interval beyond ``T_acq``;
* loss continues AFTER the link is fully up, for as long as the instantaneous
  egress is below the source rate, which the ramp guarantees for
  ``T_ramp * R_s / R_d`` seconds.

The third of these is the one that decides whether an operator should buy
faster acquisition or more buffer, and it is the one that does not appear in a
link budget at all.

UNITS
-----

All times are seconds, all rates bit/s, all data volumes bits. No module in this
package uses any other unit internally; reporting helpers convert to Gbit and
say so.

SOURCES FOR THE PHASE STRUCTURE
-------------------------------

* Acquisition, pointing and tracking as a distinct phase with its own duration
  distribution: Hemmati, H. (editor), *Deep Space Optical Communications*, JPL
  Deep-Space Communications and Navigation Series, Wiley, 2006, the chapter on
  acquisition, tracking and pointing; and Kaushal, H. and Kaddoum, G., "Optical
  Communication in Space: Challenges and Mitigation Techniques", *IEEE
  Communications Surveys and Tutorials*, volume 19, number 1, pages 57-96,
  2017.
* A separate physical layer and coding/synchronisation layer for an optical
  space link, whose synchronisation must be re-acquired before payload frames
  are accepted: CCSDS 141.0-B-1, *Optical Communications Physical Layer*,
  Recommended Standard, Blue Book, 2019, and CCSDS 142.0-B-1, *Optical
  Communications Coding and Synchronization*, Recommended Standard, Blue Book,
  2019.
* A transport session that must be re-established, with its own latency, after a
  link outage: CCSDS 727.0-B-5, *CCSDS File Delivery Protocol (CFDP)*,
  Recommended Standard, Blue Book, 2020.
* A linear ramp of the delivered rate after a restart, rather than a step:
  Allman, M., Paxson, V. and Blanton, E., RFC 5681, *TCP Congestion Control*,
  IETF, 2009, section 3.1, for the slow-start structure the ramp abstracts. The
  ramp here is the LINEAR envelope of such a ramp-up and is a declared
  modelling choice, not a reproduction of any particular congestion-control
  trajectory.
* The finite-buffer fluid formulation: Anick, D., Mitra, D. and Sondhi, M. M.,
  "Stochastic theory of a data-handling system with multiple sources", *Bell
  System Technical Journal*, volume 61, number 8, pages 1871-1894, 1982.

VALIDITY RANGE OF THE DECLARATION ITSELF
----------------------------------------

The sequence assumes the three intervals are disjoint and ordered, that the
source rate is constant in the mean over the handover window, and that the
incoming station's steady egress rate is a single number. It does not model: a
make-before-break handover in which both links are up at once (the sequence
degenerates to ``T_acq = 0``, which is accepted and gives zero acquisition
loss); a handover that fails and returns to the outgoing station; adaptive
coding and modulation that changes ``R_d`` during the pass; or any loss
mechanism on the ground segment after delivery.

None of the numeric parameter values used anywhere in this repository are
measurements from the sources above. They are declared parameters of a
synthetic generator, chosen to span a plausible range, and
``data/generate_corpus.py`` is where they are declared.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

#: Mechanism names, in the fixed order every array in this package uses.
MECHANISMS: tuple[str, str, str] = ("acquisition", "restart", "drain")


class Phase(IntEnum):
    """Index of a handover phase, used as the column order of every decomposition.

    The integer values are the mechanism order in :data:`MECHANISMS` and are
    relied on by :mod:`handoverloss.accounting`, so they are not reordered.
    """

    ACQUISITION = 0
    RESTART = 1
    DRAIN = 2


@dataclass(frozen=True)
class HandoverSequence:
    """One declared handover, in SI units, with its validation at construction.

    Parameters
    ----------
    source_rate_bps:
        Mean source (instrument) rate feeding the buffer, bit/s, strictly
        positive. Held constant in the mean over the handover window.
    downlink_rate_bps:
        Steady egress rate of the INCOMING station once the ramp is complete,
        bit/s, strictly positive.
    buffer_capacity_bits:
        On-board buffer capacity allocated to this stream, bits, strictly
        positive. Tail-drop: arrivals at a full buffer are lost.
    initial_occupancy_bits:
        Buffer occupancy at the instant the outgoing link breaks, bits, in
        ``[0, buffer_capacity_bits]``.
    acquisition_s:
        Duration of the acquisition phase, seconds, non-negative. Zero is
        accepted and means make-before-break.
    restart_s:
        Duration of the protocol-restart phase, seconds, non-negative.
    ramp_s:
        Duration of the linear egress ramp after the outage, seconds,
        non-negative. Zero means the egress steps straight to
        ``downlink_rate_bps``.
    horizon_s:
        Length of the accounting window measured from the break, seconds. Must
        be at least ``acquisition_s + restart_s + ramp_s``, because a window
        that ends inside the ramp would attribute to the drain phase only part
        of the loss that phase causes.

    Raises
    ------
    ValueError
        On any violated bound above, naming the field and the offending value.
    TypeError
        If a field is not a real number.
    """

    source_rate_bps: float
    downlink_rate_bps: float
    buffer_capacity_bits: float
    initial_occupancy_bits: float
    acquisition_s: float
    restart_s: float
    ramp_s: float
    horizon_s: float

    def __post_init__(self) -> None:
        for name in (
            "source_rate_bps",
            "downlink_rate_bps",
            "buffer_capacity_bits",
            "initial_occupancy_bits",
            "acquisition_s",
            "restart_s",
            "ramp_s",
            "horizon_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"HandoverSequence.{name} must be a real number, got "
                    f"{type(value).__name__}"
                )
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(
                    f"HandoverSequence.{name} must be finite, got {value!r}"
                )
        for name in ("source_rate_bps", "downlink_rate_bps", "buffer_capacity_bits"):
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"HandoverSequence.{name} must be strictly positive, got "
                    f"{getattr(self, name)!r}. A zero source rate has no handover to "
                    "account for and a zero capacity or egress rate is not a buffer."
                )
        for name in ("acquisition_s", "restart_s", "ramp_s", "initial_occupancy_bits"):
            if getattr(self, name) < 0.0:
                raise ValueError(
                    f"HandoverSequence.{name} must be non-negative, got "
                    f"{getattr(self, name)!r}"
                )
        if self.initial_occupancy_bits > self.buffer_capacity_bits:
            raise ValueError(
                "HandoverSequence.initial_occupancy_bits "
                f"({self.initial_occupancy_bits!r}) exceeds buffer_capacity_bits "
                f"({self.buffer_capacity_bits!r}); the buffer cannot start overfull"
            )
        minimum_horizon = self.acquisition_s + self.restart_s + self.ramp_s
        if self.horizon_s < minimum_horizon:
            raise ValueError(
                f"HandoverSequence.horizon_s ({self.horizon_s!r}) is shorter than "
                f"acquisition_s + restart_s + ramp_s ({minimum_horizon!r}). A window "
                "ending inside the ramp attributes only part of the drain-phase loss "
                "and the decomposition would not be comparable across events."
            )

    @property
    def outage_s(self) -> float:
        """Zero-egress interval ``T_o = T_acq + T_rst``, seconds."""
        return self.acquisition_s + self.restart_s

    @property
    def headroom_bits(self) -> float:
        """Free buffer space at the break, ``B - q0``, bits."""
        return self.buffer_capacity_bits - self.initial_occupancy_bits

    @property
    def headroom_s(self) -> float:
        """Seconds of source output the free buffer space can absorb."""
        return self.headroom_bits / self.source_rate_bps

    @property
    def utilisation(self) -> float:
        """``rho = R_s / R_d``. At or above 1 the backlog never drains."""
        return self.source_rate_bps / self.downlink_rate_bps

    def phase_of(self, t_s: float) -> Phase:
        """Phase containing time ``t_s`` seconds after the break.

        Times at or beyond ``outage_s`` are :attr:`Phase.DRAIN`, including times
        after the ramp has finished: the drain phase is defined as everything
        after the link comes up, so that the three mechanisms partition the
        accounting window exactly.
        """
        if t_s < 0.0:
            raise ValueError(f"t_s must be non-negative, got {t_s!r}")
        if t_s < self.acquisition_s:
            return Phase.ACQUISITION
        if t_s < self.outage_s:
            return Phase.RESTART
        return Phase.DRAIN

    def egress_rate_bps(self, t_s: float) -> float:
        """Instantaneous egress rate at ``t_s``, bit/s, from the declared ramp.

        Zero during acquisition and restart; ``R_d * (t - T_o) / T_ramp`` during
        the ramp; ``R_d`` thereafter. With ``ramp_s == 0`` the rate steps.
        """
        if t_s < 0.0:
            raise ValueError(f"t_s must be non-negative, got {t_s!r}")
        if t_s < self.outage_s:
            return 0.0
        if self.ramp_s <= 0.0:
            return self.downlink_rate_bps
        elapsed = t_s - self.outage_s
        if elapsed >= self.ramp_s:
            return self.downlink_rate_bps
        return self.downlink_rate_bps * (elapsed / self.ramp_s)
