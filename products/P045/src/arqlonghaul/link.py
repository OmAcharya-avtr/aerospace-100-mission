"""Link geometry and the bandwidth-delay product, in explicit units.

Everything downstream of this module measures time in *frame slots*: one slot is
the time to clock one frame of ``frame_bits`` bits onto the channel at
``rate_bps``.  The single number that controls every ARQ throughput expression
is

    N = (T_f + RTT) / T_f = 1 + RTT / T_f                                   (1)

the number of frame slots in one stop-and-wait cycle.  Equation (1) is the
standard normalisation used in the ARQ throughput literature (S. Lin and
D. J. Costello, Jr., *Error Control Coding*, 2nd ed., Prentice Hall, 2004,
chapter on ARQ schemes; A. S. Tanenbaum, *Computer Networks*, sliding-window
derivations).  Nothing here is specific to a space link; the space link is only
the regime where N is large, which is where the textbook window advice stops
being adequate.

Units are stated on every field and every return value.  No value in this
module is empirical: it is all arithmetic on the four inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "LinkParams",
    "PRESETS",
    "preset",
    "SPEED_OF_LIGHT_M_S",
]

SPEED_OF_LIGHT_M_S: float = 299_792_458.0
"""Speed of light in vacuum, m/s (exact, SI definition of the metre)."""


@dataclass(frozen=True)
class LinkParams:
    """A one-way data link summarised by the four numbers ARQ throughput needs.

    Attributes:
        rate_bps: Channel bit rate on the forward link, bits per second.
        rtt_s: Round-trip time, seconds.  This is the total feedback latency:
            two-way propagation plus any receiver and transmitter processing and
            any acknowledgement transmission time.  It is an input, not a
            derived quantity, because processing delays dominate propagation on
            some links and not others.
        frame_bits: Total transmitted frame length including header and frame
            check sequence, bits.
        payload_bits: Information bits carried per frame, bits.  Must not exceed
            ``frame_bits``.  Defaults to ``frame_bits`` (no overhead accounted).
        name: Free-text label, for plots and tables only.
    """

    rate_bps: float
    rtt_s: float
    frame_bits: int
    payload_bits: int | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if not self.rate_bps > 0:
            raise ValueError(f"rate_bps must be > 0, got {self.rate_bps}")
        if self.rtt_s < 0:
            raise ValueError(f"rtt_s must be >= 0, got {self.rtt_s}")
        if self.frame_bits <= 0:
            raise ValueError(f"frame_bits must be > 0, got {self.frame_bits}")
        pay = self.frame_bits if self.payload_bits is None else self.payload_bits
        if pay <= 0:
            raise ValueError(f"payload_bits must be > 0, got {self.payload_bits}")
        if pay > self.frame_bits:
            raise ValueError(
                f"payload_bits ({pay}) cannot exceed frame_bits ({self.frame_bits})"
            )

    @property
    def payload(self) -> int:
        """Information bits per frame, bits."""
        return self.frame_bits if self.payload_bits is None else self.payload_bits

    @property
    def frame_time_s(self) -> float:
        """Time to clock one frame onto the channel, seconds: ``frame_bits/rate_bps``."""
        return self.frame_bits / self.rate_bps

    @property
    def code_overhead(self) -> float:
        """Fraction of each frame that is not payload, dimensionless in [0, 1)."""
        return 1.0 - self.payload / self.frame_bits

    @property
    def slots_per_cycle(self) -> float:
        """N of equation (1): stop-and-wait cycle length in frame slots, dimensionless.

        Not rounded.  Use :meth:`n_slots` for the integer the slotted simulator
        needs.
        """
        return 1.0 + self.rtt_s / self.frame_time_s

    def n_slots(self) -> int:
        """N of equation (1) rounded to the nearest integer slot count, >= 1."""
        return max(1, int(round(self.slots_per_cycle)))

    @property
    def bdp_bits(self) -> float:
        """Bandwidth-delay product, bits: ``rate_bps * rtt_s``.

        The number of bits that are in flight on the link, unacknowledged, when
        the sender transmits continuously.
        """
        return self.rate_bps * self.rtt_s

    @property
    def bdp_bytes(self) -> float:
        """Bandwidth-delay product, bytes."""
        return self.bdp_bits / 8.0

    @property
    def bdp_frames(self) -> float:
        """Bandwidth-delay product expressed in frames, dimensionless.

        Equals ``slots_per_cycle - 1``.  A window of ``ceil(bdp_frames) + 1``
        frames is the smallest that lets the sender transmit continuously, which
        is why window-sizing advice is usually quoted as "one BDP plus one
        frame".
        """
        return self.bdp_bits / self.frame_bits

    @property
    def min_continuous_window(self) -> int:
        """Smallest window, in frames, that permits continuous transmission.

        ``N`` rounded up: a sender with this many sequence numbers outstanding
        never stalls waiting for an acknowledgement on an error-free link.
        """
        import math

        return max(1, math.ceil(self.slots_per_cycle))

    def window_bytes(self, window_frames: int) -> float:
        """Buffer implied by a window of ``window_frames`` frames, bytes."""
        if window_frames <= 0:
            raise ValueError(f"window_frames must be > 0, got {window_frames}")
        return window_frames * self.frame_bits / 8.0

    def describe(self) -> dict[str, float | int | str]:
        """Return the derived quantities as a flat dict, for printing and tests."""
        return {
            "name": self.name,
            "rate_bps": self.rate_bps,
            "rtt_s": self.rtt_s,
            "frame_bits": self.frame_bits,
            "payload_bits": self.payload,
            "frame_time_s": self.frame_time_s,
            "slots_per_cycle_N": self.slots_per_cycle,
            "N_slots_int": self.n_slots(),
            "bdp_bits": self.bdp_bits,
            "bdp_bytes": self.bdp_bytes,
            "bdp_frames": self.bdp_frames,
            "min_continuous_window_frames": self.min_continuous_window,
        }


def _rtt_from_range(range_m: float, processing_s: float = 0.0) -> float:
    """Round-trip time, seconds, for a one-way range in metres plus processing."""
    return 2.0 * range_m / SPEED_OF_LIGHT_M_S + processing_s


PRESETS: dict[str, LinkParams] = {
    # Ranges are nominal round figures used to put N in the right decade; they
    # are illustrative geometry, not ephemeris values, and are labelled as such.
    "geo": LinkParams(
        rate_bps=2.0e6,
        rtt_s=_rtt_from_range(35_786e3, processing_s=20e-3),
        frame_bits=8920,  # 1115 octets, the CCSDS AOS/TM transfer frame order of size
        payload_bits=8792,
        name="GEO relay, 2 Mbit/s, 1115-octet frame",
    ),
    "leo": LinkParams(
        rate_bps=50.0e6,
        rtt_s=_rtt_from_range(1_000e3, processing_s=5e-3),
        frame_bits=8920,
        payload_bits=8792,
        name="LEO downlink, 50 Mbit/s, 1115-octet frame",
    ),
    "lunar": LinkParams(
        rate_bps=1.0e6,
        rtt_s=_rtt_from_range(384_400e3, processing_s=50e-3),
        frame_bits=8920,
        payload_bits=8792,
        name="Lunar, 1 Mbit/s, 1115-octet frame",
    ),
    "mars_near": LinkParams(
        rate_bps=256.0e3,
        rtt_s=_rtt_from_range(0.55e9 * 1000.0, processing_s=0.0),
        frame_bits=8920,
        payload_bits=8792,
        name="Mars near conjunction-free range (0.55e9 km), 256 kbit/s",
    ),
}
"""Illustrative link presets.  Geometry is nominal, rounded, and labelled.

The round-trip times are two-way light time for the stated nominal range plus a
stated processing allowance.  They exist so that examples and tests have
realistic values of N; they are not a substitute for a mission link budget.
"""


def preset(name: str) -> LinkParams:
    """Return a named preset from :data:`PRESETS`.

    Raises:
        ValueError: if ``name`` is not a known preset.
    """
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown preset {name!r}; known presets are {sorted(PRESETS)}"
        ) from None
