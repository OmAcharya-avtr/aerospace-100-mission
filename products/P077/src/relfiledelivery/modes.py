"""The three transfer modes this harness drives, and what each one means here.

The names are taken from CCSDS 727.0-B-5, *CFDP -- CCSDS File Delivery
Protocol* (Consultative Committee for Space Data Systems, 2020), which defines
an unacknowledged (class 1) transmission mode and an acknowledged (class 2)
mode whose NAK timing is an implementation choice. **Nothing in this package
implements CFDP.** The modes below are the three retransmission disciplines a
measurement harness has to be able to distinguish; they are declared
approximations of the discipline, not of the wire format, and no PDU in this
package is CFDP-conformant. For the protocol itself use ``cfdp-py`` and
``spacepackets``.

Semantics implemented in :mod:`relfiledelivery.simulate`:

``UNACKNOWLEDGED``
    One forward pass over every protocol data unit. Nothing is retransmitted,
    so the file is delivered only if no unit is lost. There is no return
    traffic at all.

``DEFERRED_NAK``
    The sender transmits every outstanding unit back to back, then an
    end-of-file unit. The receiver replies with one negative acknowledgement
    naming everything still missing. The sender retransmits that set. Repeat
    until nothing is missing. Retransmission is therefore organised into
    *rounds*, and one turnaround is paid per round boundary.

``IMMEDIATE_NAK``
    As above, except the receiver also reports a gap as soon as a later unit
    arrives and reveals it, so a retransmission request is in flight while the
    forward stream is still draining. The end-of-file round is still needed,
    because a lost trailing unit has no successor to reveal it.
"""

from __future__ import annotations

from enum import StrEnum


class TransferMode(StrEnum):
    """Retransmission discipline driven against the link model."""

    UNACKNOWLEDGED = "unacknowledged"
    DEFERRED_NAK = "deferred_nak"
    IMMEDIATE_NAK = "immediate_nak"

    @property
    def uses_return_link(self) -> bool:
        """True when the mode sends anything on the return link."""
        return self is not TransferMode.UNACKNOWLEDGED

    @property
    def retransmits(self) -> bool:
        """True when a lost protocol data unit is sent again."""
        return self is not TransferMode.UNACKNOWLEDGED


def parse_mode(text: str) -> TransferMode:
    """Parse a mode name, raising :class:`ValueError` with the valid set.

    Accepts the enum values and the hyphenated spellings a command line is
    likely to carry (``deferred-nak``).
    """
    if isinstance(text, TransferMode):
        return text
    key = str(text).strip().lower().replace("-", "_")
    for mode in TransferMode:
        if key == mode.value:
            return mode
    valid = ", ".join(mode.value for mode in TransferMode)
    raise ValueError(f"unknown transfer mode {text!r}; expected one of {valid}")


ALL_MODES: tuple[TransferMode, ...] = tuple(TransferMode)
