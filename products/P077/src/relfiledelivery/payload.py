"""The file being delivered, its segmentation into units, and the hash check.

The completion invariant of this package is that the bytes the receiver holds
at the end of a transfer are *the same bytes* the sender started with, verified
by SHA-256 and not by a byte count. It is checked inside
:func:`relfiledelivery.simulate.simulate_transfer` on every run that completes,
and it is a hard failure, not a summary statistic. A harness that reports a
mean delivery time without ever reconstructing the file is measuring the wrong
thing, and would not notice an off-by-one in its own segment accounting.

SHA-256 is FIPS 180-4 (National Institute of Standards and Technology, 2015),
used here only as a collision-resistant content identity, not for any security
property.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


def digest(data: bytes) -> str:
    """SHA-256 of ``data`` as a lowercase hexadecimal string."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class FileTransferSpec:
    """A file and the unit size it is cut into.

    Parameters
    ----------
    file_bytes:
        Total source length in bytes. Must be positive.
    pdu_payload_bytes:
        Payload carried by one protocol data unit, bytes. The final unit
        carries the remainder and is shorter unless the file divides evenly.
    pdu_header_bytes:
        Per-unit header and trailer overhead on the wire, bytes. Declared, not
        taken from any standard: a CFDP file-data PDU header is variable and
        the value here is a constant stand-in whose only role is to make
        overhead ratios comparable across modes.
    """

    file_bytes: int
    pdu_payload_bytes: int
    pdu_header_bytes: int = 12

    def __post_init__(self) -> None:
        if not isinstance(self.file_bytes, (int, np.integer)) or self.file_bytes <= 0:
            raise ValueError(f"file_bytes must be a positive integer; got {self.file_bytes!r}")
        if not isinstance(self.pdu_payload_bytes, (int, np.integer)) or self.pdu_payload_bytes <= 0:
            raise ValueError(
                f"pdu_payload_bytes must be a positive integer; got {self.pdu_payload_bytes!r}"
            )
        if not isinstance(self.pdu_header_bytes, (int, np.integer)) or self.pdu_header_bytes < 0:
            raise ValueError(
                f"pdu_header_bytes must be a non-negative integer; got {self.pdu_header_bytes!r}"
            )
        object.__setattr__(self, "file_bytes", int(self.file_bytes))
        object.__setattr__(self, "pdu_payload_bytes", int(self.pdu_payload_bytes))
        object.__setattr__(self, "pdu_header_bytes", int(self.pdu_header_bytes))

    @property
    def pdu_count(self) -> int:
        """Number of units the file is cut into, dimensionless."""
        return math.ceil(self.file_bytes / self.pdu_payload_bytes)

    @property
    def last_payload_bytes(self) -> int:
        """Payload of the final unit, bytes. Equals ``pdu_payload_bytes`` when even."""
        remainder = self.file_bytes % self.pdu_payload_bytes
        return remainder if remainder else self.pdu_payload_bytes

    def payload_bytes_of(self, sequence: int) -> int:
        """Payload of unit ``sequence`` (zero-based), bytes."""
        if not 0 <= sequence < self.pdu_count:
            raise ValueError(
                f"sequence must lie in [0, {self.pdu_count}); got {sequence}"
            )
        return (
            self.last_payload_bytes
            if sequence == self.pdu_count - 1
            else self.pdu_payload_bytes
        )

    def wire_bytes_of(self, sequence: int) -> int:
        """Total on-wire size of unit ``sequence``, bytes, header included."""
        return self.payload_bytes_of(sequence) + self.pdu_header_bytes

    @property
    def full_wire_bytes(self) -> int:
        """On-wire size of a full-payload unit, bytes."""
        return self.pdu_payload_bytes + self.pdu_header_bytes

    @property
    def total_wire_bytes(self) -> int:
        """Sum of on-wire sizes over one clean pass of the whole file, bytes."""
        return self.file_bytes + self.pdu_header_bytes * self.pdu_count

    @property
    def minimum_overhead_ratio(self) -> float:
        """``total_wire_bytes / file_bytes`` with no loss, dimensionless and >= 1."""
        return self.total_wire_bytes / self.file_bytes

    def slices(self) -> list[tuple[int, int]]:
        """``(start, stop)`` byte offsets of every unit, in sequence order."""
        step = self.pdu_payload_bytes
        return [
            (index * step, min((index + 1) * step, self.file_bytes))
            for index in range(self.pdu_count)
        ]

    def describe(self) -> list[str]:
        """Human-readable lines for a report."""
        return [
            f"file size               : {self.file_bytes} bytes",
            f"PDU payload             : {self.pdu_payload_bytes} bytes",
            f"PDU header              : {self.pdu_header_bytes} bytes",
            f"PDU count               : {self.pdu_count}",
            f"last PDU payload        : {self.last_payload_bytes} bytes",
            f"clean-pass wire bytes   : {self.total_wire_bytes} bytes "
            f"(overhead ratio {self.minimum_overhead_ratio:.6f})",
        ]


def source_payload(spec: FileTransferSpec, seed: int) -> bytes:
    """Deterministic pseudorandom source file of ``spec.file_bytes`` bytes.

    Incompressible by construction, so no part of this package can accidentally
    benefit from structure in the payload. Identical seeds give identical bytes,
    which is asserted by test.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=spec.file_bytes, dtype=np.uint8).tobytes()


def reassemble(spec: FileTransferSpec, received: dict[int, bytes]) -> bytes:
    """Concatenate received units in sequence order.

    Raises :class:`KeyError` when a unit is missing, because a partial
    reassembly is not a file and must not be silently hashed.
    """
    pieces = []
    for sequence in range(spec.pdu_count):
        piece = received[sequence]
        expected = spec.payload_bytes_of(sequence)
        if len(piece) != expected:
            raise ValueError(
                f"unit {sequence} carries {len(piece)} bytes but the spec says {expected}"
            )
        pieces.append(piece)
    return b"".join(pieces)


class CompletionInvariantError(AssertionError):
    """Raised when a completed transfer does not reproduce the source bytes.

    Deriving from :class:`AssertionError` is deliberate: this is a violated
    invariant of the harness itself, not a condition a caller can handle by
    retrying. Every completed run checks it.
    """
