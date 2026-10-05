"""Transfer frame geometry and marker-prefixed bit streams.

CCSDS 132.0-B (TM Space Data Link Protocol) defines a fixed-length transfer
frame: a 6-octet Transfer Frame Primary Header, an optional secondary
header, the Transfer Frame Data Field, an optional Operational Control
Field, and an optional 2-octet Frame Error Control Field. This package
models the *lengths*, the Frame Error Control Field and the attached sync
marker, because those are what determine frame-level link performance. It
does not parse or build header fields -- ``ccsdspy`` and ``spacepackets``
do that, and duplicating them is not the claim here.

With RS(255,223) at interleaving depth I the CCSDS codeblock carries
``223 * I`` octets, so the frame length is 1115 octets at I = 5 and 223
octets at I = 1. Those are the two geometries used throughout the
validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .asm import ASM_32_BITS
from .crc import FECF_BITS, crc16, crc16_batch

__all__ = ["FrameGeometry", "CCSDS_PRIMARY_HEADER_OCTETS", "build_stream"]

CCSDS_PRIMARY_HEADER_OCTETS = 6
"""Transfer Frame Primary Header length, CCSDS 132.0-B."""


@dataclass(frozen=True)
class FrameGeometry:
    """Lengths of one transfer frame and its marker.

    Attributes
    ----------
    data_octets : int
        Total transfer frame length in octets, header included, Frame Error
        Control Field excluded. Positive.
    fecf : bool
        Append the 2-octet Frame Error Control Field.
    asm_bits : int
        Attached sync marker length in bits, 0 to omit the marker.
    """

    data_octets: int = 1115
    fecf: bool = True
    asm_bits: int = 32

    def __post_init__(self) -> None:
        if self.data_octets <= 0:
            raise ValueError(f"data_octets must be positive, got {self.data_octets}")
        if self.asm_bits < 0:
            raise ValueError(f"asm_bits must be non-negative, got {self.asm_bits}")

    @property
    def frame_octets(self) -> int:
        """Frame length on the channel in octets, marker excluded."""
        return self.data_octets + (FECF_BITS // 8 if self.fecf else 0)

    @property
    def frame_bits(self) -> int:
        """Frame length in bits, marker excluded. This is the ``n`` of 1-(1-p)^n."""
        return 8 * self.frame_octets

    @property
    def period_bits(self) -> int:
        """Marker plus frame, i.e. the synchroniser's frame period in bits."""
        return self.asm_bits + self.frame_bits

    def random_frames(self, n_frames: int, rng: np.random.Generator) -> np.ndarray:
        """``(n_frames, frame_octets)`` uint8 array of frames with valid FECF.

        The data field is random octets; the last two octets hold the CRC-16
        of everything before them when ``fecf`` is set.
        """
        if n_frames <= 0:
            raise ValueError(f"n_frames must be positive, got {n_frames}")
        body = rng.integers(0, 256, (int(n_frames), self.data_octets), dtype=np.uint8)
        if not self.fecf:
            return body
        crc = crc16_batch(body)
        tail = np.stack([(crc >> 8).astype(np.uint8), (crc & 0xFF).astype(np.uint8)], axis=1)
        return np.concatenate([body, tail], axis=1)

    def fecf_ok(self, frames: np.ndarray) -> np.ndarray:
        """Boolean per row: does the Frame Error Control Field check out?

        The usual receiver-side test is that the CRC recomputed over the
        whole frame *including* the two check octets equals zero; this
        implementation compares the recomputed CRC of the body against the
        transmitted field, which is equivalent and keeps the two octets
        explicit.
        """
        if not self.fecf:
            raise ValueError("geometry has no Frame Error Control Field to check")
        arr = np.atleast_2d(np.asarray(frames, dtype=np.uint8))
        if arr.shape[1] != self.frame_octets:
            raise ValueError(
                f"frames must have {self.frame_octets} octets per row, got {arr.shape[1]}"
            )
        body, tail = arr[:, : self.data_octets], arr[:, self.data_octets :]
        got = (tail[:, 0].astype(np.uint16) << 8) | tail[:, 1].astype(np.uint16)
        return crc16_batch(body) == got

    def check_single(self, frame: np.ndarray) -> bool:
        """Scalar-path Frame Error Control Field check, for cross-testing."""
        arr = np.asarray(frame, dtype=np.uint8).ravel()
        body, tail = arr[: self.data_octets], arr[self.data_octets :]
        return crc16(body) == ((int(tail[0]) << 8) | int(tail[1]))


def build_stream(
    geometry: FrameGeometry,
    n_frames: int,
    rng: np.random.Generator,
    pattern: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Marker-prefixed bit stream and the true marker offsets.

    Returns
    -------
    (stream, offsets)
        ``stream`` is a 1-D uint8 array of 0/1 bits of length
        ``n_frames * period_bits``; ``offsets`` holds the true marker start
        offsets, ``k * period_bits``.
    """
    pat = ASM_32_BITS if pattern is None else np.asarray(pattern, dtype=np.uint8).ravel()
    if geometry.asm_bits != pat.size:
        raise ValueError(
            f"geometry.asm_bits={geometry.asm_bits} does not match the {pat.size}-bit pattern"
        )
    frames = geometry.random_frames(n_frames, rng)
    bits = np.unpackbits(frames, axis=1)
    stream = np.concatenate([np.concatenate([pat, row]) for row in bits])
    offsets = np.arange(n_frames, dtype=np.int64) * geometry.period_bits
    return stream, offsets
