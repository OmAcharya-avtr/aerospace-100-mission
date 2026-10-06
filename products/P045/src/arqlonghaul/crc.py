"""Cyclic redundancy checks, implemented in-package.

Why in-package: an ARQ simulator needs a frame check sequence, because "the
receiver detected the error" is an assumption that has to be made concrete
somewhere.  ``crcmod`` and ``crc`` are the established libraries and are named
in the repository's alternatives table, but neither installs in this build
environment, so the check is implemented here with its parameters written down
and verified against two independent references:

1.  The catalogue *check* value, defined as the CRC of the ASCII byte string
    ``b"123456789"``.  The values used here were read from the
    ``crcmod`` 1.7 source distribution file ``python3/crcmod/predefined.py``
    on 2026-10-06; that file states the convention in a header comment.
2.  :func:`zlib.crc32` from the Python standard library, for CRC-32 only, over
    randomly generated payloads.  This is an independent implementation that is
    always available.

Both checks are run by ``validation/validate_crc.py``.

Parameterisation follows the usual Rocksoft-style model: width, polynomial
(without the implicit high term), initial shift-register value, input
reflection, output reflection, and final XOR.  ``crcmod`` uses a different but
equivalent convention for the initial value (it takes the value *after* the
final XOR is applied), so its table lists ``init = 0`` for CRC-32 where this
module lists ``init = 0xFFFFFFFF``; both describe the same function, and the
agreement on the check value is what demonstrates that.

This is error *detection*, not correction, and the simulator treats an
undetected error as a silent frame acceptance.  The residual undetected-error
probability of a CRC is not modelled here; see the README limitations.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CrcSpec",
    "CRC32",
    "CRC32C",
    "CRC16_CCITT_FALSE",
    "CRC16_ARC",
    "CRC8",
    "CATALOGUE",
    "crc",
    "append_fcs",
    "check_fcs",
]


def _reflect(value: int, width: int) -> int:
    """Reverse the low ``width`` bits of ``value``."""
    out = 0
    for _ in range(width):
        out = (out << 1) | (value & 1)
        value >>= 1
    return out


@dataclass(frozen=True)
class CrcSpec:
    """A CRC definition in the Rocksoft-style parameter model.

    Attributes:
        name: Catalogue name, e.g. ``"crc-32"``.
        width: Register width in bits.
        poly: Generator polynomial with the implicit x**width term removed.
        init: Initial shift-register value.
        refin: Reflect each input byte before processing.
        refout: Reflect the register before the final XOR.
        xorout: Value XORed into the reflected register to form the result.
        check: CRC of ``b"123456789"``, from the ``crcmod`` 1.7 catalogue.
    """

    name: str
    width: int
    poly: int
    init: int
    refin: bool
    refout: bool
    xorout: int
    check: int

    def __post_init__(self) -> None:
        if self.width not in (8, 16, 32, 64):
            raise ValueError(f"width must be 8, 16, 32 or 64, got {self.width}")
        mask = (1 << self.width) - 1
        for field in ("poly", "init", "xorout", "check"):
            value = getattr(self, field)
            if not 0 <= value <= mask:
                raise ValueError(
                    f"{field}={value:#x} does not fit in {self.width} bits"
                )

    @property
    def mask(self) -> int:
        """All-ones mask of the register width."""
        return (1 << self.width) - 1


CRC32 = CrcSpec("crc-32", 32, 0x04C11DB7, 0xFFFFFFFF, True, True, 0xFFFFFFFF, 0xCBF43926)
CRC32C = CrcSpec("crc-32c", 32, 0x1EDC6F41, 0xFFFFFFFF, True, True, 0xFFFFFFFF, 0xE3069283)
CRC16_CCITT_FALSE = CrcSpec(
    "crc-ccitt-false", 16, 0x1021, 0xFFFF, False, False, 0x0000, 0x29B1
)
CRC16_ARC = CrcSpec("crc-16", 16, 0x8005, 0x0000, True, True, 0x0000, 0xBB3D)
CRC8 = CrcSpec("crc-8", 8, 0x07, 0x00, False, False, 0x00, 0xF4)

CATALOGUE: dict[str, CrcSpec] = {
    spec.name: spec for spec in (CRC32, CRC32C, CRC16_CCITT_FALSE, CRC16_ARC, CRC8)
}
"""The five CRCs shipped here, keyed by catalogue name."""

_TABLES: dict[tuple[int, int, bool], tuple[int, ...]] = {}


def _table(spec: CrcSpec) -> tuple[int, ...]:
    """Byte-wise lookup table for ``spec``, memoised."""
    key = (spec.width, spec.poly, spec.refin)
    cached = _TABLES.get(key)
    if cached is not None:
        return cached
    mask = spec.mask
    top = 1 << (spec.width - 1)
    rows = []
    for byte in range(256):
        if spec.refin:
            reg = _reflect(byte, 8) << (spec.width - 8)
        else:
            reg = byte << (spec.width - 8)
        for _ in range(8):
            reg = ((reg << 1) ^ spec.poly) & mask if reg & top else (reg << 1) & mask
        rows.append(reg)
    table = tuple(rows)
    _TABLES[key] = table
    return table


def crc(data: bytes, spec: CrcSpec = CRC32) -> int:
    """CRC of ``data`` under ``spec``.

    Args:
        data: Message bytes.
        spec: CRC definition; defaults to :data:`CRC32`.

    Returns:
        The CRC value, an integer in ``[0, 2**spec.width)``.

    Raises:
        TypeError: if ``data`` is not bytes-like.
    """
    if isinstance(data, str):
        raise TypeError("crc() takes bytes, not str; encode the string first")
    try:
        view = memoryview(data).cast("B")
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError(f"data must be bytes-like, got {type(data).__name__}") from exc
    mask = spec.mask
    table = _table(spec)
    shift = spec.width - 8
    reg = spec.init
    if spec.refin:
        # Reflected algorithm: the register is held reflected throughout, which
        # is what makes the byte-wise table work with reflected input bytes.
        for byte in view:
            reg = (reg >> 8) ^ _reflect(table[(reg ^ byte) & 0xFF], spec.width)
            reg &= mask
    else:
        for byte in view:
            reg = ((reg << 8) ^ table[((reg >> shift) ^ byte) & 0xFF]) & mask
    if spec.refout != spec.refin:
        reg = _reflect(reg, spec.width)
    return (reg ^ spec.xorout) & mask


def crc_bitwise(data: bytes, spec: CrcSpec = CRC32) -> int:
    """CRC of ``data`` computed one bit at a time, as an independent check.

    This is the textbook shift-register definition.  It is slower than
    :func:`crc` by roughly a factor of eight and exists so that the table-driven
    implementation has something to be compared against; ``tests/test_crc.py``
    asserts the two agree.
    """
    if isinstance(data, str):
        raise TypeError("crc_bitwise() takes bytes, not str")
    mask = spec.mask
    top = 1 << (spec.width - 1)
    reg = spec.init
    for byte in memoryview(data).cast("B"):
        b = _reflect(byte, 8) if spec.refin else byte
        reg ^= b << (spec.width - 8) if spec.width >= 8 else b
        reg &= mask
        for _ in range(8):
            reg = ((reg << 1) ^ spec.poly) & mask if reg & top else (reg << 1) & mask
    if spec.refout:
        reg = _reflect(reg, spec.width)
    return (reg ^ spec.xorout) & mask


def append_fcs(payload: bytes, spec: CrcSpec = CRC32) -> bytes:
    """Return ``payload`` with its frame check sequence appended, big-endian."""
    value = crc(payload, spec)
    return bytes(payload) + value.to_bytes(spec.width // 8, "big")


def check_fcs(frame: bytes, spec: CrcSpec = CRC32) -> bool:
    """True if ``frame`` ends in a frame check sequence consistent with its body.

    Args:
        frame: Payload followed by ``spec.width // 8`` FCS octets.
        spec: CRC definition.

    Raises:
        ValueError: if ``frame`` is shorter than the FCS.
    """
    nbytes = spec.width // 8
    if len(frame) < nbytes + 1:
        raise ValueError(
            f"frame of {len(frame)} bytes is too short for a {nbytes}-byte FCS"
        )
    body, fcs = frame[:-nbytes], frame[-nbytes:]
    return crc(body, spec) == int.from_bytes(fcs, "big")
