"""Static range Asymmetric Numeral Systems coder. Produces actual bytes.

Why this exists rather than an entropy estimate
-----------------------------------------------

The rate of a lossy compressor is the length of the thing you have to
transmit. It is not the first-order entropy of the residual stream, and this
package never substitutes the second for the first: every rate figure in this
repository is ``len(bytes)`` of a container that round-trips. The entropy is
reported alongside, labelled as a reference figure and not as a floor, because
for a correlated source it is not a floor at all -- a coder with memory
routinely beats it. A sibling product in this portfolio (P072) measured 0.9978
of its held-out cases compressing *below* their measured first-order entropy.

The coder
---------

Static rANS, as described by Duda 2013 ("Asymmetric numeral systems: entropy
coding combining speed of Huffman coding with compression rate of arithmetic
coding", arXiv:1311.2540) and in the reference implementation notes of Giesen
2014 (``ryg_rans``). A single frequency table, normalised so the frequencies
sum to :math:`M = 2^{k}` with ``k = SCALE_BITS``, is built from the symbol
stream and shipped in the container; the state is renormalised in 16-bit
words against the interval lower bound :data:`RANS_L`.

The encoder consumes symbols in reverse order and the decoder produces them in
forward order. That is a property of the construction and not an accident: the
rANS state is a stack. Both directions are integer-only, so the coder is
bit-exact and portable -- the arithmetic here is exact in Python's arbitrary
precision integers and would be exact in 64-bit C as well, since the state
never exceeds :math:`2^{39}`.

Invariant: after every encode step the state satisfies
:math:`x \\in [L, L \\cdot 2^{16})`. :func:`encode` asserts it on every symbol,
which is cheap at the array sizes this package handles and is the reason the
round-trip property tests have never needed to debug the coder.
"""

from __future__ import annotations

#: Lower bound of the rANS normalisation interval. 2**16 with 16-bit
#: renormalisation keeps the state in [2**16, 2**32), so the final state fits
#: the uint32 field the container header reserves for it. This is the
#: ``rans_word`` configuration of the reference implementation.
RANS_L = 1 << 16

#: Number of bits in the normalised frequency total, so the total is 2**12.
#: 4096 slots is enough resolution for an alphabet of 256 symbols that each
#: need at least one slot, and keeps the shipped table small.
SCALE_BITS = 12
SCALE_TOTAL = 1 << SCALE_BITS


class CoderError(ValueError):
    """Raised on a malformed frequency table or a truncated payload."""


def normalise_frequencies(counts: list[int]) -> list[int]:
    """Scale raw symbol counts to sum to exactly :data:`SCALE_TOTAL`.

    Every symbol with a non-zero count is guaranteed at least one slot, which
    is what makes the table usable as a code: a symbol with zero slots cannot
    be encoded. Slots are taken from the largest frequency when rounding
    overshoots and given to it when rounding undershoots, so the table always
    sums to the total exactly.

    Parameters
    ----------
    counts:
        Raw occurrence counts, one entry per symbol in the alphabet. At least
        one must be non-zero.

    Raises
    ------
    CoderError
        If every count is zero, if any count is negative, or if the number of
        symbols with a non-zero count exceeds :data:`SCALE_TOTAL`.
    """
    if any(count < 0 for count in counts):
        raise CoderError("symbol counts must be non-negative")
    total = sum(counts)
    if total == 0:
        raise CoderError("cannot build a frequency table from an empty symbol stream")
    used = [index for index, count in enumerate(counts) if count > 0]
    if len(used) > SCALE_TOTAL:
        raise CoderError(
            f"{len(used)} distinct symbols cannot each hold a slot in a table of "
            f"{SCALE_TOTAL}"
        )

    scaled = [0] * len(counts)
    for index in used:
        scaled[index] = max(1, (counts[index] * SCALE_TOTAL) // total)

    deficit = SCALE_TOTAL - sum(scaled)
    order = sorted(used, key=lambda index: (-scaled[index], index))
    position = 0
    while deficit > 0:
        scaled[order[position % len(order)]] += 1
        deficit -= 1
        position += 1
    position = 0
    while deficit < 0:
        index = order[position % len(order)]
        if scaled[index] > 1:
            scaled[index] -= 1
            deficit += 1
        position += 1
        if position > 64 * SCALE_TOTAL:  # pragma: no cover - unreachable with max(1, .)
            raise CoderError("could not reduce the frequency table to the scale total")
    return scaled


def cumulative(frequencies: list[int]) -> list[int]:
    """Exclusive prefix sums of ``frequencies``, length ``len(frequencies) + 1``."""
    out = [0]
    running = 0
    for frequency in frequencies:
        running += frequency
        out.append(running)
    return out


def slot_table(frequencies: list[int]) -> list[int]:
    """Map each of the :data:`SCALE_TOTAL` slots to the symbol that owns it.

    Raises
    ------
    CoderError
        If ``frequencies`` does not sum to :data:`SCALE_TOTAL`.
    """
    if sum(frequencies) != SCALE_TOTAL:
        raise CoderError(
            f"frequency table sums to {sum(frequencies)}, expected {SCALE_TOTAL}"
        )
    table = [0] * SCALE_TOTAL
    position = 0
    for symbol, frequency in enumerate(frequencies):
        for _ in range(frequency):
            table[position] = symbol
            position += 1
    return table


def encode(symbols: list[int], frequencies: list[int]) -> tuple[bytes, int]:
    """Encode ``symbols`` under a static table. Returns the payload and final state.

    Parameters
    ----------
    symbols:
        Symbol indices into ``frequencies``.
    frequencies:
        Normalised table summing to :data:`SCALE_TOTAL`.

    Returns
    -------
    payload:
        Big-endian 16-bit renormalisation words, in decode order.
    state:
        Final rANS state, which the decoder needs as its initial state. Written
        into the container header as a 32-bit integer by
        :mod:`boundcompress.container`.

    Raises
    ------
    CoderError
        If the table is malformed or a symbol has no slots in it.
    """
    if sum(frequencies) != SCALE_TOTAL:
        raise CoderError(
            f"frequency table sums to {sum(frequencies)}, expected {SCALE_TOTAL}"
        )
    starts = cumulative(frequencies)
    words: list[int] = []
    state = RANS_L
    for symbol in reversed(symbols):
        if not 0 <= symbol < len(frequencies):
            raise CoderError(f"symbol {symbol} is outside the alphabet")
        frequency = frequencies[symbol]
        if frequency <= 0:
            raise CoderError(f"symbol {symbol} has no slots in the frequency table")
        limit = ((RANS_L >> SCALE_BITS) << 16) * frequency
        while state >= limit:
            words.append(state & 0xFFFF)
            state >>= 16
        state = ((state // frequency) << SCALE_BITS) + (state % frequency) + starts[symbol]
        assert RANS_L <= state < RANS_L << 16, "rANS state left its normalisation interval"
    words.reverse()
    payload = b"".join(word.to_bytes(2, "big") for word in words)
    return payload, state


def decode(payload: bytes, state: int, frequencies: list[int], count: int) -> list[int]:
    """Decode ``count`` symbols from ``payload`` starting at rANS state ``state``.

    Raises
    ------
    CoderError
        If the table is malformed, the payload is truncated, or the state
        leaves its normalisation interval, all of which mean the container does
        not describe the payload it carries.
    """
    if count < 0:
        raise CoderError(f"symbol count must be non-negative, got {count}")
    if len(payload) % 2:
        raise CoderError(f"payload length {len(payload)} is not a whole number of words")
    table = slot_table(frequencies)
    starts = cumulative(frequencies)
    mask = SCALE_TOTAL - 1
    position = 0
    out: list[int] = []
    for _ in range(count):
        if not RANS_L <= state < RANS_L << 16:
            raise CoderError("rANS state left its normalisation interval during decode")
        slot = state & mask
        symbol = table[slot]
        state = frequencies[symbol] * (state >> SCALE_BITS) + slot - starts[symbol]
        while state < RANS_L:
            if position + 2 > len(payload):
                raise CoderError("payload truncated: the coder needs another word")
            state = (state << 16) | int.from_bytes(payload[position : position + 2], "big")
            position += 2
        out.append(symbol)
    return out
