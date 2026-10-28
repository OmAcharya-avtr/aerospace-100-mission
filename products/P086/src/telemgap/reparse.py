"""The reinterpretation test: positive evidence that the reader, not the link, is wrong.

Why this module is the load-bearing part of the three-way classification
------------------------------------------------------------------------
A damaged frame and a correctly received frame read with the wrong decode both
present identically at the analyst's level: a channel value outside its
declared range. No screen over the engineering values can separate them,
because the engineering values are the output of the thing under suspicion.

What *does* separate them is a question about the bits. If a **different,
declared** reading of the *same* raw word produces a value that is both inside
the declared range and consistent with the frame's neighbours, then the bits
carried a plausible measurement and the reader mapped them wrongly. That is
positive evidence of a misparse. If no declared alternative reading rescues the
word, the evidence is absent and the verdict is *undecided* -- not *corrupt*.
Absence of a rescuing reinterpretation is not evidence of corruption, because
the misparse may be one this module does not model.

The alternative readings modelled
---------------------------------
Each is a real ground-system fault with a documented history, not a
hypothetical:

``byteswap``
    Word endianness reversed. CCSDS space packets are big-endian on the wire
    (CCSDS 133.0-B-2 §4.1) and little-endian hosts dominate ground segments, so
    a missing swap is the single most common decommutation fault.
``signedness``
    A two's complement field read as unsigned, or the reverse. Presents as a
    value near full scale where the true value is slightly negative.
``scale_decade_up`` / ``scale_decade_down``
    Calibration scale factor wrong by a factor of ten, the classic unit
    confusion in a hand-maintained decommutation database.
``offset_omitted``
    The additive term of equation (1) dropped, so ``value = raw * scale``.

Equation
--------
For a raw word ``w`` of ``b`` bits read as the integer ``r(w)`` and decoded by
equation (1), each alternative supplies a different pair ``(r', scale',
offset')`` and therefore a different candidate value

    v' = r'(w) * scale' + offset'                                       (6)

A candidate is accepted as evidence of a misparse when **both**

    low <= v' <= high                                                   (7a)
    |v' - m| <= tau * (1.4826 * MAD + eps)                              (7b)

hold, with ``m`` the local median of the channel from
:func:`telemgap.screens.robust_local_statistics` and ``tau`` the declared
consistency tolerance in robust standard deviations. Condition (7a) alone is
not sufficient and the reason is quantitative: a uniformly random 16-bit word
byte-swapped into a channel occupying a fraction ``f`` of its word's dynamic
range lands inside the declared range with probability of order ``f``, so for a
wide channel a chance rescue is common. Requiring (7b) as well reduces that to
the probability of landing inside a few robust standard deviations of the local
median, which is measured in ``validation/validate_classification.py`` rather
than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .record import ChannelSpec

#: Names of the alternative readings, in the order they are tried. The order
#: only decides which name is reported when several alternatives both satisfy
#: (7a) and (7b); the verdict does not depend on it.
ALTERNATIVES = (
    "byteswap",
    "signedness",
    "scale_decade_up",
    "scale_decade_down",
    "offset_omitted",
)

#: Default consistency tolerance ``tau`` of equation (7b), in robust standard
#: deviations. Deliberately tight: a loose tolerance turns the reinterpretation
#: test into a coin flip on wide channels.
DEFAULT_CONSISTENCY_TAU = 4.0


@dataclass(frozen=True)
class Reinterpretation:
    """One candidate alternative reading of a raw word, with its verdict."""

    alternative: str
    raw_as_read: int
    raw_alternative: int
    value_as_read: float
    value_alternative: float
    in_range: bool
    consistent: bool
    deviation_sigmas: float

    @property
    def rescues(self) -> bool:
        """Whether both (7a) and (7b) hold, making this positive misparse evidence."""
        return bool(self.in_range and self.consistent)


def byte_reverse(word: int, word_bits: int) -> int:
    """Reverse the byte order of ``word``, an unsigned ``word_bits``-bit integer.

    Raises ``ValueError`` if ``word_bits`` is not a positive multiple of 8 or if
    ``word`` does not fit.
    """
    if word_bits <= 0 or word_bits % 8 != 0:
        raise ValueError(f"word_bits must be a positive multiple of 8, got {word_bits!r}")
    word = int(word)
    if not 0 <= word < (1 << word_bits):
        raise ValueError(f"word {word!r} does not fit in {word_bits} bits")
    n_bytes = word_bits // 8
    result = 0
    for _ in range(n_bytes):
        result = (result << 8) | (word & 0xFF)
        word >>= 8
    return result


def as_signed(word: int, word_bits: int) -> int:
    """Two's complement interpretation of an unsigned ``word_bits``-bit pattern."""
    word = int(word)
    half = 1 << (word_bits - 1)
    return word - (1 << word_bits) if word >= half else word


def raw_as_read(word: int, channel: ChannelSpec) -> int:
    """The integer the reader used for ``word`` under ``channel``'s declared signedness."""
    return as_signed(word, channel.word_bits) if channel.signed else int(word)


def value_as_read(word: int, channel: ChannelSpec) -> float:
    """Engineering value the reader produced for ``word``, equation (1)."""
    return float(raw_as_read(word, channel) * channel.scale + channel.offset)


def alternative_value(word: int, channel: ChannelSpec, alternative: str) -> tuple[int, float]:
    """Raw integer and engineering value under one alternative reading, equation (6).

    Returns
    -------
    (raw, value)
        ``raw`` is the integer the alternative reading would use; for the two
        scale alternatives and ``offset_omitted`` the raw integer is unchanged
        and only the calibration differs.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"unknown alternative {alternative!r}; expected one of {ALTERNATIVES}")
    word = int(word)
    bits = channel.word_bits
    if alternative == "byteswap":
        swapped = byte_reverse(word, bits)
        raw = raw_as_read(swapped, channel)
        return raw, float(raw * channel.scale + channel.offset)
    if alternative == "signedness":
        raw = int(word) if channel.signed else as_signed(word, bits)
        return raw, float(raw * channel.scale + channel.offset)
    raw = raw_as_read(word, channel)
    if alternative == "scale_decade_up":
        return raw, float(raw * channel.scale * 10.0 + channel.offset)
    if alternative == "scale_decade_down":
        return raw, float(raw * channel.scale / 10.0 + channel.offset)
    return raw, float(raw * channel.scale)


def reinterpret(
    word: int,
    channel: ChannelSpec,
    local_median: float,
    scaled_mad: float,
    tau: float = DEFAULT_CONSISTENCY_TAU,
) -> tuple[Reinterpretation, ...]:
    """Evaluate every alternative reading of ``word`` against (7a) and (7b).

    Parameters
    ----------
    word
        The raw unsigned word, as retained in ``TelemetryRecord.raw_words``.
    channel
        Its :class:`~telemgap.record.ChannelSpec`.
    local_median, scaled_mad
        Local robust statistics of the channel at this frame, from
        :func:`telemgap.screens.robust_local_statistics`. ``scaled_mad`` must be
        positive; the caller's floor guarantees it.
    tau
        Consistency tolerance of (7b), in robust standard deviations.

    Returns
    -------
    tuple of Reinterpretation
        One entry per alternative, in :data:`ALTERNATIVES` order. Entries whose
        candidate value is not finite are returned with both conditions false
        rather than dropped, so the caller can report what was tried.
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau!r}")
    if not np.isfinite(scaled_mad) or scaled_mad <= 0.0:
        raise ValueError(f"scaled_mad must be positive and finite, got {scaled_mad!r}")
    read_raw = raw_as_read(word, channel)
    read_value = float(read_raw * channel.scale + channel.offset)
    results: list[Reinterpretation] = []
    for alternative in ALTERNATIVES:
        raw, value = alternative_value(word, channel, alternative)
        if np.isfinite(value):
            deviation = abs(value - local_median) / scaled_mad
            in_range = bool(channel.low <= value <= channel.high)
            consistent = bool(deviation <= tau)
        else:
            deviation, in_range, consistent = float("inf"), False, False
        results.append(
            Reinterpretation(
                alternative=alternative,
                raw_as_read=int(read_raw),
                raw_alternative=int(raw),
                value_as_read=read_value,
                value_alternative=float(value),
                in_range=in_range,
                consistent=consistent,
                deviation_sigmas=float(deviation),
            )
        )
    return tuple(results)


def best_rescue(
    candidates: tuple[Reinterpretation, ...],
) -> Reinterpretation | None:
    """The rescuing candidate closest to the local median, or ``None`` if none rescues."""
    rescuing = [candidate for candidate in candidates if candidate.rescues]
    if not rescuing:
        return None
    return min(rescuing, key=lambda candidate: candidate.deviation_sigmas)


def _raw_as_read_array(words: np.ndarray, channel: ChannelSpec) -> np.ndarray:
    """Vectorised :func:`raw_as_read` over an array of unsigned words."""
    raw = np.asarray(words, dtype=np.uint64).astype(np.int64)
    if channel.signed:
        span = np.int64(1) << np.int64(channel.word_bits)
        half = span >> np.int64(1)
        raw = np.where(raw >= half, raw - span, raw)
    return raw


def byte_reverse_array(words: np.ndarray, word_bits: int) -> np.ndarray:
    """Vectorised :func:`byte_reverse`. Returns int64 to match the other helpers."""
    if word_bits <= 0 or word_bits % 8 != 0:
        raise ValueError(f"word_bits must be a positive multiple of 8, got {word_bits!r}")
    current = np.asarray(words, dtype=np.uint64).astype(np.int64)
    result = np.zeros_like(current)
    for _ in range(word_bits // 8):
        result = (result << np.int64(8)) | (current & np.int64(0xFF))
        current = current >> np.int64(8)
    return result


def alternative_value_array(
    words: np.ndarray, channel: ChannelSpec, alternative: str
) -> np.ndarray:
    """Vectorised :func:`alternative_value`, returning engineering values only.

    Agreement with the scalar implementation is asserted by test over the whole
    16-bit word range, not assumed.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"unknown alternative {alternative!r}; expected one of {ALTERNATIVES}")
    words = np.asarray(words, dtype=np.uint64)
    if alternative == "byteswap":
        swapped = byte_reverse_array(words, channel.word_bits)
        raw = _raw_as_read_array(swapped.astype(np.uint64), channel)
        return raw.astype(np.float64) * channel.scale + channel.offset
    if alternative == "signedness":
        if channel.signed:
            raw = words.astype(np.int64)
        else:
            span = np.int64(1) << np.int64(channel.word_bits)
            half = span >> np.int64(1)
            signed = words.astype(np.int64)
            raw = np.where(signed >= half, signed - span, signed)
        return raw.astype(np.float64) * channel.scale + channel.offset
    raw = _raw_as_read_array(words, channel).astype(np.float64)
    if alternative == "scale_decade_up":
        return raw * channel.scale * 10.0 + channel.offset
    if alternative == "scale_decade_down":
        return raw * channel.scale / 10.0 + channel.offset
    return raw * channel.scale


def reinterpret_array(
    words: np.ndarray,
    channel: ChannelSpec,
    local_median: np.ndarray,
    scaled_mad: np.ndarray,
    tau: float = DEFAULT_CONSISTENCY_TAU,
) -> dict[str, np.ndarray]:
    """Evaluate (7a) and (7b) for every word and every alternative at once.

    Returns
    -------
    dict
        ``"values"``, ``"in_range"``, ``"consistent"``, ``"rescues"`` and
        ``"deviation"``, each ``(n, len(ALTERNATIVES))``, plus
        ``"any_rescue"`` ``(n,)`` bool, ``"best_deviation"`` ``(n,)`` float
        (``inf`` where nothing rescues) and ``"best_index"`` ``(n,)`` int
        (``-1`` where nothing rescues).
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau!r}")
    words = np.asarray(words, dtype=np.uint64)
    median = np.asarray(local_median, dtype=np.float64)
    scaled = np.asarray(scaled_mad, dtype=np.float64)
    if words.shape != median.shape or words.shape != scaled.shape:
        raise ValueError(
            f"words {words.shape}, local_median {median.shape} and scaled_mad {scaled.shape} "
            "must have the same shape"
        )
    if np.any(~np.isfinite(scaled)) or np.any(scaled <= 0.0):
        raise ValueError("scaled_mad must be positive and finite everywhere")
    values = np.column_stack(
        [alternative_value_array(words, channel, name) for name in ALTERNATIVES]
    )
    finite = np.isfinite(values)
    deviation = np.where(
        finite, np.abs(values - median[:, None]) / scaled[:, None], np.inf
    )
    in_range = finite & (values >= channel.low) & (values <= channel.high)
    consistent = deviation <= tau
    rescues = in_range & consistent
    masked = np.where(rescues, deviation, np.inf)
    best_index = np.where(rescues.any(axis=1), np.argmin(masked, axis=1), -1)
    best_deviation = np.where(best_index >= 0, masked.min(axis=1), np.inf)
    return {
        "values": values,
        "in_range": in_range,
        "consistent": consistent,
        "rescues": rescues,
        "deviation": deviation,
        "any_rescue": rescues.any(axis=1),
        "best_index": best_index,
        "best_deviation": best_deviation,
    }
