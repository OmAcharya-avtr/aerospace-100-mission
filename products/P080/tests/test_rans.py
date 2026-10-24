"""rANS coder tests. Round-trip, table normalisation, and malformed input."""

from __future__ import annotations

import numpy as np
import pytest

from boundcompress import rans


def test_normalise_sums_to_the_scale_total() -> None:
    counts = [0, 5, 0, 1000, 3, 0, 17]
    frequencies = rans.normalise_frequencies(counts)
    assert sum(frequencies) == rans.SCALE_TOTAL
    for count, frequency in zip(counts, frequencies, strict=True):
        assert (frequency > 0) == (count > 0)


def test_every_used_symbol_keeps_a_slot_even_when_rare() -> None:
    """One occurrence in a million must still be codable."""
    counts = [1_000_000, 1]
    frequencies = rans.normalise_frequencies(counts)
    assert frequencies[1] >= 1
    assert sum(frequencies) == rans.SCALE_TOTAL


def test_known_answer_uniform_binary_table() -> None:
    """Two equally likely symbols give 2048 slots each; hand-calculated."""
    frequencies = rans.normalise_frequencies([10, 10])
    assert frequencies[:2] == [2048, 2048]


def test_roundtrip_single_symbol_stream() -> None:
    frequencies = rans.normalise_frequencies([1])
    payload, state = rans.encode([0] * 50, frequencies)
    assert rans.decode(payload, state, frequencies, 50) == [0] * 50


def test_roundtrip_over_seeded_random_streams() -> None:
    rng = np.random.default_rng(20261024)
    for _ in range(40):
        alphabet = int(rng.integers(1, 64))
        length = int(rng.integers(1, 600))
        symbols = rng.integers(0, alphabet, size=length).tolist()
        counts = [0] * alphabet
        for symbol in symbols:
            counts[symbol] += 1
        frequencies = rans.normalise_frequencies(counts)
        payload, state = rans.encode(symbols, frequencies)
        assert state < 1 << 32, "final state must fit the container's uint32 field"
        assert rans.decode(payload, state, frequencies, length) == symbols


def test_empty_counts_are_refused() -> None:
    with pytest.raises(rans.CoderError, match="empty symbol stream"):
        rans.normalise_frequencies([0, 0, 0])


def test_negative_counts_are_refused() -> None:
    with pytest.raises(rans.CoderError, match="non-negative"):
        rans.normalise_frequencies([3, -1])


def test_encoding_a_zero_frequency_symbol_is_refused() -> None:
    frequencies = rans.normalise_frequencies([10, 0])
    with pytest.raises(rans.CoderError, match="no slots"):
        rans.encode([1], frequencies)


def test_symbol_outside_the_alphabet_is_refused() -> None:
    frequencies = rans.normalise_frequencies([10, 10])
    with pytest.raises(rans.CoderError, match="outside the alphabet"):
        rans.encode([5], frequencies)


def test_table_that_does_not_sum_to_the_total_is_refused() -> None:
    with pytest.raises(rans.CoderError, match="sums to"):
        rans.encode([0], [1, 1])
    with pytest.raises(rans.CoderError, match="sums to"):
        rans.slot_table([1, 1])


def test_truncated_payload_is_refused() -> None:
    frequencies = rans.normalise_frequencies([1, 1, 1])
    symbols = [0, 1, 2] * 60
    payload, state = rans.encode(symbols, frequencies)
    with pytest.raises(rans.CoderError, match="truncated"):
        rans.decode(payload[:4], state, frequencies, len(symbols))


def test_odd_length_payload_is_refused() -> None:
    frequencies = rans.normalise_frequencies([1, 1])
    with pytest.raises(rans.CoderError, match="whole number of words"):
        rans.decode(b"\x00", 1 << 16, frequencies, 1)


def test_negative_count_is_refused() -> None:
    frequencies = rans.normalise_frequencies([1, 1])
    with pytest.raises(rans.CoderError, match="non-negative"):
        rans.decode(b"", 1 << 16, frequencies, -1)


def test_cumulative_is_an_exclusive_prefix_sum() -> None:
    assert rans.cumulative([2, 3, 5]) == [0, 2, 5, 10]


def test_slot_table_assigns_contiguous_runs() -> None:
    frequencies = rans.normalise_frequencies([1, 1])
    table = rans.slot_table(frequencies)
    assert table[0] == 0
    assert table[-1] == 1
    assert table.count(0) == frequencies[0]
