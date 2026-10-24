"""Symbol mapping, varints, frequency-table serialisation and the entropy figure."""

from __future__ import annotations

import math

import numpy as np
import pytest

from boundcompress import entropy as E


def test_known_answer_zigzag() -> None:
    """Hand-calculated: 0->0, -1->1, 1->2, -2->3, 2->4, -3->5."""
    values = np.array([0, -1, 1, -2, 2, -3])
    assert E.zigzag(values).tolist() == [0, 1, 2, 3, 4, 5]
    assert E.unzigzag(E.zigzag(values)).tolist() == values.tolist()


def test_varint_known_answers() -> None:
    """LEB128: 0 -> 00; 127 -> 7f; 128 -> 80 01; 300 -> ac 02."""
    for value, expected in ((0, b"\x00"), (127, b"\x7f"), (128, b"\x80\x01"),
                            (300, b"\xac\x02")):
        out = bytearray()
        E.put_varint(value, out)
        assert bytes(out) == expected, value
        assert E.get_varint(bytes(out), 0) == (value, len(expected))


def test_negative_varint_is_refused() -> None:
    with pytest.raises(E.SymbolError, match="unsigned"):
        E.put_varint(-1, bytearray())


def test_truncated_varint_is_refused() -> None:
    with pytest.raises(E.SymbolError, match="truncated"):
        E.get_varint(b"\x80", 0)


def test_direct_symbols_below_the_escape() -> None:
    """Residual 127 zigzags to 254, the largest direct symbol."""
    stream = E.to_symbols(np.array([127]))
    assert stream.symbols == [254]
    assert stream.escapes == b""
    assert stream.escape_count == 0


def test_escape_path_for_a_large_residual() -> None:
    """Residual -128 zigzags to 255, which is the escape, so it goes to the side stream."""
    stream = E.to_symbols(np.array([-128]))
    assert stream.symbols == [E.ESCAPE]
    assert stream.escape_count == 1
    assert E.from_symbols(stream.symbols, stream.escapes).tolist() == [-128]


def test_roundtrip_with_mixed_direct_and_escaped_residuals() -> None:
    rng = np.random.default_rng(5)
    residuals = rng.integers(-5000, 5000, size=2000)
    stream = E.to_symbols(residuals)
    assert np.array_equal(E.from_symbols(stream.symbols, stream.escapes), residuals)
    assert stream.escape_count == stream.symbols.count(E.ESCAPE)


def test_unread_escape_bytes_are_refused() -> None:
    with pytest.raises(E.SymbolError, match="unread byte"):
        E.from_symbols([0, 1], b"\x05")


def test_symbol_outside_the_alphabet_is_refused() -> None:
    with pytest.raises(E.SymbolError, match="outside the alphabet"):
        E.from_symbols([999], b"")


def test_entropy_known_answers() -> None:
    """A fair coin is exactly 1 bit; a constant stream is exactly 0 bits."""
    assert E.shannon_entropy_bits([0, 1] * 128) == pytest.approx(1.0, abs=1e-15)
    assert E.shannon_entropy_bits([7] * 64) == pytest.approx(0.0, abs=1e-15)
    assert E.shannon_entropy_bits([]) == 0.0


def test_entropy_of_four_equally_likely_symbols_is_two_bits() -> None:
    assert E.shannon_entropy_bits([0, 1, 2, 3] * 50) == pytest.approx(2.0, abs=1e-14)


def test_entropy_never_exceeds_log2_of_the_alphabet() -> None:
    rng = np.random.default_rng(9)
    symbols = rng.integers(0, E.ALPHABET, size=5000).tolist()
    assert E.shannon_entropy_bits(symbols) <= math.log2(E.ALPHABET) + 1e-12


def test_frequency_table_roundtrip() -> None:
    frequencies = [0] * E.ALPHABET
    frequencies[3] = 100
    frequencies[200] = 7
    frequencies[E.ESCAPE] = 1
    blob = E.table_bytes(frequencies)
    back, position = E.table_from_bytes(blob, 0)
    assert back == frequencies
    assert position == len(blob)


def test_repeated_symbol_in_a_table_is_refused() -> None:
    blob = bytearray()
    E.put_varint(2, blob)
    blob.append(4)
    E.put_varint(10, blob)
    blob.append(4)
    E.put_varint(11, blob)
    with pytest.raises(E.SymbolError, match="repeats symbol"):
        E.table_from_bytes(bytes(blob), 0)


def test_truncated_table_is_refused() -> None:
    blob = bytearray()
    E.put_varint(3, blob)
    blob.append(1)
    E.put_varint(5, blob)
    with pytest.raises(E.SymbolError, match="truncated"):
        E.table_from_bytes(bytes(blob), 0)


def test_symbol_counts_cover_the_whole_alphabet() -> None:
    counts = E.symbol_counts([0, 0, 255])
    assert len(counts) == E.ALPHABET
    assert counts[0] == 2
    assert counts[255] == 1
    assert sum(counts) == 3


def test_value_entropy_known_answers() -> None:
    """Four equally likely residual values is exactly 2 bits; a constant is 0."""
    assert E.value_entropy_bits(np.array([0, 1, 2, 3] * 25)) == pytest.approx(2.0, abs=1e-14)
    assert E.value_entropy_bits(np.array([7] * 50)) == pytest.approx(0.0, abs=1e-15)
    assert E.value_entropy_bits(np.array([], dtype=np.int64)) == 0.0


def test_value_entropy_exceeds_symbol_entropy_on_an_escape_heavy_stream() -> None:
    """The reason both figures exist, asserted rather than only documented.

    Every residual here is far outside the direct-symbol range, so the symbol
    stream is a single repeated escape with zero entropy while the residual
    values carry two bits.
    """
    residuals = np.array([1000, 2000, 3000, 4000] * 25)
    stream = E.to_symbols(residuals)
    assert set(stream.symbols) == {E.ESCAPE}
    assert E.shannon_entropy_bits(stream.symbols) == pytest.approx(0.0, abs=1e-15)
    assert E.value_entropy_bits(residuals) == pytest.approx(2.0, abs=1e-14)


def test_value_entropy_equals_symbol_entropy_when_nothing_escapes() -> None:
    residuals = np.array([0, 1, -1, 2] * 30)
    stream = E.to_symbols(residuals)
    assert E.ESCAPE not in stream.symbols
    assert E.value_entropy_bits(residuals) == pytest.approx(
        E.shannon_entropy_bits(stream.symbols), rel=1e-12
    )
