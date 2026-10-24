"""Container packing, parsing, byte accounting and rejection of malformed buffers."""

from __future__ import annotations

import numpy as np
import pytest

from boundcompress import entropy, rans
from boundcompress.container import (
    FIXED_HEADER_BYTES,
    MAGIC,
    VERSION,
    Container,
    ContainerError,
    block_sizes,
    pack,
    unpack,
)


def make_container(**overrides: object) -> Container:
    """A minimal valid container, with fields overridable per test."""
    # The table spans the whole alphabet, as the codec's always does: the
    # serialised form writes only the used symbols but parses back to a
    # full-alphabet list, so a short table is not a round-trip fixed point.
    frequencies = rans.normalise_frequencies(entropy.symbol_counts([0, 1, 2, 0]))
    payload, state = rans.encode([0, 1, 2, 0], frequencies)
    fields = {
        "mode": "abs",
        "transform": 0,
        "sign": 1,
        "predictor": 2,
        "order": 2,
        "declared_bound": 1e-3,
        "step": 2e-3,
        "count": 6,
        "escape_count": 0,
        "state": state,
        "warmup": [-7, 13],
        "frequencies": frequencies,
        "escapes": b"",
        "payload": payload,
    }
    fields.update(overrides)
    return Container(**fields)  # type: ignore[arg-type]


def test_fixed_header_size_is_what_the_docstring_says() -> None:
    assert FIXED_HEADER_BYTES == 38


def test_roundtrip_preserves_every_field() -> None:
    container = make_container()
    back = unpack(pack(container))
    assert back == container


def test_negative_warmup_values_survive_the_zigzag_varint() -> None:
    container = make_container(warmup=[-1_000_000, 999_999], order=2)
    assert unpack(pack(container)).warmup == [-1_000_000, 999_999]


def test_block_sizes_sum_to_the_container_length() -> None:
    container = make_container()
    sizes = block_sizes(container)
    assert sum(sizes.values()) == len(pack(container))
    assert sizes["fixed_header"] == FIXED_HEADER_BYTES


def test_short_buffer_is_refused() -> None:
    with pytest.raises(ContainerError, match="shorter than"):
        unpack(b"BCMP")


def test_wrong_magic_is_refused() -> None:
    blob = bytearray(pack(make_container()))
    blob[0:4] = b"XXXX"
    with pytest.raises(ContainerError, match="magic is"):
        unpack(bytes(blob))


def test_wrong_version_is_refused() -> None:
    blob = bytearray(pack(make_container()))
    blob[4] = VERSION + 7
    with pytest.raises(ContainerError, match="not readable by version"):
        unpack(bytes(blob))


def test_unknown_mode_is_refused() -> None:
    blob = bytearray(pack(make_container()))
    blob[5] = 99
    with pytest.raises(ContainerError, match="unknown bound mode"):
        unpack(bytes(blob))


def test_trailing_bytes_are_refused() -> None:
    with pytest.raises(ContainerError, match="trailing byte"):
        unpack(pack(make_container()) + b"\x00")


def test_truncated_payload_block_is_refused() -> None:
    blob = pack(make_container())
    with pytest.raises(ContainerError, match="runs past the end|malformed|trailing"):
        unpack(blob[:-2])


def test_magic_is_four_bytes() -> None:
    assert len(MAGIC) == 4
    assert np.frombuffer(MAGIC, dtype=np.uint8).size == 4
