"""File segmentation, deterministic payload generation and the hash identity."""

from __future__ import annotations

import hashlib

import pytest

from relfiledelivery.payload import (
    CompletionInvariantError,
    FileTransferSpec,
    digest,
    reassemble,
    source_payload,
)


class TestFileTransferSpec:
    def test_hand_calculated_segmentation(self, tiny_spec: FileTransferSpec) -> None:
        # 950 bytes in 100-byte units: 10 units, the last carrying 50 bytes.
        # Wire size is 950 + 10 * 12 = 1070 bytes, so the overhead ratio is
        # 1070 / 950 = 1.1263157894736842.
        assert tiny_spec.pdu_count == 10
        assert tiny_spec.last_payload_bytes == 50
        assert tiny_spec.total_wire_bytes == 1070
        assert tiny_spec.minimum_overhead_ratio == pytest.approx(1070.0 / 950.0)
        assert tiny_spec.full_wire_bytes == 112

    def test_exact_multiple_has_a_full_last_unit(self) -> None:
        spec = FileTransferSpec(1000, 100)
        assert spec.pdu_count == 10
        assert spec.last_payload_bytes == 100

    def test_single_unit_file(self) -> None:
        spec = FileTransferSpec(7, 100)
        assert spec.pdu_count == 1
        assert spec.last_payload_bytes == 7
        assert spec.payload_bytes_of(0) == 7

    def test_slices_cover_the_file_exactly_once(self, tiny_spec: FileTransferSpec) -> None:
        slices = tiny_spec.slices()
        assert slices[0] == (0, 100)
        assert slices[-1] == (900, 950)
        assert sum(stop - start for start, stop in slices) == tiny_spec.file_bytes

    def test_payload_bytes_of_rejects_an_out_of_range_sequence(
        self, tiny_spec: FileTransferSpec
    ) -> None:
        with pytest.raises(ValueError, match=r"\[0, 10\)"):
            tiny_spec.payload_bytes_of(10)
        with pytest.raises(ValueError, match=r"\[0, 10\)"):
            tiny_spec.payload_bytes_of(-1)

    @pytest.mark.parametrize(
        ("file_bytes", "payload", "header"),
        [(0, 100, 12), (-5, 100, 12), (100, 0, 12), (100, -1, 12), (100, 10, -1)],
    )
    def test_invalid_sizes_are_rejected(
        self, file_bytes: int, payload: int, header: int
    ) -> None:
        with pytest.raises(ValueError):
            FileTransferSpec(file_bytes, payload, header)

    def test_non_integer_sizes_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            FileTransferSpec(100.5, 10)  # type: ignore[arg-type]

    def test_zero_header_is_allowed_and_gives_unit_overhead(self) -> None:
        spec = FileTransferSpec(1000, 100, 0)
        assert spec.minimum_overhead_ratio == pytest.approx(1.0)

    def test_describe_states_units(self, tiny_spec: FileTransferSpec) -> None:
        assert any("bytes" in line for line in tiny_spec.describe())


class TestDigestAndPayload:
    def test_digest_matches_hashlib(self) -> None:
        assert digest(b"relfiledelivery") == hashlib.sha256(b"relfiledelivery").hexdigest()

    def test_source_payload_is_deterministic_in_the_seed(
        self, tiny_spec: FileTransferSpec
    ) -> None:
        first = source_payload(tiny_spec, 1234)
        second = source_payload(tiny_spec, 1234)
        assert first == second
        assert len(first) == tiny_spec.file_bytes

    def test_different_seeds_give_different_payloads(
        self, tiny_spec: FileTransferSpec
    ) -> None:
        assert source_payload(tiny_spec, 1) != source_payload(tiny_spec, 2)


class TestReassemble:
    def test_round_trip_reproduces_the_source(self, tiny_spec: FileTransferSpec) -> None:
        source = source_payload(tiny_spec, 42)
        received = {
            index: source[start:stop] for index, (start, stop) in enumerate(tiny_spec.slices())
        }
        assert reassemble(tiny_spec, received) == source
        assert digest(reassemble(tiny_spec, received)) == digest(source)

    def test_a_missing_unit_raises_rather_than_hashing_a_partial_file(
        self, tiny_spec: FileTransferSpec
    ) -> None:
        source = source_payload(tiny_spec, 42)
        received = {
            index: source[start:stop] for index, (start, stop) in enumerate(tiny_spec.slices())
        }
        del received[4]
        with pytest.raises(KeyError):
            reassemble(tiny_spec, received)

    def test_a_wrong_length_unit_is_rejected(self, tiny_spec: FileTransferSpec) -> None:
        source = source_payload(tiny_spec, 42)
        received = {
            index: source[start:stop] for index, (start, stop) in enumerate(tiny_spec.slices())
        }
        received[2] = received[2][:-1]
        with pytest.raises(ValueError, match="carries 99 bytes"):
            reassemble(tiny_spec, received)


def test_completion_invariant_error_is_an_assertion_error() -> None:
    assert issubclass(CompletionInvariantError, AssertionError)
