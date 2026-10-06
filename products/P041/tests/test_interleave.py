"""Interleavers: permutation identity, burst dispersion, latency and memory cost.

Exercises REQ-12, REQ-13, REQ-14 of docs/REQUIREMENTS.md.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from codedfade.interleave import (
    BlockInterleaver,
    ConvolutionalInterleaver,
    burst_dispersion,
)


class TestBlockInterleaver:
    def test_known_answer_depth_3_span_4(self) -> None:
        """Write rows 0..2 of 4, read columns. Hand-computed permutation:

            rows      0  1  2  3
                      4  5  6  7
                      8  9 10 11
            columns   0 4 8 | 1 5 9 | 2 6 10 | 3 7 11
        """
        out = BlockInterleaver(3, 4).interleave(np.arange(12))
        assert out.tolist() == [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

    def test_deinterleave_inverts_interleave(self) -> None:
        il = BlockInterleaver(5, 7)
        x = np.arange(70)
        assert np.array_equal(il.deinterleave(il.interleave(x)), x)

    def test_adjacent_codeword_symbols_are_depth_apart(self) -> None:
        """With span = n, symbol j of codeword c lands at channel index j*D + c."""
        depth, span = 8, 31
        il = BlockInterleaver(depth, span)
        marks = np.zeros(depth * span, dtype=np.int64)
        marks[:span] = 1  # codeword 0 occupies the first row
        channel = il.interleave(marks)
        positions = np.nonzero(channel)[0]
        assert np.all(np.diff(positions) == depth)

    def test_depth_one_is_the_identity(self) -> None:
        il = BlockInterleaver(1, 9)
        x = np.arange(27)
        assert np.array_equal(il.interleave(x), x)

    def test_cost_formulas(self) -> None:
        cost = BlockInterleaver(64, 31).cost(1.0e6, bits_per_symbol=5)
        assert cost.latency_symbols == 2 * 64 * 31
        assert cost.latency_ms == pytest.approx(1000.0 * 2 * 64 * 31 / 1e6)
        assert cost.memory_symbols == 2 * 64 * 31
        assert cost.memory_bytes == pytest.approx(2 * 64 * 31 * 5 / 8)

    def test_max_errors_per_codeword(self) -> None:
        il = BlockInterleaver(10, 31)
        assert il.max_errors_per_codeword(0) == 0
        assert il.max_errors_per_codeword(1) == 1
        assert il.max_errors_per_codeword(10) == 1
        assert il.max_errors_per_codeword(11) == 2

    @pytest.mark.parametrize(
        "depth,span,msg",
        [(0, 4, "depth must be >= 1"), (3, 0, "span must be >= 1")],
    )
    def test_rejects_bad_geometry(self, depth: int, span: int, msg: str) -> None:
        with pytest.raises(ValueError, match=msg):
            BlockInterleaver(depth, span)

    def test_rejects_ragged_block(self) -> None:
        il = BlockInterleaver(3, 4)
        with pytest.raises(ValueError, match="not a multiple of"):
            il.interleave(np.arange(13))
        with pytest.raises(ValueError, match="not a multiple of"):
            il.deinterleave(np.arange(13))

    def test_rejects_non_1d(self) -> None:
        il = BlockInterleaver(3, 4)
        with pytest.raises(ValueError, match="1-D"):
            il.interleave(np.zeros((3, 4)))
        with pytest.raises(ValueError, match="1-D"):
            il.deinterleave(np.zeros((3, 4)))

    def test_rejects_bad_cost_arguments(self) -> None:
        il = BlockInterleaver(3, 4)
        with pytest.raises(ValueError, match="symbol_rate_hz"):
            il.cost(0.0)
        with pytest.raises(ValueError, match="bits_per_symbol"):
            il.cost(1e6, 0)

    def test_rejects_negative_burst(self) -> None:
        with pytest.raises(ValueError, match="burst_symbols"):
            BlockInterleaver(3, 4).max_errors_per_codeword(-1)


class TestConvolutionalInterleaver:
    def test_end_to_end_delay_is_measured_not_asserted(self) -> None:
        """Push a ramp through and find the delay at which the output matches."""
        ci = ConvolutionalInterleaver(4, 2)
        x = np.arange(1, 401)
        out = ci.deinterleave(ci.interleave(x))
        d = ci.total_delay_symbols
        assert d == 2 * 4 * 3
        assert np.array_equal(out[d:], x[: x.size - d])

    def test_memory_is_half_the_block_interleaver_for_the_same_spread(self) -> None:
        ci = ConvolutionalInterleaver(16, 1)
        bi = BlockInterleaver(16, 16)
        assert ci.memory_symbols < bi.cost(1e6).memory_symbols

    def test_single_branch_is_the_identity(self) -> None:
        ci = ConvolutionalInterleaver(1, 1)
        x = np.arange(20)
        assert np.array_equal(ci.interleave(x), x)
        assert ci.total_delay_symbols == 0

    def test_cost_formulas(self) -> None:
        cost = ConvolutionalInterleaver(8, 4).cost(2.0e6, bits_per_symbol=8)
        assert cost.latency_symbols == 4 * 8 * 7
        assert cost.memory_bytes == pytest.approx(4 * 8 * 7 * 8 / 8)
        assert cost.latency_ms == pytest.approx(1000.0 * 4 * 8 * 7 / 2e6)

    @pytest.mark.parametrize(
        "branches,inc,msg",
        [(0, 1, "branches must be >= 1"), (4, 0, "delay_increment must be >= 1")],
    )
    def test_rejects_bad_geometry(self, branches: int, inc: int, msg: str) -> None:
        with pytest.raises(ValueError, match=msg):
            ConvolutionalInterleaver(branches, inc)

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            ConvolutionalInterleaver(4, 1).interleave(np.zeros((2, 2)))

    def test_rejects_bad_cost_arguments(self) -> None:
        ci = ConvolutionalInterleaver(4, 1)
        with pytest.raises(ValueError, match="symbol_rate_hz"):
            ci.cost(-1.0)
        with pytest.raises(ValueError, match="bits_per_symbol"):
            ci.cost(1e6, 0)


class TestBurstDispersion:
    def test_known_answers(self) -> None:
        assert burst_dispersion(1, 20) == 20
        assert burst_dispersion(20, 20) == 1
        assert burst_dispersion(7, 20) == 3

    def test_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="depth must be >= 1"):
            burst_dispersion(0, 10)
        with pytest.raises(ValueError, match="burst_symbols must be >= 0"):
            burst_dispersion(2, -1)

    def test_depth_t_rule(self) -> None:
        """Equation (23): a burst of t*D symbols is the longest that still
        deposits at most t errors in one codeword."""
        t, depth = 5, 64
        assert burst_dispersion(depth, t * depth) == t
        assert burst_dispersion(depth, t * depth + 1) == t + 1


@settings(max_examples=40, deadline=None)
@given(
    depth=st.integers(min_value=1, max_value=12),
    span=st.integers(min_value=1, max_value=12),
    blocks=st.integers(min_value=1, max_value=4),
)
def test_block_interleaver_is_a_permutation(depth: int, span: int, blocks: int) -> None:
    il = BlockInterleaver(depth, span)
    x = np.arange(depth * span * blocks)
    y = il.interleave(x)
    assert sorted(y.tolist()) == x.tolist()
    assert np.array_equal(il.deinterleave(y), x)


@settings(max_examples=30, deadline=None)
@given(
    branches=st.integers(min_value=1, max_value=8),
    inc=st.integers(min_value=1, max_value=4),
)
def test_convolutional_interleaver_is_a_pure_delay(branches: int, inc: int) -> None:
    ci = ConvolutionalInterleaver(branches, inc)
    n = 20 * branches * inc + 50
    x = np.arange(1, n + 1)
    out = ci.deinterleave(ci.interleave(x))
    d = ci.total_delay_symbols
    assert np.array_equal(out[d:], x[: n - d])
