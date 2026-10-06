"""Cost accounting and the design search."""

import dataclasses

import numpy as np
import pytest

from interleavekit import (
    BlockInterleaver,
    ConvolutionalInterleaver,
    HelicalInterleaver,
    SRandomInterleaver,
)
from interleavekit.base import InterleaverCost
from interleavekit.cost import (
    cheapest_for_burst,
    cost_table,
    describe,
    format_cost_table,
    row_as_dict,
)


def test_latency_ms_hand_computed():
    """512 symbols at 1 Msym/s is 512 us = 0.512 ms."""
    cost = BlockInterleaver(16, 16).cost()
    assert cost.pair_latency_symbols == 512
    assert cost.latency_ms(1.0e6) == pytest.approx(0.512)
    assert cost.latency_ms(1.0e6, pair=False) == pytest.approx(0.256)


def test_memory_bytes_hand_computed():
    """512 symbols at 8 bits per symbol is 512 bytes; at 4 bits, 256 bytes."""
    cost = BlockInterleaver(16, 16).cost()
    assert cost.memory_bytes(8) == pytest.approx(512.0)
    assert cost.memory_bytes(4) == pytest.approx(256.0)
    assert cost.memory_bytes(8, pair=False) == pytest.approx(256.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_bad_symbol_rate_raises(bad):
    with pytest.raises(ValueError, match="finite positive rate"):
        BlockInterleaver(2, 2).cost().latency_ms(bad)


def test_bad_bits_per_symbol_raises():
    with pytest.raises(ValueError, match=r"bits_per_symbol must be >= 1"):
        BlockInterleaver(2, 2).cost().memory_bytes(0)


def test_cost_is_frozen():
    cost = BlockInterleaver(2, 2).cost()
    assert isinstance(cost, InterleaverCost)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cost.model = "other"  # type: ignore[misc]


def test_describe_block():
    row = describe(BlockInterleaver(8, 8), 1.0e6)
    assert row.block_symbols == 64
    assert row.pair_memory_symbols == 128
    assert row.minimum_spread == 9
    assert row.max_burst_fully_dispersed == 8
    assert row.cost_model == "full-block buffering"


def test_describe_convolutional_has_no_minimum_spread():
    row = describe(ConvolutionalInterleaver(4, 2), 1.0e6, burst_search_limit=20)
    assert row.minimum_spread is None
    assert row.max_burst_fully_dispersed >= 1
    assert row.cost_model == "shift-register bank"


def test_describe_rejects_other_types():
    with pytest.raises(TypeError, match="must be an Interleaver"):
        describe(np.arange(4), 1.0e6)  # type: ignore[arg-type]


def test_cost_table_and_formatting():
    rows = cost_table(
        [
            BlockInterleaver(8, 8),
            HelicalInterleaver(8, 8, 3),
            SRandomInterleaver(64, 5, seed=0),
            ConvolutionalInterleaver(8, 1),
        ],
        1.0e6,
        burst_search_limit=32,
        conv_window=64,
    )
    assert len(rows) == 4
    text = format_cost_table(rows, 1.0e6)
    assert "BlockInterleaver(depth=8, span=8)" in text
    assert "symbol rate 1e+06 sym/s" in text
    lines = text.splitlines()
    # 1 rate line + 2 header lines + 1 rule + one line per row
    assert len(lines) == 4 + len(rows)
    assert set(lines[3]) == {"-"}


def test_row_as_dict_round_trips_the_fields():
    row = describe(BlockInterleaver(4, 4), 1.0e6)
    d = row_as_dict(row)
    assert d["block_symbols"] == 16
    assert d["max_burst_fully_dispersed"] == 4


def test_cheapest_for_burst_finds_candidates_that_meet_the_target():
    target = 8
    best = cheapest_for_burst(target, max_block_symbols=256, max_registers=12, max_slope=12)
    assert set(best) == {"block", "helical", "convolutional"}
    for family, row in best.items():
        assert row is not None, family
        assert row.max_burst_fully_dispersed >= target, family


def test_cheapest_for_burst_prefers_the_convolutional_bank_on_memory():
    """Measured, not assumed: the shift-register bank buys the same burst
    dispersion for less memory than a buffered block of either read order."""
    best = cheapest_for_burst(16, max_block_symbols=512, max_registers=20, max_slope=20)
    conv = best["convolutional"]
    block = best["block"]
    assert conv is not None and block is not None
    assert conv.pair_memory_symbols < block.pair_memory_symbols


def test_cheapest_for_burst_reports_none_when_the_ceilings_are_too_low():
    best = cheapest_for_burst(
        200, families=("convolutional",), max_registers=2, max_slope=2
    )
    assert best["convolutional"] is None


def test_cheapest_for_burst_rejects_bad_input():
    with pytest.raises(ValueError, match=r"target_burst must be >= 1"):
        cheapest_for_burst(0)
    with pytest.raises(TypeError, match="target_burst must be an integer"):
        cheapest_for_burst(4.0)
    with pytest.raises(ValueError, match="unknown families"):
        cheapest_for_burst(4, families=("quantum",))


def test_srandom_family_is_searchable():
    best = cheapest_for_burst(2, families=("srandom",), max_block_symbols=64)
    assert best["srandom"] is None or best["srandom"].max_burst_fully_dispersed >= 2
