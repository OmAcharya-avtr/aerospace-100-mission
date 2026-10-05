"""Site sampling and exact bitwise injection."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bitflipsim.injection import (
    BitUpset,
    apply_upsets,
    distinct_site_count,
    net_flipped_bits,
    sample_upsets,
    upset_site_histogram,
)


def test_distinct_sampling_gives_exactly_that_many_net_flips():
    rng = np.random.default_rng(4)
    block = np.linspace(-2.0, 2.0, 50, dtype=np.float32)
    upsets = sample_upsets(50, 32, 37, rng)
    assert distinct_site_count(upsets) == 37
    faulty = apply_upsets(block, upsets)
    assert net_flipped_bits(block, faulty) == 37


def test_repeated_sites_cancel():
    block = np.array([1.0, 2.0], dtype=np.float32)
    once = apply_upsets(block, [BitUpset(0, 30)])
    twice = apply_upsets(block, [BitUpset(0, 30), BitUpset(0, 30)])
    assert net_flipped_bits(block, once) == 1
    assert net_flipped_bits(block, twice) == 0
    assert np.array_equal(twice, block)


def test_sampling_without_replacement_cannot_exceed_the_population():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="allow_repeat=True"):
        sample_upsets(4, 8, 33, rng)
    # with replacement it is allowed, and sites repeat
    upsets = sample_upsets(4, 8, 200, rng, allow_repeat=True)
    assert len(upsets) == 200
    assert distinct_site_count(upsets) <= 32


def test_injection_does_not_mutate_the_original():
    block = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    copy = block.copy()
    apply_upsets(block, [BitUpset(1, 31)])
    assert np.array_equal(block, copy)


def test_injection_works_on_int8_and_float16():
    for dtype in (np.int8, np.float16, np.float64):
        block = np.array([1, 2, 3], dtype=dtype)
        faulty = apply_upsets(block, [BitUpset(0, 0)])
        assert faulty.dtype == block.dtype
        assert net_flipped_bits(block, faulty) == 1


def test_histogram_is_roughly_uniform_over_bit_positions():
    rng = np.random.default_rng(7)
    upsets = sample_upsets(2000, 32, 32_000, rng)
    hist = upset_site_histogram(upsets, 32)
    assert hist.sum() == 32_000
    # Expected 1000 per position; a multinomial with p = 1/32 and n = 32000 has
    # sigma = sqrt(n p (1-p)) = sqrt(32000 * (1/32) * (31/32)) = 31.1, so a
    # +-6 sigma window is 1000 +- 187.
    assert hist.min() > 1000 - 187
    assert hist.max() < 1000 + 187


def test_injection_input_validation():
    block = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match="element_index"):
        apply_upsets(block, [BitUpset(5, 0)])
    with pytest.raises(ValueError, match="bit_position"):
        apply_upsets(block, [BitUpset(0, 32)])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_elements"):
        sample_upsets(0, 32, 1, rng)
    with pytest.raises(ValueError, match="bits_per_element"):
        sample_upsets(4, 0, 1, rng)
    with pytest.raises(ValueError, match="count"):
        sample_upsets(4, 8, -1, rng)
    with pytest.raises(ValueError, match="share shape and dtype"):
        net_flipped_bits(block, np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="bits_per_element"):
        upset_site_histogram([], 0)


def test_zero_upsets_is_a_faithful_copy():
    block = np.linspace(0, 1, 11, dtype=np.float32)
    assert sample_upsets(11, 32, 0, np.random.default_rng(0)) == []
    assert np.array_equal(apply_upsets(block, []), block)


@given(
    count=st.integers(min_value=0, max_value=100),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=120, deadline=None)
def test_property_net_flips_equal_distinct_sites(count, seed):
    rng = np.random.default_rng(seed)
    block = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    upsets = sample_upsets(20, 32, count, rng)
    assert net_flipped_bits(block, apply_upsets(block, upsets)) == count
