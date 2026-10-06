"""S-random interleaver: achieved spread, determinism and failure reporting."""

import numpy as np
import pytest

from interleavekit import SRandomInterleaver
from interleavekit.metrics import is_bijection, s_parameter


@pytest.mark.parametrize(
    "length,spread",
    [(64, 4), (128, 6), (256, 8), (512, 10)],
)
def test_achieved_s_parameter_meets_the_request(length, spread):
    """The constructed permutation satisfies the S-random condition it was asked for.

    Verified by the independent metric s_parameter(), which re-derives the
    condition from the permutation rather than trusting the search.

    The spreads here sit below the largest this bounded search reaches, which
    validation/validate_srandom.py measures as 6 at length 64 falling to 12 at
    length 1024 against a sqrt(length/2) rule of thumb of 22.6.
    """
    il = SRandomInterleaver(length, spread, seed=0)
    assert s_parameter(il.permutation()) >= spread


def test_permutation_is_a_bijection():
    assert is_bijection(SRandomInterleaver(300, 7, seed=3).permutation())


def test_round_trip_identity():
    rng = np.random.default_rng(4)
    il = SRandomInterleaver(200, 6, seed=1)
    x = rng.integers(0, 2**16, size=200)
    assert np.array_equal(il.deinterleave(il.interleave(x)), x)


def test_same_seed_gives_the_same_permutation():
    a = SRandomInterleaver(256, 8, seed=11).permutation()
    b = SRandomInterleaver(256, 8, seed=11).permutation()
    assert np.array_equal(a, b)


def test_different_seeds_give_different_permutations():
    a = SRandomInterleaver(256, 8, seed=11).permutation()
    b = SRandomInterleaver(256, 8, seed=12).permutation()
    assert not np.array_equal(a, b)


def test_permutation_returns_a_copy():
    il = SRandomInterleaver(32, 2, seed=0)
    first = il.permutation()
    first[0] = -99
    assert il.permutation()[0] != -99


def test_spread_one_is_a_plain_random_permutation():
    il = SRandomInterleaver(50, 1, seed=0)
    assert is_bijection(il.permutation())
    assert s_parameter(il.permutation()) >= 1


def test_length_one():
    il = SRandomInterleaver(1, 1, seed=0)
    assert il.permutation().tolist() == [0]


def test_spread_greater_than_length_raises():
    with pytest.raises(ValueError, match="cannot exceed length"):
        SRandomInterleaver(10, 11, seed=0)


def test_infeasible_spread_raises_with_an_actionable_message():
    """A spread far above sqrt(length/2) cannot be reached; the error says so.

    length = 64 gives sqrt(32) = 5.66 as the rule-of-thumb ceiling, so a request
    for spread 25 must fail with a message naming the ceiling rather than loop.
    """
    with pytest.raises(ValueError, match=r"S-random search failed .*sqrt\(length/2\) = 5.7"):
        SRandomInterleaver(64, 25, seed=0, max_attempts=2)


@pytest.mark.parametrize("bad", [0, -4])
def test_non_positive_parameters_raise(bad):
    with pytest.raises(ValueError, match=r">= 1"):
        SRandomInterleaver(bad, 1, seed=0)


def test_non_integer_seed_raises():
    with pytest.raises(TypeError, match="seed must be an integer"):
        SRandomInterleaver(32, 2, seed=1.5)


def test_cost_model_mentions_the_stored_index_table():
    cost = SRandomInterleaver(64, 4, seed=0).cost()
    assert cost.one_way_memory_symbols == 64
    assert cost.pair_memory_symbols == 128
    assert "index table" in cost.model


def test_attempts_used_is_reported():
    il = SRandomInterleaver(128, 5, seed=0)
    assert il.attempts_used >= 1


def test_repr():
    assert repr(SRandomInterleaver(16, 2, seed=5)) == (
        "SRandomInterleaver(length=16, spread=2, seed=5)"
    )
