"""Spread, dispersion and burst-dispersion metrics, with hand-computed cases."""

import numpy as np
import pytest

from interleavekit import BlockInterleaver, ConvolutionalInterleaver, HelicalInterleaver
from interleavekit.metrics import (
    MAX_DISPERSION_LENGTH,
    as_permutation,
    burst_dispersion,
    burst_dispersion_by_window_scan,
    burst_dispersion_profile,
    dispersion,
    is_bijection,
    longest_consecutive_run,
    max_burst_fully_dispersed,
    minimum_spread,
    s_parameter,
    transmitted_span_profile,
)

# ---------------------------------------------------------------- bijection


def test_is_bijection_accepts_a_permutation():
    assert is_bijection([2, 0, 1])


@pytest.mark.parametrize(
    "bad", [[0, 0, 1], [0, 1, 3], [-1, 0, 1], [], [[0, 1], [1, 0]], [0.0, 1.0]]
)
def test_is_bijection_rejects_non_permutations(bad):
    assert not is_bijection(np.asarray(bad))


def test_as_permutation_error_messages():
    with pytest.raises(ValueError, match="must be 1-D"):
        as_permutation(np.zeros((2, 2), dtype=int))
    with pytest.raises(ValueError, match="at least 1 element"):
        as_permutation(np.zeros(0, dtype=int))
    with pytest.raises(ValueError, match="integer dtype"):
        as_permutation(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="is not a permutation"):
        as_permutation(np.array([0, 0, 2]))


# ------------------------------------------------------------- minimum spread


def test_minimum_spread_hand_computed_4x4_block():
    """pi = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15].

    Pairs at index separation 1 have |delta pi| = 4 within a column (giving a
    spread of 5) and |delta pi| = 11 at a column boundary (spread 12). Pairs at
    index separation 4 have |delta pi| = 1 (spread 5). Nothing is smaller, so the
    minimum spread is 5.
    """
    assert minimum_spread(BlockInterleaver(4, 4).permutation()) == 5


def test_minimum_spread_hand_computed_2x2_block():
    """pi = [0, 2, 1, 3]: positions 1 and 2 give 1 + |1 - 2| = 2, the minimum."""
    assert minimum_spread(BlockInterleaver(2, 2).permutation()) == 2


def test_minimum_spread_of_the_identity_is_two():
    assert minimum_spread(np.arange(10)) == 2


def test_minimum_spread_of_the_reversal_is_two():
    """pi = [9, 8, ..., 0]: adjacent positions give 1 + 1 = 2."""
    assert minimum_spread(np.arange(9, -1, -1)) == 2


def test_minimum_spread_of_a_single_element_is_zero():
    assert minimum_spread(np.array([0])) == 0


def test_minimum_spread_matches_brute_force_on_random_permutations():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = int(rng.integers(2, 60))
        pi = rng.permutation(n)
        i, j = np.triu_indices(n, k=1)
        brute = int((np.abs(j - i) + np.abs(pi[j] - pi[i])).min())
        assert minimum_spread(pi) == brute


# -------------------------------------------------------------- S-parameter


def test_s_parameter_hand_computed_4x4_block():
    """pi = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15].

    For S = 4 the condition is: any two positions fewer than 4 apart must have
    source indices at least 4 apart. Separations 1, 2 and 3 give |delta pi| of
    4, 8 or 11 inside a column and 11, 7 or 3 across a boundary. The separation-3
    pair (3, 6) has pi = 12 and 9, |delta pi| = 3 < 4, so S = 4 fails and S = 3
    is the answer: separations 1 and 2 give |delta pi| of at least 4 and 3
    respectively.
    """
    assert s_parameter(BlockInterleaver(4, 4).permutation()) == 3


def test_s_parameter_of_the_identity_is_one():
    assert s_parameter(np.arange(20)) == 1


def test_s_parameter_bound_against_minimum_spread():
    """minimum_spread >= s_parameter + 1 for every permutation; proof in the docstring."""
    rng = np.random.default_rng(1)
    for _ in range(40):
        n = int(rng.integers(2, 80))
        pi = rng.permutation(n)
        assert minimum_spread(pi) >= s_parameter(pi) + 1


# --------------------------------------------------------------- dispersion


def test_dispersion_hand_computed_identity():
    """For the identity of length n the displacement pairs are (d, d) for
    d = 1..n-1, so there are n-1 distinct pairs out of n(n-1)/2.

    n = 5: 4 distinct out of 10, so 0.4.
    """
    assert dispersion(np.arange(5)) == pytest.approx(4 / 10)


def test_dispersion_hand_computed_2x2_block():
    """pi = [0, 2, 1, 3]. The six (delta i, delta pi) pairs are
    (1,2) (2,1) (3,3) (1,-1) (2,1) (1,2), of which (2,1) and (1,2) each repeat
    once, leaving 4 distinct out of 6.
    """
    assert dispersion(BlockInterleaver(2, 2).permutation()) == pytest.approx(4 / 6)


def test_dispersion_unnormalised():
    assert dispersion(np.arange(5), normalise=False) == 4.0


def test_dispersion_of_a_single_element_is_zero():
    assert dispersion(np.array([0])) == 0.0


def test_dispersion_is_in_the_unit_interval():
    rng = np.random.default_rng(2)
    for n in (4, 17, 64, 129):
        d = dispersion(rng.permutation(n))
        assert 0.0 < d <= 1.0


def test_dispersion_rejects_oversized_inputs():
    n = MAX_DISPERSION_LENGTH + 1
    with pytest.raises(ValueError, match=r"capped at N = 2048"):
        dispersion(np.arange(n))


# ------------------------------------------------------ longest consecutive run


@pytest.mark.parametrize(
    "indices,expected",
    [
        ([], 0),
        ([7], 1),
        ([5, 1, 2, 9, 3], 3),
        ([0, 1, 2, 3], 4),
        ([0, 2, 4, 6], 1),
        ([4, 4, 5, 5, 6], 3),
    ],
)
def test_longest_consecutive_run(indices, expected):
    assert longest_consecutive_run(indices) == expected


# ---------------------------------------------------------- burst dispersion


def test_burst_dispersion_hand_computed_4x4_block():
    """4x4 block, position_of_input = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15].

    Source symbols 0 and 1 sit at transmitted positions 0 and 4, four apart, and
    every other adjacent source pair is at least four apart except across the
    column boundary, where the separation is 11. So a burst of 4 consecutive
    transmitted symbols can never catch two adjacent source symbols: the worst
    surviving run is 1. A burst of 5, e.g. positions 0..4, catches source 0 and
    source 1, giving a surviving run of 2.
    """
    pos = BlockInterleaver(4, 4).position_of_input()
    assert burst_dispersion(pos, 1) == 1
    assert burst_dispersion(pos, 4) == 1
    assert burst_dispersion(pos, 5) == 2
    assert max_burst_fully_dispersed(pos) == 4


def test_burst_dispersion_of_the_identity_is_the_burst_length():
    """With no interleaving a burst of L damages L consecutive source symbols."""
    pos = np.arange(32)
    for length in (1, 2, 7, 16):
        assert burst_dispersion(pos, length) == length


def test_block_depth_equals_the_dispersed_burst_only_when_span_is_at_least_three():
    """Measured, not assumed: at span 2 the wrap-around pair costs one symbol.

    At span 2 the last column's source index q = span - 1 is adjacent to the next
    row's q = 0, and those two sit only depth - 1 transmitted positions apart, so
    the largest fully dispersed burst is depth - 1, not depth.
    """
    assert max_burst_fully_dispersed(BlockInterleaver(16, 2).position_of_input()) == 15
    assert max_burst_fully_dispersed(BlockInterleaver(16, 3).position_of_input()) == 16
    assert max_burst_fully_dispersed(BlockInterleaver(16, 8).position_of_input()) == 16


def test_burst_dispersion_is_monotonically_non_decreasing():
    pos = HelicalInterleaver(8, 8, 3).position_of_input()
    profile = burst_dispersion_profile(pos, np.arange(1, 33))
    assert np.all(np.diff(profile) >= 0)


def test_burst_dispersion_profile_matches_pointwise_calls():
    pos = BlockInterleaver(6, 7).position_of_input()
    lengths = np.array([1, 3, 6, 7, 12, 20])
    profile = burst_dispersion_profile(pos, lengths)
    assert profile.tolist() == [burst_dispersion(pos, int(b)) for b in lengths]


def test_burst_dispersion_profile_of_empty_lengths():
    pos = BlockInterleaver(2, 2).position_of_input()
    assert burst_dispersion_profile(pos, []).size == 0


def test_fast_path_matches_the_definitional_window_scan():
    """The O(N) reformulation must agree with the O(N*L) definition everywhere."""
    for depth in range(1, 7):
        for span in range(1, 7):
            pos = BlockInterleaver(depth, span).position_of_input()
            for length in range(1, min(depth * span, 10) + 1):
                assert burst_dispersion(pos, length) == burst_dispersion_by_window_scan(
                    pos, length
                )


def test_fast_path_matches_the_window_scan_for_the_convolutional_interleaver():
    conv = ConvolutionalInterleaver(4, 2)
    n = conv.max_delay_symbols + 40
    pos = conv.transmitted_position(np.arange(n))
    window = conv.steady_state_range(n)
    for length in range(1, 15):
        assert burst_dispersion(pos, length, window) == burst_dispersion_by_window_scan(
            pos, length, window
        )


def test_burst_dispersion_honours_the_window_range():
    """Restricting the window cannot make the worst case worse."""
    pos = BlockInterleaver(8, 8).position_of_input()
    full = burst_dispersion(pos, 10)
    narrow = burst_dispersion(pos, 10, (20, 50))
    assert narrow <= full


def test_burst_longer_than_the_window_raises():
    pos = BlockInterleaver(4, 4).position_of_input()
    with pytest.raises(ValueError, match=r"exceeds the 10 transmitted positions"):
        burst_dispersion(pos, 11, (0, 10))


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_burst_length_raises(bad):
    pos = BlockInterleaver(4, 4).position_of_input()
    with pytest.raises(ValueError, match=r"burst_length must be >= 1"):
        burst_dispersion(pos, bad)


def test_non_integer_burst_length_raises():
    pos = BlockInterleaver(4, 4).position_of_input()
    with pytest.raises(TypeError, match="burst_length must be an integer"):
        burst_dispersion(pos, 2.0)


def test_repeated_positions_raise():
    with pytest.raises(ValueError, match="repeated transmitted positions"):
        burst_dispersion(np.array([0, 1, 1, 2]), 2)


def test_empty_positions_raise():
    with pytest.raises(ValueError, match="at least 1 element"):
        burst_dispersion(np.zeros(0, dtype=np.int64), 1)


# ------------------------------------------------------ transmitted span profile


def test_transmitted_span_profile_hand_computed_4x4_block():
    """position_of_input = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15].

    m(1) = 0 by definition. m(2) is the narrowest window holding two adjacent
    source symbols: source 0 and 1 are at 0 and 4, so m(2) = 4. m(3) needs three
    adjacent source symbols: 0, 1, 2 sit at 0, 4, 8, a width of 8, and no triple
    does better, so m(3) = 8.
    """
    pos = BlockInterleaver(4, 4).position_of_input()
    assert transmitted_span_profile(pos, 3).tolist() == [0, 4, 8]


def test_transmitted_span_profile_is_non_decreasing():
    pos = HelicalInterleaver(7, 9, 2).position_of_input()
    profile = transmitted_span_profile(pos, 20)
    assert np.all(np.diff(profile) >= 0)


def test_transmitted_span_profile_of_the_identity():
    """For the identity, m(R) = R - 1."""
    assert transmitted_span_profile(np.arange(10), 5).tolist() == [0, 1, 2, 3, 4]


def test_transmitted_span_profile_sentinel_beyond_the_available_run():
    sentinel = np.iinfo(np.int64).max
    profile = transmitted_span_profile(np.array([0, 1, 2]), 5)
    assert profile[:3].tolist() == [0, 1, 2]
    assert profile[3] == sentinel
    assert profile[4] == sentinel


def test_transmitted_span_profile_rejects_bad_max_run():
    with pytest.raises(ValueError, match=r"max_run must be >= 1"):
        transmitted_span_profile(np.arange(4), 0)


def test_max_burst_fully_dispersed_equals_the_minimum_adjacent_separation():
    """max_burst_fully_dispersed is m(2), the smallest transmitted gap between
    source symbols j and j + 1. Checked against a direct computation."""
    for il in [
        BlockInterleaver(9, 5),
        HelicalInterleaver(6, 11, 3),
        BlockInterleaver(1, 12),
    ]:
        pos = il.position_of_input()
        direct = int(np.abs(np.diff(pos)).min())
        assert max_burst_fully_dispersed(pos) == direct


def test_max_burst_fully_dispersed_respects_max_search():
    pos = BlockInterleaver(32, 8).position_of_input()
    assert max_burst_fully_dispersed(pos) == 32
    assert max_burst_fully_dispersed(pos, None, 10) == 10
