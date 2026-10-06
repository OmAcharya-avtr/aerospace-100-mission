"""Block interleaver, including hand-computed permutations."""

import numpy as np
import pytest

from interleavekit import BlockInterleaver
from interleavekit.metrics import is_bijection, minimum_spread


def test_block_4x4_hand_computed_permutation():
    """Depth 4, span 4, N = 16. Permutation worked out by hand.

    Write the source indices into a 4-row by 4-column array, row by row:

        row 0:   0  1  2  3
        row 1:   4  5  6  7
        row 2:   8  9 10 11
        row 3:  12 13 14 15

    Read it out column by column, top to bottom, left to right:

        column 0:  0,  4,  8, 12
        column 1:  1,  5,  9, 13
        column 2:  2,  6, 10, 14
        column 3:  3,  7, 11, 15

    so the transmitted sequence carries source indices

        pi = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]

    Spot-checking the closed form pi[i] = (i % 4) * 4 + (i // 4):
        i = 0  ->  0 * 4 + 0 =  0
        i = 1  ->  1 * 4 + 0 =  4
        i = 5  ->  1 * 4 + 1 =  5
        i = 11 ->  3 * 4 + 2 = 14
        i = 15 ->  3 * 4 + 3 = 15
    """
    expected = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
    assert BlockInterleaver(depth=4, span=4).permutation().tolist() == expected


def test_block_2x3_hand_computed_permutation():
    """Depth 2, span 3, N = 6. Non-square, so the two parameters cannot be confused.

        write (2 rows, 3 columns)      read column by column
            0  1  2                      column 0: 0, 3
            3  4  5                      column 1: 1, 4
                                         column 2: 2, 5

        pi = [0, 3, 1, 4, 2, 5]

    Check against pi[i] = (i % 2) * 3 + (i // 2):
        i = 0 -> 0 + 0 = 0
        i = 1 -> 3 + 0 = 3
        i = 2 -> 0 + 1 = 1
        i = 3 -> 3 + 1 = 4
        i = 4 -> 0 + 2 = 2
        i = 5 -> 3 + 2 = 5
    """
    assert BlockInterleaver(depth=2, span=3).permutation().tolist() == [0, 3, 1, 4, 2, 5]


def test_block_3x2_is_the_transpose_of_2x3():
    """Exchanging depth and span inverts the permutation.

    3 rows by 2 columns:
        write     read columns
         0  1      column 0: 0, 2, 4
         2  3      column 1: 1, 3, 5
         4  5
        pi = [0, 2, 4, 1, 3, 5]

    which is exactly position_of_input for depth 2, span 3.
    """
    a = BlockInterleaver(depth=3, span=2)
    b = BlockInterleaver(depth=2, span=3)
    assert a.permutation().tolist() == [0, 2, 4, 1, 3, 5]
    assert a.permutation().tolist() == b.position_of_input().tolist()


def test_closed_form_inverse_matches_argsort():
    for depth in range(1, 9):
        for span in range(1, 9):
            il = BlockInterleaver(depth, span)
            assert np.array_equal(il.position_of_input(), np.argsort(il.permutation()))


def test_permutation_is_a_bijection():
    for depth in (1, 2, 5, 7, 16):
        for span in (1, 3, 4, 11):
            assert is_bijection(BlockInterleaver(depth, span).permutation())


def test_round_trip_identity():
    rng = np.random.default_rng(0)
    for depth, span in [(1, 1), (1, 7), (7, 1), (4, 4), (3, 11), (16, 5)]:
        il = BlockInterleaver(depth, span)
        x = rng.integers(0, 256, size=il.length)
        assert np.array_equal(il.deinterleave(il.interleave(x)), x)


def test_round_trip_preserves_trailing_axes_and_dtype():
    il = BlockInterleaver(3, 4)
    x = np.arange(12 * 2, dtype=np.float32).reshape(12, 2)
    out = il.interleave(x)
    assert out.shape == (12, 2)
    assert out.dtype == np.float32
    assert np.array_equal(il.deinterleave(out), x)


def test_depth_one_and_span_one_are_the_identity():
    assert BlockInterleaver(1, 6).permutation().tolist() == [0, 1, 2, 3, 4, 5]
    assert BlockInterleaver(6, 1).permutation().tolist() == [0, 1, 2, 3, 4, 5]
    assert BlockInterleaver(1, 1).permutation().tolist() == [0]


def test_minimum_spread_closed_form_matches_brute_force():
    for depth in range(1, 15):
        for span in range(1, 15):
            il = BlockInterleaver(depth, span)
            assert il.minimum_spread_closed_form() == minimum_spread(il.permutation())


def test_minimum_spread_closed_form_at_length_one():
    assert BlockInterleaver(1, 1).minimum_spread_closed_form() == 0


@pytest.mark.parametrize("depth,span", [(0, 4), (4, 0), (-1, 4), (4, -3)])
def test_non_positive_parameters_raise_value_error(depth, span):
    with pytest.raises(ValueError, match=r">= 1 symbol"):
        BlockInterleaver(depth, span)


@pytest.mark.parametrize("bad", [4.0, "4", None, True])
def test_non_integer_parameters_raise_type_error(bad):
    with pytest.raises(TypeError, match="must be an integer"):
        BlockInterleaver(bad, 4)


def test_wrong_block_length_names_both_lengths():
    il = BlockInterleaver(4, 4)
    with pytest.raises(ValueError, match=r"expected exactly 16 symbols .* got 17"):
        il.interleave(np.zeros(17))


def test_length_a_multiple_of_the_block_suggests_splitting():
    il = BlockInterleaver(4, 4)
    with pytest.raises(ValueError, match=r"split the input into 3 blocks of 16"):
        il.interleave(np.zeros(48))


def test_empty_input_raises():
    il = BlockInterleaver(2, 2)
    with pytest.raises(ValueError, match=r"got 0"):
        il.interleave(np.zeros(0))


def test_scalar_input_raises():
    with pytest.raises(ValueError, match="got a scalar"):
        BlockInterleaver(2, 2).interleave(np.float64(1.0))


def test_len_and_repr():
    il = BlockInterleaver(3, 5)
    assert len(il) == 15
    assert repr(il) == "BlockInterleaver(depth=3, span=5)"
