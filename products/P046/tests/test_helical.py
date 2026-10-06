"""Helical interleaver, including hand-computed permutations."""

import numpy as np
import pytest

from interleavekit import HelicalInterleaver
from interleavekit.metrics import is_bijection


def test_helical_3x3_step1_hand_computed():
    """rows = 3, columns = 3, step = 1, N = 9.

    Source array, written row by row:

        row 0:  0  1  2
        row 1:  3  4  5
        row 2:  6  7  8

    Read rule: transmitted position i takes cell (row, col) with
        col = i % columns
        row = (i // columns + step * col) mod rows

        i = 0: col 0, row (0 + 0) % 3 = 0 -> cell (0,0) = 0
        i = 1: col 1, row (0 + 1) % 3 = 1 -> cell (1,1) = 4
        i = 2: col 2, row (0 + 2) % 3 = 2 -> cell (2,2) = 8
        i = 3: col 0, row (1 + 0) % 3 = 1 -> cell (1,0) = 3
        i = 4: col 1, row (1 + 1) % 3 = 2 -> cell (2,1) = 7
        i = 5: col 2, row (1 + 2) % 3 = 0 -> cell (0,2) = 2
        i = 6: col 0, row (2 + 0) % 3 = 2 -> cell (2,0) = 6
        i = 7: col 1, row (2 + 1) % 3 = 0 -> cell (0,1) = 1
        i = 8: col 2, row (2 + 2) % 3 = 1 -> cell (1,2) = 5

        pi = [0, 4, 8, 3, 7, 2, 6, 1, 5]
    """
    assert HelicalInterleaver(3, 3, 1).permutation().tolist() == [0, 4, 8, 3, 7, 2, 6, 1, 5]


def test_helical_2x4_step1_hand_computed():
    """rows = 2, columns = 4, step = 1, N = 8.

        source array      read: col = i % 4, row = (i // 4 + col) mod 2
          0 1 2 3           i=0: col 0, row 0 -> 0
          4 5 6 7           i=1: col 1, row 1 -> 5
                            i=2: col 2, row 0 -> 2
                            i=3: col 3, row 1 -> 7
                            i=4: col 0, row 1 -> 4
                            i=5: col 1, row 0 -> 1
                            i=6: col 2, row 1 -> 6
                            i=7: col 3, row 0 -> 3

        pi = [0, 5, 2, 7, 4, 1, 6, 3]
    """
    assert HelicalInterleaver(2, 4, 1).permutation().tolist() == [0, 5, 2, 7, 4, 1, 6, 3]


def test_step_zero_is_plain_row_read_identity():
    assert HelicalInterleaver(4, 4, 0).permutation().tolist() == list(range(16))


def test_bijection_for_every_step_including_shared_factors():
    """A step sharing a factor with rows still gives a bijection."""
    for rows in range(1, 13):
        for step in range(0, 13):
            pi = HelicalInterleaver(rows, 6, step).permutation()
            assert is_bijection(pi), (rows, step)


def test_round_trip_identity():
    rng = np.random.default_rng(1)
    for rows, cols, step in [(1, 1, 1), (1, 8, 3), (8, 1, 2), (5, 7, 1), (6, 6, 4), (3, 9, 0)]:
        il = HelicalInterleaver(rows, cols, step)
        x = rng.integers(0, 1000, size=il.length)
        assert np.array_equal(il.deinterleave(il.interleave(x)), x)


def test_rows_one_or_columns_one():
    assert HelicalInterleaver(1, 5, 3).permutation().tolist() == [0, 1, 2, 3, 4]
    assert HelicalInterleaver(5, 1, 1).permutation().tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("rows,cols", [(0, 3), (3, 0), (-2, 3)])
def test_non_positive_dimensions_raise(rows, cols):
    with pytest.raises(ValueError, match=r">= 1 symbol"):
        HelicalInterleaver(rows, cols, 1)


def test_negative_step_raises():
    with pytest.raises(ValueError, match=r"step must be >= 0 rows"):
        HelicalInterleaver(4, 4, -1)


def test_non_integer_step_raises():
    with pytest.raises(TypeError, match="step must be an integer"):
        HelicalInterleaver(4, 4, 1.5)


def test_cost_matches_the_array_size():
    il = HelicalInterleaver(7, 5, 2)
    cost = il.cost()
    assert cost.one_way_memory_symbols == 35
    assert cost.pair_memory_symbols == 70
    assert cost.pair_latency_symbols == 70


def test_repr():
    assert repr(HelicalInterleaver(2, 3, 4)) == (
        "HelicalInterleaver(rows=2, columns=3, step=4)"
    )
