"""Convolutional interleaver, including hand-computed output positions."""

import numpy as np
import pytest

from interleavekit import ConvolutionalInterleaver


def test_register_delays_hand_computed():
    """registers = 4, slope = 3: register k holds k * slope symbols.

        register 0:  0 symbols
        register 1:  3 symbols
        register 2:  6 symbols
        register 3:  9 symbols
    """
    assert ConvolutionalInterleaver(4, 3).register_delays().tolist() == [0, 3, 6, 9]


def test_transmitted_positions_hand_computed_registers3_slope1():
    """registers = 3, slope = 1. Hand-computed transmitted position of each source symbol.

    Register k holds k * slope = k symbols, and a register is clocked once every
    3 symbol times, so a symbol in register k waits k * slope * registers = 3k
    symbol times. The commutator gives source symbol i the register i % 3:

        i = 0: register 0, delay 0 * 1 * 3 = 0  ->  position 0 + 0  =  0
        i = 1: register 1, delay 1 * 1 * 3 = 3  ->  position 1 + 3  =  4
        i = 2: register 2, delay 2 * 1 * 3 = 6  ->  position 2 + 6  =  8
        i = 3: register 0, delay 0            ->  position 3 + 0  =  3
        i = 4: register 1, delay 3            ->  position 4 + 3  =  7
        i = 5: register 2, delay 6            ->  position 5 + 6  = 11
        i = 6: register 0, delay 0            ->  position 6
        i = 7: register 1, delay 3            ->  position 10
        i = 8: register 2, delay 6            ->  position 14

    so the first nine transmitted positions are [0, 4, 8, 3, 7, 11, 6, 10, 14].
    """
    ci = ConvolutionalInterleaver(registers=3, slope=1)
    got = ci.transmitted_position(np.arange(9)).tolist()
    assert got == [0, 4, 8, 3, 7, 11, 6, 10, 14]


def test_transmitted_positions_hand_computed_registers2_slope2():
    """registers = 2, slope = 2: delay of register 1 is 1 * 2 * 2 = 4 symbol times.

        i = 0: register 0, delay 0 -> 0
        i = 1: register 1, delay 4 -> 5
        i = 2: register 0, delay 0 -> 2
        i = 3: register 1, delay 4 -> 7
        i = 4: register 0           -> 4
        i = 5: register 1           -> 9
    """
    ci = ConvolutionalInterleaver(registers=2, slope=2)
    assert ci.transmitted_position(np.arange(6)).tolist() == [0, 5, 2, 7, 4, 9]


def test_interleaved_stream_hand_computed_registers3_slope1():
    """Source 1..12 through registers = 3, slope = 1, fill 0.

    From the position table above, source symbol i (value i + 1) lands at
    transmitted position i + (i % 3) * 3. Transmitted positions 1, 2, 5, 12, 15
    and 16 are not reached by any of the twelve source symbols within the
    18-symbol stream, so they carry the fill value 0:

        position : 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
        source i : 0  -  -  3  1  -  6  4  2  9  7  5  -  10  8  -  -  11
        value    : 1  0  0  4  2  0  7  5  3 10  8  6  0  11  9  0  0  12
    """
    ci = ConvolutionalInterleaver(registers=3, slope=1)
    out = ci.interleave(np.arange(1, 13))
    assert out.tolist() == [1, 0, 0, 4, 2, 0, 7, 5, 3, 10, 8, 6, 0, 11, 9, 0, 0, 12]


def test_pair_latency_matches_the_derived_constant():
    """pair latency = registers * slope * (registers - 1), derived in the module docstring.

    registers = 4, slope = 3: 4 * 3 * 3 = 36 symbol times.
    """
    assert ConvolutionalInterleaver(4, 3).cost().pair_latency_symbols == 36


def test_one_way_memory_is_the_sum_of_register_lengths():
    """slope * R * (R - 1) / 2, i.e. sum of k * slope for k = 0..R-1.

    registers = 4, slope = 3: 0 + 3 + 6 + 9 = 18 symbols.
    """
    cost = ConvolutionalInterleaver(4, 3).cost()
    assert cost.one_way_memory_symbols == 18
    assert cost.pair_memory_symbols == 36


def test_end_to_end_delay_is_the_same_for_every_symbol():
    """The pair delay is constant, which is what makes the de-interleaver usable."""
    for registers, slope in [(2, 1), (3, 1), (4, 2), (5, 3), (8, 1)]:
        ci = ConvolutionalInterleaver(registers, slope)
        n = 4 * ci.max_delay_symbols + 20
        x = np.arange(1, n + 1)
        assert np.array_equal(ci.deinterleave(ci.interleave(x)), x)


def test_round_trip_identity():
    rng = np.random.default_rng(2)
    for registers, slope in [(1, 1), (2, 0), (2, 1), (3, 2), (6, 1), (4, 5)]:
        ci = ConvolutionalInterleaver(registers, slope)
        n = ci.max_delay_symbols + 30
        x = rng.integers(0, 500, size=n)
        assert np.array_equal(ci.deinterleave(ci.interleave(x)), x)


def test_slope_zero_and_single_register_are_the_identity():
    x = np.arange(1, 11)
    assert ConvolutionalInterleaver(4, 0).interleave(x).tolist() == x.tolist()
    assert ConvolutionalInterleaver(1, 7).interleave(x).tolist() == x.tolist()


def test_steady_state_range():
    ci = ConvolutionalInterleaver(4, 2)
    assert ci.max_delay_symbols == 3 * 2 * 4
    assert ci.steady_state_range(100) == (24, 100)


def test_transmitted_length():
    ci = ConvolutionalInterleaver(4, 2)
    assert ci.transmitted_length(100) == 124
    assert ci.interleave(np.arange(100)).size == 124


@pytest.mark.parametrize("registers,slope", [(0, 1), (-2, 1)])
def test_bad_registers_raise(registers, slope):
    with pytest.raises(ValueError, match=r"registers must be >= 1"):
        ConvolutionalInterleaver(registers, slope)


def test_negative_slope_raises():
    with pytest.raises(ValueError, match=r"slope must be >= 0 symbols"):
        ConvolutionalInterleaver(4, -1)


def test_non_integer_parameters_raise():
    with pytest.raises(TypeError, match="slope must be an integer"):
        ConvolutionalInterleaver(4, 1.5)
    with pytest.raises(TypeError, match="registers must be an integer"):
        ConvolutionalInterleaver("4", 1)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="at least 1 symbol"):
        ConvolutionalInterleaver(3, 1).interleave(np.zeros(0))


def test_two_dimensional_input_raises():
    with pytest.raises(ValueError, match="1-D stream"):
        ConvolutionalInterleaver(3, 1).interleave(np.zeros((4, 4)))


def test_stream_too_short_to_recover_raises():
    ci = ConvolutionalInterleaver(4, 2)  # max_delay_symbols == 24
    with pytest.raises(ValueError, match=r"too short to recover anything"):
        ci.deinterleave(np.zeros(10))


def test_negative_source_index_raises():
    with pytest.raises(ValueError, match=r"source_index must be >= 0"):
        ConvolutionalInterleaver(3, 1).transmitted_position(np.array([-1, 0]))


def test_repr():
    assert repr(ConvolutionalInterleaver(3, 2)) == (
        "ConvolutionalInterleaver(registers=3, slope=2)"
    )
