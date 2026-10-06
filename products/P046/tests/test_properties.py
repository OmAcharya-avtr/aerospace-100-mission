"""Hypothesis property tests for the algebraic identities every construction obeys."""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from interleavekit import (
    BlockInterleaver,
    ConvolutionalInterleaver,
    HelicalInterleaver,
    SRandomInterleaver,
)
from interleavekit.metrics import (
    burst_dispersion,
    is_bijection,
    max_burst_fully_dispersed,
    minimum_spread,
    s_parameter,
    transmitted_span_profile,
)

SLOW = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

small = st.integers(min_value=1, max_value=12)
mid = st.integers(min_value=1, max_value=24)


# ------------------------------------------------- de-interleave o interleave = id


@SLOW
@given(depth=small, span=small, data=st.data())
def test_block_round_trip_is_the_identity(depth, span, data):
    il = BlockInterleaver(depth, span)
    x = np.asarray(
        data.draw(
            st.lists(
                st.integers(min_value=-1000, max_value=1000),
                min_size=il.length,
                max_size=il.length,
            )
        )
    )
    assert np.array_equal(il.deinterleave(il.interleave(x)), x)


@SLOW
@given(rows=small, cols=small, step=st.integers(min_value=0, max_value=12), data=st.data())
def test_helical_round_trip_is_the_identity(rows, cols, step, data):
    il = HelicalInterleaver(rows, cols, step)
    x = np.asarray(
        data.draw(
            st.lists(
                st.integers(min_value=-1000, max_value=1000),
                min_size=il.length,
                max_size=il.length,
            )
        )
    )
    assert np.array_equal(il.deinterleave(il.interleave(x)), x)


@SLOW
@given(
    registers=st.integers(min_value=1, max_value=8),
    slope=st.integers(min_value=0, max_value=4),
    extra=st.integers(min_value=1, max_value=40),
)
def test_convolutional_round_trip_is_the_identity(registers, slope, extra):
    ci = ConvolutionalInterleaver(registers, slope)
    n = ci.max_delay_symbols + extra
    x = np.arange(1, n + 1)
    assert np.array_equal(ci.deinterleave(ci.interleave(x)), x)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    length=st.integers(min_value=4, max_value=200),
    seed=st.integers(min_value=0, max_value=50),
)
def test_srandom_round_trip_is_the_identity(length, seed):
    # Capped well below the sqrt(length/2) rule of thumb: the search is bounded
    # and the achievable spread is measured in validation/validate_srandom.py.
    spread = max(1, min(4, int(np.sqrt(length / 2))))
    il = SRandomInterleaver(length, spread, seed=seed)
    x = np.arange(length) * 3 + 1
    assert np.array_equal(il.deinterleave(il.interleave(x)), x)


# ------------------------------------------------------------------- bijection


@SLOW
@given(depth=mid, span=mid)
def test_block_permutation_is_a_bijection(depth, span):
    assert is_bijection(BlockInterleaver(depth, span).permutation())


@SLOW
@given(rows=mid, cols=mid, step=st.integers(min_value=0, max_value=24))
def test_helical_permutation_is_a_bijection(rows, cols, step):
    assert is_bijection(HelicalInterleaver(rows, cols, step).permutation())


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(length=st.integers(min_value=1, max_value=200), seed=st.integers(0, 50))
def test_srandom_permutation_is_a_bijection(length, seed):
    spread = max(1, min(4, int(np.sqrt(length / 2))))
    assert is_bijection(SRandomInterleaver(length, spread, seed=seed).permutation())


@SLOW
@given(
    registers=st.integers(min_value=1, max_value=10),
    slope=st.integers(min_value=0, max_value=5),
    n=st.integers(min_value=1, max_value=60),
)
def test_convolutional_positions_are_injective(registers, slope, n):
    """Two source symbols can never occupy the same transmitted slot."""
    pos = ConvolutionalInterleaver(registers, slope).transmitted_position(np.arange(n))
    assert np.unique(pos).size == n


# ---------------------------------------------------------------- spread bounds


@SLOW
@given(depth=mid, span=mid)
def test_block_closed_form_spread_matches_the_metric(depth, span):
    il = BlockInterleaver(depth, span)
    assert il.minimum_spread_closed_form() == minimum_spread(il.permutation())


@SLOW
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(min_value=2, max_value=120))
def test_minimum_spread_exceeds_the_s_parameter(seed, n):
    """minimum_spread >= s_parameter + 1 for every permutation.

    If |i-j| >= S the spread is at least S + 1 because |delta pi| >= 1; if
    |i-j| < S then |delta pi| >= S by the definition of S, so the spread is at
    least S + 1 again.
    """
    pi = np.random.default_rng(seed).permutation(n)
    assert minimum_spread(pi) >= s_parameter(pi) + 1


@SLOW
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(min_value=2, max_value=120))
def test_minimum_spread_is_at_least_two_and_at_most_n(seed, n):
    pi = np.random.default_rng(seed).permutation(n)
    spread = minimum_spread(pi)
    assert 2 <= spread <= 2 * n


@SLOW
@given(
    length=st.integers(min_value=16, max_value=300),
    seed=st.integers(min_value=0, max_value=40),
)
def test_srandom_meets_its_requested_spread(length, seed):
    spread = max(1, min(4, int(np.sqrt(length / 2))))
    il = SRandomInterleaver(length, spread, seed=seed)
    assert s_parameter(il.permutation()) >= spread


# -------------------------------------------------------- burst metric identities


@SLOW
@given(depth=small, span=small)
def test_span_profile_is_non_decreasing(depth, span):
    pos = BlockInterleaver(depth, span).position_of_input()
    profile = transmitted_span_profile(pos, min(depth * span, 8))
    assert np.all(np.diff(profile) >= 0)


@SLOW
@given(depth=small, span=small, burst=st.integers(min_value=1, max_value=8))
def test_burst_dispersion_never_exceeds_the_burst_length(depth, span, burst):
    pos = BlockInterleaver(depth, span).position_of_input()
    burst = min(burst, depth * span)
    assert 1 <= burst_dispersion(pos, burst) <= burst


@SLOW
@given(rows=small, cols=small, step=st.integers(min_value=0, max_value=8))
def test_fully_dispersed_burst_is_exactly_the_minimum_adjacent_gap(rows, cols, step):
    pos = HelicalInterleaver(rows, cols, step).position_of_input()
    if pos.size < 2:
        return
    direct = int(np.abs(np.diff(pos)).min())
    assert max_burst_fully_dispersed(pos) == direct


@SLOW
@given(depth=small, span=small)
def test_burst_dispersion_is_one_up_to_the_fully_dispersed_length(depth, span):
    pos = BlockInterleaver(depth, span).position_of_input()
    limit = max_burst_fully_dispersed(pos)
    if limit >= 1:
        assert burst_dispersion(pos, limit) == 1
    if limit + 1 <= depth * span:
        assert burst_dispersion(pos, limit + 1) >= 2
