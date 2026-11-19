"""Brackets from independent mathematics, and the conditioning of their slack."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from passyield.bounds import (
    MACHINE_EPSILON,
    SLACK_OPERATION_COUNT,
    bracket_slack_mbit,
    capacity_bracket,
    inside,
    shortfall_bracket,
)
from passyield.decompose import decompose_pass
from passyield.link import DEFAULT_BUDGET


def test_shortfall_bracket_hand_calculation():
    """Hand-calculated: ``Y_nom = 1000`` Mbit, ``T = 200`` s, ``R_max = 250`` Mbit/s.

    ``R_max * T = 50000`` Mbit, so the bracket is
    ``[1000 - 50000, 1000] = [-49000, 1000]`` before the slack. The slack is
    ``8 * eps * max(1000, 50000) = 8 * 2.22e-16 * 50000 = 8.88e-11`` Mbit, so
    the bracket is ``[-49000.0000000001, 1000.0000000001]`` to that order.
    """
    low, high = shortfall_bracket(1000.0, 200.0, 250.0)
    assert low == pytest.approx(-49000.0, rel=0.0, abs=1.0e-9)
    assert high == pytest.approx(1000.0, rel=0.0, abs=1.0e-9)
    assert low < -49000.0
    assert high > 1000.0


def test_capacity_bracket_hand_calculation():
    """Hand-calculated: ``T = 200`` s, ``R_min = 20``, ``R_max = 250`` Mbit/s.

    The mean-value inequality gives ``[20*200, 250*200] = [4000, 50000]``
    Mbit, widened by the slack.
    """
    low, high = capacity_bracket(200.0, 20.0, 250.0)
    assert low == pytest.approx(4000.0, rel=0.0, abs=1.0e-9)
    assert high == pytest.approx(50000.0, rel=0.0, abs=1.0e-9)


def test_slack_follows_the_operands_and_not_the_result():
    """The slack scales with the magnitude of the bound expression's operands.

    A slack floored at an absolute constant makes a tight bracket vacuous: a
    previous product of this mission had exactly that falsified, with a
    bracket of width 4.5e-23 admitted by a slack floored at 1.8e-15. Here the
    slack of two operands of magnitude ``M`` is ``8 * eps * M`` whatever
    their difference comes to.
    """
    small = bracket_slack_mbit(1.0, 1.0 + 1.0e-12)
    large = bracket_slack_mbit(1.0e9, 1.0e9 + 1.0)
    assert small == pytest.approx(SLACK_OPERATION_COUNT * MACHINE_EPSILON * (1.0 + 1e-12),
                                  rel=1.0e-12, abs=0.0)
    assert large / small == pytest.approx(1.0e9, rel=1.0e-6, abs=0.0)


def test_slack_of_no_operands_is_zero():
    """A bound with no operands needs no slack."""
    assert bracket_slack_mbit() == 0.0


def test_inside_is_inclusive():
    """The bracket is inclusive at both ends."""
    assert inside(1.0, (1.0, 2.0))
    assert inside(2.0, (1.0, 2.0))
    assert not inside(0.999, (1.0, 2.0))


@pytest.mark.parametrize(
    ("nominal", "duration", "ceiling"),
    [(1000.0, -1.0, 250.0), (1000.0, 200.0, 0.0), (1000.0, 200.0, -5.0)],
)
def test_shortfall_bracket_rejects_out_of_range_inputs(nominal, duration, ceiling):
    """Input validation."""
    with pytest.raises(ValueError):
        shortfall_bracket(nominal, duration, ceiling)


@pytest.mark.parametrize(
    ("duration", "minimum", "maximum"),
    [(-1.0, 10.0, 20.0), (10.0, -1.0, 20.0), (10.0, 30.0, 20.0)],
)
def test_capacity_bracket_rejects_out_of_range_inputs(duration, minimum, maximum):
    """Input validation."""
    with pytest.raises(ValueError):
        capacity_bracket(duration, minimum, maximum)


def test_real_decomposition_lies_inside_both_brackets(satrec, station, long_window, events):
    """A measured pass sits inside both independent brackets.

    Neither bracket uses the composite Simpson quadrature, the cumulative
    rate integral or any distribution, so this is a check of a different kind
    from "the terms sum to the shortfall": it can catch a unit slip or a sign
    error that the identity cannot.
    """
    item = decompose_pass(satrec, station, long_window, events)
    from passyield.decompose import rate_at_offsets

    offsets = np.linspace(0.0, long_window.duration_s, 257)
    rates, _clamp = rate_at_offsets(satrec, station, long_window, offsets)
    assert inside(
        item.shortfall_bits,
        shortfall_bracket(
            item.nominal_bits, item.window_duration_s, DEFAULT_BUDGET.max_rate_mbit_s
        ),
    )
    assert inside(
        item.capacity_bits,
        capacity_bracket(item.window_duration_s, float(rates.min()), float(rates.max())),
    )


def test_negative_control_a_doubled_shortfall_leaves_the_bracket(
    satrec, station, long_window, events
):
    """Negative control, anchored on the TRUE value and verified to FIRE.

    The control is not an offset applied to an output: the true shortfall is
    at most ``Y_nom`` by the bracket's own upper end, so a value of
    ``Y_nom + R_max * T`` is outside it by construction and the control must
    fire. An offset small enough to leave the value inside the bracket would
    read as evidence and prove nothing.
    """
    item = decompose_pass(satrec, station, long_window, events)
    bracket = shortfall_bracket(
        item.nominal_bits, item.window_duration_s, DEFAULT_BUDGET.max_rate_mbit_s
    )
    injected = item.nominal_bits + DEFAULT_BUDGET.max_rate_mbit_s * item.window_duration_s
    assert not inside(injected, bracket), "the negative control did not fire"


def test_negative_control_a_sign_flip_leaves_the_bracket(satrec, station, long_window, events):
    """Negative control: the capacity with a flipped sign is outside its bracket.

    The capacity bracket's lower end is ``R_min * T > 0``, so ``-Y_cap`` is
    outside it whatever the quadrature did. Verified to fire.
    """
    item = decompose_pass(satrec, station, long_window, events)
    from passyield.decompose import rate_at_offsets

    offsets = np.linspace(0.0, long_window.duration_s, 257)
    rates, _clamp = rate_at_offsets(satrec, station, long_window, offsets)
    bracket = capacity_bracket(
        item.window_duration_s, float(rates.min()), float(rates.max())
    )
    assert not inside(-item.capacity_bits, bracket), "the negative control did not fire"


def test_negative_control_a_bit_versus_mbit_slip_leaves_the_bracket(
    satrec, station, long_window, events
):
    """Negative control: a factor of 1e6 unit slip is outside the bracket.

    This is the error class the bracket exists for: a Mbit quantity reported
    in bit would move the convolution and the Monte Carlo together, so no
    agreement test could see it. Verified to fire.
    """
    item = decompose_pass(satrec, station, long_window, events)
    bracket = shortfall_bracket(
        item.nominal_bits, item.window_duration_s, DEFAULT_BUDGET.max_rate_mbit_s
    )
    assert not inside(item.shortfall_bits * 1.0e6, bracket), "the negative control did not fire"


@given(
    nominal=st.floats(min_value=0.0, max_value=1.0e6),
    duration=st.floats(min_value=1.0e-3, max_value=1.0e4),
)
@settings(max_examples=200, deadline=None)
def test_shortfall_bracket_always_contains_the_endpoints_it_is_built_from(nominal, duration):
    """Property: the bracket contains ``Y_nom`` and ``Y_nom - R_max*T``.

    The two extreme realised yields -- nothing delivered and the ceiling
    delivered for the whole window -- are the ends of the bracket, so the
    bracket must contain both. If it did not, the slack would have the wrong
    sign.
    """
    ceiling = DEFAULT_BUDGET.max_rate_mbit_s
    bracket = shortfall_bracket(nominal, duration, ceiling)
    assert inside(nominal, bracket)
    assert inside(nominal - ceiling * duration, bracket)
