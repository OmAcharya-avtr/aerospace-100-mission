"""Property-based tests over the algebraic identities this package relies on.

Every counterexample Hypothesis finds here is pinned as an explicit named
regression test in ``tests/test_regressions.py`` with its literal inputs,
because ``.hypothesis/`` is gitignored and an example found by one run is lost
on a fresh clone.

No test in this file asserts bit-exact equality of a floating-point quantity,
and none asserts equality of a boolean produced by a strict threshold
comparison. What is asserted is agreement to a two-sided tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from downlinkpick.contacts import (
    ContactPlan,
    ContactWindow,
    constant_rate_profile,
    rate_profile_from_elevation,
)
from downlinkpick.exact import SOLVER_ABS_GAP_VALUE_UNITS, enumerate_select, exact_select
from downlinkpick.greedy import dantzig_bound, greedy_select
from downlinkpick.inventory import DataProduct, Inventory
from downlinkpick.selection import SelectionProblem, check_invariants
from downlinkpick.stability import jaccard

SLOW = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

rates = st.floats(min_value=0.0, max_value=1.0e9, allow_nan=False, allow_infinity=False)
positive_rates = st.floats(
    min_value=1.0, max_value=1.0e9, allow_nan=False, allow_infinity=False
)
durations = st.floats(min_value=1.0, max_value=3.0e3, allow_nan=False, allow_infinity=False)


@given(rate=positive_rates, duration=durations)
@settings(max_examples=100, deadline=None)
def test_flat_capacity_is_rate_times_duration(rate: float, duration: float) -> None:
    window = ContactWindow("w", 0.0, duration, constant_rate_profile(0.0, duration, rate))
    expected = rate * duration
    assert window.capacity_bits() == pytest.approx(expected, rel=1e-12)


@given(
    duration=durations,
    peak=positive_rates,
    floor=st.floats(min_value=0.0, max_value=1.0),
    points=st.integers(min_value=3, max_value=21),
    split=st.floats(min_value=0.01, max_value=0.99),
)
@settings(max_examples=100, deadline=None)
def test_capacity_is_additive_over_any_split(
    duration: float, peak: float, floor: float, points: int, split: float
) -> None:
    window = ContactWindow(
        "w", 0.0, duration, rate_profile_from_elevation(
            0.0, duration, peak, floor_fraction=floor, points=points
        )
    )
    whole = window.delivered_bits(0.0, duration)
    assume(whole > 0.0)
    middle = split * duration
    parts = window.delivered_bits(0.0, middle) + window.delivered_bits(middle, duration)
    assert parts == pytest.approx(whole, rel=1e-10)


@given(
    duration=durations,
    peak=positive_rates,
    fraction=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=100, deadline=None)
def test_time_for_bits_inverts_delivered_bits(
    duration: float, peak: float, fraction: float
) -> None:
    window = ContactWindow(
        "w",
        0.0,
        duration,
        rate_profile_from_elevation(0.0, duration, peak, floor_fraction=0.2, points=9),
    )
    capacity = window.capacity_bits()
    assume(capacity > 1.0)
    target = fraction * capacity
    seconds = window.time_for_bits(target)
    assert 0.0 <= seconds <= duration + 1e-9
    assert window.delivered_bits(0.0, seconds) == pytest.approx(target, rel=1e-8, abs=1.0)


@given(
    duration=durations,
    peak=positive_rates,
    extra=st.floats(min_value=0.0, max_value=1.0e3),
)
@settings(max_examples=100, deadline=None)
def test_extension_is_monotone_in_capacity(
    duration: float, peak: float, extra: float
) -> None:
    window = ContactWindow(
        "w",
        0.0,
        duration,
        rate_profile_from_elevation(0.0, duration, peak, floor_fraction=0.3, points=9),
    )
    assert window.extended(extra).capacity_bits() >= window.capacity_bits() - 1e-6


@given(
    rates_list=st.lists(rates, min_size=2, max_size=12),
    duration=durations,
)
@settings(max_examples=100, deadline=None)
def test_capacity_is_bounded_by_peak_times_duration(
    rates_list: list[float], duration: float
) -> None:
    times = np.linspace(0.0, duration, len(rates_list))
    breakpoints = tuple(
        (float(time), float(rate)) for time, rate in zip(times, rates_list, strict=True)
    )
    window = ContactWindow("w", 0.0, duration, breakpoints)
    capacity = window.capacity_bits()
    assert capacity <= max(rates_list) * duration * (1.0 + 1e-12) + 1e-6
    assert capacity >= min(rates_list) * duration * (1.0 - 1e-12) - 1e-6


#: A product is generated as a ``(value, size_bits)`` pair so that the two lists
#: are the same length by construction. Generating two independent lists and
#: discarding the mismatched pairs with ``assume`` filters out six inputs in
#: seven and trips Hypothesis's filter_too_much health check.
items = st.lists(
    st.tuples(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=8.0, max_value=1.0e7, allow_nan=False, allow_infinity=False),
    ),
    min_size=1,
    max_size=7,
)

positive_items = st.lists(
    st.tuples(
        st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=8.0, max_value=1.0e7, allow_nan=False, allow_infinity=False),
    ),
    min_size=1,
    max_size=6,
)

capacities = st.floats(
    min_value=8.0, max_value=5.0e7, allow_nan=False, allow_infinity=False
)

reliabilities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


def _problem(
    values: list[float], sizes: list[float], capacity: float, reliability: float
) -> SelectionProblem:
    inventory = Inventory(
        [
            DataProduct(
                f"p{index}",
                raw_bytes=size / 8.0,
                age_s=0.0,
                expiry_s=1000.0,
                declared_value=value,
                compressibility=1.0,
            )
            for index, (value, size) in enumerate(zip(values, sizes, strict=True))
        ]
    )
    plan = ContactPlan(
        [
            ContactWindow(
                "w",
                100.0,
                200.0,
                constant_rate_profile(100.0, 200.0, capacity / 100.0),
                reliability=reliability,
            )
        ]
    )
    return SelectionProblem.from_declared(inventory, plan)


@given(products=items, capacity=capacities, reliability=reliabilities)
@SLOW
def test_greedy_never_exceeds_the_optimum(
    products: list[tuple[float, float]], capacity: float, reliability: float
) -> None:
    values = [value for value, _size in products]
    sizes = [size for _value, size in products]
    problem = _problem(values, sizes, capacity, reliability)
    optimum = exact_select(problem).total_value
    heuristic = greedy_select(problem).total_value
    # A two-sided tolerance, not an ordering of exact floats. The absolute term
    # is the solver's own MIP gap: HiGHS stops when its incumbent is within
    # SOLVER_ABS_GAP_VALUE_UNITS of its dual bound, so greedy may legitimately
    # exceed the reported optimum by that much. A Hypothesis run found exactly
    # that with a product worth 5.96e-08; see tests/test_regressions.py.
    assert heuristic <= optimum * (1.0 + 1e-9) + SOLVER_ABS_GAP_VALUE_UNITS


@given(products=items, capacity=capacities)
@SLOW
def test_milp_agrees_with_enumeration(
    products: list[tuple[float, float]], capacity: float
) -> None:
    values = [value for value, _size in products]
    sizes = [size for _value, size in products]
    problem = _problem(values, sizes, capacity, 1.0)
    solved = exact_select(problem).total_value
    enumerated = enumerate_select(problem).total_value
    # The absolute term is the solver's MIP gap, not a fudge: the enumeration
    # proves optimality with no tolerance and the MILP to within that gap.
    assert solved == pytest.approx(
        enumerated, rel=1e-9, abs=SOLVER_ABS_GAP_VALUE_UNITS
    )


@given(products=items, capacity=capacities, reliability=reliabilities)
@SLOW
def test_both_selectors_satisfy_every_invariant(
    products: list[tuple[float, float]], capacity: float, reliability: float
) -> None:
    values = [value for value, _size in products]
    sizes = [size for _value, size in products]
    problem = _problem(values, sizes, capacity, reliability)
    for selection in (greedy_select(problem), exact_select(problem)):
        report = check_invariants(problem, selection)
        assert report.ok, report.violations


@given(products=items, capacity=capacities)
@SLOW
def test_dantzig_bound_dominates_the_optimum(
    products: list[tuple[float, float]], capacity: float
) -> None:
    values = [value for value, _size in products]
    sizes = [size for _value, size in products]
    problem = _problem(values, sizes, capacity, 1.0)
    optimum = exact_select(problem).total_value
    assert dantzig_bound(problem) >= optimum * (1.0 - 1e-9) - SOLVER_ABS_GAP_VALUE_UNITS


@given(
    products=positive_items,
    capacity=capacities,
    scale=st.floats(min_value=0.01, max_value=100.0, allow_nan=False),
)
@SLOW
def test_scaling_every_value_scales_the_objective(
    products: list[tuple[float, float]], capacity: float, scale: float
) -> None:
    # A uniform rescaling of the values leaves every constraint untouched, so
    # the optimal objective scales by the same factor. The SELECTION is not
    # asserted to be identical: an equal-value tie can be broken either way and
    # the solver is free to return either vertex.
    values = [value for value, _size in products]
    sizes = [size for _value, size in products]
    problem = _problem(values, sizes, capacity, 1.0)
    base = exact_select(problem).total_value
    scaled_problem = problem.with_values(
        {name: scale * float(problem.values[name]) for name in problem.inventory.names},
        value_source="scaled",
    )
    scaled = exact_select(scaled_problem).total_value
    assert scaled == pytest.approx(
        scale * base, rel=1e-7, abs=SOLVER_ABS_GAP_VALUE_UNITS * max(scale, 1.0)
    )


@given(
    left=st.frozensets(st.sampled_from("abcdef"), max_size=6),
    right=st.frozensets(st.sampled_from("abcdef"), max_size=6),
)
@settings(max_examples=200, deadline=None)
def test_jaccard_is_symmetric_and_bounded(
    left: frozenset[str], right: frozenset[str]
) -> None:
    value = jaccard(left, right)
    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(jaccard(right, left), rel=0.0, abs=0.0)
    if left == right:
        assert value == pytest.approx(1.0, rel=1e-15)


@given(
    sizes=st.lists(
        st.floats(min_value=8.0, max_value=1.0e6, allow_nan=False), min_size=1, max_size=8
    ),
    capacity=capacities,
)
@SLOW
def test_zero_values_select_nothing(sizes: list[float], capacity: float) -> None:
    problem = _problem([0.0] * len(sizes), sizes, capacity, 1.0)
    assert exact_select(problem).total_value == pytest.approx(0.0, abs=1e-12)
    assert greedy_select(problem).assignments == ()
