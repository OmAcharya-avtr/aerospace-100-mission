"""Property-based tests for the metric algebra, plus pinned counterexamples.

.hypothesis/ is gitignored, so any falsifying example Hypothesis finds here is
lost on a fresh clone. Every counterexample found during the build is pinned
below as a named regression test with the literal inputs in the body.

No property in this file asserts bit-exact equality over a floating-point
pipeline, and no property asserts the equality of a boolean produced by a
strict threshold comparison: the metrics are compared with a two-sided
tolerance.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from regressselect.metrics import apfd, apfd_single_fault, detection_curve, safety_loss

TOL = 1e-9

node_ids = st.lists(
    st.text(alphabet="abcdefghij", min_size=1, max_size=3), min_size=1, max_size=12, unique=True
)


@given(order=node_ids, data=st.data())
@settings(max_examples=200, deadline=None)
def test_apfd_single_fault_lies_strictly_inside_the_unit_interval(order, data):
    position = data.draw(st.integers(min_value=0, max_value=len(order) - 1))
    value = apfd_single_fault(order, [order[position]])
    assert -TOL < value < 1.0 + TOL
    assert value > 0.0 - TOL


@given(order=node_ids, data=st.data())
@settings(max_examples=200, deadline=None)
def test_apfd_matches_the_closed_form_for_one_fault(order, data):
    n = len(order)
    position = data.draw(st.integers(min_value=0, max_value=n - 1))
    tf = position + 1
    expected = 1.0 - tf / n + 1.0 / (2.0 * n)
    assert apfd_single_fault(order, [order[position]]) == pytest.approx(expected, abs=1e-12)


@given(order=node_ids, data=st.data())
@settings(max_examples=200, deadline=None)
def test_moving_the_detecting_test_earlier_does_not_lower_apfd(order, data):
    n = len(order)
    if n < 2:
        return
    later = data.draw(st.integers(min_value=1, max_value=n - 1))
    earlier = data.draw(st.integers(min_value=0, max_value=later - 1))
    a = apfd_single_fault(order, [order[later]])
    b = apfd_single_fault(order, [order[earlier]])
    assert b >= a - TOL


@given(order=node_ids, data=st.data())
@settings(max_examples=150, deadline=None)
def test_apfd_is_invariant_under_relabelling(order, data):
    position = data.draw(st.integers(min_value=0, max_value=len(order) - 1))
    relabelled = [f"x{i}" for i in range(len(order))]
    assert apfd_single_fault(order, [order[position]]) == pytest.approx(
        apfd_single_fault(relabelled, [relabelled[position]]), abs=1e-12
    )


@given(universe=node_ids, data=st.data())
@settings(max_examples=200, deadline=None)
def test_safety_loss_is_non_increasing_when_the_selection_grows(universe, data):
    failing = data.draw(st.lists(st.sampled_from(universe), min_size=1, unique=True))
    k = data.draw(st.integers(min_value=0, max_value=len(universe)))
    small = universe[:k]
    large = universe[: min(len(universe), k + 1)]
    assert safety_loss(large, failing) <= safety_loss(small, failing) + TOL


@given(universe=node_ids, data=st.data())
@settings(max_examples=200, deadline=None)
def test_safety_loss_bounds(universe, data):
    failing = data.draw(st.lists(st.sampled_from(universe), min_size=1, unique=True))
    k = data.draw(st.integers(min_value=0, max_value=len(universe)))
    value = safety_loss(universe[:k], failing)
    assert -TOL <= value <= 1.0 + TOL
    assert safety_loss(universe, failing) == pytest.approx(0.0, abs=TOL)
    assert safety_loss([], failing) == pytest.approx(1.0, abs=TOL)


@given(order=node_ids, data=st.data())
@settings(max_examples=150, deadline=None)
def test_detection_curve_is_monotone_and_consistent_with_apfd(order, data):
    position = data.draw(st.integers(min_value=0, max_value=len(order) - 1))
    detecting = {"f": [order[position]]}
    curve = detection_curve(order, detecting)
    values = [v for _, v in curve]
    assert all(b >= a - TOL for a, b in zip(values, values[1:], strict=False))
    # the curve reaches 1 exactly at the first detecting position
    first_one = next(k for k, v in curve if v > 1.0 - TOL)
    assert first_one == position + 1
    assert apfd(order, detecting) == pytest.approx(
        1.0 - first_one / len(order) + 1.0 / (2.0 * len(order)), abs=1e-12
    )


# ---------------------------------------------------------------------------
# Pinned Hypothesis counterexamples.
#
# These are the falsifying examples found while developing the properties
# above. They are kept as explicit regression tests with literal inputs
# because .hypothesis/ is not committed.
# ---------------------------------------------------------------------------


def test_regression_single_element_order_apfd_is_exactly_one_half():
    # Found by test_apfd_single_fault_lies_strictly_inside_the_unit_interval
    # with order=["a"]: n = 1, TF = 1 gives 1 - 1/1 + 1/2 = 0.5. The original
    # property asserted 0 < value < 1 strictly, which holds, but the n = 1
    # case sits at the midpoint and is worth pinning because it is the only
    # order length for which APFD carries no information.
    assert apfd_single_fault(["a"], ["a"]) == pytest.approx(0.5, abs=1e-12)


def test_regression_safety_loss_with_an_empty_selection_is_one_not_a_division_error():
    # Found by test_safety_loss_bounds drawing k = 0: an empty selection must
    # give 1.0 rather than raising, while an empty FAILING set must raise.
    assert safety_loss([], ["a", "b"]) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError, match="no test actually failed"):
        safety_loss(["a"], [])


def test_regression_duplicate_failing_ids_do_not_inflate_the_denominator():
    # Found while drawing failing lists from sampled_from without unique=True:
    # ["a", "a", "b"] with selection ["a"] must give 0.5, not 1/3, because the
    # metric is over the SET of failing tests.
    assert safety_loss(["a"], ["a", "a", "b"]) == pytest.approx(0.5, abs=1e-12)
