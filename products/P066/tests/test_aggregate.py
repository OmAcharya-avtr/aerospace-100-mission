"""Known-answer tests for each aggregation rule, with the hand arithmetic shown.

Every expected value in this file was computed by hand before the code was
run, and the arithmetic is written out in the comment above the assertion.
Comparisons use a tolerance on both sides: the identities below are exact in
exact arithmetic and agree to within rounding in IEEE 754 double precision.
"""

from __future__ import annotations

import math

import pytest

from budgetroll import (
    Budget,
    Correlation,
    Item,
    MarginPolicy,
    Node,
    combine,
    linear_rollup_uncertainty,
    rollup,
    rss_rollup_uncertainty,
)
from budgetroll.errors import AggregationError

POLICY = MarginPolicy(
    name="t",
    description="test policy",
    source="declared in tests/test_aggregate.py",
    levels={"mature": 0.0, "new": 0.20},
)


def build(
    rule: str,
    items: list[tuple[str, float, float, str]],
    allocation: float | None = None,
    k_sigma: float = 1.0,
    correlations: tuple[Correlation, ...] = (),
) -> Budget:
    """Build a one-level budget from ``(id, value, uncertainty, maturity)`` rows."""
    children = tuple(
        Item(id=i, name=i, value=v, uncertainty=u, maturity=m) for i, v, u, m in items
    )
    root = Node(id="root", name="root", combination=rule, children=children, allocation=allocation)
    return Budget(
        id="b",
        name="b",
        unit="kg",
        quantity="mass",
        policy=POLICY,
        root=root,
        correlations=correlations,
        k_sigma=k_sigma,
    )


# --- linear combination -------------------------------------------------


def test_linear_combination_known_answer():
    # Hand arithmetic: 0.3 + 0.4 + 1.2 = 1.9
    assert combine("linear", [0.3, 0.4, 1.2]) == pytest.approx(1.9, rel=0, abs=1e-12)


def test_linear_rollup_known_answer():
    # Hand arithmetic, uncertainties 0.3, 0.4, 1.2 summed:
    #   0.3 + 0.4 + 1.2 = 1.9 kg
    budget = build(
        "linear",
        [("a", 1.0, 0.3, "mature"), ("b", 1.0, 0.4, "mature"), ("c", 1.0, 1.2, "mature")],
    )
    assert rollup(budget).root.uncertainty == pytest.approx(1.9, rel=0, abs=1e-12)


# --- root-sum-square combination ----------------------------------------


def test_rss_combination_pythagorean_triple():
    # Hand arithmetic: sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    assert combine("rss", [3.0, 4.0]) == pytest.approx(5.0, rel=0, abs=1e-12)


def test_rss_combination_second_triple():
    # Hand arithmetic: sqrt(5^2 + 12^2) = sqrt(25 + 144) = sqrt(169) = 13
    assert combine("rss", [5.0, 12.0]) == pytest.approx(13.0, rel=0, abs=1e-12)


def test_rss_rollup_known_answer():
    # Hand arithmetic, uncertainties 0.10, 0.08, 0.05, 0.06:
    #   0.0100 + 0.0064 + 0.0025 + 0.0036 = 0.0225
    #   sqrt(0.0225) = 0.15 kg
    budget = build(
        "rss",
        [
            ("a", 1.0, 0.10, "mature"),
            ("b", 1.0, 0.08, "mature"),
            ("c", 1.0, 0.05, "mature"),
            ("d", 1.0, 0.06, "mature"),
        ],
    )
    assert rollup(budget).root.uncertainty == pytest.approx(0.15, rel=0, abs=1e-12)


# --- correlated combination ---------------------------------------------


def test_correlated_rho_half_known_answer():
    # Hand arithmetic, u = (3, 4), rho = 0.5:
    #   var = 9 + 16 + 2 * 0.5 * 3 * 4 = 25 + 12 = 37
    #   u = sqrt(37) = 6.0827625302982193...
    budget = build(
        "correlated",
        [("a", 1.0, 3.0, "mature"), ("b", 1.0, 4.0, "mature")],
        correlations=(Correlation("a", "b", 0.5, "declared in the test"),),
    )
    assert rollup(budget).root.uncertainty == pytest.approx(math.sqrt(37.0), rel=1e-12)


def test_correlated_rho_one_equals_linear():
    # Hand arithmetic, u = (3, 4), rho = 1:
    #   var = 9 + 16 + 2 * 1 * 12 = 49, u = 7 = 3 + 4
    budget = build(
        "correlated",
        [("a", 1.0, 3.0, "mature"), ("b", 1.0, 4.0, "mature")],
        correlations=(Correlation("a", "b", 1.0, "declared in the test"),),
    )
    assert rollup(budget).root.uncertainty == pytest.approx(7.0, rel=1e-12)


def test_correlated_rho_minus_one_equals_absolute_difference():
    # Hand arithmetic, u = (3, 4), rho = -1:
    #   var = 9 + 16 - 24 = 1, u = 1 = |3 - 4|
    budget = build(
        "correlated",
        [("a", 1.0, 3.0, "mature"), ("b", 1.0, 4.0, "mature")],
        correlations=(Correlation("a", "b", -1.0, "declared in the test"),),
    )
    assert rollup(budget).root.uncertainty == pytest.approx(1.0, rel=0, abs=1e-9)


# --- contingency --------------------------------------------------------


def test_contingency_is_summed_not_root_sum_squared():
    # Hand arithmetic: 10 * 1.20 + 20 * 1.00 = 12 + 20 = 32 kg predicted,
    # with allowance 10 * 0.20 + 20 * 0.00 = 2.0 kg. If the allowances were
    # root-sum-squared the allowance would be sqrt(4 + 0) = 2.0 here, so the
    # case uses a second non-zero allowance below to separate the two.
    budget = build("rss", [("a", 10.0, 0.0, "new"), ("b", 20.0, 0.0, "mature")])
    root = rollup(budget).root
    assert root.nominal == pytest.approx(30.0, rel=0, abs=1e-12)
    assert root.allowance == pytest.approx(2.0, rel=0, abs=1e-12)
    assert root.predicted == pytest.approx(32.0, rel=0, abs=1e-12)


def test_two_contingencies_sum_rather_than_rss():
    # Hand arithmetic: allowances 10 * 0.2 = 2.0 and 5 * 0.2 = 1.0.
    #   linear   2.0 + 1.0 = 3.0      <- what this package does
    #   rss      sqrt(4 + 1) = 2.236  <- what it must never do
    budget = build("rss", [("a", 10.0, 0.0, "new"), ("b", 5.0, 0.0, "new")])
    allowance = rollup(budget).root.allowance
    assert allowance == pytest.approx(3.0, rel=0, abs=1e-12)
    assert allowance != pytest.approx(math.sqrt(5.0), rel=1e-6)


# --- margin -------------------------------------------------------------


def test_margin_known_answer():
    # Hand arithmetic: nominal 10 + 20 = 30, allowance 10 * 0.2 = 2,
    #   predicted 32, allocation 40, margin 8, fraction 8 / 40 = 0.20,
    #   uncertainty sqrt(3^2 + 4^2) = 5, margin at k = 1 is 40 - 37 = 3.
    budget = build(
        "rss",
        [("a", 10.0, 3.0, "new"), ("b", 20.0, 4.0, "mature")],
        allocation=40.0,
        k_sigma=1.0,
    )
    root = rollup(budget).root
    assert root.predicted == pytest.approx(32.0, rel=0, abs=1e-12)
    assert root.margin_absolute == pytest.approx(8.0, rel=0, abs=1e-12)
    assert root.margin_fraction == pytest.approx(0.20, rel=0, abs=1e-12)
    assert root.uncertainty == pytest.approx(5.0, rel=0, abs=1e-12)
    assert root.margin_at_k == pytest.approx(3.0, rel=0, abs=1e-12)


def test_margin_is_none_without_allocation():
    budget = build("rss", [("a", 1.0, 0.1, "mature"), ("b", 1.0, 0.1, "mature")])
    root = rollup(budget).root
    assert root.allocation is None
    assert root.margin_absolute is None
    assert root.margin_fraction is None
    assert root.margin_at_k is None


def test_margin_fraction_is_none_for_zero_allocation():
    budget = build("rss", [("a", 0.0, 0.0, "mature"), ("b", 0.0, 0.0, "mature")], allocation=0.0)
    root = rollup(budget).root
    assert root.margin_absolute == pytest.approx(0.0, abs=1e-12)
    assert root.margin_fraction is None


# --- hierarchy ----------------------------------------------------------


def test_hierarchical_rss_matches_flat_rss_to_rounding():
    # sqrt(sqrt(a^2 + b^2)^2 + sqrt(c^2 + d^2)^2) = sqrt(a^2 + b^2 + c^2 + d^2)
    # exactly in exact arithmetic; here it is checked to within rounding.
    left = Node(
        id="l",
        name="l",
        combination="rss",
        children=(
            Item(id="a", name="a", value=1.0, uncertainty=0.3, maturity="mature"),
            Item(id="b", name="b", value=1.0, uncertainty=0.4, maturity="mature"),
        ),
    )
    right = Node(
        id="r",
        name="r",
        combination="rss",
        children=(
            Item(id="c", name="c", value=1.0, uncertainty=0.5, maturity="mature"),
            Item(id="d", name="d", value=1.0, uncertainty=1.2, maturity="mature"),
        ),
    )
    root = Node(id="root", name="root", combination="rss", children=(left, right))
    budget = Budget(
        id="b", name="b", unit="kg", quantity="mass", policy=POLICY, root=root
    )
    assert rollup(budget).root.uncertainty == pytest.approx(
        rss_rollup_uncertainty(budget), rel=1e-12
    )


def test_consumption_of_parent_allocation():
    # Hand arithmetic: predicted 32 against a root allocation of 40 is 0.80.
    budget = build(
        "rss", [("a", 10.0, 0.0, "new"), ("b", 20.0, 0.0, "mature")], allocation=40.0
    )
    roll = rollup(budget)
    # 12 / 40 = 0.30 and 20 / 40 = 0.50
    assert roll.by_id("a").consumption_of_parent_allocation == pytest.approx(0.30, abs=1e-12)
    assert roll.by_id("b").consumption_of_parent_allocation == pytest.approx(0.50, abs=1e-12)


def test_linear_and_rss_helpers_agree_with_hand_values():
    budget = build(
        "rss",
        [("a", 1.0, 3.0, "mature"), ("b", 1.0, 4.0, "mature"), ("c", 1.0, 12.0, "mature")],
    )
    # linear 3 + 4 + 12 = 19; rss sqrt(9 + 16 + 144) = sqrt(169) = 13
    assert linear_rollup_uncertainty(budget) == pytest.approx(19.0, abs=1e-12)
    assert rss_rollup_uncertainty(budget) == pytest.approx(13.0, abs=1e-12)


def test_combine_rejects_correlated_rule():
    with pytest.raises(AggregationError, match="cannot be evaluated over immediate children"):
        combine("correlated", [1.0, 2.0])


def test_combine_rejects_unknown_rule():
    with pytest.raises(AggregationError, match="unknown"):
        combine("unknown", [1.0])


def test_rollup_by_id_raises_for_missing_identifier():
    budget = build("rss", [("a", 1.0, 0.1, "mature"), ("b", 1.0, 0.1, "mature")])
    with pytest.raises(KeyError, match="no node or item with id"):
        rollup(budget).by_id("nope")
