"""Sensitivity: value share, variance share and measured margin elasticity."""

from __future__ import annotations

import math

import pytest

from budgetroll import (
    Budget,
    Correlation,
    Item,
    MarginPolicy,
    Node,
    margin_elasticity,
    rollup,
    sensitivity,
    value_shares,
    variance_shares,
)
from budgetroll.errors import DeclarationError

POLICY = MarginPolicy(
    name="t",
    description="sensitivity policy",
    source="declared in tests/test_sensitivity.py",
    levels={"mature": 0.0, "new": 0.20},
)


def build(rows, allocation=None, correlations=(), rule="rss"):
    children = tuple(
        Item(id=i, name=i, value=v, uncertainty=u, maturity=m) for i, v, u, m in rows
    )
    root = Node(
        id="root", name="root", combination=rule, children=children, allocation=allocation
    )
    return Budget(
        id="b",
        name="B",
        unit="kg",
        quantity="mass",
        policy=POLICY,
        root=root,
        correlations=correlations,
    )


def test_value_shares_known_answer():
    # Predicted values 12 (10 at 20 %) and 20 (20 at 0 %); total 32.
    #   12 / 32 = 0.375 and 20 / 32 = 0.625
    budget = build([("a", 10.0, 0.0, "new"), ("b", 20.0, 0.0, "mature")])
    shares = value_shares(rollup(budget))
    assert shares["a"] == pytest.approx(0.375, rel=0, abs=1e-12)
    assert shares["b"] == pytest.approx(0.625, rel=0, abs=1e-12)


def test_value_shares_are_none_for_a_zero_total():
    budget = build([("a", 0.0, 0.0, "mature"), ("b", 0.0, 0.0, "mature")])
    assert set(value_shares(rollup(budget)).values()) == {None}


def test_variance_shares_known_answer_independent():
    # u = (3, 4), independent: variance 25, shares 9/25 = 0.36 and 16/25 = 0.64
    budget = build([("a", 1.0, 3.0, "mature"), ("b", 1.0, 4.0, "mature")])
    shares = variance_shares(budget)
    assert shares["a"] == pytest.approx(0.36, rel=0, abs=1e-12)
    assert shares["b"] == pytest.approx(0.64, rel=0, abs=1e-12)


def test_variance_shares_known_answer_correlated():
    # u = (3, 4), rho = 0.5: variance = 9 + 16 + 12 = 37
    #   share_a = 3 * (3 + 0.5 * 4) / 37 = 3 * 5 / 37 = 15/37
    #   share_b = 4 * (0.5 * 3 + 4) / 37 = 4 * 5.5 / 37 = 22/37
    budget = build(
        [("a", 1.0, 3.0, "mature"), ("b", 1.0, 4.0, "mature")],
        correlations=(Correlation("a", "b", 0.5, "r"),),
    )
    shares = variance_shares(budget)
    assert shares["a"] == pytest.approx(15.0 / 37.0, rel=1e-12)
    assert shares["b"] == pytest.approx(22.0 / 37.0, rel=1e-12)
    assert math.fsum(shares.values()) == pytest.approx(1.0, rel=0, abs=1e-12)


def test_variance_share_can_be_negative_under_negative_correlation():
    # u = (3, 4), rho = -0.9: variance = 25 - 21.6 = 3.4
    #   share_a = 3 * (3 - 0.9 * 4) / 3.4 = 3 * (-0.6) / 3.4 = -1.8 / 3.4
    budget = build(
        [("a", 1.0, 3.0, "mature"), ("b", 1.0, 4.0, "mature")],
        correlations=(Correlation("a", "b", -0.9, "r"),),
    )
    shares = variance_shares(budget)
    assert shares["a"] == pytest.approx(-1.8 / 3.4, rel=1e-12)
    assert shares["a"] < 0.0
    assert math.fsum(shares.values()) == pytest.approx(1.0, rel=0, abs=1e-12)


def test_variance_shares_are_none_when_every_uncertainty_is_zero():
    budget = build([("a", 1.0, 0.0, "mature"), ("b", 1.0, 0.0, "mature")])
    assert set(variance_shares(budget).values()) == {None}


def test_margin_elasticity_known_answer():
    # Leaf a: value 10, contingency 20 %, so predicted 12.
    # A +10 % step takes the value to 11 and the predicted value to 13.2,
    # so the margin falls by exactly 1.2 kg in exact arithmetic.
    budget = build(
        [("a", 10.0, 0.0, "new"), ("b", 20.0, 0.0, "mature")], allocation=100.0
    )
    deltas = margin_elasticity(budget, step=0.10)
    assert deltas["a"] == pytest.approx(-1.2, rel=0, abs=1e-12)
    # Leaf b: 20 at 0 % contingency, +10 % is +2.0, margin falls by 2.0.
    assert deltas["b"] == pytest.approx(-2.0, rel=0, abs=1e-12)


def test_margin_elasticity_matches_the_closed_form():
    # The finite difference equals step * value * (1 + contingency) exactly
    # in exact arithmetic for this model; checked here to within rounding.
    budget = build([("a", 7.3, 0.5, "new"), ("b", 2.1, 0.2, "mature")], allocation=50.0)
    deltas = margin_elasticity(budget, step=0.07)
    assert deltas["a"] == pytest.approx(-0.07 * 7.3 * 1.20, rel=1e-12)
    assert deltas["b"] == pytest.approx(-0.07 * 2.1 * 1.00, rel=1e-12)


def test_margin_elasticity_is_none_without_an_allocation():
    budget = build([("a", 1.0, 0.1, "mature"), ("b", 1.0, 0.1, "mature")])
    assert set(margin_elasticity(budget).values()) == {None}


@pytest.mark.parametrize("step", [0.0, float("inf"), float("nan")])
def test_margin_elasticity_rejects_a_bad_step(step):
    budget = build([("a", 1.0, 0.1, "mature"), ("b", 1.0, 0.1, "mature")], allocation=10.0)
    with pytest.raises(DeclarationError, match="finite non-zero fraction"):
        margin_elasticity(budget, step=step)


def test_sensitivity_is_ranked_by_margin_effect():
    budget = build(
        [("small", 1.0, 5.0, "mature"), ("large", 50.0, 0.1, "mature")], allocation=100.0
    )
    rows = sensitivity(budget)
    assert [r.id for r in rows] == ["large", "small"]


def test_sensitivity_falls_back_to_value_share_without_an_allocation():
    budget = build([("small", 1.0, 5.0, "mature"), ("large", 50.0, 0.1, "mature")])
    rows = sensitivity(budget)
    assert [r.id for r in rows] == ["large", "small"]
    assert all(r.margin_delta is None for r in rows)


def test_sensitivity_ordering_is_deterministic_on_ties():
    budget = build(
        [("a", 5.0, 0.1, "mature"), ("b", 5.0, 0.1, "mature"), ("c", 5.0, 0.1, "mature")],
        allocation=100.0,
    )
    assert [r.id for r in sensitivity(budget)] == ["a", "b", "c"]


def test_sensitivity_on_the_cubesat_power_case(cubesat_power):
    rows = sensitivity(cubesat_power)
    assert rows[0].id == "radio_tx"
    # Transmitter: 1.60 kg nominal at 10 % contingency gives 1.76 predicted;
    # a +10 % step is 0.16 * 1.10 = 0.176 W of additional predicted load.
    assert rows[0].margin_delta == pytest.approx(-0.176, rel=1e-12)
    assert math.fsum(r.value_share for r in rows) == pytest.approx(1.0, rel=0, abs=1e-12)
    assert math.fsum(r.variance_share for r in rows) == pytest.approx(1.0, rel=0, abs=1e-12)
