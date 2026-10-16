"""Edge cases: zeros, single items, deep nesting, negative values, extremes."""

from __future__ import annotations

import math

import pytest

from budgetroll import (
    Budget,
    Correlation,
    Item,
    MarginPolicy,
    Node,
    rollup,
    run_checks,
    sensitivity,
    text_report,
)

POLICY = MarginPolicy(
    name="t",
    description="edge policy",
    source="declared in tests/test_edge_cases.py",
    levels={"mature": 0.0, "new": 0.20},
)


def wrap(root, correlations=(), k_sigma=1.0, unit="kg"):
    return Budget(
        id="b",
        name="B",
        unit=unit,
        quantity="mass",
        policy=POLICY,
        root=root,
        correlations=correlations,
        k_sigma=k_sigma,
    )


def test_single_leaf_budget():
    root = Node(
        id="root",
        name="root",
        combination="linear",
        children=(Item(id="a", name="a", value=3.0, uncertainty=0.5, maturity="new"),),
    )
    result = rollup(wrap(root)).root
    assert result.predicted == pytest.approx(3.6, rel=0, abs=1e-12)
    assert result.uncertainty == pytest.approx(0.5, rel=0, abs=1e-12)


def test_all_zero_budget():
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(
            Item(id="a", name="a", value=0.0, uncertainty=0.0, maturity="mature"),
            Item(id="b", name="b", value=0.0, uncertainty=0.0, maturity="mature"),
        ),
    )
    result = rollup(wrap(root)).root
    assert result.predicted == 0.0
    assert result.uncertainty == 0.0
    assert result.contingency is None  # undefined, not reported as zero per cent


def test_negative_values_in_a_log_domain_budget():
    # dB losses entered negative; contingency makes a loss larger in magnitude.
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(
            Item(id="a", name="a", value=-2.5, uncertainty=0.3, maturity="new"),
            Item(id="b", name="b", value=-1.0, uncertainty=0.2, maturity="mature"),
        ),
    )
    result = rollup(wrap(root, unit="dB")).root
    # -2.5 * 1.20 = -3.0 and -1.0 * 1.00 = -1.0, so predicted is -4.0 dB.
    assert result.predicted == pytest.approx(-4.0, rel=0, abs=1e-12)
    assert result.uncertainty == pytest.approx(math.sqrt(0.13), rel=1e-12)


def test_deep_nesting_is_preserved():
    node: Item | Node = Item(id="leaf", name="leaf", value=1.0, uncertainty=0.1, maturity="mature")
    for depth in range(10):
        node = Node(id=f"n{depth}", name=f"n{depth}", combination="linear", children=(node,))
    budget = wrap(node)
    roll = rollup(budget)
    assert roll.by_id("leaf").depth == 10
    assert roll.root.uncertainty == pytest.approx(0.1, rel=0, abs=1e-12)
    # Every one of the ten single-child nodes except the linear ones is
    # silent; all ten are linear, so BR007 does not fire.
    assert "BR007" not in [f.code for f in run_checks(budget)]


def test_k_sigma_zero_makes_margin_at_k_equal_margin():
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(
            Item(id="a", name="a", value=1.0, uncertainty=5.0, maturity="mature"),
            Item(id="b", name="b", value=1.0, uncertainty=5.0, maturity="mature"),
        ),
        allocation=10.0,
    )
    result = rollup(wrap(root, k_sigma=0.0)).root
    assert result.margin_at_k == pytest.approx(result.margin_absolute, rel=0, abs=1e-12)


def test_perfect_negative_correlation_cancels_to_zero():
    root = Node(
        id="root",
        name="root",
        combination="correlated",
        children=(
            Item(id="a", name="a", value=1.0, uncertainty=2.0, maturity="mature"),
            Item(id="b", name="b", value=1.0, uncertainty=2.0, maturity="mature"),
        ),
    )
    budget = wrap(root, (Correlation("a", "b", -1.0, "exactly opposed"),))
    # var = 4 + 4 - 2 * 2 * 2 = 0; clipped at zero rather than NaN.
    assert rollup(budget).root.uncertainty == pytest.approx(0.0, rel=0, abs=1e-9)


def test_very_large_magnitudes_do_not_overflow():
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(
            Item(id="a", name="a", value=1e150, uncertainty=1e150, maturity="mature"),
            Item(id="b", name="b", value=1e150, uncertainty=1e150, maturity="mature"),
        ),
    )
    value = rollup(wrap(root)).root.uncertainty
    assert math.isfinite(value)
    assert value == pytest.approx(math.sqrt(2.0) * 1e150, rel=1e-12)


def test_rss_overflows_above_the_square_root_of_the_double_maximum():
    # 1e160 squared is 1e320, which is above DBL_MAX = 1.797e308, so the
    # square overflows to infinity. Documented in README.md Limitations.
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(
            Item(id="a", name="a", value=1.0, uncertainty=1e160, maturity="mature"),
            Item(id="b", name="b", value=1.0, uncertainty=1e160, maturity="mature"),
        ),
    )
    with pytest.warns(RuntimeWarning, match="overflow"):
        value = rollup(wrap(root)).root.uncertainty
    assert math.isinf(value)


def test_zero_uncertainty_everywhere_gives_no_sensitivity_variance_shares():
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(
            Item(id="a", name="a", value=1.0, uncertainty=0.0, maturity="mature"),
            Item(id="b", name="b", value=2.0, uncertainty=0.0, maturity="mature"),
        ),
        allocation=10.0,
    )
    rows = sensitivity(wrap(root))
    assert all(row.variance_share is None for row in rows)
    assert rows[0].id == "b"


def test_text_report_renders_for_an_all_zero_budget():
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(
            Item(id="a", name="a", value=0.0, uncertainty=0.0, maturity="mature"),
            Item(id="b", name="b", value=0.0, uncertainty=0.0, maturity="mature"),
        ),
    )
    assert "Verdict:" in text_report(wrap(root))


def test_a_very_long_name_is_truncated_in_the_table_not_dropped():
    long_name = "x" * 200
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(
            Item(id="a", name=long_name, value=1.0, uncertainty=0.1, maturity="mature"),
            Item(id="b", name="b", value=1.0, uncertainty=0.1, maturity="mature"),
        ),
    )
    report = text_report(wrap(root))
    assert "x" * 29 in report
    # The full name survives in the sensitivity ranking data even though the
    # table column truncates it.
    assert sensitivity(wrap(root))[0].name == long_name
