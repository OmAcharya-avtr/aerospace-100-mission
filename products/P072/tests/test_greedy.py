"""The greedy baseline: ordering, packing, precedence, rollback, determinism."""

from __future__ import annotations

import pytest

from downlinkpick.contacts import ContactPlan, ContactWindow, constant_rate_profile
from downlinkpick.dataset import generate_instance
from downlinkpick.greedy import (
    PLACEMENT_RULES,
    _window_rank,
    dantzig_bound,
    density_order,
    greedy_select,
)
from downlinkpick.inventory import DataProduct, Inventory
from downlinkpick.selection import SelectionProblem, check_invariants, sizes_from_ratios


def test_known_answer_on_the_tiny_problem(tiny_problem: SelectionProblem) -> None:
    # Hand calculation. Sizes: cal 4e6, img 4e6, bulk 2.16e7 bits. Windows carry
    # 2e7 bits each; img expires at 500 s so only w1 is feasible for it, and
    # precedence then forces cal into w1 as well. The {cal, img} bundle has
    # density (5 + 8) / 8e6; bulk has 20 / 2.16e7, which is lower. Taking the
    # bundle into w1 costs 8e6 of w1's 2e7 bits and yields 0.8 * (5 + 8) = 10.4.
    # bulk at 2.16e7 bits then fits in neither window's remainder, so the greedy
    # total is 10.4.
    selection = greedy_select(tiny_problem)
    assert selection.method == "greedy"
    assert selection.status == "feasible"
    assert selection.bound is None
    assert selection.selected_set == {"cal", "img"}
    assert selection.window_of == {"cal": "w1", "img": "w1"}
    assert selection.total_value == pytest.approx(10.4, rel=1e-12)
    assert check_invariants(tiny_problem, selection).ok


def test_greedy_is_deterministic(instance_problem: SelectionProblem) -> None:
    first = greedy_select(instance_problem)
    second = greedy_select(instance_problem)
    assert first.assignments == second.assignments
    assert first.total_value == second.total_value


def test_greedy_never_violates_an_invariant() -> None:
    for seed in range(1, 41):
        instance = generate_instance(seed, n_products=18, n_windows=3, capacity_fraction=0.4)
        values = {
            name: float(instance.inventory[name].declared_value or 0.0)
            for name in instance.inventory.names
        }
        problem = SelectionProblem(
            instance.inventory,
            instance.plan,
            values,
            sizes_from_ratios(instance.inventory, instance.true_ratios),
        )
        report = check_invariants(problem, greedy_select(problem))
        assert report.ok, (seed, report.violations)


def test_bundle_rollback_leaves_no_orphan() -> None:
    # child needs parent, and parent alone exceeds the only window's capacity,
    # so neither may be selected and the window must be left for `other`.
    inventory = Inventory(
        [
            DataProduct("parent", 1.0e9, 0.0, 1000.0, declared_value=1.0, compressibility=1.0),
            DataProduct(
                "child",
                1.0e3,
                0.0,
                1000.0,
                declared_value=50.0,
                compressibility=1.0,
                depends_on=("parent",),
            ),
            DataProduct("other", 1.0e4, 0.0, 1000.0, declared_value=3.0, compressibility=1.0),
        ]
    )
    plan = ContactPlan(
        [ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 1.0e5))]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    selection = greedy_select(problem)
    assert selection.selected_set == {"other"}
    assert check_invariants(problem, selection).ok


def test_both_placement_rules_are_tried() -> None:
    # Reliability-first would park `cal` in the later, more reliable window and
    # strand `img`, which expires before it. Earliest-first keeps both. Greedy
    # takes the better of the two, so it must find the 10.4 solution.
    inventory = Inventory(
        [
            DataProduct("cal", 1.0e6, 0.0, 1000.0, declared_value=5.0, compressibility=0.5),
            DataProduct(
                "img",
                2.0e6,
                0.0,
                500.0,
                declared_value=8.0,
                compressibility=0.25,
                depends_on=("cal",),
            ),
        ]
    )
    plan = ContactPlan(
        [
            ContactWindow(
                "early", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 2.0e5), reliability=0.8
            ),
            ContactWindow(
                "late", 600.0, 700.0, constant_rate_profile(600.0, 700.0, 2.0e5), reliability=1.0
            ),
        ]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    selection = greedy_select(problem)
    assert selection.selected_set == {"cal", "img"}
    assert selection.total_value == pytest.approx(10.4, rel=1e-12)
    assert PLACEMENT_RULES == ("reliability", "earliest")


def test_window_rank_orders_as_documented(tiny_problem: SelectionProblem) -> None:
    by_reliability = sorted(
        tiny_problem.plan.names, key=lambda w: _window_rank(tiny_problem, w, "reliability")
    )
    by_earliest = sorted(
        tiny_problem.plan.names, key=lambda w: _window_rank(tiny_problem, w, "earliest")
    )
    assert by_reliability == ["w2", "w1"]
    assert by_earliest == ["w1", "w2"]
    with pytest.raises(ValueError, match="unknown placement rule"):
        _window_rank(tiny_problem, "w1", "nonsense")


def test_zero_value_products_are_not_selected() -> None:
    inventory = Inventory(
        [DataProduct("p", 1.0e3, 0.0, 1000.0, declared_value=0.0, compressibility=1.0)]
    )
    plan = ContactPlan(
        [ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 1.0e6))]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    assert greedy_select(problem).assignments == ()


def test_everything_expired_selects_nothing() -> None:
    inventory = Inventory(
        [DataProduct("p", 1.0e3, 0.0, 10.0, declared_value=5.0, compressibility=1.0)]
    )
    plan = ContactPlan(
        [ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 1.0e6))]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    assert greedy_select(problem).assignments == ()


def test_density_order_is_descending(tiny_problem: SelectionProblem) -> None:
    order = density_order(tiny_problem)
    densities = [
        float(tiny_problem.values[name]) / tiny_problem.sizes_bits[name] for name in order
    ]
    assert densities == sorted(densities, reverse=True)
    # Hand calculation: img 8/4e6 = 2.0e-6, cal 5/4e6 = 1.25e-6,
    # bulk 20/2.16e7 = 9.26e-7.
    assert order == ("img", "cal", "bulk")


def test_dantzig_bound_is_an_upper_bound_on_greedy(
    instance_problem: SelectionProblem,
) -> None:
    assert dantzig_bound(instance_problem) >= greedy_select(instance_problem).total_value


def test_dantzig_bound_by_hand() -> None:
    # One window of 1e7 bits at reliability 1.0; two products of 5e6 bits with
    # values 10 and 4, plus one of 1e7 bits with value 1. Taking the first two
    # fills the knapsack exactly, so the bound is 14.
    inventory = Inventory(
        [
            DataProduct("a", 625000.0, 0.0, 1000.0, declared_value=10.0, compressibility=1.0),
            DataProduct("b", 625000.0, 0.0, 1000.0, declared_value=4.0, compressibility=1.0),
            DataProduct("c", 1250000.0, 0.0, 1000.0, declared_value=1.0, compressibility=1.0),
        ]
    )
    plan = ContactPlan(
        [ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 1.0e5))]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    assert problem.plan.total_capacity_bits() == pytest.approx(1.0e7, rel=1e-15)
    assert dantzig_bound(problem) == pytest.approx(14.0, rel=1e-12)


def test_dantzig_bound_splits_the_last_item() -> None:
    # Capacity 1e7 bits; a is 8e6 bits worth 8, b is 8e6 bits worth 4. a fits
    # whole, b is taken at 2e6/8e6 = 0.25 of its value, so the bound is 9.
    inventory = Inventory(
        [
            DataProduct("a", 1.0e6, 0.0, 1000.0, declared_value=8.0, compressibility=1.0),
            DataProduct("b", 1.0e6, 0.0, 1000.0, declared_value=4.0, compressibility=1.0),
        ]
    )
    plan = ContactPlan(
        [ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 1.0e5))]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    assert dantzig_bound(problem) == pytest.approx(9.0, rel=1e-12)


def test_dantzig_bound_ignores_an_expired_product() -> None:
    inventory = Inventory(
        [
            DataProduct("live", 1.0e3, 0.0, 1000.0, declared_value=1.0, compressibility=1.0),
            DataProduct("dead", 1.0e3, 0.0, 10.0, declared_value=99.0, compressibility=1.0),
        ]
    )
    plan = ContactPlan(
        [ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 1.0e6))]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    assert dantzig_bound(problem) == pytest.approx(1.0, rel=1e-12)
