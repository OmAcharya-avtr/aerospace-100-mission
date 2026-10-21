"""The selection problem, equation (2), and the invariant checker."""

from __future__ import annotations

import pytest

from downlinkpick.contacts import ContactPlan, ContactWindow, constant_rate_profile
from downlinkpick.inventory import DataProduct, Inventory
from downlinkpick.selection import (
    Assignment,
    Selection,
    SelectionProblem,
    check_invariants,
    sizes_from_ratios,
)


def test_pair_value_is_value_times_reliability(tiny_problem: SelectionProblem) -> None:
    # Hand calculation: cal declared 5.0 in w1 at reliability 0.8 gives 4.0.
    assert tiny_problem.pair_value("cal", "w1") == pytest.approx(4.0, rel=1e-15)
    assert tiny_problem.pair_value("cal", "w2") == pytest.approx(5.0, rel=1e-15)
    # img expires at 500 s and w2 ends at 700 s, so the pair is worth nothing.
    assert tiny_problem.pair_value("img", "w2") == 0.0


def test_pair_value_rejects_an_unknown_window(tiny_problem: SelectionProblem) -> None:
    with pytest.raises(KeyError, match="no window named"):
        tiny_problem.pair_value("cal", "absent")


def test_feasible_windows_respect_expiry(tiny_problem: SelectionProblem) -> None:
    assert tiny_problem.feasible_windows("cal") == ("w1", "w2")
    assert tiny_problem.feasible_windows("img") == ("w1",)
    assert tiny_problem.feasible_windows("bulk") == ("w1", "w2")


def test_feasible_pairs_are_in_model_order(tiny_problem: SelectionProblem) -> None:
    assert tiny_problem.feasible_pairs() == (
        ("cal", "w1"),
        ("cal", "w2"),
        ("img", "w1"),
        ("bulk", "w1"),
        ("bulk", "w2"),
    )


def test_from_declared_names_every_missing_declaration() -> None:
    inventory = Inventory(
        [
            DataProduct("a", 1.0e6, 0.0, 1000.0, compressibility=0.5),
            DataProduct("b", 1.0e6, 0.0, 1000.0, compressibility=0.5),
        ]
    )
    plan = ContactPlan(
        [ContactWindow("w", 0.0, 100.0, constant_rate_profile(0.0, 100.0, 1.0e6))]
    )
    with pytest.raises(ValueError, match="declared_value is None for: a, b"):
        SelectionProblem.from_declared(inventory, plan)


def test_from_declared_names_every_missing_ratio() -> None:
    inventory = Inventory([DataProduct("a", 1.0e6, 0.0, 1000.0, declared_value=1.0)])
    plan = ContactPlan(
        [ContactWindow("w", 0.0, 100.0, constant_rate_profile(0.0, 100.0, 1.0e6))]
    )
    with pytest.raises(ValueError, match="compressibility is None for: a"):
        SelectionProblem.from_declared(inventory, plan)


def test_missing_value_or_size_rejected(
    tiny_inventory: Inventory, tiny_plan: ContactPlan
) -> None:
    with pytest.raises(ValueError, match="no value supplied"):
        SelectionProblem(tiny_inventory, tiny_plan, {"cal": 1.0}, {"cal": 1.0})
    values = dict.fromkeys(tiny_inventory.names, 1.0)
    with pytest.raises(ValueError, match="no size supplied"):
        SelectionProblem(tiny_inventory, tiny_plan, values, {"cal": 1.0})


def test_negative_value_and_zero_size_rejected(
    tiny_inventory: Inventory, tiny_plan: ContactPlan
) -> None:
    values = dict.fromkeys(tiny_inventory.names, 1.0)
    sizes = dict.fromkeys(tiny_inventory.names, 1.0)
    values["cal"] = -1.0
    with pytest.raises(ValueError, match="value must be non-negative"):
        SelectionProblem(tiny_inventory, tiny_plan, values, sizes)
    values["cal"] = 1.0
    sizes["cal"] = 0.0
    with pytest.raises(ValueError, match="size must be strictly positive"):
        SelectionProblem(tiny_inventory, tiny_plan, values, sizes)


def test_with_values_sizes_and_plan_are_copies(tiny_problem: SelectionProblem) -> None:
    other = tiny_problem.with_values({"cal": 1.0, "img": 1.0, "bulk": 1.0}, value_source="x")
    assert other.value_source == "x"
    assert tiny_problem.values["cal"] == 5.0
    bigger = tiny_problem.with_sizes({"cal": 1.0, "img": 1.0, "bulk": 1.0})
    assert bigger.sizes_bits["cal"] == 1.0
    assert tiny_problem.sizes_bits["cal"] == pytest.approx(4.0e6)
    extended = tiny_problem.with_plan(tiny_problem.plan.extended_window("w1", 10.0))
    assert extended.plan["w1"].end_s == pytest.approx(210.0)
    assert tiny_problem.plan["w1"].end_s == pytest.approx(200.0)


def test_unreachable_identifies_expiry_oversize_and_dependency() -> None:
    inventory = Inventory(
        [
            DataProduct("late", 1.0e3, 0.0, 10.0, declared_value=1.0, compressibility=1.0),
            DataProduct("huge", 1.0e9, 0.0, 1000.0, declared_value=1.0, compressibility=1.0),
            DataProduct(
                "child",
                1.0e3,
                0.0,
                1000.0,
                declared_value=1.0,
                compressibility=1.0,
                depends_on=("huge",),
            ),
            DataProduct("fine", 1.0e3, 0.0, 1000.0, declared_value=1.0, compressibility=1.0),
        ]
    )
    plan = ContactPlan(
        [ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 1.0e5))]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    assert problem.unreachable() == ("late", "huge", "child")


def test_score_uses_this_problems_values(tiny_problem: SelectionProblem) -> None:
    selection = Selection(
        (Assignment("cal", "w1", 4.0e6, 4.0),), 4.0, "manual"
    )
    assert tiny_problem.score(selection) == pytest.approx(4.0, rel=1e-15)
    rescaled = tiny_problem.with_values(
        {"cal": 10.0, "img": 8.0, "bulk": 20.0}, value_source="x"
    )
    assert rescaled.score(selection) == pytest.approx(8.0, rel=1e-15)


def test_total_value_available(tiny_problem: SelectionProblem) -> None:
    assert tiny_problem.total_value_available() == pytest.approx(33.0, rel=1e-15)


def test_capacity_bits_matches_the_plan(tiny_problem: SelectionProblem) -> None:
    assert tiny_problem.capacity_bits() == tiny_problem.plan.capacity_bits()


def test_selection_accessors() -> None:
    assignments = (
        Assignment("a", "w1", 10.0, 1.0),
        Assignment("b", "w1", 20.0, 2.0),
        Assignment("c", "w2", 30.0, 3.0),
    )
    selection = Selection(assignments, 6.0, "manual", bound=7.0)
    assert selection.selected == ("a", "b", "c")
    assert selection.selected_set == frozenset({"a", "b", "c"})
    assert selection.window_of == {"a": "w1", "b": "w1", "c": "w2"}
    assert selection.bits_by_window() == {"w1": 30.0, "w2": 30.0}
    assert selection.value_by_window() == {"w1": 3.0, "w2": 3.0}
    assert selection.total_bits() == 60.0
    assert selection.bound == 7.0


def test_invariants_hold_for_a_valid_selection(tiny_problem: SelectionProblem) -> None:
    selection = Selection(
        (
            Assignment("cal", "w1", 4.0e6, 4.0),
            Assignment("img", "w1", 4.0e6, 6.4),
        ),
        10.4,
        "manual",
    )
    report = check_invariants(tiny_problem, selection)
    assert report.ok
    assert report.violations == ()
    # Hand calculation: w1 capacity 2e7 bits, 8e6 committed, so headroom -1.2e7.
    assert report.worst_capacity_overshoot_bits == pytest.approx(-1.2e7, rel=1e-12)
    assert report.value_residual_rel == pytest.approx(0.0, abs=1e-15)


def test_invariants_catch_precedence_violation(tiny_problem: SelectionProblem) -> None:
    # img is delivered in w1 (ends 200 s) and its prerequisite cal in w2
    # (ends 700 s), so the calibration frame arrives after the image.
    selection = Selection(
        (
            Assignment("img", "w1", 4.0e6, 6.4),
            Assignment("cal", "w2", 4.0e6, 5.0),
        ),
        11.4,
        "manual",
    )
    report = check_invariants(tiny_problem, selection)
    assert not report.ok
    assert any("precedence violated" in v for v in report.violations)


def test_invariants_catch_missing_dependency(tiny_problem: SelectionProblem) -> None:
    selection = Selection((Assignment("img", "w1", 4.0e6, 6.4),), 6.4, "manual")
    report = check_invariants(tiny_problem, selection)
    assert any("dependency unmet" in v for v in report.violations)


def test_invariants_catch_capacity_overflow(tiny_problem: SelectionProblem) -> None:
    selection = Selection(
        (
            Assignment("bulk", "w1", 2.16e7, 16.0),
            Assignment("cal", "w1", 4.0e6, 4.0),
        ),
        20.0,
        "manual",
    )
    report = check_invariants(tiny_problem, selection)
    assert any("capacity violated" in v for v in report.violations)
    assert report.worst_capacity_overshoot_bits > 0.0


def test_invariants_catch_double_downlink(tiny_problem: SelectionProblem) -> None:
    selection = Selection(
        (
            Assignment("cal", "w1", 4.0e6, 4.0),
            Assignment("cal", "w2", 4.0e6, 5.0),
        ),
        9.0,
        "manual",
    )
    report = check_invariants(tiny_problem, selection)
    assert any("double downlink" in v for v in report.violations)


def test_invariants_catch_unknown_names(tiny_problem: SelectionProblem) -> None:
    report = check_invariants(
        tiny_problem, Selection((Assignment("ghost", "w1", 1.0, 0.0),), 0.0, "manual")
    )
    assert any("unknown product" in v for v in report.violations)
    report = check_invariants(
        tiny_problem, Selection((Assignment("cal", "ghost", 4.0e6, 0.0),), 0.0, "manual")
    )
    assert any("unknown window" in v for v in report.violations)


def test_value_conservation_uses_a_two_sided_tolerance(
    tiny_problem: SelectionProblem,
) -> None:
    # The reported total is perturbed by less than the tolerance, so the
    # invariant holds: what is guaranteed is agreement to within rounding, not
    # bit-exact equality of two independently accumulated sums.
    selection = Selection(
        (Assignment("cal", "w1", 4.0e6, 4.0),), 4.0 * (1.0 + 1.0e-12), "manual"
    )
    assert check_invariants(tiny_problem, selection).ok
    worse = Selection((Assignment("cal", "w1", 4.0e6, 4.0),), 4.0 * (1.0 + 1.0e-6), "manual")
    assert not check_invariants(tiny_problem, worse).ok


def test_empty_selection_is_valid(tiny_problem: SelectionProblem) -> None:
    report = check_invariants(tiny_problem, Selection((), 0.0, "manual"))
    assert report.ok
    assert report.worst_capacity_overshoot_bits == pytest.approx(-2.0e7, rel=1e-12)


def test_sizes_from_ratios_rejects_out_of_range(tiny_inventory: Inventory) -> None:
    good = sizes_from_ratios(tiny_inventory, {"cal": 0.5, "img": 0.25, "bulk": 0.9})
    assert good["cal"] == pytest.approx(4.0e6, rel=1e-15)
    with pytest.raises(ValueError, match="no compression ratio supplied"):
        sizes_from_ratios(tiny_inventory, {"cal": 0.5})
    with pytest.raises(ValueError, match="compression ratio must lie in"):
        sizes_from_ratios(tiny_inventory, {"cal": 50.0, "img": 0.25, "bulk": 0.9})
