"""Marginal value of contact time, capacity sweeps and the left-behind report."""

from __future__ import annotations

import pytest

from downlinkpick.contacts import ContactPlan, ContactWindow, constant_rate_profile
from downlinkpick.exact import exact_select
from downlinkpick.greedy import greedy_select
from downlinkpick.inventory import DataProduct, Inventory
from downlinkpick.marginal import (
    DEFAULT_DELTA_S,
    LEFT_BEHIND_REASONS,
    best_window_to_extend,
    capacity_sweep,
    left_behind_report,
    marginal_value_of_contact,
    value_left_behind,
)
from downlinkpick.selection import SelectionProblem


def _one_window_problem(capacity_rate: float, values: dict[str, float]) -> SelectionProblem:
    inventory = Inventory(
        [
            DataProduct(
                name, 125000.0, 0.0, 1000.0, declared_value=value, compressibility=1.0
            )
            for name, value in values.items()
        ]
    )
    plan = ContactPlan(
        [
            ContactWindow(
                "w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, capacity_rate)
            )
        ]
    )
    return SelectionProblem.from_declared(inventory, plan)


def test_marginal_value_known_answer() -> None:
    # Each product is 125000 bytes = 1e6 bits. The window carries 1e6 bit/s for
    # 100 s = 1e8 bits... too large. Use 1e4 bit/s: 1e6 bits, so exactly one
    # product fits. Extending by 100 s adds another 1e6 bits and admits the
    # second-best product, worth 4.0, so the price is 4.0 / 100 = 0.04 per second.
    problem = _one_window_problem(1.0e4, {"a": 10.0, "b": 4.0, "c": 1.0})
    assert problem.plan.total_capacity_bits() == pytest.approx(1.0e6, rel=1e-15)
    base = exact_select(problem)
    assert base.selected_set == {"a"}
    marginals = marginal_value_of_contact(problem, base=base, delta_s=100.0)
    assert len(marginals) == 1
    marginal = marginals[0]
    assert marginal.delta_bits == pytest.approx(1.0e6, rel=1e-12)
    assert marginal.delta_value == pytest.approx(4.0, rel=1e-12)
    assert marginal.value_per_second == pytest.approx(0.04, rel=1e-12)
    assert marginal.value_per_gbit == pytest.approx(4.0 / 1.0e-3, rel=1e-12)
    assert marginal.entering == ("b",)
    assert marginal.leaving == ()
    assert marginal.expiry_crossings == ()


def test_marginal_value_is_zero_when_nothing_is_binding() -> None:
    problem = _one_window_problem(1.0e6, {"a": 10.0, "b": 4.0})
    marginals = marginal_value_of_contact(problem, delta_s=10.0)
    assert marginals[0].delta_value == pytest.approx(0.0, abs=1e-12)
    assert best_window_to_extend(marginals) is None


def test_expiry_crossing_is_reported() -> None:
    # The window ends at 200 s and `soon` expires at 240 s, so an extension of
    # 60 s pushes the delivery epoch past its deadline and it loses the window.
    inventory = Inventory(
        [
            DataProduct("soon", 125000.0, 0.0, 240.0, declared_value=9.0, compressibility=1.0),
            DataProduct("later", 125000.0, 0.0, 9000.0, declared_value=1.0, compressibility=1.0),
        ]
    )
    plan = ContactPlan(
        [ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 1.0e4))]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    base = exact_select(problem)
    assert base.selected_set == {"soon"}
    marginals = marginal_value_of_contact(problem, base=base, delta_s=60.0)
    assert marginals[0].expiry_crossings == ("soon",)
    # The extension buys capacity and spends feasibility: the objective falls.
    assert marginals[0].delta_value < 0.0
    assert marginals[0].value_per_second < 0.0
    assert best_window_to_extend(marginals) is None


def test_marginal_rejects_a_bad_step_and_an_unknown_window(
    tiny_problem: SelectionProblem,
) -> None:
    with pytest.raises(ValueError, match="delta_s must be strictly positive"):
        marginal_value_of_contact(tiny_problem, delta_s=0.0)
    with pytest.raises(KeyError, match="no window named"):
        marginal_value_of_contact(tiny_problem, windows=("absent",))
    assert DEFAULT_DELTA_S == 1.0


def test_marginal_handles_an_extension_that_would_overlap(
    tiny_problem: SelectionProblem,
) -> None:
    # Extending w1 by 1000 s would overlap w2, so the price is reported as zero
    # rather than raising: the extension is not available.
    marginals = marginal_value_of_contact(tiny_problem, delta_s=1000.0, windows=("w1",))
    assert marginals[0].delta_bits == 0.0
    assert marginals[0].value_per_second == 0.0
    assert marginals[0].value_per_gbit is None


def test_marginal_under_greedy_and_exact_may_differ(
    instance_problem: SelectionProblem,
) -> None:
    by_greedy = marginal_value_of_contact(
        instance_problem, selector=greedy_select, delta_s=60.0
    )
    by_exact = marginal_value_of_contact(
        instance_problem, selector=exact_select, delta_s=60.0
    )
    assert len(by_greedy) == len(by_exact) == len(instance_problem.plan)
    assert all(m.delta_bits > 0.0 for m in by_exact)


def test_capacity_sweep_is_non_decreasing_without_expiry_crossings() -> None:
    problem = _one_window_problem(1.0e4, {"a": 10.0, "b": 4.0, "c": 1.0})
    points = capacity_sweep(problem, "w", (0.0, 50.0, 100.0, 200.0, 400.0))
    assert [p.n_expiry_crossings for p in points] == [0, 0, 0, 0, 0]
    values = [p.total_value for p in points]
    assert values == sorted(values)
    assert points[0].total_value == pytest.approx(10.0, rel=1e-12)
    assert points[-1].total_value == pytest.approx(15.0, rel=1e-12)
    assert points[-1].n_selected == 3
    assert points[0].capacity_bits == pytest.approx(1.0e6, rel=1e-12)


def test_capacity_sweep_validation(tiny_problem: SelectionProblem) -> None:
    with pytest.raises(KeyError, match="no window named"):
        capacity_sweep(tiny_problem, "absent", (0.0,))
    with pytest.raises(ValueError, match="extra_seconds must be non-negative"):
        capacity_sweep(tiny_problem, "w1", (-1.0,))


def test_left_behind_reasons_are_exhaustive_and_correct() -> None:
    inventory = Inventory(
        [
            DataProduct("late", 1.0e3, 0.0, 10.0, declared_value=1.0, compressibility=1.0),
            DataProduct("huge", 1.0e9, 0.0, 1000.0, declared_value=2.0, compressibility=1.0),
            DataProduct(
                "child",
                1.0e3,
                0.0,
                1000.0,
                declared_value=3.0,
                compressibility=1.0,
                depends_on=("huge",),
            ),
            # Window capacity is 1e3 bit/s * 100 s = 1e5 bits. small is 8e3 bits
            # and big is 9.6e4 bits, so big fits in an empty window but not
            # alongside small, which is worth more: big is left behind for
            # capacity and not because it could never fit.
            DataProduct("big", 12000.0, 0.0, 1000.0, declared_value=4.0, compressibility=1.0),
            DataProduct("small", 1000.0, 0.0, 1000.0, declared_value=9.0, compressibility=1.0),
        ]
    )
    plan = ContactPlan(
        [ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 1.0e3))]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    selection = exact_select(problem)
    rows = left_behind_report(problem, selection)
    reasons = {row.product: row.reason for row in rows}
    assert reasons["late"] == "expired"
    assert reasons["huge"] == "oversized"
    assert reasons["child"] == "dependency"
    assert reasons["big"] == "capacity"
    assert set(reasons) | selection.selected_set == set(inventory.names)
    assert all(row.reason in LEFT_BEHIND_REASONS for row in rows)
    # The report is ordered most valuable first.
    assert [row.value for row in rows] == sorted((row.value for row in rows), reverse=True)


def test_left_behind_shortfall_in_seconds_is_bits_over_final_rate() -> None:
    # Capacity 1e6 bits, products of 1e6 bits each. The optimum takes one, so the
    # other is short by its whole size, 1e6 bits, at 1e4 bit/s = 100 s.
    problem = _one_window_problem(1.0e4, {"a": 10.0, "b": 4.0})
    selection = exact_select(problem)
    rows = left_behind_report(problem, selection)
    assert len(rows) == 1
    assert rows[0].product == "b"
    assert rows[0].extra_bits_needed == pytest.approx(1.0e6, rel=1e-9)
    assert rows[0].extra_seconds_needed == pytest.approx(100.0, rel=1e-9)
    assert rows[0].best_window == "w"


def test_left_behind_is_empty_when_everything_fits() -> None:
    problem = _one_window_problem(1.0e6, {"a": 1.0, "b": 2.0})
    rows = left_behind_report(problem, exact_select(problem))
    assert rows == ()
    assert value_left_behind(rows) == dict.fromkeys(LEFT_BEHIND_REASONS, 0.0)


def test_value_left_behind_has_every_reason_key() -> None:
    problem = _one_window_problem(1.0e4, {"a": 10.0, "b": 4.0})
    totals = value_left_behind(left_behind_report(problem, exact_select(problem)))
    assert set(totals) == set(LEFT_BEHIND_REASONS)
    assert totals["capacity"] == pytest.approx(4.0, rel=1e-12)


def test_left_behind_zero_rate_window_gives_infinite_shortfall() -> None:
    inventory = Inventory(
        [DataProduct("p", 1.0e3, 0.0, 1000.0, declared_value=1.0, compressibility=1.0)]
    )
    plan = ContactPlan(
        [ContactWindow("w", 100.0, 200.0, ((100.0, 1.0e3), (199.0, 1.0e3), (200.0, 0.0)))]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    rows = left_behind_report(problem, type(exact_select(problem))((), 0.0, "manual"))
    assert rows[0].extra_seconds_needed == float("inf")


def test_best_window_to_extend_picks_the_highest_price(
    instance_problem: SelectionProblem,
) -> None:
    marginals = marginal_value_of_contact(instance_problem, delta_s=120.0)
    best = best_window_to_extend(marginals)
    if best is not None:
        assert best.value_per_second == max(m.value_per_second for m in marginals)
        assert best.delta_value > 0.0
