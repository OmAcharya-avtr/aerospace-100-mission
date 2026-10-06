"""The select_rate entry point: method dispatch, canonicalisation, guards."""

from __future__ import annotations

import pytest

from coderateopt import LognormalFade, RateProblem, illustrative_modcod_table, select_rate
from coderateopt.select import CANONICAL_SUBSET_BUDGET


def _problem(**overrides) -> RateProblem:
    kwargs = {
        "modcods": illustrative_modcod_table(),
        "fade": LognormalFade(0.2),
        "margin_db": 12.0,
        "availability_target": 0.99,
        "max_entries": 2,
        "mode": "long_run",
    }
    kwargs.update(overrides)
    return RateProblem(**kwargs)


@pytest.mark.parametrize("method", ["auto", "milp", "exhaustive", "closed_form"])
def test_every_method_reaches_the_same_optimum(method):
    problem = _problem()
    reference = select_rate(problem, method="exhaustive").expected_goodput
    assert select_rate(problem, method=method).expected_goodput == pytest.approx(
        reference, abs=1e-9
    )


def test_auto_canonicalises_by_default_and_not_when_asked_not_to():
    problem = _problem()
    assert select_rate(problem).canonical is True
    assert select_rate(problem).method == "exhaustive"
    assert select_rate(problem, canonicalise=False).canonical is False
    assert select_rate(problem, canonicalise=False).method == "milp"


def test_canonical_result_is_identical_across_repeated_calls():
    problem = _problem(availability_target=0.95)
    first = select_rate(problem)
    for _ in range(5):
        again = select_rate(problem)
        assert again.support == first.support
        assert again.tied_supports == first.tied_supports


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method must be"):
        select_rate(_problem(), method="simplex")  # type: ignore[arg-type]


def test_budget_constant_is_large_enough_for_the_shipped_table():
    from coderateopt import count_subsets

    assert count_subsets(9, 3) < CANONICAL_SUBSET_BUDGET


def test_support_names_round_trip():
    problem = _problem()
    solution = select_rate(problem)
    names = solution.support_names(problem.modcods)
    assert len(names) == len(solution.support)
    assert all(name in problem.modcods.names for name in names)
