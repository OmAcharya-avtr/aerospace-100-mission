"""Known-answer and validation tests for the requirement language.

Every expected number below is hand-calculated in a comment beside the
assertion. The sign-agreement property itself lives in
``tests/test_sign_agreement.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from falsifyloop.requirements import (
    BOUND_ATOL,
    Abs,
    Always,
    And,
    Difference,
    Eventually,
    Or,
    Predicate,
    Signal,
    check_horizon,
    robustness,
    satisfies,
    violated,
)
from falsifyloop.traces import Trace

# y = 0,1,2,3,4,5,4,3,2,1,0 on t = 0.0 .. 1.0 s at dt = 0.1 s: a triangle.
TRIANGLE = Trace(
    np.arange(11) * 0.1,
    {"y": np.array([0.0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]), "z": np.full(11, -2.0)},
)


def test_signal_term_known_answer() -> None:
    np.testing.assert_allclose(Signal("y").values(TRIANGLE), TRIANGLE.signal("y"))
    assert Signal("y").signals() == frozenset({"y"})
    assert str(Signal("y")) == "y"


def test_difference_term_known_answer() -> None:
    # Backward difference of the triangle at dt = 0.1 s is +10 on the rise and
    # -10 on the fall; index 0 copies index 1, so it is +10.
    d = Difference("y").values(TRIANGLE)
    assert d[0] == pytest.approx(10.0)
    assert d[1] == pytest.approx(10.0)
    assert d[5] == pytest.approx(10.0)
    assert d[6] == pytest.approx(-10.0)
    assert d.size == TRIANGLE.length
    assert str(Difference("y")) == "d/dt(y)"


def test_abs_term_known_answer() -> None:
    np.testing.assert_allclose(Abs(Signal("z")).values(TRIANGLE), np.full(11, 2.0))
    assert str(Abs(Signal("z"))) == "|z|"


def test_predicate_le_known_answer() -> None:
    # rho = (bound - y)/scale with bound 4, scale 2: at y = 5 that is -0.5.
    rho = Predicate(Signal("y"), "<=", 4.0, scale=2.0).rho(TRIANGLE)
    assert rho[5] == pytest.approx(-0.5)
    assert rho[0] == pytest.approx(2.0)


def test_predicate_ge_known_answer() -> None:
    # rho = (y - bound)/scale with bound 1, scale 1: at y = 0 that is -1.
    rho = Predicate(Signal("y"), ">=", 1.0).rho(TRIANGLE)
    assert rho[0] == pytest.approx(-1.0)
    assert rho[5] == pytest.approx(4.0)


def test_predicate_boundary_is_satisfied_with_zero_robustness() -> None:
    # y = 5 against bound 5: the non-strict predicate holds and rho is exactly 0.
    formula = Predicate(Signal("y"), "<=", 5.0)
    assert formula.rho(TRIANGLE)[5] == 0.0
    assert bool(formula.sat(TRIANGLE)[5]) is True
    assert not violated(Always(formula, 0.0, 1.0), TRIANGLE)


def test_always_known_answer() -> None:
    # always[0,1] y <= 4 over the whole triangle: min(4 - y) = 4 - 5 = -1.
    formula = Always(Predicate(Signal("y"), "<=", 4.0), 0.0, 1.0)
    assert robustness(formula, TRIANGLE) == pytest.approx(-1.0)
    assert satisfies(formula, TRIANGLE) is False
    assert formula.horizon() == pytest.approx(1.0)


def test_always_window_offset_known_answer() -> None:
    # always[0.6,1.0] y <= 4 looks only at indices 6..10 where y <= 4, so the
    # robustness is min(4 - [4,3,2,1,0]) = 0.
    formula = Always(Predicate(Signal("y"), "<=", 4.0), 0.6, 1.0)
    assert robustness(formula, TRIANGLE) == pytest.approx(0.0)
    assert satisfies(formula, TRIANGLE) is True


def test_eventually_known_answer() -> None:
    # eventually[0,1] y >= 5 is satisfied at index 5 with margin 0.
    formula = Eventually(Predicate(Signal("y"), ">=", 5.0), 0.0, 1.0)
    assert robustness(formula, TRIANGLE) == pytest.approx(0.0)
    assert satisfies(formula, TRIANGLE) is True


def test_empty_window_conventions() -> None:
    # At index 10 the window [0.5, 0.7] s starts at index 15, past the end.
    always = Always(Predicate(Signal("y"), "<=", -99.0), 0.5, 0.7)
    eventually = Eventually(Predicate(Signal("y"), ">=", 99.0), 0.5, 0.7)
    assert always.rho(TRIANGLE)[10] == np.inf
    assert bool(always.sat(TRIANGLE)[10]) is True
    assert eventually.rho(TRIANGLE)[10] == -np.inf
    assert bool(eventually.sat(TRIANGLE)[10]) is False


def test_nested_eventually_always_known_answer() -> None:
    # eventually[0,1] always[0,0.2] y <= 2 on the triangle, by hand.
    # Inner at index i is min over j in i..i+2 of (2 - y[j]), clipped at the end:
    #   i=0: min(2,1,0) = 0        i=8:  min(0,1,2) = 0
    #   i=9: window clips to 9..10, min(1,2) = 1
    #   i=10: window clips to 10 alone, 2 - 0 = 2
    # The outer max over i = 0..10 is therefore 2, attained at the last sample
    # where the inner window has been clipped to a single point. That is the
    # vacuous-satisfaction trap, and it is exactly why check_horizon exists: the
    # formula needs 1.2 s of trace and this one is 1.0 s long.
    formula = Eventually(Always(Predicate(Signal("y"), "<=", 2.0), 0.0, 0.2), 0.0, 1.0)
    assert robustness(formula, TRIANGLE) == pytest.approx(2.0)
    assert satisfies(formula, TRIANGLE) is True
    assert formula.horizon() == pytest.approx(1.2)
    with pytest.raises(ValueError, match="clipped or empty window"):
        check_horizon(formula, TRIANGLE)


def test_nested_eventually_always_on_a_long_enough_trace() -> None:
    # The same formula on a 2 s trace where y stays at 7: the inner always is
    # 2 - 7 = -5 everywhere with a full window, so the outer max is -5 and the
    # requirement is violated.
    flat = Trace(np.arange(21) * 0.1, {"y": np.full(21, 7.0)})
    formula = Eventually(Always(Predicate(Signal("y"), "<=", 2.0), 0.0, 0.2), 0.0, 1.0)
    check_horizon(formula, flat)
    assert robustness(formula, flat) == pytest.approx(-5.0)
    assert satisfies(formula, flat) is False


def test_and_or_known_answers() -> None:
    low = Predicate(Signal("y"), "<=", 10.0)  # rho[0] = 10
    high = Predicate(Signal("y"), ">=", 3.0)  # rho[0] = -3
    assert robustness(And(low, high), TRIANGLE) == pytest.approx(-3.0)
    assert robustness(Or(low, high), TRIANGLE) == pytest.approx(10.0)
    assert satisfies(And(low, high), TRIANGLE) is False
    assert satisfies(Or(low, high), TRIANGLE) is True


def test_and_horizon_is_the_deepest_branch() -> None:
    formula = And(
        Always(Predicate(Signal("y"), "<=", 9.0), 0.0, 0.3),
        Always(Predicate(Signal("y"), "<=", 9.0), 0.0, 0.9),
    )
    assert formula.horizon() == pytest.approx(0.9)


def test_signals_propagate_through_the_tree() -> None:
    formula = And(
        Always(Predicate(Signal("y"), "<=", 1.0), 0.0, 0.1),
        Eventually(Predicate(Abs(Difference("z")), "<=", 1.0), 0.0, 0.1),
    )
    assert formula.signals() == frozenset({"y", "z"})


def test_str_round_trips_the_structure() -> None:
    formula = Always(Predicate(Abs(Difference("y")), "<=", 12.0), 0.0, 0.5)
    assert str(formula) == "always[0,0.5] (|d/dt(y)| <= 12)"


def test_missing_signal_in_formula_raises_keyerror() -> None:
    formula = Always(Predicate(Signal("absent"), "<=", 1.0), 0.0, 0.1)
    with pytest.raises(KeyError, match="does not have"):
        robustness(formula, TRIANGLE)
    with pytest.raises(KeyError, match="does not have"):
        satisfies(formula, TRIANGLE)


def test_time_bound_off_the_sample_grid_is_rejected() -> None:
    # dt = 0.1 s, so 0.15 s is half a sample and must not be silently rounded.
    formula = Always(Predicate(Signal("y"), "<=", 1.0), 0.0, 0.15)
    with pytest.raises(ValueError, match="integer multiple"):
        robustness(formula, TRIANGLE)


def test_time_bound_within_tolerance_is_accepted() -> None:
    formula = Always(Predicate(Signal("y"), "<=", 9.0), 0.0, 0.5 + BOUND_ATOL / 10.0)
    assert robustness(formula, TRIANGLE) == pytest.approx(4.0)


def test_check_horizon_accepts_a_long_enough_trace() -> None:
    check_horizon(Always(Predicate(Signal("y"), "<=", 9.0), 0.0, 1.0), TRIANGLE)


def test_check_horizon_rejects_a_short_trace() -> None:
    with pytest.raises(ValueError, match="clipped or empty window"):
        check_horizon(Always(Predicate(Signal("y"), "<=", 9.0), 0.0, 5.0), TRIANGLE)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: Signal(""), "non-empty string"),
        (lambda: Difference(""), "non-empty string"),
        (lambda: Predicate(Signal("y"), "<", 1.0), "op must be one of"),
        (lambda: Predicate(Signal("y"), "<=", np.inf), "bound must be finite"),
        (lambda: Predicate(Signal("y"), "<=", 1.0, scale=0.0), "strictly positive"),
        (lambda: Predicate(Signal("y"), "<=", 1.0, scale=-1.0), "strictly positive"),
        (lambda: And(), "at least one argument"),
        (lambda: Or(), "at least one argument"),
        (lambda: Always(Predicate(Signal("y"), "<=", 1.0), -0.1, 1.0), "non-negative"),
        (lambda: Always(Predicate(Signal("y"), "<=", 1.0), 1.0, 0.5), "lo <= hi"),
        (lambda: Always(Predicate(Signal("y"), "<=", 1.0), 0.0, np.inf), "must be finite"),
    ],
)
def test_constructor_validation(factory, match) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Abs("y"),
        lambda: Predicate("y", "<=", 1.0),
        lambda: And(Predicate(Signal("y"), "<=", 1.0), "x"),
        lambda: Always("x", 0.0, 1.0),
    ],
)
def test_type_validation(factory) -> None:
    with pytest.raises(TypeError):
        factory()


def test_top_level_functions_type_check_their_arguments() -> None:
    formula = Predicate(Signal("y"), "<=", 1.0)
    with pytest.raises(TypeError, match="Formula"):
        robustness("not a formula", TRIANGLE)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Trace"):
        robustness(formula, "not a trace")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Formula"):
        satisfies("not a formula", TRIANGLE)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Trace"):
        satisfies(formula, "not a trace")  # type: ignore[arg-type]
