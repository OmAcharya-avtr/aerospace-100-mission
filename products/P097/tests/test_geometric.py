"""The mathematical core: known answers computed by hand, identities, validation.

Every known-answer test shows the arithmetic in a comment. The identities are
driven with Hypothesis where one exists, because the geometric expectation has
several, and every counterexample Hypothesis has found is pinned as a named
regression test with literal inputs in ``tests/test_regressions.py``.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from flakerank.geometric import (
    expected_attempts_by_recursion,
    expected_attempts_to_first_pass,
    expected_failed_attempts,
    expected_failed_attempts_by_recursion,
    expected_wasted_seconds,
    failure_probability_for_expected_attempts,
)

# ---------------------------------------------------------------------------
# Known answers, arithmetic shown
# ---------------------------------------------------------------------------


def test_expected_attempts_at_p_one_quarter() -> None:
    """p = 1/4: E[N] = 1/(1 - 1/4) = 1/(3/4) = 4/3 = 1.333333...

    Hand arithmetic: 1 - 0.25 = 0.75; 1 / 0.75 = 1.3333333333333333.
    """
    assert expected_attempts_to_first_pass(0.25) == pytest.approx(4.0 / 3.0, rel=1e-15, abs=0.0)


def test_expected_failed_attempts_at_p_one_quarter() -> None:
    """p = 1/4: E[F] = p/(1-p) = 0.25/0.75 = 1/3 = 0.333333...

    Hand arithmetic: 0.25 / 0.75 = 0.3333333333333333.
    """
    assert expected_failed_attempts(0.25) == pytest.approx(1.0 / 3.0, rel=1e-15, abs=0.0)


def test_expected_failed_attempts_budget_three_at_p_one_quarter() -> None:
    """p = 1/4, m = 3, equation (5): (p - p^3)/(1 - p) + p^3.

    Hand arithmetic:
        p    = 0.25
        p^3  = 0.015625
        p - p^3        = 0.234375
        0.234375/0.75  = 0.3125
        0.3125 + 0.015625 = 0.328125
    The recursion agrees: m=1 -> 0.25; m=2 -> 0.25*(1+0.25) = 0.3125;
    m=3 -> 0.25*(1+0.3125) = 0.328125.
    """
    assert expected_failed_attempts(0.25, retry_budget=3) == pytest.approx(
        0.328125, rel=1e-15, abs=0.0
    )


def test_expected_failed_attempts_budget_one_is_p() -> None:
    """m = 1 is the no-retry policy, so E[F_1] = p exactly in exact arithmetic."""
    for probability in (0.0, 0.25, 1.0 / 44.0, 0.9, 1.0):
        assert expected_failed_attempts(probability, retry_budget=1) == pytest.approx(
            probability, rel=1e-15, abs=0.0
        )


def test_the_mission_case_one_in_fortyfour() -> None:
    """p = 1/44: E[F] = (1/44)/(43/44) = 1/43 = 0.023255813953488372.

    Hand arithmetic: 1 - 1/44 = 43/44; (1/44) / (43/44) = 1/43.
    In a 600-second test that is 600/43 = 13.953488372093023 s per invocation.
    """
    probability = 1.0 / 44.0
    assert expected_failed_attempts(probability) == pytest.approx(1.0 / 43.0, rel=1e-15, abs=0.0)
    assert expected_wasted_seconds(probability, 600.0) == pytest.approx(
        600.0 / 43.0, rel=1e-15, abs=0.0
    )


def test_the_dashboard_inversion_factor() -> None:
    """The product's thesis as arithmetic, which a reader can check in one line.

    A 1-in-44 failure in a 600 s test costs 600/43 = 13.953488372093023 s per
    invocation. A 1-in-2 failure in a 0.1 s test costs 0.1 * (0.5/0.5) = 0.1 s.
    Ratio of costs: 13.953488372093023 / 0.1 = 139.53488372093023, the slow
    rare flake being the expensive one. Ratio of flake rates: 0.5 / 0.0227... =
    22.0, the fast frequent one being the "worse" one. So the two rankings
    disagree on this pair by a factor of 139.53488372093023 * 22.0 = 3069.767...
    """
    slow = expected_wasted_seconds(1.0 / 44.0, 600.0)
    fast = expected_wasted_seconds(0.5, 0.1)
    assert slow == pytest.approx(600.0 / 43.0, rel=1e-15, abs=0.0)
    assert fast == pytest.approx(0.1, rel=1e-15, abs=0.0)
    assert slow / fast == pytest.approx(139.53488372093023, rel=1e-13, abs=0.0)
    rate_ratio = 0.5 / (1.0 / 44.0)
    assert rate_ratio == pytest.approx(22.0, rel=1e-15, abs=0.0)


def test_expected_failed_attempts_at_p_one_half() -> None:
    """p = 1/2: E[F] = 0.5/0.5 = 1. One wasted attempt per invocation on average."""
    assert expected_failed_attempts(0.5) == pytest.approx(1.0, rel=1e-15, abs=0.0)


def test_expected_failed_attempts_at_p_zero_is_zero() -> None:
    """A test that never fails wastes nothing, bounded or not."""
    assert expected_failed_attempts(0.0) == 0.0
    assert expected_failed_attempts(0.0, retry_budget=1) == 0.0
    assert expected_failed_attempts(0.0, retry_budget=12) == 0.0


def test_expected_failed_attempts_at_p_one_with_budget_is_the_budget() -> None:
    """At p = 1 every one of the m attempts fails, so E[F_m] = m exactly."""
    for budget in (1, 2, 3, 7, 100):
        assert expected_failed_attempts(1.0, retry_budget=budget) == float(budget)


def test_expected_wasted_seconds_scales_with_invocations() -> None:
    """S = 330 invocations at p = 1/44 and 600 s: 330 * 600/43 = 4604.651162790698.

    Hand arithmetic: 600/43 = 13.953488372093023; times 330 = 4604.651162790698.
    """
    assert expected_wasted_seconds(
        1.0 / 44.0, 600.0, invocations=330
    ) == pytest.approx(198000.0 / 43.0, rel=1e-14, abs=0.0)


# ---------------------------------------------------------------------------
# The divergence, raised rather than returned
# ---------------------------------------------------------------------------


def test_unbounded_at_p_one_raises_and_does_not_return_inf() -> None:
    """The divergence is an error with an actionable message, never ``inf``."""
    with pytest.raises(ValueError, match="diverges at failure_probability = 1"):
        expected_failed_attempts(1.0)
    with pytest.raises(ValueError, match="retry_budget"):
        expected_failed_attempts(1.0)
    with pytest.raises(ValueError, match="diverges"):
        expected_attempts_to_first_pass(1.0)
    with pytest.raises(ValueError, match="diverges"):
        expected_wasted_seconds(1.0, 10.0)


def test_unbounded_near_one_is_large_but_finite() -> None:
    """p = 1 - 1e-12 gives E[F] near 1e12: large, finite, and not an error.

    Hand arithmetic: 1 - (1 - 1e-12) = 1.000088900582341e-12 in double
    precision, so p/(1-p) is about 9.999e11. The value is asserted only to a
    loose relative tolerance because the input itself cannot be represented
    exactly; the point of the test is that the function returns a finite number
    rather than raising or returning inf.
    """
    value = expected_failed_attempts(1.0 - 1e-12)
    assert math.isfinite(value)
    assert value == pytest.approx(1e12, rel=1e-3, abs=0.0)


def test_bounded_form_is_finite_where_the_unbounded_one_diverges() -> None:
    """The two forms differ exactly where the retry budget matters."""
    with pytest.raises(ValueError):
        expected_failed_attempts(1.0)
    assert expected_failed_attempts(1.0, retry_budget=4) == 4.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("probability", [-0.001, -1.0, 1.0001, 2.0, 1e9])
def test_probability_outside_unit_interval_raises(probability: float) -> None:
    """A per-attempt failure probability cannot leave [0, 1]."""
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        expected_failed_attempts(probability)


def test_nan_probability_raises() -> None:
    """NaN is rejected explicitly rather than falling through a range test."""
    with pytest.raises(ValueError, match="NaN"):
        expected_failed_attempts(float("nan"))


@pytest.mark.parametrize("probability", ["0.5", None, [0.5], True])
def test_non_numeric_probability_raises_type_error(probability: object) -> None:
    """A string or a bool is a TypeError, not a silently coerced float."""
    with pytest.raises(TypeError):
        expected_failed_attempts(probability)  # type: ignore[arg-type]


@pytest.mark.parametrize("budget", [0, -1, -100])
def test_non_positive_retry_budget_raises(budget: int) -> None:
    """A budget of m means at most m ATTEMPTS, so m >= 1."""
    with pytest.raises(ValueError, match="retry_budget must be >= 1"):
        expected_failed_attempts(0.3, retry_budget=budget)


@pytest.mark.parametrize("budget", [1.5, "3", True])
def test_non_integer_retry_budget_raises_type_error(budget: object) -> None:
    """A fractional or boolean budget is a TypeError."""
    with pytest.raises(TypeError):
        expected_failed_attempts(0.3, retry_budget=budget)  # type: ignore[arg-type]


@pytest.mark.parametrize("seconds", [-1.0, float("inf"), float("nan")])
def test_invalid_runtime_raises(seconds: float) -> None:
    """A runtime is a finite non-negative number of seconds."""
    with pytest.raises(ValueError, match="failing_attempt_seconds"):
        expected_wasted_seconds(0.2, seconds)


@pytest.mark.parametrize("invocations", [0, -3])
def test_non_positive_invocations_raises(invocations: int) -> None:
    """A test present in a history ran at least once."""
    with pytest.raises(ValueError, match="invocations must be an integer >= 1"):
        expected_wasted_seconds(0.2, 1.0, invocations=invocations)


def test_inverse_rejects_expectation_below_one() -> None:
    """E[N] < 1 is impossible on the support {1, 2, 3, ...}."""
    with pytest.raises(ValueError, match="must be finite and >= 1"):
        failure_probability_for_expected_attempts(0.5)


def test_recursion_form_rejects_bad_iterations() -> None:
    """The cross-check form validates its own iteration count."""
    with pytest.raises(ValueError, match="iterations"):
        expected_attempts_by_recursion(0.3, iterations=0)


# ---------------------------------------------------------------------------
# Algebraic identities, driven with Hypothesis
# ---------------------------------------------------------------------------

_PROBABILITY = st.floats(
    min_value=0.0, max_value=0.84, allow_nan=False, allow_infinity=False, width=64
)
_BUDGET = st.integers(min_value=1, max_value=60)


@given(_PROBABILITY)
@settings(max_examples=250, deadline=None)
def test_identity_expected_failed_is_expected_attempts_minus_one(probability: float) -> None:
    """E[F] = E[N] - 1, equation (2) against equation (1)."""
    attempts = expected_attempts_to_first_pass(probability)
    failed = expected_failed_attempts(probability)
    assert failed == pytest.approx(attempts - 1.0, rel=1e-12, abs=1e-15)


@given(_PROBABILITY)
@settings(max_examples=250, deadline=None)
def test_identity_recursion_matches_closed_form(probability: float) -> None:
    """E[N] = 1 + p E[N]: the fixed-point iteration matches the closed form.

    The default 200 iterations leave a residual of ``p**200/(1-p)``, which is
    below double precision for ``p <= 0.84``; that is why the strategy stops
    there rather than the strategy being widened and the tolerance loosened.
    """
    assert expected_attempts_by_recursion(probability) == pytest.approx(
        expected_attempts_to_first_pass(probability), rel=1e-12, abs=1e-15
    )


@given(_PROBABILITY, _BUDGET)
@settings(max_examples=400, deadline=None)
def test_identity_bounded_closed_form_matches_its_recursion(
    probability: float, budget: int
) -> None:
    """E[F_m] = p (1 + E[F_{m-1}]) matches equation (5).

    The recursion involves no division at all, so it cannot share a
    cancellation with the closed form, which divides by ``1 - p``.
    """
    closed = expected_failed_attempts(probability, retry_budget=budget)
    recursive = expected_failed_attempts_by_recursion(probability, budget)
    assert closed == pytest.approx(recursive, rel=1e-11, abs=1e-15)


@given(_PROBABILITY, _BUDGET)
@settings(max_examples=300, deadline=None)
def test_identity_cap_never_costs_more_than_unbounded(
    probability: float, budget: int
) -> None:
    """A retry cap can only remove attempts, so E[F_m] <= E[F].

    Compared with a tolerance rather than strictly, because at small ``p`` the
    two agree to within rounding and a strict comparison on a difference of two
    nearly equal numbers is the one thing floating point cannot promise.
    """
    capped = expected_failed_attempts(probability, retry_budget=budget)
    unbounded = expected_failed_attempts(probability)
    assert capped <= unbounded + 1e-12 * max(1.0, unbounded)


@given(_PROBABILITY, st.integers(min_value=1, max_value=40))
@settings(max_examples=300, deadline=None)
def test_identity_bounded_is_monotone_in_the_budget(probability: float, budget: int) -> None:
    """Raising the budget cannot lower the expected wasted-attempt count."""
    lower = expected_failed_attempts(probability, retry_budget=budget)
    higher = expected_failed_attempts(probability, retry_budget=budget + 1)
    assert higher >= lower - 1e-13 * max(1.0, lower)


#: Double-precision unit roundoff, used where a tolerance must follow the
#: CONDITIONING of the expression rather than be a constant.
_UNIT_ROUNDOFF = 2.0**-53


@given(st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False))
@settings(max_examples=250, deadline=None)
def test_identity_probability_expectation_round_trip(expectation: float) -> None:
    """``E[N] -> p -> E[N]`` returns where it started, to within the conditioning.

    The tolerance is NOT a constant, and a constant one was wrong here: a first
    draft asserted ``rel=1e-9`` and Hypothesis falsified it at
    ``expectation = 25929728.0``, where the round trip returns
    25929728.03245014. That is not a defect in either direction of the map; it
    is the conditioning of the round trip, derived as follows.

    Write ``u = 1/E``. Then ``p = fl(1 - u)`` carries an absolute error of
    order ``eps`` once ``p`` is near 1, because the rounding of a quantity near
    1 is ``eps`` in absolute terms. Recovering ``E' = 1/(1 - p)`` divides by
    ``1 - p``, whose true value is ``u = 1/E``, so a fixed absolute error
    ``eps`` in ``p`` is a RELATIVE error of ``eps / u = eps * E`` in the
    denominator and therefore in ``E'``.

    So the admissible relative error grows linearly with ``E`` and the right
    tolerance is ``SAFETY * eps * E``, which at ``E = 2.6e7`` is about 1.2e-8
    and at ``E = 10`` is about 4.4e-15. The counterexample is pinned with its
    literal input in ``tests/test_regressions.py``.
    """
    probability = failure_probability_for_expected_attempts(expectation)
    assume(probability < 1.0)
    tolerance = 4.0 * _UNIT_ROUNDOFF * expectation
    assert expected_attempts_to_first_pass(probability) == pytest.approx(
        expectation, rel=tolerance, abs=0.0
    )


@given(
    _PROBABILITY,
    st.floats(min_value=1e-3, max_value=1e4, allow_nan=False, allow_infinity=False),
    st.integers(min_value=1, max_value=500),
)
@settings(max_examples=300, deadline=None)
def test_identity_wasted_seconds_is_linear_in_runtime_and_invocations(
    probability: float, seconds: float, invocations: int
) -> None:
    """Equation (3) is linear in both ``t`` and ``S``, exactly in exact arithmetic."""
    one = expected_wasted_seconds(probability, seconds, None, 1)
    many = expected_wasted_seconds(probability, seconds, None, invocations)
    assert many == pytest.approx(one * invocations, rel=1e-12, abs=1e-18)


@given(_BUDGET)
@settings(max_examples=60, deadline=None)
def test_identity_bounded_at_p_one_equals_the_budget(budget: int) -> None:
    """The analytic limit of equation (5) at p = 1 is the budget itself."""
    assert expected_failed_attempts(1.0, retry_budget=budget) == float(budget)


@given(st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, deadline=None)
def test_identity_large_budget_recovers_the_unbounded_form(probability: float) -> None:
    """E[F_m] -> p/(1-p) as m grows; at p <= 0.5 and m = 80, p**80 is negligible."""
    capped = expected_failed_attempts(probability, retry_budget=80)
    unbounded = expected_failed_attempts(probability)
    assert capped == pytest.approx(unbounded, rel=1e-12, abs=1e-20)
