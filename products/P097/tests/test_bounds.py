"""The independently derived bracket, and the negative controls that must fire.

Every negative control here is anchored on a value derived from the BOUND, not
on the accumulator's output: an offset applied to a value that already sits
inside a bracket does not guarantee it leaves it, and an earlier product of
this mission shipped two of eleven controls silently too weak that way. Each
control asserts that it fired.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flakerank.baselines import accumulated_wasted_time
from flakerank.bounds import SAFETY, UNIT_ROUNDOFF, wasted_time_bracket
from flakerank.history import RunHistory, history_from_mapping

# ---------------------------------------------------------------------------
# Hand-calculated known answers
# ---------------------------------------------------------------------------


def test_bracket_on_the_tiny_history(tiny: RunHistory) -> None:
    """One failure of 2.0 s; durations are 2.0, 1.0, 0.5, 1.0, 0.5 s.

    Hand arithmetic:
        F = 1 failed attempt
        t_min over ALL attempts = 0.5 s, t_max = 2.0 s
        order-statistic bracket = [1 * 0.5, 1 * 2.0] = [0.5, 2.0] s
        sum t^2 = 4.0 + 1.0 + 0.25 + 1.0 + 0.25 = 6.5 s^2
        Cauchy-Schwarz upper = sqrt(1 * 6.5) = 2.5495097567963922 s
        tightest upper = min(2.0, 2.5495...) = 2.0 s
        accumulated = 2.0 s, which sits exactly on the upper bound, as it must
        when the single wasted attempt IS the longest attempt in the history.
    """
    bracket = wasted_time_bracket(tiny, 2.0)
    assert bracket.failed_attempts == 1
    assert bracket.min_attempt_seconds == pytest.approx(0.5, rel=0.0, abs=1e-12)
    assert bracket.max_attempt_seconds == pytest.approx(2.0, rel=0.0, abs=1e-12)
    assert bracket.order_statistic_low == pytest.approx(0.5, rel=0.0, abs=1e-12)
    assert bracket.order_statistic_high == pytest.approx(2.0, rel=0.0, abs=1e-12)
    assert bracket.second_moment_seconds_squared == pytest.approx(6.5, rel=0.0, abs=1e-12)
    assert bracket.cauchy_schwarz_high == pytest.approx(
        float(np.sqrt(6.5)), rel=1e-15, abs=0.0
    )
    assert bracket.tightest_high_seconds == pytest.approx(2.0, rel=0.0, abs=1e-12)
    assert bracket.brackets


def test_slack_formula_is_the_declared_one(tiny: RunHistory) -> None:
    """slack = SAFETY * F * eps * sum|t|, equation (3).

    Hand arithmetic on the fixture: sum|t| = 2.0 + 1.0 + 0.5 + 1.0 + 0.5 = 5.0 s,
    F = 1, so slack = 4 * 1 * 2**-53 * 5.0 = 2.220446049250313e-15 s.
    """
    bracket = wasted_time_bracket(tiny, 2.0)
    assert bracket.absolute_total_seconds == pytest.approx(5.0, rel=0.0, abs=1e-12)
    assert bracket.slack_seconds == pytest.approx(
        SAFETY * 1 * UNIT_ROUNDOFF * 5.0, rel=1e-15, abs=0.0
    )
    assert bracket.slack_seconds == pytest.approx(2.220446049250313e-15, rel=1e-13, abs=0.0)


def test_slack_scales_with_the_conditioning_not_with_a_constant() -> None:
    """Scaling every duration by 1000 scales the slack by 1000.

    The build guide requires the slack to follow the conditioning of the bound
    expression and not an absolute constant, because an absolute slack floored
    at 1.8e-15 once made a bracket of width 4.5e-23 vacuous. This test is the
    statement that the slack here is proportional to the magnitude being
    summed.
    """
    from _support import tiny_history

    small = history_from_mapping(
        tiny_history(attempts=[(0, 0, 1, 1, 10), (0, 0, 2, 0, 10), (0, 1, 1, 0, 10)])
    )
    large = history_from_mapping(
        tiny_history(
            attempts=[(0, 0, 1, 1, 10000), (0, 0, 2, 0, 10000), (0, 1, 1, 0, 10000)]
        )
    )
    small_slack = wasted_time_bracket(small, 0.01).slack_seconds
    large_slack = wasted_time_bracket(large, 10.0).slack_seconds
    assert large_slack / small_slack == pytest.approx(1000.0, rel=1e-12, abs=0.0)


# ---------------------------------------------------------------------------
# The real history
# ---------------------------------------------------------------------------


def test_bracket_contains_the_accumulated_total(history: RunHistory) -> None:
    """The bracket from a failure count and order statistics holds the total.

    Measured on the committed history: 1417 failures, durations from 0.011 s to
    1848.115 s, accumulated 253493.035 s, order-statistic bracket
    [15.587, 2618178.955] s. The check is reported with its numbers by
    ``validation/validate_bounds.py``.
    """
    truth = accumulated_wasted_time(history, resamples=20, seed=5)
    bracket = wasted_time_bracket(history, truth.total_seconds)
    assert bracket.brackets
    assert bracket.margin_low_seconds > 0.0
    assert bracket.margin_high_seconds > 0.0
    assert bracket.failed_attempts == 1417


def test_cauchy_schwarz_is_the_tighter_upper_bound(history: RunHistory) -> None:
    """On the real history Cauchy-Schwarz beats the maximum-times-count bound.

    Hand reasoning: the maximum bound pays ``t_max`` for every failure, while
    Cauchy-Schwarz pays the quadratic mean, which is smaller whenever the
    durations are not all equal. The committed history spans four orders of
    magnitude, so the gap is large and is reported by
    ``validation/validate_bounds.py``.
    """
    truth = accumulated_wasted_time(history, resamples=20, seed=5)
    bracket = wasted_time_bracket(history, truth.total_seconds)
    assert bracket.cauchy_schwarz_high < bracket.order_statistic_high
    assert bracket.tightest_high_seconds == bracket.cauchy_schwarz_high


def test_the_slack_is_not_doing_the_work(history: RunHistory) -> None:
    """The slack is many orders of magnitude below the bracket width.

    A bound check whose slack is comparable to its width is vacuous. Measured
    on the committed history the slack is below 1e-5 s against a width above
    1e5 s, so the ratio is below 1e-9. The threshold is stated as 1e-6 to leave
    room without making the assertion meaningless.
    """
    truth = accumulated_wasted_time(history, resamples=20, seed=5)
    bracket = wasted_time_bracket(history, truth.total_seconds)
    width = bracket.tightest_high_seconds - bracket.order_statistic_low
    assert width > 0.0
    assert bracket.slack_seconds / width < 1e-6


@given(st.integers(min_value=1, max_value=300), st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=60, deadline=None)
def test_bracket_holds_on_random_windows(size: int, seed: int) -> None:
    """The bracket holds for an arbitrary synthetic history, not just the real one.

    Built from integer millisecond durations and random outcomes, so every
    input is of the kind the loader accepts. The bracket is a mathematical
    consequence of counting and order statistics, so it must hold on every
    such history; a failure would mean the accumulator and the bound disagree
    about what is being summed.
    """
    from _support import tiny_history

    generator = np.random.default_rng(seed)
    attempts: list[tuple[int, int, int, int, int]] = []
    for session in range(size):
        failed = bool(generator.integers(0, 2))
        attempts.append((session, 0, 1, int(failed), int(generator.integers(1, 100000))))
        if failed:
            attempts.append((session, 0, 2, 0, int(generator.integers(1, 100000))))
    history = history_from_mapping(
        tiny_history(attempts=attempts, n_tests=1, n_sessions=size)
    )
    truth = accumulated_wasted_time(history, resamples=10, seed=seed % 1000)
    bracket = wasted_time_bracket(history, truth.total_seconds)
    assert bracket.brackets


# ---------------------------------------------------------------------------
# Negative controls, anchored on the bound and asserted to fire
# ---------------------------------------------------------------------------


def test_control_a_total_above_the_upper_bound_is_caught(history: RunHistory) -> None:
    """CONTROL, must fire: a total set to 1.5x the upper bound leaves the bracket.

    Anchored on the BOUND, not on the accumulator's output: the injected value
    is ``1.5 * tightest_high``, which is strictly above ``tightest_high +
    slack`` because the slack is twelve orders of magnitude smaller. An offset
    applied to the accumulated value would not be guaranteed to leave the
    bracket, which is how an earlier product of this mission shipped two
    silently weak controls.
    """
    truth = accumulated_wasted_time(history, resamples=10, seed=5)
    clean = wasted_time_bracket(history, truth.total_seconds)
    assert clean.brackets, "the control is only meaningful if the clean case passes"
    injected = 1.5 * clean.tightest_high_seconds
    assert injected > clean.tightest_high_seconds + clean.slack_seconds
    fired = not wasted_time_bracket(history, injected).brackets
    assert fired, "the upper-bound control did not fire"


def test_control_a_total_below_the_lower_bound_is_caught(history: RunHistory) -> None:
    """CONTROL, must fire: a total of half the lower bound leaves the bracket.

    Anchored on the bound again: ``0.5 * order_statistic_low`` is strictly
    below ``order_statistic_low - slack`` whenever the lower bound exceeds
    twice the slack, which is asserted here rather than assumed.
    """
    truth = accumulated_wasted_time(history, resamples=10, seed=5)
    clean = wasted_time_bracket(history, truth.total_seconds)
    assert clean.order_statistic_low > 2.0 * clean.slack_seconds
    injected = 0.5 * clean.order_statistic_low
    assert injected < clean.order_statistic_low - clean.slack_seconds
    fired = not wasted_time_bracket(history, injected).brackets
    assert fired, "the lower-bound control did not fire"


def test_control_a_doubled_accumulator_is_caught_on_a_narrow_history() -> None:
    """CONTROL, must fire: on a near-constant-duration history, doubling is caught.

    The bracket is as wide as the duration spread, so on the real history a
    doubled total still fits inside it -- which is a real limitation of the
    bound and is recorded in README.md. The control therefore uses a history
    whose durations are nearly constant, where the bracket IS tight: 40
    attempts of 1000 ms each, 20 of them failing, gives an order-statistic
    bracket of exactly [20.0, 20.0] s and an accumulated total of 20.0 s, so
    any multiplicative error at all leaves it.
    """
    from _support import tiny_history

    attempts: list[tuple[int, int, int, int, int]] = []
    for session in range(20):
        attempts.append((session, 0, 1, 1, 1000))
        attempts.append((session, 0, 2, 0, 1000))
    history = history_from_mapping(
        tiny_history(attempts=attempts, n_tests=1, n_sessions=20)
    )
    truth = accumulated_wasted_time(history, resamples=10, seed=1)
    assert truth.total_seconds == pytest.approx(20.0, rel=0.0, abs=1e-12)
    clean = wasted_time_bracket(history, truth.total_seconds)
    assert clean.brackets
    assert clean.order_statistic_low == pytest.approx(20.0, rel=0.0, abs=1e-12)
    assert clean.order_statistic_high == pytest.approx(20.0, rel=0.0, abs=1e-12)
    fired = not wasted_time_bracket(history, 2.0 * truth.total_seconds).brackets
    assert fired, "the doubled-accumulator control did not fire"


def test_control_the_bracket_does_not_fire_on_the_true_value(history: RunHistory) -> None:
    """CONTROL, must STAY SILENT: the accumulated total passes.

    Both directions of every control are asserted, because a check that cannot
    pass is as useless as one that cannot fail.
    """
    truth = accumulated_wasted_time(history, resamples=10, seed=5)
    assert wasted_time_bracket(history, truth.total_seconds).brackets


def test_control_the_bracket_does_not_fire_on_the_bound_itself(history: RunHistory) -> None:
    """CONTROL, must STAY SILENT: a total sitting exactly on the upper bound passes.

    The bracket is a closed interval, so a value equal to an end is inside it.
    That is the right convention: both ends are attainable -- the tiny-history
    case above attains the upper end exactly -- and excluding them would make
    the verdict depend on a floating-point tie.
    """
    truth = accumulated_wasted_time(history, resamples=10, seed=5)
    clean = wasted_time_bracket(history, truth.total_seconds)
    assert wasted_time_bracket(history, clean.tightest_high_seconds).brackets
    assert wasted_time_bracket(history, clean.order_statistic_low).brackets


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_accumulated_total_rejected(tiny: RunHistory, value: float) -> None:
    """Wasted time is a sum of non-negative durations."""
    with pytest.raises(ValueError, match="accumulated_seconds"):
        wasted_time_bracket(tiny, value)


@pytest.mark.parametrize("value", ["2.0", None, True])
def test_non_numeric_accumulated_total_rejected(tiny: RunHistory, value: object) -> None:
    """A string or a bool is a TypeError."""
    with pytest.raises(TypeError):
        wasted_time_bracket(tiny, value)  # type: ignore[arg-type]


def test_relative_width_is_infinite_at_a_zero_total(tiny: RunHistory) -> None:
    """A zero total has no relative bracket width, reported as ``inf``."""
    from _support import tiny_history

    history = history_from_mapping(
        tiny_history(attempts=[(0, 0, 1, 0, 1000), (0, 1, 1, 0, 1000)])
    )
    bracket = wasted_time_bracket(history, 0.0)
    assert bracket.failed_attempts == 0
    assert np.isinf(bracket.relative_width)
