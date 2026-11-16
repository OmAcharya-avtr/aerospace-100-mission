"""The closed-form detection power, its inversions and its independent brackets.

Known-answer tests carry the hand arithmetic in the test body. The headline
constant is ``1 - (43/44)**2``: 43/44 = 0.9772727272727273, squared is
0.9550619834710744, so the answer is 0.044938016528925595. That is the
detection probability a run-twice reproducibility check earns at the rate this
mission shipped.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from seedaudit import power

RATE = power.MISSION_RATE


def test_mission_rate_is_one_in_forty_four() -> None:
    assert RATE == pytest.approx(1.0 / 44.0, rel=0.0, abs=0.0)


def test_known_answer_two_runs_at_the_mission_rate() -> None:
    # Hand arithmetic: (43/44)**2 = 0.9550619834710744, so the reference-design
    # answer is 0.044938016528925595; the pairwise design subtracts (1/44)**2.
    assert power.detection_probability_reference(RATE, 2) == pytest.approx(
        0.044938016528925595, rel=1e-15, abs=0.0
    )
    assert power.detection_probability_pairwise(RATE, 2) == pytest.approx(
        0.044421487603305776, rel=1e-13, abs=0.0
    )
    assert power.detection_probability_pairwise(RATE, 2) == pytest.approx(
        1.0 - (1.0 / 44.0) ** 2 - (43.0 / 44.0) ** 2, rel=1e-13, abs=0.0
    )


def test_the_two_designs_differ_by_exactly_p_to_the_n() -> None:
    gap = power.detection_probability_reference(0.3, 4) - power.detection_probability_pairwise(
        0.3, 4
    )
    assert gap == pytest.approx(0.3**4, rel=1e-12, abs=0.0)
    # The case where conflating the designs costs a third of the answer.
    assert power.detection_probability_reference(0.5, 2) == pytest.approx(0.75, rel=1e-15, abs=0.0)
    assert power.detection_probability_pairwise(0.5, 2) == pytest.approx(0.5, rel=1e-15, abs=0.0)


def test_degenerate_rates_and_repeat_counts() -> None:
    # One run has nothing to disagree with; at rate 1 every run manifests, so
    # run-and-diff with no reference is blind to a change that happens always.
    assert power.detection_probability_pairwise(0.4, 1) == 0.0
    assert power.detection_probability_pairwise(1.0, 10) == 0.0
    assert power.detection_probability_pairwise(0.0, 1000) == 0.0
    assert power.detection_probability_reference(1.0, 1) == 1.0
    assert power.detection_probability_reference(0.0, 1000) == 0.0


def test_pairwise_is_symmetric_under_rate_complement() -> None:
    assert power.detection_probability_pairwise(0.017, 6) == pytest.approx(
        power.detection_probability_pairwise(1.0 - 0.017, 6), rel=1e-12, abs=0.0
    )


def test_reference_is_monotone_in_both_arguments() -> None:
    over_n = [power.detection_probability_reference(0.05, n) for n in range(1, 40)]
    over_p = [power.detection_probability_reference(p, 5) for p in np.linspace(0.0, 1.0, 50)]
    assert all(b >= a for a, b in zip(over_n, over_n[1:], strict=False))
    assert all(b >= a for a, b in zip(over_p, over_p[1:], strict=False))


def test_reference_is_stable_where_the_naive_form_loses_the_answer() -> None:
    assert power.detection_probability_reference(1e-17, 3) == pytest.approx(
        3e-17, rel=1e-9, abs=0.0
    )
    assert 1.0 - (1.0 - 1e-17) ** 3 == 0.0


def test_the_escaped_rate_bound_is_the_smallest_detectable_rate() -> None:
    # Two claims, numerically one expression: the smallest rate the design
    # catches with probability 1-alpha is the largest rate consistent with
    # catching nothing. The identity is checked, not assumed.
    for n in (1, 2, 3, 7, 44, 131, 1000):
        assert power.escaped_rate_upper_bound(n) == pytest.approx(
            power.smallest_detectable_rate(n), rel=1e-15, abs=0.0
        )
    assert power.smallest_detectable_rate(2) == pytest.approx(
        0.7763932022500211, rel=1e-14, abs=0.0
    )
    assert power.escaped_rate_upper_bound(1, 0.05) == pytest.approx(0.95, rel=1e-15, abs=0.0)


def test_the_escaped_rate_bound_decreases_towards_the_rule_of_three() -> None:
    values = [power.escaped_rate_upper_bound(n) for n in range(1, 60)]
    assert all(b < a for a, b in zip(values, values[1:], strict=False))
    assert power.escaped_rate_upper_bound(1000) == pytest.approx(
        3.0 / 1000.0, rel=5e-3, abs=0.0
    )


def test_repeats_for_detection_at_the_mission_rate_is_one_hundred_and_thirty_one() -> None:
    # log(0.05)/log(43/44) = 130.3086..., so 131 runs, and 130 is not enough.
    assert power.repeats_for_detection(RATE) == 131
    assert power.detection_probability_reference(RATE, 131) >= 0.95
    assert power.detection_probability_reference(RATE, 130) < 0.95
    assert power.repeats_for_detection(1.0) == 1


def test_repeats_for_an_escaped_rate_bound_is_minimal() -> None:
    for target in (0.5, 0.1, 0.02, 0.005):
        n = power.repeats_for_escaped_rate_bound(target)
        assert power.escaped_rate_upper_bound(n) <= target
        assert power.escaped_rate_upper_bound(n - 1) > target
    assert power.repeats_for_escaped_rate_bound(1.0) == 1


def test_the_union_bound_sits_above_the_closed_form() -> None:
    for n in (2, 3, 10, 50):
        for p in (0.001, 0.01, RATE, 0.1, 0.4):
            assert power.union_bound_reference(p, n) >= power.detection_probability_reference(p, n)
    assert power.union_bound_reference(0.5, 10) == 1.0


def test_the_bonferroni_bracket_is_degenerate_at_two_runs() -> None:
    # With two events S1 - S2 = 2p - p**2 = 1 - (1-p)**2 exactly, so the
    # bracket carries no information at n = 2 and the test below uses n >= 3.
    low, high = power.bonferroni_bracket_reference(RATE, 2)
    exact = power.detection_probability_reference(RATE, 2)
    assert low == pytest.approx(exact, rel=1e-14, abs=0.0)
    assert high >= exact


def test_the_bonferroni_bracket_contains_the_closed_form() -> None:
    for n in (3, 4, 8, 20):
        for p in (1e-5, 1e-3, RATE, 0.05):
            low, high = power.bonferroni_bracket_reference(p, n)
            exact = power.detection_probability_reference(p, n)
            # Slack follows the CONDITIONING of the bracket, set by the third
            # inclusion-exclusion term C(n,3)*p**3, not by an absolute constant
            # and not by the operation count.
            slack = 4.0 * math.comb(n, 3) * p**3 + 8.0 * np.finfo(float).eps * max(exact, 1e-300)
            assert low - slack <= exact <= high + slack


def test_the_exponential_bracket_contains_the_closed_form() -> None:
    for n in (1, 2, 5, 30, 200):
        for p in (1e-6, 1e-3, RATE, 0.2, 0.7):
            low, high = power.exponential_bracket_reference(p, n)
            exact = power.detection_probability_reference(p, n)
            slack = 16.0 * np.finfo(float).eps * max(exact, 1e-300)
            assert low - slack <= exact <= high + slack
    low, high = power.exponential_bracket_reference(1e-9, 2)
    assert high - low < 1e-17
    assert power.exponential_bracket_reference(1.0, 4) == (1.0, 1.0)


def test_the_seed_axis_is_hypergeometric_and_not_binomial() -> None:
    # Population 10, 5 sensitive, 2 drawn without replacement:
    # miss = C(5,2)/C(10,2) = 10/45, so detection = 35/45, not 1 - 0.5**2.
    assert power.seed_axis_detection(10, 5, 2) == pytest.approx(35.0 / 45.0, rel=1e-15, abs=0.0)
    assert power.seed_axis_detection(10, 5, 2) != pytest.approx(0.75, rel=1e-6, abs=0.0)
    assert power.seed_axis_detection(100, 100, 1) == 1.0
    assert power.seed_axis_detection(100, 0, 1) == 0.0
    assert power.seed_axis_detection(10, 6, 5) == 1.0
    assert power.seed_axis_detection(10, 5, 0) == 0.0


def test_monte_carlo_agrees_with_the_closed_forms() -> None:
    # The derivation is checked against a simulation rather than against the
    # algebra that produced it.
    rng = np.random.default_rng(1000044)
    ref, ref_se, pair, pair_se = power.monte_carlo_detection(RATE, 2, 200000, rng)
    assert abs(ref - power.detection_probability_reference(RATE, 2)) <= 4.0 * ref_se
    assert abs(pair - power.detection_probability_pairwise(RATE, 2)) <= 4.0 * pair_se
    ref2, ref2_se, pair2, pair2_se = power.monte_carlo_detection(0.5, 3, 100000, rng)
    assert abs(ref2 - power.detection_probability_reference(0.5, 3)) <= 4.0 * ref2_se
    assert abs(pair2 - power.detection_probability_pairwise(0.5, 3)) <= 4.0 * pair2_se
    assert ref_se >= 0.0 and pair_se >= 0.0


def test_the_power_table_matches_the_scalar_functions_and_keeps_order() -> None:
    rows = power.power_table((44, 2, 10), RATE)
    assert [row.repeats for row in rows] == [44, 2, 10]
    for row in rows:
        assert row.reference == power.detection_probability_reference(RATE, row.repeats)
        assert row.pairwise == power.detection_probability_pairwise(RATE, row.repeats)
        assert row.escaped_upper_bound == power.escaped_rate_upper_bound(row.repeats)
    with pytest.raises(ValueError, match="non-empty"):
        power.power_table(())


def test_the_bootstrap_interval_brackets_its_point_estimate() -> None:
    rng = np.random.default_rng(1000044)
    values = np.zeros(400)
    values[:20] = 1.0
    point, low, high = power.bootstrap_interval(values, rng, 2000)
    assert point == pytest.approx(0.05, rel=0.0, abs=0.0)
    assert low <= point <= high
    assert high > low
    assert power.bootstrap_interval(np.ones(50), np.random.default_rng(7), 500) == (1.0, 1.0, 1.0)


def test_the_bootstrap_interval_is_reproducible_and_validated() -> None:
    values = np.array([0.0, 1.0] * 50)
    assert power.bootstrap_interval(values, np.random.default_rng(1000044), 500) == (
        power.bootstrap_interval(values, np.random.default_rng(1000044), 500)
    )
    with pytest.raises(ValueError, match="only 0 and 1"):
        power.bootstrap_interval(np.array([0.0, 0.5, 1.0]), np.random.default_rng(1))
    with pytest.raises(ValueError, match="non-empty"):
        power.bootstrap_interval(np.array([]), np.random.default_rng(1))
    with pytest.raises(ValueError, match="confidence"):
        power.bootstrap_interval(np.ones(4), np.random.default_rng(1), 10, confidence=1.0)


@pytest.mark.parametrize("bad", [-0.001, 1.001, float("nan"), float("inf")])
def test_a_rate_outside_the_unit_interval_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        power.detection_probability_reference(bad, 2)


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_a_non_positive_repeat_count_is_rejected(bad: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        power.detection_probability_reference(0.1, bad)


@pytest.mark.parametrize("bad", [2.5, "2", None, True])
def test_a_non_integer_repeat_count_is_rejected(bad: object) -> None:
    with pytest.raises(TypeError):
        power.detection_probability_reference(0.1, bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_an_alpha_outside_the_open_unit_interval_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        power.smallest_detectable_rate(5, bad)


def test_the_remaining_input_validation() -> None:
    with pytest.raises(ValueError, match="never detected"):
        power.repeats_for_detection(0.0)
    with pytest.raises(ValueError, match="power"):
        power.repeats_for_detection(0.1, 1.0)
    with pytest.raises(ValueError, match="bounds the rate at 0"):
        power.repeats_for_escaped_rate_bound(0.0)
    with pytest.raises(ValueError, match="sensitive"):
        power.seed_axis_detection(5, 6, 1)
    with pytest.raises(ValueError, match="draws"):
        power.seed_axis_detection(5, 1, 6)
    with pytest.raises(TypeError):
        power.seed_axis_detection(5, 1, 1.5)  # type: ignore[arg-type]


@given(
    p=st.floats(min_value=1e-12, max_value=1.0 - 1e-12, allow_nan=False),
    n=st.integers(min_value=1, max_value=400),
)
@settings(max_examples=250, deadline=None)
def test_property_reference_is_bracketed_by_the_exponential_bound(p: float, n: int) -> None:
    low, high = power.exponential_bracket_reference(p, n)
    exact = power.detection_probability_reference(p, n)
    slack = 32.0 * np.finfo(float).eps * max(exact, 1e-300)
    assert low - slack <= exact <= high + slack


@given(
    p=st.floats(min_value=1e-12, max_value=1.0 - 1e-12, allow_nan=False),
    n=st.integers(min_value=1, max_value=200),
)
@settings(max_examples=250, deadline=None)
def test_property_pairwise_never_exceeds_reference(p: float, n: int) -> None:
    pair = power.detection_probability_pairwise(p, n)
    ref = power.detection_probability_reference(p, n)
    assert pair <= ref + 32.0 * np.finfo(float).eps


@given(n=st.integers(min_value=1, max_value=5000))
@settings(max_examples=200, deadline=None)
def test_property_the_escaped_bound_recovers_its_own_detection_probability(n: int) -> None:
    bound = power.escaped_rate_upper_bound(n, 0.05)
    assert power.detection_probability_reference(bound, n) == pytest.approx(
        0.95, rel=1e-9, abs=0.0
    )


@given(p=st.floats(min_value=1e-9, max_value=0.5, allow_nan=False))
@settings(max_examples=150, deadline=None)
def test_property_repeats_for_detection_is_minimal(p: float) -> None:
    n = power.repeats_for_detection(p, 0.95)
    assert power.detection_probability_reference(p, n) >= 0.95
    if n > 1:
        assert power.detection_probability_reference(p, n - 1) < 0.95


def test_regression_pinned_literals_quoted_in_the_readme() -> None:
    # Pinned as literals because .hypothesis/ is gitignored, so nothing a
    # property run finds survives a cold clone unless it is written here. The
    # argument that a two-run reproducibility report is uninformative rests on
    # the first of these.
    assert power.escaped_rate_upper_bound(2, 0.05) == pytest.approx(
        0.7763932022500211, rel=1e-15, abs=0.0
    )
    assert power.detection_probability_pairwise(1.0, 2) == 0.0
    assert power.detection_probability_pairwise(1.0, 440) == 0.0
    assert power.escaped_rate_upper_bound(131, 0.05) == pytest.approx(
        0.022608690169639987, rel=1e-13, abs=0.0
    )


def test_agreement_probability_is_the_complement_of_pairwise_detection() -> None:
    for p in (0.0, 1e-4, RATE, 0.5, 0.9, 1.0):
        for n in (1, 2, 5, 20):
            assert power.agreement_probability(p, n) == pytest.approx(
                1.0 - power.detection_probability_pairwise(p, n), rel=1e-14, abs=1e-15
            )
    # n = 1: one run always agrees with itself.
    assert power.agreement_probability(0.3, 1) == pytest.approx(1.0, rel=0.0, abs=1e-15)


def test_the_classifier_misclassifies_a_merely_non_deterministic_script() -> None:
    # Measured first, then derived. Running the two-arrangement classifier over
    # the injected corpus at n = 2 classified an intrinsic rate-0.5 script as a
    # self-polluting check: its pristine pair happened to agree and its reused
    # pair happened to differ. q = p**n + (1-p)**n = 0.5 at p = 0.5, n = 2, so
    # q(1-q) = 0.25.
    assert power.self_pollution_misclassification(0.5, 2) == pytest.approx(
        0.25, rel=1e-14, abs=0.0
    )
    assert power.self_pollution_misclassification(0.0, 2) == 0.0
    assert power.self_pollution_misclassification(1.0, 2) == 0.0
    assert power.self_pollution_misclassification(RATE, 2) == pytest.approx(
        power.agreement_probability(RATE, 2) * power.detection_probability_pairwise(RATE, 2),
        rel=1e-14,
        abs=0.0,
    )


def test_more_runs_per_arrangement_do_not_lower_the_worst_case() -> None:
    # q(1-q) peaks at 0.25 when q = 0.5, and q = p**n + (1-p)**n reaches 0.5
    # for every n >= 2. Raising the repeat count moves the rate at which the
    # classifier is worst and does not lower how bad it gets there.
    for n in (2, 3, 5, 10, 20, 100):
        assert power.worst_case_misclassification(n) == pytest.approx(0.25, rel=0.0, abs=0.0)
        worst_rate = power.worst_case_misclassification_rate(n)
        assert power.agreement_probability(worst_rate, n) == pytest.approx(
            0.5, rel=1e-9, abs=0.0
        )
        assert power.self_pollution_misclassification(worst_rate, n) == pytest.approx(
            0.25, rel=1e-8, abs=0.0
        )
    assert power.worst_case_misclassification(1) == 0.0


def test_the_worst_case_rate_moves_towards_zero_as_repeats_grow() -> None:
    rates = [power.worst_case_misclassification_rate(n) for n in (2, 3, 5, 10, 20)]
    assert rates[0] == pytest.approx(0.5, rel=0.0, abs=0.0)
    assert all(b < a for a, b in zip(rates, rates[1:], strict=False))


def test_independent_repetitions_of_the_whole_experiment_do_lower_it() -> None:
    # The mitigation is more TRIALS, not more runs per arrangement.
    for trials in (1, 2, 3, 4):
        assert power.worst_case_misclassification(2, trials) == pytest.approx(
            0.25**trials, rel=1e-14, abs=0.0
        )
    assert power.self_pollution_misclassification(0.5, 2, 3) == pytest.approx(
        0.015625, rel=1e-14, abs=0.0
    )


def test_the_misclassification_model_validates_its_input() -> None:
    with pytest.raises(ValueError, match="cannot fire below 2 runs"):
        power.worst_case_misclassification_rate(1)
    with pytest.raises(ValueError, match="at least 1"):
        power.self_pollution_misclassification(0.5, 2, 0)
    with pytest.raises(ValueError):
        power.agreement_probability(1.5, 2)


def test_the_duration_marker_is_a_line_local_keyword_test() -> None:
    # It GROUPS findings and never removes one. A line whose unit lives in a
    # column header elsewhere carries no marker and is counted as unmarked, so
    # the unmarked count is an upper bound on the non-timing findings.
    from seedaudit.report import DURATION_MARKERS, looks_like_a_duration

    assert looks_like_a_duration("wall clock : <#> s")
    assert looks_like_a_duration("one simulation : <#> ms")
    assert looks_like_a_duration("elapsed: <#> s on this machine")
    assert looks_like_a_duration("<#> energy [W*h] <#>")
    assert looks_like_a_duration("difficulty estimate : <#> simulations, <#> s")
    assert not looks_like_a_duration("mean estimate <#> of <#> draws")
    assert not looks_like_a_duration("tiny-enumerable <#> <#> <#>")
    assert not looks_like_a_duration("climatology baseline <#> <#> <#>")
    assert len(set(DURATION_MARKERS)) == len(DURATION_MARKERS)
