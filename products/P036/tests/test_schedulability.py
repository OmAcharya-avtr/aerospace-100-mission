"""Schedulability tests with every expected number hand-computed in a comment.

The task set used throughout is::

    t1: T = 7 s,  C = 3 s
    t2: T = 12 s, C = 3 s
    t3: T = 20 s, C = 5 s

chosen because its utilization sits above the Liu & Layland bound but it is
schedulable, so it separates a sufficient test from an exact one.
"""

from __future__ import annotations

import math

import pytest

from rtclock.schedulability import (
    edf_test,
    hyperbolic_bound_test,
    priority_ceiling_blocking,
    response_time,
    response_time_analysis,
    rm_utilization_bound,
    rm_utilization_test,
)
from rtclock.taskset import PeriodicTask, TaskSet

# Liu & Layland 1973, Theorem 5: U_lub(n) = n (2^(1/n) - 1).
# Hand values, computed to 12 decimals:
#   n = 1 : 1 * (2       - 1) = 1.000000000000
#   n = 2 : 2 * (1.41421356237310 - 1) = 0.828427124746
#   n = 3 : 3 * (1.25992104989487 - 1) = 0.779763149685
#   n = 4 : 4 * (1.18920711500272 - 1) = 0.756828460011
#   n = 5 : 5 * (1.14869835499704 - 1) = 0.743491774985
#   n = 10: 10 * (1.07177346253629 - 1) = 0.717734625363
RM_BOUND_HAND = {
    1: 1.000000000000,
    2: 0.828427124746,
    3: 0.779763149685,
    4: 0.756828460011,
    5: 0.743491774985,
    10: 0.717734625363,
}


@pytest.mark.parametrize(("n", "expected"), sorted(RM_BOUND_HAND.items()))
def test_rm_bound_hand_values(n, expected):
    assert rm_utilization_bound(n) == pytest.approx(expected, abs=5e-13)


def test_rm_bound_n_equals_two_is_the_textbook_0_8284():
    # The most-quoted single value of the bound.
    assert rm_utilization_bound(2) == pytest.approx(0.8284, abs=5e-5)


def test_rm_bound_is_strictly_decreasing_and_tends_to_ln2():
    bounds = [rm_utilization_bound(n) for n in range(1, 2001)]
    assert all(b > a for a, b in zip(bounds[1:], bounds[:-1], strict=True))
    assert all(b > math.log(2.0) for b in bounds)
    # Asymptotics: 2^(1/n) = exp(ln2/n) = 1 + ln2/n + (ln2)^2/(2n^2) + ...
    # so n(2^(1/n)-1) = ln2 + (ln2)^2/(2n) + O(1/n^2). Hand value of the
    # leading correction at n = 2000: (0.6931471805599453)^2 / (2*2000)
    #   = 0.4804530139182014 / 4000 = 1.201132534795e-04.
    gap = bounds[-1] - math.log(2.0)
    assert gap == pytest.approx(1.201132534795e-04, rel=2e-4)
    assert rm_utilization_bound(10**6) == pytest.approx(math.log(2.0), abs=2.5e-7)


def test_rm_bound_validates_input():
    with pytest.raises(ValueError, match="n must be >= 1"):
        rm_utilization_bound(0)
    with pytest.raises(TypeError, match="must be an int"):
        rm_utilization_bound(3.0)
    with pytest.raises(TypeError, match="must be an int"):
        rm_utilization_bound(True)


def standard_set() -> TaskSet:
    """t1/t2/t3 with rate-monotonic priorities."""
    return TaskSet(
        [
            PeriodicTask("t1", 7.0, 3.0),
            PeriodicTask("t2", 12.0, 3.0),
            PeriodicTask("t3", 20.0, 5.0),
        ]
    ).rate_monotonic()


# Hand computation: U = 3/7 + 3/12 + 5/20
#                     = 0.428571428571 + 0.25 + 0.25 = 0.928571428571
def test_standard_set_utilization_hand_value():
    assert standard_set().total_utilization == pytest.approx(0.928571428571, abs=5e-13)


def test_liu_layland_bound_is_inconclusive_for_the_standard_set():
    # U = 0.9286 > 0.7798, so the sufficient test says nothing.
    result = rm_utilization_test(standard_set())
    assert result.schedulable is False
    assert result.strength == "sufficient"
    assert result.bound == pytest.approx(0.779763149685, abs=5e-13)
    assert result.margin == pytest.approx(0.779763149685 - 0.928571428571, abs=1e-12)


# Hand computation of the hyperbolic bound for the standard set:
#   (1 + 3/7)(1 + 1/4)(1 + 1/4) = (10/7)(5/4)(5/4) = 250/112 = 2.232142857143
def test_hyperbolic_bound_hand_value():
    result = hyperbolic_bound_test(standard_set())
    assert result.schedulable is False
    assert "2.232142857143" in result.detail


def test_hyperbolic_bound_is_tighter_than_liu_layland():
    # Two harmonic tasks at U = 0.5 each: U = 1.0 > 0.828, so L&L is
    # inconclusive. Hyperbolic: (1.5)(1.5) = 2.25 > 2, also inconclusive.
    # A set where hyperbolic accepts and L&L does not: U1 = 0.7, U2 = 0.1.
    # L&L: U = 0.8 <= 0.8284 -> accepted too. Use U1 = 0.75, U2 = 0.08:
    # L&L: 0.83 > 0.828427 -> rejected. Hyperbolic: 1.75 * 1.08 = 1.89 <= 2.
    ts = TaskSet(
        [PeriodicTask("a", 1.0, 0.75), PeriodicTask("b", 100.0, 8.0)]
    ).rate_monotonic()
    assert ts.total_utilization == pytest.approx(0.83, abs=1e-12)
    assert rm_utilization_test(ts).schedulable is False
    assert hyperbolic_bound_test(ts).schedulable is True


def test_edf_is_exact_for_implicit_deadlines():
    result = edf_test(standard_set())
    assert result.strength == "exact"
    assert result.schedulable is True
    assert result.bound == 1.0


def test_edf_density_test_is_only_sufficient_for_constrained_deadlines():
    ts = TaskSet([PeriodicTask("a", 10.0, 4.0, deadline_s=5.0)]).rate_monotonic()
    result = edf_test(ts)
    assert result.strength == "sufficient"
    assert "processor-demand" in result.detail


def test_utilization_bounds_refuse_constrained_deadlines():
    ts = TaskSet([PeriodicTask("a", 10.0, 1.0, deadline_s=5.0)]).rate_monotonic()
    with pytest.raises(ValueError, match="assumes D == T"):
        rm_utilization_test(ts)
    with pytest.raises(ValueError, match="assumes D == T"):
        hyperbolic_bound_test(ts)


def test_tests_reject_empty_sets():
    for fn in (rm_utilization_test, hyperbolic_bound_test, edf_test):
        with pytest.raises(ValueError, match="at least one task"):
            fn(TaskSet([]))
    with pytest.raises(ValueError, match="at least one task"):
        response_time_analysis(TaskSet([]))


# Hand computation of the RTA recurrence (Joseph & Pandya 1986; the iteration
# form of Audsley et al. 1993), R = C + B + sum_j ceil(R/T_j) C_j:
#
#   t1 (highest, no interference): R = 3.                       3 <= 7   ok
#   t2: R^(0) = 3
#       R^(1) = 3 + ceil(3/7)*3  = 3 + 1*3 = 6
#       R^(2) = 3 + ceil(6/7)*3  = 3 + 1*3 = 6   -> fixed point R2 = 6
#                                                   6 <= 12  ok
#   t3: R^(0) = 5
#       R^(1) = 5 + ceil(5/7)*3  + ceil(5/12)*3  = 5 + 3 + 3 = 11
#       R^(2) = 5 + ceil(11/7)*3 + ceil(11/12)*3 = 5 + 6 + 3 = 14
#       R^(3) = 5 + ceil(14/7)*3 + ceil(14/12)*3 = 5 + 6 + 6 = 17
#       R^(4) = 5 + ceil(17/7)*3 + ceil(17/12)*3 = 5 + 9 + 6 = 20
#       R^(5) = 5 + ceil(20/7)*3 + ceil(20/12)*3 = 5 + 9 + 6 = 20
#               -> fixed point R3 = 20, exactly equal to D3 = 20: met.
RTA_HAND = {"t1": 3.0, "t2": 6.0, "t3": 20.0}
RTA_HAND_ITERATES_T3 = (5.0, 11.0, 14.0, 17.0, 20.0, 20.0)


def test_rta_matches_hand_computed_response_times():
    results = {r.name: r for r in response_time_analysis(standard_set())}
    for name, expected in RTA_HAND.items():
        assert results[name].response_s == pytest.approx(expected, abs=1e-12)
        assert results[name].converged is True
        assert results[name].meets_deadline is True
    assert results["t3"].slack_s == pytest.approx(0.0, abs=1e-12)


def test_rta_iterate_sequence_matches_the_hand_computation_step_by_step():
    results = {r.name: r for r in response_time_analysis(standard_set())}
    assert results["t3"].iterates == pytest.approx(RTA_HAND_ITERATES_T3, abs=1e-12)
    assert results["t2"].iterates == pytest.approx((3.0, 6.0, 6.0), abs=1e-12)
    assert results["t1"].iterates == pytest.approx((3.0, 3.0), abs=1e-12)


# Hand computation, harmonic set a(T=4,C=1), b(T=8,C=2), c(T=16,C=8):
#   U = 1/4 + 2/8 + 8/16 = 1.0, above the n=3 bound 0.7798, so the
#   sufficient test is inconclusive -- yet the set is schedulable, which is
#   the classic demonstration that the Liu & Layland bound is not necessary.
#   R_a = 1.                                                  1 <= 4  ok
#   R_b: R^(0) = 2; R^(1) = 2 + ceil(2/4)*1 = 3; R^(2) = 2 + ceil(3/4)*1 = 3
#        -> R_b = 3                                           3 <= 8  ok
#   R_c: R^(0) = 8
#        R^(1) = 8 + ceil(8/4)*1  + ceil(8/8)*2  = 8 + 2 + 2 = 12
#        R^(2) = 8 + ceil(12/4)*1 + ceil(12/8)*2 = 8 + 3 + 4 = 15
#        R^(3) = 8 + ceil(15/4)*1 + ceil(15/8)*2 = 8 + 4 + 4 = 16
#        R^(4) = 8 + ceil(16/4)*1 + ceil(16/8)*2 = 8 + 4 + 4 = 16
#        -> R_c = 16 = D_c: met.
def test_harmonic_set_at_u_equals_one_is_schedulable():
    ts = TaskSet(
        [PeriodicTask("a", 4.0, 1.0), PeriodicTask("b", 8.0, 2.0), PeriodicTask("c", 16.0, 8.0)]
    ).rate_monotonic()
    assert ts.total_utilization == pytest.approx(1.0, abs=1e-15)
    assert rm_utilization_test(ts).schedulable is False  # sufficient test fails
    results = {r.name: r.response_s for r in response_time_analysis(ts)}
    assert results == pytest.approx({"a": 1.0, "b": 3.0, "c": 16.0}, abs=1e-12)
    assert all(r.meets_deadline for r in response_time_analysis(ts))


# Hand computation of priority-ceiling blocking (Sha, Rajkumar & Lehoczky
# 1990). Four tasks, RM priorities t1=3 > t2=2 > t3=1 > t4=0. Semaphore s1 is
# used by t3 (0.001 s) and t4 (1.0 s), so ceiling(s1) = prio(t3) = 1.
#   B_1: lower-priority users of a semaphore with ceiling >= 3 -> none  -> 0
#   B_2: ceiling(s1) = 1 < 2                                   -> none  -> 0
#   B_3: t4 is lower priority and ceiling(s1) = 1 >= 1         -> 1.0 s
#   B_4: no lower-priority task                                        -> 0
# Then for t3: R = C + B + ceil(R/7)*3 + ceil(R/12)*3 with C = 5, B = 1:
#   R^(0) = 6
#   R^(1) = 6 + ceil(6/7)*3  + ceil(6/12)*3  = 6 + 3 + 3 = 12
#   R^(2) = 6 + ceil(12/7)*3 + ceil(12/12)*3 = 6 + 6 + 3 = 15
#   R^(3) = 6 + ceil(15/7)*3 + ceil(15/12)*3 = 6 + 9 + 6 = 21
#   R^(4) = 6 + ceil(21/7)*3 + ceil(21/12)*3 = 6 + 9 + 6 = 21
#   -> R_3 = 21 > D_3 = 20: the 1 s blocking term turns a set that just met
#      its deadline into one that misses it by 1 s.
def test_priority_ceiling_blocking_and_its_effect_on_rta():
    ts = TaskSet(
        [
            PeriodicTask("t1", 7.0, 3.0),
            PeriodicTask("t2", 12.0, 3.0),
            PeriodicTask("t3", 20.0, 5.0),
            PeriodicTask("t4", 50.0, 5.0),
        ]
    ).rate_monotonic()
    blocking = priority_ceiling_blocking(ts, {"t4": {"s1": 1.0}, "t3": {"s1": 0.001}})
    assert blocking == pytest.approx({"t1": 0.0, "t2": 0.0, "t3": 1.0, "t4": 0.0}, abs=0.0)
    results = {r.name: r for r in response_time_analysis(ts, blocking_s=blocking)}
    assert results["t3"].response_s == pytest.approx(21.0, abs=1e-12)
    assert results["t3"].meets_deadline is False
    assert results["t3"].slack_s == pytest.approx(-1.0, abs=1e-12)


# Hand computation: a semaphore used by the highest-priority task t1 and the
# lowest t3 has ceiling(s1) = prio(t1) = 2, so it blocks t1 and t2 alike:
#   B_1 = 1.0 (t3 holds s1, ceiling 2 >= 2)
#   B_2 = 1.0 (t3 lower priority, ceiling 2 >= 1)
#   B_3 = 0   (nothing below it)
def test_ceiling_above_a_task_blocks_it_even_without_sharing():
    ts = standard_set()
    blocking = priority_ceiling_blocking(ts, {"t3": {"s1": 1.0}, "t1": {"s1": 0.001}})
    assert blocking == pytest.approx({"t1": 1.0, "t2": 1.0, "t3": 0.0}, abs=0.0)


def test_blocking_takes_the_maximum_not_the_sum():
    # PCP blocks a task at most once per release, so two critical sections of a
    # lower-priority task contribute max(0.4, 0.9) = 0.9, not 1.3.
    ts = standard_set()
    blocking = priority_ceiling_blocking(
        ts, {"t3": {"s1": 0.4, "s2": 0.9}, "t1": {"s1": 0.001, "s2": 0.001}}
    )
    assert blocking["t1"] == pytest.approx(0.9, abs=0.0)


def test_blocking_with_explicit_ceilings():
    ts = standard_set()
    blocking = priority_ceiling_blocking(ts, {"t3": {"s1": 0.5}}, ceilings={"s1": 2})
    assert blocking["t1"] == pytest.approx(0.5, abs=0.0)
    # A ceiling below t1's priority cannot block t1.
    blocking = priority_ceiling_blocking(ts, {"t3": {"s1": 0.5}}, ceilings={"s1": 0})
    assert blocking["t1"] == 0.0


def test_blocking_validates_input():
    ts = standard_set()
    with pytest.raises(ValueError, match="unknown task"):
        priority_ceiling_blocking(ts, {"nope": {"s1": 0.1}})
    with pytest.raises(ValueError, match="duration > 0"):
        priority_ceiling_blocking(ts, {"t3": {"s1": 0.0}})
    with pytest.raises(ValueError, match="never used"):
        priority_ceiling_blocking(ts, {"t3": {"s1": 0.1}}, ceilings={"s1": 1, "s9": 1})
    with pytest.raises(ValueError, match="no ceiling given"):
        priority_ceiling_blocking(ts, {"t3": {"s1": 0.1, "s2": 0.1}}, ceilings={"s1": 1})


def test_response_time_validates_input():
    task = PeriodicTask("a", 1.0, 0.1)
    with pytest.raises(ValueError, match="blocking_s must be >= 0"):
        response_time(task, [], blocking_s=-1.0)
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        response_time(task, [], max_iterations=0)


def test_response_time_stops_at_the_deadline_on_an_overloaded_set():
    # An overloaded set: hp utilization 1.5 > 1, so the recurrence diverges.
    # Iteration must stop at the first iterate past the deadline rather than
    # running to floating-point overflow.
    task = PeriodicTask("low", 1000.0, 1.0)
    hp = [PeriodicTask("hi", 1.0, 1.5)]
    rt = response_time(task, hp, max_iterations=10_000)
    assert rt.converged is False
    assert rt.stopped_past_deadline is True
    assert rt.meets_deadline is False
    assert rt.response_s > rt.deadline_s
    assert rt.iterations < 100
    assert all(math.isfinite(x) for x in rt.iterates)


def test_response_time_stops_immediately_when_c_plus_b_exceeds_the_deadline():
    # C + B = 0.6 + 0.5 = 1.1 s > D = 1.0 s: unschedulable before any
    # interference is added, so no iteration happens at all.
    rt = response_time(PeriodicTask("a", 1.0, 0.6), [], blocking_s=0.5)
    assert rt.iterations == 0
    assert rt.converged is False
    assert rt.stopped_past_deadline is True
    assert rt.response_s == pytest.approx(1.1, abs=1e-12)


def test_response_time_honours_the_iteration_cap():
    # t3 of the standard set needs 5 iterations; cap it at 2.
    ts = standard_set()
    t3 = next(t for t in ts if t.name == "t3")
    hp = [t for t in ts if t.name != "t3"]
    rt = response_time(t3, hp, max_iterations=2)
    assert rt.iterations == 2
    assert rt.converged is False
    assert rt.stopped_past_deadline is False


def test_response_time_analysis_rejects_unknown_blocking_names():
    with pytest.raises(ValueError, match="names tasks not in the set"):
        response_time_analysis(standard_set(), blocking_s={"ghost": 1.0})


def test_response_time_analysis_requires_priorities():
    ts = TaskSet([PeriodicTask("a", 1.0, 0.1)])
    with pytest.raises(ValueError, match="have no priority"):
        response_time_analysis(ts)
