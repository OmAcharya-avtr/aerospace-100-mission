"""Scheduler tests, including hand-solved instances with known optima.

Value convention throughout: availability=None, so
E[data] = data_rate_gbps * duration_s.  With rate = 1 Gbit/s the value of a
pass in Gbit equals its duration in seconds, which makes the hand arithmetic
below trivial to check.
"""

from datetime import datetime, timedelta, timezone
from itertools import combinations

import pytest

from passplanner import Station, schedule_greedy, schedule_ilp
from passplanner.passes import Pass
from passplanner.scheduler import passes_conflict

T0 = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
S1 = Station(name="S1", lat_deg=0.0, lon_deg=0.0, data_rate_gbps=1.0)
S2 = Station(name="S2", lat_deg=10.0, lon_deg=20.0, data_rate_gbps=1.0)


def mk(sat: str, station: Station, start_s: float, end_s: float) -> Pass:
    """Synthetic pass from start_s to end_s (seconds after T0)."""
    return Pass(satellite=sat, station=station,
                t_rise=T0 + timedelta(seconds=start_s),
                t_set=T0 + timedelta(seconds=end_s),
                t_culminate=T0 + timedelta(seconds=0.5 * (start_s + end_s)),
                max_elevation_deg=45.0)


def brute_force_optimum(passes, setup_time_s=0.0):
    """Exhaustive max-weight independent set (exponential; small n only)."""
    best = 0.0
    best_set: tuple = ()
    n = len(passes)
    for k in range(n + 1):
        for combo in combinations(range(n), k):
            if any(passes_conflict(passes[i], passes[j], setup_time_s)
                   for i, j in combinations(combo, 2)):
                continue
            val = sum(passes[i].duration_s * passes[i].station.data_rate_gbps for i in combo)
            if val > best:
                best, best_set = val, combo
    return best, best_set


# --- Hand-solved instance A: the classic greedy trap ------------------------
# One station S1 (1 Gbit/s), one satellite SAT-A, three candidate passes:
#   A: 0 s .. 1000 s   value 1000 Gbit
#   B: 0 s ..  600 s   value  600 Gbit
#   C: 700 s .. 1400 s value  700 Gbit
# Conflicts: A-B (overlap 0-600), A-C (overlap 700-1000). B and C are
# disjoint (B ends at 600 < 700 when C rises).
# HAND SOLUTION: feasible sets are {}, {A}=1000, {B}=600, {C}=700, {B,C}=1300.
#   -> optimum = {B, C} = 1300 Gbit.
# Greedy takes the largest value first (A = 1000) and is then blocked by both
#   -> greedy = 1000 Gbit, gap = (1300 - 1000)/1300 = 23.0769 %.
INSTANCE_A = [mk("SAT-A", S1, 0, 1000), mk("SAT-A", S1, 0, 600), mk("SAT-A", S1, 700, 1400)]


def test_instance_a_ilp_finds_hand_optimum():
    res = schedule_ilp(INSTANCE_A)
    assert res.total_value == pytest.approx(1300.0)
    assert {(p.t_rise, p.t_set) for p in res.selected} == {
        (INSTANCE_A[1].t_rise, INSTANCE_A[1].t_set),
        (INSTANCE_A[2].t_rise, INSTANCE_A[2].t_set)}


def test_instance_a_greedy_matches_hand_value_and_gap():
    greedy = schedule_greedy(INSTANCE_A)
    assert greedy.total_value == pytest.approx(1000.0)
    assert len(greedy.selected) == 1
    gap = (1300.0 - greedy.total_value) / 1300.0
    assert gap == pytest.approx(0.230769, abs=1e-6)


# --- Hand-solved instance B: per-satellite constraint across stations -------
# One satellite SAT-B with simultaneous passes over two different stations.
# The satellite has a single downlink terminal, so only one can be used.
#   D: S1, 0 s .. 500 s  value 500
#   E: S2, 100 s .. 900 s value 800
# HAND SOLUTION: D and E overlap (100-500) and share the satellite -> conflict.
#   optimum = {E} = 800 Gbit; greedy also picks E first -> greedy = optimum.
INSTANCE_B = [mk("SAT-B", S1, 0, 500), mk("SAT-B", S2, 100, 900)]


def test_instance_b_satellite_constraint():
    assert passes_conflict(INSTANCE_B[0], INSTANCE_B[1])
    ilp = schedule_ilp(INSTANCE_B)
    greedy = schedule_greedy(INSTANCE_B)
    assert ilp.total_value == pytest.approx(800.0)
    assert greedy.total_value == pytest.approx(800.0)
    assert len(ilp.selected) == 1


# --- Hand-solved instance C: per-station constraint across satellites -------
# One station S1, two satellites with overlapping passes; a third pass at the
# other station S2 is unconstrained.
#   F: SAT-A @ S1, 0 s .. 400 s   value 400
#   G: SAT-B @ S1, 200 s .. 900 s value 700
#   H: SAT-A @ S2, 0 s .. 300 s   value 300  (different station AND different
#      satellite from G; shares SAT-A with F and overlaps it -> conflict)
# HAND SOLUTION: conflicts are F-G (same station, overlap 200-400) and F-H
#   (same satellite, overlap 0-300). G-H share neither -> compatible.
#   Feasible maxima: {G, H} = 700 + 300 = 1000; {F} = 400.
#   -> optimum = 1000 Gbit. Greedy: G (700) first, then H (300, compatible),
#   then F blocked -> greedy = 1000 = optimum.
INSTANCE_C = [mk("SAT-A", S1, 0, 400), mk("SAT-B", S1, 200, 900), mk("SAT-A", S2, 0, 300)]


def test_instance_c_mixed_constraints():
    assert passes_conflict(INSTANCE_C[0], INSTANCE_C[1])     # same station
    assert passes_conflict(INSTANCE_C[0], INSTANCE_C[2])     # same satellite
    assert not passes_conflict(INSTANCE_C[1], INSTANCE_C[2])  # neither shared
    ilp = schedule_ilp(INSTANCE_C)
    greedy = schedule_greedy(INSTANCE_C)
    assert ilp.total_value == pytest.approx(1000.0)
    assert greedy.total_value == pytest.approx(1000.0)


# --- Hand-solved instance D: setup time creates conflicts -------------------
# Two passes at S1, 0..200 s and 260..500 s: a 60 s gap.
# HAND SOLUTION with setup_time_s = 0: disjoint -> both, 200 + 240 = 440 Gbit.
# With setup_time_s = 120 s the padded intervals overlap -> pick the larger,
#   240 Gbit.
INSTANCE_D = [mk("SAT-A", S1, 0, 200), mk("SAT-A", S1, 260, 500)]


def test_instance_d_setup_time_changes_optimum():
    assert schedule_ilp(INSTANCE_D, setup_time_s=0.0).total_value == pytest.approx(440.0)
    assert schedule_ilp(INSTANCE_D, setup_time_s=120.0).total_value == pytest.approx(240.0)
    assert schedule_greedy(INSTANCE_D, setup_time_s=120.0).total_value == pytest.approx(240.0)


@pytest.mark.parametrize("instance,expected", [
    (INSTANCE_A, 1300.0), (INSTANCE_B, 800.0), (INSTANCE_C, 1000.0), (INSTANCE_D, 440.0)])
def test_ilp_equals_brute_force_optimum(instance, expected):
    bf, _ = brute_force_optimum(instance)
    assert bf == pytest.approx(expected)
    assert schedule_ilp(instance).total_value == pytest.approx(bf)


def test_ilp_matches_brute_force_on_random_instances():
    import random
    rng = random.Random(20260306)
    for _ in range(12):
        n = rng.randint(4, 9)
        passes = []
        for _i in range(n):
            start = rng.uniform(0.0, 3000.0)
            passes.append(mk(rng.choice(["SAT-A", "SAT-B"]), rng.choice([S1, S2]),
                             start, start + rng.uniform(100.0, 900.0)))
        bf, _ = brute_force_optimum(passes)
        ilp = schedule_ilp(passes)
        greedy = schedule_greedy(passes)
        assert ilp.total_value == pytest.approx(bf, rel=1e-9)
        assert greedy.total_value <= ilp.total_value + 1e-9


def test_greedy_and_ilp_agree_when_no_conflicts():
    passes = [mk("SAT-A", S1, 0, 100), mk("SAT-A", S1, 200, 300), mk("SAT-A", S1, 400, 500)]
    g, i = schedule_greedy(passes), schedule_ilp(passes)
    assert g.total_value == pytest.approx(i.total_value) == pytest.approx(300.0)
    assert len(g.selected) == len(i.selected) == 3


def test_selected_schedule_is_feasible():
    for instance in (INSTANCE_A, INSTANCE_B, INSTANCE_C, INSTANCE_D):
        for res in (schedule_greedy(instance), schedule_ilp(instance)):
            for a, b in combinations(res.selected, 2):
                assert not passes_conflict(a, b)


def test_results_are_sorted_by_rise_time():
    res = schedule_ilp(INSTANCE_A)
    rises = [p.t_rise for p in res.selected]
    assert rises == sorted(rises)


def test_empty_input():
    for res in (schedule_greedy([]), schedule_ilp([])):
        assert res.selected == ()
        assert res.total_value == 0.0
        assert res.n_candidates == 0


def test_schedulers_are_deterministic():
    a = schedule_greedy(INSTANCE_A).total_value
    b = schedule_greedy(INSTANCE_A).total_value
    c = schedule_ilp(INSTANCE_A).total_value
    d = schedule_ilp(INSTANCE_A).total_value
    assert a == b and c == d


def test_negative_setup_time_rejected():
    with pytest.raises(ValueError, match="setup_time_s"):
        schedule_greedy(INSTANCE_A, setup_time_s=-1.0)
