"""Validation 2: response-time analysis against hand computations.

Four task sets, each hand-computed in full in the printed output so the
arithmetic can be checked line by line without running anything.

Reference
  M. Joseph and P. Pandya, "Finding Response Times in a Real-Time System",
  The Computer Journal 29(5) (1986) -- the worst-case response time of a
  fixed-priority preemptive task set.
  N. C. Audsley, A. Burns, M. Richardson, K. Tindell and A. J. Wellings,
  "Applying new scheduling theory to static priority pre-emptive
  scheduling", Software Engineering Journal 8(5) (1993) -- the iterative
  fixed-point form used here.
  L. Sha, R. Rajkumar and J. P. Lehoczky, "Priority Inheritance Protocols:
  An Approach to Real-Time Synchronization", IEEE Transactions on Computers
  39(9) (1990) -- blocked at most once under the priority ceiling protocol.
  G. C. Buttazzo, Hard Real-Time Computing Systems, 3rd ed., Springer (2011),
  Sec. 4.5-4.6 and 7.5 -- textbook anchor.

Recurrence:  R_i = C_i + B_i + sum_{j in hp(i)} ceil(R_i / T_j) * C_j

Run from products/P036/:  python validation/validate_response_times.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtclock.schedulability import (  # noqa: E402
    priority_ceiling_blocking,
    response_time_analysis,
)
from rtclock.taskset import PeriodicTask, TaskSet  # noqa: E402

TOL = 1e-12
failures: list[str] = []

print("=" * 78)
print("VALIDATION 2 -- response-time analysis vs hand computation")
print("=" * 78)
print()
print("Recurrence (Joseph & Pandya 1986; iterative form Audsley et al. 1993):")
print("    R_i = C_i + B_i + sum_{j in hp(i)} ceil(R_i / T_j) * C_j")
print(f"Tolerance on every response time: {TOL:.1e} s absolute.")
print()


def check(label: str, ts: TaskSet, hand: dict[str, float],
          blocking: dict[str, float] | None = None,
          hand_iterates: dict[str, tuple[float, ...]] | None = None) -> None:
    results = {r.name: r for r in response_time_analysis(ts, blocking_s=blocking)}
    print(f"--- {label}")
    print(f"{'task':>6} {'prio':>5} {'T [s]':>8} {'C [s]':>8} {'B [s]':>8} "
          f"{'D [s]':>8} {'R computed':>12} {'R by hand':>12} {'diff':>10}  verdict")
    for t in ts.by_priority():
        r = results[t.name]
        exp = hand[t.name]
        diff = abs(r.response_s - exp)
        ok = diff <= TOL
        if not ok:
            failures.append(f"{label} / {t.name}: {r.response_s} vs {exp}")
        print(f"{t.name:>6} {t.priority:>5} {t.period_s:>8.3f} {t.wcet_s:>8.3f} "
              f"{r.blocking_s:>8.3f} {r.deadline_s:>8.3f} {r.response_s:>12.6f} "
              f"{exp:>12.6f} {diff:>10.2e}  {'PASS' if ok else 'FAIL'}")
    if hand_iterates:
        for name, seq in hand_iterates.items():
            got = results[name].iterates
            pairs = zip(got, seq, strict=False)
            ok = len(got) == len(seq) and all(abs(a - b) <= TOL for a, b in pairs)
            if not ok:
                failures.append(f"{label} / {name} iterates: {got} vs {seq}")
            print(f"       iterates for {name}: computed {[f'{x:g}' for x in got]}")
            print(f"       iterates for {name}: by hand  {[f'{x:g}' for x in seq]}"
                  f"   {'PASS' if ok else 'FAIL'}")
    print()


# ---------------------------------------------------------------- example 2.1
print("2.1  Standard three-task set, no blocking.")
print("     t1: T=7  C=3   t2: T=12 C=3   t3: T=20 C=5     (D = T for all)")
print("     U = 3/7 + 3/12 + 5/20 = 0.428571428571 + 0.25 + 0.25 = 0.928571428571")
print()
print("     Hand computation:")
print("       t1 (highest, no interference):  R = C = 3            3 <= 7   met")
print("       t2: R0 = 3")
print("           R1 = 3 + ceil(3/7)*3  = 3 + 1*3 = 6")
print("           R2 = 3 + ceil(6/7)*3  = 3 + 1*3 = 6   fixed       6 <= 12  met")
print("       t3: R0 = 5")
print("           R1 = 5 + ceil(5/7)*3  + ceil(5/12)*3  = 5 + 3 + 3 = 11")
print("           R2 = 5 + ceil(11/7)*3 + ceil(11/12)*3 = 5 + 6 + 3 = 14")
print("           R3 = 5 + ceil(14/7)*3 + ceil(14/12)*3 = 5 + 6 + 6 = 17")
print("           R4 = 5 + ceil(17/7)*3 + ceil(17/12)*3 = 5 + 9 + 6 = 20")
print("           R5 = 5 + ceil(20/7)*3 + ceil(20/12)*3 = 5 + 9 + 6 = 20  fixed")
print("                                                    20 <= 20  met exactly")
print()
standard = TaskSet(
    [PeriodicTask("t1", 7.0, 3.0), PeriodicTask("t2", 12.0, 3.0), PeriodicTask("t3", 20.0, 5.0)]
).rate_monotonic()
check(
    "2.1 standard set",
    standard,
    {"t1": 3.0, "t2": 6.0, "t3": 20.0},
    hand_iterates={
        "t1": (3.0, 3.0),
        "t2": (3.0, 6.0, 6.0),
        "t3": (5.0, 11.0, 14.0, 17.0, 20.0, 20.0),
    },
)

# ---------------------------------------------------------------- example 2.2
print("2.2  Harmonic set at U = 1.0 exactly -- schedulable although the")
print("     Liu & Layland bound (0.779763149685 for n=3) rejects it.")
print("     a: T=4  C=1   b: T=8  C=2   c: T=16 C=8")
print()
print("     Hand computation:")
print("       R_a = 1                                              1 <= 4   met")
print("       R_b: R0 = 2; R1 = 2 + ceil(2/4)*1 = 3;")
print("            R2 = 2 + ceil(3/4)*1 = 3   fixed                3 <= 8   met")
print("       R_c: R0 = 8")
print("            R1 = 8 + ceil(8/4)*1  + ceil(8/8)*2  = 8 + 2 + 2 = 12")
print("            R2 = 8 + ceil(12/4)*1 + ceil(12/8)*2 = 8 + 3 + 4 = 15")
print("            R3 = 8 + ceil(15/4)*1 + ceil(15/8)*2 = 8 + 4 + 4 = 16")
print("            R4 = 8 + ceil(16/4)*1 + ceil(16/8)*2 = 8 + 4 + 4 = 16  fixed")
print("                                                   16 <= 16  met exactly")
print()
harmonic = TaskSet(
    [PeriodicTask("a", 4.0, 1.0), PeriodicTask("b", 8.0, 2.0), PeriodicTask("c", 16.0, 8.0)]
).rate_monotonic()
check(
    "2.2 harmonic set at U = 1",
    harmonic,
    {"a": 1.0, "b": 3.0, "c": 16.0},
    hand_iterates={"c": (8.0, 12.0, 15.0, 16.0, 16.0)},
)

# ---------------------------------------------------------------- example 2.3
print("2.3  Priority-ceiling blocking turns a met deadline into a missed one.")
print("     t1: T=7 C=3  t2: T=12 C=3  t3: T=20 C=5  t4: T=50 C=5")
print("     Semaphore s1 used by t3 (0.001 s) and t4 (1.0 s).")
print("     RM priorities: t1=3 > t2=2 > t3=1 > t4=0, so ceiling(s1) = 1.")
print()
print("     Blocking by hand (Sha, Rajkumar & Lehoczky 1990, blocked at most once):")
print("       B_1: no lower-priority user of a semaphore with ceiling >= 3 -> 0")
print("       B_2: ceiling(s1) = 1 < 2                                    -> 0")
print("       B_3: t4 is lower priority and ceiling(s1) = 1 >= 1          -> 1.0 s")
print("       B_4: nothing below it                                       -> 0")
print()
print("     Then for t3, R = 5 + 1 + ceil(R/7)*3 + ceil(R/12)*3:")
print("       R0 = 6")
print("       R1 = 6 + ceil(6/7)*3  + ceil(6/12)*3  = 6 + 3 + 3 = 12")
print("       R2 = 6 + ceil(12/7)*3 + ceil(12/12)*3 = 6 + 6 + 3 = 15")
print("       R3 = 6 + ceil(15/7)*3 + ceil(15/12)*3 = 6 + 9 + 6 = 21")
print("       21 > D = 20: the 1 s blocking term costs the deadline by 1 s.")
print("       Iteration stops at the first iterate past the deadline.")
print()
print("     t4 also misses, because U = 0.928571 + 0.1 = 1.028571 > 1.")
print("     R_4 = 5 + ceil(R/7)*3 + ceil(R/12)*3 + ceil(R/20)*5:")
print("       R0 = 5")
print("       R1 = 5 + ceil(5/7)*3  + ceil(5/12)*3  + ceil(5/20)*5  = 5+3+3+5     = 16")
print("       R2 = 5 + ceil(16/7)*3 + ceil(16/12)*3 + ceil(16/20)*5 = 5+9+6+5     = 25")
print("       R3 = 5 + ceil(25/7)*3 + ceil(25/12)*3 + ceil(25/20)*5 = 5+12+9+10   = 36")
print("       R4 = 5 + ceil(36/7)*3 + ceil(36/12)*3 + ceil(36/20)*5 = 5+18+9+10   = 42")
print("       R5 = 5 + ceil(42/7)*3 + ceil(42/12)*3 + ceil(42/20)*5 = 5+18+12+15  = 50")
print("       R6 = 5 + ceil(50/7)*3 + ceil(50/12)*3 + ceil(50/20)*5 = 5+24+15+15  = 59")
print("       59 > D = 50, so iteration stops there and reports R_4 = 59.")
print("       (Past its own period the recurrence is no longer the exact worst")
print("        case -- see the known-limitation note in rtclock.schedulability --")
print("        but the deadline-miss verdict is sound either way.)")
print()
blocked = TaskSet(
    [
        PeriodicTask("t1", 7.0, 3.0),
        PeriodicTask("t2", 12.0, 3.0),
        PeriodicTask("t3", 20.0, 5.0),
        PeriodicTask("t4", 50.0, 5.0),
    ]
).rate_monotonic()
blocking = priority_ceiling_blocking(blocked, {"t4": {"s1": 1.0}, "t3": {"s1": 0.001}})
hand_blocking = {"t1": 0.0, "t2": 0.0, "t3": 1.0, "t4": 0.0}
ok_b = all(abs(blocking[k] - v) <= TOL for k, v in hand_blocking.items())
if not ok_b:
    failures.append(f"2.3 blocking terms: {blocking} vs {hand_blocking}")
print(f"     computed blocking terms [s]: {blocking}")
print(f"     hand blocking terms     [s]: {hand_blocking}   {'PASS' if ok_b else 'FAIL'}")
print()
results_23 = {r.name: r for r in response_time_analysis(blocked, blocking_s=blocking)}
check(
    "2.3 with priority-ceiling blocking",
    blocked,
    {"t1": 3.0, "t2": 6.0, "t3": 21.0, "t4": 59.0},
    blocking=blocking,
    hand_iterates={
        "t3": (6.0, 12.0, 15.0, 21.0),
        "t4": (5.0, 16.0, 25.0, 36.0, 42.0, 50.0, 59.0),
    },
)
miss_ok = results_23["t3"].meets_deadline is False and results_23["t3"].slack_s < 0.0
if not miss_ok:
    failures.append("2.3: t3 should miss its deadline")
print(f"     t3 slack [s]: {results_23['t3'].slack_s:+.6f}  "
      f"deadline met: {results_23['t3'].meets_deadline}   {'PASS' if miss_ok else 'FAIL'}")
print()

# ---------------------------------------------------------------- example 2.4
print("2.4  Constrained deadlines (D < T) under deadline-monotonic priorities.")
print("     a: T=50 C=5  D=10   b: T=20 C=4  D=20   (DM: a has the shorter")
print("     deadline so a gets the higher priority, which rate-monotonic")
print("     would have got the other way round.)")
print()
print("     Hand computation:")
print("       R_a = 5                                            5 <= 10  met")
print("       R_b: R0 = 4; R1 = 4 + ceil(4/50)*5 = 9;")
print("            R2 = 4 + ceil(9/50)*5 = 9  fixed              9 <= 20  met")
print()
dm = TaskSet(
    [
        PeriodicTask("a", 50.0, 5.0, deadline_s=10.0),
        PeriodicTask("b", 20.0, 4.0, deadline_s=20.0),
    ]
).deadline_monotonic()
check("2.4 constrained deadlines, DM priorities", dm, {"a": 5.0, "b": 9.0},
      hand_iterates={"b": (4.0, 9.0, 9.0)})

print("=" * 78)
if failures:
    print(f"RESULT: {len(failures)} FAILURE(S)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: all checks PASS")
print("=" * 78)
