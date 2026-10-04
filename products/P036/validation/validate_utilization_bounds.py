"""Validation 1: utilization bounds against textbook values.

Checks
  1a. RM least upper bound n(2^(1/n)-1) for n = 1..10 against values computed
      independently from mpmath-free high-precision arithmetic (Python's
      ``decimal`` at 40 digits), then rounded to 12 decimals.
  1b. The n = 2 value against the textbook 0.8284 and the n -> inf limit
      against ln 2 = 0.6931471805599453.
  1c. The asymptotic expansion n(2^(1/n)-1) = ln2 + (ln2)^2/(2n) + O(1/n^2).
  1d. EDF: U <= 1 is exact for implicit deadlines (Liu & Layland 1973
      Theorem 7) -- demonstrated on a set at U = 1 exactly that RM also
      happens to schedule, and on a set at U = 1.2 that neither does.
  1e. The Liu & Layland bound is sufficient but NOT necessary: a harmonic
      set at U = 1.0 is schedulable under RM although it is far above the
      n = 3 bound.
  1f. The hyperbolic bound dominates the Liu & Layland bound over a 200 000
      point random sweep of 2-6 task sets.

Reference
  C. L. Liu and J. W. Layland, "Scheduling Algorithms for Multiprogramming in
  a Hard-Real-Time Environment", Journal of the ACM 20(1), 46-61 (1973),
  Theorems 5 and 7.
  G. C. Buttazzo, Hard Real-Time Computing Systems, 3rd ed., Springer (2011),
  Sec. 4.3 (Liu & Layland and hyperbolic bounds).

Run from products/P036/:  python validation/validate_utilization_bounds.py
"""

from __future__ import annotations

import math
import sys
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtclock.schedulability import (  # noqa: E402
    edf_test,
    hyperbolic_bound_test,
    response_time_analysis,
    rm_utilization_bound,
    rm_utilization_test,
)
from rtclock.taskset import PeriodicTask, TaskSet  # noqa: E402

getcontext().prec = 40
TOL = 5e-13
failures: list[str] = []


def independent_bound(n: int) -> float:
    """n(2^(1/n)-1) computed at 40 decimal digits with the decimal module."""
    two = Decimal(2)
    return float(Decimal(n) * (two ** (Decimal(1) / Decimal(n)) - Decimal(1)))


print("=" * 78)
print("VALIDATION 1 -- utilization bounds (Liu & Layland 1973)")
print("=" * 78)
print()
print("1a. RM least upper bound U_lub(n) = n(2^(1/n) - 1), n = 1..10")
print("    reference: 40-digit decimal arithmetic; tolerance 5e-13 absolute")
print()
print(f"{'n':>3}  {'rtclock':>20}  {'decimal(40 digits)':>20}  {'abs diff':>12}  verdict")
for n in range(1, 11):
    got = rm_utilization_bound(n)
    ref = independent_bound(n)
    diff = abs(got - ref)
    ok = diff <= TOL
    if not ok:
        failures.append(f"1a n={n}: |{got} - {ref}| = {diff} > {TOL}")
    print(f"{n:>3}  {got:>20.15f}  {ref:>20.15f}  {diff:>12.3e}  {'PASS' if ok else 'FAIL'}")
print()

print("1b. Named values")
ln2 = math.log(2.0)
checks_1b = [
    ("U_lub(1)", rm_utilization_bound(1), 1.0, 0.0),
    ("U_lub(2) vs textbook 0.8284", rm_utilization_bound(2), 0.8284, 5e-5),
    ("U_lub(2) full precision", rm_utilization_bound(2), 0.8284271247461903, 1e-15),
    ("U_lub(3)", rm_utilization_bound(3), 0.7797631496846196, 1e-15),
    ("U_lub(10)", rm_utilization_bound(10), 0.7177346253629313, 1e-15),
    ("U_lub(1e6) -> ln 2", rm_utilization_bound(10**6), ln2, 2.5e-7),
]
print(f"{'check':>34}  {'computed':>20}  {'expected':>20}  {'tol':>9}  verdict")
for label, got, exp, tol in checks_1b:
    ok = abs(got - exp) <= tol
    if not ok:
        failures.append(f"1b {label}: {got} vs {exp} tol {tol}")
    print(f"{label:>34}  {got:>20.16f}  {exp:>20.16f}  {tol:>9.1e}  {'PASS' if ok else 'FAIL'}")
print()

print("1c. Asymptotic expansion: U_lub(n) - ln2 ~ (ln2)^2/(2n)")
print(f"    (ln 2)^2 / 2 = {ln2**2 / 2:.16f}")
print(f"{'n':>8}  {'U_lub(n)-ln2':>18}  {'(ln2)^2/(2n)':>18}  {'ratio':>12}  verdict")
for n in (100, 1000, 10_000, 100_000):
    gap = rm_utilization_bound(n) - ln2
    lead = ln2**2 / (2 * n)
    ratio = gap / lead
    ok = abs(ratio - 1.0) < 0.01
    if not ok:
        failures.append(f"1c n={n}: ratio {ratio} not within 1% of 1")
    print(f"{n:>8}  {gap:>18.12e}  {lead:>18.12e}  {ratio:>12.9f}  {'PASS' if ok else 'FAIL'}")
print()

print("1d. EDF utilization test (exact for implicit deadlines)")
set_u1 = TaskSet(
    [PeriodicTask("a", 4.0, 1.0), PeriodicTask("b", 8.0, 2.0), PeriodicTask("c", 16.0, 8.0)]
).rate_monotonic()
set_u12 = TaskSet(
    [PeriodicTask("a", 10.0, 4.0), PeriodicTask("b", 10.0, 4.0), PeriodicTask("c", 10.0, 4.0)]
).rate_monotonic()
for label, ts, expected_u, expected_sched in (
    ("harmonic set, U = 1.0", set_u1, 1.0, True),
    ("overloaded set, U = 1.2", set_u12, 1.2, False),
):
    r = edf_test(ts)
    ok = (
        abs(r.utilization - expected_u) < 1e-12
        and r.schedulable is expected_sched
        and r.strength == "exact"
    )
    if not ok:
        failures.append(f"1d {label}: U={r.utilization} sched={r.schedulable}")
    print(f"    {label:<26} U = {r.utilization:.12f}  EDF {r.strength}: "
          f"{'feasible' if r.schedulable else 'infeasible'}  {'PASS' if ok else 'FAIL'}")
print()

print("1e. The Liu & Layland bound is SUFFICIENT, not necessary")
ll = rm_utilization_test(set_u1)
hyp = hyperbolic_bound_test(set_u1)
rts = response_time_analysis(set_u1)
all_met = all(rt.meets_deadline for rt in rts)
print(f"    harmonic set a(T=4,C=1) b(T=8,C=2) c(T=16,C=8), U = {set_u1.total_utilization:.12f}")
print(f"    Liu & Layland bound for n=3            : {ll.bound:.12f}")
print(f"    Liu & Layland verdict                  : "
      f"{'schedulable' if ll.schedulable else 'INCONCLUSIVE'}")
print(f"    hyperbolic verdict                     : "
      f"{'schedulable' if hyp.schedulable else 'INCONCLUSIVE'}")
print(f"    exact RTA response times [s]           : "
      f"{{{', '.join(f'{rt.name}: {rt.response_s:g}' for rt in rts)}}}")
print(f"    exact RTA verdict                      : "
      f"{'all deadlines met' if all_met else 'deadline missed'}")
ok_1e = (not ll.schedulable) and all_met
if not ok_1e:
    failures.append("1e: expected an inconclusive bound and a schedulable RTA result")
print(f"    a set rejected by the bound yet schedulable: {'PASS' if ok_1e else 'FAIL'}")
print()

print("1f. Hyperbolic bound dominates the Liu & Layland bound (random sweep)")
rng = np.random.default_rng(20261004)
n_sets = 200_000
accept_ll = accept_hyp = both = ll_only = 0
for _ in range(n_sets):
    k = int(rng.integers(2, 7))
    us = rng.random(k) * (1.1 / k)
    prod = float(np.prod(us + 1.0))
    total = float(us.sum())
    a_ll = total <= rm_utilization_bound(k)
    a_hyp = prod <= 2.0
    accept_ll += a_ll
    accept_hyp += a_hyp
    both += a_ll and a_hyp
    ll_only += a_ll and not a_hyp
ok_1f = ll_only == 0 and accept_hyp >= accept_ll
if not ok_1f:
    failures.append(f"1f: {ll_only} sets accepted by L&L but rejected by hyperbolic")
print(f"    task sets drawn                        : {n_sets}")
print(f"    accepted by Liu & Layland              : {accept_ll}")
print(f"    accepted by hyperbolic                 : {accept_hyp}")
print(f"    accepted by both                       : {both}")
print(f"    accepted by L&L but not hyperbolic     : {ll_only}  (must be 0)")
print(f"    verdict                                : {'PASS' if ok_1f else 'FAIL'}")
print()

print("=" * 78)
if failures:
    print(f"RESULT: {len(failures)} FAILURE(S)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: all checks PASS")
print("=" * 78)
