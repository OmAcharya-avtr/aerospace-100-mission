"""Validation 3: percentiles against exact computation over the stored samples.

Both implemented definitions are checked against two independent references:

  3a. ``nearest_rank`` against an explicit order-statistic index computed in
      this script -- k = max(1, ceil(p/100 * N)), x sorted, take x[k-1]. Over
      every N in a fixed list and ten percentiles each, requiring **bit-exact
      equality**, because the result must be one of the stored samples.
  3b. ``linear`` against ``numpy.percentile(..., method="linear")``, the
      Hyndman & Fan (1996) type 7 definition. Bit-exact or within 1e-16.
  3c. A 10-sample hand-computed table printed in full for both definitions.
  3d. Invariants over 20 000 random samples: monotonicity in p, membership of
      the sample set for nearest-rank, agreement at p = 0 and p = 100,
      invariance to sample order, positive homogeneity.
  3e. The two definitions are shown to DISAGREE where they must, so that no
      reader can treat them as interchangeable.

Reference
  R. J. Hyndman and Y. Fan, "Sample Quantiles in Statistical Packages",
  The American Statistician 50(4), 361-365 (1996), Definition 7.

Run from products/P036/:  python validation/validate_percentiles.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtclock.histogram import LatencyHistogram, percentile  # noqa: E402

failures: list[str] = []
PS = (0.0, 0.1, 1.0, 25.0, 33.333, 50.0, 66.667, 90.0, 99.0, 99.9, 100.0)
SIZES = (1, 2, 3, 5, 10, 13, 64, 101, 1000, 20_000)

print("=" * 78)
print("VALIDATION 3 -- exact percentiles, both definitions named")
print("=" * 78)
print()
print("Definitions implemented (rtclock.histogram):")
print("  nearest_rank : k = max(1, ceil(p/100 * N)); P = x_sorted[k-1]")
print("                 the result is always one of the stored samples")
print("  linear       : h = (N-1) * p/100; P = x[floor h] + frac * (x[ceil h] - x[floor h])")
print("                 Hyndman & Fan (1996) type 7, = numpy method='linear'")
print()

print("3a. nearest_rank vs an explicit order statistic (bit-exact required)")
rng = np.random.default_rng(20261004)
worst_a = 0.0
n_checks_a = 0
for n in SIZES:
    xs = (rng.random(n) * 1e-3).tolist()
    srt = sorted(xs)
    for p in PS:
        k = max(1, math.ceil(p / 100.0 * n))
        ref = srt[k - 1]
        got = percentile(xs, p, "nearest_rank")
        n_checks_a += 1
        if got != ref:
            worst_a = max(worst_a, abs(got - ref))
            failures.append(f"3a N={n} p={p}: {got!r} != {ref!r}")
print(f"    sample sizes            : {list(SIZES)}")
print(f"    percentiles per size    : {list(PS)}")
print(f"    comparisons             : {n_checks_a}")
print(f"    worst absolute deviation: {worst_a:.1e}  (bit-exact equality required)")
print(f"    verdict                 : {'PASS' if worst_a == 0.0 else 'FAIL'}")
print()

print("3b. linear vs numpy.percentile(method='linear')")
worst_b = 0.0
n_checks_b = 0
for n in SIZES:
    xs = rng.random(n) * 1e-3
    for p in PS:
        got = percentile(xs.tolist(), p, "linear")
        ref = float(np.percentile(xs, p, method="linear"))
        worst_b = max(worst_b, abs(got - ref))
        n_checks_b += 1
        if abs(got - ref) > 1e-16:
            failures.append(f"3b N={n} p={p}: {got!r} vs numpy {ref!r}")
print(f"    comparisons             : {n_checks_b}")
print(f"    worst absolute deviation: {worst_b:.3e} s  (tolerance 1e-16 s)")
print(f"    verdict                 : {'PASS' if worst_b <= 1e-16 else 'FAIL'}")
print()

print("3c. Hand-computed table, samples 1..10 (unsorted input, N = 10)")
ten = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 10.0, 6.0]
# nearest_rank: k = max(1, ceil(p/100*10)); linear: h = 9*p/100
hand = {
    0.0: (1.0, 1.0),
    1.0: (1.0, 1.09),
    10.0: (1.0, 1.9),
    11.0: (2.0, 1.99),
    25.0: (3.0, 3.25),
    50.0: (5.0, 5.5),
    55.0: (6.0, 5.95),
    90.0: (9.0, 9.1),
    95.0: (10.0, 9.55),
    99.0: (10.0, 9.91),
    100.0: (10.0, 10.0),
}
print(f"{'p':>7}  {'k=ceil(p*N/100)':>16}  {'nearest_rank':>13}  {'hand':>7}  "
      f"{'h=(N-1)p/100':>13}  {'linear':>9}  {'hand':>7}  verdict")
for p, (nr_hand, lin_hand) in sorted(hand.items()):
    k = max(1, math.ceil(p / 100.0 * 10))
    h = 9 * p / 100.0
    nr = percentile(ten, p, "nearest_rank")
    lin = percentile(ten, p, "linear")
    ok = nr == nr_hand and abs(lin - lin_hand) <= 1e-12
    if not ok:
        failures.append(f"3c p={p}: nr {nr} vs {nr_hand}, lin {lin} vs {lin_hand}")
    print(f"{p:>7.1f}  {k:>16}  {nr:>13.4f}  {nr_hand:>7.2f}  {h:>13.4f}  "
          f"{lin:>9.4f}  {lin_hand:>7.2f}  {'PASS' if ok else 'FAIL'}")
print()

print("3d. Invariants over a 20 000-sample right-skewed latency trace")
trace = (1e-3 + rng.gamma(shape=2.0, scale=5e-5, size=20_000)).tolist()
hist = LatencyHistogram(label="synthetic latency")
hist.extend(trace)
mono_ok = True
prev_nr = prev_lin = -math.inf
for p in [i * 0.5 for i in range(201)]:
    nr = hist.percentile(p, "nearest_rank")
    lin = hist.percentile(p, "linear")
    if nr < prev_nr or lin < prev_lin:
        mono_ok = False
    prev_nr, prev_lin = nr, lin
member_ok = all(hist.percentile(p, "nearest_rank") in trace for p in PS)
ends_ok = (
    hist.percentile(0.0, "nearest_rank") == min(trace)
    and hist.percentile(100.0, "nearest_rank") == max(trace)
    and hist.percentile(0.0, "linear") == min(trace)
    and hist.percentile(100.0, "linear") == max(trace)
)
order_ok = all(
    percentile(trace, p, m) == percentile(list(reversed(trace)), p, m)
    for p in PS
    for m in ("nearest_rank", "linear")
)
scale = 1e3
homog_ok = all(
    math.isclose(
        percentile([x * scale for x in trace], p, m),
        percentile(trace, p, m) * scale,
        rel_tol=1e-12,
    )
    for p in PS
    for m in ("nearest_rank", "linear")
)
for label, ok in (
    ("monotone in p (401 percentiles, both methods)", mono_ok),
    ("nearest_rank result is a stored sample", member_ok),
    ("both methods agree at p = 0 and p = 100", ends_ok),
    ("invariant to sample order", order_ok),
    ("positively homogeneous under a 1e3 scaling", homog_ok),
):
    if not ok:
        failures.append(f"3d {label}")
    print(f"    {label:<48} {'PASS' if ok else 'FAIL'}")
print()
print("    summary of the same trace, both definitions (units s):")
s_nr = hist.summary(percentiles=(50.0, 90.0, 99.0, 99.9, 100.0), method="nearest_rank")
s_lin = hist.summary(percentiles=(50.0, 90.0, 99.0, 99.9, 100.0), method="linear")
print(f"      count = {int(s_nr['count'])}  min = {s_nr['min_s']:.9e}  "
      f"mean = {s_nr['mean_s']:.9e}  max = {s_nr['max_s']:.9e}")
for p in ("p50", "p90", "p99", "p99_9"):
    print(f"      {p:>6}  nearest_rank {s_nr[f'{p}_nearest_rank_s']:.9e}   "
          f"linear {s_lin[f'{p}_linear_s']:.9e}")
print()

print("3e. The definitions are NOT interchangeable")
diff_50 = percentile(ten, 50.0, "linear") - percentile(ten, 50.0, "nearest_rank")
diff_995 = hist.percentile(99.5, "linear") - hist.percentile(99.5, "nearest_rank")
ok_e = abs(diff_50) > 0.4
if not ok_e:
    failures.append("3e: the two definitions failed to differ on the 10-sample case")
print(f"    p50 on 1..10          : nearest_rank 5.0, linear 5.5, difference "
      f"{diff_50:+.4f}")
print(f"    p99.5 on the 20k trace: difference {diff_995:+.6e} s")
print(f"    verdict               : {'PASS' if ok_e else 'FAIL'}")
print()

print("3f. The nearest-rank float edge, demonstrated rather than hidden")
print("    p arrives as a binary64 float. The stored value of 99.9 is")
print("    99.900000000000005684..., so p/100 * N can land just above an integer")
print("    and push the rank up by one. No tolerance is applied to mask this.")
print()
edge = [(99.9, 20000), (99.9, 10000), (99.9, 1000), (99.95, 20000), (99.0, 100), (50.0, 10)]
print(f"{'p':>8}  {'N':>7}  {'p/100*N (repr)':>24}  {'ceil':>7}  {'naive':>7}  shifted?")
ok_f = True
for pv, nv in edge:
    prod = pv / 100.0 * nv
    k = math.ceil(prod)
    naive = round(pv * nv / 100.0)
    shifted = k != naive
    print(f"{pv:>8.2f}  {nv:>7}  {prod!r:>24}  {k:>7}  {naive:>7}  "
          f"{'YES (+1 rank)' if shifted else 'no'}")
ramp = [float(i) for i in range(20_000)]
got_edge = percentile(ramp, 99.9, "nearest_rank")
expected_edge = ramp[19_980]
ok_f = got_edge == expected_edge and (got_edge - ramp[19_979]) == 1.0
if not ok_f:
    failures.append("3f: the documented nearest-rank edge behaviour did not reproduce")
print()
print(f"    on the ramp x[i] = i, N = 20000: p99.9 nearest_rank = {got_edge:.1f}")
print(f"    the 19980th order statistic is {ramp[19_979]:.1f}, the 19981st is "
      f"{ramp[19_980]:.1f}")
print(f"    error bounded by exactly one order statistic: "
      f"{got_edge - ramp[19_979]:.1f}")
print(f"    verdict               : {'PASS' if ok_f else 'FAIL'}")
print()

print("=" * 78)
if failures:
    print(f"RESULT: {len(failures)} FAILURE(S)")
    for f in failures[:20]:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: all checks PASS")
print("=" * 78)
