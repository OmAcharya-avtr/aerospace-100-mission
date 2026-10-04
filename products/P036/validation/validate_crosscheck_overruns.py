"""Validation 6: overrun accounting, and the P031/P036 independent cross-check.

Batch 04 requires P036 RtClock and P031 HilForge to agree on overrun counts
for the same injected timing trace, from independent implementations.

Protocol
  If ``products/P031/validation/crosscheck_overruns.json`` exists, this script
  reads its ``trace`` and ``period_s``, runs rtclock's own overrun accounting
  over them, and writes the independent result to
  ``products/P036/validation/crosscheck_overruns.json`` in the same shape. It
  never edits P031's file.

  If that file does not exist, this script defines the trace itself, in the
  same shape and with a fixed seed, and writes it out. The coordinator then
  diffs the two files.

Convention, stated so that a disagreement can only be a real disagreement
  An iteration overruns when ``latency_s > period_s``, **strictly**. Equality
  is not an overrun. No tolerance is applied: the comparison is on the raw
  binary64 values, so the count is a pure function of the stored trace and
  reproduces bit-exactly. Indices are zero-based.

Checks
  6a. Hand-computed overrun accounting on a 10-sample trace.
  6b. The strict-inequality boundary: a sample exactly equal to the period is
      not an overrun; the next representable float above it is.
  6c. An independent count computed by a separate expression in this script
      (a NumPy boolean sum) must agree exactly with rtclock's count.
  6d. The cross-check trace, with its count, indices, longest consecutive run
      and worst overshoot, written to JSON.
  6e. If P031 also publishes a *cascade* overrun count -- late completion
      after earlier overruns have pushed the start -- reproduce it with
      FixedRateLoop on a SimulatedTimebase, which implements that recurrence
      directly, and report agreement.

Run from products/P036/:  python validation/validate_crosscheck_overruns.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[0] / "src"))

from rtclock.histogram import overrun_report  # noqa: E402
from rtclock.loop import FixedRateLoop  # noqa: E402
from rtclock.timebase import SimulatedTimebase  # noqa: E402

failures: list[str] = []
P031_FILE = ROOT / "P031" / "validation" / "crosscheck_overruns.json"
OUT_FILE = HERE / "crosscheck_overruns.json"

print("=" * 78)
print("VALIDATION 6 -- overrun accounting and the P031/P036 cross-check")
print("=" * 78)
print()
print("Convention: an iteration overruns when latency_s > period_s, STRICTLY.")
print("            Equality is not an overrun. No tolerance. Indices are")
print("            zero-based. The count is a pure function of the trace.")
print()

print("6a. Hand-computed accounting on a 10-sample trace, period 2.5 ms")
# trace in ms: 1.0, 2.0, 2.5, 2.5000001, 3.0, 3.1, 1.0, 9.0, 2.4999999, 2.5
# period 2.5 ms. Strictly greater than 2.5 ms: indices 3 (2.5000001),
# 4 (3.0), 5 (3.1), 7 (9.0)  ->  count 4
# consecutive runs among {3,4,5,7}: 3-4-5 is a run of 3, 7 alone is 1 -> 3
# worst overshoot: 9.0 - 2.5 = 6.5 ms
hand_trace_ms = [1.0, 2.0, 2.5, 2.5000001, 3.0, 3.1, 1.0, 9.0, 2.4999999, 2.5]
hand_trace = [x * 1e-3 for x in hand_trace_ms]
rep = overrun_report(hand_trace, 2.5e-3)
hand = {"count": 4, "indices": (3, 4, 5, 7), "run": 3, "overshoot_ms": 6.5}
ok_6a = (
    rep.count == hand["count"]
    and rep.indices == hand["indices"]
    and rep.longest_consecutive_run == hand["run"]
    and abs(rep.worst_overshoot_s * 1e3 - hand["overshoot_ms"]) < 1e-9
)
if not ok_6a:
    failures.append(f"6a: {rep} vs {hand}")
print(f"    trace [ms]                  : {hand_trace_ms}")
print("    period [ms]                 : 2.5")
print(f"    count    computed / by hand : {rep.count} / {hand['count']}")
print(f"    indices  computed / by hand : {list(rep.indices)} / {list(hand['indices'])}")
print(f"    run      computed / by hand : {rep.longest_consecutive_run} / {hand['run']}")
print(f"    overshoot computed / by hand: {rep.worst_overshoot_s * 1e3:.7f} / "
      f"{hand['overshoot_ms']} ms")
print(f"    fraction                    : {rep.fraction:.4f}")
print(f"    verdict                     : {'PASS' if ok_6a else 'FAIL'}")
print()

print("6b. The strict-inequality boundary at one ulp")
period = 2.5e-3
equal = overrun_report([period], period)
just_above = overrun_report([math.nextafter(period, math.inf)], period)
just_below = overrun_report([math.nextafter(period, 0.0)], period)
ok_6b = equal.count == 0 and just_above.count == 1 and just_below.count == 0
if not ok_6b:
    failures.append(f"6b: equal={equal.count} above={just_above.count} below={just_below.count}")
print(f"    latency exactly   2.500000000000000e-03 s -> overruns {equal.count}  (expect 0)")
print(f"    latency nextafter {math.nextafter(period, math.inf):.15e} s -> overruns "
      f"{just_above.count}  (expect 1)")
print(f"    latency nextafter {math.nextafter(period, 0.0):.15e} s -> overruns "
      f"{just_below.count}  (expect 0)")
print(f"    one ulp at the period       : {math.ulp(period):.6e} s")
print(f"    verdict                     : {'PASS' if ok_6b else 'FAIL'}")
print()

# --------------------------------------------------------------- the trace
if P031_FILE.exists():
    source = "P031"
    payload = json.loads(P031_FILE.read_text())
    trace = [float(x) for x in payload["trace"]]
    period_s = float(payload["period_s"])
    print(f"6c/6d. P031's cross-check file was found at {P031_FILE}")
    print("       Its trace and period are used; rtclock's accounting is run")
    print("       over them independently and written to this product's own file.")
    p031_count = payload.get("overrun_count")
    p031_indices = payload.get("overrun_indices")
else:
    source = "P036"
    print(f"6c/6d. P031's cross-check file is NOT present at {P031_FILE}")
    print("       This product therefore DEFINES the cross-check trace, in the")
    print("       agreed shape and with a fixed seed, and writes it out. The")
    print("       cross-check itself is PENDING the coordinator's diff.")
    rng = np.random.default_rng(20261004)
    period_s = 2.5e-3
    n = 2000
    # A plausible control-loop latency trace: a lognormal body of ~1.2 ms with
    # a 3% heavy-tail contamination that pushes past the 2.5 ms period. Fixed
    # seed, so the trace regenerates bit-exactly.
    base = rng.lognormal(mean=math.log(1.2e-3), sigma=0.25, size=n)
    spikes = rng.random(n) < 0.03
    base[spikes] *= 1.0 + 2.0 * rng.random(int(spikes.sum()))
    trace = [float(x) for x in base]
    p031_count = None
    p031_indices = None
print()

rep = overrun_report(trace, period_s)
arr = np.asarray(trace, dtype=np.float64)
independent_count = int(np.count_nonzero(arr > period_s))
independent_indices = [int(i) for i in np.flatnonzero(arr > period_s)]
ok_6c = independent_count == rep.count and independent_indices == list(rep.indices)
if not ok_6c:
    failures.append(f"6c: rtclock {rep.count} vs numpy {independent_count}")
print("6c. rtclock's accounting vs a separate expression in this script")
print(f"    trace length                : {len(trace)}")
print(f"    period [s]                  : {period_s:.12e}")
print(f"    rtclock overrun count       : {rep.count}")
print(f"    numpy  (arr > period).sum() : {independent_count}")
print(f"    indices identical           : {independent_indices == list(rep.indices)}")
print(f"    verdict                     : {'PASS' if ok_6c else 'FAIL'}")
print()

print("6d. Cross-check result")
print(f"    trace defined by            : {source}")
print(f"    overrun count               : {rep.count}")
print(f"    overrun fraction            : {rep.fraction:.6f}")
print(f"    longest consecutive run     : {rep.longest_consecutive_run}")
print(f"    worst overshoot [s]         : {rep.worst_overshoot_s:.12e}")
print(f"    first 20 overrun indices    : {list(rep.indices[:20])}")
if p031_count is not None:
    agree = p031_count == rep.count and list(p031_indices or []) == list(rep.indices)
    if not agree:
        failures.append(
            f"6d DISAGREEMENT with P031: P031 count {p031_count}, rtclock {rep.count}"
        )
    print(f"    P031 reported count         : {p031_count}")
    print(f"    agreement                   : {'PASS' if agree else 'DISAGREEMENT'}")
else:
    print("    agreement                   : PENDING -- P031's file was absent when")
    print("                                  this ran; the coordinator diffs the two.")

# ------------------------------------------------------------------ 6e
cascade_count = None
cascade_indices: list[int] = []
cascade_longest = 0
if P031_FILE.exists() and "cascade_overrun_count" in payload:
    print("6e. The second accounting P031 publishes: CASCADE overruns")
    print("    P031 also reports a cascade count under the recurrence")
    print("        r[i] = i*T ; s[i] = max(r[i], c[i-1]) ; c[i] = s[i] + d[i]")
    print("        overrun iff c[i] > r[i] + deadline,  with c[-1] = 0")
    print("    which is a DIFFERENT question from the direct one above: it asks")
    print("    whether an iteration finished late once earlier overruns have")
    print("    pushed its start, not whether its own duration exceeded the")
    print("    budget. rtclock.loop.FixedRateLoop on a SimulatedTimebase")
    print("    implements exactly that recurrence (release at t0 + k*T, wait")
    print("    skipped when already past it), so it reproduces the quantity")
    print("    independently rather than reimplementing P031's expression.")
    print()
    deadline_s = float(payload.get("deadline_s", period_s))
    tb = SimulatedTimebase()
    loop = FixedRateLoop(period_s=period_s, timebase=tb, mode="absolute",
                         deadline_s=deadline_s)
    report = loop.run(body=lambda k: tb.sleep(trace[k]), iterations=len(trace))
    cascade_indices = [
        rec.index for rec in report.records
        if rec.end_s > rec.scheduled_s + deadline_s
    ]
    cascade_count = len(cascade_indices)
    longest = run = 0
    previous = -2
    for i in cascade_indices:
        run = run + 1 if i == previous + 1 else 1
        longest = max(longest, run)
        previous = i
    cascade_longest = longest
    p031_cascade = int(payload["cascade_overrun_count"])
    p031_cascade_idx = [int(i) for i in payload.get("cascade_overrun_indices", [])]
    p031_longest = payload.get("max_consecutive_cascade")
    agree_cascade = cascade_count == p031_cascade and cascade_indices == p031_cascade_idx
    if not agree_cascade:
        failures.append(
            f"6e cascade DISAGREEMENT: P031 {p031_cascade}, rtclock {cascade_count}"
        )
    print(f"    deadline [s]                : {deadline_s:.12e}")
    print(f"    rtclock cascade count       : {cascade_count}")
    print(f"    P031 cascade count          : {p031_cascade}")
    print(f"    cascade indices identical   : {cascade_indices == p031_cascade_idx}")
    print(f"    rtclock longest cascade run : {cascade_longest}")
    print(f"    P031 max_consecutive_cascade: {p031_longest}")
    print(f"    late releases reported      : {report.late_releases} of {len(trace)}")
    print(f"    agreement                   : "
          f"{'PASS' if agree_cascade else 'DISAGREEMENT'}")
    if p031_longest is not None and int(p031_longest) != cascade_longest:
        print("    note: the longest-run figures use different definitions and are")
        print("          reported side by side rather than asserted equal.")
    print()

OUT_FILE.write_text(
    json.dumps(
        {
            "product": "P036 rtclock",
            "convention": "overrun iff latency_s > period_s, strict, no tolerance",
            "indices": "zero-based",
            "trace_defined_by": source,
            "trace": trace,
            "period_s": period_s,
            "overrun_count": rep.count,
            "overrun_indices": list(rep.indices),
            "longest_consecutive_run": rep.longest_consecutive_run,
            "worst_overshoot_s": rep.worst_overshoot_s,
            "total_samples": rep.total_samples,
            "cascade_definition": (
                "r[i]=i*T; s[i]=max(r[i], c[i-1]); c[i]=s[i]+d[i]; "
                "overrun iff c[i] > r[i]+deadline_s; c[-1]=0"
            ),
            "cascade_overrun_count": cascade_count,
            "cascade_overrun_indices": cascade_indices,
            "max_consecutive_cascade": cascade_longest,
        },
        indent=1,
    )
    + "\n"
)
print(f"    written to                  : {OUT_FILE}")
print()

print("=" * 78)
if failures:
    print(f"RESULT: {len(failures)} FAILURE(S)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: all checks PASS"
      + ("" if p031_count is not None else "; the P031 cross-check is PENDING"))
print("=" * 78)
