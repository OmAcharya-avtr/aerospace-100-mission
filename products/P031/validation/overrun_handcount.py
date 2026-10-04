"""Check 3 — overrun accounting reproduces a hand-counted sequence exactly.

Requirement: "overrun accounting reproduces a hand-counted overrun sequence
exactly (show the hand count in a test comment)". The hand count is worked out
in full in ``tests/test_timing.py`` above
``test_overrun_hand_counted_sequence`` and repeated here so the validation
output stands alone.

This script also writes ``validation/crosscheck_overruns.json``, the
independent cross-check record for P036 RtClock. Both overrun definitions go
into the file, with the formula for each, because the two definitions disagree
and a cross-check that does not say which one it means is worthless.

Run: ``python validation/overrun_handcount.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))

from hilforge.backends import make_backend_pair
from hilforge.loop import HilLoop, LoopConfig
from hilforge.timing import PeriodSpec, overrun_report

# --- the hand-counted trace -------------------------------------------------
# Period T = 0.010 s, implicit deadline D = T. Thirteen iterations.
HAND_TRACE = [
    0.004,
    0.012,
    0.003,
    0.011,
    0.011,
    0.002,
    0.009,
    0.010,
    0.0101,
    0.005,
    0.016,
    0.006,
    0.003,
]
HAND_PERIOD = 0.010

# DIRECT definition, d[i] > D:
#   i : d[i]    verdict
#   0 : 0.0040  no        7 : 0.0100  no  (equal is a met deadline)
#   1 : 0.0120  OVERRUN   8 : 0.0101  OVERRUN
#   2 : 0.0030  no        9 : 0.0050  no
#   3 : 0.0110  OVERRUN  10 : 0.0160  OVERRUN
#   4 : 0.0110  OVERRUN  11 : 0.0060  no
#   5 : 0.0020  no       12 : 0.0030  no
#   6 : 0.0090  no
HAND_DIRECT = [1, 3, 4, 8, 10]

# CASCADE definition, r[i] = i T, s[i] = max(r[i], c[i-1]), c[i] = s[i] + d[i],
# D[i] = r[i] + T, overrun iff c[i] > D[i], c[-1] = 0:
#   i :    r      s        c       D     verdict
#   0 : 0.000  0.0000  0.0040  0.0100  no
#   1 : 0.010  0.0100  0.0220  0.0200  OVERRUN
#   2 : 0.020  0.0220  0.0250  0.0300  no
#   3 : 0.030  0.0300  0.0410  0.0400  OVERRUN
#   4 : 0.040  0.0410  0.0520  0.0500  OVERRUN  (late because i=3 ran over)
#   5 : 0.050  0.0520  0.0540  0.0600  no
#   6 : 0.060  0.0600  0.0690  0.0700  no
#   7 : 0.070  0.0700  0.0800  0.0800  no       (exactly on the deadline)
#   8 : 0.080  0.0800  0.0901  0.0900  OVERRUN
#   9 : 0.090  0.0901  0.0951  0.1000  no
#  10 : 0.100  0.1000  0.1160  0.1100  OVERRUN
#  11 : 0.110  0.1160  0.1220  0.1200  OVERRUN  (d = 0.006 < D, late anyway)
#  12 : 0.120  0.1220  0.1250  0.1300  no
HAND_CASCADE = [1, 3, 4, 8, 10, 11]
HAND_COMPLETIONS = [
    0.0040,
    0.0220,
    0.0250,
    0.0410,
    0.0520,
    0.0540,
    0.0690,
    0.0800,
    0.0901,
    0.0951,
    0.1160,
    0.1220,
    0.1250,
]
HAND_MAX_RUN = 2
HAND_RUN_START = 3

# --- the cross-check trace for P036 ----------------------------------------
# A longer deterministic trace, under the 2000-sample limit the cross-check
# asks for. Generated from a fixed seed so P036 can regenerate it from the
# JSON rather than from this code.
CROSSCHECK_N = 1200
CROSSCHECK_SEED = 36031
CROSSCHECK_PERIOD = 0.010


def crosscheck_trace() -> np.ndarray:
    """Per-iteration durations [s] for the P036 cross-check.

    Gamma body with a Bernoulli exponential spike, rounded to whole
    nanoseconds so the trace survives JSON round-tripping exactly: a
    cross-check that disagrees because of a decimal-printing difference would
    be a false finding.
    """
    rng = np.random.Generator(np.random.PCG64(CROSSCHECK_SEED))
    body = rng.gamma(5.0, 0.0062 / 5.0, size=CROSSCHECK_N)
    spike = (rng.random(CROSSCHECK_N) < 0.09) * rng.exponential(0.0075, size=CROSSCHECK_N)
    return np.round((body + spike) * 1e9) / 1e9


def main() -> int:
    print("HilForge validation 3 — overrun accounting against a hand count")
    print("=" * 74)
    print(f"period T = {HAND_PERIOD:.6e} s, implicit deadline D = T")
    print(f"trace    = {HAND_TRACE}")
    print()
    account = overrun_report(HAND_TRACE, HAND_PERIOD)
    checks: list[tuple[str, object, object]] = [
        ("direct overrun indices", HAND_DIRECT, account.direct_indices),
        ("direct overrun count", len(HAND_DIRECT), account.direct_count),
        ("cascade overrun indices", HAND_CASCADE, account.cascade_indices),
        ("cascade overrun count", len(HAND_CASCADE), account.cascade_count),
        ("max consecutive cascade", HAND_MAX_RUN, account.max_consecutive_cascade),
        ("first cascade run start", HAND_RUN_START, account.first_cascade_run_start),
    ]
    failures = 0
    print("3a. the account against the hand count")
    for label, want, got in checks:
        ok = want == got
        failures += 0 if ok else 1
        print(f"  {label:<26} hand={want!s:<22} code={got!s:<22} "
              f"{'PASS' if ok else 'FAIL'}")
    resid = float(np.max(np.abs(np.asarray(account.completion_s) - HAND_COMPLETIONS)))
    ok = resid <= 1e-15
    failures += 0 if ok else 1
    print(f"  {'completion times':<26} max |difference| = {resid:.3e} s "
          f"(tolerance 1e-15)  {'PASS' if ok else 'FAIL'}")
    print()

    print("3b. the same trace through the loop, not just the accounting function")
    sim, _ = make_backend_pair(seed=1, sample_dt_s=HAND_PERIOD)
    cfg = LoopConfig(
        period=PeriodSpec(period_s=HAND_PERIOD),
        n_iterations=len(HAND_TRACE),
        injected_durations_s=tuple(HAND_TRACE),
    )
    record = HilLoop(sim, cfg).run()
    loop_direct = [it.index for it in record.iterations if it.direct_overrun]
    loop_cascade = [it.index for it in record.iterations if it.cascade_overrun]
    for label, want, got in (
        ("loop direct indices", HAND_DIRECT, loop_direct),
        ("loop cascade indices", HAND_CASCADE, loop_cascade),
        ("loop account direct", len(HAND_DIRECT), record.overruns.direct_count),
        ("loop account cascade", len(HAND_CASCADE), record.overruns.cascade_count),
    ):
        ok = want == got
        failures += 0 if ok else 1
        print(f"  {label:<26} hand={want!s:<22} code={got!s:<22} "
              f"{'PASS' if ok else 'FAIL'}")
    print()

    print("3c. cross-check record for P036 RtClock")
    trace = crosscheck_trace()
    cross = overrun_report(trace, CROSSCHECK_PERIOD)
    payload = {
        "trace": [float(x) for x in trace],
        "period_s": CROSSCHECK_PERIOD,
        "overrun_count": cross.direct_count,
        "overrun_indices": list(cross.direct_indices),
        "definition": (
            "overrun_count and overrun_indices use the DIRECT definition: "
            "iteration i overruns iff d[i] > deadline_s. Equality is a met "
            "deadline, not an overrun."
        ),
        "deadline_s": CROSSCHECK_PERIOD,
        "n_iterations": int(trace.size),
        "cascade_overrun_count": cross.cascade_count,
        "cascade_overrun_indices": list(cross.cascade_indices),
        "cascade_definition": (
            "r[i] = i * period_s; s[i] = max(r[i], c[i-1]); c[i] = s[i] + d[i]; "
            "D[i] = r[i] + deadline_s; overrun iff c[i] > D[i]; c[-1] = 0"
        ),
        "max_consecutive_cascade": cross.max_consecutive_cascade,
        "mean_duration_s": float(trace.mean()),
        "utilisation": float(trace.mean() / CROSSCHECK_PERIOD),
        "trace_units": "seconds, rounded to whole nanoseconds",
        "generator": (
            f"numpy PCG64 seed {CROSSCHECK_SEED}: gamma(shape=5, scale=0.0062/5) "
            f"plus Bernoulli(0.09) * exponential(0.0075), rounded to 1 ns"
        ),
        "source": "hilforge 0.1.0, validation/overrun_handcount.py",
        "is_hardware": False,
        "note": (
            "Synthetic injected trace. Not a hardware measurement. Written for "
            "the independent cross-check with P036 RtClock."
        ),
    }
    out = HERE / "crosscheck_overruns.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  samples                    : {trace.size} (limit 2000)")
    print(f"  period                     : {CROSSCHECK_PERIOD:.6e} s")
    print(f"  mean duration              : {float(trace.mean()):.9e} s")
    print(f"  utilisation                : {float(trace.mean() / CROSSCHECK_PERIOD):.6f}")
    print(f"  direct overrun count       : {cross.direct_count}")
    print(f"  cascade overrun count      : {cross.cascade_count}")
    print(f"  max consecutive cascade    : {cross.max_consecutive_cascade}")
    print(f"  first 12 direct indices    : {cross.direct_indices[:12]}")
    print(f"  written                    : {out.name}")
    # The JSON must round-trip the trace exactly, or the cross-check could
    # disagree over decimal printing rather than over arithmetic.
    reread = json.loads(out.read_text())
    exact = np.array_equal(np.asarray(reread["trace"], dtype=np.float64), trace)
    failures += 0 if exact else 1
    print(f"  JSON round-trips the trace : {exact}  "
          f"{'PASS' if exact else 'FAIL'}")
    print()
    print(f"verdict: {'PASS' if failures == 0 else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
