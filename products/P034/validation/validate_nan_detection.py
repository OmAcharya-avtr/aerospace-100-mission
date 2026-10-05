"""Validation 4: NaN injection is detected rather than silently absorbed.

Claim under test (Level 2 requirement): an injected NaN must be *detected*.
Two distinct failure modes are checked, because only one of them is obvious:

  * the NaN propagates and wrecks the trace -- easy to notice;
  * the target quietly replaces it (a ``nan_to_num``, an ``if not isfinite``
    guard) so the trace stays finite and the severity score sees almost
    nothing.  A campaign tool that reported "no effect" here would be lying.
    :class:`faultinject.wrapper.NumericalMonitor` watches the wrapper boundary
    for exactly this case and reports it as ``absorbed``.

Checks
  4a. ``classify_value`` against a hand-written table of IEEE 754 values.
  4b. NaN on each channel, plain target: detected, verdict, zero detection
      latency, non-finite trace, severity 1.0.
  4c. NaN on the position channel with a sanitising target: the trace stays
      finite and the severity drops, but the monitor still reports the NaN and
      labels it ``absorbed``.
  4d. Negative control: a fault-free run reports nothing.
  4e. Every numerical_nan coverage cell (12 of them) is detected.
  4f. Denormal and overflow injections, with the documented limitation that a
      large-but-finite magnitude below the surveillance threshold is NOT
      flagged by the monitor and is caught only by the severity score.

Run from products/P034/:  python validation/validate_nan_detection.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faultinject.campaign import FaultCase, case_in_cell, execute_case  # noqa: E402
from faultinject.coverage import all_cells  # noqa: E402
from faultinject.faults import SUBNORMAL_MIN, Injection  # noqa: E402
from faultinject.harness import nominal_trace, run_case  # noqa: E402
from faultinject.severity import score  # noqa: E402
from faultinject.target import SanitisingGncController  # noqa: E402
from faultinject.taxonomy import FaultKind  # noqa: E402
from faultinject.wrapper import (  # noqa: E402
    LARGE_MAGNITUDE,
    SMALLEST_NORMAL,
    classify_value,
)

SEED = 3
N_STEPS = 150
START = 50
DURATION = 20

failures: list[str] = []

print("=" * 78)
print("VALIDATION 4 -- NaN injection is detected, not silently absorbed")
print("=" * 78)
print()
print(f"Smallest normal binary64          {SMALLEST_NORMAL:.17e}")
print(f"Smallest subnormal binary64       {SUBNORMAL_MIN:.17e}")
print(f"Surveillance threshold (sqrt max) {LARGE_MAGNITUDE:.17e}")
print()

# ------------------------------------------------------------------------- 4a
print("4a. classify_value against a hand-written table")
table = (
    (float("nan"), "nan"),
    (-float("nan"), "nan"),
    (float("inf"), "inf"),
    (-float("inf"), "inf"),
    (0.0, None),
    (-0.0, None),
    (1.0, None),
    (-1e100, None),
    (SMALLEST_NORMAL, None),
    (SMALLEST_NORMAL / 2.0, "subnormal"),
    (SUBNORMAL_MIN, "subnormal"),
    (-SUBNORMAL_MIN, "subnormal"),
    (LARGE_MAGNITUDE, None),
    (LARGE_MAGNITUDE * 2.0, "large"),
    (-1e200, "large"),
)
print(f"    {'value':>26} {'expected':>10} {'got':>10} {'ok':>5}")
print("    " + "-" * 55)
for value, expected in table:
    got = classify_value(value)
    ok = got == expected
    shown = "nan" if math.isnan(value) else f"{value:.6e}"
    print(f"    {shown:>26} {str(expected):>10} {str(got):>10} {str(ok):>5}")
    if not ok:
        failures.append(f"4a: classify_value({shown}) = {got}, expected {expected}")
print()

# ------------------------------------------------------------------------- 4b
print("4b. NaN injected on each channel, plain target")
print()
print(f"    {'channel':>8} {'detected':>9} {'verdict':>11} {'first step':>11} "
      f"{'latency':>8} {'finite':>7} {'severity':>9}")
print("    " + "-" * 70)
for channel in ("pos", "vel", "u"):
    inj = Injection.create(FaultKind.NUMERICAL_NAN, channel, {}, START, DURATION)
    case = FaultCase(inj, SEED, N_STEPS)
    res = execute_case(case, keep_trace=True)
    mon = res.monitor
    verdict = mon["verdicts"].get("nan", "absent")  # type: ignore[union-attr]
    first = mon["first_in"].get("nan") or mon["first_out"].get("nan")  # type: ignore[union-attr]
    step = first[0] if first else -1
    latency = step - START if first else None
    finite = res.trace.finite() if res.trace else None
    print(f"    {channel:>8} {str(mon['detected']):>9} {verdict:>11} {step:>11} "
          f"{str(latency):>8} {str(finite):>7} {res.severity.severity:>9.6f}")
    if not mon["detected"]:
        failures.append(f"4b: NaN on {channel} was not detected")
    if latency != 0:
        failures.append(f"4b: NaN on {channel} detection latency {latency}, expected 0")
    if res.severity.severity != 1.0:
        failures.append(
            f"4b: NaN on {channel} severity {res.severity.severity}, expected exactly 1.0"
        )
    if finite:
        failures.append(f"4b: NaN on {channel} left a fully finite trace")
print()
print("    Verdict meanings: 'propagated' entered the target and left it;")
print("    'emitted' was written on the actuator side so it never entered.")
print()

# ------------------------------------------------------------------------- 4c
print("4c. The hard case: a target that silently replaces the NaN")
inj = Injection.create(FaultKind.NUMERICAL_NAN, "pos", {}, START, DURATION)
sanitised = run_case([inj], SEED, N_STEPS, target=SanitisingGncController())
nominal = nominal_trace(SEED, N_STEPS)
rep = score(sanitised, nominal)
verdict = sanitised.monitor["verdicts"].get("nan", "absent")  # type: ignore[union-attr]
print("    target                  SanitisingGncController (replaces non-finite with 0)")
print(f"    trace finite            {sanitised.finite()}")
print(f"    severity score          {rep.severity:.6f} ({rep.label})")
print(f"    monitor detected        {sanitised.monitor['detected']}")
print(f"    monitor verdict         {verdict}")
print(f"    first seen entering     {sanitised.monitor['first_in'].get('nan')}")
print(f"    first seen leaving      {sanitised.monitor['first_out'].get('nan')}")
if not sanitised.monitor["detected"]:
    failures.append("4c: an absorbed NaN was not detected at all")
elif verdict != "absorbed":
    failures.append(f"4c: absorbed NaN reported as {verdict!r}, expected 'absorbed'")
elif not sanitised.finite():
    failures.append("4c: the sanitising target did not actually keep the trace finite")
else:
    print("    PASS  the severity score alone would have called this a mild fault;")
    print("          the boundary monitor is what makes it visible.")
print()

# ------------------------------------------------------------------------- 4d
print("4d. Negative control -- fault-free run")
nom = nominal_trace(SEED, N_STEPS)
print(f"    detected {nom.monitor['detected']}, classes {nom.monitor['classes']}")
if nom.monitor["detected"]:
    failures.append("4d: the monitor fired on a fault-free run")
else:
    print("    PASS")
print()

# ------------------------------------------------------------------------- 4e
print("4e. Every numerical_nan coverage cell is detected")
nan_cells = [c for c in all_cells() if c.kind is FaultKind.NUMERICAL_NAN]
missed = []
for cell in nan_cells:
    case = case_in_cell(cell, seed=SEED, n_steps=N_STEPS)
    res = execute_case(case)
    if not res.monitor["detected"]:
        missed.append(cell.label())
print(f"    cells   {len(nan_cells)}")
print(f"    missed  {len(missed)}")
if missed:
    failures.append(f"4e: {len(missed)} numerical_nan cell(s) undetected: {missed}")
else:
    print("    PASS")
print()

# ------------------------------------------------------------------------- 4f
print("4f. Denormal and overflow, including the documented blind spot")
print()
print(f"    {'kind':>20} {'channel':>8} {'magnitude':>12} {'classes':>22} "
      f"{'verdict':>11} {'severity':>9}")
print("    " + "-" * 90)
rows = [(FaultKind.NUMERICAL_DENORMAL, ch, {}) for ch in ("pos", "vel", "u")]
rows += [
    (FaultKind.NUMERICAL_OVERFLOW, "pos", {"magnitude": m})
    for m in (1e8, 1e50, 1e150, 1e250)
]
blind_spot_seen = False
for kind, channel, params in rows:
    inj = Injection.create(kind, channel, params, START, DURATION)
    res = execute_case(FaultCase(inj, SEED, N_STEPS))
    classes = res.monitor["classes"]
    verdicts = res.monitor["verdicts"]
    verdict = ",".join(f"{v}" for v in verdicts.values()) or "-"  # type: ignore[union-attr]
    mag = params.get("magnitude", float("nan"))
    mstr = "-" if math.isnan(mag) else f"{mag:.1e}"
    print(f"    {kind.value:>20} {channel:>8} {mstr:>12} {str(classes):>22} "
          f"{verdict:>11} {res.severity.severity:>9.6f}")
    if kind is FaultKind.NUMERICAL_DENORMAL and "subnormal" not in classes:
        failures.append(f"4f: denormal on {channel} not detected as subnormal")
    if kind is FaultKind.NUMERICAL_OVERFLOW and mag <= LARGE_MAGNITUDE:
        if classes:
            failures.append(
                f"4f: magnitude {mag:.1e} below the threshold was flagged as {classes}; "
                "the documented blind spot no longer matches the code"
            )
        else:
            blind_spot_seen = True
            if res.severity.severity < 0.6:
                failures.append(
                    f"4f: magnitude {mag:.1e} is neither flagged by the monitor nor "
                    f"severe ({res.severity.severity:.3f}); it would be missed entirely"
                )
print()
if blind_spot_seen:
    print("    DOCUMENTED LIMITATION, verified above: an injected magnitude below")
    print(f"    {LARGE_MAGNITUDE:.6e} is a finite normal number and the numerical monitor")
    print("    does not flag it. Those cases are caught by the severity score's")
    print("    divergence test instead, and the table above shows their severities.")
else:
    failures.append("4f: the blind-spot rows did not run, so the limitation is unverified")
print()

print("=" * 78)
if failures:
    print(f"RESULT: FAILED ({len(failures)} check(s))")
    for f in failures:
        print(f"  FAILED  {f}")
    sys.exit(1)
print("RESULT: PASS -- all checks in validation 4 passed")
print("=" * 78)
