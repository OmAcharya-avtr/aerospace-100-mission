"""Validation 2: coverage accounting against a hand enumeration.

Claim under test (Level 2 requirement): the coverage accounting matches a hand
enumeration on a small taxonomy subset.

The subset is ``{sensor_bias, timing_late_sample}``.  It was chosen because it
spans two fault classes, two channel sets (two sensor channels, one transport
channel), a log-scaled continuous parameter with three bins and an
integer-valued parameter with two bins -- that is, every structural feature of
the binning appears at least once, in 32 cells that fit on a page.

Checks
  2a. The 32 cell labels of the subset, typed out by hand below, equal
      ``coverage.all_cells`` cell for cell and in the same order.
  2b. Per-kind cell counts of the FULL taxonomy equal the hand-evaluated
      product ``channels x start bins x duration bins x parameter bins`` for all
      sixteen kinds, and the total equals 248.
  2c. Coverage after a hand-specified list of six injections equals the
      hand-computed value, with the binning arithmetic shown step by step.
  2d. Every one of the 248 cells is reachable: ``case_in_cell`` produces a case
      that lands back in the requested cell. An unreachable cell would make the
      coverage denominator a lie.
  2e. An injection outside the tracked subset is rejected instead of silently
      pushing coverage above one.

Run from products/P034/:  python validation/validate_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faultinject.campaign import case_in_cell  # noqa: E402
from faultinject.coverage import CoverageTracker, all_cells, cell_of  # noqa: E402
from faultinject.faults import Injection  # noqa: E402
from faultinject.taxonomy import (  # noqa: E402
    DURATION_FRAC,
    START_FRAC,
    FaultKind,
    kinds,
    spec,
)

N_STEPS = 150
failures: list[str] = []

SUBSET = (FaultKind.SENSOR_BIAS, FaultKind.TIMING_LATE_SAMPLE)

# --- hand enumeration -------------------------------------------------------
# sensor_bias: channels (pos, vel) x start bins (0,1) x duration bins (0,1)
#              x offset bins (0,1,2)                                    = 24
# timing_late_sample: channel (bus) x (0,1) x (0,1) x late_steps bins (0,1) = 8
HAND_CELLS = (
    "sensor_bias/pos/s0/d0/p[0]",
    "sensor_bias/pos/s0/d0/p[1]",
    "sensor_bias/pos/s0/d0/p[2]",
    "sensor_bias/pos/s0/d1/p[0]",
    "sensor_bias/pos/s0/d1/p[1]",
    "sensor_bias/pos/s0/d1/p[2]",
    "sensor_bias/pos/s1/d0/p[0]",
    "sensor_bias/pos/s1/d0/p[1]",
    "sensor_bias/pos/s1/d0/p[2]",
    "sensor_bias/pos/s1/d1/p[0]",
    "sensor_bias/pos/s1/d1/p[1]",
    "sensor_bias/pos/s1/d1/p[2]",
    "sensor_bias/vel/s0/d0/p[0]",
    "sensor_bias/vel/s0/d0/p[1]",
    "sensor_bias/vel/s0/d0/p[2]",
    "sensor_bias/vel/s0/d1/p[0]",
    "sensor_bias/vel/s0/d1/p[1]",
    "sensor_bias/vel/s0/d1/p[2]",
    "sensor_bias/vel/s1/d0/p[0]",
    "sensor_bias/vel/s1/d0/p[1]",
    "sensor_bias/vel/s1/d0/p[2]",
    "sensor_bias/vel/s1/d1/p[0]",
    "sensor_bias/vel/s1/d1/p[1]",
    "sensor_bias/vel/s1/d1/p[2]",
    "timing_late_sample/bus/s0/d0/p[0]",
    "timing_late_sample/bus/s0/d0/p[1]",
    "timing_late_sample/bus/s0/d1/p[0]",
    "timing_late_sample/bus/s0/d1/p[1]",
    "timing_late_sample/bus/s1/d0/p[0]",
    "timing_late_sample/bus/s1/d0/p[1]",
    "timing_late_sample/bus/s1/d1/p[0]",
    "timing_late_sample/bus/s1/d1/p[1]",
)

# kind -> hand-evaluated channels x 2 x 2 x product(parameter bins)
HAND_COUNTS = {
    "sensor_bias": 2 * 2 * 2 * 3,
    "sensor_drift": 2 * 2 * 2 * 3,
    "sensor_stuck": 2 * 2 * 2,
    "sensor_dropout": 2 * 2 * 2 * 3,
    "sensor_quant_collapse": 2 * 2 * 2 * 3,
    "actuator_loss_effectiveness": 1 * 2 * 2 * 3,
    "actuator_stuck": 1 * 2 * 2,
    "actuator_runaway": 1 * 2 * 2 * 3,
    "bus_delay": 1 * 2 * 2 * 3,
    "bus_reorder": 1 * 2 * 2 * 3,
    "bus_loss": 1 * 2 * 2 * 3,
    "timing_late_sample": 1 * 2 * 2 * 2,
    "timing_overrun": 1 * 2 * 2 * 3,
    "numerical_nan": 3 * 2 * 2,
    "numerical_denormal": 3 * 2 * 2,
    "numerical_overflow": 3 * 2 * 2 * 3,
}
HAND_TOTAL = 248

print("=" * 78)
print("VALIDATION 2 -- coverage accounting vs a hand enumeration")
print("=" * 78)
print()
print("Binning in use:")
print(f"  start_frac    range [{START_FRAC.lo}, {START_FRAC.hi}], "
      f"{START_FRAC.n_bins} linear bins, edges {START_FRAC.edges()}")
print(f"  duration_frac range [{DURATION_FRAC.lo}, {DURATION_FRAC.hi}], "
      f"{DURATION_FRAC.n_bins} linear bins, edges {DURATION_FRAC.edges()}")
bias_param = spec(FaultKind.SENSOR_BIAS).param("offset")
late_param = spec(FaultKind.TIMING_LATE_SAMPLE).param("late_steps")
print(f"  sensor_bias.offset  range [{bias_param.lo}, {bias_param.hi}] "
      f"{bias_param.unit}, {bias_param.n_bins} log bins, edges "
      + ", ".join(f"{e:.6f}" for e in bias_param.edges()))
print(f"  timing_late_sample.late_steps range [{late_param.lo}, {late_param.hi}] "
      f"{late_param.unit}, {late_param.n_bins} linear bins, edges "
      + ", ".join(f"{e:.6f}" for e in late_param.edges()))
print()

# ------------------------------------------------------------------------- 2a
print("2a. Hand enumeration of the subset vs coverage.all_cells")
computed = tuple(c.label() for c in all_cells(SUBSET))
print(f"    hand cells      {len(HAND_CELLS)}")
print(f"    computed cells  {len(computed)}")
if computed == HAND_CELLS:
    print("    PASS  identical, cell for cell and in the same order")
else:
    only_hand = [c for c in HAND_CELLS if c not in computed]
    only_comp = [c for c in computed if c not in HAND_CELLS]
    failures.append(
        f"2a: hand enumeration differs; only-hand={only_hand[:5]} only-computed={only_comp[:5]}"
    )
    print("    FAILED")
    for c in only_hand[:10]:
        print(f"      only in hand list: {c}")
    for c in only_comp[:10]:
        print(f"      only computed:     {c}")
print()

# ------------------------------------------------------------------------- 2b
print("2b. Per-kind cell counts of the full taxonomy vs hand-evaluated products")
print()
print(f"    {'kind':<30} {'hand':>6} {'computed':>9} {'match':>6}")
print("    " + "-" * 54)
total_computed = 0
for kind in kinds():
    sp = spec(kind)
    hand = HAND_COUNTS[kind.value]
    got = sp.n_cells
    total_computed += got
    ok = hand == got
    print(f"    {kind.value:<30} {hand:>6} {got:>9} {str(ok):>6}")
    if not ok:
        failures.append(f"2b: {kind.value} cells hand={hand} computed={got}")
print("    " + "-" * 54)
print(f"    {'TOTAL':<30} {HAND_TOTAL:>6} {total_computed:>9} "
      f"{str(HAND_TOTAL == total_computed):>6}")
if total_computed != HAND_TOTAL:
    failures.append(f"2b: total cells hand={HAND_TOTAL} computed={total_computed}")
print()

# ------------------------------------------------------------------------- 2c
print("2c. Coverage after a hand-specified campaign, arithmetic shown")
print()
campaign = [
    ("sensor_bias", "pos", {"offset": 0.2}, 10, 20, "sensor_bias/pos/s0/d0/p[0]"),
    ("sensor_bias", "pos", {"offset": 1.0}, 10, 20, "sensor_bias/pos/s0/d0/p[1]"),
    ("sensor_bias", "pos", {"offset": 0.2}, 10, 20, "sensor_bias/pos/s0/d0/p[0]"),
    ("sensor_bias", "vel", {"offset": 5.0}, 120, 30, "sensor_bias/vel/s1/d0/p[2]"),
    ("timing_late_sample", "bus", {"late_steps": 1.0}, 0, 150,
     "timing_late_sample/bus/s0/d1/p[0]"),
    ("timing_late_sample", "bus", {"late_steps": 4.0}, 75, 75,
     "timing_late_sample/bus/s1/d0/p[1]"),
]
tracker = CoverageTracker(SUBSET)
for kind, channel, params, start, dur, expected in campaign:
    inj = Injection.create(kind, channel, params, start, dur)
    sf = start / N_STEPS
    df = dur / N_STEPS
    got = cell_of(inj, N_STEPS)
    tracker.add(inj, N_STEPS)
    pname = next(iter(params))
    pspec = spec(kind).param(pname)
    print(f"    {kind}/{channel} {pname}={params[pname]:g} start={start} duration={dur}")
    print(f"      start_frac    = {start}/{N_STEPS} = {sf:.6f} -> bin "
          f"{START_FRAC.bin_of(sf)}")
    print(f"      duration_frac = {dur}/{N_STEPS} = {df:.6f} -> bin "
          f"{DURATION_FRAC.bin_of(df)}")
    print(f"      {pname:<13} = {params[pname]:g} -> bin {pspec.bin_of(params[pname])}")
    print(f"      hand cell     = {expected}")
    print(f"      computed cell = {got.label()}")
    if got.label() != expected:
        failures.append(f"2c: {kind}/{channel} hand={expected} computed={got.label()}")
        print("      FAILED")
    print()
hand_covered = 5
hand_fraction = hand_covered / 32
print(f"    hand-computed distinct cells covered : {hand_covered} of 32 "
      f"= {hand_fraction:.12f}")
print(f"    tracker.covered / tracker.total      : {tracker.covered} of {tracker.total} "
      f"= {tracker.fraction:.12f}")
print(f"    six injections, one repeated, so hit count of the repeated cell = 2: "
      f"{max(tracker.hits.values())}")
if tracker.covered != hand_covered or tracker.total != 32:
    failures.append(
        f"2c: coverage hand=({hand_covered}/32) computed=({tracker.covered}/{tracker.total})"
    )
elif abs(tracker.fraction - hand_fraction) > 0:
    failures.append("2c: coverage fraction differs from the hand value")
else:
    print("    PASS")
print()
per_kind = tracker.per_kind()
print("    per-kind breakdown (hand: sensor_bias 3 of 24, timing_late_sample 2 of 8)")
for kind, (cov, tot) in per_kind.items():
    print(f"      {kind.value:<22} {cov} of {tot}")
if per_kind[FaultKind.SENSOR_BIAS] != (3, 24) or per_kind[
    FaultKind.TIMING_LATE_SAMPLE
] != (2, 8):
    failures.append(f"2c: per-kind breakdown {per_kind} differs from the hand values")
print()

# ------------------------------------------------------------------------- 2d
print("2d. Every cell of the full taxonomy is reachable by a constructed case")
unreachable = []
for cell in all_cells():
    try:
        case = case_in_cell(cell, seed=1, n_steps=N_STEPS)
        if case.cell() != cell:
            unreachable.append(cell.label())
    except (ValueError, AssertionError) as exc:
        unreachable.append(f"{cell.label()}: {exc}")
print(f"    cells tested     {len(all_cells())}")
print(f"    unreachable      {len(unreachable)}")
if unreachable:
    failures.append(f"2d: {len(unreachable)} unreachable cell(s): {unreachable[:5]}")
    for u in unreachable[:10]:
        print(f"      {u}")
else:
    print("    PASS")
print()

# ------------------------------------------------------------------------- 2e
print("2e. An injection outside the tracked subset is rejected")
outside = Injection.create(FaultKind.BUS_LOSS, "bus", {"loss_prob": 0.5}, 10, 20)
try:
    CoverageTracker(SUBSET).add(outside, N_STEPS)
    failures.append("2e: an out-of-subset injection was accepted")
    print("    FAILED  accepted silently")
except ValueError as exc:
    print(f"    PASS    ValueError raised: {exc}")
print()

print("=" * 78)
if failures:
    print(f"RESULT: FAILED ({len(failures)} check(s))")
    for f in failures:
        print(f"  FAILED  {f}")
    sys.exit(1)
print("RESULT: PASS -- all checks in validation 2 passed")
print("=" * 78)
