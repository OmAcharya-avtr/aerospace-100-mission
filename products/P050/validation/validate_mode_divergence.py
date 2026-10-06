"""Validation: the two readings of "availability" give different answers.

``mode="per_interval"`` requires every MODCOD in the operating table to meet
the availability target on its own. ``mode="long_run"`` requires only the
time-averaged delivery probability of the mix to meet it. The two coincide at
``K = 1`` and diverge as soon as mixing is allowed, and the divergence is a
real engineering trade: long-run buys goodput and pays for it in the
availability of the worst interval.

This script measures, over a grid of margins, scintillation indices and
targets:

* how often the two modes give a different answer at ``K = 2``;
* the goodput gained by the long-run reading;
* the worst-interval availability given up to get it, in probability and as a
  dB shortfall against the target.

It also confirms the claimed identity at ``K = 1``.

Runtime: about 60 s on one contended core.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from coderateopt import (  # noqa: E402
    InfeasibleProblem,
    LognormalFade,
    RateProblem,
    illustrative_modcod_table,
    select_rate,
)

TABLE = illustrative_modcod_table()
MARGINS = np.arange(8.0, 22.01, 0.5)
SCINTILLATIONS = (0.05, 0.1, 0.2, 0.4, 0.8)
TARGETS = (0.90, 0.95, 0.99, 0.995, 0.999)
failures: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


def build(margin: float, scintillation: float, target: float, mode: str, k: int) -> RateProblem:
    return RateProblem(
        modcods=TABLE,
        fade=LognormalFade(scintillation),
        margin_db=margin,
        availability_target=target,
        max_entries=k,
        mode=mode,
    )


print("1. the two modes coincide at K = 1")
identical = 0
compared = 0
for margin in MARGINS:
    for scintillation in SCINTILLATIONS:
        for target in TARGETS:
            try:
                a = select_rate(build(float(margin), scintillation, target, "per_interval", 1))
                b = select_rate(build(float(margin), scintillation, target, "long_run", 1))
            except InfeasibleProblem:
                continue
            compared += 1
            if a.support == b.support and abs(a.expected_goodput - b.expected_goodput) < 1e-12:
                identical += 1
print(f"  feasible grid points compared {compared}")
print(f"  identical answers             {identical}")
check("per_interval and long_run agree at K = 1 everywhere", identical == compared)

print()
print("2. divergence at K = 2")
diverged = 0
feasible = 0
gains: list[float] = []
availability_costs: list[float] = []
records = []
for margin in MARGINS:
    for scintillation in SCINTILLATIONS:
        for target in TARGETS:
            try:
                strict = select_rate(
                    build(float(margin), scintillation, target, "per_interval", 2)
                )
                loose = select_rate(build(float(margin), scintillation, target, "long_run", 2))
            except InfeasibleProblem:
                continue
            feasible += 1
            gain = loose.expected_goodput / strict.expected_goodput - 1.0
            if gain > 1e-12 or loose.support != strict.support:
                diverged += 1
                gains.append(gain)
                cost = target - loose.worst_interval_availability
                availability_costs.append(cost)
                records.append(
                    (gain, cost, float(margin), scintillation, target, strict, loose)
                )
print(f"  feasible grid points          {feasible}")
print(f"  answers that differ           {diverged} ({diverged / feasible:.4f})")
if gains:
    gain_array = np.asarray(gains)
    cost_array = np.asarray(availability_costs)
    print(
        f"  goodput gain from long_run: mean {gain_array.mean():.6f}  median "
        f"{np.median(gain_array):.6f}  max {gain_array.max():.6f}"
    )
    print(
        f"  worst-interval availability shortfall against target: mean "
        f"{cost_array.mean():.6f}  median {np.median(cost_array):.6f}  max "
        f"{cost_array.max():.6f}"
    )
    check("every divergence trades goodput for worst-interval availability", bool(
        np.all(cost_array > -1e-12)
    ))

print()
print("3. the largest divergence, in full")
if records:
    records.sort(key=lambda r: r[0], reverse=True)
    gain, cost, margin, scintillation, target, strict, loose = records[0]
    print(f"  margin {margin:.2f} dB, scintillation index {scintillation}, target {target}")
    print(
        f"  per_interval: {strict.support_names(TABLE)} goodput "
        f"{strict.expected_goodput:.6f}  worst-interval availability "
        f"{strict.worst_interval_availability:.6f}"
    )
    mix = ", ".join(
        f"{TABLE.names[i]} x={loose.time_fractions[i]:.6f}" for i in loose.support
    )
    print(f"  long_run:     {mix}")
    print(
        f"                goodput {loose.expected_goodput:.6f}  long-run availability "
        f"{loose.achieved_availability:.6f}  worst-interval availability "
        f"{loose.worst_interval_availability:.6f}"
    )
    print(f"  goodput gain {gain:.6f}; worst-interval availability shortfall {cost:.6f}")
    unavailable_strict = 1.0 - strict.worst_interval_availability
    unavailable_loose = 1.0 - loose.worst_interval_availability
    print(
        f"  unavailability during the worst interval rises from "
        f"{unavailable_strict:.6e} to {unavailable_loose:.6e}, a factor of "
        f"{unavailable_loose / max(unavailable_strict, 1e-300):.2f}"
    )
    check(
        "the long-run mix meets the long-run target",
        loose.achieved_availability >= target - 1e-9,
    )
    check(
        "the long-run mix violates the target during its worst interval",
        loose.worst_interval_availability < target - 1e-12,
    )

print()
print("4. a worked divergence at the documented operating point")
point = {"margin": 12.0, "scintillation": 0.2, "target": 0.99}
POINT_ARGS = (point["margin"], point["scintillation"], point["target"])
strict = select_rate(build(*POINT_ARGS, "per_interval", 2))
loose = select_rate(build(*POINT_ARGS, "long_run", 2))
print(f"  margin {point['margin']:.2f} dB, scintillation index {point['scintillation']}, "
      f"target {point['target']}")
_point_availabilities = build(*POINT_ARGS, "long_run", 2).availabilities()
print(f"  availabilities {np.array2string(_point_availabilities, precision=6)}")
print(
    f"  per_interval -> {strict.support_names(TABLE)} at {strict.expected_goodput:.6f} "
    f"bits/symbol"
)
print(
    "  long_run     -> "
    + ", ".join(f"{TABLE.names[i]} x={loose.time_fractions[i]:.6f}" for i in loose.support)
    + f" at {loose.expected_goodput:.6f} bits/symbol"
)
print(
    f"  goodput gain {loose.expected_goodput / strict.expected_goodput - 1.0:.6f}; "
    f"worst-interval availability {loose.worst_interval_availability:.6f} against a "
    f"{point['target']} target"
)

print()
print(f"checks run: {len(failures)} failed")
if failures:
    for item in failures:
        print(f"  FAILED: {item}")
    sys.exit(1)
print("the two availability readings diverge exactly where the formulation says they must")
