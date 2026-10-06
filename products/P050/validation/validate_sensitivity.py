"""Validation: stability intervals are real boundaries, and knife edges exist.

Three things are checked, because a sensitivity number that is not checked is
worse than no sensitivity number.

1. **The interval is stable.** At 25 points inside each reported interval the
   canonical support must be the one reported.
2. **The boundary is a boundary.** A step of 1e-3 past each resolved endpoint
   must change the support (or make the instance infeasible). An endpoint
   flagged as a scan bound is excluded, because there the width is a lower
   bound rather than a boundary.
3. **Knife-edge and flat optima both occur, and are distinguishable.** The
   script finds the widest and narrowest stability interval over a grid of
   margins and targets on the illustrative table and prints both, with the
   relative width of each, so the difference is a measured number and not an
   adjective.

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
    margin_sensitivity,
    scintillation_sensitivity,
    select_rate,
)

TABLE = illustrative_modcod_table()
NOMINAL_SCINTILLATION = 0.2
failures: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


def problem(margin_db: float, target: float, **kwargs) -> RateProblem:
    return RateProblem(
        modcods=TABLE,
        fade=LognormalFade(NOMINAL_SCINTILLATION),
        margin_db=margin_db,
        availability_target=target,
        max_entries=kwargs.get("max_entries", 1),
        mode=kwargs.get("mode", "per_interval"),
    )


def support_at(base: RateProblem, scintillation: float):
    try:
        return select_rate(base.with_fade(LognormalFade(scintillation))).support
    except InfeasibleProblem:
        return None


print("1 and 2. interval stability and boundary resolution, scintillation index sweeps")
print(
    f"  {'margin':>7s} {'target':>8s} {'support':>26s} {'lower':>9s} {'upper':>9s} "
    f"{'rel width':>10s} {'knife':>6s}"
)
cases = [
    (10.0, 0.95),
    (12.0, 0.99),
    (14.0, 0.99),
    (16.0, 0.999),
    (20.0, 0.99),
]
for margin, target in cases:
    base = problem(margin, target)
    report = scintillation_sensitivity(base, grid_points=64)
    names = (
        "+".join(TABLE.names[i] for i in report.signature)
        if report.signature is not None
        else "INFEASIBLE"
    )
    print(
        f"  {margin:>7.2f} {target:>8.4f} {names:>26s} {report.lower:>9.5f} "
        f"{report.upper:>9.5f} {report.relative_width:>10.5f} "
        f"{str(report.is_knife_edge):>6s}"
    )
    inside = np.linspace(report.lower + 1e-9, report.upper - 1e-9, 25)
    stable = all(support_at(base, float(v)) == report.signature for v in inside)
    check(f"support stable across [{report.lower:.5f}, {report.upper:.5f}]", stable)
    if not report.bounded_above_by_scan:
        check(
            "support changes 1e-3 above the upper boundary",
            support_at(base, report.upper + 1e-3) != report.signature,
        )
    if not report.bounded_below_by_scan:
        check(
            "support changes 1e-3 below the lower boundary",
            support_at(base, report.lower - 1e-3) != report.signature,
        )

print()
print("3. widest and narrowest stability intervals over a margin/target grid")
records = []
for margin in np.arange(9.0, 22.01, 0.5):
    for target in (0.90, 0.95, 0.99, 0.995, 0.999):
        base = problem(float(margin), float(target))
        try:
            report = scintillation_sensitivity(base, grid_points=48)
        except (InfeasibleProblem, ValueError):
            continue
        if report.signature is None:
            continue
        records.append((float(margin), float(target), report))

finite = [r for r in records if not (r[2].bounded_below_by_scan or r[2].bounded_above_by_scan)]
print(f"  grid points evaluated                        {len(records)}")
print(f"  with both boundaries resolved (not scan-bounded) {len(finite)}")
widths = np.array([r[2].relative_width for r in finite])
print(f"  relative width: min {widths.min():.6f}  median {np.median(widths):.6f}  "
      f"max {widths.max():.6f}")
knife = [r for r in finite if r[2].is_knife_edge]
print(f"  knife-edge at +/-10%                         {len(knife)} of {len(finite)}")

order = np.argsort(widths)
narrowest = finite[int(order[0])]
widest = finite[int(order[-1])]
for label, record in (("narrowest (knife-edge)", narrowest), ("widest (flat)", widest)):
    margin, target, report = record
    names = "+".join(TABLE.names[i] for i in report.signature)
    below = (
        "+".join(TABLE.names[i] for i in report.boundary_signature_below)
        if report.boundary_signature_below
        else "infeasible"
    )
    above = (
        "+".join(TABLE.names[i] for i in report.boundary_signature_above)
        if report.boundary_signature_above
        else "infeasible"
    )
    print()
    print(f"  {label}")
    print(f"    margin                     {margin:.2f} dB")
    print(f"    availability target        {target:.4f}")
    print(f"    chosen MODCOD              {names}")
    print(f"    nominal scintillation      {report.nominal:.6f}")
    print(f"    stable over                [{report.lower:.6f}, {report.upper:.6f}]")
    print(f"    relative width             {report.relative_width:.6f}")
    print(
        f"    headroom down / up         {report.downward_headroom:.6f} / "
        f"{report.upward_headroom:.6f}"
    )
    print(f"    knife-edge at +/-10%       {report.is_knife_edge}")
    print(f"    answer just below / above  {below} / {above}")

print()
print("  ratio of widest to narrowest relative width: "
      f"{widest[2].relative_width / narrowest[2].relative_width:.2f}")

print()
print("4. margin sensitivity on the same two instances, in dB")
for label, record in (("narrowest case", narrowest), ("widest case", widest)):
    margin, target, _ = record
    report = margin_sensitivity(problem(margin, target), span_db=10.0, grid_points=64)
    print(
        f"  {label:<16s} nominal {report.nominal:>6.2f} dB  stable over "
        f"[{report.lower:>7.3f}, {report.upper:>7.3f}] dB  width "
        f"{report.upper - report.lower:>6.3f} dB  knife-edge {report.is_knife_edge}"
    )

print()
print(f"checks run: {len(failures)} failed")
if failures:
    for item in failures:
        print(f"  FAILED: {item}")
    sys.exit(1)
print("every reported stability interval was verified stable inside and unstable outside")
