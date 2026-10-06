"""A knife-edge optimum and a flat one, side by side.

Writes ``../screenshots/knife_edge_vs_flat.png``.

Each panel plots the expected goodput ``R_m A_m`` of every MODCOD against the
scintillation index at a fixed margin and availability target. A MODCOD's
curve is drawn solid where it meets the availability target and dotted where
it does not -- a MODCOD can have the highest goodput and still be ineligible,
which is exactly the trap the fixed-margin rule of thumb falls into. The thick
black line is the constrained optimum, and the shaded band is the interval
over which the *chosen MODCOD does not change*.

Left: a knife-edge case, where the band is narrower than a 10 % error in the
scintillation index. Right: a flat case, where it is much wider. Both
instances and both widths come from ``validation/validate_sensitivity.py``.

Runtime: about 40 s on one contended core.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from coderateopt import (  # noqa: E402
    InfeasibleProblem,
    LognormalFade,
    RateProblem,
    illustrative_modcod_table,
    scintillation_sensitivity,
    select_rate,
)

TABLE = illustrative_modcod_table()
CASES = (
    ("knife-edge", 14.0, 0.999),
    ("flat", 22.0, 0.990),
)
SWEEP = np.linspace(0.02, 0.9, 260)
OUT = Path(__file__).resolve().parents[1] / "screenshots" / "knife_edge_vs_flat.png"

figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

for axis, (label, margin, target), y_top in zip(axes, CASES, (1.1, 3.2), strict=True):
    base = RateProblem(
        modcods=TABLE,
        fade=LognormalFade(0.2),
        margin_db=margin,
        availability_target=target,
        max_entries=1,
        mode="per_interval",
    )
    report = scintillation_sensitivity(base, grid_points=64)
    chosen = report.signature[0] if report.signature else None

    availability = np.zeros((len(TABLE), SWEEP.size))
    goodput = np.zeros_like(availability)
    optimum = np.full(SWEEP.size, np.nan)
    for column, scintillation in enumerate(SWEEP):
        instance = base.with_fade(LognormalFade(float(scintillation)))
        availability[:, column] = instance.availabilities()
        goodput[:, column] = instance.goodputs()
        try:
            optimum[column] = select_rate(instance).expected_goodput
        except InfeasibleProblem:
            continue

    colours = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(TABLE)))
    for index, entry in enumerate(TABLE):
        eligible = availability[index] >= target
        axis.plot(
            SWEEP,
            np.where(eligible, goodput[index], np.nan),
            color=colours[index],
            linewidth=1.4,
            label=entry.name,
        )
        axis.plot(
            SWEEP,
            np.where(eligible, np.nan, goodput[index]),
            color=colours[index],
            linewidth=0.9,
            linestyle=":",
        )
    axis.plot(SWEEP, optimum, color="black", linewidth=2.6, label="constrained optimum")
    axis.axvspan(
        report.lower,
        report.upper,
        color="#dce8f2",
        zorder=0,
        label="interval where the choice is unchanged",
    )
    axis.axvline(report.nominal, color="black", linestyle="--", linewidth=0.9)
    name = TABLE.names[chosen] if chosen is not None else "infeasible"
    axis.set_title(
        f"{label}: margin {margin:.1f} dB, target {target}\n"
        f"chosen {name}; stable over "
        f"[{report.lower:.4f}, {report.upper:.4f}], relative width "
        f"{report.relative_width:.3f}"
    )
    axis.set_xlabel("scintillation index")
    axis.set_ylim(0.0, y_top)
    axis.grid(True, alpha=0.3)
    print(
        f"{label}: margin {margin:.1f} dB target {target} chosen {name} "
        f"interval [{report.lower:.6f}, {report.upper:.6f}] relative width "
        f"{report.relative_width:.6f} knife_edge {report.is_knife_edge} "
        f"headroom down/up {report.downward_headroom:.6f}/{report.upward_headroom:.6f}"
    )

for axis in axes:
    axis.set_ylabel("expected goodput, bits/symbol")
axes[1].legend(loc="upper right", fontsize=7.5, ncol=2)
figure.tight_layout()
figure.savefig(OUT, dpi=140)
print(f"wrote {OUT.parent.name}/{OUT.name}")
print("dotted segments are MODCODs whose availability is below the target at that point")
