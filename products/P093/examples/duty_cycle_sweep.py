"""The three baselines against each other across the duty-cycle sweep.

Run it:

    python examples/duty_cycle_sweep.py

It prints the measured background floor, then for every grid cell the energy per
inference (both variants), the analytic duty-cycle model's energy per useful
decision and the directly integrated ground truth, and writes
``screenshots/duty_cycle_sweep.png``, which also shows the rail power trace of
one cycle so the sleep window and the wake transient are visible.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _common  # noqa: E402

ROOT = _common.bootstrap()

from powerenvelope.baselines import (  # noqa: E402
    RAMP_WINDOW_S,
    duty_cycle_model,
    energy_per_inference,
    floor_power,
    ground_truth,
)
from powerenvelope.energy import active_windows  # noqa: E402
from powerenvelope.report import floor_lines  # noqa: E402

NAME = "duty_cycle_sweep"
REFERENCE_DIMENSION = 64


def main() -> int:
    campaign, is_measured = _common.campaign()
    out = _common.Output(NAME, is_measured)
    out(campaign.provenance.banner())
    out("")

    floor = floor_power(campaign)
    out("THE MEASURED BACKGROUND FLOOR")
    for line in floor_lines(floor):
        out(f"  {line}")
    out("")

    out("THE THREE BASELINES, per grid cell")
    out("")
    header = (
        f"  {'cell':<12}{'duty':>6}{'E/inf meas':>12}{'E/inf nom':>12}"
        f"{'analytic':>12}{'truth':>12}{'F':>9}"
    )
    out(header)
    out("  " + "-" * (len(header) - 2))
    sweep: dict[int, list[tuple[float, float, float, float]]] = {}
    for cell in campaign.grid_cells():
        measured = energy_per_inference(cell, "measured-active")
        nominal = energy_per_inference(cell, "nominal-rail")
        analytic = duty_cycle_model(campaign, cell.label, floor=floor)
        truth = ground_truth(cell)
        out(
            f"  {cell.label:<12}{cell.duty:>6.2f}{measured.energy_mj:>12.5f}"
            f"{nominal.energy_mj:>12.5f}"
            f"{analytic.energy_per_useful_decision_mj:>12.4f}"
            f"{truth.energy_per_useful_decision_mj:>12.4f}"
            f"{truth.energy_per_useful_decision_mj / measured.energy_mj:>9.3f}"
        )
        sweep.setdefault(cell.frame_dim, []).append(
            (
                cell.duty,
                measured.energy_mj,
                analytic.energy_per_useful_decision_mj,
                truth.energy_per_useful_decision_mj,
            )
        )
    out("")
    out("All five columns are in mJ except F, which is dimensionless. The analytic")
    out("column is Baseline 2 and the truth column is Baseline 3; the two are different")
    out("quantities only in that one is modelled from measured components and the other")
    out("is the integral divided by the counted accepted decisions.")

    handle, save = _common.figure(
        NAME,
        is_measured,
        "Energy per inference against energy per useful decision, across the duty sweep",
    )
    axes = handle.subplots(1, 2)
    for dimension, rows in sorted(sweep.items()):
        rows.sort()
        duties = [row[0] for row in rows]
        axes[0].plot(
            duties, [row[3] for row in rows], marker="o", label=f"truth, frame {dimension}"
        )
        axes[0].plot(
            duties,
            [row[2] for row in rows],
            marker="s",
            linestyle="--",
            label=f"analytic, frame {dimension}",
        )
        axes[0].plot(
            duties,
            [row[1] for row in rows],
            marker="^",
            linestyle=":",
            label=f"energy per inference, frame {dimension}",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("duty cycle")
    axes[0].set_ylabel("mJ (log scale)")
    axes[0].set_title("per useful decision, against per inference")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].grid(alpha=0.3, which="both")

    cell = campaign.require(f"w{REFERENCE_DIMENSION}_d050")
    starts, ends = active_windows(cell.require_decisions())
    t0 = float(cell.power.t[0])
    window = (cell.power.t - t0 >= 2.0) & (cell.power.t - t0 <= 6.0)
    axes[1].plot(
        cell.power.t[window] - t0,
        cell.power.vdd_in_mw[window],
        drawstyle="steps-post",
        linewidth=1.2,
        label="VDD_IN power",
    )
    axes[1].axhline(
        floor.mean_mw, color="grey", linestyle="--", linewidth=1, label="measured floor"
    )
    for lo, hi in zip(starts, ends, strict=True):
        if 2.0 <= lo - t0 <= 6.0:
            axes[1].axvspan(lo - t0, min(hi - t0, 6.0), color="tab:orange", alpha=0.18)
            axes[1].axvspan(
                lo - t0, min(lo - t0 + RAMP_WINDOW_S, 6.0), color="tab:red", alpha=0.22
            )
    # The sampler is spawned about half a second before the pipeline's own t0,
    # so the origin here is the FIRST RAIL SAMPLE and not the run start. The
    # shaded active windows are placed from the decisions' own CLOCK_MONOTONIC
    # timestamps, which is why they do not begin at a whole second.
    axes[1].set_xlabel("seconds after the first rail sample")
    axes[1].set_ylabel("mW")
    axes[1].set_title(f"two cycles of {cell.label}: active shaded, settling darker")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].annotate(
        f"current quantum {float(np.mean(cell.power.quantisation_floor_mw)):.1f} mW",
        xy=(0.02, 0.92),
        xycoords="axes fraction",
        fontsize=8,
        color="grey",
    )

    written = save()
    out("")
    out(f"figure written to {written}")
    out(f"raw output written to {_common.relative(out.path)}")
    out.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
