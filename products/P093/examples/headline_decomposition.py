"""The headline factor and its decomposition, across the duty-cycle grid.

Run it:

    python examples/headline_decomposition.py

It prints the decomposition for the reference cell in both Baseline-1 variants,
then the factor and its three named factors plus the residual for every cell of
the grid, and writes ``screenshots/headline_decomposition.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _common  # noqa: E402

ROOT = _common.bootstrap()

from powerenvelope.decomposition import decompose  # noqa: E402
from powerenvelope.report import MEASUREMENT_MODEL  # noqa: E402

NAME = "headline_decomposition"


def main() -> int:
    campaign, is_measured = _common.campaign()
    out = _common.Output(NAME, is_measured)
    out(MEASUREMENT_MODEL)
    out("")
    out(campaign.provenance.banner())
    out("")

    out("THE REFERENCE CELL, both variants of energy per inference")
    out("")
    for variant in ("measured-active", "nominal-rail"):
        for line in decompose(campaign.require("w64_d050"), variant).lines():
            out(f"  {line}")
        out("")

    out("EVERY GRID CELL, measured-active variant")
    out("")
    header = (
        f"  {'cell':<12}{'duty':>6}{'a':>8}{'F':>10}{'N/A':>9}"
        f"{'E_tot/E_act':>13}{'closure':>11}{'occupancy':>11}"
    )
    out(header)
    out("  " + "-" * (len(header) - 2))
    rows = []
    for cell in campaign.grid_cells():
        result = decompose(cell)
        rows.append(result)
        out(
            f"  {cell.label:<12}{cell.duty:>6.2f}{result.acceptance_rate:>8.4f}"
            f"{result.factor:>10.4f}{result.rejection_factor:>9.4f}"
            f"{result.duty_cycle_factor:>13.4f}{result.closure_error:>11.2e}"
            f"{result.occupancy.occupancy:>11.6f}"
        )

    out("")
    out("The three named factors multiply to F exactly: E_use = E_total/A and")
    out("E_inf = E_active/N are shares of the same two integrals over the run and its")
    out("declared phases, so there is no residual term. The closure column is the")
    out("floating-point residual of that identity and nothing else. The occupancy")
    out("column is a DIAGNOSTIC: how much of the declared active phase the inferences")
    out("actually filled.")

    handle, save = _common.figure(
        NAME,
        is_measured,
        "Energy per inference understates energy per useful decision by F, "
        "decomposed exactly into rejection and duty cycle",
    )
    axes = handle.subplots(1, 2)
    duties = sorted({cell.duty for cell in campaign.grid_cells()})
    dimensions = sorted({cell.frame_dim for cell in campaign.grid_cells()})
    for dimension in dimensions:
        factors = [
            decompose(campaign.require(f"w{dimension}_d{int(round(duty * 100)):03d}")).factor
            for duty in duties
        ]
        axes[0].plot(duties, factors, marker="o", label=f"frame {dimension}x{dimension}")
    axes[0].set_xlabel("duty cycle")
    axes[0].set_ylabel("understatement factor F")
    axes[0].set_title("F against duty cycle")
    axes[0].axhline(1.0, color="grey", linestyle=":", linewidth=1)
    axes[0].annotate(
        "F = 1: the quoted unit would be adequate",
        xy=(duties[0], 1.0),
        xytext=(duties[0], 1.0 + 0.1 * max(1.0, max(factors))),
        fontsize=8,
        color="grey",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    reference = [result for result in rows if result.cell.startswith("w64_")]
    labels = [result.cell for result in reference]
    positions = np.arange(len(labels))
    width = 0.27
    axes[1].bar(
        positions - width,
        [result.rejection_factor for result in reference],
        width,
        label="rejection 1/a",
    )
    axes[1].bar(
        positions,
        [result.duty_cycle_factor for result in reference],
        width,
        label="duty cycle E_tot/E_act",
    )
    axes[1].bar(
        positions + width,
        [result.factor for result in reference],
        width,
        label="F (their product)",
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("factor")
    axes[1].set_title("the factors, frame 64x64")
    axes[1].axhline(1.0, color="grey", linestyle=":", linewidth=1)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    written = save()
    out("")
    out(f"figure written to {written}")
    out(f"raw output written to {_common.relative(out.path)}")
    out.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
