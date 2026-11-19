"""Which mechanism dominates, as a map over the two quantities that decide it.

A map over buffer headroom (seconds of source output the free space absorbs) and
the product ``rho * T_ramp`` (which sets the drain term, because the closed form
gives ``L_drain = R_s * rho * T_ramp / 2`` once the buffer is full). The
protocol restart is held at three declared values so its effect is visible as
three panels rather than averaged away.

The figure is the product's positioning in one picture: the answer to "which
mechanism dominates" is not a constant, so a tool that reports only a total
cannot answer the question an operator is asking.

Writes ``screenshots/dominance_map.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _common  # noqa: E402

ROOT = _common.bootstrap()

from handoverloss.baselines import fluid_decomposition_array  # noqa: E402
from handoverloss.sequence import MECHANISMS  # noqa: E402

#: Grid resolution. The whole figure is closed-form arithmetic, so it is cheap.
GRID = 40 if _common.FAST else 140

#: Declared restart latencies, seconds, one panel each.
RESTARTS = (3.0, 20.0, 55.0)

#: Fixed for the map: a 1 Gbit/s ground station and a 40 Gbit buffer. The source
#: rate is set by the utilisation, which is one of the swept axes.
DOWNLINK_BPS = 1.0e9
CAPACITY_BITS = 4.0e10
ACQUISITION_S = 35.0


def main() -> int:
    headroom_s = np.linspace(1.0, 120.0, GRID)
    rho = np.linspace(0.04, 0.96, GRID)
    ramp_s = 60.0

    handle, save = _common.figure("dominance_map", size=(13.0, 4.6))
    axes = handle.subplots(1, len(RESTARTS), sharey=True)
    colours = np.array([[0.21, 0.38, 0.56], [0.85, 0.65, 0.13], [0.55, 0.20, 0.35]])
    fractions = {}

    for panel, restart in enumerate(RESTARTS):
        head_grid, rho_grid = np.meshgrid(headroom_s, rho, indexing="ij")
        source = rho_grid * DOWNLINK_BPS
        # headroom_s = (B - q0) / R_s, so the occupancy that gives a target
        # headroom in seconds is B - R_s * headroom_s, clipped into the buffer.
        occupancy = np.clip(CAPACITY_BITS - source * head_grid, 0.0, CAPACITY_BITS)
        horizon = ACQUISITION_S + restart + ramp_s + 400.0
        losses = fluid_decomposition_array(
            source,
            np.full_like(source, DOWNLINK_BPS),
            np.full_like(source, CAPACITY_BITS),
            occupancy,
            np.full_like(source, ACQUISITION_S),
            np.full_like(source, restart),
            np.full_like(source, ramp_s),
            np.full_like(source, horizon),
        )
        dominant = np.argmax(losses, axis=-1)
        # Where there is no loss at all, the argmax is meaningless; mask it.
        total = losses.sum(axis=-1)
        image = colours[dominant]
        image[total <= 0.0] = 0.92
        axes[panel].imshow(
            image,
            origin="lower",
            aspect="auto",
            extent=(rho.min(), rho.max(), headroom_s.min(), headroom_s.max()),
        )
        axes[panel].set_xlabel(r"utilisation $\rho = R_s / R_d$")
        axes[panel].set_title(f"declared restart {restart:.0f} s")
        counted = {
            name: float(np.mean((dominant == column) & (total > 0.0)))
            for column, name in enumerate(MECHANISMS)
        }
        fractions[restart] = (counted, float(np.mean(total <= 0.0)))
    axes[0].set_ylabel("buffer headroom at the break [s of source output]")
    handle.suptitle(
        "Which mechanism dominates the handover loss (baseline 2, closed form)\n"
        "blue = acquisition, gold = protocol restart, plum = buffer drain, "
        "grey = no loss"
    )

    path = save()
    print("WHAT TO NOTICE")
    print(
        "  The three regions move as the declared protocol restart grows. At a 3 s\n"
        "  restart the picture is acquisition against drain and the restart mechanism is\n"
        "  almost absent; at 55 s the restart region takes most of the plane. Nothing\n"
        "  about the total loss would have told an operator that, which is the product's\n"
        "  positioning: a network simulator reports the total, and the total does not\n"
        "  say whether to buy faster acquisition or more buffer."
    )
    print("")
    print(f"  grid                         {GRID} x {GRID} per panel, closed form only")
    print(f"  fixed R_d                    {DOWNLINK_BPS:.3e} bit/s")
    print(f"  fixed buffer                 {CAPACITY_BITS:.3e} bits")
    print(f"  fixed acquisition            {ACQUISITION_S:.1f} s")
    print(f"  fixed ramp                   {ramp_s:.1f} s")
    print("")
    for restart, (counted, empty) in fractions.items():
        rendered = "  ".join(f"{name}={value:.4f}" for name, value in counted.items())
        print(f"  restart {restart:5.1f} s: {rendered}   no-loss fraction={empty:.4f}")
    print("")
    print(f"figure written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
