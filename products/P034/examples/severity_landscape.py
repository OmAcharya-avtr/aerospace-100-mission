"""Severity over two fault-parameter sweeps.

Left: a position sensor bias, swept over magnitude and injection start step.
Right: actuator loss of effectiveness, swept over retained effectiveness and
start step. Both show the severity score of the same target under the same
seed, so the two panels are directly comparable.

Writes ../screenshots/severity_landscape.png.
Runtime on the 1-core build container: about 3 s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faultinject.campaign import FaultCase, execute_case  # noqa: E402
from faultinject.faults import Injection  # noqa: E402
from faultinject.severity import SEVERE_THRESHOLD  # noqa: E402
from faultinject.taxonomy import FaultKind  # noqa: E402

SEED = 7
N_STEPS = 150
STARTS = list(range(0, 141, 14))
OFFSETS = np.logspace(-1, 1, 11)
RETAINED = np.linspace(0.0, 0.9, 11)


def sweep(kind, channel, param_name, values, starts):
    grid = np.zeros((len(values), len(starts)))
    for i, v in enumerate(values):
        for j, s in enumerate(starts):
            duration = N_STEPS - s
            inj = Injection.create(kind, channel, {param_name: float(v)}, s, duration)
            grid[i, j] = execute_case(FaultCase(inj, SEED, N_STEPS)).severity.severity
    return grid


bias = sweep(FaultKind.SENSOR_BIAS, "pos", "offset", OFFSETS, STARTS)
loss = sweep(
    FaultKind.ACTUATOR_LOSS_EFFECTIVENESS, "u", "retained", RETAINED, STARTS
)

print(f"sensor_bias severities: min {bias.min():.4f} max {bias.max():.4f} "
      f"severe cells {int((bias >= SEVERE_THRESHOLD).sum())}/{bias.size}")
print(f"actuator_loss severities: min {loss.min():.4f} max {loss.max():.4f} "
      f"severe cells {int((loss >= SEVERE_THRESHOLD).sum())}/{loss.size}")

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
for ax, grid, yvals, ylabel, title in (
    (axes[0], bias, OFFSETS, "bias offset (m)", "sensor_bias on pos"),
    (
        axes[1],
        loss,
        RETAINED,
        "retained effectiveness (fraction)",
        "actuator_loss_effectiveness on u",
    ),
):
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        extent=(STARTS[0], STARTS[-1], 0, len(yvals) - 1),
    )
    ax.set_yticks(range(len(yvals)))
    ax.set_yticklabels([f"{v:.3g}" for v in yvals], fontsize=8)
    ax.set_xlabel("injection start step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}, seed {SEED}")
    fig.colorbar(im, ax=ax, label="severity")

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "screenshots" / "severity_landscape.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
