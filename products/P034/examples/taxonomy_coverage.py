"""Taxonomy size and coverage growth under three search strategies.

Left panel: coverage cells contributed by each of the sixteen fault kinds.
Right panel: fraction of the 248 cells reached against campaign budget, for
uniform-random, coverage-greedy and the learned prioritiser.

Writes ../screenshots/taxonomy_coverage.png.
Runtime on the 1-core build container: about 10 s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faultinject.campaign import build_pool, evaluate_pool  # noqa: E402
from faultinject.search import coverage_greedy, learned, uniform_random  # noqa: E402
from faultinject.taxonomy import kinds, spec, total_cells  # noqa: E402

BUDGET = 150

# Three replicates per cell, so uniform random can and does repeat cells and
# the coverage-greedy strategy has something to be greedy about. With one case
# per cell every strategy would trivially open a new cell on every execution.
pool = build_pool(pool_seed=1, replicates=3)
sev = evaluate_pool(pool)


def oracle(i: int) -> float:
    return sev[i]


print(f"taxonomy: {len(kinds())} kinds, {total_cells()} coverage cells")
print(f"pool: {len(pool)} cases, three per cell")

curves = {}
ur = uniform_random(pool, oracle, BUDGET, np.random.default_rng(1))
curves["uniform random"] = ur.coverage_curve
cg = coverage_greedy(pool, oracle, BUDGET, np.random.default_rng(1))
curves["coverage greedy"] = cg.coverage_curve
lr, _ = learned(pool, oracle, BUDGET, np.random.default_rng(1), warmup=32)
curves["learned prioritiser"] = lr.coverage_curve

for name, curve in curves.items():
    print(f"  {name:<22} coverage after {BUDGET} executions: {curve[-1]:.4f}")

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.0, 5.2))

labels = [k.value for k in kinds()]
counts = [spec(k).n_cells for k in kinds()]
classes = [spec(k).fault_class.value for k in kinds()]
palette = {
    "sensor": "#4c72b0",
    "actuator": "#dd8452",
    "bus": "#55a868",
    "timing": "#c44e52",
    "numerical": "#8172b3",
}
colours = [palette[c] for c in classes]
ax_left.barh(labels, counts, color=colours)
ax_left.set_xlabel("coverage cells")
ax_left.set_title(f"Cells per fault kind (total {total_cells()})")
ax_left.invert_yaxis()
ax_left.grid(axis="x", alpha=0.3)
for cls, colour in palette.items():
    ax_left.barh([], [], color=colour, label=cls)
ax_left.legend(title="fault class", loc="lower right", fontsize=8)

for name, curve in curves.items():
    ax_right.plot(range(1, len(curve) + 1), curve, label=name)
ax_right.plot(
    [1, BUDGET],
    [1 / 248, BUDGET / 248],
    linestyle=":",
    color="black",
    label="one new cell per execution",
)
ax_right.set_xlabel("campaign budget (executions)")
ax_right.set_ylabel("fraction of the 248 cells covered")
ax_right.set_title("Coverage growth")
ax_right.grid(alpha=0.3)
ax_right.legend(fontsize=8, loc="upper left")

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "screenshots" / "taxonomy_coverage.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
