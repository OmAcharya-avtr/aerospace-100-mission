"""Severe faults found against campaign budget, for all four strategies.

Shaded bands are the 2.5th to 97.5th percentile over strategy seeds on a fixed
case pool. The dashed line is the expectation of uniform random sampling,
``budget x (fraction of the pool that is severe)``, computed from the
exhaustively evaluated pool.

The 'kind mean' curve is the non-learned ablation: rank untried cases by the
running mean severity of their fault kind. It is in this figure because it is
the honest comparison for the learned model, not because the specification
asked for it.

Writes ../screenshots/search_comparison.png.
Runtime on the 1-core build container: about 12 s.
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
from faultinject.search import (  # noqa: E402
    coverage_greedy,
    kind_mean,
    learned,
    uniform_random,
)
from faultinject.severity import SEVERE_THRESHOLD  # noqa: E402

BUDGET = 80
WARMUP = 32
SEEDS = (1, 2, 3, 4, 5, 6)

pool = build_pool(pool_seed=1, replicates=2)
sev = evaluate_pool(pool)
severe_frac = float(np.mean([1.0 if s >= SEVERE_THRESHOLD else 0.0 for s in sev]))


def oracle(i: int) -> float:
    return sev[i]


print(f"pool {len(pool)} cases, severe fraction {severe_frac:.4f}, "
      f"threshold {SEVERE_THRESHOLD}")

curves: dict[str, np.ndarray] = {}
for name in ("uniform random", "coverage greedy", "kind mean", "learned"):
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        if name == "uniform random":
            res = uniform_random(pool, oracle, BUDGET, rng)
        elif name == "coverage greedy":
            res = coverage_greedy(pool, oracle, BUDGET, rng)
        elif name == "kind mean":
            res = kind_mean(pool, oracle, BUDGET, rng, warmup=WARMUP)
        else:
            res, _ = learned(pool, oracle, BUDGET, rng, warmup=WARMUP)
        rows.append(res.found_curve)
    curves[name] = np.asarray(rows, dtype=float)
    final = curves[name][:, -1]
    print(f"  {name:<17} severe found at budget {BUDGET}: mean {final.mean():.2f}, "
          f"range {final.min():.0f}-{final.max():.0f}")

fig, ax = plt.subplots(figsize=(8.4, 5.4))
x = np.arange(1, BUDGET + 1)
for name, rows in curves.items():
    mean = rows.mean(axis=0)
    lo = np.percentile(rows, 2.5, axis=0)
    hi = np.percentile(rows, 97.5, axis=0)
    line, = ax.plot(x, mean, label=name)
    ax.fill_between(x, lo, hi, alpha=0.18, color=line.get_color())
ax.plot(
    x,
    x * severe_frac,
    linestyle="--",
    color="black",
    linewidth=1.0,
    label="random expectation",
)
ax.axvline(WARMUP, color="grey", linewidth=0.8, linestyle=":")
ax.text(
    WARMUP + 1,
    1.0,
    "learned warm-up ends",
    rotation=90,
    fontsize=8,
    color="grey",
    va="bottom",
)
ax.set_xlabel("campaign budget (executions)")
ax.set_ylabel(f"severe cases found (severity >= {SEVERE_THRESHOLD})")
ax.set_title(
    f"Severe faults found per budget, {len(SEEDS)} strategy seeds, pool of {len(pool)}"
)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "screenshots" / "search_comparison.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
