"""The optimality gap: the greedy baseline against the proven optimum.

Left: per-instance objective, greedy against exact, over the seeded family.
Right: the distribution of the ratio, with the instances where greedy ties the
optimum counted separately, because a mean alone hides that most of the family
is a tie and the loss is concentrated in a few instances.

Run: ``python examples/greedy_vs_exact.py``
"""

from __future__ import annotations

import math
import time

import numpy as np
from _common import EXACT, GREEDY, LEARNED, add_src_to_path, save, style

add_src_to_path()

import matplotlib.pyplot as plt  # noqa: E402

from activityplan.exact import solve_exact  # noqa: E402
from activityplan.greedy import plan_greedy  # noqa: E402
from activityplan.instances import instance_family  # noqa: E402
from activityplan.timeline import RasterisedProblem  # noqa: E402

BASE_SEED = 9000
FAMILY_SIZE = 24


def main() -> int:
    style()
    greedy_values: list[float] = []
    exact_values: list[float] = []
    totals: list[float] = []
    names: list[str] = []
    greedy_times: list[float] = []
    exact_times: list[float] = []
    started = time.perf_counter()
    for spec in instance_family(FAMILY_SIZE, BASE_SEED):
        rasterised = RasterisedProblem.build(spec.analytic_problem(), spec.step_s)
        t0 = time.perf_counter()
        greedy = plan_greedy(rasterised)
        greedy_times.append(time.perf_counter() - t0)
        exact = solve_exact(rasterised, time_limit_s=150.0)
        exact_times.append(exact.solve_s)
        greedy_values.append(greedy.value)
        exact_values.append(exact.value)
        totals.append(rasterised.problem.total_priority)
        names.append(str(spec.seed))
    elapsed = time.perf_counter() - started

    greedy_array = np.array(greedy_values, dtype=float)
    exact_array = np.array(exact_values, dtype=float)
    ratio = greedy_array / exact_array
    order = np.argsort(ratio)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), width_ratios=(2.0, 1.0))

    ax = axes[0]
    positions = np.arange(FAMILY_SIZE)
    ax.bar(
        positions,
        np.array(totals)[order],
        color="#ececec",
        width=0.78,
        label="total requested priority",
    )
    ax.bar(positions - 0.19, exact_array[order], color=EXACT, width=0.36, label="exact optimum")
    ax.bar(
        positions + 0.19, greedy_array[order], color=GREEDY, width=0.36, label="greedy baseline"
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([names[index] for index in order], rotation=90, fontsize=6.5)
    ax.set_xlabel("instance seed, ordered by greedy/exact ratio")
    ax.set_ylabel("priority units")
    ax.set_title(
        f"{FAMILY_SIZE} instances from seed {BASE_SEED}. Grey is what was requested; "
        "black is the proven optimum."
    )
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.hist(ratio, bins=np.linspace(0.84, 1.005, 12), color=LEARNED, alpha=0.85)
    ax.axvline(1.0, color=EXACT, linestyle="--", linewidth=1.0, label="greedy ties the optimum")
    ax.axvline(
        float(np.mean(ratio)),
        color=GREEDY,
        linewidth=1.4,
        label=f"mean {float(np.mean(ratio)):.4f}",
    )
    ties = int(np.sum(np.abs(greedy_array - exact_array) <= 1e-9))
    ax.set_xlabel("greedy objective / exact optimum")
    ax.set_ylabel("instances")
    ax.set_title(
        f"ties: {ties} of {FAMILY_SIZE}; worst {float(np.min(ratio)):.4f}", fontsize=9
    )
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "activityplan: what the greedy baseline gives up against the exact optimum",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.88, wspace=0.22)
    path = save(fig, "greedy_vs_exact.png")
    print(f"wrote {path.relative_to(add_src_to_path())}")
    print("")
    pooled = math.fsum(greedy_values) / math.fsum(exact_values)
    print(f"instances                 : {FAMILY_SIZE} from seed {BASE_SEED}")
    print(f"greedy/exact mean ratio   : {float(np.mean(ratio)):.4f}")
    print(f"greedy/exact median ratio : {float(np.median(ratio)):.4f}")
    print(f"greedy/exact worst ratio  : {float(np.min(ratio)):.4f}")
    print(f"instances greedy ties     : {ties} of {FAMILY_SIZE}")
    print(f"pooled shortfall          : {100.0 * (1.0 - pooled):.2f} %")
    print(f"greedy mean wall clock    : {float(np.mean(greedy_times)):.3f} s")
    print(f"exact mean wall clock     : {float(np.mean(exact_times)):.3f} s")
    print(f"total wall clock          : {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
