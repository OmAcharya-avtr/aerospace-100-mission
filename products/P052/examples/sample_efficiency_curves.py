"""The headline figure: sample-efficiency curves with bootstrap bands.

One panel per instance, five strategies each, plus a ninth panel with the
unweighted aggregate. Saves ``../screenshots/sample_efficiency_curves.png``.

Read the per-instance panels first. The aggregate panel is last because it hides
the thing a falsification user needs: whether the strategy that wins on average
is the one that loses on their instance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from _bootstrap import add_src_to_path  # noqa: E402

ROOT = add_src_to_path()

from falsifyloop.benchmark import run_benchmark  # noqa: E402
from falsifyloop.instances import suite  # noqa: E402
from falsifyloop.search import BASELINE, STRATEGIES  # noqa: E402

BUDGET = 100
REPEATS = 20
BASE_SEED = 7100
N_BOOT = 1000

COLOURS = {
    "uniform-random": "#111111",
    "latin-hypercube": "#1f77b4",
    "simulated-annealing": "#ff7f0e",
    "cross-entropy": "#2ca02c",
    "surrogate-guided": "#d62728",
}


def main() -> int:
    report = run_benchmark(budget=BUDGET, repeats=REPEATS, base_seed=BASE_SEED)
    instances = suite()
    n_panels = len(instances) + 1
    cols = 3
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.0 * rows), sharex=True, sharey=True)
    flat = axes.ravel()
    n = np.arange(1, BUDGET + 1)

    for ax, inst in zip(flat, instances, strict=False):
        for name in report.strategy_names:
            cell = report.cell(inst.identifier, name)
            curve = cell.curve()
            lower, upper = cell.band(n_boot=N_BOOT, seed=BASE_SEED)
            width = 2.4 if name == BASELINE else 1.5
            ax.plot(n, curve, color=COLOURS[name], lw=width, label=name)
            ax.fill_between(n, lower, upper, color=COLOURS[name], alpha=0.12, linewidth=0)
        ax.set_title(
            f"{inst.identifier}  [{inst.tier}]\n"
            f"design target p = {inst.design_target_probability:.4f}",
            fontsize=10,
        )
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)

    ax = flat[len(instances)]
    for name in report.strategy_names:
        curve = report.aggregate_curve(name)
        lower, upper = report.aggregate_band(name, n_boot=N_BOOT, seed=BASE_SEED)
        width = 2.4 if name == BASELINE else 1.5
        ax.plot(n, curve, color=COLOURS[name], lw=width, label=name)
        ax.fill_between(n, lower, upper, color=COLOURS[name], alpha=0.12, linewidth=0)
    ax.set_title(
        "AGGREGATE (unweighted mean of the panels above)\nread the panels first",
        fontsize=10,
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)

    for extra in flat[n_panels:]:
        extra.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("simulations")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(violation found by n)")

    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(STRATEGIES),
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "falsifyloop sample-efficiency curves: "
        f"budget {BUDGET} simulations, {REPEATS} seeds, 95 % pointwise bootstrap bands "
        f"({N_BOOT} resamples)\n"
        "uniform random (black) is the baseline. Falsification is one-sided: finding no "
        "violation is not evidence of correctness.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))

    out = ROOT / "screenshots" / "sample_efficiency_curves.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)

    # A short textual summary so running the example is informative on its own.
    lines = [f"wrote {out.relative_to(ROOT)}", ""]
    lines.append(f"{'strategy':<22s} {'aggregate mean P':>17s} {'instances lost to baseline':>28s}")
    for name in report.strategy_names:
        lost = 0 if name == BASELINE else len(report.baseline_wins(name))
        lines.append(
            f"{name:<22s} {report.aggregate_mean_probability(name):>17.4f} {lost:>28d}"
        )
    lines.append("")
    lines.append(f"hardest instance for the baseline: {report.hardest_instance()}")
    lines.append(f"benchmark wall clock: {report.wall_clock_seconds:.1f} s")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
