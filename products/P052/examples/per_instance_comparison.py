"""Per-instance strategy comparison, with the baseline's wins marked.

A grouped bar chart of the mean curve probability -- the average probability of
having found a violation over the whole budget -- for every strategy on every
instance, ordered from easiest to hardest. Bars where uniform random beats the
strategy are hatched, because those are the result a reader needs to see.

Saves ``../screenshots/per_instance_comparison.png``.
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
from falsifyloop.search import BASELINE  # noqa: E402

BUDGET = 100
REPEATS = 20
BASE_SEED = 7300

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
    names = report.strategy_names
    width = 0.8 / len(names)
    x = np.arange(len(instances))

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [3, 2]}
    )

    for k, name in enumerate(names):
        values = [report.cell(i.identifier, name).mean_curve_probability for i in instances]
        baseline = [
            report.cell(i.identifier, BASELINE).mean_curve_probability for i in instances
        ]
        offsets = x + (k - (len(names) - 1) / 2) * width
        pairs = zip(values, baseline, strict=True)
        hatches = ["///" if (name != BASELINE and b > v) else "" for v, b in pairs]
        bars = top.bar(offsets, values, width, label=name, color=COLOURS[name], edgecolor="white")
        for bar, hatch in zip(bars, hatches, strict=True):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor("black")
    top.set_xticks(x)
    top.set_xticklabels(
        [f"{i.identifier}\n[{i.tier}]" for i in instances], fontsize=9
    )
    top.set_ylabel("mean curve probability")
    top.set_ylim(0, 1.05)
    top.grid(axis="y", alpha=0.25)
    top.legend(ncol=len(names), fontsize=9, frameon=False, loc="upper right")
    top.set_title(
        f"Mean curve probability per instance, budget {BUDGET}, {REPEATS} seeds. "
        "Hatched bars are the ones uniform random (black) beats.",
        fontsize=11,
    )

    # Lower panel: the number of simulations the median run needed, censored.
    for k, name in enumerate(names):
        medians = []
        for inst in instances:
            value = report.cell(inst.identifier, name).median_simulations
            medians.append(BUDGET * 1.12 if value is None else value)
        offsets = x + (k - (len(names) - 1) / 2) * width
        bottom.bar(offsets, medians, width, color=COLOURS[name], edgecolor="white")
    bottom.axhline(BUDGET, color="black", lw=1.0, ls="--")
    bottom.text(
        -0.45,
        BUDGET * 1.14,
        f"bars at this height = median not reached within {BUDGET} simulations (censored)",
        fontsize=8.5,
        va="bottom",
    )
    bottom.set_xticks(x)
    bottom.set_xticklabels([i.identifier for i in instances], fontsize=9)
    bottom.set_ylabel("median simulations to first violation")
    bottom.set_ylim(0, BUDGET * 1.3)
    bottom.grid(axis="y", alpha=0.25)
    bottom.set_title(
        "Median simulations to the first violation. Lower is better; a censored bar "
        "means more than half the runs found nothing.",
        fontsize=11,
    )

    fig.suptitle(
        "Falsification is one-sided: finding no violation is not evidence of correctness.",
        fontsize=10,
        y=0.005,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = ROOT / "screenshots" / "per_instance_comparison.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)

    print(f"wrote {out.relative_to(ROOT)}")
    print("")
    print(f"{'strategy':<22s} {'instances the baseline beats it on':<50s}")
    for name in names:
        if name == BASELINE:
            continue
        losses = report.baseline_wins(name)
        print(f"{name:<22s} {(', '.join(losses) if losses else 'none'):<50s}")
    print("")
    print(f"benchmark wall clock: {report.wall_clock_seconds:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
