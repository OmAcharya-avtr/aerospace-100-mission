"""Bytes per predictor at identical distortion, and the model's own byte cost.

The left panel is the rate comparison the specification asks for. The right
panel is the part a rate comparison alone hides: the learned predictor's
parameters have to be shipped, and the break-even array count says how many
arrays the mission must send before the bytes saved pay for the bytes held.
"""

from __future__ import annotations

from _common import PREDICTOR_COLOURS, bootstrap, fit_models, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from boundcompress import benchmark, dataset  # noqa: E402
from boundcompress.predictors import CLASSICAL_NAMES  # noqa: E402
from boundcompress.ratedistortion import geometric_bounds  # noqa: E402

style()

NAMES = (*CLASSICAL_NAMES, "learned_linear", "learned_mlp")


def main() -> None:
    models = fit_models()
    signals = {
        name: signal.values
        for name, signal in dataset.generate_all(samples=4096).items()
    }
    bounds = {
        name: geometric_bounds(values, count=5) for name, values in signals.items()
    }
    cells = benchmark.run(signals, bounds, dict(models))

    totals = {name: 0 for name in NAMES}
    for cell in cells:
        totals[cell.predictor] += cell.encoded_bytes

    figure = plt.figure(figsize=(10.0, 4.2))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.28)
    left = figure.add_subplot(grid[0, 0])
    right = figure.add_subplot(grid[0, 1])

    signal_names = list(signals)
    width = 0.13
    positions = np.arange(len(signal_names))
    for index, predictor in enumerate(NAMES):
        heights = []
        for name in signal_names:
            bound = bounds[name][len(bounds[name]) // 2]
            cell = next(
                cell
                for cell in cells
                if cell.signal == name
                and cell.bound == bound
                and cell.predictor == predictor
            )
            heights.append(cell.encoded_bytes)
        left.bar(
            positions + (index - 2.5) * width,
            heights,
            width=width,
            color=PREDICTOR_COLOURS[predictor],
            label=predictor,
        )
    left.set_xticks(positions)
    left.set_xticklabels(signal_names, rotation=30, ha="right", fontsize=7)
    left.set_ylabel("encoded bytes (actual)")
    left.set_title("Bytes at the median swept bound, identical distortion")
    left.legend(fontsize=6.8, ncol=3)

    verdict = benchmark.summarise(cells)
    labels = []
    heights = []
    colours = []
    for label, model in models.items():
        break_even = benchmark.break_even_arrays(cells, label, model.model_bytes)
        labels.append(f"learned_{label}\n{model.model_bytes} B model")
        heights.append(0.0 if break_even == float("inf") else break_even)
        colours.append(PREDICTOR_COLOURS[f"learned_{label}"])
    bars = right.bar(labels, heights, color=colours, width=0.5)
    for bar, height in zip(bars, heights, strict=True):
        text = "never pays back" if height == 0.0 else f"{height:.1f} arrays"
        right.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.6,
            text,
            ha="center",
            fontsize=8,
        )
    right.set_ylabel("arrays to break even on model bytes")
    right.set_ylim(0, max(max(heights), 1.0) * 1.35)
    right.set_title(
        f"learned wins {verdict.learned_wins} of "
        f"{verdict.learned_wins + verdict.classical_wins} cells,\n"
        f"median gain {verdict.median_learned_gain_pct:.2f}%, "
        f"worst loss {verdict.worst_learned_loss_pct:.2f}%"
    )
    right.tick_params(labelsize=7.5)

    print(save(figure, "predictor_comparison"))
    print(
        f"classical_wins={verdict.classical_wins} learned_wins={verdict.learned_wins} "
        f"best_classical={verdict.best_classical_overall} "
        f"best_learned={verdict.best_learned_overall}"
    )
    for name in sorted(totals, key=lambda key: totals[key]):
        print(f"  total bytes {name:<16s} {totals[name]}")


if __name__ == "__main__":
    main()
