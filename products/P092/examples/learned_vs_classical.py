"""The learned predictor against the two baselines on both held-out splits.

Draws, from the MEASURED campaign:

- left: predicted against measured window miss rate on the held-out-mix split,
  for the learned forest, the classical speed-scaling predictor and the
  constant-rate null model, with the learned predictor's reported one-sigma
  uncertainty as error bars;
- right: the mean absolute error of each predictor on each split, side by side,
  so whichever predictor wins is visible rather than argued.

Writes ``screenshots/learned_vs_classical.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _common  # noqa: E402

ROOT = _common.bootstrap()

from deadlinerisk.predictor import (  # noqa: E402
    ConstantRatePredictor,
    ContentionMissPredictor,
    score,
)
from deadlinerisk.speed_scaling import BOARD_CORES, predicted_miss_probability  # noqa: E402
from deadlinerisk.trace import TraceDataAbsentError  # noqa: E402
from deadlinerisk.windows import extract_windows, grouped_splits, time_split  # noqa: E402

HOLDOUT_FRACTION = 0.3


def main() -> int:
    try:
        campaign = _common.measured_campaign()
    except TraceDataAbsentError as error:
        print(f"REFUSED: {error}")
        return 1

    windows = extract_windows(campaign)
    idle = campaign.jobs["mix_c0"]

    truth: list[float] = []
    learned: list[float] = []
    sigma: list[float] = []
    constant: list[float] = []
    classical: list[float] = []
    for _label, train, test in grouped_splits(windows):
        model = ContentionMissPredictor().fit(
            windows.features[train], windows.targets[train], windows.feature_names
        )
        prediction = model.predict(windows.features[test], windows.releases[test])
        null = ConstantRatePredictor().fit(windows.features[train], windows.targets[train])
        workers = int(windows.features[test, 2][0])
        truth.extend(windows.targets[test].tolist())
        learned.extend(prediction.rate.tolist())
        sigma.extend(prediction.sigma.tolist())
        constant.extend(null.predict(windows.features[test]).rate.tolist())
        classical.extend(
            [
                predicted_miss_probability(
                    idle.exec_s, campaign.deadline_s, BOARD_CORES, workers
                )
            ]
            * int(test.size)
        )
    truth_array = np.asarray(truth)
    grouped = [
        score("learned forest", truth_array, np.asarray(learned), np.asarray(sigma)),
        score("constant rate", truth_array, np.asarray(constant)),
        score("speed scaling", truth_array, np.asarray(classical)),
    ]

    train, test = time_split(windows, HOLDOUT_FRACTION)
    time_model = ContentionMissPredictor().fit(
        windows.features[train], windows.targets[train], windows.feature_names
    )
    time_prediction = time_model.predict(windows.features[test], windows.releases[test])
    time_null = ConstantRatePredictor().fit(windows.features[train], windows.targets[train])
    time_classical = np.asarray(
        [
            predicted_miss_probability(
                idle.exec_s, campaign.deadline_s, BOARD_CORES, int(workers)
            )
            for workers in windows.features[test, 2]
        ]
    )
    timed = [
        score("learned forest", windows.targets[test], time_prediction),
        score("constant rate", windows.targets[test], time_null.predict(
            windows.features[test]).rate),
        score("speed scaling", windows.targets[test], time_classical),
    ]

    handle, save = _common.figure("learned_vs_classical")
    left = handle.add_subplot(1, 2, 1)
    left.errorbar(
        truth_array, np.asarray(learned), yerr=np.asarray(sigma),
        fmt="o", ms=4, lw=0.8, capsize=2, alpha=0.75,
        label=f"learned forest, +/-1 reported sigma (MAE {grouped[0].mae:.5f})",
    )
    left.plot(
        truth_array, np.asarray(classical), "^", ms=5, alpha=0.8, color="darkorange",
        label=f"classical speed scaling (MAE {grouped[2].mae:.5f})",
    )
    left.plot(
        truth_array, np.asarray(constant), "s", ms=4, alpha=0.6, color="grey",
        label=f"constant rate null (MAE {grouped[1].mae:.5f})",
    )
    limit = float(max(truth_array.max(), np.asarray(learned).max(),
                      np.asarray(classical).max(), 1e-6)) * 1.1
    left.plot([0.0, limit], [0.0, limit], "k-", lw=1.0, label="perfect prediction")
    left.set_xlim(-0.02 * limit, limit)
    left.set_ylim(-0.02 * limit, limit)
    left.set_xlabel("measured window miss rate")
    left.set_ylabel("predicted window miss rate")
    left.set_title("held-out workload mixes (leave one run out)")
    left.grid(alpha=0.3)
    left.legend(fontsize=7, loc="upper left")

    right = handle.add_subplot(1, 2, 2)
    names = [item.label for item in grouped]
    width = 0.38
    positions = np.arange(len(names))
    right.bar(
        positions - width / 2, [item.mae for item in grouped], width,
        label="held-out mixes", color="steelblue",
    )
    right.bar(
        positions + width / 2, [item.mae for item in timed], width,
        label="held-out time", color="seagreen",
    )
    right.set_xticks(positions)
    right.set_xticklabels(names, fontsize=9)
    right.set_ylabel("mean absolute error on the window miss rate")
    right.set_title("lower is better; the winner is whichever bar is shortest")
    right.grid(alpha=0.3, axis="y")
    right.legend(fontsize=8)

    path = save()
    print(f"windows                  {len(windows)}")
    print(f"held-out runs            {len(grouped_splits(windows))}")
    print(f"time split train/test    {train.size} / {test.size}")
    print("")
    print("held-out workload mixes")
    for item in grouped:
        cover = "n/a" if item.coverage != item.coverage else f"{item.coverage:.3f}"
        print(f"  {item.label:<16} MAE {item.mae:.6f}  RMSE {item.rmse:.6f}  "
              f"bias {item.bias:+.6f}  coverage {cover}")
    print("held-out time")
    for item in timed:
        cover = "n/a" if item.coverage != item.coverage else f"{item.coverage:.3f}"
        print(f"  {item.label:<16} MAE {item.mae:.6f}  RMSE {item.rmse:.6f}  "
              f"bias {item.bias:+.6f}  coverage {cover}")
    best_grouped = min(grouped, key=lambda item: item.mae).label
    best_time = min(timed, key=lambda item: item.mae).label
    print("")
    print(f"lowest MAE, held-out mixes {best_grouped}")
    print(f"lowest MAE, held-out time  {best_time}")
    print(f"figure written to        {path}")
    print(
        "What to notice: whether the learned points sit closer to the diagonal than "
        "the orange speed-scaling markers, and whether the error bars reach the "
        "diagonal. A baseline winning either split is the finding, not a defect."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
