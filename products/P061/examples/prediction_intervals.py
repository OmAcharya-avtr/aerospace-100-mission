"""What the learned component's uncertainty output is worth.

Left: predicted against realised duration on held-out activities, with the 90 %
conformal interval drawn as an error bar.
Middle: coverage by duration tercile for both conformal variants against the
nominal 0.90. The constant-width (absolute) variant undercovers long activities
badly; that is the figure's point.
Right: the interval width against the analytic duration, showing that the
normalised variant spends its width where the error is.

Run: ``python examples/prediction_intervals.py``
"""

from __future__ import annotations

import numpy as np
from _common import ANALYTIC, EXACT, GREEDY, LEARNED, ROBUST, add_src_to_path, save, style

add_src_to_path()

import matplotlib.pyplot as plt  # noqa: E402

from activityplan.features import (  # noqa: E402
    analytic_duration_s,
    feature_matrix,
    realise,
    sample_requests,
)
from activityplan.predictor import (  # noqa: E402
    DurationEnergyPredictor,
    Interval,
    analytic_predictions,
    coverage,
    mean_interval_width,
)

SEED = 61
N_TOTAL = 1400
N_TEST = 400
ALPHA = 0.10


def main() -> int:
    style()
    rng = np.random.default_rng(SEED)
    requests = sample_requests(N_TOTAL, rng)
    realised = [realise(request, rng) for request in requests]
    features = feature_matrix(list(requests))
    durations = np.array([item.duration_s for item in realised], dtype=float)
    energies = np.array([item.energy_wh for item in realised], dtype=float)
    split = N_TOTAL - N_TEST
    analytic_test = np.array(
        [analytic_duration_s(request) for request in requests[split:]], dtype=float
    )

    predictors = {}
    intervals = {}
    for mode in ("absolute", "normalised"):
        predictor = DurationEnergyPredictor(
            alpha=ALPHA, mode=mode, n_estimators=160, random_state=SEED
        ).fit(features[:split], durations[:split], energies[:split])
        predictors[mode] = predictor
        intervals[mode] = predictor.duration.predict(features[split:])

    truth = durations[split:]
    analytic_point, _ = analytic_predictions(requests[split:])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    ax = axes[0]
    interval = intervals["normalised"]
    show = np.argsort(truth)[::7]
    ax.errorbar(
        truth[show],
        interval.point[show],
        yerr=np.vstack(
            (
                interval.point[show] - interval.lower[show],
                interval.upper[show] - interval.point[show],
            )
        ),
        fmt="o",
        markersize=2.6,
        linewidth=0.7,
        color=LEARNED,
        ecolor=LEARNED,
        alpha=0.65,
        label="learned, 90 % conformal",
    )
    ax.scatter(
        truth[show],
        analytic_point[show],
        s=7,
        color=ANALYTIC,
        marker="x",
        linewidths=0.8,
        label="analytic model",
    )
    limits = (float(min(truth)) * 0.9, float(max(truth)) * 1.05)
    ax.plot(limits, limits, color=EXACT, linestyle="--", linewidth=1.0, label="perfect")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("realised duration [s]")
    ax.set_ylabel("predicted duration [s]")
    ax.set_title(
        f"held-out activities (every 7th of {N_TEST} shown)\n"
        f"learned MAE {float(np.mean(np.abs(truth - interval.point))):.2f} s, "
        f"analytic MAE {float(np.mean(np.abs(truth - analytic_point))):.2f} s"
    )
    ax.legend(loc="upper left", fontsize=7)

    ax = axes[1]
    cuts = np.percentile(analytic_test, [100.0 / 3.0, 200.0 / 3.0])
    masks = {
        "short": analytic_test <= cuts[0],
        "middle": (analytic_test > cuts[0]) & (analytic_test <= cuts[1]),
        "long": analytic_test > cuts[1],
    }
    positions = np.arange(3)
    width = 0.36
    measured: dict[str, list[float]] = {}
    for offset, (mode, colour) in enumerate((("absolute", GREEDY), ("normalised", LEARNED))):
        values = []
        for label in ("short", "middle", "long"):
            mask = masks[label]
            sliced = Interval(
                point=intervals[mode].point[mask],
                lower=intervals[mode].lower[mask],
                upper=intervals[mode].upper[mask],
            )
            values.append(coverage(truth[mask], sliced))
        measured[mode] = values
        bars = ax.bar(
            positions + (offset - 0.5) * width, values, width=width, color=colour, label=mode
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                fontsize=7,
                color=colour,
            )
    ax.axhline(
        1.0 - ALPHA, color=EXACT, linestyle="--", linewidth=1.1, label=f"nominal {1.0 - ALPHA:.2f}"
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(["short", "middle", "long"])
    ax.set_xlabel("tercile of analytic duration")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0.0, 1.08)
    ax.set_title(
        "conditional coverage. Marginal coverage is guaranteed;\nconditional coverage is not,"
        " and the constant-width\nvariant fails it on the long tercile."
    )
    ax.legend(loc="lower left", fontsize=7)

    ax = axes[2]
    for mode, colour in (("absolute", GREEDY), ("normalised", LEARNED)):
        order = np.argsort(analytic_test)
        ax.plot(
            analytic_test[order],
            intervals[mode].width[order],
            color=colour,
            linewidth=1.2,
            label=f"{mode}, mean {mean_interval_width(intervals[mode]):.1f} s",
        )
    ax.plot(
        np.sort(analytic_test),
        2.0 * 1.6449 * 0.08 * np.sort(analytic_test),
        color=ROBUST,
        linestyle=":",
        linewidth=1.3,
        label="generator's own 90 % band",
    )
    ax.set_xlabel("analytic duration [s]")
    ax.set_ylabel("interval width [s]")
    ax.set_title(
        "where each variant spends its width.\nThe dotted line is the width the generator's\n"
        "heteroscedastic noise actually calls for."
    )
    ax.legend(loc="upper left", fontsize=7)

    fig.suptitle(
        "activityplan: the learned predictor's interval, measured rather than asserted",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.80, wspace=0.28)
    path = save(fig, "prediction_intervals.png")
    print(f"wrote {path.relative_to(add_src_to_path())}")
    print("")
    print(f"seed                        : {SEED}, {N_TOTAL} activities, {N_TEST} held out")
    for mode in ("absolute", "normalised"):
        print(
            f"{mode:11s} marginal coverage {coverage(truth, intervals[mode]):.4f}, "
            f"mean width {mean_interval_width(intervals[mode]):.4f} s"
        )
        print(
            f"            terciles short {measured[mode][0]:.4f}, "
            f"middle {measured[mode][1]:.4f}, long {measured[mode][2]:.4f}"
        )
    print(f"nominal coverage            : {1.0 - ALPHA:.4f}")
    print(
        f"learned MAE {float(np.mean(np.abs(truth - intervals['normalised'].point))):.4f} s "
        f"against analytic MAE {float(np.mean(np.abs(truth - analytic_point))):.4f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
