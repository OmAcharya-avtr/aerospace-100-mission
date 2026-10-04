"""Link-availability predictors: reliability diagram and score comparison.

Produces ``screenshots/calibration.png``:

* left panel -- reliability diagram for the climatology baseline, the
  logistic-regression baseline and the learned model, with Wilson 95 %
  intervals on the observed frequency and a histogram of forecast counts per
  bin underneath;
* right panel -- the three Brier-score components (reliability, resolution,
  uncertainty) side by side, because a single accuracy number would hide the
  thing that matters.

CALIBRATION is the headline.  On this dataset the two baselines are better
calibrated than the learned model; the figure shows that rather than hiding
it.  Full numbers are in ``validation/VALIDATION.md``.

Run: ``python examples/example_calibration.py``
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from constellink.availability import (  # noqa: E402
    ClimatologyBaseline,
    LinkAvailabilityModel,
    LogisticBaseline,
    PredictorScores,
    grouped_split,
    score_predictor,
)
from constellink.metrics import reliability_curve  # noqa: E402
from constellink.synthdata import DatasetConfig, generate_dataset  # noqa: E402

N_BINS = 10
SPLIT_SEED = 0
MODEL_SEED = 1
OUT = os.path.join(os.path.dirname(__file__), "..", "screenshots",
                   "calibration.png")


def main() -> int:
    cfg = DatasetConfig()
    data = generate_dataset(cfg)
    train, test = grouped_split(data, test_fraction=0.3, seed=SPLIT_SEED)

    clim = ClimatologyBaseline().fit(data.x[train], data.y[train],
                                      data.stratum[train])
    logi = LogisticBaseline().fit(data.x[train], data.y[train])
    learned = LinkAvailabilityModel(seed=MODEL_SEED).fit(data.x[train],
                                                          data.y[train])
    preds = {
        "climatology baseline": clim.predict_proba(data.x[test],
                                                    data.stratum[test]),
        "logistic baseline": logi.predict_proba(data.x[test]),
        "learned (bagged GBM)": learned.predict_proba(data.x[test]),
    }
    ens_std = learned.predict_std(data.x[test])
    scores = {name: score_predictor(name, p, data.y[test], n_bins=N_BINS)
              for name, p in preds.items()}

    colours = {"climatology baseline": "tab:blue",
               "logistic baseline": "tab:orange",
               "learned (bagged GBM)": "tab:green"}

    fig = plt.figure(figsize=(13.0, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, 0])
    axh = fig.add_subplot(gs[1, 0], sharex=ax)
    axb = fig.add_subplot(gs[:, 1])

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for name, p in preds.items():
        rc = reliability_curve(p, data.y[test], n_bins=N_BINS)
        m = rc.count > 0
        ax.errorbar(rc.mean_forecast[m], rc.observed_frequency[m],
                    yerr=[rc.observed_frequency[m] - rc.ci_low[m],
                          rc.ci_high[m] - rc.observed_frequency[m]],
                    marker="o", ms=4, lw=1.3, capsize=2,
                    color=colours[name], label=name)
        axh.step(rc.bin_centre, rc.count, where="mid", color=colours[name])
    ax.set_ylabel("observed frequency of a closed contact")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Reliability diagram, grouped split by link (seed "
                 f"{SPLIT_SEED}), {N_BINS} equal-width bins\n"
                 f"{test.size} held-out contacts, test base rate "
                 f"{data.y[test].mean():.3f}; bars are Wilson 95 %")
    axh.set_xlabel("forecast probability")
    axh.set_ylabel("rows per bin")
    axh.set_yscale("symlog", linthresh=1)
    axh.grid(alpha=0.3)

    names = list(scores)
    width = 0.26
    pos = np.arange(3)
    for i, name in enumerate(names):
        s = scores[name]
        axb.bar(pos + (i - 1) * width,
                [s.reliability, s.resolution, s.uncertainty],
                width=width, color=colours[name], label=name)
    axb.set_xticks(pos)
    axb.set_xticklabels(["reliability\n(lower better)",
                         "resolution\n(higher better)",
                         "uncertainty\n(data property)"])
    axb.set_ylabel("Brier component")
    axb.grid(alpha=0.3, axis="y")
    axb.legend(loc="upper center", fontsize=8)
    best_rel = min(scores, key=lambda k: scores[k].reliability)
    axb.set_title("Murphy (1973) decomposition of the Brier score\n"
                  f"best calibrated here: {best_rel}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)

    print(f"dataset: {len(data)} contacts, base rate {data.base_rate:.4f}")
    print(f"grouped split: {train.size} train / {test.size} test")
    print("")
    print(PredictorScores.header())
    print("-" * len(PredictorScores.header()))
    for s in scores.values():
        print(s.format_row())
    print("")
    print(f"learned-model ensemble std: mean {ens_std.mean():.4f}, "
          f"max {ens_std.max():.4f}")
    print(f"best calibrated (lowest reliability term): {best_rel}")
    print(f"best Brier score: {min(scores, key=lambda k: scores[k].brier)}")
    print(f"written: {os.path.normpath(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
