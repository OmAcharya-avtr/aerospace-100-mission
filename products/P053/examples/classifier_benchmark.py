"""The learned classifier against the analytic baselines, and its calibration.

Saves ``../screenshots/classifier_benchmark.png``. What to notice: the left
panel shows the learned model ahead on the two mean-shift scenarios and well
behind on the noise-variance one; the middle panel shows the only result that
matters, a sign-flipped change of the same size that the baselines detect
every time and the learned model almost never does; the right panel is the
reliability diagram that makes the confidence output usable.
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

from twininvalidate import (  # noqa: E402
    N_FEATURES,
    SCENARIOS,
    AssetChange,
    DetectorSpec,
    DriftClassifier,
    StreamSpec,
    bracket_threshold,
    brier_score,
    calibrate_threshold,
    calibration_set,
    delay_after_onset,
    expected_calibration_error,
    in_control_streams,
    reliability_diagram,
    simulate_residuals,
    training_set,
    window_features,
)

TARGET = 1000.0
N_RUNS = 200
N_SAMPLES = 2500
BASELINES = ("cusum", "ewma", "glr")


def main() -> int:
    train = training_set()
    cal = calibration_set()
    clf = DriftClassifier().fit(train.x, train.y, cal.x, cal.y)
    bank = in_control_streams(n_runs=N_RUNS, n_samples=N_SAMPLES, seed=53001)
    thresholds = {
        n: calibrate_threshold(DetectorSpec(n), bank, TARGET).threshold for n in BASELINES
    }
    bracket = bracket_threshold(clf.statistic(bank), TARGET)
    onset = clf.window

    scenarios = [(n, SCENARIOS[n], 53200 + i) for i, n in enumerate(SCENARIOS)]
    ood = [("gain +1 %\n(sign flipped)", AssetChange("parameter_step", 0, +0.01), 53300),
           ("gain -3 %\n(larger)", AssetChange("parameter_step", 0, -0.03), 53302),
           ("process noise x 5", AssetChange("noise_variance", 0, 5.0), 53303)]

    def measure(change: AssetChange, seed: int) -> tuple[dict[str, float], dict[str, float]]:
        shifted = AssetChange(change.kind, onset, change.magnitude, change.ramp_samples)
        oc = simulate_residuals(
            StreamSpec(change=shifted, n_runs=N_RUNS, n_samples=N_SAMPLES, seed=seed)
        )
        delays, fractions = {}, {}
        for n in BASELINES:
            est, _ = delay_after_onset(DetectorSpec(n).statistic(oc), thresholds[n], onset)
            delays[n], fractions[n] = est.value, est.detection_fraction
        est, _ = delay_after_onset(clf.statistic(oc), bracket.conservative, onset)
        delays["learned"], fractions["learned"] = est.value, est.detection_fraction
        return delays, fractions

    in_dist = {name: measure(change, seed) for name, change, seed in scenarios}
    out_dist = {name: measure(change, seed) for name, change, seed in ood}

    fig = plt.figure(figsize=(14.0, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.32)

    series = [*BASELINES, "learned"]
    colours = {"cusum": "C0", "ewma": "C1", "glr": "C2", "learned": "C3"}

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(in_dist))
    width = 0.2
    for k, name in enumerate(series):
        ax.bar(
            x + (k - 1.5) * width,
            [in_dist[s][0][name] for s in in_dist],
            width,
            color=colours[name],
            label="learned classifier" if name == "learned" else DetectorSpec(name).label(),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(in_dist), fontsize=8)
    ax.set_ylabel("detection delay (samples)")
    ax.set_title(
        f"In distribution\nbaselines at ARL0 = {TARGET:.0f}, learned at "
        f"{bracket.conservative_arl0:.0f} (quantised)",
        fontsize=9.5,
    )
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, axis="y")

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(out_dist))
    for k, name in enumerate(series):
        frac = [out_dist[s][1][name] for s in out_dist]
        ax.bar(x + (k - 1.5) * width, frac, width, color=colours[name])
    ax.set_xticks(x)
    ax.set_xticklabels(list(out_dist), fontsize=8)
    ax.set_ylabel("fraction of runs detected")
    ax.set_ylim(0, 1.08)
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
    ax.set_title("Out of distribution: detection fraction", fontsize=9.5)
    ax.grid(alpha=0.25, axis="y")
    worst = out_dist["gain +1 %\n(sign flipped)"][1]["learned"]
    ax.annotate(
        f"learned: {worst:.2f}",
        xy=(0 + 1.5 * width, worst),
        xytext=(0.04, 0.52),
        fontsize=8,
        color="C3",
        arrowprops=dict(arrowstyle="->", color="C3", lw=1.0),
    )

    eval_in = in_control_streams(n_runs=40, n_samples=700, seed=53120)
    parts_x, parts_y = [], []
    f = window_features(eval_in, clf.window).reshape(-1, N_FEATURES)
    parts_x.append(f)
    parts_y.append(np.zeros(f.shape[0], dtype=int))
    for i, name in enumerate(SCENARIOS):
        z = simulate_residuals(
            StreamSpec(change=SCENARIOS[name], n_runs=14, n_samples=700, seed=53121 + i)
        )
        f = window_features(z, clf.window).reshape(-1, N_FEATURES)
        parts_x.append(f)
        parts_y.append(np.ones(f.shape[0], dtype=int))
    conf = clf.confidence(np.concatenate(parts_x))
    labels = np.concatenate(parts_y)
    bins = reliability_diagram(conf, labels, n_bins=10)
    ax = fig.add_subplot(gs[0, 2])
    ax.plot([0, 1], [0, 1], color="0.4", ls=":", lw=1.1, label="perfect calibration")
    ax.plot(
        [b.mean_confidence for b in bins],
        [b.observed_frequency for b in bins],
        marker="o",
        color="C3",
        lw=1.4,
        label="isotonic-calibrated forest",
    )
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed frequency of change")
    ax.set_title(
        f"Reliability\nBrier {brier_score(conf, labels):.3f}, "
        f"ECE {expected_calibration_error(bins):.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Learned drift classifier against the analytic baselines, thresholds set "
        "from a declared false-alarm target\n"
        "twininvalidate: the CUSUM is ahead on the step and far ahead on the "
        "variance change; the classifier collapses on a sign flip",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.06, right=0.98)
    out = ROOT / "screenshots" / "classifier_benchmark.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    print(f"classifier threshold {bracket.conservative:.6f}, ARL0 "
          f"{bracket.conservative_arl0:.1f}")
    for name, (delays, fractions) in {**in_dist, **out_dist}.items():
        cells = " ".join(
            f"{n}={delays[n]:.0f}/{fractions[n]:.2f}" for n in series
        )
        print(f"  {name.replace(chr(10), ' '):<28}{cells}")
    print(f"Brier {brier_score(conf, labels):.4f}  "
          f"ECE {expected_calibration_error(bins):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
