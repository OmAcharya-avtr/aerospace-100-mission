"""ROC curves for every detection method, plus the detection-delay spread.

Left: window-level ROC for the two chi-squared windows, the channel CUSUM
bank, the classical GLR bank and the learned classifier, all on the same
held-out windows.  Right: the distribution of detection delay per method at a
common calibrated false-alarm rate.

This runs a **reduced** campaign so that it finishes in about a minute.  The
numbers quoted in the README and in ``validation/VALIDATION.md`` come from
``validation/detection_benchmark.py``, which uses three times the data; cite
those, not these.

    python examples/roc_curves.py
"""

from __future__ import annotations

import numpy as np
from _plotstyle import COLORS, save

import matplotlib.pyplot as plt  # noqa: E402  (backend set in _plotstyle)

from fdiscope import (  # noqa: E402
    BenchmarkConfig,
    FaultClassifier,
    build_default_bank,
    calibrate_all_thresholds,
    evaluate_detection,
    harvest_training_rows,
    healthy_calibration_runs,
    method_names,
    roc_curve,
    run_scenarios,
    sample_scenarios,
    window_scores,
)

N_TRAIN = 80
N_TEST = 80
N_CALIB = 60
N_TREES = 100
TARGET_RUN_FAR = 0.10


def main() -> None:
    cfg = BenchmarkConfig()
    bank = build_default_bank(cfg)
    train = sample_scenarios(N_TRAIN, 1000)
    test = sample_scenarios(N_TEST, 5000)
    train_runs = run_scenarios(train, cfg)
    test_runs = run_scenarios(test, cfg)
    x, y = harvest_training_rows(train, train_runs, cfg)
    clf = FaultClassifier(n_estimators=N_TREES, random_state=0).fit(x, y)
    _, calib = healthy_calibration_runs(N_CALIB, 9000, cfg)
    thresholds = calibrate_all_thresholds(calib, cfg, bank, clf, TARGET_RUN_FAR)

    pos, neg = window_scores(test, test_runs, cfg, bank, clf)
    results = evaluate_detection(test, test_runs, cfg, bank, clf, thresholds)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    ax = axes[0]
    curves = {}
    for name in pos:
        curve = roc_curve(pos[name], neg[name], name)
        curves[name] = curve
        ax.plot(curve.fpr, curve.tpr, color=COLORS[name], label=f"{name} (AUC {curve.auc:.3f})")
    ax.plot([0, 1], [0, 1], color=COLORS["onset"], lw=0.8, ls=":", label="chance")
    ax.set_xlabel("false-positive rate [-]")
    ax.set_ylabel("true-positive rate [-]")
    ax.set_title(
        f"window-level ROC, {pos['glr'].size} faulted and {neg['glr'].size} fault-free windows"
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")

    ax = axes[1]
    names = method_names()
    data = [results[n].delays for n in names]
    parts = ax.boxplot(
        data, tick_labels=names, showfliers=False, widths=0.6, patch_artist=True
    )
    for patch, name in zip(parts["boxes"], names, strict=True):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.35)
    for name, values in zip(names, data, strict=True):
        ax.scatter(
            [names.index(name) + 1], [np.mean(values)], color=COLORS[name], marker="D", zorder=3
        )
    ax.set_ylabel("detection delay [samples]")
    ax.set_title(f"delay at a matched {TARGET_RUN_FAR:.0%} per-run false-alarm probability")
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    path = save(fig, "roc_curves.png")
    print(f"saved {path}")
    print(f"reduced campaign: {N_TRAIN} training / {N_TEST} held-out scenarios, "
          f"{N_CALIB} calibration runs, {N_TREES} trees")
    print("NOT the published numbers; see validation/detection_benchmark.py")
    print(f"{'method':>12} {'AUC':>8} {'TPR@1%':>8} {'FAR/run':>9} {'det rate':>9} "
          f"{'mean delay':>11} {'median':>8}")
    for name in names:
        m = results[name]
        curve = curves.get(name)
        auc = f"{curve.auc:.4f}" if curve else "-"
        tpr = f"{curve.tpr_at_fpr(0.01):.4f}" if curve else "-"
        far = m.far_runs[0] / m.far_runs[1] if m.far_runs[1] else float("nan")
        print(
            f"{name:>12} {auc:>8} {tpr:>8} {far:>9.4f} {m.detection_rate:>9.4f} "
            f"{np.mean(m.delays):>11.2f} {np.median(m.delays):>8.1f}"
        )


if __name__ == "__main__":
    main()
