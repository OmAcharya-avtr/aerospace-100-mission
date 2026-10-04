"""Benchmark the two baselines against the learned overrun predictor.

Produces ``screenshots/overrun_predictor.png``: ROC curves on both trace
families, the three metrics that decide this at a matched flag rate, and the
learned model's reliability curve. The honest headline is in the figure
itself: on the jittery family every curve sits on the diagonal.

Run: ``python examples/overrun_predictor.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hilforge.predict import (
    FixedThresholdPredictor,
    LearnedOverrunPredictor,
    OverrunMetrics,
    QueueingOverrunPredictor,
    TraceConfig,
    build_dataset,
    evaluate_predictor,
    generate_trace,
    split_trace,
)

HORIZON = 3
N_TRACE = 30_000
SEED = 4242
TARGET_FLAG_RATE = 0.15
OUT = ROOT / "screenshots" / "overrun_predictor.png"
COLOURS = {
    "fixed_threshold": "#44618c",
    "queueing_markov": "#9a7b2f",
    "learned_hgb": "#2f6f4e",
}


def _case(preset: str):
    cfg = TraceConfig.preset(preset, n_iterations=N_TRACE)
    trace = generate_trace(cfg, seed=SEED)
    train, test = split_trace(trace, train_fraction=0.6)
    deadline = cfg.period_s
    x_tr, y_tr, _ = build_dataset(train.stage_s, deadline, horizon=HORIZON)
    x_te, y_te, idx = build_dataset(test.stage_s, deadline, horizon=HORIZON)
    over = test.overruns()
    window = np.column_stack([over[idx + h] for h in range(1, HORIZON + 1)])
    predictors = [
        FixedThresholdPredictor(deadline_s=deadline).fit(x_tr, y_tr),
        QueueingOverrunPredictor(deadline_s=deadline, horizon=HORIZON).fit(x_tr, y_tr),
        LearnedOverrunPredictor().fit(x_tr, y_tr),
    ]
    for p in predictors:
        p.calibrate_flag_rate(x_tr, TARGET_FLAG_RATE)
    metrics = [
        evaluate_predictor(p, x_te, y_te, window, horizon=HORIZON) for p in predictors
    ]
    return trace, predictors, metrics, x_te, y_te


def main() -> int:
    results = {}
    for preset in ("jittery", "bursty"):
        trace, predictors, metrics, x_te, y_te = _case(preset)
        results[preset] = (trace, predictors, metrics, x_te, y_te)
        print(f"=== {preset} ===")
        print(f"utilisation                 : {trace.utilisation:.6f}")
        print(f"per-iteration overrun rate  : {float(trace.overruns().mean()):.6f}")
        print(f"operating point             : matched flag rate, target "
              f"{TARGET_FLAG_RATE:.2f}")
        print(OverrunMetrics.header())
        for m in metrics:
            print(m.as_row())
        print()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    fig.suptitle(
        "HilForge: deadline-overrun prediction — two baselines and a learned model\n"
        "left column jittery (near-independent overruns), right column bursty "
        "(autocorrelated)",
        fontsize=12,
    )

    for col, preset in enumerate(("jittery", "bursty")):
        trace, predictors, metrics, x_te, y_te = results[preset]
        ax = axes[0, col]
        for p, m in zip(predictors, metrics, strict=True):
            fpr, tpr, _ = roc_curve(y_te, p.score(x_te))
            ax.plot(fpr, tpr, lw=1.6, color=COLOURS[p.name],
                    label=f"{p.name} (AUC {m.roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="chance")
        ax.set_xlabel("false-alarm rate")
        ax.set_ylabel("true-positive rate")
        ax.set_title(
            f"{preset}: ROC on the held-out 40 %\n"
            f"utilisation {trace.utilisation:.3f}, "
            f"overrun rate {float(trace.overruns().mean()):.3f}"
        )
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax = axes[1, 0]
    width = 0.26
    offsets = np.arange(3)
    for k, (p, _m) in enumerate(
        zip(results["bursty"][1], results["bursty"][2], strict=True)
    ):
        jit = results["jittery"][2][k]
        bur = results["bursty"][2][k]
        values = [
            bur.missed_overrun_rate,
            bur.false_alarm_rate,
            jit.missed_overrun_rate,
        ]
        ax.bar(offsets + (k - 1) * width, values, width, color=COLOURS[p.name],
               label=p.name)
    ax.set_xticks(offsets)
    ax.set_xticklabels(
        ["missed overrun\n(bursty)", "false alarm\n(bursty)", "missed overrun\n(jittery)"]
    )
    ax.set_ylabel("rate")
    ax.set_title(
        f"the three metrics at a matched flag rate ({TARGET_FLAG_RATE:.2f})\n"
        "on the jittery family every predictor misses almost everything"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    for preset, marker in (("jittery", "s"), ("bursty", "o")):
        _, predictors, _, x_te, y_te = results[preset]
        learned = predictors[-1]
        table = learned.reliability_table(x_te, y_te, n_bins=12)
        good = np.isfinite(table[:, 2])
        ax.plot(table[good, 2], table[good, 3], marker=marker, lw=1.4,
                label=f"learned_hgb, {preset}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="perfect calibration")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(
        "learned model's calibrated probability output\n"
        "(the uncertainty output, held to a reliability curve)"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"figure: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
