"""The learned forest against the three baselines, on both held-out splits.

Writes ``screenshots/model_against_baselines.png``.

What to notice in the figure. The left panel is the rank metrics, where a
classical baseline wins on both splits -- that is the finding and it is drawn
rather than hidden. The right panel is the cost-weighted capture, which is the
operational question: the flake-rate ranking's bar is short because it does not
know how long a test takes, and the three cost-aware predictors are all close
to the oracle.

Runs from any working directory. Reads no clock, and measures no fit time.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from flakerank.history import load_history  # noqa: E402
from flakerank.predictor import (  # noqa: E402
    IntermittencyCostPredictor,
    baseline_scores,
    build_rows,
    evaluate_split,
    make_splits,
)

LOOKBACK = 120
HORIZON = 30
TRAIN_ORIGINS = (120, 150, 180, 210)
TIME_ORIGINS = (240, 270)
MINIMUM_ATTEMPTS = 8
TOP_K = 10


def main() -> int:
    """Fit once, score four predictors on two splits, draw both."""
    _bootstrap.style()
    import matplotlib.pyplot as plt

    history = load_history(ROOT / _bootstrap.HISTORY)
    rows = build_rows(
        history,
        origins=TRAIN_ORIGINS + TIME_ORIGINS,
        lookback=LOOKBACK,
        horizon=HORIZON,
        minimum_attempts=MINIMUM_ATTEMPTS,
    )
    splits = make_splits(
        rows, train_origins=TRAIN_ORIGINS, time_origins=TIME_ORIGINS, seed=_bootstrap.SEED
    )
    trees = 30 if _bootstrap.FAST else 300
    model = IntermittencyCostPredictor(
        n_estimators=trees, random_state=_bootstrap.SEED
    ).fit(splits["train"])

    collected: dict[str, dict[str, object]] = {}
    for key in ("heldout_tests", "heldout_time"):
        split = splits[key]
        prediction = model.predict(split)
        scores = {
            "learned forest": evaluate_split(
                "learned forest", split, prediction.mean,
                seconds_scale=True, prediction=prediction, top_k=TOP_K,
            )
        }
        for name, vector in baseline_scores(
            split, horizon=HORIZON, lookback=LOOKBACK
        ).items():
            scores[name] = evaluate_split(
                name, split, vector, seconds_scale=(name != "flake rate"), top_k=TOP_K
            )
        scores["oracle"] = evaluate_split(
            "oracle", split, split.target, seconds_scale=True, top_k=TOP_K
        )
        collected[key] = scores

    order = (
        "flake rate",
        "accumulated persistence",
        "analytic expected cost",
        "learned forest",
        "oracle",
    )
    labels = ("held-out TESTS", "held-out TIME")
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    positions = np.arange(len(order))
    width = 0.38
    for offset, (key, label) in enumerate(zip(("heldout_tests", "heldout_time"), labels,
                                              strict=True)):
        taus = [collected[key][name].kendall_tau_b for name in order]
        caps = [collected[key][name].captured_fraction for name in order]
        shift = (offset - 0.5) * width
        left.bar(positions + shift, taus, width=width, label=label)
        right.bar(positions + shift, caps, width=width, label=label)

    for axis, title, ylabel in (
        (
            left,
            "rank metric: Kendall tau-b against the realised cost\n"
            "a classical baseline wins on both splits -- that is the finding",
            "Kendall tau-b (dimensionless)",
        ),
        (
            right,
            f"operational metric: fraction of the realised wasted time\n"
            f"held by each predictor's top {TOP_K}",
            "cost-weighted capture (dimensionless)",
        ),
    ):
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [name.replace(" ", "\n") for name in order], fontsize=7.5
        )
        axis.set_ylim(0.0, 1.05)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8, loc="lower right")

    figure.tight_layout()
    destination = ROOT / _bootstrap.SCREENSHOTS / "model_against_baselines.png"
    figure.savefig(destination)
    plt.close(figure)

    print("flakerank -- the learned forest against the three baselines")
    print("")
    print(f"lookback {LOOKBACK} sessions, horizon {HORIZON} sessions, "
          f"minimum {MINIMUM_ATTEMPTS} attempts per row")
    print(f"training origins {list(TRAIN_ORIGINS)}, held-out TIME origins "
          f"{list(TIME_ORIGINS)}")
    print(f"trees {trees}, seed {_bootstrap.SEED}, no model serialised")
    print("")
    for key, label in zip(("heldout_tests", "heldout_time"), labels, strict=True):
        split = splits[key]
        print(f"--- {label}, {split.n_rows} rows ---")
        print(f"{'predictor':>26} {'tau-b':>10} {'rho':>10} {'capture':>10} {'mae':>10}")
        for name in order:
            score = collected[key][name]
            mae = "n/a" if np.isnan(score.mae_log1p) else f"{score.mae_log1p:.6f}"
            print(f"{name:>26} {score.kendall_tau_b:>10.6f} {score.spearman:>10.6f}"
                  f" {score.captured_fraction:>10.6f} {mae:>10}")
        print("")

    print("THE FINDING: the analytic expected-cost baseline matches or beats the")
    print("learned forest on the rank and error metrics on both splits. The")
    print("forest's largest feature importances are the analytic cost and the")
    print("accumulated wasted time, which are the baselines' own outputs. The")
    print("result is published as the finding and was not retuned away.")
    print("")
    print(f"figure written to {_bootstrap.relative(destination)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
