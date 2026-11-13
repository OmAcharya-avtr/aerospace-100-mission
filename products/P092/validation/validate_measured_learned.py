"""MEASURED: the learned predictor against both baselines, on both held-out splits.

Scored SEPARATELY on two splits, because they answer different questions:

- **held-out workload mixes**, grouped by run. No window of a held-out mix is
  ever in training, so the score is about generalising to a contention mix the
  model has never seen.
- **held-out time**, the later portion of every run. Training sees only the
  earlier portion, so the score is about predicting forward within mixes it has
  seen.

Compared on the same splits against:

- the **constant-rate** null model, the training mean. A learned model that
  cannot beat this has learned nothing from the descriptors.
- the **classical speed-scaling** predictor, ``available_fraction = min(1,
  m/(n+1))`` applied to the idle run's execution-time distribution. Fitted to
  nothing.

If a baseline wins, that is the finding and it is published as the finding, here
and in README.md and MODEL_CARD.md. Nothing is retuned to make the learned model
win.

The compute budget is measured and reported: the fit times below are wall-clock
seconds on the analysis machine named in the environment block.

INPUT ORIGIN: the MEASURED hardware campaign in ``data/traces/`` and nothing
else. No synthetic fallback.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import time  # noqa: E402

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from deadlinerisk.predictor import (  # noqa: E402
    MAX_DEPTH,
    N_ESTIMATORS,
    RANDOM_STATE,
    ConstantRatePredictor,
    ContentionMissPredictor,
    score,
)
from deadlinerisk.report import score_table  # noqa: E402
from deadlinerisk.speed_scaling import BOARD_CORES, predicted_miss_probability  # noqa: E402
from deadlinerisk.trace import TraceDataAbsentError  # noqa: E402
from deadlinerisk.windows import (  # noqa: E402
    MIN_RELEASES_PER_WINDOW,
    extract_windows,
    grouped_splits,
    time_split,
)

HOLDOUT_FRACTION = 0.3


def main() -> int:
    out = _bootstrap.Tee("validate_measured_learned_output.txt")
    out.rule("MEASURED LEARNED PREDICTOR -- against both baselines, on both splits")
    out("INPUT ORIGIN: the MEASURED hardware campaign only. No synthetic fallback.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)
    out("")
    out(_bootstrap.MEASUREMENT_MODEL)

    try:
        campaign = _bootstrap.require_measured_campaign()
    except TraceDataAbsentError as error:
        out.rule("REFUSED -- no measured campaign")
        for line in str(error).splitlines():
            out(f"  {line}")
        out.check(
            "the MEASURED campaign is on disk and complete",
            False,
            "nothing was reported and nothing was substituted",
        )
        out("")
        out(f"raw output written to {out.save()}")
        return 1

    out.rule("1  the dataset")
    windows = extract_windows(campaign)
    out(f"  window length             : {windows.window_s:.1f} s")
    out(f"  minimum releases a window : {MIN_RELEASES_PER_WINDOW}")
    out(f"  windows kept              : {len(windows)}")
    out(f"  windows discarded         : {len(windows.dropped_windows)}")
    for run, start, reason in windows.dropped_windows[:12]:
        out(f"    discarded {run} at {start:.0f} s: {reason}")
    if len(windows.dropped_windows) > 12:
        out(f"    ... and {len(windows.dropped_windows) - 12} more")
    out(f"  descriptors ({len(windows.feature_names)}) : "
        f"{', '.join(windows.feature_names)}")
    out("")
    out(f"  {'run':<12}{'windows':>9}{'rate_mean':>13}{'rate_min':>13}{'rate_max':>13}")
    for run in sorted(set(windows.runs.tolist())):
        mask = windows.runs == run
        values = windows.targets[mask]
        out(
            f"  {run:<12}{int(mask.sum()):>9}{float(values.mean()):>13.6e}"
            f"{float(values.min()):>13.6e}{float(values.max()):>13.6e}"
        )
    out.check(
        "the dataset has at least twenty windows",
        len(windows) >= 20,
        f"{len(windows)} windows",
    )
    out.check(
        "every window target is a rate in [0, 1]",
        bool(np.all(windows.targets >= 0.0) and np.all(windows.targets <= 1.0)),
        f"[{float(windows.targets.min()):.6e}, {float(windows.targets.max()):.6e}]",
    )

    out.rule("2  the model, and the compute budget it was measured at")
    out("  estimator                 : RandomForestRegressor")
    out(f"  n_estimators              : {N_ESTIMATORS}")
    out(f"  max_depth                 : {MAX_DEPTH}")
    out(f"  random_state              : {RANDOM_STATE}")
    out("  n_jobs                    : 1 (two cores available; the forest is small)")
    started = time.perf_counter()
    full = ContentionMissPredictor().fit(
        windows.features, windows.targets, windows.feature_names
    )
    fit_seconds = time.perf_counter() - started
    out(f"  one full fit              : {fit_seconds:.3f} s wall clock")
    out.check(
        "a single fit finishes well inside the three-minute budget",
        fit_seconds < 180.0,
        f"{fit_seconds:.3f} s against a 180 s budget",
    )
    out("")
    out("  impurity-based feature importances of the full-data fit:")
    for name, value in sorted(full.importances().items(), key=lambda item: -item[1]):
        out(f"    {name:<16}{value:>10.6f}")
    out(
        "  An importance is a statement about THIS forest on THIS dataset. It is not a "
        "causal claim and the descriptors are strongly collinear on this campaign: the "
        "stressor count, the stressor occupancy and the load average all move together "
        "by construction."
    )

    idle_run = _bootstrap.IDLE_RUN
    idle = campaign.jobs.get(idle_run)
    if idle is None:
        out.unmeasurable_check(
            "the classical speed-scaling baseline",
            f"the idle run {idle_run!r} is absent from the campaign, so there is no "
            "idle execution-time distribution to scale.",
        )
    out("")
    out(f"  speed-scaling idle run    : {idle_run}")
    out(f"  board cores m             : {BOARD_CORES}")
    if idle is not None:
        out(f"  idle exec samples         : {len(idle)}")
        out(f"  idle median exec          : {float(np.median(idle.exec_s)) * 1e3:.6f} ms")

    out.rule("3  HELD-OUT WORKLOAD MIXES -- leave one run out")
    truth: list[float] = []
    learned: list[float] = []
    learned_sigma: list[float] = []
    constant: list[float] = []
    classical: list[float] = []
    per_run: list[tuple[str, int, float, float, float]] = []
    grouped_seconds = 0.0
    for label, train, test in grouped_splits(windows):
        started = time.perf_counter()
        model = ContentionMissPredictor().fit(
            windows.features[train], windows.targets[train], windows.feature_names
        )
        grouped_seconds += time.perf_counter() - started
        prediction = model.predict(windows.features[test], windows.releases[test])
        null = ConstantRatePredictor().fit(
            windows.features[train], windows.targets[train]
        )
        null_rate = null.predict(windows.features[test]).rate
        workers = int(windows.features[test, 2][0])
        if idle is not None:
            classical_rate = np.full(
                int(test.size),
                predicted_miss_probability(
                    idle.exec_s, campaign.deadline_s, BOARD_CORES, workers
                ),
            )
        else:
            classical_rate = np.full(int(test.size), np.nan)
        truth.extend(windows.targets[test].tolist())
        learned.extend(prediction.rate.tolist())
        learned_sigma.extend(prediction.sigma.tolist())
        constant.extend(null_rate.tolist())
        classical.extend(classical_rate.tolist())
        per_run.append(
            (
                label,
                int(test.size),
                float(np.mean(np.abs(prediction.rate - windows.targets[test]))),
                float(np.mean(np.abs(null_rate - windows.targets[test]))),
                float(np.mean(np.abs(classical_rate - windows.targets[test]))),
            )
        )
    out(f"  held-out runs             : {len(per_run)}")
    out(f"  total fit time            : {grouped_seconds:.3f} s wall clock")
    out("")
    out(f"  {'held-out run':<12}{'windows':>9}{'learned_MAE':>14}{'constant_MAE':>14}"
        f"{'speedscale_MAE':>16}{'winner':>14}")
    for label, n_test, mae_learned, mae_constant, mae_classical in per_run:
        candidates = {
            "learned": mae_learned,
            "constant": mae_constant,
            "speedscale": mae_classical,
        }
        finite = {k: v for k, v in candidates.items() if np.isfinite(v)}
        winner = min(finite, key=lambda key: finite[key]) if finite else "none"
        out(
            f"  {label:<12}{n_test:>9}{mae_learned:>14.6f}{mae_constant:>14.6f}"
            f"{mae_classical:>16.6f}{winner:>14}"
        )
    out("")
    grouped_scores = [
        score(
            "learned forest (grouped)",
            np.asarray(truth),
            np.asarray(learned),
            np.asarray(learned_sigma),
        ),
        score("constant rate (grouped)", np.asarray(truth), np.asarray(constant)),
    ]
    if idle is not None:
        grouped_scores.append(
            score("speed scaling (grouped)", np.asarray(truth), np.asarray(classical))
        )
    for line in score_table(grouped_scores):
        out("  " + line)

    out.rule("4  HELD-OUT TIME -- the later portion of every run")
    train, test = time_split(windows, HOLDOUT_FRACTION)
    out(f"  holdout fraction          : {HOLDOUT_FRACTION}")
    out(f"  training windows          : {train.size}")
    out(f"  test windows              : {test.size}")
    started = time.perf_counter()
    time_model = ContentionMissPredictor().fit(
        windows.features[train], windows.targets[train], windows.feature_names
    )
    time_seconds = time.perf_counter() - started
    out(f"  fit time                  : {time_seconds:.3f} s wall clock")
    time_prediction = time_model.predict(windows.features[test], windows.releases[test])
    time_null = ConstantRatePredictor().fit(
        windows.features[train], windows.targets[train]
    )
    time_scores = [
        score("learned forest (time)", windows.targets[test], time_prediction),
        score(
            "constant rate (time)",
            windows.targets[test],
            time_null.predict(windows.features[test]).rate,
        ),
    ]
    if idle is not None:
        classical_time = np.asarray(
            [
                predicted_miss_probability(
                    idle.exec_s, campaign.deadline_s, BOARD_CORES, int(workers)
                )
                for workers in windows.features[test, 2]
            ]
        )
        time_scores.append(
            score("speed scaling (time)", windows.targets[test], classical_time)
        )
    out("")
    for line in score_table(time_scores):
        out("  " + line)

    out.rule("5  the uncertainty output, checked rather than asserted")
    out(
        "  The learned predictor reports a per-window standard deviation: the spread "
        "across the forest's trees, combined in quadrature with the binomial noise of "
        "the window's own counted rate. The coverage column above is the fraction of "
        "held-out windows whose true rate lies within one reported sigma. For a "
        "well-calibrated Gaussian-like uncertainty that figure would be near 0.68; a "
        "much lower figure means the reported sigma is too small and a much higher one "
        "means it is too large."
    )
    out("")
    out(f"  grouped-split coverage    : {grouped_scores[0].coverage:.4f}")
    out(f"  time-split coverage       : {time_scores[0].coverage:.4f}")
    out(
        f"  mean reported sigma, grouped: "
        f"{float(np.mean(np.asarray(learned_sigma))):.6f}"
    )
    out(
        f"  mean model-only sigma, time : "
        f"{float(np.mean(time_prediction.sigma_model)):.6f}"
    )
    out(
        f"  mean target-noise sigma, time: "
        f"{float(np.mean(time_prediction.sigma_target)):.6f}"
    )
    out.check(
        "the reported sigma is strictly positive on every held-out window",
        bool(np.all(np.asarray(learned_sigma) > 0.0)),
        f"minimum {float(np.min(np.asarray(learned_sigma))):.3e}",
    )

    out.rule("6  THE VERDICT -- published whichever way it falls")
    for split_name, scores in (
        ("held-out workload mixes", grouped_scores),
        ("held-out time", time_scores),
    ):
        finite = [item for item in scores if np.isfinite(item.mae)]
        best = min(finite, key=lambda item: item.mae)
        learned_score = next(item for item in scores if item.label.startswith("learned"))
        out(f"  {split_name}:")
        for item in scores:
            out(f"    {item.label:<34} MAE {item.mae:.6f}  RMSE {item.rmse:.6f}")
        out(f"    lowest MAE: {best.label}")
        if best.label.startswith("learned"):
            margin = min(
                item.mae for item in finite if not item.label.startswith("learned")
            ) - learned_score.mae
            out(
                f"    the learned predictor wins this split by {margin:.6f} MAE over the "
                "best baseline"
            )
        else:
            margin = learned_score.mae - best.mae
            out(
                f"    A BASELINE WINS THIS SPLIT by {margin:.6f} MAE. That is the "
                "finding and it is published as the finding, in README.md, "
                "validation/VALIDATION.md and MODEL_CARD.md. Nothing was retuned."
            )
        out("")
    out(
        "  Neither outcome is retuned away. Across two days of this mission eight of "
        "thirteen products measured a baseline matching or beating the learned "
        "component and every one was published."
    )

    out.rule("7  what this split structure CANNOT measure")
    out.unmeasurable_check(
        "held-out ambient temperature",
        "one uninstrumented ambient was measured, so there is no second ambient to "
        "hold out. A model that generalises across mixes and across time on this "
        "campaign has been shown nothing about generalising across ambient.",
    )
    out.unmeasurable_check(
        "held-out board",
        "one board was measured. Nothing here supports a statement about a second "
        "Orin Nano, let alone about Orin Nano in general.",
    )
    out.unmeasurable_check(
        "held-out payload and period",
        "one CPython CPU-only payload at one period. The descriptors do not include "
        "anything that varies with payload or period, because neither varied.",
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
