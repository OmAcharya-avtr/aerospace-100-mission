"""Does the forecast survive dropping the descriptors that need the task running?

The specification requires the learned predictor to map **co-resident workload
descriptors an operator could observe WITHOUT running the task** to a miss
probability. The measured feature importances do not respect that requirement:
the dominant descriptor of the full-data fit is ``task_pct``, the measured CPU
occupancy of the task itself, which is only available to someone who is already
running the task. ``load1`` is contaminated the same way, more weakly, because
the run queue it averages includes the task.

So the headline forecasting result in ``validate_measured_learned.py`` is scored
on a descriptor set that is wider than the specification allows. That result is
NOT withdrawn and NOT retuned -- it is published as measured -- and this script
reports, beside it, the same forests and the same two splits on two narrower
descriptor sets:

* ``no_task_pct``  -- every descriptor except ``task_pct``.
* ``operator_only`` -- also without ``load1``, leaving only the declared
  stressor counts and quantities an outside observer can read: foreign
  occupancy, stressor occupancy, core clocks and die temperature.

Nothing here is tuned. Same estimator, same seed, same splits, same scoring. The
only change is which columns the model is allowed to see, and the point of the
script is to report what the restriction costs.

INPUT ORIGIN: the MEASURED hardware campaign in ``data/traces/`` and nothing
else. There is no synthetic fallback.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from deadlinerisk.predictor import (  # noqa: E402
    ConstantRatePredictor,
    ContentionMissPredictor,
    score,
)
from deadlinerisk.trace import TraceDataAbsentError  # noqa: E402
from deadlinerisk.windows import extract_windows, grouped_splits, time_split  # noqa: E402

HOLDOUT_FRACTION = 0.3

#: Descriptor sets, declared here rather than chosen after seeing a score.
SETS: dict[str, tuple[str, ...]] = {
    "full (as scored in validate_measured_learned.py)": (),
    "no_task_pct": ("task_pct",),
    "operator_only": ("task_pct", "load1"),
}


def _subset(windows, drop: tuple[str, ...]):
    keep = [i for i, name in enumerate(windows.feature_names) if name not in drop]
    return windows.features[:, keep], tuple(
        windows.feature_names[i] for i in keep
    )


def main() -> int:
    out = _bootstrap.Tee("validate_descriptor_leak_output.txt")
    out.rule("DESCRIPTOR LEAK -- what the forecast costs when the task's own "
             "occupancy is removed")
    out("INPUT ORIGIN: the MEASURED hardware campaign only. No synthetic fallback.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)
    out("")

    try:
        campaign = _bootstrap.require_measured_campaign()
    except TraceDataAbsentError as error:
        out.rule("REFUSED -- no measured campaign")
        for line in str(error).splitlines():
            out(f"  {line}")
        out.check("the MEASURED campaign is on disk and complete", False,
                  "nothing was reported and nothing was substituted")
        out(f"raw output written to {out.save()}")
        return 1

    windows = extract_windows(campaign)
    out(f"  windows                   : {len(windows.targets)}")
    out(f"  descriptors available     : {', '.join(windows.feature_names)}")
    out("")

    out.rule("1  held-out workload mixes, grouped by run")
    out(f"  {'descriptor set':<48}{'MAE':>12}{'RMSE':>12}{'bias':>12}")
    grouped: dict[str, float] = {}
    for label, drop in SETS.items():
        features, _names = _subset(windows, drop)
        predictions = np.empty(len(windows.targets), dtype=np.float64)
        for _run, train, test in grouped_splits(windows):
            model = ContentionMissPredictor().fit(features[train], windows.targets[train])
            predictions[test] = model.predict(features[test]).rate
        result = score(label, windows.targets, predictions)
        grouped[label] = result.mae
        out(f"  {label:<48}{result.mae:>12.6f}{result.rmse:>12.6f}{result.bias:>+12.6f}")
    constant = np.empty(len(windows.targets), dtype=np.float64)
    for _run, train, test in grouped_splits(windows):
        model = ConstantRatePredictor().fit(
            windows.features[train], windows.targets[train]
        )
        constant[test] = model.predict(windows.features[test]).rate
    constant_grouped = score('constant rate', windows.targets, constant)
    out(f"  {'constant rate (for reference)':<48}{constant_grouped.mae:>12.6f}"
        f"{constant_grouped.rmse:>12.6f}{constant_grouped.bias:>+12.6f}")
    out("")

    out.rule("2  held-out time, the later portion of every run")
    train, test = time_split(windows, HOLDOUT_FRACTION)
    out(f"  {'descriptor set':<48}{'MAE':>12}{'RMSE':>12}{'bias':>12}")
    timed: dict[str, float] = {}
    for label, drop in SETS.items():
        features, _names = _subset(windows, drop)
        model = ContentionMissPredictor().fit(features[train], windows.targets[train])
        result = score(label, windows.targets[test], model.predict(features[test]))
        timed[label] = result.mae
        out(f"  {label:<48}{result.mae:>12.6f}{result.rmse:>12.6f}{result.bias:>+12.6f}")
    out("")

    out.rule("3  what the restriction costs")
    base = next(iter(SETS))
    for label in SETS:
        if label == base:
            continue
        out(f"  {label}: grouped MAE {grouped[label]:.6f} against "
            f"{grouped[base]:.6f} full "
            f"({grouped[label] / grouped[base]:.4f}x), "
            f"time MAE {timed[label]:.6f} against {timed[base]:.6f} full "
            f"({timed[label] / timed[base]:.4f}x)")
    out("")
    out("  The restricted sets are the ones the specification actually licenses. "
        "Where a restricted set is worse, the full-set figure is an upper bound on "
        "what an operator could achieve without running the task, and the honest "
        "forecasting claim is the restricted one.")
    out("")

    out.check(
        "every descriptor set produced a finite score on both splits",
        all(np.isfinite(v) for v in (*grouped.values(), *timed.values())),
        f"{len(SETS)} sets x 2 splits",
    )
    out.check(
        "the operator_only set excludes both task-dependent descriptors",
        set(SETS["operator_only"]) == {"task_pct", "load1"},
        "task_pct and load1 removed",
    )
    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
