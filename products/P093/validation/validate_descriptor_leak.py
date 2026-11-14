"""What the learned predictor's accuracy costs when descriptors are taken away.

INPUT ORIGIN: SYNTHETIC. The measured version of this comparison is in
``validate_measured_learned.py``, which refuses until the campaign's traces
exist. The SHAPE of the check is the same in both places, and this script exists
so that the shape is exercised and reviewable before the data lands.

Why the check exists. The specification licenses exactly the descriptors an
operator could state from the configuration and read from the platform's own
environment without running the pipeline, and
:func:`powerenvelope.predictor.assert_descriptors_licensed` enforces that in
code. But two of the licensed descriptors are weaker than the licence suggests
and the specification says so rather than hiding it:

* ``env_load1`` is the 1-minute load average, which includes the pipeline's own
  run-queue contribution. The specification names it as licensed. It is used,
  and the contamination is real.
* ``env_foreign_busy`` is ``/proc/stat`` occupancy restricted to monitor seconds
  with no inference in them. With a 2.0 s cycle no such second exists at duty
  0.75 or 1.00, so for those cells the value is IMPUTED from the floor cell and
  flagged by ``env_foreign_imputed``.

So this script refits the same estimator, with the same seed, on the same two
grouped splits, over four nested descriptor sets, and reports what each
restriction costs. Nothing is tuned. The only change is which columns the model
is allowed to see.

The descriptor sets, declared here rather than chosen after seeing a score:

* ``licensed``        -- all ten, as scored in ``validate_measured_learned.py``.
* ``no_load1``        -- without the contaminated load average.
* ``no_foreign``      -- also without the occupancy descriptor and its flag.
* ``configuration``   -- only duty, cycle period, frame dimension, tau and the
                         core count: nothing read from the platform at all.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from powerenvelope.baselines import floor_power  # noqa: E402
from powerenvelope.predictor import (  # noqa: E402
    ActivePowerPredictor,
    AnalyticActivePowerModel,
    ConstantPowerPredictor,
    score,
)
from powerenvelope.report import score_table, window_lines  # noqa: E402
from powerenvelope.windows import extract_windows, grouped_splits  # noqa: E402

#: Descriptor sets as the columns DROPPED from the licensed set.
SETS: dict[str, tuple[str, ...]] = {
    "licensed (all ten)": (),
    "no_load1": ("env_load1",),
    "no_foreign": ("env_load1", "env_foreign_busy", "env_foreign_imputed"),
    "configuration only": (
        "env_load1",
        "env_foreign_busy",
        "env_foreign_imputed",
        "env_tj_c",
        "env_clk_mean_khz",
    ),
}


def main() -> int:
    out = _bootstrap.Tee("validate_descriptor_leak_output.txt")
    out.rule("DESCRIPTOR RESTRICTION -- what each narrower set costs")
    out("INPUT ORIGIN: SYNTHETIC. The measured version is validate_measured_learned.py.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    campaign = _bootstrap.synthetic_campaign()
    windows = extract_windows(campaign)
    floor = floor_power(campaign)

    out.rule("1  the dataset")
    for line in window_lines(windows):
        out(f"  {line}")

    out.rule("2  the licence check is enforced in code, not by convention")
    out(
        "  powerenvelope.predictor.assert_descriptors_licensed is called by "
        "ActivePowerPredictor.fit, so an unlicensed column cannot reach the estimator. "
        "validate_negative_controls.py asserts that it fires on each excluded class."
    )

    out.rule("3  the same estimator, the same splits, four descriptor sets")
    summary: dict[str, dict[str, float]] = {}
    for label, dropped in SETS.items():
        matrix, names = windows.without(dropped)
        rows = []
        for grouping in ("duty", "frame_dim"):
            for split_label, train, test in grouped_splits(windows, grouping):
                model = ActivePowerPredictor().fit(
                    matrix[train], windows.targets[train], names
                )
                rows.append(
                    score(
                        f"{grouping}: {split_label}",
                        windows.targets[test],
                        model.predict(matrix[test]),
                    )
                )
        summary[label] = {
            "mae": float(np.mean([row.mae_mw for row in rows])),
            "rmse": float(np.mean([row.rmse_mw for row in rows])),
            "n_descriptors": float(len(names)),
            "coverage": float(np.nanmean([row.coverage for row in rows])),
        }
        out("")
        out(f"  descriptor set: {label}  ({len(names)} column(s))")
        out(f"  columns: {', '.join(names)}")
        for line in score_table(rows):
            out(line)

    out.rule("4  the baselines on the same splits, for scale")
    baseline_rows = []
    for grouping in ("duty", "frame_dim"):
        for split_label, train, test in grouped_splits(windows, grouping):
            analytic = AnalyticActivePowerModel().fit(
                campaign, windows.cells[train], floor_mw=floor.mean_mw
            )
            constant = ConstantPowerPredictor().fit(
                windows.features[train], windows.targets[train]
            )
            baseline_rows.append(
                score(
                    f"analytic duty-cycle / {split_label}",
                    windows.targets[test],
                    analytic.predict(windows.features[test]),
                )
            )
            baseline_rows.append(
                score(
                    f"constant (null) / {split_label}",
                    windows.targets[test],
                    constant.predict(windows.features[test]),
                )
            )
    for line in score_table(baseline_rows):
        out(line)
    analytic_mae = float(
        np.mean([row.mae_mw for row in baseline_rows if row.label.startswith("analytic")])
    )
    constant_mae = float(
        np.mean([row.mae_mw for row in baseline_rows if row.label.startswith("constant")])
    )

    out.rule("5  the cost of each restriction")
    reference = summary["licensed (all ten)"]["mae"]
    out(f"  {'descriptor set':<24}{'cols':>6}{'mean MAE mW':>14}{'vs licensed':>14}{'cover':>8}")
    out("  " + "-" * 64)
    for label, values in summary.items():
        delta = values["mae"] - reference
        out(
            f"  {label:<24}{values['n_descriptors']:>6.0f}{values['mae']:>14.4f}"
            f"{delta:>+14.4f}{values['coverage']:>8.3f}"
        )
    out("")
    out(f"  analytic duty-cycle model, mean MAE : {analytic_mae:.4f} mW")
    out(f"  constant null model, mean MAE       : {constant_mae:.4f} mW")

    out.rule("6  what is asserted")
    out.check(
        "every descriptor set scores on both groupings without error",
        len(summary) == len(SETS),
        f"{len(SETS)} set(s) x 8 split(s)",
    )
    out.check(
        "removing descriptors never improves the mean error by more than noise",
        all(
            values["mae"] >= reference - 0.05 * max(reference, 1e-9)
            for values in summary.values()
        ),
        "a narrower set that beat the wider one would mean the wider one is "
        "overfitting the extra columns, which would itself be the finding",
    )
    out(
        "  NOTE: the number that matters is the MEASURED one. On the synthetic campaign "
        "the environment descriptors are deterministic functions of the duty, so the "
        "restriction cost measured here is a property of the generator and not of any "
        "board. The measured comparison is validate_measured_learned.py section 5, "
        "which scores the learned predictor and three classical ones on the "
        "decision-level target rather than on the power alone."
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
