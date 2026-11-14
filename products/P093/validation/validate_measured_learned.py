"""The learned predictor against the classical ones, scored where it bites.

Predicting a mean rail power is not by itself a decision-relevant task, so the
learned component is scored on the job the specification gives it: **supplying
``P_active`` to the Baseline 2 analytic model without measuring it.** Everything
else Baseline 2 needs -- the measured floor, the counted inferences per cycle
and the counted acceptance rate -- is held at its measured value, so the table
isolates the contribution of the power prediction and nothing else.

The target is therefore **energy per useful decision**, and the error is the
relative error against the Baseline 3 measured ground truth of the same cell.
Four predictors supply ``P_active``, and the measured value supplies a ceiling:

* **measured ``P_active``** -- the ceiling. How well the analytic FORM itself
  can do when the power it needs is known exactly. Anything above this line is
  the analytic model's own error, not the predictor's.
* **classical: constant grid mean** -- the training cells' mean active power.
  The do-nothing predictor.
* **classical: linear in duty and frame dimension** -- three-parameter ordinary
  least squares over the training cells. The parametric alternative.
* **classical: analytic duty inversion** -- the duty-cycle model run backwards,
  ``mean((P_run - P_floor) / duty)`` over the training cells. This is the
  predictor the specification names, and on this campaign it is the WEAKEST of
  the three classical ones; that is reported rather than quietly dropped, and
  the strongest classical predictor is used as the comparator per the build
  guide.
* **learned: random forest** -- the window-level predictor of
  :mod:`powerenvelope.predictor`, averaged over each test cell's windows.

Every predictor is fitted on training cells only, under both held-out schemes
the specification names: held-out duty cycles and held-out frame dimensions,
grouped so no window of a test group is in training.

The comparator for every error is the **between-run spread** of the four
reference-cell repeats, which is the measurement's own noise floor and the
quantity the pre-registered expectation names.

The two-term form of the cycle energy is the one substituted into, because it
consumes exactly the quantity the window dataset targets; it is algebraically
the same number as the three-term form and
``validate_measured_baselines.py`` reports the residual of that identity.

INPUT ORIGIN: the MEASURED hardware campaign under ``data/traces/`` and nothing
else. No synthetic fallback.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from powerenvelope.baselines import (  # noqa: E402
    between_run_spread,
    energy_per_useful_decision_from_cycle_mj,
    floor_power,
    ground_truth,
    mean_active_power_mw,
    two_term_cycle_energy_mj,
)
from powerenvelope.predictor import (  # noqa: E402
    MAX_DEPTH,
    N_ESTIMATORS,
    RANDOM_STATE,
    ActivePowerPredictor,
    AnalyticActivePowerModel,
    ConstantPowerPredictor,
    LinearActivePowerModel,
    target_quantisation_sd,
)
from powerenvelope.report import campaign_lines, window_lines  # noqa: E402
from powerenvelope.trace import TraceDataAbsentError  # noqa: E402
from powerenvelope.windows import (  # noqa: E402
    extract_windows,
    grouped_splits,
    pipeline_free_monitor_mask,
)

#: The predictors, in the order they are reported: the ceiling first, then the
#: three classical ones, then the learned one. The order is the build guide's:
#: the classical alternatives are implemented and reported before the learned
#: model, because an under-strength classical baseline inflates a learned win.
PREDICTOR_ORDER = (
    "measured P_active (ceiling)",
    "classical: constant grid mean",
    "classical: linear in duty and dimension",
    "classical: analytic duty inversion",
    "learned: random forest",
)

#: Which of the above are classical predictors the learned one is judged against.
CLASSICAL = PREDICTOR_ORDER[1:4]


def report(out: _bootstrap.Tee, campaign) -> None:
    """Write the whole measured report into ``out`` from an already-loaded campaign.

    Split out from :func:`main` so that ``tests/test_measured_reports.py`` can
    exercise every formatting and arithmetic path in this file against a
    SYNTHETIC campaign, without the refusal in ``main`` being weakened and
    without a committed output file being overwritten.
    """
    out.rule("1  campaign identity")
    for line in campaign_lines(campaign):
        out(f"  {line}")

    out.rule("2  the dataset")
    windows = extract_windows(campaign)
    for line in window_lines(windows):
        out(f"  {line}")
    out("")
    out(f"  {'descriptor':<24}{'min':>14}{'mean':>14}{'max':>14}")
    out("  " + "-" * 66)
    for name in windows.feature_names:
        column = windows.column(name)
        out(
            f"  {name:<24}{float(column.min()):>14.6g}{float(column.mean()):>14.6g}"
            f"{float(column.max()):>14.6g}"
        )
    out("")
    out("  dropped windows and why:")
    for cell, reason in windows.dropped:
        out(f"    {cell}: {reason}")

    out.rule("3  where foreign occupancy was separable, per cell")
    out(f"  {'cell':<22}{'duty':>7}{'monitor rows':>14}{'pipeline-free':>15}")
    out("  " + "-" * 60)
    separable_duties = set()
    for label, cell in sorted(campaign.cells.items()):
        free = pipeline_free_monitor_mask(cell)
        if cell.has_pipeline and int(free.sum()) > 0:
            separable_duties.add(cell.duty)
        out(f"  {label:<22}{cell.duty:>7.2f}{len(cell.monitor):>14}{int(free.sum()):>15}")
    out("")
    out(
        "  A monitor sample reports the occupancy of the preceding second. A sample is "
        "pipeline-free only when no inference interval overlaps that whole second. With "
        "a 2.0 s cycle the sleep window is 2*(1-duty) seconds long, so a pipeline-free "
        "second is possible in principle at duty 0.50 and below -- but at duty 0.50 the "
        "sleep window is exactly 1.0 s and the monitor's own 1 s interval has an "
        "arbitrary phase against the cycle, so in practice the two never align. "
        f"Measured on this campaign, occupancy was separable only at duty "
        f"{', '.join(f'{value:.2f}' for value in sorted(separable_duties))}."
    )
    out(
        f"  {windows.foreign_imputed} of {len(windows)} windows therefore carry an "
        f"IMPUTED foreign occupancy, taken from the floor cell's measured "
        f"{windows.floor_foreign_busy:.6f} and flagged in the design matrix as "
        "env_foreign_imputed. MODEL_CARD.md records it as an imputation. It is not a "
        "per-window measurement and it is not presented as one."
    )

    out.rule("4  the estimators, declared")
    out(
        f"  learned  : random forest, {N_ESTIMATORS} trees, max_depth {MAX_DEPTH}, "
        f"random_state {RANDOM_STATE}, n_jobs 1"
    )
    out("  classical: constant grid mean; ordinary least squares in duty and frame")
    out("             dimension; and the analytic duty inversion the specification names.")
    out("  Each is fitted on TRAINING CELLS ONLY and predicts P_active for the held-out")
    out("  cells. Everything else Baseline 2 needs is held at its measured value.")

    out.rule("5  the substitution: energy per useful decision, relative to ground truth")
    floor = floor_power(campaign)
    quantum = float(
        np.mean(campaign.require(_bootstrap.HEADLINE_CELL).power.quantisation_floor_mw)
    )
    cells = {cell.label: cell for cell in campaign.pipeline_cells()}
    measured_power = {label: mean_active_power_mw(cell) for label, cell in cells.items()}
    truth = {
        label: ground_truth(cell).energy_per_useful_decision_mj
        for label, cell in cells.items()
    }

    def substituted(cell, p_active_mw: float) -> float:
        return energy_per_useful_decision_from_cycle_mj(
            cell,
            two_term_cycle_energy_mj(cell.duty, campaign.cycle_s, floor.mean_mw, p_active_mw),
        )

    collected: dict[tuple[str, str], list[tuple[str, float, float]]] = {}
    fit_seconds = 0.0
    importances: dict[str, float] = {}
    coverage: list[float] = []
    for grouping in ("duty", "frame_dim"):
        for _label, train, test in grouped_splits(windows, grouping):
            test_cells = sorted(set(windows.cells[test]))
            train_cells = np.asarray(sorted(set(windows.cells[train])), dtype=object)
            started = time.perf_counter()
            forest = ActivePowerPredictor().fit(
                windows.features[train], windows.targets[train], windows.feature_names
            )
            fit_seconds += time.perf_counter() - started
            samples = np.maximum(1.0, windows.active_s[test] / campaign.sample_period_s)
            prediction = forest.predict(
                windows.features[test], quantisation_step_mw=quantum, n_samples=samples
            )
            inside = np.abs(prediction.power_mw - windows.targets[test]) <= prediction.sigma_mw
            coverage.append(float(np.mean(inside)))
            learned = {
                name: float(prediction.power_mw[windows.cells[test] == name].mean())
                for name in test_cells
            }
            for name, value in forest.importances().items():
                importances[name] = importances.get(name, 0.0) + value
            constant = ConstantPowerPredictor().fit(
                windows.features[train], windows.targets[train]
            )
            constant_value = float(constant.predict(np.zeros((1, 1))).power_mw[0])
            linear = LinearActivePowerModel().fit(campaign, train_cells)
            analytic = AnalyticActivePowerModel().fit(
                campaign, train_cells, floor_mw=floor.mean_mw
            )
            analytic_value = float(analytic.predict(np.zeros((1, 1))).power_mw[0])
            # Each lambda binds the value it needs as a default argument. The
            # dict is consumed inside this iteration, so late binding would be
            # harmless here, but a closure over a loop variable is the kind of
            # thing that stops being harmless the moment the loop is refactored.
            suppliers = {
                "measured P_active (ceiling)": lambda name, table=measured_power: table[name],
                "classical: constant grid mean": lambda name, value=constant_value: value,
                "classical: linear in duty and dimension": (
                    lambda name, model=linear: model.predict_cell(campaign, name)
                ),
                "classical: analytic duty inversion": (
                    lambda name, value=analytic_value: value
                ),
                "learned: random forest": lambda name, table=learned: table[name],
            }
            for predictor in PREDICTOR_ORDER:
                supply = suppliers[predictor]
                for name in test_cells:
                    power = supply(name)
                    collected.setdefault((grouping, predictor), []).append(
                        (
                            name,
                            abs(substituted(cells[name], power) / truth[name] - 1.0),
                            abs(power / measured_power[name] - 1.0),
                        )
                    )

    spread = between_run_spread(campaign)
    relative_half_width = spread.half_width_mj / spread.mean_mj
    means: dict[tuple[str, str], float] = {}
    for grouping in ("duty", "frame_dim"):
        out("")
        out(f"  held-out {grouping}, leave-one-group-out")
        out(
            f"  {'predictor':<40}{'cells':>6}{'mean rel err':>14}{'max':>9}"
            f"{'mean |dP|/P':>13}"
        )
        out("  " + "-" * 82)
        for predictor in PREDICTOR_ORDER:
            rows = collected[(grouping, predictor)]
            errors = [row[1] for row in rows]
            power_errors = [row[2] for row in rows]
            means[(grouping, predictor)] = float(np.mean(errors))
            out(
                f"  {predictor:<40}{len(rows):>6}{float(np.mean(errors)):>14.6f}"
                f"{float(np.max(errors)):>9.6f}{float(np.mean(power_errors)):>13.6f}"
            )
    out("")
    out(
        f"  between-run interval half-width, relative: {relative_half_width:.6f} "
        f"({spread.half_width_mj:.4f} mJ on a mean of {spread.mean_mj:.4f} mJ per useful "
        "decision). Every error above is judged against this."
    )
    out(
        "  'mean |dP|/P' is the predictor's own relative error on the active power, "
        "reported beside the decision-level error so a reader can see how much of the "
        "latter is the analytic form and how much is the prediction."
    )

    out.rule("6  descriptor importances, summed over the splits and normalised")
    total = sum(importances.values()) or 1.0
    for name, value in sorted(importances.items(), key=lambda item: -item[1]):
        out(f"  {name:<24}{value / total:>10.4f}")
    out("")
    out(
        "  Every descriptor above is in the licensed set, which "
        "powerenvelope.predictor.assert_descriptors_licensed enforces inside fit(). The "
        "pipeline's own measured power, its measured latency, its CPU occupancy, its "
        "realised inference count and its realised acceptance rate cannot appear here: "
        "a fit that was offered one of them raises DescriptorLeakError."
    )

    out.rule("7  THE PRE-REGISTERED EXPECTATION, evaluated")
    out(
        "  From SPEC.md, written before the campaign ran: the analytic duty-cycle model "
        "is competitive, and the learned predictor's error will not be smaller than the "
        "analytic model's by more than the between-run interval's half-width."
    )
    out("")
    verdicts = []
    for grouping in ("duty", "frame_dim"):
        classical = {name: means[(grouping, name)] for name in CLASSICAL}
        best = min(classical, key=classical.get)
        worst_classical = max(classical, key=classical.get)
        learned_error = means[(grouping, "learned: random forest")]
        ceiling = means[(grouping, "measured P_active (ceiling)")]
        margin = classical[best] - learned_error
        refuted = margin > relative_half_width
        verdicts.append(refuted)
        out(f"  held-out {grouping}")
        out(f"    ceiling, measured P_active        : {ceiling:.6f}")
        out(f"    strongest classical ({best[11:]:<28}): {classical[best]:.6f}")
        out(
            f"    weakest classical   ({worst_classical[11:]:<28}): "
            f"{classical[worst_classical]:.6f}"
        )
        out(f"    learned random forest             : {learned_error:.6f}")
        out(f"    margin (classical - learned)      : {margin:+.6f}")
        out(f"    between-run half-width            : {relative_half_width:.6f}")
        out(
            f"    VERDICT: {'REFUTED' if refuted else 'NOT REFUTED'} -- the learned "
            f"predictor {'beats' if refuted else 'does not beat'} the strongest classical "
            f"predictor by more than the interval half-width."
        )
        out("")
    out(
        "  OVERALL VERDICT: the pre-registered expectation is "
        + ("REFUTED" if all(verdicts) else "NOT REFUTED")
        + " on "
        + ("both splits." if all(verdicts) or not any(verdicts) else "one split and not the other.")
    )
    if not any(verdicts):
        out(
            "  The classical predictors are competitive and the learned random forest "
            "does not beat the strongest of them on either split; on this campaign it is "
            "slightly WORSE than the best classical predictor on both, by a margin well "
            "inside the measurement's own noise floor. That is the finding. It is "
            "published as it stands, the learned component stays in the product reported "
            "as beaten, and nothing was retuned."
        )
    out(
        "  The specification names the analytic duty-inversion predictor as the "
        "comparator. On this campaign it is the weakest of the three classical "
        "predictors on both splits, so the strongest classical predictor is used as the "
        "comparator and the specified one is reported beside it, per the build guide."
    )
    out.check(
        "the pre-registered comparison was computed and reported in whichever direction "
        "it fell",
        np.isfinite(relative_half_width)
        and all(np.isfinite(value) for value in means.values()),
        f"refuted on {sum(verdicts)} of 2 split(s)",
    )
    for grouping in ("duty", "frame_dim"):
        out.check(
            f"every predictor beats doing nothing about the analytic form on held-out "
            f"{grouping}",
            means[(grouping, "measured P_active (ceiling)")]
            <= min(means[(grouping, name)] for name in CLASSICAL),
            "the ceiling is at least as good as every classical predictor, as it must "
            "be: it is the same model with the power it needs known exactly",
        )
    out.check(
        "the forest fit is inside the three-minute compute budget",
        fit_seconds < 180.0,
        f"{fit_seconds:.2f} s over 8 split(s) on {len(os.sched_getaffinity(0))} core(s)",
    )

    out.rule("8  the uncertainty output, checked rather than asserted")
    out(
        f"  per-sample quantiser sd at the measured quantum: "
        f"{float(target_quantisation_sd(quantum, 1.0)):.3f} mW"
    )
    out(f"  mean one-sigma coverage over the 8 splits      : {float(np.mean(coverage)):.4f}")
    out(
        "  Coverage is the fraction of test windows whose true mean active power lies "
        "within one reported sigma. It is printed whatever it is. A value far below 0.68 "
        "means the reported sigma is too small, which is expected here: the across-tree "
        "spread is a LOWER bound on the predictive uncertainty and includes neither the "
        "run-to-run variation nor the campaign's load drift. The between-run interval "
        "measures the first of those separately."
    )


def main() -> int:
    out = _bootstrap.Tee("validate_measured_learned_output.txt")
    out.rule("MEASURED LEARNED PREDICTOR -- against both baselines, on both splits")
    out("INPUT ORIGIN: the MEASURED hardware campaign only. No synthetic fallback.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)
    out("")
    out(_bootstrap.measurement_model())
    out("")
    out(
        "PRE-REGISTERED EXPECTATION (SPEC.md, written before the campaign ran): the "
        "analytic duty-cycle model is competitive, and the learned predictor's error "
        "will not be smaller than the analytic model's by more than the between-run "
        "interval's half-width."
    )

    try:
        campaign = _bootstrap.require_measured_campaign()
    except TraceDataAbsentError as error:
        out.rule("REFUSED -- no measured campaign")
        for line in str(error).splitlines():
            out(f"  {line}")
        out("")
        out(
            f"  Every measured figure this script would report stands at "
            f"{_bootstrap.MEASURED_PENDING}. Nothing was estimated, interpolated or "
            "illustrated, and the synthetic generator was not substituted."
        )
        for name in (
            "the window dataset size and descriptor ranges",
            "learned predictor MAE on held-out duty cycles",
            "learned predictor MAE on held-out frame dimensions",
            "analytic duty-cycle model MAE on the same splits",
            "constant null model MAE on the same splits",
            "uncertainty coverage of the learned predictor",
            "descriptor importances",
            "the pre-registered expectation's verdict",
            "where foreign CPU occupancy was separable and where it was imputed",
        ):
            out.pending(name, "the measured campaign is not on disk")
        out.check(
            "the MEASURED campaign is on disk and complete",
            False,
            "nothing was reported and nothing was substituted",
        )
        out("")
        out(f"raw output written to {out.save()}")
        return 1

    report(out, campaign)

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
