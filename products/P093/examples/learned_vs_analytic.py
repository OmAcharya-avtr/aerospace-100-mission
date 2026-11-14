"""The learned predictor against the classical ones, scored where it bites.

Run it:

    python examples/learned_vs_analytic.py

Predicting a mean rail power is not by itself a decision-relevant task, so this
scores every predictor on the job the specification gives the learned one:
supplying ``P_active`` to the Baseline 2 analytic model without measuring it.
Everything else Baseline 2 needs is held at its measured value, so the table
isolates the contribution of the power prediction. The error reported is the
relative error of the resulting energy per useful decision against the
Baseline 3 measured ground truth, judged against the between-run spread of the
four reference-cell repeats. It writes
``screenshots/learned_vs_analytic.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _common  # noqa: E402

ROOT = _common.bootstrap()

from powerenvelope.baselines import (  # noqa: E402
    between_run_spread,
    energy_per_useful_decision_from_cycle_mj,
    floor_power,
    ground_truth,
    mean_active_power_mw,
    two_term_cycle_energy_mj,
)
from powerenvelope.predictor import (  # noqa: E402
    ActivePowerPredictor,
    AnalyticActivePowerModel,
    ConstantPowerPredictor,
    LinearActivePowerModel,
)
from powerenvelope.report import window_lines  # noqa: E402
from powerenvelope.windows import extract_windows, grouped_splits  # noqa: E402

NAME = "learned_vs_analytic"

ORDER = (
    "measured (ceiling)",
    "classical: constant",
    "classical: linear",
    "classical: duty inversion",
    "learned: forest",
)
CLASSICAL = ORDER[1:4]


def main() -> int:
    campaign, is_measured = _common.campaign()
    out = _common.Output(NAME, is_measured)
    out(campaign.provenance.banner())
    out("")

    windows = extract_windows(campaign)
    for line in window_lines(windows):
        out(f"  {line}")
    out("")

    floor = floor_power(campaign)
    cells = {cell.label: cell for cell in campaign.pipeline_cells()}
    measured_power = {label: mean_active_power_mw(cell) for label, cell in cells.items()}
    truth = {
        label: ground_truth(cell).energy_per_useful_decision_mj
        for label, cell in cells.items()
    }

    def substituted(cell, power_mw: float) -> float:
        return energy_per_useful_decision_from_cycle_mj(
            cell,
            two_term_cycle_energy_mj(cell.duty, campaign.cycle_s, floor.mean_mw, power_mw),
        )

    collected: dict[tuple[str, str], list[float]] = {}
    for grouping in ("duty", "frame_dim"):
        for _label, train, test in grouped_splits(windows, grouping):
            test_cells = sorted(set(windows.cells[test]))
            train_cells = np.asarray(sorted(set(windows.cells[train])), dtype=object)
            forest = ActivePowerPredictor().fit(
                windows.features[train], windows.targets[train], windows.feature_names
            )
            predicted = forest.predict(windows.features[test]).power_mw
            learned = {
                name: float(predicted[windows.cells[test] == name].mean())
                for name in test_cells
            }
            constant = float(
                ConstantPowerPredictor()
                .fit(windows.features[train], windows.targets[train])
                .predict(np.zeros((1, 1)))
                .power_mw[0]
            )
            linear = LinearActivePowerModel().fit(campaign, train_cells)
            inversion = float(
                AnalyticActivePowerModel()
                .fit(campaign, train_cells, floor_mw=floor.mean_mw)
                .predict(np.zeros((1, 1)))
                .power_mw[0]
            )
            suppliers = {
                "measured (ceiling)": lambda name, t=measured_power: t[name],
                "classical: constant": lambda name, v=constant: v,
                "classical: linear": lambda name, m=linear: m.predict_cell(campaign, name),
                "classical: duty inversion": lambda name, v=inversion: v,
                "learned: forest": lambda name, t=learned: t[name],
            }
            for predictor in ORDER:
                supply = suppliers[predictor]
                for name in test_cells:
                    collected.setdefault((grouping, predictor), []).append(
                        abs(substituted(cells[name], supply(name)) / truth[name] - 1.0)
                    )

    spread = between_run_spread(campaign)
    relative_half_width = spread.half_width_mj / spread.mean_mj
    means = {key: float(np.mean(values)) for key, values in collected.items()}

    out("ENERGY PER USEFUL DECISION, relative error against measured ground truth")
    out("Only P_active is predicted; the floor, the inference count and the acceptance")
    out("rate are held at their measured values.")
    out("")
    for grouping in ("duty", "frame_dim"):
        out(f"  held-out {grouping}")
        out(f"    {'predictor':<30}{'mean rel err':>14}{'max':>10}")
        out("    " + "-" * 54)
        for predictor in ORDER:
            values = collected[(grouping, predictor)]
            out(
                f"    {predictor:<30}{float(np.mean(values)):>14.6f}"
                f"{float(np.max(values)):>10.6f}"
            )
        out("")

    out(f"between-run interval half-width, relative: {relative_half_width:.6f}")
    out("")
    out("THE PRE-REGISTERED COMPARISON (SPEC.md, written before the campaign ran)")
    for grouping in ("duty", "frame_dim"):
        classical = {name: means[(grouping, name)] for name in CLASSICAL}
        best = min(classical, key=classical.get)
        margin = classical[best] - means[(grouping, "learned: forest")]
        out(
            f"  held-out {grouping}: strongest classical is {best} at "
            f"{classical[best]:.6f}; learned {means[(grouping, 'learned: forest')]:.6f}; "
            f"margin {margin:+.6f} against a half-width of {relative_half_width:.6f}"
        )
        out(
            "    VERDICT: "
            + (
                "REFUTED, the learned predictor wins by more than the interval."
                if margin > relative_half_width
                else "NOT REFUTED, the classical predictor is competitive."
            )
        )
    out("")
    out("Whichever way it falls, that is the finding. The learned component stays in the")
    out("product and is reported as beaten where it is beaten.")

    handle, save = _common.figure(
        NAME,
        is_measured,
        "Supplying P_active to the analytic model: learned against classical",
    )
    axes = handle.subplots(1, 2)
    width = 0.35
    positions = np.arange(len(ORDER))
    for offset, grouping in ((-width / 2, "duty"), (width / 2, "frame_dim")):
        axes[0].bar(
            positions + offset,
            [means[(grouping, name)] for name in ORDER],
            width,
            label=f"held-out {grouping}",
        )
    axes[0].axhline(
        relative_half_width,
        color="black",
        linestyle="--",
        linewidth=1,
        label="between-run half-width",
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(ORDER, rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("mean relative error of energy per useful decision")
    axes[0].set_title("lower is better; the ceiling is the analytic form's own error")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    forest = ActivePowerPredictor().fit(
        windows.features, windows.targets, windows.feature_names
    )
    importances = forest.importances()
    names = sorted(importances, key=lambda name: importances[name])
    axes[1].barh(np.arange(len(names)), [importances[name] for name in names])
    axes[1].set_yticks(np.arange(len(names)))
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].set_xlabel("impurity importance, full-data fit")
    axes[1].set_title("licensed descriptors only; the excluded set cannot appear")
    axes[1].grid(alpha=0.3, axis="x")

    written = save()
    out("")
    out(f"figure written to {written}")
    out(f"raw output written to {_common.relative(out.path)}")
    out.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
