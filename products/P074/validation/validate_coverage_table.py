"""The interval coverage table, stratified, including where it is miscalibrated.

Marginal coverage is the easy number and it hides the interesting failure. The
rows of this dataset are grouped -- twenty-eight epoch ages of the same object
share its features and its physics -- so a split-conformal calibration drawn by
object is exchangeable at the level of objects and not at the level of rows.
Conformal prediction then delivers about its nominal coverage *marginally*
while coverage conditional on altitude or on epoch age does not, and that is
what the tables below show.

Every cell carries a Clopper-Pearson interval and a verdict. A cell is marked
``under`` or ``over`` when the nominal level falls outside that interval, so
the data are inconsistent with the claim; ``thin`` means the cell was too
small to be inconsistent with anything, which is an absence of evidence and
not a pass.

The row-level interval is too narrow for grouped data, which is stated at every
table. The object-level coverage in ``validate_predictors_output.txt`` is the
conservative counterpart.
"""

from __future__ import annotations

import sys

from _bootstrap import Tee, bootstrap, environment_lines, is_fast, load_dataset

bootstrap()

from orbitdrift import run_benchmark  # noqa: E402
from orbitdrift.benchmark import run_extrapolation_benchmark  # noqa: E402

NOMINAL = 0.90
_ESTIMATORS = 40 if is_fast() else None
"""Boosting stages for the learned model: reduced in probe mode only."""
HEADLINE = (
    "rule_of_thumb",
    "analytic_linear",
    "analytic_power_altitude",
    "analytic_power_altitude_conformal",
    "learned_gbr_quantile",
    "learned_gbr_conformal",
)
EXTRAPOLATION_HEADLINE = (
    "own_curve_power",
    "own_curve_power_conformal",
    "own_curve_linear_conformal",
    "own_curve_drag_exponential_conformal",
    "learned_with_curve_features",
    "learned_with_curve_features_conformal",
)


def _cells(out, cells, title: str) -> int:
    out("")
    out(f"  {title}")
    out("    stratum                 n   coverage   Clopper-Pearson 95 pct   "
        "median log width   verdict")
    flagged = 0
    for cell in cells:
        if cell.verdict in ("under", "over"):
            flagged += 1
        out(
            f"    {cell.stratum:20s} {cell.n:5d}   {cell.coverage:8.4f}   "
            f"[{cell.coverage_ci[0]:.4f}, {cell.coverage_ci[1]:.4f}]           "
            f"{cell.median_log_width:14.4f}   {cell.verdict}"
        )
    return flagged


def main() -> int:
    out = Tee("validate_coverage_table_output.txt")
    out.rule("orbitdrift -- measured interval coverage, stratified")
    for line in environment_lines():
        out(line)

    table = load_dataset()
    out("")
    out(f"  dataset       : {table.detail}")
    out(f"  nominal level : {NOMINAL}")
    out("")
    out("  The rows are GROUPED: 28 epoch ages of each object share its features")
    out("  and its physics, so the effective sample size of a cell is closer to the")
    out("  number of objects in it than to the number of rows. The Clopper-Pearson")
    out("  interval below is computed on the row count and is therefore TOO NARROW.")
    out("  Treat a cell flagged 'under' or 'over' as evidence; treat a cell flagged")
    out("  'ok' as weaker evidence than its interval suggests.")

    cold = run_benchmark(
        table, nominal_coverage=NOMINAL, learned_estimators=_ESTIMATORS
    )
    extrapolation = run_extrapolation_benchmark(
        table, nominal_coverage=NOMINAL, learned_estimators=_ESTIMATORS
    )

    out.rule("task 1: cold prediction -- coverage by perigee altitude")
    flagged_altitude = 0
    for name in HEADLINE:
        flagged_altitude += _cells(out, cold.coverage_by_altitude[name], name)

    out.rule("task 1: cold prediction -- coverage by epoch age")
    flagged_age = 0
    for name in HEADLINE:
        flagged_age += _cells(out, cold.coverage_by_age[name], name)

    out.rule("task 2: extrapolation -- coverage by perigee altitude")
    flagged_extrapolation_altitude = 0
    for name in EXTRAPOLATION_HEADLINE:
        flagged_extrapolation_altitude += _cells(
            out, extrapolation.coverage_by_altitude[name], name
        )

    out.rule("task 2: extrapolation -- coverage by epoch age")
    flagged_extrapolation_age = 0
    for name in EXTRAPOLATION_HEADLINE:
        flagged_extrapolation_age += _cells(out, extrapolation.coverage_by_age[name], name)

    out.rule("where it is miscalibrated: the summary")
    out("  cells inconsistent with the nominal level, by table")
    out(f"    task 1, by altitude : {flagged_altitude}")
    out(f"    task 1, by age      : {flagged_age}")
    out(f"    task 2, by altitude : {flagged_extrapolation_altitude}")
    out(f"    task 2, by age      : {flagged_extrapolation_age}")

    out("")
    out("  The best-calibrated predictor on each task, cell by cell:")
    for label, result, names in (
        ("cold prediction", cold, HEADLINE),
        ("extrapolation", extrapolation, EXTRAPOLATION_HEADLINE),
    ):
        scores = []
        for name in names:
            cells = list(result.coverage_by_altitude[name]) + list(result.coverage_by_age[name])
            bad = sum(1 for cell in cells if cell.verdict in ("under", "over"))
            under = sum(1 for cell in cells if cell.verdict == "under")
            scores.append((bad, under, name, len(cells)))
        scores.sort()
        out(f"    {label}")
        for bad, under, name, total in scores:
            out(
                f"      {name:40s} {bad:2d} of {total:2d} cells inconsistent "
                f"({under} of them under-covering)"
            )

    out.rule("the conditional miscalibration conformal prediction does not promise")
    out("  Split conformal prediction gives MARGINAL coverage at or above the")
    out("  nominal level for exchangeable data. It promises nothing about coverage")
    out("  conditional on a feature, and the tables above show it failing to")
    out("  deliver conditionally while delivering marginally. That is not a defect")
    out("  in the method; it is the method's stated scope, and a product that")
    out("  reported only the marginal number would be hiding it.")
    out("")
    for label, result, name in (
        ("cold prediction", cold, "analytic_power_altitude_conformal"),
        ("extrapolation", extrapolation, "own_curve_power_conformal"),
    ):
        row = result.row(name)
        cells = result.coverage_by_altitude[name]
        spread = max(cell.coverage for cell in cells) - min(cell.coverage for cell in cells)
        out(
            f"  {label}: {name} covers {row.metrics.coverage:.4f} marginally, and its "
            f"altitude cells span {min(cell.coverage for cell in cells):.4f} to "
            f"{max(cell.coverage for cell in cells):.4f}, a spread of {spread:.4f}"
        )

    out.rule("checks")
    flagged_total = (
        flagged_altitude
        + flagged_age
        + flagged_extrapolation_altitude
        + flagged_extrapolation_age
    )
    out.check(
        "at least one cell in the tables is flagged, so the flagging works",
        flagged_total > 0,
        f"{flagged_total} cells inconsistent with the nominal level",
    )
    out.check(
        "the rule of thumb is flagged in every altitude cell",
        all(
            cell.verdict == "under" for cell in cold.coverage_by_altitude["rule_of_thumb"]
        ),
        ", ".join(
            f"{cell.stratum}:{cell.verdict}"
            for cell in cold.coverage_by_altitude["rule_of_thumb"]
        ),
    )
    conformal_cells = cold.coverage_by_altitude["analytic_power_altitude_conformal"]
    conformal_spread = max(cell.coverage for cell in conformal_cells) - min(
        cell.coverage for cell in conformal_cells
    )
    if is_fast():
        out("")
        out("  probe mode: the dataset is reduced, so the population-level recorded")
        out("  result below is not evaluated. Only the exit status is being probed.")
    else:
        out.check(
            "RECORDED RESULT: a conformally calibrated predictor is conditionally "
            "miscalibrated across altitude even where it is marginally calibrated",
            conformal_spread > 0.10,
            f"altitude-cell coverage spread {conformal_spread:.4f} against a marginal "
            f"{cold.row('analytic_power_altitude_conformal').metrics.coverage:.4f}",
        )
    out.check(
        "every cell with 20 or more rows carries a verdict other than 'thin'",
        all(
            cell.verdict != "thin"
            for name in HEADLINE
            for cell in cold.coverage_by_altitude[name]
            if cell.n >= 20
        ),
        "",
    )

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())
