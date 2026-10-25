"""The analytic baselines against the learned predictor, on measured coverage and width.

Two tasks, because the question has two forms.

**Cold prediction.** Nothing about the object has been observed; a model must
predict its drift from orbital elements and epoch age alone. Held-out objects,
never held-out rows of the same object.

**Extrapolation.** The operator has the element set and can run the divergence
computation over the first two days in seconds. The question is then how far
the curve goes. The analytic forms are fitted to the object's own curve inside
the horizon and extrapolated beyond it; the learned model is trained on the
beyond-horizon rows of the training objects and is given the object's own
fitted amplitude, exponent and residual spread as three extra features, so it
is not handicapped by being denied information the baseline has.

The analytic baselines are fitted and measured first in both tasks, and
``validate_growth_baseline.py`` has already established that they are as strong
as the physics allows. The ordering is enforced in
:func:`orbitdrift.benchmark.run_benchmark` rather than left to this script.

How the winner is decided, and why it is not the narrowest interval. Narrowing
an interval always improves its score until it stops covering, so width is
only comparable among predictors whose measured coverage is consistent with
the level they claim. Three rankings are therefore reported: over everything,
over the predictors whose Clopper-Pearson interval contains the nominal level,
and over the predictors that do not under-cover at all. The third is the
operational one, because for a staleness decision an interval that covers less
than it claims gives a horizon that is LATE.
"""

from __future__ import annotations

import sys

from _bootstrap import Tee, Timer, bootstrap, environment_lines, is_fast, load_dataset

bootstrap()


from orbitdrift import run_benchmark  # noqa: E402
from orbitdrift.benchmark import (  # noqa: E402
    EXTRA_CURVE_FEATURES,
    calibrated_rows,
    not_under_covering_rows,
    run_extrapolation_benchmark,
)

NOMINAL = 0.90
_ESTIMATORS = 40 if is_fast() else None
"""Boosting stages for the learned model: reduced in probe mode only."""


def _table(out, result) -> None:
    out("")
    out(
        "  predictor                             kind      coverage   "
        "Clopper-Pearson     median   interval   median |log"
    )
    out(
        "                                                             "
        "95 pct interval   log width   score km   error|"
    )
    for row in result.rows:
        metrics = row.metrics
        out(
            f"  {row.name:37s} {row.kind:9s} {metrics.coverage:8.4f}   "
            f"[{metrics.coverage_ci[0]:.4f}, {metrics.coverage_ci[1]:.4f}]   "
            f"{metrics.median_log_width:9.4f}   {metrics.interval_score:8.2f}   "
            f"{row.median_absolute_log_error:.4f}"
        )


def _group_table(out, result) -> None:
    out("")
    out("  Object-level coverage: one Bernoulli trial per object, counted as covered")
    out("  when more than half its rows are. The rows are grouped -- all the ages of")
    out("  one object share its features and its physics -- so the row-level")
    out("  interval above is too narrow and this one is conservative in the other")
    out("  direction. Neither is presented as the answer.")
    out("")
    out("  predictor                             objects   object coverage   interval")
    for row in result.rows:
        out(
            f"  {row.name:37s} {row.n_groups:7d}   {row.group_coverage:15.4f}   "
            f"[{row.group_coverage_ci[0]:.4f}, {row.group_coverage_ci[1]:.4f}]"
        )


def _rankings(out, result, label: str) -> None:
    out("")
    out(f"  {label}")
    for line in result.verdict_line().splitlines():
        out(f"    {line}")
    kept = calibrated_rows(result.rows, result.nominal_coverage)
    safe = not_under_covering_rows(result.rows, result.nominal_coverage)
    out("")
    out(f"    calibrated predictors ({len(kept)}): " + ", ".join(row.name for row in kept))
    out(f"    not under-covering ({len(safe)}): " + ", ".join(row.name for row in safe))


def main() -> int:
    out = Tee("validate_predictors_output.txt")
    out.rule("orbitdrift -- analytic baselines against the learned predictor")
    for line in environment_lines():
        out(line)

    table = load_dataset()
    out("")
    out(f"  dataset        : {table.detail}")
    out(f"  nominal level  : {NOMINAL}")
    out("  split          : by OBJECT, never by row. Splitting by row would put an")
    out("                   object's early ages in training and its late ages in")
    out("                   test, where the features are identical and only the age")
    out("                   column differs, so a model could interpolate its own")
    out("                   training rows.")

    timer_cold = Timer()
    with timer_cold:
        cold = run_benchmark(
            table, nominal_coverage=NOMINAL, learned_estimators=_ESTIMATORS
        )

    out.rule("task 1: cold prediction from orbital elements and epoch age")
    out(f"  training : {cold.n_train_rows} rows over {cold.n_train_objects} objects")
    out(f"  test     : {cold.n_test_rows} rows over {cold.n_test_objects} objects")
    _table(out, cold)
    _group_table(out, cold)
    _rankings(out, cold, "rankings")

    out.rule("task 1: the analytic forms' fitted parameters")
    out("  predictor                             effective exponent   log-resid sd   tau d")
    for row in cold.rows:
        if "effective_exponent" not in row.extra:
            continue
        out(
            f"  {row.name:37s} {row.extra['effective_exponent']:18.4f}   "
            f"{row.extra['log_residual_std']:12.5f}   "
            f"{row.extra['timescale_days']:7.2f}"
        )

    timer_extrapolation = Timer()
    with timer_extrapolation:
        extrapolation = run_extrapolation_benchmark(
            table,
            fit_horizon_days=2.0,
            nominal_coverage=NOMINAL,
            learned_estimators=_ESTIMATORS,
        )

    out.rule("task 2: extrapolation beyond a two-day fitting horizon")
    out(f"  fitting horizon        : {extrapolation.fit_horizon_days} days")
    out(f"  rows inside the horizon: {extrapolation.n_fit_rows}")
    out(f"  rows beyond it         : {extrapolation.n_target_rows}")
    out(f"  measured on            : {extrapolation.n_test_target_rows} rows over "
        f"{extrapolation.n_test_objects} held-out objects")
    out(f"  trained on             : {extrapolation.n_train_objects} objects")
    out(f"  extra learned features : {', '.join(EXTRA_CURVE_FEATURES)}")
    out(f"  best-fitting form inside the horizon, per object: {extrapolation.form_counts}")
    _table(out, extrapolation)
    _group_table(out, extrapolation)
    _rankings(out, extrapolation, "rankings")

    out.rule("what the object's own curve is worth, separately from the learning")
    with_curve = extrapolation.row("learned_with_curve_features")
    without = extrapolation.row("learned_features_only")
    out("  Two learned models, identical except that one sees the object's own")
    out("  inside-horizon fit and the other does not.")
    out("")
    out(f"  with the curve    : median |log error| {with_curve.median_absolute_log_error:.4f}, "
        f"interval score {with_curve.metrics.interval_score:.2f} km")
    out(f"  without the curve : median |log error| {without.median_absolute_log_error:.4f}, "
        f"interval score {without.metrics.interval_score:.2f} km")
    ratio = without.median_absolute_log_error / with_curve.median_absolute_log_error
    out(f"  ratio             : {ratio:.4f}")
    if not is_fast():
        out.check(
            "the object's own curve is worth more than a factor of two in point error",
            ratio > 2.0,
            f"{ratio:.4f}; the information is in the curve, not in the learning",
        )

    out.rule("the central measured fact about nominal intervals")
    out("  An analytic fit's own in-sample residual spread is what a practitioner")
    out("  would naturally use as an interval. Measured on held-out objects beyond")
    out("  the horizon it covers far less than it claims, and that is the reason")
    out("  this product measures coverage instead of quoting it.")
    out("")
    out("  form                nominal interval coverage   conformal coverage")
    for form in ("linear", "quadratic", "power", "drag_exponential"):
        nominal_row = extrapolation.row(f"own_curve_{form}")
        conformal_row = extrapolation.row(f"own_curve_{form}_conformal")
        out(
            f"  {form:16s}    {nominal_row.metrics.coverage:24.4f}   "
            f"{conformal_row.metrics.coverage:18.4f}"
        )
    power_nominal = extrapolation.row("own_curve_power").metrics.coverage
    power_conformal = extrapolation.row("own_curve_power_conformal").metrics.coverage
    if not is_fast():
        out.check(
            "the nominal interval of the own-curve power law under-covers badly",
            power_nominal < 0.60,
            f"{power_nominal:.4f} against a claimed {NOMINAL:.2f}",
        )
        out.check(
            "conformal calibration repairs it",
            power_conformal > power_nominal + 0.25,
            f"{power_conformal:.4f} against {power_nominal:.4f}",
        )

    out.rule("the verdict, in one place")
    for task, result in (("cold prediction", cold), ("extrapolation", extrapolation)):
        kept = calibrated_rows(result.rows, result.nominal_coverage)
        safe = not_under_covering_rows(result.rows, result.nominal_coverage)
        best_point_analytic = min(
            (row for row in result.rows if row.kind == "analytic"),
            key=lambda row: row.median_absolute_log_error,
        )
        best_point_learned = min(
            (row for row in result.rows if row.kind == "learned"),
            key=lambda row: row.median_absolute_log_error,
        )
        out("")
        out(f"  {task}")
        out(
            f"    point prediction : analytic {best_point_analytic.name} "
            f"{best_point_analytic.median_absolute_log_error:.4f} against learned "
            f"{best_point_learned.name} {best_point_learned.median_absolute_log_error:.4f}"
        )
        if kept:
            best = min(kept, key=lambda row: row.metrics.interval_score)
            out(
                f"    calibrated best  : {best.name} ({best.kind}), score "
                f"{best.metrics.interval_score:.2f} km, coverage "
                f"{best.metrics.coverage:.4f}"
            )
        else:
            out("    calibrated best  : none; every interval is miscalibrated")
        if safe:
            best = min(safe, key=lambda row: row.metrics.interval_score)
            out(
                f"    not under-cover  : {best.name} ({best.kind}), score "
                f"{best.metrics.interval_score:.2f} km, coverage "
                f"{best.metrics.coverage:.4f}"
            )
        else:
            out("    not under-cover  : none; every predictor under-covers")

    out.rule("checks")
    out.check(
        "the benchmark fits and measures every analytic form before the learned one",
        [row.kind for row in cold.rows].index("learned")
        == sum(1 for row in cold.rows if row.kind == "analytic"),
        "ordering enforced in orbitdrift.benchmark.run_benchmark",
    )
    out.check(
        "no object appears in both the training and the test split",
        cold.n_train_objects + cold.n_test_objects == table.n_objects,
        f"{cold.n_train_objects} + {cold.n_test_objects} = {table.n_objects}",
    )
    out.check(
        "the rule of thumb a practitioner would use is badly calibrated",
        is_fast() or cold.row("rule_of_thumb").metrics.coverage < 0.75,
        f"coverage {cold.row('rule_of_thumb').metrics.coverage:.4f} at a claimed "
        f"{NOMINAL:.2f}, interval score "
        f"{cold.row('rule_of_thumb').metrics.interval_score:.2f} km",
    )
    analytic_safe = [
        row
        for row in not_under_covering_rows(extrapolation.rows, NOMINAL)
        if row.kind == "analytic"
    ]
    learned_safe = [
        row
        for row in not_under_covering_rows(extrapolation.rows, NOMINAL)
        if row.kind == "learned"
    ]
    out.check(
        "on the extrapolation task at least one ANALYTIC predictor reaches its "
        "nominal coverage",
        bool(analytic_safe),
        ", ".join(f"{row.name} {row.metrics.coverage:.4f}" for row in analytic_safe)
        or "none",
    )
    out("")
    if is_fast():
        out("  probe mode: the dataset is reduced, so the population-level recorded")
        out("  result below is not evaluated. Only the exit status is being probed.")
    else:
        out("  The next line is the specification's predicted outcome, reported as")
        out("  measured. It is a baseline result and is not softened.")
        best_learned = max(
            row.metrics.coverage for row in extrapolation.rows if row.kind == "learned"
        )
        out.check(
            "RECORDED RESULT: on the extrapolation task no LEARNED predictor reaches "
            "its nominal coverage, so the only operationally usable intervals are "
            "analytic",
            not learned_safe,
            ", ".join(f"{row.name} {row.metrics.coverage:.4f}" for row in learned_safe)
            or f"no learned predictor at or above {NOMINAL:.2f}; best learned "
            f"coverage {best_learned:.4f}",
        )

    out.rule("timings (UNSTABLE: wall clock on a shared two-core container)")
    out(f"  cold-prediction benchmark     : {timer_cold.elapsed_s:.1f} s")
    out(f"  extrapolation benchmark       : {timer_extrapolation.elapsed_s:.1f} s")

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())
