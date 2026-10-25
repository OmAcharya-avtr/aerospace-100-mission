"""The baseline-against-learned benchmark, run in a fixed order.

The order is enforced in code, not left to the caller, because it is the thing
most easily got wrong: :func:`run_benchmark` fits and measures every analytic
form **before** it constructs the learned model at all, and the returned
:class:`BenchmarkResult` carries the analytic rows first. A learned model whose
hyperparameters were chosen after seeing the baseline's test score would be a
different and much weaker experiment, and in this mission a published learned
win was once inflated by about half by an under-strength baseline.

The comparison is on held-out **objects**, never on held-out rows of the same
object: :func:`orbitdrift.dataset.split_by_object` makes the split. Each
predictor appears twice, once with its own nominal interval and once wrapped in
split conformal calibration, so the table separates two questions that are
usually confused -- how good the point prediction is, and whether the interval
around it was honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calibration import (
    CoverageCell,
    IntervalMetrics,
    age_stratum,
    altitude_stratum,
    coverage_by_group,
    coverage_table,
    interval_metrics,
)
from .dataset import DivergenceTable, split_by_object
from .errors import InsufficientDataError
from .features import FEATURE_NAMES
from .predictors import (
    ANALYTIC_FORMS,
    AnalyticGrowthPredictor,
    ConformalPredictor,
    LearnedQuantilePredictor,
    RuleOfThumbPredictor,
)

_PERIGEE = FEATURE_NAMES.index("perigee_altitude_km")


@dataclass(frozen=True)
class PredictorRow:
    """One predictor's measured behaviour on the held-out objects."""

    name: str
    kind: str
    metrics: IntervalMetrics
    group_coverage: float
    group_coverage_ci: tuple[float, float]
    n_groups: int
    median_absolute_log_error: float
    extra: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    """Analytic rows, then the learned rows, then the coverage tables."""

    rows: tuple[PredictorRow, ...]
    coverage_by_altitude: dict[str, tuple[CoverageCell, ...]]
    coverage_by_age: dict[str, tuple[CoverageCell, ...]]
    n_train_rows: int
    n_test_rows: int
    n_train_objects: int
    n_test_objects: int
    nominal_coverage: float

    def row(self, name: str) -> PredictorRow:
        """One row by predictor name."""
        for candidate in self.rows:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no predictor named {name!r}; have {[r.name for r in self.rows]}")

    @property
    def analytic_rows(self) -> tuple[PredictorRow, ...]:
        return tuple(row for row in self.rows if row.kind == "analytic")

    @property
    def learned_rows(self) -> tuple[PredictorRow, ...]:
        return tuple(row for row in self.rows if row.kind == "learned")

    def best_by_interval_score(self, kind: str | None = None) -> PredictorRow:
        """Row with the lowest mean Winkler interval score, optionally within a kind."""
        candidates = [row for row in self.rows if kind is None or row.kind == kind]
        if not candidates:
            raise KeyError(f"no rows of kind {kind!r}")
        return min(candidates, key=lambda row: row.metrics.interval_score)

    def calibrated_rows(self) -> tuple[PredictorRow, ...]:
        """Rows whose measured coverage interval contains the nominal level."""
        return calibrated_rows(self.rows, self.nominal_coverage)

    def not_under_covering_rows(self) -> tuple[PredictorRow, ...]:
        """Rows whose measured coverage is at or above the nominal level."""
        return not_under_covering_rows(self.rows, self.nominal_coverage)

    def verdict_line(self) -> str:
        """The verdict, over all predictors, the calibrated ones and the safe ones."""
        return _verdict(self.rows, self.nominal_coverage, "cold-prediction task")


def _measure(
    name: str,
    kind: str,
    predictor,
    x_test: np.ndarray,
    y_test: np.ndarray,
    groups_test: np.ndarray,
    nominal: float,
    extra: dict[str, float] | None = None,
) -> tuple[PredictorRow, np.ndarray, np.ndarray]:
    lower, upper = predictor.predict_interval_km(x_test, nominal)
    median = np.asarray(predictor.predict_median_km(x_test), dtype=float)
    metrics = interval_metrics(lower, upper, y_test, nominal=nominal)
    fraction, interval, n_groups = coverage_by_group(lower, upper, y_test, groups_test)
    usable = (median > 0.0) & (y_test > 0.0)
    log_error = np.abs(np.log(y_test[usable]) - np.log(median[usable]))
    return (
        PredictorRow(
            name=name,
            kind=kind,
            metrics=metrics,
            group_coverage=fraction,
            group_coverage_ci=interval,
            n_groups=n_groups,
            median_absolute_log_error=(
                float(np.median(log_error)) if log_error.size else float("nan")
            ),
            extra=dict(extra or {}),
        ),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )


def run_benchmark(
    table: DivergenceTable,
    *,
    nominal_coverage: float = 0.90,
    test_fraction: float = 0.3,
    seed: int = 20740825,
    rule_of_thumb_km_per_day: float = 1.0,
    learned_estimators: int | None = None,
) -> BenchmarkResult:
    """Fit every analytic baseline, then the learned model, then measure all of them.

    Returns a :class:`BenchmarkResult` whose ``rows`` are ordered analytic
    first. The learned model's quantile pair is derived from
    ``nominal_coverage`` symmetrically.

    ``learned_estimators`` overrides the number of boosting stages. It exists
    so that the working-directory probe in
    ``validation/validate_cwd_independence.py``, which runs every script four
    times and only checks the exit status, can afford to. Every published
    number uses the default.
    """
    train_index, test_index = split_by_object(table, test_fraction=test_fraction, seed=seed)
    design = table.design()
    drift = table.column("drift_km")
    objects = table.column("object_index")
    x_train, y_train, g_train = design[train_index], drift[train_index], objects[train_index]
    x_test, y_test, g_test = design[test_index], drift[test_index], objects[test_index]

    rows: list[PredictorRow] = []
    altitude_cells: dict[str, tuple[CoverageCell, ...]] = {}
    age_cells: dict[str, tuple[CoverageCell, ...]] = {}
    altitude_labels = altitude_stratum(x_test[:, _PERIGEE])
    age_labels = age_stratum(x_test[:, -1])

    def record(name: str, kind: str, predictor, extra: dict[str, float] | None = None) -> None:
        row, lower, upper = _measure(
            name, kind, predictor, x_test, y_test, g_test, nominal_coverage, extra
        )
        rows.append(row)
        altitude_cells[name] = tuple(
            coverage_table(lower, upper, y_test, altitude_labels, nominal=nominal_coverage)
        )
        age_cells[name] = tuple(
            coverage_table(lower, upper, y_test, age_labels, nominal=nominal_coverage)
        )

    # --- Step 1: the rule of thumb the product is positioned against.
    record(
        "rule_of_thumb",
        "analytic",
        RuleOfThumbPredictor(rate_km_per_day=rule_of_thumb_km_per_day).fit(x_train, y_train),
        {"rate_km_per_day": rule_of_thumb_km_per_day},
    )

    # --- Step 2: every analytic growth form, nominal interval then conformal.
    for form in ANALYTIC_FORMS:
        fitted = AnalyticGrowthPredictor(form=form).fit(x_train, y_train)
        record(
            f"analytic_{form}",
            "analytic",
            fitted,
            {
                "effective_exponent": fitted.effective_exponent,
                "log_residual_std": fitted.log_residual_std_,
                "timescale_days": fitted.timescale_days_,
            },
        )
        conformal = ConformalPredictor(
            AnalyticGrowthPredictor(form=form), coverage=nominal_coverage, random_state=seed
        )
        conformal.fit(x_train, y_train, groups=g_train)
        record(
            f"analytic_{form}_conformal",
            "analytic",
            conformal,
            {
                "log_half_width": conformal.log_half_width_,
                "n_calibration": float(conformal.n_calibration_),
            },
        )

    # --- Step 3: only now the learned model.
    tail = 0.5 * (1.0 - nominal_coverage)
    estimators = (
        {} if learned_estimators is None else {"n_estimators": int(learned_estimators)}
    )
    learned = LearnedQuantilePredictor(
        lower_quantile=tail, upper_quantile=1.0 - tail, random_state=seed, **estimators
    ).fit(x_train, y_train)
    record("learned_gbr_quantile", "learned", learned)
    learned_conformal = ConformalPredictor(
        LearnedQuantilePredictor(
            lower_quantile=tail, upper_quantile=1.0 - tail, random_state=seed, **estimators
        ),
        coverage=nominal_coverage,
        random_state=seed,
    )
    learned_conformal.fit(x_train, y_train, groups=g_train)
    record(
        "learned_gbr_conformal",
        "learned",
        learned_conformal,
        {
            "log_half_width": learned_conformal.log_half_width_,
            "n_calibration": float(learned_conformal.n_calibration_),
        },
    )

    return BenchmarkResult(
        rows=tuple(rows),
        coverage_by_altitude=altitude_cells,
        coverage_by_age=age_cells,
        n_train_rows=int(train_index.size),
        n_test_rows=int(test_index.size),
        n_train_objects=int(np.unique(g_train).size),
        n_test_objects=int(np.unique(g_test).size),
        nominal_coverage=float(nominal_coverage),
    )


# ---------------------------------------------------------------------------
# The second, and more realistic, task.
#
# The benchmark above is a cold prediction: nothing about the object has been
# observed, and a model must guess its drift from orbital elements alone. That
# is the hard version of the question and it is worth measuring, but it is not
# the one an operator faces. An operator has the element set in hand and can
# run the divergence computation of this package over the first couple of days
# in a few seconds. The question is then an extrapolation: given the curve so
# far, how far does it go?
#
# For that task the analytic growth forms are not a population regression.
# They are fitted to the object's own curve, which is the strongest classical
# baseline available and the one the specification predicts will match. Both
# families are measured on exactly the same rows -- the ages beyond the fitting
# horizon, for held-out objects only -- and the learned model is additionally
# given the object's own fitted amplitude, exponent and residual spread as
# features, so that it is not handicapped by being denied information the
# baseline has.
# ---------------------------------------------------------------------------

EXTRA_CURVE_FEATURES: tuple[str, ...] = (
    "own_log_amplitude_km",
    "own_exponent",
    "own_log_residual_std",
)
"""Features derived from the object's own curve inside the fitting horizon."""


@dataclass(frozen=True)
class ExtrapolationResult:
    """Per-object analytic extrapolation against a learned model, on the same rows."""

    rows: tuple[PredictorRow, ...]
    coverage_by_altitude: dict[str, tuple[CoverageCell, ...]]
    coverage_by_age: dict[str, tuple[CoverageCell, ...]]
    fit_horizon_days: float
    n_fit_rows: int
    n_target_rows: int
    n_test_target_rows: int
    n_train_objects: int
    n_test_objects: int
    nominal_coverage: float
    form_counts: dict[str, int]

    def row(self, name: str) -> PredictorRow:
        for candidate in self.rows:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no predictor named {name!r}; have {[r.name for r in self.rows]}")

    def best_by_interval_score(self, kind: str | None = None) -> PredictorRow:
        candidates = [row for row in self.rows if kind is None or row.kind == kind]
        if not candidates:
            raise KeyError(f"no rows of kind {kind!r}")
        return min(candidates, key=lambda row: row.metrics.interval_score)

    def calibrated_rows(self) -> tuple[PredictorRow, ...]:
        """Rows whose measured coverage interval contains the nominal level."""
        return calibrated_rows(self.rows, self.nominal_coverage)

    def not_under_covering_rows(self) -> tuple[PredictorRow, ...]:
        """Rows whose measured coverage is at or above the nominal level."""
        return not_under_covering_rows(self.rows, self.nominal_coverage)

    def verdict_line(self) -> str:
        """The verdict, over all predictors, the calibrated ones and the safe ones."""
        return _verdict(self.rows, self.nominal_coverage, "extrapolation task")


def _own_curve_fits(
    table: DivergenceTable, fit_horizon_days: float
) -> tuple[dict[float, dict[str, object]], dict[float, np.ndarray]]:
    """Fit every analytic form to each object's curve inside the horizon."""
    from .growth import fit_all_forms

    objects = table.column("object_index")
    ages = table.column("epoch_age_days")
    drift = table.column("drift_km")
    fits: dict[float, dict[str, object]] = {}
    extras: dict[float, np.ndarray] = {}
    for value in np.unique(objects):
        inside = (objects == value) & (ages <= fit_horizon_days)
        if np.count_nonzero(inside) < 3:
            continue
        object_fits = fit_all_forms(ages[inside], drift[inside])
        if not object_fits:
            continue
        fits[float(value)] = object_fits
        power = object_fits.get("power")
        if power is None:
            extras[float(value)] = np.array([np.nan, np.nan, np.nan])
        else:
            extras[float(value)] = np.array(
                [
                    np.log(max(power.amplitude_km, 1e-12)),
                    power.exponent,
                    power.log_residual_std,
                ]
            )
    return fits, extras


@dataclass(frozen=True)
class _ExtrapolationPrediction:
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def _extrapolate(
    form: str,
    fits: dict[float, dict[str, object]],
    objects: np.ndarray,
    ages: np.ndarray,
    coverage: float,
) -> _ExtrapolationPrediction:
    median = np.full(ages.size, np.nan)
    lower = np.full(ages.size, np.nan)
    upper = np.full(ages.size, np.nan)
    for value in np.unique(objects):
        mask = objects == value
        object_fits = fits.get(float(value))
        if object_fits is None or form not in object_fits:
            continue
        fit = object_fits[form]
        median[mask] = fit.predict_km(ages[mask])
        low, high = fit.interval_km(ages[mask], coverage=coverage)
        lower[mask] = low
        upper[mask] = high
    return _ExtrapolationPrediction(median, lower, upper)


def run_extrapolation_benchmark(
    table: DivergenceTable,
    *,
    fit_horizon_days: float = 2.0,
    nominal_coverage: float = 0.90,
    test_fraction: float = 0.3,
    seed: int = 20740825,
    learned_estimators: int | None = None,
) -> ExtrapolationResult:
    """Per-object analytic extrapolation against a learned model on the same rows.

    Every object's curve is split at ``fit_horizon_days``. The analytic forms
    are fitted to the part inside the horizon and extrapolated beyond it; the
    learned model is trained on the beyond-horizon rows of the **training**
    objects, using the static features, the age, and the three
    :data:`EXTRA_CURVE_FEATURES` taken from the same inside-horizon fit the
    baseline used. Both are measured on the beyond-horizon rows of the
    **test** objects.

    Two conformal variants are included. The analytic one takes its
    half-width from the extrapolation residuals of the *training* objects,
    which is the honest way to calibrate an extrapolation: the object's own
    inside-horizon residual spread says nothing about how its fit behaves
    outside the range it was fitted on, and the measured coverage of the
    nominal interval in the result shows exactly how little.
    """
    from .growth import fit_all_forms  # noqa: F401 - used via _own_curve_fits

    train_index, test_index = split_by_object(table, test_fraction=test_fraction, seed=seed)
    objects = table.column("object_index")
    ages = table.column("epoch_age_days")
    drift = table.column("drift_km")
    static = table.design()[:, :-1]

    fits, extras = _own_curve_fits(table, fit_horizon_days)
    form_counts: dict[str, int] = {}
    for object_fits in fits.values():
        from .growth import best_form as _best

        name = _best(object_fits)  # type: ignore[arg-type]
        form_counts[name] = form_counts.get(name, 0) + 1

    beyond = ages > fit_horizon_days
    has_fit = np.array([float(value) in fits for value in objects])
    is_train = np.zeros(objects.size, dtype=bool)
    is_train[train_index] = True
    train_target = np.flatnonzero(beyond & has_fit & is_train)
    test_target = np.flatnonzero(beyond & has_fit & ~is_train)
    if test_target.size == 0 or train_target.size == 0:
        raise InsufficientDataError(
            f"the horizon {fit_horizon_days} days left {train_target.size} training and "
            f"{test_target.size} test rows beyond it"
        )

    extra_matrix = np.array([extras[float(value)] for value in objects])
    learned_design = np.column_stack([static, ages, extra_matrix])
    plain_design = np.column_stack([static, ages])

    rows: list[PredictorRow] = []
    altitude_cells: dict[str, tuple[CoverageCell, ...]] = {}
    age_cells: dict[str, tuple[CoverageCell, ...]] = {}
    altitude_labels = altitude_stratum(static[test_target, _PERIGEE])
    age_labels = age_stratum(ages[test_target])
    truth = drift[test_target]
    test_objects = objects[test_target]

    def record(
        name: str,
        kind: str,
        median: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        extra: dict[str, float] | None = None,
    ) -> None:
        metrics = interval_metrics(lower, upper, truth, nominal=nominal_coverage)
        fraction, interval, n_groups = coverage_by_group(lower, upper, truth, test_objects)
        usable = np.isfinite(median) & (median > 0.0) & (truth > 0.0)
        log_error = np.abs(np.log(truth[usable]) - np.log(median[usable]))
        rows.append(
            PredictorRow(
                name=name,
                kind=kind,
                metrics=metrics,
                group_coverage=fraction,
                group_coverage_ci=interval,
                n_groups=n_groups,
                median_absolute_log_error=float(np.median(log_error))
                if log_error.size
                else float("nan"),
                extra=dict(extra or {}),
            )
        )
        altitude_cells[name] = tuple(
            coverage_table(lower, upper, truth, altitude_labels, nominal=nominal_coverage)
        )
        age_cells[name] = tuple(
            coverage_table(lower, upper, truth, age_labels, nominal=nominal_coverage)
        )

    # --- Step 1: the analytic extrapolations, nominal and then conformal.
    for form in ("linear", "quadratic", "power", "drag_exponential"):
        test_prediction = _extrapolate(
            form, fits, objects[test_target], ages[test_target], nominal_coverage
        )
        record(
            f"own_curve_{form}",
            "analytic",
            test_prediction.median,
            test_prediction.lower,
            test_prediction.upper,
        )
        train_prediction = _extrapolate(
            form, fits, objects[train_target], ages[train_target], nominal_coverage
        )
        usable = (
            np.isfinite(train_prediction.median)
            & (train_prediction.median > 0.0)
            & (drift[train_target] > 0.0)
        )
        residual = np.abs(
            np.log(drift[train_target][usable]) - np.log(train_prediction.median[usable])
        )
        n = residual.size
        index = min(n, int(np.ceil((n + 1) * nominal_coverage)))
        half_width = float(np.sort(residual)[index - 1]) if n else float("nan")
        factor = float(np.exp(half_width))
        record(
            f"own_curve_{form}_conformal",
            "analytic",
            test_prediction.median,
            test_prediction.median / factor,
            test_prediction.median * factor,
            {"log_half_width": half_width, "n_calibration": float(n)},
        )

    # --- Step 2: only now the learned models.
    tail = 0.5 * (1.0 - nominal_coverage)
    estimators = (
        {} if learned_estimators is None else {"n_estimators": int(learned_estimators)}
    )
    for label, design in (("with_curve_features", learned_design), ("features_only", plain_design)):
        learned = LearnedQuantilePredictor(
            lower_quantile=tail, upper_quantile=1.0 - tail, random_state=seed, **estimators
        ).fit(design[train_target], drift[train_target])
        low, high = learned.predict_interval_km(design[test_target], nominal_coverage)
        record(
            f"learned_{label}",
            "learned",
            learned.predict_median_km(design[test_target]),
            low,
            high,
        )
        conformal = ConformalPredictor(
            LearnedQuantilePredictor(
                lower_quantile=tail,
                upper_quantile=1.0 - tail,
                random_state=seed,
                **estimators,
            ),
            coverage=nominal_coverage,
            random_state=seed,
        )
        conformal.fit(design[train_target], drift[train_target], groups=objects[train_target])
        low, high = conformal.predict_interval_km(design[test_target])
        record(
            f"learned_{label}_conformal",
            "learned",
            conformal.predict_median_km(design[test_target]),
            low,
            high,
            {
                "log_half_width": conformal.log_half_width_,
                "n_calibration": float(conformal.n_calibration_),
            },
        )

    return ExtrapolationResult(
        rows=tuple(rows),
        coverage_by_altitude=altitude_cells,
        coverage_by_age=age_cells,
        fit_horizon_days=float(fit_horizon_days),
        n_fit_rows=int(np.count_nonzero(~beyond)),
        n_target_rows=int(np.count_nonzero(beyond)),
        n_test_target_rows=int(test_target.size),
        n_train_objects=int(np.unique(objects[train_target]).size),
        n_test_objects=int(np.unique(test_objects).size),
        nominal_coverage=float(nominal_coverage),
        form_counts=form_counts,
    )

def calibrated_rows(rows, nominal: float) -> tuple[PredictorRow, ...]:
    """Rows whose measured coverage interval contains ``nominal``.

    A predictor whose interval does not cover at the level it claims cannot be
    compared on width against one that does: narrowing an interval always
    improves its score until it stops covering, so width is only meaningful
    among predictors that pass coverage first. Every verdict in this module is
    therefore stated over all predictors and then over this subset, and the
    second is the one to read.
    """
    return tuple(
        row
        for row in rows
        if row.metrics.coverage_ci[0] <= nominal <= row.metrics.coverage_ci[1]
    )


def not_under_covering_rows(rows, nominal: float) -> tuple[PredictorRow, ...]:
    """Rows whose measured coverage is at least ``nominal``.

    The two failure directions are not symmetric for the decision this package
    serves. An interval that covers 95 per cent when it claims 90 is wider than
    it needs to be, and a staleness horizon taken from it is early: the user
    refreshes an element set sooner than strictly necessary. An interval that
    covers 86 per cent when it claims 90 gives a horizon that is late, and the
    user acts on an element set that is already outside the tolerance they
    declared. So a conservative miscalibration is a cost and an optimistic one
    is a defect, and the ranking that matters operationally is over the
    predictors that do not under-cover.
    """
    return tuple(row for row in rows if row.metrics.coverage >= nominal)


def _verdict(rows, nominal: float, task: str) -> str:
    lines = []
    overall = min(rows, key=lambda row: row.metrics.interval_score)
    lines.append(
        f"{task}: lowest mean interval score overall is {overall.name} "
        f"({overall.kind}) at {overall.metrics.interval_score:.4f} km, with measured "
        f"coverage {overall.metrics.coverage:.4f} against {nominal:.4f} nominal"
    )

    point = min(rows, key=lambda row: row.median_absolute_log_error)
    point_analytic = min(
        (row for row in rows if row.kind == "analytic"),
        key=lambda row: row.median_absolute_log_error,
    )
    point_learned = min(
        (row for row in rows if row.kind == "learned"),
        key=lambda row: row.median_absolute_log_error,
    )
    lines.append(
        f"{task}: best point prediction is {point.name} ({point.kind}) at a median "
        f"absolute log error of {point.median_absolute_log_error:.4f}; best analytic "
        f"{point_analytic.name} {point_analytic.median_absolute_log_error:.4f} against "
        f"best learned {point_learned.name} "
        f"{point_learned.median_absolute_log_error:.4f}"
    )

    kept = calibrated_rows(rows, nominal)
    if not kept:
        lines.append(
            f"{task}: NO predictor's measured coverage interval contains the nominal "
            f"{nominal:.4f}; every interval in this table is miscalibrated and none of "
            "the widths is comparable"
        )
    else:
        best = min(kept, key=lambda row: row.metrics.interval_score)
        lines.append(
            f"{task}: among the {len(kept)} predictor(s) whose measured coverage is "
            f"consistent with {nominal:.4f}, the best interval score is {best.name} "
            f"({best.kind}) at {best.metrics.interval_score:.4f} km, coverage "
            f"{best.metrics.coverage:.4f} "
            f"[{best.metrics.coverage_ci[0]:.4f}, {best.metrics.coverage_ci[1]:.4f}]"
        )
        kinds = {row.kind for row in kept}
        if "learned" not in kinds:
            lines.append(
                f"{task}: no learned predictor is calibrated on this task; the only "
                "intervals consistent with their nominal level are analytic"
            )
        elif "analytic" not in kinds:
            lines.append(
                f"{task}: no analytic predictor is calibrated on this task; the only "
                "intervals consistent with their nominal level are learned"
            )
        else:
            analytic = min(
                (row for row in kept if row.kind == "analytic"),
                key=lambda row: row.metrics.interval_score,
            )
            learned = min(
                (row for row in kept if row.kind == "learned"),
                key=lambda row: row.metrics.interval_score,
            )
            ratio = learned.metrics.interval_score / analytic.metrics.interval_score
            winner = "analytic" if ratio > 1.0 else "learned"
            lines.append(
                f"{task}: among calibrated predictors the {winner} family wins -- "
                f"{analytic.name} {analytic.metrics.interval_score:.4f} km against "
                f"{learned.name} {learned.metrics.interval_score:.4f} km, ratio "
                f"{max(ratio, 1.0 / ratio):.4f}"
            )

    safe = not_under_covering_rows(rows, nominal)
    if not safe:
        lines.append(
            f"{task}: EVERY predictor under-covers at {nominal:.4f}. For a staleness "
            "decision that means every horizon in this table is late, and the table "
            "has no operationally usable row."
        )
        return "\n".join(lines)
    best_safe = min(safe, key=lambda row: row.metrics.interval_score)
    lines.append(
        f"{task}: among the {len(safe)} predictor(s) that do not under-cover "
        f"(measured coverage at or above {nominal:.4f}), the best interval score is "
        f"{best_safe.name} ({best_safe.kind}) at "
        f"{best_safe.metrics.interval_score:.4f} km, coverage "
        f"{best_safe.metrics.coverage:.4f}"
    )
    safe_kinds = {row.kind for row in safe}
    if "learned" not in safe_kinds:
        lines.append(
            f"{task}: no learned predictor reaches its nominal coverage; every "
            "learned interval here is optimistic, and a staleness horizon taken "
            "from one would be late"
        )
    return "\n".join(lines)
