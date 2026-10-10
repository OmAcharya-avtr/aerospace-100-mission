"""The coverage audit: the deliverable of this package.

One audit replicate draws a fit sample and a calibration sample from the
calibration distribution and one test sample per declared shift severity. Each
interval method is calibrated on the same calibration sample and evaluated on
every test sample, so every number in a row of the output table is computed on
identical data. Results are pooled over replicates and reported with an exact
binomial interval.

Methods compared
----------------
``parametric``
    The baseline. ``yhat +/- z_{1-alpha/2} sigma_hat`` with ``sigma_hat`` the
    residual standard deviation on the calibration sample. Assumes
    homoscedastic Gaussian residuals and an unbiased mean model.
``split``
    Marginal split conformal. Distribution-free under exchangeability.
``mondrian``
    Split conformal conditional on a tercile of the calibration prediction.
``weighted_declared``
    Weighted conformal with the exact likelihood ratio of the declared shift.
    At severity zero every weight is 1 and it coincides with ``split``.
``weighted_learned``
    Weighted conformal with the likelihood ratio estimated by logistic
    regression from unlabelled covariates.

Reading the table
-----------------
An unbounded interval covers by construction, so coverage alone cannot be read
without ``infinite_fraction``. ``mean_width`` is the mean over **finite**
intervals only and is stated as such. ``ess_fraction`` is Kish's effective
sample size divided by the calibration size and is the early warning that a
weighted method is running out of usable calibration points.

Which confidence interval
-------------------------
Test points inside one replicate share a calibration quantile, so they are
**not** independent Bernoulli trials and a Clopper-Pearson interval on the
pooled count is too narrow. The primary interval reported here is therefore a
Student-t interval across replicate coverages, which treats the replicate as
the unit of independence. The Clopper-Pearson interval is reported alongside
as ``cp_low``/``cp_high`` so the size of the difference is visible; it is
typically two to three times narrower and should not be used to decide whether
a method holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .baseline import GaussianResidualInterval, PhysicsRegressor
from .bounds import clopper_pearson, effective_sample_size, split_conformal_coverage_bound
from .conformal import (
    Interval,
    MondrianConformal,
    SplitConformal,
    WeightedConformal,
    assign_bins,
    tercile_edges,
)
from .data import Dataset, make_dataset
from .learned import LearnedRegressor, LearnedWeightEstimator
from .shift import SHIFT_LEVELS, CovariateShift

METHODS: tuple[str, ...] = (
    "parametric",
    "split",
    "mondrian",
    "weighted_declared",
    "weighted_learned",
)
"""Interval methods the audit compares, baseline first."""

MODELS: tuple[str, ...] = ("physics", "learned")
"""Point predictors the audit compares, analytic baseline first."""

DEFAULT_ALPHA = 0.1
DEFAULT_REPLICATES = 40
DEFAULT_N_FIT = 1500
DEFAULT_N_CALIBRATION = 500
DEFAULT_N_TEST = 1000
DEFAULT_SEED = 57001


@dataclass(frozen=True)
class CoverageRow:
    """One (model, method, severity) cell of the audit table.

    Attributes
    ----------
    model, method:
        Point predictor and interval method names.
    severity:
        Declared shift severity [-].
    mahalanobis:
        Mahalanobis displacement of the declared shift [-].
    nominal:
        ``1 - alpha`` [-].
    coverage:
        Pooled empirical coverage over all replicates [-].
    ci_low, ci_high:
        95 % Student-t interval across replicate coverages [-]. This is the
        interval to read.
    cp_low, cp_high:
        Clopper-Pearson 95 % interval on the pooled count [-]. Reported for
        comparison only; it is too narrow because test points within a
        replicate are correlated.
    per_replicate_sd, replicate_se:
        Standard deviation of the per-replicate coverage and its standard
        error [-].
    mean_width, median_width:
        Interval width over finite intervals only [Wh].
    infinite_fraction:
        Fraction of test points whose interval was unbounded [-].
    ess, ess_fraction:
        Mean Kish effective sample size of the calibration weights and that
        divided by the calibration size [-]. Both are 1.0 and the calibration
        size for unweighted methods.
    replicates, test_points:
        Counts behind the row.
    bound_exact, bound_lower, bound_upper:
        The finite-sample split-conformal coverage bound for this calibration
        size, shown on every row so the reader can see what exchangeability
        would have bought.
    """

    model: str
    method: str
    severity: float
    mahalanobis: float
    nominal: float
    coverage: float
    ci_low: float
    ci_high: float
    cp_low: float
    cp_high: float
    per_replicate_sd: float
    replicate_se: float
    mean_width: float
    median_width: float
    infinite_fraction: float
    ess: float
    ess_fraction: float
    replicates: int
    test_points: int
    bound_exact: float
    bound_lower: float
    bound_upper: float

    @property
    def covers_nominal(self) -> bool:
        """``True`` when coverage is **demonstrably at or above** nominal.

        That is ``ci_low >= nominal``. Demonstrating a one-sided guarantee
        takes far more replicates than detecting its violation, so this is
        ``False`` on many rows that are not broken.
        """
        return self.ci_low >= self.nominal

    @property
    def below_nominal(self) -> bool:
        """``True`` when coverage is **demonstrably below** nominal.

        That is ``ci_high < nominal``: the whole 95 % replicate-level
        interval lies under ``1 - alpha``. This is the criterion the
        breaking-point search uses, because it is the one that detects a
        failure rather than the one that certifies success.
        """
        return self.ci_high < self.nominal


@dataclass(frozen=True)
class AuditResult:
    """All rows of one audit plus the configuration that produced them."""

    rows: tuple[CoverageRow, ...]
    alpha: float
    replicates: int
    n_fit: int
    n_calibration: int
    n_test: int
    seed: int
    severities: tuple[float, ...]

    def select(self, *, model: str | None = None, method: str | None = None) -> list[CoverageRow]:
        """Rows filtered by model and/or method, in table order."""
        return [
            r
            for r in self.rows
            if (model is None or r.model == model) and (method is None or r.method == method)
        ]

    def table(self) -> str:
        """Fixed-width rendering of the audit table."""
        head = (
            f"{'model':8s} {'method':18s} {'sev':>4s} {'mahal':>6s} {'coverage':>9s} "
            f"{'ci_low':>7s} {'ci_high':>8s} {'repl_sd':>8s} {'width_Wh':>9s} "
            f"{'inf_frac':>9s} {'ess':>7s} {'ess_fr':>7s}"
        )
        lines = [head, "-" * len(head)]
        for r in self.rows:
            lines.append(
                f"{r.model:8s} {r.method:18s} {r.severity:4.1f} {r.mahalanobis:6.3f} "
                f"{r.coverage:9.5f} {r.ci_low:7.5f} {r.ci_high:8.5f} {r.per_replicate_sd:8.5f} "
                f"{r.mean_width:9.5f} {r.infinite_fraction:9.5f} {r.ess:7.1f} {r.ess_fraction:7.4f}"
            )
        return "\n".join(lines)


@dataclass
class _Accumulator:
    covered: int = 0
    total: int = 0
    widths: list[np.ndarray] = field(default_factory=list)
    infinite: int = 0
    ess: list[float] = field(default_factory=list)
    per_replicate: list[float] = field(default_factory=list)

    def add(self, interval: Interval, y_true: np.ndarray, ess: float) -> None:
        covered = interval.covers(y_true)
        width = interval.width
        finite = np.isfinite(width)
        self.covered += int(covered.sum())
        self.total += int(covered.size)
        self.infinite += int((~finite).sum())
        self.widths.append(width[finite])
        self.ess.append(float(ess))
        self.per_replicate.append(float(covered.mean()))


def _summarise(acc: _Accumulator, confidence: float = 0.95) -> dict[str, float]:
    """Pooled coverage, its replicate-level t interval and the pooled width stats."""
    widths = np.concatenate(acc.widths) if acc.widths else np.array([np.nan])
    per_replicate = np.asarray(acc.per_replicate, dtype=float)
    coverage = acc.covered / acc.total
    if per_replicate.size > 1:
        sd = float(np.std(per_replicate, ddof=1))
        se = sd / np.sqrt(per_replicate.size)
        half = float(stats.t.ppf(0.5 + confidence / 2.0, per_replicate.size - 1)) * se
    else:
        sd = 0.0
        se = float("nan")
        half = float("nan")
    cp_low, cp_high = clopper_pearson(acc.covered, acc.total, confidence)
    return {
        "coverage": coverage,
        "ci_low": coverage - half,
        "ci_high": coverage + half,
        "cp_low": cp_low,
        "cp_high": cp_high,
        "per_replicate_sd": sd,
        "replicate_se": se,
        "mean_width": float(np.mean(widths)) if widths.size else float("nan"),
        "median_width": float(np.median(widths)) if widths.size else float("nan"),
        "infinite_fraction": acc.infinite / acc.total,
        "ess": float(np.mean(acc.ess)),
    }


def _fit_model(name: str, dataset: Dataset, *, random_state: int):
    if name == "physics":
        fitted = PhysicsRegressor().fit(dataset.features, dataset.energy)
        return fitted, PhysicsRegressor.n_parameters
    if name == "learned":
        return LearnedRegressor(random_state=random_state).fit(dataset.features, dataset.energy), 0
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


def _replicate_data(
    *,
    seed: int,
    n_fit: int,
    n_calibration: int,
    n_test: int,
    severities: tuple[float, ...],
) -> tuple[Dataset, Dataset, dict[float, Dataset]]:
    rng = np.random.default_rng(seed)
    fit = make_dataset(n_fit, rng=rng, severity=0.0)
    calibration = make_dataset(n_calibration, rng=rng, severity=0.0)
    tests = {
        severity: make_dataset(n_test, rng=rng, shift=CovariateShift(severity=severity))
        for severity in severities
    }
    return fit, calibration, tests


def coverage_audit(
    *,
    alpha: float = DEFAULT_ALPHA,
    replicates: int = DEFAULT_REPLICATES,
    n_fit: int = DEFAULT_N_FIT,
    n_calibration: int = DEFAULT_N_CALIBRATION,
    n_test: int = DEFAULT_N_TEST,
    seed: int = DEFAULT_SEED,
    severities: tuple[float, ...] = SHIFT_LEVELS,
    models: tuple[str, ...] = MODELS,
    methods: tuple[str, ...] = METHODS,
) -> AuditResult:
    """Run the coverage audit and return one row per (model, method, severity).

    Parameters
    ----------
    alpha:
        Miscoverage level in (0, 1).
    replicates:
        Independent replicates; each draws fresh fit, calibration and test
        samples.
    n_fit, n_calibration, n_test:
        Sample sizes per replicate.
    seed:
        Base seed; replicate ``r`` uses ``seed + r``.
    severities:
        Declared shift severities to evaluate.
    models:
        Subset of :data:`MODELS`, baseline first.
    methods:
        Subset of :data:`METHODS`, baseline first.

    Returns
    -------
    AuditResult
    """
    if replicates <= 0:
        raise ValueError(f"replicates must be > 0, got {replicates}")
    unknown_methods = set(methods) - set(METHODS)
    if unknown_methods:
        raise ValueError(f"unknown methods {sorted(unknown_methods)}; choose from {METHODS}")
    unknown_models = set(models) - set(MODELS)
    if unknown_models:
        raise ValueError(f"unknown models {sorted(unknown_models)}; choose from {MODELS}")
    bound = split_conformal_coverage_bound(n_calibration, alpha)

    accumulators: dict[tuple[str, str, float], _Accumulator] = {
        (model, method, severity): _Accumulator()
        for model in models
        for method in methods
        for severity in severities
    }

    for index in range(replicates):
        fit_set, calibration, tests = _replicate_data(
            seed=seed + index,
            n_fit=n_fit,
            n_calibration=n_calibration,
            n_test=n_test,
            severities=tuple(severities),
        )
        for model_name in models:
            model, n_parameters = _fit_model(model_name, fit_set, random_state=seed + index)
            pred_cal = np.asarray(model.predict(calibration.features), dtype=float).ravel()
            parametric = GaussianResidualInterval(alpha, n_parameters=n_parameters).fit(
                calibration.energy, pred_cal
            )
            split = SplitConformal(alpha).calibrate(calibration.energy, pred_cal)
            edges = tercile_edges(pred_cal)
            bins_cal = assign_bins(pred_cal, edges)
            mondrian = MondrianConformal(alpha).calibrate(calibration.energy, pred_cal, bins_cal)

            for severity, test in tests.items():
                shift = CovariateShift(severity=severity)
                pred_test = np.asarray(model.predict(test.features), dtype=float).ravel()
                for method in methods:
                    if method == "parametric":
                        interval = parametric.interval(pred_test)
                        ess = float(n_calibration)
                    elif method == "split":
                        interval = split.interval(pred_test)
                        ess = float(n_calibration)
                    elif method == "mondrian":
                        interval = mondrian.interval(pred_test, assign_bins(pred_test, edges))
                        ess = float(n_calibration)
                    elif method == "weighted_declared":
                        w_cal = shift.likelihood_ratio(calibration.mass, calibration.headwind)
                        w_test = shift.likelihood_ratio(test.mass, test.headwind)
                        weighted = WeightedConformal(alpha).calibrate(
                            calibration.energy, pred_cal, w_cal
                        )
                        interval = weighted.interval(pred_test, w_test)
                        ess = effective_sample_size(w_cal)
                    else:
                        estimator = LearnedWeightEstimator().fit(
                            calibration.features, test.features
                        )
                        w_cal = estimator.weights(calibration.features)
                        w_test = estimator.weights(test.features)
                        weighted = WeightedConformal(alpha).calibrate(
                            calibration.energy, pred_cal, w_cal
                        )
                        interval = weighted.interval(pred_test, w_test)
                        ess = effective_sample_size(w_cal)
                    accumulators[(model_name, method, severity)].add(interval, test.energy, ess)

    rows: list[CoverageRow] = []
    for model_name in models:
        for method in methods:
            for severity in severities:
                summary = _summarise(accumulators[(model_name, method, severity)])
                rows.append(
                    CoverageRow(
                        model=model_name,
                        method=method,
                        severity=float(severity),
                        mahalanobis=CovariateShift(severity=severity).mahalanobis,
                        nominal=1.0 - alpha,
                        coverage=summary["coverage"],
                        ci_low=summary["ci_low"],
                        ci_high=summary["ci_high"],
                        cp_low=summary["cp_low"],
                        cp_high=summary["cp_high"],
                        per_replicate_sd=summary["per_replicate_sd"],
                        replicate_se=summary["replicate_se"],
                        mean_width=summary["mean_width"],
                        median_width=summary["median_width"],
                        infinite_fraction=summary["infinite_fraction"],
                        ess=summary["ess"],
                        ess_fraction=summary["ess"] / n_calibration,
                        replicates=replicates,
                        test_points=accumulators[(model_name, method, severity)].total,
                        bound_exact=bound.exact,
                        bound_lower=bound.lower,
                        bound_upper=bound.upper,
                    )
                )
    return AuditResult(
        rows=tuple(rows),
        alpha=alpha,
        replicates=replicates,
        n_fit=n_fit,
        n_calibration=n_calibration,
        n_test=n_test,
        seed=seed,
        severities=tuple(float(s) for s in severities),
    )


@dataclass(frozen=True)
class BreakingPointRow:
    """One assumed-weight fraction of the breaking-point sweep."""

    fraction: float
    assumed_severity: float
    coverage: float
    ci_low: float
    ci_high: float
    cp_low: float
    cp_high: float
    replicate_se: float
    mean_width: float
    infinite_fraction: float
    ess: float
    ess_fraction: float
    nominal: float
    test_points: int

    @property
    def below_nominal(self) -> bool:
        """``True`` when the whole 95 % replicate-level interval is under nominal."""
        return self.ci_high < self.nominal

    @property
    def holds(self) -> bool:
        """``True`` when coverage is not demonstrably below nominal."""
        return not self.below_nominal

    @property
    def demonstrated(self) -> bool:
        """``True`` when coverage is demonstrably at or above nominal."""
        return self.ci_low >= self.nominal


@dataclass(frozen=True)
class BreakingPointResult:
    """Where weighted conformal stops holding when its weights are wrong.

    Attributes
    ----------
    rows:
        One row per assumed fraction, ascending.
    last_holding_fraction:
        Smallest fraction at or below 1.0 that is **not** demonstrably below
        nominal, scanning down from 1.0 with no intervening failure.
    breaking_fraction:
        The next grid point below that: the largest assumed fraction whose
        coverage is demonstrably below nominal. ``None`` when no grid point
        at or below 1.0 fails.
    true_severity:
        Severity the test data was actually drawn under.
    """

    rows: tuple[BreakingPointRow, ...]
    last_holding_fraction: float | None
    breaking_fraction: float | None
    true_severity: float
    alpha: float
    replicates: int
    n_calibration: int
    n_test: int
    seed: int
    model: str

    def table(self) -> str:
        """Fixed-width rendering of the sweep."""
        head = (
            f"{'fraction':>8s} {'assumed':>8s} {'coverage':>9s} {'ci_low':>7s} "
            f"{'ci_high':>8s} {'width_Wh':>9s} {'inf_frac':>9s} {'ess':>7s} {'holds':>6s}"
        )
        lines = [head, "-" * len(head)]
        for r in self.rows:
            lines.append(
                f"{r.fraction:8.2f} {r.assumed_severity:8.3f} {r.coverage:9.5f} "
                f"{r.ci_low:7.5f} {r.ci_high:8.5f} {r.mean_width:9.5f} "
                f"{r.infinite_fraction:9.5f} {r.ess:7.1f} {str(r.holds):>6s}"
            )
        return "\n".join(lines)


def breaking_point_sweep(
    *,
    true_severity: float = 2.0,
    fractions: tuple[float, ...] | None = None,
    alpha: float = DEFAULT_ALPHA,
    replicates: int = DEFAULT_REPLICATES,
    n_fit: int = DEFAULT_N_FIT,
    n_calibration: int = DEFAULT_N_CALIBRATION,
    n_test: int = DEFAULT_N_TEST,
    seed: int = DEFAULT_SEED,
    model: str = "learned",
) -> BreakingPointResult:
    """Measure weighted conformal coverage as its declared weights go wrong.

    The test data is drawn under ``true_severity``. The weights are computed
    from ``CovariateShift(true_severity * fraction)``, so ``fraction = 1``
    is the correct weight model, ``fraction < 1`` understates the shift and
    ``fraction > 1`` overstates it.

    Parameters
    ----------
    true_severity:
        Severity the test sample is drawn under, positive.
    fractions:
        Grid of assumed fractions. Defaults to 0.00 to 1.50 in steps of 0.05.
    alpha, replicates, n_fit, n_calibration, n_test, seed:
        As in :func:`coverage_audit`.
    model:
        Point predictor, one of :data:`MODELS`.

    Returns
    -------
    BreakingPointResult
    """
    if true_severity <= 0.0:
        raise ValueError(f"true_severity must be > 0, got {true_severity}")
    grid = (
        tuple(round(0.05 * i, 2) for i in range(31)) if fractions is None else tuple(fractions)
    )
    if any(f < 0.0 for f in grid):
        raise ValueError("fractions must be >= 0")
    true_shift = CovariateShift(severity=true_severity)
    accumulators = {f: _Accumulator() for f in grid}

    for index in range(replicates):
        fit_set, calibration, tests = _replicate_data(
            seed=seed + index,
            n_fit=n_fit,
            n_calibration=n_calibration,
            n_test=n_test,
            severities=(true_severity,),
        )
        test = tests[true_severity]
        fitted, _ = _fit_model(model, fit_set, random_state=seed + index)
        pred_cal = np.asarray(fitted.predict(calibration.features), dtype=float).ravel()
        pred_test = np.asarray(fitted.predict(test.features), dtype=float).ravel()
        for fraction in grid:
            assumed = true_shift.scaled(fraction)
            w_cal = assumed.likelihood_ratio(calibration.mass, calibration.headwind)
            w_test = assumed.likelihood_ratio(test.mass, test.headwind)
            weighted = WeightedConformal(alpha).calibrate(calibration.energy, pred_cal, w_cal)
            interval = weighted.interval(pred_test, w_test)
            accumulators[fraction].add(interval, test.energy, effective_sample_size(w_cal))

    nominal = 1.0 - alpha
    rows: list[BreakingPointRow] = []
    for fraction in grid:
        acc = accumulators[fraction]
        summary = _summarise(acc)
        rows.append(
            BreakingPointRow(
                fraction=float(fraction),
                assumed_severity=float(true_severity * fraction),
                coverage=summary["coverage"],
                ci_low=summary["ci_low"],
                ci_high=summary["ci_high"],
                cp_low=summary["cp_low"],
                cp_high=summary["cp_high"],
                replicate_se=summary["replicate_se"],
                mean_width=summary["mean_width"],
                infinite_fraction=summary["infinite_fraction"],
                ess=summary["ess"],
                ess_fraction=summary["ess"] / n_calibration,
                nominal=nominal,
                test_points=acc.total,
            )
        )

    at_or_below = [r for r in rows if r.fraction <= 1.0]
    last_holding: float | None = None
    breaking: float | None = None
    for row in reversed(at_or_below):
        if row.below_nominal:
            breaking = row.fraction
            break
        last_holding = row.fraction
    return BreakingPointResult(
        rows=tuple(rows),
        last_holding_fraction=last_holding,
        breaking_fraction=breaking,
        true_severity=float(true_severity),
        alpha=float(alpha),
        replicates=int(replicates),
        n_calibration=int(n_calibration),
        n_test=int(n_test),
        seed=int(seed),
        model=model,
    )


def stratified_coverage(
    *,
    alpha: float = DEFAULT_ALPHA,
    replicates: int = 20,
    n_fit: int = DEFAULT_N_FIT,
    n_calibration: int = DEFAULT_N_CALIBRATION,
    n_test: int = DEFAULT_N_TEST,
    seed: int = DEFAULT_SEED,
    severity: float = 0.0,
    model: str = "learned",
) -> dict[str, dict[int, tuple[float, float, float, int]]]:
    """Per-stratum coverage of split against Mondrian conformal.

    Returns
    -------
    dict
        ``{method: {bin: (coverage, mean_width, nominal, n_points)}}`` with
        bins indexed 0, 1, 2 over terciles of the calibration prediction.
    """
    tallies: dict[str, dict[int, list[list[float]]]] = {
        "split": {b: [[], []] for b in range(3)},
        "mondrian": {b: [[], []] for b in range(3)},
    }
    for index in range(replicates):
        fit_set, calibration, tests = _replicate_data(
            seed=seed + index,
            n_fit=n_fit,
            n_calibration=n_calibration,
            n_test=n_test,
            severities=(severity,),
        )
        test = tests[severity]
        fitted, _ = _fit_model(model, fit_set, random_state=seed + index)
        pred_cal = np.asarray(fitted.predict(calibration.features), dtype=float).ravel()
        pred_test = np.asarray(fitted.predict(test.features), dtype=float).ravel()
        edges = tercile_edges(pred_cal)
        bins_cal = assign_bins(pred_cal, edges)
        bins_test = assign_bins(pred_test, edges)
        split = SplitConformal(alpha).calibrate(calibration.energy, pred_cal)
        mondrian = MondrianConformal(alpha).calibrate(calibration.energy, pred_cal, bins_cal)
        for name, interval in (
            ("split", split.interval(pred_test)),
            ("mondrian", mondrian.interval(pred_test, bins_test)),
        ):
            covered = interval.covers(test.energy)
            width = interval.width
            for b in range(3):
                mask = bins_test == b
                if not np.any(mask):
                    continue
                tallies[name][b][0].extend(covered[mask].tolist())
                tallies[name][b][1].extend(width[mask].tolist())
    out: dict[str, dict[int, tuple[float, float, float, int]]] = {}
    for name, per_bin in tallies.items():
        out[name] = {}
        for b, (covered, widths) in per_bin.items():
            if not covered:
                continue
            out[name][b] = (
                float(np.mean(covered)),
                float(np.mean(widths)),
                1.0 - alpha,
                len(covered),
            )
    return out
