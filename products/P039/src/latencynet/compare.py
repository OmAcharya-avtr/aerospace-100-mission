"""The three-way comparison, on held-out pipelines, in both dependence regimes.

Protocol, fixed before any result was looked at
-----------------------------------------------
1. Generate a population of pipelines in one regime and split it by *pipeline*
   into train / calibration / test. A held-out pipeline is one the model has
   never seen; no pass from a test pipeline appears anywhere in fitting.
2. Fit baseline 1 (analytic sum-of-stages, nothing to fit), baseline 2 (OLS)
   and the learned model on the train split only.
3. Calibrate a split-conformal wrapper for each model on the calibration
   split, so that every model's interval is produced by the same procedure and
   what remains to compare is interval width at matched coverage.
4. Score on the test split: accuracy in log space, native interval coverage,
   conformal interval coverage, and a paired t-test of each model's absolute
   log errors against the analytic baseline's on the same pipelines.
5. The winner is the model with the smallest **mean absolute log error** on
   the test split. A win is called *significant* only when the paired t-test
   against the analytic baseline gives ``p < 0.05``. Ties within that are
   reported as ties.

The covariance-aware analytic variant is scored too, as a diagnostic. It is
not one of the two baselines the specification names; it is there so that the
reader can see how much of any learned-model advantage is just "knowing the
covariance", which is information a profiler can measure directly.

Expected outcome, stated in advance
-----------------------------------
In the independent regime the analytic baseline should win: equation (2) is
exact there, and nothing a learned model can do with a few hundred noisy
targets improves on an identity. In the correlated regime the
independence-mode analytic baseline is biased low, and the models with access
to the dependence features should win. Whether that is what happens is an
empirical question and the answer, either way, is the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conformal import ConformalPredictor, coverage_quantisation
from .dataset import PipelineDataset, log_target
from .learned import LearnedTailPredictor
from .linear import LinearTailPredictor
from .metrics import (
    AccuracyResult,
    CoverageResult,
    interval_coverage,
    log_accuracy,
    paired_difference_test,
)
from .predictors import AnalyticTailPredictor, TailPredictor

#: Name of the baseline every other model is compared against.
REFERENCE_MODEL = "analytic_sum_indep"


@dataclass(frozen=True)
class ModelScore:
    """Held-out score for one model at one tail probability.

    Attributes
    ----------
    name:
        Model name.
    is_baseline:
        Whether this model is one of the two specified baselines.
    accuracy:
        Point-prediction accuracy in log space on the test split.
    native_coverage:
        Coverage of the model's own interval, or ``None`` if it has none.
    conformal_coverage:
        Coverage of the split-conformal interval built on the calibration
        split.
    paired_mean_difference:
        Mean of ``|error| - |error of the analytic baseline|`` over the test
        pipelines. Negative means this model is more accurate.
    paired_p_value:
        Two-sided p-value of the paired t-test against the analytic baseline.
        ``nan`` for the baseline itself.
    log_errors:
        Signed log errors on the test split, retained so a caller can plot
        them.
    """

    name: str
    is_baseline: bool
    accuracy: AccuracyResult
    native_coverage: CoverageResult | None
    conformal_coverage: CoverageResult
    paired_mean_difference: float
    paired_p_value: float
    log_errors: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of the three-way comparison in one regime at one probability.

    Attributes
    ----------
    regime:
        ``"independent"`` or ``"correlated"``.
    p:
        Tail probability compared.
    level:
        Nominal interval coverage used.
    scores:
        One :class:`ModelScore` per model, in fitting order.
    winner:
        Name of the model with the smallest mean absolute log error.
    winner_is_significant:
        Whether the winner's paired t-test against the analytic baseline gave
        ``p < 0.05``. ``False`` when the winner *is* the analytic baseline.
    n_train, n_calibration, n_test:
        Split sizes, in pipelines.
    coverage_quantisation:
        ``1 / (m + 1)``, the smallest coverage step the calibration split can
        resolve.
    mean_reference_relative_se:
        Mean relative Monte Carlo standard error of the reference targets on
        the test split. Model errors below this figure are not resolvable with
        this reference sample size, and the comparison says so rather than
        reading a winner out of the noise.
    """

    regime: str
    p: float
    level: float
    scores: tuple[ModelScore, ...]
    winner: str
    winner_is_significant: bool
    n_train: int
    n_calibration: int
    n_test: int
    coverage_quantisation: float
    mean_reference_relative_se: float

    def score(self, name: str) -> ModelScore:
        """Look up one model's score by name."""
        for s in self.scores:
            if s.name == name:
                return s
        raise KeyError(f"no model named {name!r}; have {[s.name for s in self.scores]}")


def default_models(level: float = 0.9) -> tuple[TailPredictor, ...]:
    """The two baselines, the learned model, and the covariance diagnostic.

    Order is fixed: the analytic baseline first, because it is the reference
    every other model is tested against.
    """
    return (
        AnalyticTailPredictor(assume_independent=True),
        LinearTailPredictor(),
        LearnedTailPredictor(native_interval_level=level),
        AnalyticTailPredictor(assume_independent=False),
    )


_BASELINE_NAMES = ("analytic_sum_indep", "linear_ols")


def compare_models(
    dataset: PipelineDataset,
    p: float,
    level: float = 0.9,
    models: tuple[TailPredictor, ...] | None = None,
) -> ComparisonResult:
    """Run the protocol in the module docstring and return the scored result."""
    if not (0.0 < float(p) < 1.0):
        raise ValueError(f"p must lie strictly in (0, 1), got {p!r}")
    if not (0.0 < float(level) < 1.0):
        raise ValueError(f"level must lie strictly in (0, 1), got {level!r}")
    models = models if models is not None else default_models(level=level)
    if not models:
        raise ValueError("models must be non-empty")

    truth = log_target(dataset.test, p)
    ref_rel_se = float(
        np.mean(
            [
                r.reference_quantile_se_s[float(p)] / r.reference_quantile_s[float(p)]
                for r in dataset.test
            ]
        )
    )

    predictions: dict[str, np.ndarray] = {}
    native: dict[str, CoverageResult | None] = {}
    conformal: dict[str, CoverageResult] = {}

    for model in models:
        model.fit(dataset.train, p)
        predictions[model.name] = model.predict_log(dataset.test)
        try:
            native[model.name] = interval_coverage(
                model.predict_log_interval(dataset.test, level=level), truth
            )
        except (NotImplementedError, RuntimeError):
            native[model.name] = None
        wrapper = ConformalPredictor.from_fitted(model, float(p), level=level)
        wrapper.calibrate(dataset.calibration)
        conformal[model.name] = interval_coverage(
            wrapper.predict_log_interval(dataset.test), truth
        )

    ref_err = predictions[REFERENCE_MODEL] - truth
    scores: list[ModelScore] = []
    for model in models:
        err = predictions[model.name] - truth
        if model.name == REFERENCE_MODEL:
            mean_diff, p_value = 0.0, float("nan")
        else:
            mean_diff, _, p_value = paired_difference_test(err, ref_err)
        scores.append(
            ModelScore(
                name=model.name,
                is_baseline=model.name in _BASELINE_NAMES,
                accuracy=log_accuracy(predictions[model.name], truth),
                native_coverage=native[model.name],
                conformal_coverage=conformal[model.name],
                paired_mean_difference=mean_diff,
                paired_p_value=p_value,
                log_errors=err,
            )
        )

    best = min(scores, key=lambda s: s.accuracy.mean_abs_log_error)
    significant = (
        best.name != REFERENCE_MODEL
        and np.isfinite(best.paired_p_value)
        and best.paired_p_value < 0.05
    )
    return ComparisonResult(
        regime=dataset.regime,
        p=float(p),
        level=float(level),
        scores=tuple(scores),
        winner=best.name,
        winner_is_significant=bool(significant),
        n_train=len(dataset.train),
        n_calibration=len(dataset.calibration),
        n_test=len(dataset.test),
        coverage_quantisation=coverage_quantisation(len(dataset.calibration)),
        mean_reference_relative_se=ref_rel_se,
    )


def format_comparison(result: ComparisonResult) -> str:
    """Render a comparison as a fixed-width table, for a validation log."""
    lines = [
        f"regime={result.regime}  p={result.p}  nominal level={result.level}  "
        f"train={result.n_train} cal={result.n_calibration} test={result.n_test}",
        f"reference-target mean relative MC standard error: "
        f"{result.mean_reference_relative_se * 100:.3f} %",
        "",
        f"{'model':<26}{'mean|dlnq|':>11}{'med|dlnq|':>11}{'bias':>9}"
        f"{'native cov':>12}{'conf cov':>10}{'conf width':>12}{'p vs base':>11}",
        "-" * 102,
    ]
    for s in result.scores:
        native = "n/a" if s.native_coverage is None else f"{s.native_coverage.measured:.3f}"
        p_txt = "ref" if not np.isfinite(s.paired_p_value) else f"{s.paired_p_value:.4f}"
        lines.append(
            f"{s.name:<26}{s.accuracy.mean_abs_log_error:>11.5f}"
            f"{s.accuracy.median_abs_log_error:>11.5f}{s.accuracy.bias_log:>9.4f}"
            f"{native:>12}{s.conformal_coverage.measured:>10.3f}"
            f"{s.conformal_coverage.mean_log_width:>12.4f}{p_txt:>11}"
        )
    verdict = "significant" if result.winner_is_significant else "not significant vs the baseline"
    lines += ["-" * 102, f"winner by mean absolute log error: {result.winner} ({verdict})"]
    return "\n".join(lines)
