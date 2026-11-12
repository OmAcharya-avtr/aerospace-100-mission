"""Build step 5: the learned thermal-state predictor, and the honest comparison.

The task. Given the first ``window_s`` seconds of a run -- die temperature,
mean CPU clock and achieved duty -- predict the EVENTUAL steady state: the
equilibrium die temperature the run will settle at. The point is to avoid
running the 12-minute soak.

**The comparison is against baseline 2 given the same inputs, and that is what
makes it a comparison.** The RC baseline is not handed the whole soak while the
learned model is handed 90 seconds. :func:`rc_warmup_prediction` fits the
first-order relaxation to exactly the same warm-up window the learned model
sees and extrapolates its asymptote, which is the strongest honest form of the
classical method on this input. The mission's build guide records two products
whose apparent learned margin was inflated by a weak baseline; fitting the RC
form on the full run and the model on 90 seconds would be a third.

**The expected and acceptable result is that the learned model does not win.**
A first-order relaxation is the right functional form for this system, it has
three parameters, and it is fitted to the very curve whose asymptote is the
target. A learned model with a handful of features and six runs of training
data has no obvious advantage. :func:`compare_against_rc` therefore reports the
margin against the run-to-run bootstrap interval of baseline 3 and returns the
verdict ``NOT_DISTINGUISHABLE`` whenever the margin is smaller than that
interval's half-width, regardless of its sign. A margin inside the interval is
never reported as an improvement.

**Two models, and the choice between them is an argument, not a default.**

- ``ridge`` -- standardised features into ``sklearn.linear_model.Ridge``, wrapped
  in a group bootstrap: ``n_models`` ridge fits on run-resampled training sets.
  The prediction is their mean, and the uncertainty is the spread across fits
  (epistemic) combined in quadrature with the in-sample residual spread
  (aleatoric). This is the PRIMARY model, for one reason that matters on this
  evaluation axis: held-out duty cycle means training on cooler duties and
  predicting a hotter one, which is extrapolation in the target, and a linear
  model extrapolates while a tree cannot.
- ``forest`` -- ``sklearn.ensemble.RandomForestRegressor``, uncertainty from the
  spread across trees. Reported beside the ridge because a tree ensemble is the
  obvious default choice and its failure mode on this axis is worth showing: its
  prediction is bounded by the training targets, so on the held-out duty-1.0
  split it cannot predict a steady state hotter than the hottest run it trained
  on, and it does not.

**Dataset size and correlation, stated plainly.** The campaign has six soak runs
across four duty cycles. The dataset is built by taking several warm-up window
lengths per run, which multiplies the row count but not the information: rows
from one run are nested inside it and are strongly correlated. Every split in
this module is therefore grouped by RUN, and the held-out-duty evaluation is
grouped by DUTY. Reporting a random-row cross-validation score on this dataset
would be meaningless and is not offered.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from . import bootstrap as bs
from .equilibrium import (
    DEFAULT_SLOPE_THRESHOLD_C_PER_S,
    DEFAULT_WINDOW_S,
    equilibrium_test,
    fit_linear,
)
from .provenance import Origin, meet
from .rc_model import RelaxationFitError, fit_exponential_relaxation

#: Warm-up window lengths, seconds, one dataset row per run per length.
DEFAULT_WINDOW_LENGTHS_S: tuple[float, ...] = (60.0, 75.0, 90.0, 105.0, 120.0, 150.0)

#: Features, in the order :func:`warmup_features` returns them.
FEATURE_NAMES: tuple[str, ...] = (
    "window_s",
    "achieved_duty_mean",
    "die_first_c",
    "die_last_c",
    "die_mean_c",
    "die_rise_c",
    "die_slope_c_per_s",
    "die_slope_second_half_c_per_s",
    "die_slope_ratio",
    "clock_mean_khz",
    "clock_rel_range",
)

#: Ridge penalty. Declared, not searched: with six runs there is no honest
#: held-out set to tune it on, and a penalty chosen by looking at the test duty
#: would make the held-out score meaningless.
RIDGE_ALPHA: float = 1.0

#: Group-bootstrap fits behind the ridge predictor's uncertainty.
N_MODELS: int = 200

#: Trees in the comparison forest.
N_TREES: int = 200

#: Seed for every random draw in this module unless overridden.
SEED: int = 910091

#: Fewest finite samples a warm-up window must contain.
MIN_WINDOW_SAMPLES: int = 20


class NotEnoughRunsError(ValueError):
    """Raised when a split leaves too few runs to train or to score."""


def warmup_features(trace, window_s: float) -> np.ndarray:
    """Feature vector from the first ``window_s`` seconds of ``trace``.

    Every feature is computable from a short warm-up trace alone: nothing here
    reads the end of the run, and nothing reads the target.

    Raises
    ------
    ValueError
        If the window holds fewer than :data:`MIN_WINDOW_SAMPLES` finite die
        samples, or the trace is shorter than the window.
    """
    if window_s <= 0.0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    if trace.n_rows == 0 or float(trace.t_rel[-1]) < window_s:
        raise ValueError(
            f"{trace.label!r} is {0.0 if trace.n_rows == 0 else float(trace.t_rel[-1]):g} s "
            f"long, shorter than the {window_s:g} s warm-up window"
        )
    window = trace.window(0.0, window_s)
    die = window.die_temperature_c
    time_s = window.t_rel
    keep = np.isfinite(die) & np.isfinite(time_s)
    if int(np.sum(keep)) < MIN_WINDOW_SAMPLES:
        raise ValueError(
            f"{trace.label!r} has {int(np.sum(keep))} finite die samples in the first "
            f"{window_s:g} s, fewer than the {MIN_WINDOW_SAMPLES} required"
        )
    die_finite, time_finite = die[keep], time_s[keep]
    whole = fit_linear(time_finite, die_finite)
    midpoint = 0.5 * (float(time_finite[0]) + float(time_finite[-1]))
    second = time_finite >= midpoint
    if int(np.sum(second)) >= 3:
        half = fit_linear(time_finite[second], die_finite[second])
        slope_second = half.slope
    else:
        slope_second = whole.slope
    ratio = slope_second / whole.slope if whole.slope != 0.0 else math.nan

    clock = window.mean_cpu_khz
    clock_finite = clock[np.isfinite(clock)]
    clock_mean = float(clock_finite.mean()) if clock_finite.size else math.nan
    clock_range = (
        float(np.ptp(clock_finite)) / float(clock_finite.max())
        if clock_finite.size and float(clock_finite.max()) > 0.0
        else math.nan
    )
    duty = window.data["achieved_duty"]
    duty_finite = duty[np.isfinite(duty)]
    return np.array(
        [
            window_s,
            float(duty_finite.mean()) if duty_finite.size else math.nan,
            float(die_finite[0]),
            float(die_finite[-1]),
            float(die_finite.mean()),
            float(die_finite[-1] - die_finite[0]),
            whole.slope,
            slope_second,
            ratio,
            clock_mean,
            clock_range,
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class Dataset:
    """Warm-up features with measured steady-state targets, grouped by run.

    Attributes
    ----------
    features:
        ``(n_rows, len(FEATURE_NAMES))``.
    die_target_c:
        Measured equilibrium-window mean die temperature of the row's run.
    throughput_target:
        Measured equilibrium-window mean throughput of the row's run.
    run, duty, window_s:
        Row metadata. ``run`` is the grouping key for every split.
    excluded:
        Run labels that contributed no rows, with the reason.
    origin:
        Meet over the contributing runs.
    """

    features: np.ndarray
    die_target_c: np.ndarray
    throughput_target: np.ndarray
    run: tuple[str, ...]
    duty: np.ndarray
    window_s: np.ndarray
    excluded: tuple[tuple[str, str], ...]
    origin: Origin

    @property
    def n_rows(self) -> int:
        """Rows in the dataset."""
        return int(self.features.shape[0])

    @property
    def runs(self) -> tuple[str, ...]:
        """Distinct run labels, sorted."""
        return tuple(sorted(set(self.run)))

    @property
    def duties(self) -> tuple[float, ...]:
        """Distinct commanded duty cycles, ascending."""
        return tuple(sorted({float(value) for value in self.duty}))


def build_dataset(
    traces: Sequence,
    *,
    window_lengths_s: Sequence[float] = DEFAULT_WINDOW_LENGTHS_S,
    equilibrium_window_s: float = DEFAULT_WINDOW_S,
    threshold_c_per_s: float = DEFAULT_SLOPE_THRESHOLD_C_PER_S,
) -> Dataset:
    """Build the supervised dataset from soak traces.

    A run is used only if it passes the equilibrium test, because an
    unequilibrated run has no measured steady state to be the target. Excluded
    runs are named in :attr:`Dataset.excluded`.

    Raises
    ------
    NotEnoughRunsError
        If fewer than two runs contribute rows.
    """
    rows: list[np.ndarray] = []
    die: list[float] = []
    throughput: list[float] = []
    run_labels: list[str] = []
    duties: list[float] = []
    windows: list[float] = []
    excluded: list[tuple[str, str]] = []
    origins: list[Origin] = []

    for trace in traces:
        verdict = equilibrium_test(
            trace, window_s=equilibrium_window_s, threshold_c_per_s=threshold_c_per_s
        )
        if not verdict.equilibrated:
            excluded.append((trace.label, f"{verdict.status}: {verdict.reason}"))
            continue
        tail = trace.tail(equilibrium_window_s)
        die_samples = tail.die_temperature_c[np.isfinite(tail.die_temperature_c)]
        throughput_samples = tail.throughput_iters_per_s[
            np.isfinite(tail.throughput_iters_per_s)
        ]
        if die_samples.size == 0 or throughput_samples.size == 0:
            excluded.append((trace.label, "equilibrium window has no finite target samples"))
            continue
        used_any = False
        for window_s in window_lengths_s:
            try:
                features = warmup_features(trace, window_s)
            except ValueError:
                continue
            if not np.all(np.isfinite(features)):
                continue
            rows.append(features)
            die.append(float(die_samples.mean()))
            throughput.append(float(throughput_samples.mean()))
            run_labels.append(trace.label)
            duties.append(float(trace.duty))
            windows.append(float(window_s))
            used_any = True
        if used_any:
            origins.append(trace.origin)
        else:
            excluded.append(
                (trace.label, "no warm-up window of the declared lengths was usable")
            )

    if len({label for label in run_labels}) < 2:
        raise NotEnoughRunsError(
            "the dataset has fewer than 2 contributing runs; excluded: "
            + "; ".join(f"{label} ({reason})" for label, reason in excluded)
        )
    return Dataset(
        features=np.vstack(rows),
        die_target_c=np.asarray(die, dtype=float),
        throughput_target=np.asarray(throughput, dtype=float),
        run=tuple(run_labels),
        duty=np.asarray(duties, dtype=float),
        window_s=np.asarray(windows, dtype=float),
        excluded=tuple(excluded),
        origin=meet(origins),
    )


@dataclass
class SteadyStatePredictor:
    """Group-bootstrapped ridge (or forest) predictor with an uncertainty output.

    Attributes
    ----------
    kind:
        ``"ridge"`` or ``"forest"``.
    alpha:
        Ridge penalty, ignored for the forest.
    n_models:
        Group-bootstrap fits for the ridge. The forest uses :data:`N_TREES`.
    seed:
        Seed for the resampling and for the forest.
    """

    kind: str = "ridge"
    alpha: float = RIDGE_ALPHA
    n_models: int = N_MODELS
    seed: int = SEED
    _models: list = field(default_factory=list, repr=False)
    _mean: np.ndarray | None = field(default=None, repr=False)
    _scale: np.ndarray | None = field(default=None, repr=False)
    _residual_sigma: float = field(default=math.nan, repr=False)
    _forest: object | None = field(default=None, repr=False)

    def fit(self, features: np.ndarray, target: np.ndarray, groups: Sequence[str]):
        """Fit on ``features`` and ``target``, resampling by ``groups``.

        Raises
        ------
        NotEnoughRunsError
            If fewer than two groups are present: a group bootstrap cannot
            resample one group, and an uncertainty from one group would be zero
            and would be a lie.
        ValueError
            On a non-finite entry, or an unknown ``kind``.
        """
        features = np.asarray(features, dtype=float)
        target = np.asarray(target, dtype=float)
        if features.ndim != 2 or features.shape[0] != target.size:
            raise ValueError(
                f"features {features.shape} and target {target.shape} do not agree"
            )
        if not (np.all(np.isfinite(features)) and np.all(np.isfinite(target))):
            raise ValueError("features and target must be finite; filter before fitting")
        unique_groups = sorted(set(groups))
        if len(unique_groups) < 2:
            raise NotEnoughRunsError(
                f"fitting needs at least 2 groups, got {len(unique_groups)}: "
                f"{unique_groups}"
            )
        if self.kind not in ("ridge", "forest"):
            raise ValueError(f"kind must be 'ridge' or 'forest', got {self.kind!r}")

        self._mean = features.mean(axis=0)
        spread = features.std(axis=0)
        spread[spread == 0.0] = 1.0
        self._scale = spread
        scaled = (features - self._mean) / self._scale

        if self.kind == "forest":
            from sklearn.ensemble import RandomForestRegressor

            forest = RandomForestRegressor(
                n_estimators=N_TREES, random_state=self.seed, n_jobs=1, min_samples_leaf=1
            )
            forest.fit(scaled, target)
            self._forest = forest
            predictions = np.column_stack(
                [tree.predict(scaled) for tree in forest.estimators_]
            )
            self._residual_sigma = float(np.std(target - predictions.mean(axis=1), ddof=1))
            return self

        from sklearn.linear_model import Ridge

        rng = np.random.default_rng(self.seed)
        index_by_group = {
            group: np.nonzero(np.asarray(groups) == group)[0] for group in unique_groups
        }
        self._models = []
        for _ in range(self.n_models):
            drawn = rng.integers(0, len(unique_groups), size=len(unique_groups))
            indices = np.concatenate([index_by_group[unique_groups[i]] for i in drawn])
            model = Ridge(alpha=self.alpha)
            model.fit(scaled[indices], target[indices])
            self._models.append(model)
        full = Ridge(alpha=self.alpha)
        full.fit(scaled, target)
        self._residual_sigma = float(np.std(target - full.predict(scaled), ddof=1))
        return self

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predictions and one-sigma uncertainties.

        Returns
        -------
        (prediction, sigma)
            ``sigma`` combines the spread across the ensemble (epistemic) with
            the in-sample residual spread (aleatoric) in quadrature. It is a
            one-sigma figure, not an interval, and the report layer multiplies
            it where an interval is wanted.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._mean is None or self._scale is None:
            raise RuntimeError("predict() called before fit()")
        features = np.atleast_2d(np.asarray(features, dtype=float))
        scaled = (features - self._mean) / self._scale
        if self.kind == "forest":
            if self._forest is None:
                raise RuntimeError("predict() called before fit()")
            stack = np.column_stack(
                [tree.predict(scaled) for tree in self._forest.estimators_]
            )
        else:
            if not self._models:
                raise RuntimeError("predict() called before fit()")
            stack = np.column_stack([model.predict(scaled) for model in self._models])
        mean = stack.mean(axis=1)
        epistemic = stack.std(axis=1, ddof=1)
        sigma = np.sqrt(epistemic**2 + self._residual_sigma**2)
        return mean, sigma


def rc_warmup_prediction(trace, window_s: float) -> float:
    """Baseline 2 on the SAME warm-up window: the fitted asymptote, degrees C.

    Returns ``nan`` when the relaxation fit refuses, which it does for a window
    too short or too flat to identify an asymptote. A refusal is reported as a
    refusal and is not replaced by a fallback value.
    """
    window = trace.window(0.0, window_s)
    try:
        fit = fit_exponential_relaxation(window.t_rel, window.die_temperature_c)
    except (RelaxationFitError, ValueError):
        return math.nan
    return fit.asymptote_c


@dataclass(frozen=True)
class SplitScore:
    """Errors of the learned model and of the RC baseline on one held-out group."""

    held_out: str
    n_rows: int
    learned_mae_c: float
    rc_mae_c: float
    learned_coverage: float
    rows: tuple[tuple[str, float, float, float, float, float], ...]

    @property
    def margin_c(self) -> float:
        """RC error minus learned error. Positive means the learned model is better."""
        return self.rc_mae_c - self.learned_mae_c


@dataclass(frozen=True)
class HeldOutDutyResult:
    """Leave-one-duty-out evaluation of the learned predictor against baseline 2.

    Attributes
    ----------
    splits:
        One :class:`SplitScore` per held-out duty cycle.
    learned_mae_c, rc_mae_c:
        Mean absolute error pooled over all held-out rows, degrees Celsius.
    margin_c:
        ``rc_mae_c - learned_mae_c``. Positive means the learned model is better.
    coverage:
        Fraction of held-out rows whose true steady state lies inside the
        learned prediction's plus-or-minus-two-sigma band. A calibrated
        two-sigma band would cover about 0.95.
    kind, origin:
        Which model, and the origin of the data.
    skipped:
        Duty cycles that could not be held out, with the reason: a duty whose
        removal leaves fewer than two training runs is not a usable split.
    """

    splits: tuple[SplitScore, ...]
    learned_mae_c: float
    rc_mae_c: float
    margin_c: float
    coverage: float
    kind: str
    origin: Origin
    skipped: tuple[tuple[float, str], ...]


def leave_one_duty_out(
    dataset: Dataset,
    traces: Sequence,
    *,
    kind: str = "ridge",
    alpha: float = RIDGE_ALPHA,
    n_models: int = N_MODELS,
    seed: int = SEED,
) -> HeldOutDutyResult:
    """Train on all duty cycles but one, predict the held-out one, per duty.

    The RC baseline is evaluated on exactly the same held-out rows and the same
    warm-up windows, so the two columns of the result are comparable.

    Raises
    ------
    NotEnoughRunsError
        If no duty cycle can be held out with at least two training runs
        remaining.
    """
    by_label = {trace.label: trace for trace in traces}
    splits: list[SplitScore] = []
    skipped: list[tuple[float, str]] = []
    learned_errors: list[float] = []
    rc_errors: list[float] = []
    covered: list[bool] = []

    for duty in dataset.duties:
        test = np.isclose(dataset.duty, duty)
        train = ~test
        train_runs = {label for label, keep in zip(dataset.run, train, strict=True) if keep}
        if len(train_runs) < 2:
            skipped.append(
                (duty, f"removing duty {duty:g} leaves {len(train_runs)} training run(s)")
            )
            continue
        predictor = SteadyStatePredictor(
            kind=kind, alpha=alpha, n_models=n_models, seed=seed
        )
        predictor.fit(
            dataset.features[train],
            dataset.die_target_c[train],
            [label for label, keep in zip(dataset.run, train, strict=True) if keep],
        )
        prediction, sigma = predictor.predict(dataset.features[test])
        truth = dataset.die_target_c[test]
        rows: list[tuple[str, float, float, float, float, float]] = []
        split_learned: list[float] = []
        split_rc: list[float] = []
        for position, index in enumerate(np.nonzero(test)[0]):
            label = dataset.run[index]
            window_s = float(dataset.window_s[index])
            rc = rc_warmup_prediction(by_label[label], window_s)
            learned_error = abs(float(prediction[position]) - float(truth[position]))
            rc_error = abs(rc - float(truth[position])) if math.isfinite(rc) else math.nan
            rows.append(
                (
                    label,
                    window_s,
                    float(truth[position]),
                    float(prediction[position]),
                    float(sigma[position]),
                    rc,
                )
            )
            split_learned.append(learned_error)
            if math.isfinite(rc_error):
                split_rc.append(rc_error)
                learned_errors.append(learned_error)
                rc_errors.append(rc_error)
            covered.append(
                abs(float(prediction[position]) - float(truth[position]))
                <= 2.0 * float(sigma[position])
            )
        splits.append(
            SplitScore(
                held_out=f"duty {duty:g}",
                n_rows=int(np.sum(test)),
                learned_mae_c=float(np.mean(split_learned)) if split_learned else math.nan,
                rc_mae_c=float(np.mean(split_rc)) if split_rc else math.nan,
                learned_coverage=float(np.mean(covered[-int(np.sum(test)) :])),
                rows=tuple(rows),
            )
        )
    if not splits:
        raise NotEnoughRunsError(
            "no duty cycle could be held out with 2 or more training runs remaining: "
            + "; ".join(f"{duty:g} ({reason})" for duty, reason in skipped)
        )
    learned_mae = float(np.mean(learned_errors)) if learned_errors else math.nan
    rc_mae = float(np.mean(rc_errors)) if rc_errors else math.nan
    return HeldOutDutyResult(
        splits=tuple(splits),
        learned_mae_c=learned_mae,
        rc_mae_c=rc_mae,
        margin_c=rc_mae - learned_mae,
        coverage=float(np.mean(covered)) if covered else math.nan,
        kind=kind,
        origin=dataset.origin,
        skipped=tuple(skipped),
    )


@dataclass(frozen=True)
class ModelVerdict:
    """Whether the learned margin is distinguishable from the baseline.

    Attributes
    ----------
    verdict:
        ``"NOT_DISTINGUISHABLE"``, ``"LEARNED_BETTER"`` or ``"BASELINE_BETTER"``.
    margin_c:
        ``rc_mae_c - learned_mae_c`` in degrees Celsius. Positive favours the
        learned model.
    reference_half_width_c:
        Half-width of baseline 3's run-to-run bootstrap interval on the steady
        state, degrees Celsius. The yardstick: a margin smaller than this is
        inside the noise of the quantity being predicted.
    ratio:
        ``|margin| / reference_half_width``.
    statement:
        One sentence stating the comparison as measured.
    """

    verdict: str
    margin_c: float
    reference_half_width_c: float
    ratio: float
    statement: str


def compare_against_rc(
    result: HeldOutDutyResult, reference: bs.BootstrapResult
) -> ModelVerdict:
    """Compare a learned margin against baseline 3's run-to-run interval.

    Parameters
    ----------
    result:
        Output of :func:`leave_one_duty_out`.
    reference:
        Baseline 3's bootstrap on the steady-state die temperature. Its
        half-width is the yardstick.

    Returns
    -------
    ModelVerdict
        ``NOT_DISTINGUISHABLE`` whenever ``|margin| <= half_width``, in either
        direction. This is the whole point of the function: a margin inside the
        run-to-run interval is not evidence, and no sign of it is reported as an
        improvement.
    """
    half_width = reference.half_width
    margin = result.margin_c
    if not (math.isfinite(margin) and math.isfinite(half_width)) or half_width <= 0.0:
        return ModelVerdict(
            verdict="NOT_DISTINGUISHABLE",
            margin_c=margin,
            reference_half_width_c=half_width,
            ratio=math.nan,
            statement=(
                "the margin or the reference interval is not finite, so no comparison is "
                "possible and nothing is claimed"
            ),
        )
    ratio = abs(margin) / half_width
    if ratio <= 1.0:
        return ModelVerdict(
            verdict="NOT_DISTINGUISHABLE",
            margin_c=margin,
            reference_half_width_c=half_width,
            ratio=ratio,
            statement=(
                f"the learned model's margin over the RC baseline is {margin:+.4f} degC, "
                f"which is {ratio:.3f} of the run-to-run bootstrap half-width "
                f"{half_width:.4f} degC. The margin is inside the interval, so the learned "
                f"model is NOT DISTINGUISHABLE from the baseline on this axis"
            ),
        )
    better = "learned model" if margin > 0.0 else "RC baseline"
    return ModelVerdict(
        verdict="LEARNED_BETTER" if margin > 0.0 else "BASELINE_BETTER",
        margin_c=margin,
        reference_half_width_c=half_width,
        ratio=ratio,
        statement=(
            f"the {better} is better by {abs(margin):.4f} degC mean absolute error, which is "
            f"{ratio:.3f} times the run-to-run bootstrap half-width {half_width:.4f} degC"
        ),
    )
