"""Baselines 1, 2 and 3, and the headline number they exist to produce.

Build order is the specification's and is deliberate: the classical baselines
are implemented and reported before any learned model exists, and the headline
of this product is a measured difference between two of them, whatever that
difference turns out to be.

- **Baseline 1, cold-burst extrapolation** (:func:`burst_baseline`). Sustained
  throughput is taken equal to the throughput of a short run from a cold board.
  This is what a datasheet figure and a bench benchmark produce, and it is
  implemented here as the thing being tested, not as a strawman: it gets the
  same equilibrium test, the same sample hygiene and the same bootstrap
  treatment as baseline 3.
- **Baseline 2, first-order thermal RC network** (:func:`rc_baseline`). Fit the
  relaxation to the measured die temperature, take the asymptote as the
  steady-state die temperature, map that to a steady-state clock through the
  MEASURED clock-versus-temperature relation, and map clock to throughput
  through the declared proportionality anchored on the burst runs. Tau is also
  estimated independently from the cooldown traces and both values are
  reported; neither is picked over the other.
- **Baseline 3, ground truth** (:func:`sustained_truth`). The directly measured
  sustained throughput and die temperature over the equilibrium window of the
  duty-1.0 soaks, with a two-level bootstrap interval across runs and within
  runs. A run that fails the equilibrium test is EXCLUDED and named, not
  averaged.

**The headline** (:func:`overstatement`) is baseline 1 against baseline 3: the
sustained throughput as a fraction of the burst figure, and the burst figure's
overstatement of sustained throughput, both from the same resample draws so the
two agree. **If the interval on the overstatement contains zero, the finding is
that sustained throughput is statistically indistinguishable from burst
throughput on this board with this payload, and that is reported as the
finding.** There is no default, no tolerance and no code path in this module
that treats a zero gap as a failure to be explained away.

Sample hygiene, declared once. The first :data:`BURST_SKIP_S` seconds of every
run are dropped before any throughput statistic. The reason is visible in the
measured traces: every one of the EIGHT runs with a non-zero duty cycle reports
a first one-second sample below that run's own post-skip mean, because the six
worker processes are still starting during that first second and flush their
counters into shared memory only every 0.25 s, so the first sample accounts for
less work than was actually running.

Measured first sample against post-skip mean, iterations per second, all eight
load runs:

===============  ============  ================  =======
run              first sample  post-skip mean    deficit
===============  ============  ================  =======
burst_1.00_a         19447.34          26077.93   25.4 %
burst_1.00_b         19251.80          26242.87   26.6 %
soak_0.25             2906.69           3032.94    4.2 %
soak_0.50             8537.74          10317.68   17.3 %
soak_0.75            14738.28          19447.17   24.2 %
soak_1.00_r1         19138.60          25727.20   25.6 %
soak_1.00_r2         19011.22          25741.82   26.1 %
soak_1.00_r3         19550.24          25827.46   24.3 %
===============  ============  ================  =======

The deficit is NOT constant across duty cycles: it is about a quarter at duty
0.75 and above, 17.3 per cent at duty 0.50 and only 4.2 per cent at duty 0.25,
which is what a fixed start-up cost looks like when it is amortised against a
smaller amount of work per window. The skip is justified in every case, and the
figure to quote is the table and not a single number. That first sample is an
artefact of the harness, not a measurement of the payload.

Two corrections, both dated 2026-11-12, both recorded rather than quietly
applied.

1. By the coordinating session: this docstring originally stated that the first
   row of the measured cold-burst trace carries ``iters_cum = 0``. It does not;
   the measured value is 19468 for ``burst_1.00_a`` and 19272 for
   ``burst_1.00_b``. That sentence described data that had not been seen when
   it was written.
2. By the build session, on re-measuring: the replacement text said the deficit
   was "consistently near 25 per cent" and appeared in "all six load runs".
   There are eight non-zero-duty runs, not six, and the deficit is 4.2 per cent
   at duty 0.25, which is not near 25 per cent. The table above is the measured
   result and replaces both claims.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from . import bootstrap as bs
from .clock_throughput import (
    ClockTemperatureRelation,
    ProportionalityExponent,
    clock_temperature_relation,
    proportionality_exponent,
)
from .equilibrium import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_SLOPE_THRESHOLD_C_PER_S,
    DEFAULT_WINDOW_S,
    EquilibriumVerdict,
    equilibrium_test,
)
from .provenance import Origin, meet
from .rc_model import (
    RelaxationFit,
    RelaxationFitError,
    TauEstimate,
    fit_exponential_relaxation,
    tau_from_integral,
    tau_from_log_differences,
)

#: Seconds dropped from the start of every run before any throughput statistic.
BURST_SKIP_S: float = 2.0

#: Fewest finite throughput samples a burst run must contribute.
MIN_BURST_SAMPLES: int = 10

#: Seed for every bootstrap in this module unless overridden.
SEED: int = 910091


def _throughput_samples(trace, *, skip_s: float = BURST_SKIP_S) -> np.ndarray:
    """Finite ``iters_per_s`` samples of ``trace`` after the declared skip."""
    mask = trace.t_rel > skip_s
    values = trace.throughput_iters_per_s[mask]
    return values[np.isfinite(values)]


def _die_samples(trace, *, skip_s: float = BURST_SKIP_S) -> np.ndarray:
    """Finite die-temperature samples of ``trace`` after the declared skip."""
    mask = trace.t_rel > skip_s
    values = trace.die_temperature_c[mask]
    return values[np.isfinite(values)]


@dataclass(frozen=True)
class BurstBaseline:
    """Baseline 1: the cold-burst figure, with its own interval.

    Attributes
    ----------
    throughput:
        Two-level bootstrap of iterations per second across the burst runs.
    t_interval:
        ``(point, low, high)`` Student-t interval over the per-run means, as an
        independent cross-check on the bootstrap.
    per_run_mean:
        Mean throughput of each burst run.
    mean_die_c:
        Mean die temperature over the pooled burst samples, degrees Celsius.
    equilibrium:
        Equilibrium verdict for each burst run. A 20 s burst is expected to
        return ``INSUFFICIENT_SAMPLES``: it is nowhere near equilibrium, which is
        the premise of the whole product, and the test says so rather than this
        module asserting it.
    n_runs, skip_s, origin:
        Provenance of the figure.
    throughput_groups:
        The per-run sample arrays, kept so that :func:`overstatement` can
        resample exactly the samples this baseline averaged rather than a
        recomputed approximation of them.
    """

    throughput: bs.BootstrapResult
    t_interval: tuple[float, float, float]
    per_run_mean: dict[str, float]
    mean_die_c: float
    equilibrium: dict[str, EquilibriumVerdict]
    n_runs: int
    skip_s: float
    origin: Origin
    throughput_groups: tuple[np.ndarray, ...] = field(default=(), repr=False)

    @property
    def prediction_iters_per_s(self) -> float:
        """Baseline 1's prediction of SUSTAINED throughput: the burst figure itself."""
        return self.throughput.point


def burst_baseline(
    traces: Sequence,
    *,
    skip_s: float = BURST_SKIP_S,
    seed: int = SEED,
    n_resamples: int = bs.DEFAULT_RESAMPLES,
) -> BurstBaseline:
    """Baseline 1 over the given burst traces.

    Raises
    ------
    ValueError
        If fewer than two burst runs are given, or any contributes fewer than
        :data:`MIN_BURST_SAMPLES` finite samples after the skip. Two runs is
        already a weak basis for a run-level interval and the result says so
        through ``n_distinct_group_draws``; one is not a basis at all.
    """
    if len(traces) < 2:
        raise ValueError(
            f"baseline 1 needs at least 2 burst runs for a run-level interval, got "
            f"{len(traces)}"
        )
    groups = []
    per_run: dict[str, float] = {}
    for trace in traces:
        samples = _throughput_samples(trace, skip_s=skip_s)
        if samples.size < MIN_BURST_SAMPLES:
            raise ValueError(
                f"burst run {trace.label!r} contributes {samples.size} finite throughput "
                f"sample(s) after dropping the first {skip_s:g} s, fewer than the "
                f"{MIN_BURST_SAMPLES} required"
            )
        groups.append(samples)
        per_run[trace.label] = float(samples.mean())
    die = np.concatenate([_die_samples(trace, skip_s=skip_s) for trace in traces])
    return BurstBaseline(
        throughput=bs.bootstrap_cluster_mean(groups, seed=seed, n_resamples=n_resamples),
        t_interval=bs.t_interval_of_group_means(groups),
        per_run_mean=per_run,
        mean_die_c=float(die.mean()) if die.size else math.nan,
        equilibrium={trace.label: equilibrium_test(trace) for trace in traces},
        n_runs=len(traces),
        skip_s=skip_s,
        origin=meet([trace.origin for trace in traces]),
        throughput_groups=tuple(groups),
    )


@dataclass(frozen=True)
class SustainedTruth:
    """Baseline 3: directly measured sustained throughput and die temperature.

    Attributes
    ----------
    throughput, die_c:
        Two-level bootstrap across the included runs and within their
        equilibrium windows.
    t_interval_throughput:
        Independent Student-t interval over the per-run throughput means.
    within_run_throughput:
        Single-level bootstrap of the equilibrium-window samples of each
        included run, which is the within-run half of the spread the
        specification asks for, reported separately so the reader can see which
        level dominates.
    verdicts:
        Equilibrium verdict for every candidate run, included or not.
    included, excluded:
        Run labels, with the reason for each exclusion.
    window_s, threshold_c_per_s:
        The equilibrium criterion applied.
    origin:
        Meet over the included runs.
    """

    throughput: bs.BootstrapResult
    die_c: bs.BootstrapResult
    t_interval_throughput: tuple[float, float, float]
    within_run_throughput: dict[str, bs.BootstrapResult]
    verdicts: dict[str, EquilibriumVerdict]
    included: tuple[str, ...]
    excluded: tuple[tuple[str, str], ...]
    window_s: float
    threshold_c_per_s: float
    origin: Origin
    throughput_groups: tuple[np.ndarray, ...] = field(default=(), repr=False)


def sustained_truth(
    traces: Sequence,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    threshold_c_per_s: float = DEFAULT_SLOPE_THRESHOLD_C_PER_S,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    seed: int = SEED,
    n_resamples: int = bs.DEFAULT_RESAMPLES,
) -> SustainedTruth:
    """Baseline 3 over soak traces, excluding any run that is not equilibrated.

    Raises
    ------
    ValueError
        If fewer than two runs pass the equilibrium test. The verdicts are in
        the exception message by name, because "no sustained figure could be
        computed, and here is which run failed and by how much" is the honest
        outcome and is not the same as a zero.
    """
    verdicts: dict[str, EquilibriumVerdict] = {}
    groups: list[np.ndarray] = []
    die_groups: list[np.ndarray] = []
    included: list[str] = []
    excluded: list[tuple[str, str]] = []
    origins: list[Origin] = []
    for trace in traces:
        verdict = equilibrium_test(
            trace,
            window_s=window_s,
            threshold_c_per_s=threshold_c_per_s,
            min_samples=min_samples,
        )
        verdicts[trace.label] = verdict
        if not verdict.equilibrated:
            excluded.append((trace.label, f"{verdict.status}: {verdict.reason}"))
            continue
        tail = trace.tail(window_s)
        throughput = tail.throughput_iters_per_s
        throughput = throughput[np.isfinite(throughput)]
        die = tail.die_temperature_c
        die = die[np.isfinite(die)]
        if throughput.size < min_samples:
            excluded.append(
                (
                    trace.label,
                    f"equilibrated but only {throughput.size} finite throughput samples in "
                    f"the window, fewer than {min_samples}",
                )
            )
            continue
        groups.append(throughput)
        die_groups.append(die)
        included.append(trace.label)
        origins.append(trace.origin)

    if len(groups) < 2:
        raise ValueError(
            "baseline 3 needs at least 2 equilibrated runs for a run-level interval; "
            f"included {included}, excluded "
            + "; ".join(f"{label} ({reason})" for label, reason in excluded)
        )
    return SustainedTruth(
        throughput=bs.bootstrap_cluster_mean(groups, seed=seed, n_resamples=n_resamples),
        die_c=bs.bootstrap_cluster_mean(die_groups, seed=seed + 1, n_resamples=n_resamples),
        t_interval_throughput=bs.t_interval_of_group_means(groups),
        within_run_throughput={
            label: bs.bootstrap_mean(group, seed=seed + 2 + index, n_resamples=n_resamples)
            for index, (label, group) in enumerate(zip(included, groups, strict=True))
        },
        verdicts=verdicts,
        included=tuple(included),
        excluded=tuple(excluded),
        window_s=window_s,
        threshold_c_per_s=threshold_c_per_s,
        origin=meet(origins),
        throughput_groups=tuple(groups),
    )


@dataclass(frozen=True)
class RCBaseline:
    """Baseline 2: first-order RC network, with tau from rise and from cooldown.

    Attributes
    ----------
    rise_fits:
        Relaxation fit per heating run used.
    cooldown_fits:
        Relaxation fit per cooldown run that was usable.
    tau_rise_s, tau_cool_s:
        The two independent tau populations, seconds, reported side by side.
        Neither is preferred and neither is combined into a single number.
    independent_tau:
        Additional tau estimates from the log-difference and area routes on the
        primary rise, which share no algebra with the least-squares fit.
    predicted_die_c:
        Mean fitted asymptote over the heating runs, degrees Celsius: baseline
        2's prediction of the steady-state die temperature.
    predicted_die_spread_c:
        Half-range of the fitted asymptotes across runs, as a crude spread. With
        three runs there is no interval worth quoting and this is labelled a
        half-range, not a confidence interval.
    relation:
        The measured clock-versus-temperature relation.
    predicted_clock_khz, clock_degenerate, clock_extrapolating:
        The predicted steady-state clock and the two flags from
        :meth:`ClockTemperatureRelation.clock_at`.
    anchor_throughput_iters_per_s, anchor_clock_khz:
        The burst-run anchor the clock-to-throughput map is scaled from.
    predicted_throughput_iters_per_s:
        Baseline 2's prediction.
    exponent:
        The measured clock-to-throughput exponent, or a non-identifiable result.
    reduces_to_baseline_1:
        True when the clock relation is degenerate, so baseline 2 predicts the
        unthrottled clock and its prediction coincides with baseline 1's. This
        is a result about the board, and the report prints it as one.
    notes:
        Anything a reader needs in order not to over-read the prediction.
    origin:
        Meet over every trace used.
    """

    rise_fits: dict[str, RelaxationFit]
    cooldown_fits: dict[str, RelaxationFit]
    tau_rise_s: dict[str, float]
    tau_cool_s: dict[str, float]
    independent_tau: tuple[TauEstimate, ...]
    predicted_die_c: float
    predicted_die_spread_c: float
    relation: ClockTemperatureRelation
    predicted_clock_khz: float
    clock_degenerate: bool
    clock_extrapolating: bool
    anchor_throughput_iters_per_s: float
    anchor_clock_khz: float
    predicted_throughput_iters_per_s: float
    exponent: ProportionalityExponent
    reduces_to_baseline_1: bool
    notes: tuple[str, ...]
    origin: Origin

    @property
    def tau_rise_mean_s(self) -> float:
        """Mean of the per-run rise tau values, seconds."""
        values = [value for value in self.tau_rise_s.values() if math.isfinite(value)]
        return float(np.mean(values)) if values else math.nan

    @property
    def tau_cool_mean_s(self) -> float:
        """Mean of the per-run cooldown tau values, seconds."""
        values = [value for value in self.tau_cool_s.values() if math.isfinite(value)]
        return float(np.mean(values)) if values else math.nan

    @property
    def tau_ratio(self) -> float:
        """Cooldown tau divided by rise tau. One for a true single-pole system."""
        if not math.isfinite(self.tau_rise_mean_s) or self.tau_rise_mean_s == 0.0:
            return math.nan
        return self.tau_cool_mean_s / self.tau_rise_mean_s


def rc_baseline(
    heating: Sequence,
    cooldowns: Sequence,
    bursts: Sequence,
    relation_traces: Sequence | None = None,
    *,
    skip_s: float = BURST_SKIP_S,
) -> RCBaseline:
    """Baseline 2 from the heating, cooldown and burst traces.

    Parameters
    ----------
    heating:
        Loaded runs whose die-temperature rise the RC form is fitted to. The
        duty-1.0 soaks.
    cooldowns:
        Duty-0 runs after a loaded run, which carry the independent tau.
    bursts:
        Burst runs, used only as the anchor of the clock-to-throughput map. The
        anchor is deliberately NOT the soak equilibrium: anchoring on the
        quantity baseline 3 measures would make baseline 2's prediction
        partially circular.
    relation_traces:
        Traces pooled for the clock-versus-temperature relation. Defaults to
        every heating and burst trace given.

    Raises
    ------
    ValueError
        If no heating run yields a relaxation fit, or no burst anchor is usable.
    """
    if not heating:
        raise ValueError("baseline 2 needs at least one heating run to fit")
    if not bursts:
        raise ValueError("baseline 2 needs at least one burst run as the throughput anchor")

    notes: list[str] = []
    rise_fits: dict[str, RelaxationFit] = {}
    for trace in heating:
        try:
            rise_fits[trace.label] = fit_exponential_relaxation(
                trace.t_rel, trace.die_temperature_c
            )
        except RelaxationFitError as error:
            notes.append(f"rise fit refused for {trace.label}: {error}")
    if not rise_fits:
        raise ValueError(
            "no heating run produced a relaxation fit; baseline 2 has no steady-state "
            "temperature to predict from. " + " ".join(notes)
        )

    cooldown_fits: dict[str, RelaxationFit] = {}
    for trace in cooldowns:
        try:
            cooldown_fits[trace.label] = fit_exponential_relaxation(
                trace.t_rel, trace.die_temperature_c
            )
        except RelaxationFitError as error:
            notes.append(f"cooldown fit refused for {trace.label}: {error}")
    if not cooldown_fits:
        notes.append(
            "no cooldown trace yielded a relaxation fit, so the independent cooldown "
            "estimate of tau is NOT AVAILABLE for this campaign"
        )

    primary_label = sorted(rise_fits, key=lambda label: -rise_fits[label].n)[0]
    primary = next(trace for trace in heating if trace.label == primary_label)
    independent: list[TauEstimate] = []
    for estimator, keywords in (
        (tau_from_log_differences, {}),
        (tau_from_integral, {"asymptote_c": rise_fits[primary_label].asymptote_c}),
    ):
        try:
            independent.append(
                estimator(primary.t_rel, primary.die_temperature_c, **keywords)
            )
        except RelaxationFitError as error:
            notes.append(f"independent tau route refused on {primary_label}: {error}")

    asymptotes = [fit.asymptote_c for fit in rise_fits.values()]
    predicted_die = float(np.mean(asymptotes))
    spread = 0.5 * (max(asymptotes) - min(asymptotes)) if len(asymptotes) > 1 else math.nan

    pooled = list(relation_traces) if relation_traces is not None else [*heating, *bursts]
    relation = clock_temperature_relation(pooled)
    predicted_clock, degenerate, extrapolating = relation.clock_at(predicted_die)

    anchor_throughput_samples = np.concatenate(
        [_throughput_samples(trace, skip_s=skip_s) for trace in bursts]
    )
    anchor_clock_values = np.concatenate(
        [
            trace.mean_cpu_khz[np.isfinite(trace.mean_cpu_khz) & (trace.t_rel > skip_s)]
            for trace in bursts
        ]
    )
    if anchor_throughput_samples.size == 0 or anchor_clock_values.size == 0:
        raise ValueError(
            "the burst anchor has no finite throughput or clock samples after the "
            f"{skip_s:g} s skip"
        )
    anchor_throughput = float(anchor_throughput_samples.mean())
    anchor_clock = float(anchor_clock_values.mean())
    predicted_throughput = anchor_throughput * (predicted_clock / anchor_clock)

    exponent = proportionality_exponent([*heating, *bursts], duty=1.0)
    if degenerate:
        notes.append(
            "the clock-versus-temperature relation is NOT IDENTIFIABLE on this campaign, "
            "so baseline 2 predicts the unthrottled clock and its sustained-throughput "
            "prediction coincides with baseline 1's. That is a measured result about this "
            "board under this payload, not a failure of the fit"
        )
    if extrapolating:
        notes.append(
            f"the predicted steady-state die temperature {predicted_die:.2f} degC lies "
            f"OUTSIDE the measured temperature range "
            f"{relation.temperature_range_c[0]:.2f}--{relation.temperature_range_c[1]:.2f} "
            "degC; the predicted clock is an extrapolation of the fitted relation and is "
            "labelled unmeasured at that temperature"
        )
    if not exponent.identifiable:
        notes.append(
            "the clock-to-throughput exponent is not identifiable from this campaign; the "
            "map is the ASSUMED proportionality, stated in clock_throughput.py, and is not "
            "measured here"
        )
    return RCBaseline(
        rise_fits=rise_fits,
        cooldown_fits=cooldown_fits,
        tau_rise_s={label: fit.tau_s for label, fit in rise_fits.items()},
        tau_cool_s={label: fit.tau_s for label, fit in cooldown_fits.items()},
        independent_tau=tuple(independent),
        predicted_die_c=predicted_die,
        predicted_die_spread_c=spread,
        relation=relation,
        predicted_clock_khz=predicted_clock,
        clock_degenerate=degenerate,
        clock_extrapolating=extrapolating,
        anchor_throughput_iters_per_s=anchor_throughput,
        anchor_clock_khz=anchor_clock,
        predicted_throughput_iters_per_s=predicted_throughput,
        exponent=exponent,
        reduces_to_baseline_1=degenerate,
        notes=tuple(notes),
        origin=meet([trace.origin for trace in [*heating, *cooldowns, *bursts]]),
    )


@dataclass(frozen=True)
class Overstatement:
    """The headline: baseline 1 against baseline 3, with an interval.

    Attributes
    ----------
    sustained_fraction:
        Sustained throughput as a fraction of the burst figure, with a
        percentile interval. 1.0 means the burst figure is sustained.
    overstatement:
        ``burst / sustained - 1``: the fraction by which the burst figure
        overstates sustained throughput. 0.0 means it does not.
    verdict:
        ``"INDISTINGUISHABLE"`` when the overstatement interval contains zero,
        ``"OVERSTATES"`` when the whole interval is above zero,
        ``"UNDERSTATES"`` when the whole interval is below zero.
    burst_iters_per_s, sustained_iters_per_s:
        The two point estimates.
    n_resamples, confidence, seed:
        Reproducibility of the interval.
    origin:
        Meet of the two baselines.
    """

    sustained_fraction: bs.BootstrapResult
    overstatement: bs.BootstrapResult
    verdict: str
    burst_iters_per_s: float
    sustained_iters_per_s: float
    n_resamples: int
    confidence: float
    seed: int
    origin: Origin

    @property
    def statement(self) -> str:
        """One sentence stating the headline exactly as measured."""
        over = self.overstatement
        where = (
            "On the measured board with the declared payload"
            if self.origin is Origin.MEASURED
            else "On SYNTHETIC fixture data, which is not a measurement of any board"
        )
        if self.verdict == "INDISTINGUISHABLE":
            return (
                f"{where}, sustained throughput is statistically "
                f"indistinguishable from burst throughput: the cold-burst overstatement is "
                f"{100.0 * over.point:+.3f} per cent with a "
                f"{100.0 * self.confidence:g} per cent interval "
                f"[{100.0 * over.ci_low:+.3f}, {100.0 * over.ci_high:+.3f}] per cent, which "
                f"contains zero."
            )
        direction = "overstates" if self.verdict == "OVERSTATES" else "UNDERSTATES"
        return (
            f"{where}: the cold-burst figure {direction} sustained throughput by "
            f"{100.0 * over.point:+.3f} per cent, {100.0 * self.confidence:g} per cent "
            f"interval [{100.0 * over.ci_low:+.3f}, {100.0 * over.ci_high:+.3f}] per cent."
        )


def overstatement(
    burst_groups: Sequence[Sequence[float] | np.ndarray],
    sustained_groups: Sequence[Sequence[float] | np.ndarray],
    *,
    n_resamples: int = bs.DEFAULT_RESAMPLES,
    confidence: float = bs.DEFAULT_CONFIDENCE,
    seed: int = SEED,
    origin: Origin = Origin.SYNTHETIC,
) -> Overstatement:
    """The headline number, from the same resample draws for both forms.

    The two sides are resampled independently, which is the right model: a burst
    run and a soak run are separate executions and nothing pairs them. Both
    reported quantities are computed from the SAME pair of draw arrays, so
    ``sustained_fraction`` and ``overstatement`` are exact algebraic transforms
    of each other on every resample rather than two bootstraps that happen to
    nearly agree.

    The two reported INTERVALS are nevertheless not exact transforms of each
    other, and the reason is percentile interpolation rather than rounding:
    ``numpy.percentile`` interpolates linearly between the bracketing order
    statistics, and a monotone nonlinear transform does not commute with linear
    interpolation. Measured on this repository's throttled synthetic fixture the
    disagreement is about 1.6e-8 relative, against an interval width of order
    1e-2. ``tests/test_baselines.py`` asserts the agreement to 1e-6 relative and
    says why; it does not assert equality.

    Raises
    ------
    ValueError
        If either side has fewer than two non-empty groups, or any resample
        yields a non-positive throughput, which would make the ratio undefined.
    """
    burst_draws = bs.cluster_resampled_means(burst_groups, n_resamples=n_resamples, seed=seed)
    sustained_draws = bs.cluster_resampled_means(
        sustained_groups, n_resamples=n_resamples, seed=seed + 7919
    )
    if np.any(burst_draws <= 0.0) or np.any(sustained_draws <= 0.0):
        raise ValueError(
            "a resample produced a non-positive throughput; the ratio is undefined. "
            "This means at least one input group contains non-positive throughput "
            "samples, which an idle or cooldown run does"
        )
    burst_point = bs.observed_cluster_mean(burst_groups)
    sustained_point = bs.observed_cluster_mean(sustained_groups)
    if burst_point <= 0.0 or sustained_point <= 0.0:
        raise ValueError(
            f"observed burst {burst_point:g} and sustained {sustained_point:g} must both "
            "be positive for the ratio to be defined"
        )

    fraction_draws = sustained_draws / burst_draws
    overstatement_draws = burst_draws / sustained_draws - 1.0
    fraction_point = sustained_point / burst_point
    overstatement_point = burst_point / sustained_point - 1.0

    n_groups = min(len(burst_groups), len(sustained_groups))
    fraction_low, fraction_high = bs.percentile_interval(fraction_draws, confidence=confidence)
    over_low, over_high = bs.percentile_interval(overstatement_draws, confidence=confidence)
    fraction = bs.BootstrapResult(
        point=fraction_point,
        ci_low=fraction_low,
        ci_high=fraction_high,
        confidence=confidence,
        n_resamples=n_resamples,
        standard_error=float(np.std(fraction_draws, ddof=1)),
        bias=float(fraction_draws.mean() - fraction_point),
        n_groups=n_groups,
        n_samples=int(
            sum(len(group) for group in burst_groups)
            + sum(len(group) for group in sustained_groups)
        ),
        n_distinct_group_draws=n_groups**n_groups,
    )
    over = bs.BootstrapResult(
        point=overstatement_point,
        ci_low=over_low,
        ci_high=over_high,
        confidence=confidence,
        n_resamples=n_resamples,
        standard_error=float(np.std(overstatement_draws, ddof=1)),
        bias=float(overstatement_draws.mean() - overstatement_point),
        n_groups=n_groups,
        n_samples=fraction.n_samples,
        n_distinct_group_draws=n_groups**n_groups,
    )
    if over.ci_low <= 0.0 <= over.ci_high:
        verdict = "INDISTINGUISHABLE"
    elif over.ci_low > 0.0:
        verdict = "OVERSTATES"
    else:
        verdict = "UNDERSTATES"
    return Overstatement(
        sustained_fraction=fraction,
        overstatement=over,
        verdict=verdict,
        burst_iters_per_s=burst_point,
        sustained_iters_per_s=sustained_point,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
        origin=origin,
    )
