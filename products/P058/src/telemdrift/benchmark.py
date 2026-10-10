"""The benchmark harness: one declared experiment configuration, used everywhere.

This module is the only place the experiment's sizes and seeds are chosen, so
every figure, table and validation script in this repository shares them and
no two numbers in the README were measured under different conditions.

Compute budget, measured not estimated
--------------------------------------
Two cores (``os.cpu_count() == 2``, ``len(os.sched_getaffinity(0)) == 2``),
shared with sibling build agents. The sizes in :class:`BenchmarkConfig` were
chosen from a measured cost probe, not guessed: the per-sample cost of the
detectors was measured at 0.5 us (CUSUM), 2.2 us (ADWIN) and 26 us per KS
evaluation. The configuration below is what fits, and every ARL figure derived
from it is published with its Monte Carlo standard error rather than as a point
estimate, because at these sizes the standard error is of order 3-5 % and
quoting four significant figures would be a lie about precision.

A smaller experiment with honest error bars is worth more than a larger one that
did not finish. See ``validation/VALIDATION.md`` section 1 for the measured
wall-clock of each script.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .detectors import (
    ADWIN,
    ANALYTIC_DETECTORS,
    CUSUM,
    EWMA,
    Detector,
    PageHinkley,
    WindowedKS,
)
from .features import WINDOW
from .learned import LearnedDetector, score_stream
from .scoring import (
    ARL0Result,
    ARL1Result,
    TradeoffPoint,
    blind_fraction,
    measure_arl0,
    measure_arl1,
    wilson_interval,
)
from .streams import ChangeSpec, ar1_stationary, change_stream, stationary
from .thresholds import CalibrationResult, calibrate_threshold

__all__ = [
    "BenchmarkConfig",
    "DETECTOR_LABELS",
    "STANDARD",
    "SCORED_CHANGES",
    "analytic_factory",
    "ar1_stream_fn",
    "blind_fraction_from_scores",
    "blind_fraction_table",
    "calibrate_all_analytic",
    "change_stream_fn",
    "calibrate_learned_threshold",
    "default_threshold_operating_points",
    "learned_arl0",
    "learned_arl1",
    "measure_change_response",
    "tradeoff_curve",
    "transient_response",
]

#: Display labels, keyed as the CLI and every table key them.
DETECTOR_LABELS = {
    "cusum": "CUSUM",
    "page_hinkley": "Page-Hinkley",
    "ewma": "EWMA",
    "ks": "Windowed KS",
    "adwin": "ADWIN",
    "learned": "Learned RF",
}

#: The scored change types with the magnitude each is scored at. The transient
#: is in this list as the negative control: an alarm on it is a false alarm.
SCORED_CHANGES: tuple[tuple[str, ChangeSpec], ...] = (
    ("mean_step_1.0", ChangeSpec("mean_step", 1.0)),
    ("mean_step_0.5", ChangeSpec("mean_step", 0.5)),
    ("variance_step_2.0", ChangeSpec("variance_step", 2.0)),
    ("drift_ramp_0.02", ChangeSpec("drift_ramp", 0.02)),
    ("transient_4.0x20", ChangeSpec("transient", 4.0, 20)),
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Every size and seed the benchmark uses. Nothing is chosen elsewhere."""

    target_arl0: float = 500.0
    #: Seeds used to fit thresholds. Disjoint from ``eval_seeds`` by construction.
    cal_seeds: tuple[int, ...] = (58_101, 58_102, 58_103, 58_104, 58_105, 58_106)
    #: Seeds used to report achieved ARL0. Never used for fitting.
    eval_seeds: tuple[int, ...] = (58_201, 58_202, 58_203, 58_204, 58_205, 58_206,
                                   58_207, 58_208)
    cal_length: int = 30_000
    eval_length: int = 50_000
    #: Reduced ARL0 budget used at each point of the trade-off sweep.
    sweep_length: int = 25_000
    sweep_seeds: tuple[int, ...] = (58_301, 58_302, 58_303, 58_304)
    #: ARL1 replicates. One seeded stream per replicate.
    arl1_replicates: int = 300
    arl1_seed_base: int = 58_400
    #: Stationary samples before the change. Set to two target ARL0s so the
    #: detector reaches its steady-state distribution before the change arrives,
    #: and comfortably above the longest detector warm-up (300 samples: the
    #: windowed KS test's 200-sample reference plus 100-sample detection window).
    pre_length: int = 1_000
    #: Post-change samples. Delays are right-censored here.
    arl1_budget: int = 1_500
    window: int = WINDOW

    def arl1_seeds(self, offset: int = 0) -> tuple[int, ...]:
        """Seeds for an ARL1 experiment, offset so distinct experiments differ."""
        base = self.arl1_seed_base + 1_000_000 * offset
        return tuple(base + i for i in range(self.arl1_replicates))

    @property
    def seeds_disjoint(self) -> bool:
        return not (set(self.cal_seeds) & set(self.eval_seeds))


#: The one configuration every script in this repository uses.
STANDARD = BenchmarkConfig()


def analytic_factory(key: str, threshold: float):
    """Return a zero-argument factory for an analytic detector at ``threshold``."""
    if key == "cusum":
        return lambda: CUSUM(h=threshold)
    if key == "page_hinkley":
        return lambda: PageHinkley(lambda_=threshold)
    if key == "ewma":
        return lambda: EWMA(L=threshold)
    if key == "ks":
        return lambda: WindowedKS(c=threshold)
    if key == "adwin":
        return lambda: ADWIN(delta=threshold)
    raise ValueError(f"unknown analytic detector {key!r}; expected {ANALYTIC_DETECTORS}")


def _default_threshold(key: str) -> float:
    return {
        "cusum": CUSUM,
        "page_hinkley": PageHinkley,
        "ewma": EWMA,
        "ks": WindowedKS,
        "adwin": ADWIN,
    }[key].default_threshold()


def _threshold_clip(key: str) -> tuple[float, float] | None:
    if key == "adwin":
        return (1e-12, 0.999_999)
    if key == "ks":
        return (1e-6, 1.0)
    return None


def stationary_stream_fn(length: int, seed: int) -> np.ndarray:
    """``stream_fn`` for the i.i.d. Gaussian stationary stream."""
    return stationary(length, seed)


def ar1_stream_fn(phi: float):
    """``stream_fn`` factory for an AR(1) stationary stream of lag-one ``phi``."""

    def fn(length: int, seed: int) -> np.ndarray:
        return ar1_stationary(length, seed, phi=phi)

    return fn


def change_stream_fn(spec: ChangeSpec):
    """``stream_fn`` factory for :func:`telemdrift.scoring.measure_arl1`."""

    def fn(pre_length: int, post_length: int, seed: int):
        return change_stream(pre_length, post_length, spec, seed)

    return fn


def calibrate_all_analytic(
    config: BenchmarkConfig = STANDARD,
    target_arl0: float | None = None,
    keys=ANALYTIC_DETECTORS,
    stream_fn=stationary_stream_fn,
) -> dict[str, CalibrationResult]:
    """Calibrate every analytic detector to one target ARL0.

    Returns a mapping ``key -> CalibrationResult`` in the order of ``keys``.
    """
    target = config.target_arl0 if target_arl0 is None else float(target_arl0)
    out: dict[str, CalibrationResult] = {}
    for key in keys:
        out[key] = calibrate_threshold(
            name=DETECTOR_LABELS[key],
            factory_from_threshold=lambda th, k=key: analytic_factory(k, th)(),
            default_threshold=_default_threshold(key),
            stream_fn=stream_fn,
            target_arl0=target,
            calibration_seeds=config.cal_seeds,
            evaluation_seeds=config.eval_seeds,
            calibration_length=config.cal_length,
            evaluation_length=config.eval_length,
            clip=_threshold_clip(key),
        )
    return out


# --------------------------------------------------------------------------
# Learned-detector fast paths.
#
# The forest costs milliseconds per single-row call and microseconds per row in
# a batch. A score series is a causal function of the stream alone, so scoring
# the whole stream once and then simulating the alarm-plus-refractory logic over
# the score series gives results identical to the online path at a fraction of
# the cost. tests/test_learned.py pins the identity on real streams; the
# speed-up is what makes the learned detector affordable on two cores at all.
# --------------------------------------------------------------------------


def _run_lengths_from_scores(scores: np.ndarray, p: float, window: int) -> tuple[list[int], int]:
    """Run lengths of the learned detector from a score series.

    Mirrors :meth:`LearnedDetector.alarms_from_scores` exactly, including the
    warm-up and the post-alarm refractory period, both ``window - 1`` samples.
    The two must stay identical: ``tests/test_benchmark.py`` compares them on
    real streams.
    """
    lengths: list[int] = []
    since = 0
    refractory = window - 1
    for t in range(scores.size):
        since += 1
        if refractory > 0:
            refractory -= 1
            continue
        if scores[t] > p:
            lengths.append(since)
            since = 0
            refractory = window - 1
    return lengths, since


def learned_arl0(
    model,
    p_threshold: float,
    config: BenchmarkConfig = STANDARD,
    seeds=None,
    stream_length: int | None = None,
    stream_fn=stationary_stream_fn,
    cached_scores: list[np.ndarray] | None = None,
) -> ARL0Result:
    """ARL0 of the learned detector, via the batch score path.

    ``cached_scores`` lets a calibration sweep reuse one scoring pass over all
    streams for every candidate ``p_threshold``, which is the difference between
    a 2-second calibration and a 2-minute one.
    """
    seeds = list(config.eval_seeds if seeds is None else seeds)
    length = config.eval_length if stream_length is None else stream_length
    if cached_scores is None:
        cached_scores = [
            score_stream(model, stream_fn(length, s), config.window) for s in seeds
        ]
    all_lengths: list[int] = []
    censored = 0
    for scores in cached_scores:
        lengths, tail = _run_lengths_from_scores(scores, p_threshold, config.window)
        all_lengths.extend(lengths)
        censored += tail
    total = sum(s.size for s in cached_scores)
    if not all_lengths:
        return ARL0Result(float(total), float("nan"), 0, total, censored,
                          np.asarray([], dtype=float))
    arr = np.asarray(all_lengths, dtype=float)
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
    return ARL0Result(float(arr.mean()), sem, arr.size, total, censored, arr)


def calibrate_learned_threshold(
    model,
    config: BenchmarkConfig = STANDARD,
    target_arl0: float | None = None,
    stream_fn=stationary_stream_fn,
    grid_size: int = 97,
) -> CalibrationResult:
    """Choose ``p*`` so the learned detector's measured ARL0 matches the target.

    A grid search over ``p*`` is used instead of bisection because the score
    series is cached: all ``grid_size`` candidate thresholds cost one scoring
    pass between them. The grid is in score space, ``(0, 1)``, and the chosen
    ``p*`` is the grid point whose calibration ARL0 is closest to target in log
    space. Achieved ARL0 is then measured on the disjoint evaluation seeds.
    """
    target = config.target_arl0 if target_arl0 is None else float(target_arl0)
    cal_scores = [
        score_stream(model, stream_fn(config.cal_length, s), config.window)
        for s in config.cal_seeds
    ]
    grid = np.linspace(0.01, 0.99, grid_size)
    best_p = float(grid[0])
    best_err = float("inf")
    best_arl0 = float("nan")
    for p in grid:
        res = learned_arl0(model, float(p), config, cached_scores=cal_scores)
        if res.n_runs == 0:
            continue
        err = abs(np.log(max(res.arl0, 1e-9)) - np.log(target))
        if err < best_err:
            best_err, best_p, best_arl0 = err, float(p), res.arl0
    achieved = learned_arl0(
        model, best_p, config, seeds=config.eval_seeds,
        stream_length=config.eval_length, stream_fn=stream_fn,
    )
    return CalibrationResult(
        detector=DETECTOR_LABELS["learned"],
        threshold=best_p,
        target_arl0=target,
        calibration_arl0=best_arl0,
        achieved=achieved,
        iterations=grid_size,
        bracket=(float(grid[0]), float(grid[-1])),
        bracketing_failed=not np.isfinite(best_err),
    )


def learned_arl1(
    model,
    p_threshold: float,
    spec: ChangeSpec,
    config: BenchmarkConfig = STANDARD,
    seed_offset: int = 0,
    replicates: int | None = None,
) -> ARL1Result:
    """Detection delay of the learned detector, steady-state convention.

    Uses the batch score path: one scoring pass per replicate, then the same
    alarm-plus-refractory simulation the online detector runs. Identical result,
    affordable cost.
    """
    seeds = list(config.arl1_seeds(seed_offset))
    if replicates is not None:
        seeds = seeds[:replicates]
    det = LearnedDetector(model, p_threshold, config.window)
    delays: list[int] = []
    pre_alarm_replicates = 0
    censored = 0
    unarmed = 0
    for s in seeds:
        stream, idx = change_stream(config.pre_length, config.arl1_budget, spec, s)
        scores = score_stream(model, stream, config.window)
        alarms = det.alarms_from_scores(scores)
        pre = alarms[alarms < idx]
        post = alarms[alarms >= idx]
        if pre.size:
            pre_alarm_replicates += 1
            # Un-armed at the change if the last pre-change alarm is within one
            # window of it: the refractory period has not expired.
            if idx - int(pre[-1]) <= config.window:
                unarmed += 1
        if post.size == 0:
            delays.append(config.arl1_budget)
            censored += 1
        else:
            delays.append(int(post[0]) - idx)
    arr = np.asarray(delays, dtype=float)
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
    return ARL1Result(float(arr.mean()), sem, arr.size, len(seeds), pre_alarm_replicates,
                      censored, config.arl1_budget, arr, unarmed)


# --------------------------------------------------------------------------
# Change response and trade-off curve.
# --------------------------------------------------------------------------


def measure_change_response(
    key: str,
    threshold: float,
    spec: ChangeSpec,
    config: BenchmarkConfig = STANDARD,
    seed_offset: int = 0,
    replicates: int | None = None,
    model=None,
) -> ARL1Result:
    """ARL1 for any detector key, analytic or learned, on one change spec."""
    if key == "learned":
        if model is None:
            raise ValueError("model is required for the learned detector")
        return learned_arl1(model, threshold, spec, config, seed_offset, replicates)
    seeds = list(config.arl1_seeds(seed_offset))
    if replicates is not None:
        seeds = seeds[:replicates]
    return measure_arl1(
        analytic_factory(key, threshold),
        change_stream_fn(spec),
        seeds,
        config.pre_length,
        config.arl1_budget,
    )


def tradeoff_curve(
    key: str,
    thresholds,
    spec: ChangeSpec,
    config: BenchmarkConfig = STANDARD,
    model=None,
    replicates: int = 200,
) -> list[TradeoffPoint]:
    """Measure (ARL0, ARL1) at each threshold: the delay-vs-false-alarm curve.

    This is the figure that makes the detectors comparable. A detector is better
    than another only if its curve lies below and to the right of it; a single
    (ARL0, ARL1) pair compares nothing, and a comparison at default thresholds
    compares two arbitrary points on two different curves.
    """
    points: list[TradeoffPoint] = []
    cached: list[np.ndarray] | None = None
    if key == "learned":
        if model is None:
            raise ValueError("model is required for the learned detector")
        cached = [
            score_stream(model, stationary(config.sweep_length, s), config.window)
            for s in config.sweep_seeds
        ]
    for th in thresholds:
        if key == "learned":
            a0 = learned_arl0(model, float(th), config, cached_scores=cached)
        else:
            a0 = measure_arl0(
                analytic_factory(key, float(th)),
                stationary_stream_fn,
                config.sweep_seeds,
                config.sweep_length,
            )
        a1 = measure_change_response(
            key, float(th), spec, config, seed_offset=0, replicates=replicates, model=model
        )
        points.append(TradeoffPoint(DETECTOR_LABELS[key], float(th), a0, a1))
    return points


def default_threshold_operating_points(
    config: BenchmarkConfig = STANDARD,
    keys=ANALYTIC_DETECTORS,
) -> dict[str, ARL0Result]:
    """Measured ARL0 of each detector **at its own declared default threshold**.

    This is the number that shows why comparing at defaults is the usual error:
    the five defaults do not land on one false-alarm rate, they land decades
    apart.
    """
    out: dict[str, ARL0Result] = {}
    for key in keys:
        out[key] = measure_arl0(
            analytic_factory(key, _default_threshold(key)),
            stationary_stream_fn,
            config.eval_seeds,
            config.eval_length,
        )
    return out


def timed(fn, *args, **kwargs):
    """Run ``fn`` and return ``(result, seconds)``. Wall clock only, no CPU time.

    Wall-clock figures in this container move 10-40 % between runs under
    contention; nothing in this repository quotes a single-run microsecond
    figure in prose without saying so.
    """
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


@dataclass
class DetectorRow:
    """One row of a detector comparison table, for text and JSON reporting."""

    key: str
    label: str
    threshold: float
    arl0: float
    arl0_sem: float
    results: dict[str, ARL1Result] = field(default_factory=dict)


def detector_instance(key: str, threshold: float, model=None) -> Detector:
    """Construct any detector by key, including the learned one."""
    if key == "learned":
        if model is None:
            raise ValueError("model is required for the learned detector")
        return LearnedDetector(model, threshold)
    return analytic_factory(key, threshold)()


def blind_fraction_from_scores(scores: np.ndarray, p: float, window: int) -> float:
    """Un-armed fraction of the learned detector, from a precomputed score series.

    The online path would need one forest call per sample, which this container
    cannot afford (measured at milliseconds per call). The logic is the same as
    :func:`telemdrift.scoring.blind_fraction` applied to
    :meth:`LearnedDetector.alarms_from_scores`.
    """
    s = np.asarray(scores, dtype=float)
    blind = 0
    refractory = window - 1
    for t in range(s.size):
        if refractory > 0:
            blind += 1
            refractory -= 1
            continue
        if s[t] > p:
            refractory = window - 1
    return blind / s.size if s.size else 0.0


def blind_fraction_table(
    thresholds: dict[str, float],
    config: BenchmarkConfig = STANDARD,
    seeds=None,
    stream_length: int = 30_000,
    model=None,
) -> dict[str, float]:
    """Measured un-armed fraction on a stationary stream, per detector.

    ``thresholds`` maps a detector key to its calibrated scalar, so the figure
    is reported at the operating point the detector is actually compared at.
    The learned detector goes through the batch score path for cost; every other
    detector runs its real online loop.
    """
    seeds = list(config.eval_seeds[:2] if seeds is None else seeds)
    out: dict[str, float] = {}
    for key, th in thresholds.items():
        total = 0.0
        for s in seeds:
            stream = stationary(stream_length, s)
            if key == "learned":
                if model is None:
                    raise ValueError("model is required for the learned detector")
                total += blind_fraction_from_scores(
                    score_stream(model, stream, config.window), th, config.window
                )
            else:
                total += blind_fraction(detector_instance(key, th, model=model), stream)
        out[key] = total / len(seeds)
    return out


def transient_response(
    key: str,
    threshold: float,
    amplitude: float,
    duration: int,
    config: BenchmarkConfig = STANDARD,
    replicates: int = 300,
    grace: int = 50,
    seed_offset: int = 2,
    model=None,
) -> dict[str, float]:
    """Firing rate inside a transient excursion against a matched stationary baseline.

    A transient is a negative control: the channel recovers, so there is no
    change and every alarm is a false alarm. But a detector calibrated to
    ARL0 = 500 will false-alarm inside *any* 70-sample window about 13 % of the
    time, so a raw firing rate on a transient stream says nothing on its own.
    This function therefore measures both:

    * ``transient_rate`` -- fraction of replicates alarming in
      ``[change_index, change_index + duration + grace)`` on a stream containing
      the excursion;
    * ``baseline_rate`` -- the same measurement on a stationary stream with the
      same seeds and no excursion;

    and reports ``excess = transient_rate - baseline_rate``, which is the part
    attributable to the transient. ``excess`` near zero means the detector is
    genuinely indifferent to the excursion; ``excess`` near ``1 - baseline``
    means it treats the transient exactly as it would a real change.

    Returns
    -------
    dict
        Keys ``transient_fired``, ``transient_rate``, ``baseline_fired``,
        ``baseline_rate``, ``excess``, ``replicates``, ``window``, plus Wilson
        interval bounds ``transient_lo``/``transient_hi`` and
        ``baseline_lo``/``baseline_hi``.
    """
    seeds = list(config.arl1_seeds(seed_offset))[:replicates]
    spec = ChangeSpec("transient", amplitude, duration)
    window = duration + grace

    def fired_in_window(stream: np.ndarray, idx: int) -> bool:
        if key == "learned":
            if model is None:
                raise ValueError("model is required for the learned detector")
            det = LearnedDetector(model, threshold, config.window)
            alarms = det.alarms_from_scores(score_stream(model, stream, config.window))
            return bool(np.any((alarms >= idx) & (alarms < idx + window)))
        det = analytic_factory(key, threshold)()
        det.reset()
        external = not getattr(det, "self_resetting", False)
        for i in range(idx):
            if det.update(stream[i]) and external:
                det.reset()
        for i in range(idx, min(idx + window, stream.size)):
            if det.update(stream[i]):
                return True
        return False

    t_fired = 0
    b_fired = 0
    for s in seeds:
        stream, idx = change_stream(config.pre_length, config.arl1_budget, spec, s)
        if fired_in_window(stream, idx):
            t_fired += 1
        base = stationary(config.pre_length + config.arl1_budget, s + 500_000)
        if fired_in_window(base, config.pre_length):
            b_fired += 1
    n = len(seeds)
    t_lo, t_hi = wilson_interval(t_fired, n)
    b_lo, b_hi = wilson_interval(b_fired, n)
    return {
        "transient_fired": t_fired,
        "transient_rate": t_fired / n,
        "transient_lo": t_lo,
        "transient_hi": t_hi,
        "baseline_fired": b_fired,
        "baseline_rate": b_fired / n,
        "baseline_lo": b_lo,
        "baseline_hi": b_hi,
        "excess": (t_fired - b_fired) / n,
        "replicates": n,
        "window": window,
    }
