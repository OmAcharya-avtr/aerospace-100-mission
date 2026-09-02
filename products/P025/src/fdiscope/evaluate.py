"""Benchmark harness: detection delay, false-alarm rate, isolation, ROC.

The comparison protocol, fixed before any number was measured:

**False-alarm rate** is measured on dedicated fault-free runs over samples
``[FAR_START, n_steps)``, so no faulted sample can ever count as a false
alarm.  Two numbers are reported for every method: the per-sample alarm
fraction and the per-run probability of at least one alarm.  They answer
different questions and neither substitutes for the other.

**Detection delay** is the number of samples from fault onset to the first
alarm at or after onset, on faulted runs.  Alarms before onset are false
alarms and are never counted as fast detections.  Runs that never alarm are
reported as censored, with their count, and are excluded from the mean --
averaging an ``inf`` away silently would be the easiest possible way to fake a
good delay.

**Isolation** is evaluated on one window per run, ``[onset, onset + ISO_WINDOW)``,
identical for every method.  Both the GLR bank and the classifier see exactly
the same residuals.  This assumes the onset sample is known, which is an
idealisation shared by both methods; the misalignment sensitivity is measured
separately in ``validation/isolation_confusion.py``.

**ROC** uses window-level scores: positives are windows starting at fixed
offsets after onset on faulted runs, negatives are pre-onset windows from every
run.  Each method contributes one score per window and the curves are computed
by :func:`fdiscope.metrics.roc_curve`.

**Threshold calibration.** Detection delay can be bought with false alarms, so
comparing delays at different false-alarm rates is meaningless.  Every method
is therefore calibrated on dedicated *fault-free* runs to the same per-run
false-alarm probability before its delay is measured
(:func:`calibrate_all_thresholds`); no held-out run touches a threshold.

That calibration is not free, and the asymmetry it hides is itself a result:
the chi-squared and CUSUM thresholds also follow from their design formulas
(:func:`design_thresholds`) with **no data at all**, while the GLR bank and the
classifier have no such formula and cannot be operated without a fault-free
calibration set.  Both threshold sets are reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .analytic import chi2_threshold, cusum_threshold_for_arl0
from .classifier import FaultClassifier
from .detectors import ChiSquaredDetector, CusumBank, CusumDetector, detection_delay
from .faults import FAULT_CLASSES, FaultSpec, FaultType, class_index
from .features import _vectorised_features, feature_matrix
from .isolation import SignatureBank, build_signature_bank
from .plant import ControllerGains, PlantConfig
from .scenarios import Scenario, sample_scenario, sample_scenarios
from .simulate import LoopConfig, LoopRun, simulate_loop

__all__ = [
    "BenchmarkConfig",
    "MethodResult",
    "IsolationOutcome",
    "default_signature_specs",
    "build_cusum_bank",
    "run_scenarios",
    "harvest_training_rows",
    "sequential_alarms",
    "evaluate_detection",
    "window_scores",
    "evaluate_isolation",
    "calibrate_threshold",
    "calibrate_all_thresholds",
    "healthy_calibration_runs",
    "window_features_batch",
    "sequential_scores",
    "method_names",
    "design_thresholds",
    "default_scenario_sets",
    "build_default_bank",
    "class_labels",
    "FAR_START",
]

#: First sample used for the false-alarm measurement.  The loop starts at the
#: filter's steady-state covariance so there is no transient to skip, but the
#: first samples are kept out anyway so that every detector's window is full.
FAR_START: int = 300


@dataclass(frozen=True)
class BenchmarkConfig:
    """Fixed protocol parameters.

    Parameters
    ----------
    det_window : int
        Window of the short chi-squared detector [samples].
    iso_window : int
        Window used for isolation, the long chi-squared detector, the GLR bank
        and the classifier [samples].
    alpha : float
        Design per-window false-alarm probability of the chi-squared tests and
        the family-wise level of the GLR bank.
    cusum_mu : float
        Design mean shift of the channel CUSUMs, in residual sigmas.
    cusum_arl0 : float
        Design mean time between CUSUM false alarms [samples].
    plant : PlantConfig
    gains : ControllerGains
    roc_offsets : tuple of int
        Offsets after onset at which positive ROC windows start [samples].
    """

    det_window: int = 25
    iso_window: int = 100
    alpha: float = 1.0e-3
    cusum_mu: float = 1.0
    cusum_arl0: float = 2000.0
    plant: PlantConfig = field(default_factory=PlantConfig)
    gains: ControllerGains = field(default_factory=ControllerGains)
    roc_offsets: tuple[int, ...] = (0, 25, 50, 100)

    def __post_init__(self) -> None:
        if self.det_window < 3 or self.iso_window < 3:
            raise ValueError("windows must be at least 3 samples")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha}")


def default_signature_specs(plant: PlantConfig | None = None) -> dict[FaultType, FaultSpec]:
    """One representative fault per hypothesis, for signature construction.

    Magnitudes sit near the middle of the sampling ranges of
    :mod:`fdiscope.scenarios`.  The *direction* of a signature is what matters
    for the GLR bank, and for the additive sensor faults it does not depend on
    magnitude at all; for the multiplicative actuator faults it does, and that
    dependence is one of the measured limitations.
    """
    p = plant if plant is not None else PlantConfig()
    sigma_a = float(np.sqrt(p.attitude_var_rad2))
    sigma_r = float(np.sqrt(p.gyro_var_rad2_s2))
    return {
        FaultType.SENSOR_BIAS: FaultSpec(FaultType.SENSOR_BIAS, 0, 4.0 * sigma_a, 0),
        FaultType.SENSOR_DRIFT: FaultSpec(FaultType.SENSOR_DRIFT, 0, 0.15 * sigma_r, 1),
        FaultType.SENSOR_STUCK: FaultSpec(FaultType.SENSOR_STUCK, 0, 0.0, 0),
        FaultType.SENSOR_DROPOUT: FaultSpec(FaultType.SENSOR_DROPOUT, 0, 0.0, 0),
        FaultType.ACTUATOR_LOSS_OF_EFFECT: FaultSpec(
            FaultType.ACTUATOR_LOSS_OF_EFFECT, 0, 0.6, 0
        ),
        FaultType.ACTUATOR_STUCK: FaultSpec(FaultType.ACTUATOR_STUCK, 0, 0.0, 0),
        FaultType.ACTUATOR_RUNAWAY: FaultSpec(FaultType.ACTUATOR_RUNAWAY, 0, 1.0e-4, 0),
    }


def build_cusum_bank(mu: float, threshold: float) -> CusumBank:
    """Four one-sided CUSUMs, one per residual channel and sign.

    This is the textbook practical CUSUM implementation: a two-sided test on
    each residual component.  It isolates *which measurement channel* has
    shifted and in which direction, which is not the same as isolating which
    of seven faults occurred -- hence the separate GLR stage.
    """
    directions = {
        "ch0_pos": np.array([1.0, 0.0]),
        "ch0_neg": np.array([-1.0, 0.0]),
        "ch1_pos": np.array([0.0, 1.0]),
        "ch1_neg": np.array([0.0, -1.0]),
    }
    return CusumBank(
        detectors={
            name: CusumDetector(direction=d, mu=mu, threshold=threshold, label=name)
            for name, d in directions.items()
        }
    )


def run_scenarios(
    scenarios: list[Scenario], cfg: BenchmarkConfig | None = None
) -> list[LoopRun]:
    """Simulate every scenario once and return the runs in order."""
    bc = cfg if cfg is not None else BenchmarkConfig()
    out: list[LoopRun] = []
    for sc in scenarios:
        loop = LoopConfig(
            plant=bc.plant, gains=bc.gains, n_steps=sc.n_steps, seed=sc.seed, noise=True
        )
        out.append(simulate_loop(loop, sc.fault))
    return out


def harvest_training_rows(
    scenarios: list[Scenario],
    runs: list[LoopRun],
    cfg: BenchmarkConfig,
    fault_offsets: tuple[int, ...] = (0, 10, 25, 50),
    null_offsets: tuple[int, ...] = (-150, -300),
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Feature rows and class labels for classifier training.

    Windows starting at ``onset + offset`` for each ``fault_offsets`` entry
    carry the scenario's own class; windows starting at ``onset + offset`` for
    each (negative) ``null_offsets`` entry carry ``FaultType.NONE``, because
    that stretch of the run is genuinely fault-free.  A fault-free scenario
    contributes ``NONE`` rows from both sets.

    Returns
    -------
    (x, y) : tuple
        ``x`` has shape ``(n, 16)``, ``y`` holds class indices.
    """
    w = int(cfg.iso_window)
    rows: list[NDArray[np.float64]] = []
    labels: list[int] = []
    for sc, run in zip(scenarios, runs, strict=True):
        onset = sc.onset_step
        for off in fault_offsets:
            start = onset + int(off)
            if start < 0 or start + w > run.residual.shape[0]:
                continue
            feats, _ = feature_matrix(run.residual[start : start + w], w, 1)
            rows.append(feats[0])
            labels.append(class_index(sc.label))
        for off in null_offsets:
            start = onset + int(off)
            if start < 0 or start + w > onset:
                continue
            feats, _ = feature_matrix(run.residual[start : start + w], w, 1)
            rows.append(feats[0])
            labels.append(class_index(FaultType.NONE))
    if not rows:
        raise ValueError("no training rows harvested; check window and onset ranges")
    return np.stack(rows, axis=0), np.asarray(labels, dtype=np.int64)


def _glr_sliding_max(
    residual: NDArray[np.float64], bank: SignatureBank, start: int, stop: int
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Max GLR statistic over the bank for every window ending in a range."""
    w = bank.window
    ends = np.arange(max(start, w - 1), min(stop, residual.shape[0]), dtype=np.intp)
    if ends.size == 0:
        return np.zeros(0), ends
    starts = ends - w + 1
    idx = starts[:, None] + np.arange(w)[None, :]
    windows = residual[idx].reshape(ends.size, -1)
    proj = windows @ bank.matrix.T
    return np.max(proj * proj, axis=1), ends


def method_names(with_classifier: bool = True) -> list[str]:
    """Benchmark method keys in report order."""
    base = ["chi2_short", "chi2_long", "cusum", "glr"]
    return base + ["learned"] if with_classifier else base


def sequential_scores(
    run: LoopRun,
    cfg: BenchmarkConfig,
    bank: SignatureBank,
    classifier: FaultClassifier | None,
    span: tuple[int, int],
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.intp]]]:
    """Per-sample detection statistic of every method over ``span``.

    A method's statistic is attributed to the *last* sample it uses, so every
    decision is causal and the methods are directly comparable sample by
    sample.

    Returns
    -------
    dict of str to (score, index)
        ``score[i]`` is the statistic attributed to sample ``index[i]``.
    """
    lo, hi = int(span[0]), int(span[1])
    res = run.residual
    stop = min(hi, res.shape[0])
    out: dict[str, tuple[NDArray[np.float64], NDArray[np.intp]]] = {}

    for name, window in (("chi2_short", cfg.det_window), ("chi2_long", cfg.iso_window)):
        det = ChiSquaredDetector(window=window, dim=2, alpha=cfg.alpha)
        stat = det.run(res).statistic
        idx = np.arange(max(lo, window - 1), stop, dtype=np.intp)
        out[name] = (stat[idx], idx)

    cusum = build_cusum_bank(cfg.cusum_mu, 1.0)
    g = cusum.run(res).statistic
    idx = np.arange(max(lo, 0), stop, dtype=np.intp)
    out["cusum"] = (g[idx], idx)

    out["glr"] = _glr_sliding_max(res, bank, lo, stop)

    if classifier is not None:
        w = cfg.iso_window
        feats, ends = feature_matrix(res, w, 1, start=max(lo, w - 1))
        keep = ends < stop
        feats, ends = feats[keep], ends[keep]
        score = classifier.detection_score(feats) if ends.size else np.zeros(0)
        out["learned"] = (score, ends)
    return out


def sequential_alarms(
    run: LoopRun,
    cfg: BenchmarkConfig,
    bank: SignatureBank,
    classifier: FaultClassifier | None,
    thresholds: dict[str, float],
    span: tuple[int, int],
) -> dict[str, tuple[NDArray[np.bool_], NDArray[np.intp]]]:
    """Per-sample alarm flags for every method over ``span``.

    Thresholding of :func:`sequential_scores`; ``score > threshold`` alarms.
    """
    scores = sequential_scores(run, cfg, bank, classifier, span)
    return {
        name: (score > float(thresholds[name]), idx) for name, (score, idx) in scores.items()
    }


def calibrate_all_thresholds(
    runs: list[LoopRun],
    cfg: BenchmarkConfig,
    bank: SignatureBank,
    classifier: FaultClassifier | None,
    target_run_far: float = 0.10,
) -> dict[str, float]:
    """Match every method to the same per-run false-alarm probability.

    Parameters
    ----------
    runs : list of LoopRun
        **Fault-free** calibration runs.  Passing a faulted run silently
        corrupts the calibration, so the caller must filter.
    cfg, bank, classifier
        As elsewhere.
    target_run_far : float
        Probability that one fault-free run of length ``n_steps`` produces at
        least one alarm, in ``(0, 1)``.

    Returns
    -------
    dict of str to float
        Threshold per method.

    Notes
    -----
    Per-*run* calibration is used rather than per-*sample* because the
    sliding-window statistics are strongly autocorrelated: a 1700-sample run
    with a 100-sample window carries only about 17 independent windows, so an
    empirical 1e-3 per-sample quantile is estimated from a fraction of one
    effective exceedance and is worthless.  The run maximum is one
    approximately independent draw per run, so ``n`` runs give ``n`` effective
    samples and the quantile is as good as ``n`` allows.

    Detection delay can always be bought with false alarms, so delays measured
    at different false-alarm rates cannot be compared; this function is what
    makes the delay table in ``validation/`` mean anything.
    """
    t = float(target_run_far)
    if not (0.0 < t < 1.0):
        raise ValueError(f"target_run_far must lie in (0, 1), got {target_run_far}")
    if not runs:
        raise ValueError("calibration needs at least one fault-free run")
    maxima: dict[str, list[float]] = {}
    for run in runs:
        scores = sequential_scores(
            run, cfg, bank, classifier, (FAR_START, run.residual.shape[0])
        )
        for name, (score, _) in scores.items():
            maxima.setdefault(name, []).append(float(np.max(score)) if score.size else -np.inf)
    return {
        name: float(np.nextafter(np.quantile(np.asarray(v), 1.0 - t, method="higher"), np.inf))
        for name, v in maxima.items()
    }


def healthy_calibration_runs(
    n: int, seed0: int, cfg: BenchmarkConfig
) -> tuple[list[Scenario], list[LoopRun]]:
    """``n`` dedicated fault-free runs for threshold calibration.

    Kept separate from the training scenario set so that the calibration
    sample can be enlarged without changing the class balance the classifier
    is trained on.
    """
    scenarios = [
        sample_scenario(int(seed0) + i, index=i, fault_class=FaultType.NONE)
        for i in range(int(n))
    ]
    return scenarios, run_scenarios(scenarios, cfg)


@dataclass(frozen=True)
class MethodResult:
    """Detection performance of one method.

    Attributes
    ----------
    name : str
        Method key.
    delays : ndarray
        Detection delay in samples for each *detected* faulted run.
    censored : int
        Faulted runs that never alarmed.
    n_faulted : int
        Faulted runs evaluated.
    far_samples : tuple of int
        ``(alarming_samples, total_samples)`` on fault-free runs.
    far_runs : tuple of int
        ``(runs_with_any_alarm, total_runs)`` on fault-free runs.
    threshold : float
        Threshold used.
    per_run_delay : ndarray
        One entry per faulted run, in scenario order, ``nan`` where the run
        was never detected.  Use this for per-class breakdowns; ``delays`` is
        the same data with the ``nan`` entries removed.
    labels : tuple of FaultType
        Ground-truth class of each entry of ``per_run_delay``.
    """

    name: str
    delays: NDArray[np.float64]
    censored: int
    n_faulted: int
    far_samples: tuple[int, int]
    far_runs: tuple[int, int]
    threshold: float
    per_run_delay: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    labels: tuple[FaultType, ...] = ()

    def delays_for(self, fault: FaultType) -> NDArray[np.float64]:
        """Delays of one fault class, ``nan`` retained for censored runs."""
        mask = np.asarray([lab is fault for lab in self.labels], dtype=bool)
        return self.per_run_delay[mask]

    @property
    def far_per_sample(self) -> float:
        """Alarming fault-free samples divided by fault-free samples."""
        return self.far_samples[0] / self.far_samples[1] if self.far_samples[1] else float("nan")

    @property
    def detection_rate(self) -> float:
        """Fraction of faulted runs detected at all."""
        return (self.n_faulted - self.censored) / self.n_faulted if self.n_faulted else float("nan")


def calibrate_threshold(scores: NDArray[np.float64], target_far: float) -> float:
    """Smallest threshold whose fault-free alarm fraction is ``<= target_far``.

    Parameters
    ----------
    scores : ndarray
        Scores from fault-free samples.
    target_far : float
        Target per-sample false-alarm rate, in ``(0, 1)``.

    Returns
    -------
    float
        The ``1 - target_far`` empirical quantile of ``scores``, nudged up by
        one ULP so that the alarm test ``score > threshold`` is strict.

    Notes
    -----
    This is the honest cost of a method with no analytic threshold: it needs
    fault-free data to be usable at all, and the achieved rate is only as good
    as that sample.  With ``n`` calibration samples a target below ``1/n``
    cannot be resolved.
    """
    s = np.asarray(scores, dtype=float).reshape(-1)
    t = float(target_far)
    if s.size == 0:
        raise ValueError("need at least one calibration score")
    if not (0.0 < t < 1.0):
        raise ValueError(f"target_far must lie in (0, 1), got {target_far}")
    q = float(np.quantile(s, 1.0 - t, method="higher"))
    return float(np.nextafter(q, np.inf))


def evaluate_detection(
    scenarios: list[Scenario],
    runs: list[LoopRun],
    cfg: BenchmarkConfig,
    bank: SignatureBank,
    classifier: FaultClassifier | None,
    thresholds: dict[str, float],
    persistence: int = 1,
    delay_horizon: int = 600,
) -> dict[str, MethodResult]:
    """Detection delay and false-alarm rate for every method.

    Faulted runs supply delays; fault-free runs supply false-alarm rates.
    """
    names = method_names(classifier is not None)
    delays: dict[str, list[float]] = {n: [] for n in names}
    per_run: dict[str, list[float]] = {n: [] for n in names}
    fault_labels: list[FaultType] = []
    censored: dict[str, int] = dict.fromkeys(names, 0)
    far_hits: dict[str, int] = dict.fromkeys(names, 0)
    far_total: dict[str, int] = dict.fromkeys(names, 0)
    far_runs_hit: dict[str, int] = dict.fromkeys(names, 0)
    n_faulted = 0
    n_healthy = 0

    for sc, run in zip(scenarios, runs, strict=True):
        if sc.label is FaultType.NONE:
            n_healthy += 1
            span = (FAR_START, run.residual.shape[0])
            alarms = sequential_alarms(run, cfg, bank, classifier, thresholds, span)
            for name in names:
                flags, _ = alarms[name]
                far_hits[name] += int(np.count_nonzero(flags))
                far_total[name] += int(flags.size)
                far_runs_hit[name] += int(bool(np.any(flags)))
        else:
            n_faulted += 1
            fault_labels.append(sc.label)
            onset = sc.onset_step
            span = (onset, min(onset + int(delay_horizon), run.residual.shape[0]))
            alarms = sequential_alarms(run, cfg, bank, classifier, thresholds, span)
            for name in names:
                flags, idx = alarms[name]
                if idx.size == 0:
                    censored[name] += 1
                    per_run[name].append(float("nan"))
                    continue
                full = np.zeros(run.residual.shape[0], dtype=bool)
                full[idx] = flags
                d = detection_delay(full, onset, persistence=persistence)
                if np.isfinite(d):
                    delays[name].append(float(d))
                    per_run[name].append(float(d))
                else:
                    censored[name] += 1
                    per_run[name].append(float("nan"))

    return {
        name: MethodResult(
            name=name,
            delays=np.asarray(delays[name], dtype=float),
            censored=censored[name],
            n_faulted=n_faulted,
            far_samples=(far_hits[name], far_total[name]),
            far_runs=(far_runs_hit[name], n_healthy),
            threshold=float(thresholds[name]),
            per_run_delay=np.asarray(per_run[name], dtype=float),
            labels=tuple(fault_labels),
        )
        for name in names
    }


def window_scores(
    scenarios: list[Scenario],
    runs: list[LoopRun],
    cfg: BenchmarkConfig,
    bank: SignatureBank,
    classifier: FaultClassifier | None,
    null_offsets: tuple[int, ...] = (-150, -300, -450),
) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
    """Window-level scores for ROC curves.

    Returns
    -------
    (positive, negative) : tuple of dict
        Score arrays per method name.  Positives come from faulted runs at
        ``cfg.roc_offsets`` after onset; negatives from pre-onset windows of
        every run at ``null_offsets``.
    """
    w = cfg.iso_window
    pos_windows: list[NDArray[np.float64]] = []
    neg_windows: list[NDArray[np.float64]] = []

    for sc, run in zip(scenarios, runs, strict=True):
        res = run.residual
        onset = sc.onset_step
        sink = pos_windows if sc.label is not FaultType.NONE else neg_windows
        for off in cfg.roc_offsets:
            start = onset + int(off)
            if start + w <= res.shape[0]:
                sink.append(res[start : start + w])
        for off in null_offsets:
            start = onset + int(off)
            if start >= 0 and start + w <= onset:
                neg_windows.append(res[start : start + w])

    if not pos_windows or not neg_windows:
        raise ValueError("need at least one positive and one negative window")
    return (
        _score_windows(np.stack(pos_windows), cfg, bank, classifier),
        _score_windows(np.stack(neg_windows), cfg, bank, classifier),
    )


def _score_windows(
    windows: NDArray[np.float64],
    cfg: BenchmarkConfig,
    bank: SignatureBank,
    classifier: FaultClassifier | None,
) -> dict[str, NDArray[np.float64]]:
    """Batch window-level scores, shape ``(n, W, 2)`` in, dict of ``(n,)`` out.

    Every method is restarted at the window boundary, including the CUSUM, so
    that no method carries information from before the window into its score.
    """
    n, w, _ = windows.shape
    flat = windows.reshape(n, -1)
    out = {"chi2_long": np.sum(flat * flat, axis=1)}

    mu = float(cfg.cusum_mu)
    proj = np.stack(
        [windows[:, :, 0], -windows[:, :, 0], windows[:, :, 1], -windows[:, :, 1]], axis=2
    )
    incr = mu * proj - 0.5 * mu * mu
    s = np.cumsum(incr, axis=1)
    g = s - np.minimum(0.0, np.minimum.accumulate(s, axis=1))
    out["cusum"] = np.max(g[:, -1, :], axis=1)

    glr = flat @ bank.matrix.T
    out["glr"] = np.max(glr * glr, axis=1)

    if classifier is not None:
        feats = np.stack([window_features_batch(windows)], axis=0)[0]
        out["learned"] = classifier.detection_score(feats)
    return out


def window_features_batch(windows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Features for a stack of windows, shape ``(n, W, 2)`` -> ``(n, 16)``."""
    n, w, _ = windows.shape
    stacked = windows.reshape(n * w, 2)
    ends = np.arange(w - 1, n * w, w, dtype=np.intp)
    return _vectorised_features(stacked, ends, w)


@dataclass(frozen=True)
class IsolationOutcome:
    """Isolation predictions for one method.

    Attributes
    ----------
    truth : ndarray of int
        Ground-truth class indices.
    predicted : ndarray of int
        Predicted class indices.
    confidence : ndarray
        Confidence of each prediction; ``nan`` where the method declined to
        declare a fault.
    """

    truth: NDArray[np.int64]
    predicted: NDArray[np.int64]
    confidence: NDArray[np.float64]


def evaluate_isolation(
    scenarios: list[Scenario],
    runs: list[LoopRun],
    cfg: BenchmarkConfig,
    bank: SignatureBank,
    classifier: FaultClassifier | None,
    glr_alpha: float | None = None,
    offset: int = 0,
) -> dict[str, IsolationOutcome]:
    """Isolate one window per run with the GLR bank and the classifier.

    Parameters
    ----------
    offset : int
        Shift of the isolation window relative to the true onset [samples].
        Non-zero values measure how much the methods lose when the onset is
        not known exactly.
    """
    from .isolation import isolate_window

    w = cfg.iso_window
    a = float(glr_alpha) if glr_alpha is not None else cfg.alpha
    truth: list[int] = []
    glr_pred: list[int] = []
    glr_conf: list[float] = []
    iso_windows: list[NDArray[np.float64]] = []

    for sc, run in zip(scenarios, runs, strict=True):
        start = sc.onset_step + int(offset)
        if start < 0 or start + w > run.residual.shape[0]:
            continue
        win = run.residual[start : start + w]
        truth.append(class_index(sc.label))
        result = isolate_window(win, bank, alpha=a)
        glr_pred.append(class_index(result.fault))
        glr_conf.append(result.confidence)
        if classifier is not None:
            iso_windows.append(win)

    out = {
        "glr": IsolationOutcome(
            truth=np.asarray(truth, dtype=np.int64),
            predicted=np.asarray(glr_pred, dtype=np.int64),
            confidence=np.asarray(glr_conf, dtype=float),
        )
    }
    if classifier is not None and iso_windows:
        pred = classifier.predict_with_confidence(window_features_batch(np.stack(iso_windows)))
        out["learned"] = IsolationOutcome(
            truth=np.asarray(truth, dtype=np.int64),
            predicted=np.asarray([class_index(c) for c in pred.classes], dtype=np.int64),
            confidence=np.asarray(pred.confidence, dtype=float),
        )
    return out


def design_thresholds(cfg: BenchmarkConfig) -> dict[str, float]:
    """Thresholds that follow from the design formulas alone, no data used."""
    return {
        "chi2_short": chi2_threshold(cfg.alpha, cfg.det_window * 2),
        "chi2_long": chi2_threshold(cfg.alpha, cfg.iso_window * 2),
        "cusum": cusum_threshold_for_arl0(cfg.cusum_arl0, cfg.cusum_mu),
    }


def default_scenario_sets(
    n_train: int = 240, n_test: int = 240, seed_train: int = 1000, seed_test: int = 5000
) -> tuple[list[Scenario], list[Scenario]]:
    """The training and held-out scenario sets used throughout the package.

    Seeds are disjoint by construction, and the class cycle makes both sets
    exactly balanced when their sizes are multiples of eight.
    """
    return (
        sample_scenarios(n_train, seed_train),
        sample_scenarios(n_test, seed_test),
    )


def build_default_bank(cfg: BenchmarkConfig, n_onsets: int = 8) -> SignatureBank:
    """Signature bank averaged over ``n_onsets`` phases of the reference cycle.

    The reference period is 60 s at 0.1 s per sample, so eight onsets spaced
    75 samples apart cover exactly one period.
    """
    loop = LoopConfig(plant=cfg.plant, gains=cfg.gains, n_steps=1400, noise=False)
    period_steps = int(round(60.0 / cfg.plant.dt_s))
    spacing = max(1, period_steps // int(n_onsets))
    onsets = [600 + i * spacing for i in range(int(n_onsets))]
    return build_signature_bank(loop, default_signature_specs(cfg.plant), cfg.iso_window, onsets)


def class_labels() -> tuple[str, ...]:
    """Short class names in :data:`fdiscope.faults.FAULT_CLASSES` order."""
    return tuple(f.value for f in FAULT_CLASSES)
