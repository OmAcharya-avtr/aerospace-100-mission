"""The matched-false-alarm-rate comparison harness.

This module is the narrow defensible claim of the package.  It does one thing:
it brings every detector to the *same* window false-alarm probability on
independent nominal data, and only then measures what each one costs in
detection delay and detection probability.  A detection-delay table produced any
other way is not a comparison, because the fastest detector is always the one
with the loosest threshold.

The procedure, in order
-----------------------
1. **Train** every detector that needs fitting on a nominal block.  Seed
   ``seed + 1``.
2. **Calibrate** each detector's scalar threshold on a second, independent
   nominal block so that exactly ``ceil(target_alpha_w * n_cal)`` of those
   windows alarm (:func:`telemetryool.calibration.calibrate_threshold`).  Seed
   ``seed + 2``.
3. **Measure** the false-alarm rate on a third, independent nominal block, and
   report it with its binomial standard error and the signed discrepancy from
   the target in units of that error.  Seed ``seed + 3``.  This is the number
   that proves the match; the calibration-set rate is in-sample and proves
   nothing.
4. **Evaluate** each anomaly scenario on its own block, seed
   ``seed + 10 + scenario_index``, reporting the full confusion matrix against
   the step-3 nominal block, the detection delay conditional on detection, and
   the window-level ROC.

Two sources of error are kept visible rather than averaged away.  The
calibration threshold is itself estimated from a finite sample, so the measured
false-alarm rate has roughly ``sqrt(2)`` times the binomial standard error of a
known-threshold estimate; both the calibration size and the measurement size are
printed.  And the measured rate is compared against the *target*, not against
the calibration set's achieved rate, so a mis-calibration cannot hide.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .calibration import Calibration, RateEstimate, calibrate_threshold, estimate_rate
from .detectors import WindowDetector
from .metrics import (
    ConfusionMatrix,
    DelayStats,
    RocCurve,
    confusion_matrix,
    delay_stats,
    window_roc,
)
from .runs import any_run
from .synthetic import Anomaly, NominalModel, generate_anomalous, generate_nominal

__all__ = [
    "ScenarioSpec",
    "MethodScenarioResult",
    "MethodResult",
    "ComparisonResult",
    "run_comparison",
]


@dataclass(frozen=True)
class ScenarioSpec:
    """One anomaly scenario to evaluate every detector against.

    Parameters
    ----------
    name
        Short label used in the result tables.
    anomaly
        The anomaly injected into each window of this scenario's block.
    n_windows
        Anomalous windows to generate, >= 1.  Detection probability near 0.5
        estimated from ``n`` windows has standard error ``0.5 / sqrt(n)``.
    """

    name: str
    anomaly: Anomaly
    n_windows: int = 2000

    def __post_init__(self) -> None:
        if self.n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {self.n_windows!r}")


@dataclass(frozen=True)
class MethodScenarioResult:
    """One detector against one scenario, at the matched operating point."""

    method: str
    scenario: str
    confusion: ConfusionMatrix
    delay: DelayStats
    roc: RocCurve


@dataclass(frozen=True)
class MethodResult:
    """Everything measured for one detector.

    Attributes
    ----------
    name
        Detector identifier.
    persistence
        Debounce count.
    calibration
        The in-sample calibration record.
    false_alarm
        Independent measurement of the window false-alarm probability, with its
        binomial standard error and interval estimates.
    design_alpha_w
        Analytic design value from :mod:`telemetryool.arl` where one exists for
        this detector's configuration (single-channel, persistence 1), else
        ``None``.  It is reported *beside* the measurement, never in place of it.
    per_scenario
        Scenario name -> :class:`MethodScenarioResult`.
    """

    name: str
    persistence: int
    calibration: Calibration
    false_alarm: RateEstimate
    design_alpha_w: float | None
    per_scenario: dict[str, MethodScenarioResult] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonResult:
    """Result of :func:`run_comparison`, with its own text rendering."""

    target_alpha_w: float
    window_length: int
    n_channels: int
    n_train_windows: int
    n_cal_windows: int
    n_measure_windows: int
    seed: int
    methods: list[MethodResult]
    scenarios: list[ScenarioSpec]
    elapsed_s: float

    def combined_standard_error(self) -> float:
        """Standard error of the measured rate including the calibration's own error.

        ``sqrt(alpha (1 - alpha) (1 / n_cal + 1 / n_measure))``.  The calibration
        threshold is an order statistic of the calibration trigger levels, so by
        the quantile/ECDF duality the out-of-sample rate it induces has variance
        ``alpha (1 - alpha) / n_cal`` to first order, independent of the
        measurement's own binomial variance.  Judging agreement with the target
        on the measurement binomial error alone therefore overstates the
        discrepancy; this is the error to use, and its adequacy is checked
        empirically across seeds in ``validation/validate_matched_far.py``.
        """
        a = self.target_alpha_w
        return float(
            np.sqrt(a * (1.0 - a) * (1.0 / self.n_cal_windows + 1.0 / self.n_measure_windows))
        )

    # -- rendering --------------------------------------------------------- #

    def far_table(self) -> str:
        """Design, calibration and measured false-alarm rate, side by side."""
        lines = [
            "Window false-alarm probability: target, calibration (in-sample), "
            "measurement (independent)",
            f"  target alpha_W = {self.target_alpha_w:.6f}   W = {self.window_length} samples"
            f"   channels = {self.n_channels}",
            f"  calibration windows = {self.n_cal_windows}   measurement windows = "
            f"{self.n_measure_windows}",
            "",
            f"  {'method':24s} {'threshold':>12s} {'cal':>9s} {'measured':>9s} "
            f"{'binom SE':>9s} {'comb SE':>9s} {'z comb':>8s} {'95% Wilson':>20s}",
        ]
        se_comb = self.combined_standard_error()
        for m in self.methods:
            fa = m.false_alarm
            z = (fa.rate - self.target_alpha_w) / se_comb
            lines.append(
                f"  {m.name:24s} {m.calibration.threshold:12.6f} "
                f"{m.calibration.achieved_alpha_w:9.5f} {fa.rate:9.5f} "
                f"{fa.standard_error:9.5f} {se_comb:9.5f} {z:8.2f} "
                f"[{fa.wilson_low:.5f}, {fa.wilson_high:.5f}]"
            )
        extra = [m for m in self.methods if m.design_alpha_w is not None]
        if extra:
            lines.append("")
            lines.append("  analytic single-channel design value, for reference only:")
            for m in extra:
                lines.append(f"    {m.name:24s} alpha_W(design) = {m.design_alpha_w:.6f}")
        return "\n".join(lines)

    def delay_table(self) -> str:
        """Detection probability and delay per method per scenario."""
        out = []
        for sc in self.scenarios:
            out.append(f"Scenario {sc.name!r}: {sc.anomaly.kind} onset={sc.anomaly.onset} "
                       f"magnitude={sc.anomaly.magnitude} channels="
                       f"{sc.anomaly.channels if sc.anomaly.channels is not None else [0]} "
                       f"n_windows={sc.n_windows}")
            out.append(
                f"  {'method':24s} {'Pd':>8s} {'SE':>8s} {'mean':>8s} {'median':>8s} "
                f"{'p10':>6s} {'p90':>6s} {'early':>7s} {'AUC':>8s}"
            )
            for m in self.methods:
                r = m.per_scenario.get(sc.name)
                if r is None:
                    continue
                d = r.delay
                out.append(
                    f"  {m.name:24s} {d.detection_probability:8.4f} "
                    f"{d.detection_probability_se:8.4f} {d.mean_delay:8.2f} "
                    f"{d.median_delay:8.1f} {d.p10_delay:6.1f} {d.p90_delay:6.1f} "
                    f"{d.n_early:7d} {r.roc.auc:8.4f}"
                )
            out.append("")
        return "\n".join(out)

    def confusion_report(self) -> str:
        """Every confusion matrix in full, with its derived rates."""
        out = []
        for sc in self.scenarios:
            for m in self.methods:
                r = m.per_scenario.get(sc.name)
                if r is None:
                    continue
                out.append(f"Confusion matrix: method={m.name!r} scenario={sc.name!r}")
                out.append(r.confusion.table(indent="  "))
                out.append(f"  {r.confusion.summary()}")
                out.append("")
        return "\n".join(out)

    def to_text(self) -> str:
        """The whole result as plain text, suitable for a validation log."""
        header = [
            "Matched-false-alarm-rate comparison",
            f"  seed={self.seed}  elapsed={self.elapsed_s:.1f} s  "
            f"train_windows={self.n_train_windows}",
            "",
        ]
        return "\n".join(header + [self.far_table(), "", self.delay_table(),
                                   self.confusion_report()])

    def fastest_at_matched_far(self, scenario: str) -> str:
        """Name of the method with the lowest mean delay among those with Pd >= 0.5.

        Returns the empty string if no method reaches Pd = 0.5 on that scenario.
        Ranking on mean delay alone is meaningless when detection probabilities
        differ, which is why the Pd floor is part of the definition and why the
        tables always print both.
        """
        best, best_delay = "", np.inf
        for m in self.methods:
            r = m.per_scenario.get(scenario)
            if r is None or r.delay.detection_probability < 0.5:
                continue
            if r.delay.mean_delay < best_delay:
                best, best_delay = m.name, r.delay.mean_delay
        return best


def run_comparison(
    detectors: Sequence[WindowDetector],
    model: NominalModel,
    scenarios: Sequence[ScenarioSpec],
    target_alpha_w: float = 0.05,
    window_length: int = 100,
    n_train_windows: int = 1500,
    n_cal_windows: int = 10000,
    n_measure_windows: int = 20000,
    seed: int = 20261005,
    design_alpha: dict[str, float] | None = None,
) -> ComparisonResult:
    """Train, calibrate, measure and evaluate every detector at a matched false-alarm rate.

    Parameters
    ----------
    detectors
        Detectors to compare.  Names must be unique.
    model
        Nominal telemetry model used for every block.
    scenarios
        Anomaly scenarios.  Names must be unique.
    target_alpha_w
        Common window false-alarm probability in (0, 1).
    window_length
        ``W``, samples per window, >= 1.
    n_train_windows, n_cal_windows, n_measure_windows
        Nominal block sizes.  ``n_cal_windows * target_alpha_w`` must be at
        least 1 for the target to be representable.
    seed
        Base seed.  Each block uses a distinct derived seed so the blocks are
        independent and the whole run is reproducible.
    design_alpha
        Optional mapping detector name -> analytic design ``alpha_W``, printed
        for reference beside the measurement.

    Returns
    -------
    ComparisonResult
    """
    if not detectors:
        raise ValueError("at least one detector is required")
    if not scenarios:
        raise ValueError("at least one scenario is required")
    names = [d.name for d in detectors]
    if len(set(names)) != len(names):
        raise ValueError(f"detector names must be unique, got {names}")
    sc_names = [s.name for s in scenarios]
    if len(set(sc_names)) != len(sc_names):
        raise ValueError(f"scenario names must be unique, got {sc_names}")
    if not (0.0 < float(target_alpha_w) < 1.0):
        raise ValueError(f"target_alpha_w must lie in (0, 1), got {target_alpha_w!r}")

    t_start = time.perf_counter()
    design_alpha = dict(design_alpha or {})

    train = generate_nominal(model, n_train_windows, window_length,
                             np.random.default_rng(seed + 1))
    cal = generate_nominal(model, n_cal_windows, window_length,
                           np.random.default_rng(seed + 2))
    meas = generate_nominal(model, n_measure_windows, window_length,
                            np.random.default_rng(seed + 3))
    anomalous: dict[str, NDArray[np.float64]] = {}
    for i, sc in enumerate(scenarios):
        rng = np.random.default_rng(seed + 10 + i)
        anomalous[sc.name] = generate_anomalous(
            model, sc.anomaly, sc.n_windows, window_length, rng
        )

    results: list[MethodResult] = []
    for det in detectors:
        if det.requires_fit:
            det.fit(train)
        cal_scores = det.scores(cal)
        calib = calibrate_threshold(cal_scores, det.persistence, target_alpha_w)
        meas_scores = det.scores(meas)
        alarmed_nominal = any_run(meas_scores > calib.threshold, det.persistence)
        fa = estimate_rate(int(alarmed_nominal.sum()), int(alarmed_nominal.size))

        per_scenario: dict[str, MethodScenarioResult] = {}
        for sc in scenarios:
            a_scores = det.scores(anomalous[sc.name])
            alarmed_anom = any_run(a_scores > calib.threshold, det.persistence)
            per_scenario[sc.name] = MethodScenarioResult(
                method=det.name,
                scenario=sc.name,
                confusion=confusion_matrix(alarmed_anom, alarmed_nominal),
                delay=delay_stats(
                    a_scores, calib.threshold, det.persistence, sc.anomaly.onset
                ),
                roc=window_roc(meas_scores, a_scores, det.persistence),
            )
        results.append(
            MethodResult(
                name=det.name,
                persistence=det.persistence,
                calibration=calib,
                false_alarm=fa,
                design_alpha_w=design_alpha.get(det.name),
                per_scenario=per_scenario,
            )
        )

    return ComparisonResult(
        target_alpha_w=float(target_alpha_w),
        window_length=int(window_length),
        n_channels=int(model.n_channels),
        n_train_windows=int(n_train_windows),
        n_cal_windows=int(n_cal_windows),
        n_measure_windows=int(n_measure_windows),
        seed=int(seed),
        methods=results,
        scenarios=list(scenarios),
        elapsed_s=time.perf_counter() - t_start,
    )
