"""Benchmarks: learned against threshold logic, and what conservatism costs.

Two measurements, both on held-out data:

1. **Detector comparison at an equal false-alarm rate.** The threshold-logic
   baseline has a fixed declared operating point. Its false-alarm rate -- the
   fraction of no-fault observations at which it declares some fault -- is
   measured on a calibration split, and the learned classifier's
   declare-a-fault threshold is then tuned on the SAME split to match that
   rate. Both are then evaluated on a test split that neither was tuned on.
   Comparing at a matched false-alarm rate is the only comparison that means
   anything: a detector can always buy identification accuracy with false
   alarms.

2. **Policy cost, four policies.** Oracle (perfect identification), always
   fault-specific, confidence-routed, and always conservative. Cost is the
   declared resolution cost in minutes of mission outage; the cost of
   conservatism is the difference between the confidence-routed policy and the
   oracle, and separately between the confidence-routed policy and the
   always-specific policy. Mis-routes, missed detections and false alarms are
   counted, never folded into the cost.

Everything here is a function of a seed and the declaration. Nothing is tuned
on the test split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .automaton import DetectionAssumption
from .baseline import ThresholdLogicClassifier
from .calibration import CalibrationResult, brier_score, reliability_curve
from .data import NOMINAL_LABEL, ObservationSet, make_observations
from .learned import LearnedFaultClassifier, accuracy_ceiling
from .policy import FaultResponsePolicy
from .spec import SafeModeSpec

__all__ = [
    "DetectorScore",
    "PolicyScore",
    "Split",
    "compare_detectors",
    "compare_policies",
    "conservatism_sweep",
    "false_alarm_rate",
    "identification_accuracy",
    "learned_decisions",
    "noise_sweep",
    "split_observations",
]


@dataclass(frozen=True)
class Split:
    """A train / calibration / test partition of one observation set.

    The calibration split is where the learned detector's operating point is
    chosen and where the baseline's false-alarm rate is measured. The test
    split is touched by neither.
    """

    train: ObservationSet
    calibration: ObservationSet
    test: ObservationSet

    def sizes(self) -> tuple[int, int, int]:
        """Row counts of the three parts."""
        return (self.train.n_samples, self.calibration.n_samples, self.test.n_samples)


def _subset(observations: ObservationSet, index: np.ndarray) -> ObservationSet:
    return ObservationSet(
        features=observations.features[index],
        labels=observations.labels[index],
        channel_ids=observations.channel_ids,
        class_names=observations.class_names,
        noise_scale=observations.noise_scale,
    )


def split_observations(
    observations: ObservationSet,
    *,
    train_fraction: float = 0.5,
    calibration_fraction: float = 0.25,
    seed: int = 0,
) -> Split:
    """Partition an observation set into train, calibration and test.

    Args:
        observations: the set to split.
        train_fraction: share used for fitting, in (0, 1).
        calibration_fraction: share used for choosing operating points, in
            (0, 1). The remainder is the test split, which must be non-empty.
        seed: seed for the permutation.

    The split is a plain random permutation, not stratified, so class
    proportions differ between parts by sampling noise. The counts per class in
    each part are reported by ``ObservationSet.counts``.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            f"calibration_fraction must be in (0, 1), got {calibration_fraction}"
        )
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError(
            "train_fraction + calibration_fraction must be < 1 so the test split is "
            f"non-empty, got {train_fraction} + {calibration_fraction}"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(observations.n_samples)
    n_train = int(round(train_fraction * observations.n_samples))
    n_calibration = int(round(calibration_fraction * observations.n_samples))
    return Split(
        train=_subset(observations, order[:n_train]),
        calibration=_subset(observations, order[n_train : n_train + n_calibration]),
        test=_subset(observations, order[n_train + n_calibration :]),
    )


def false_alarm_rate(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of no-fault samples at which some fault was declared.

    Dimensionless in [0, 1]. Returns ``float('nan')`` when there is no no-fault
    sample to measure on.
    """
    predicted = np.asarray(predicted).astype(str)
    truth = np.asarray(truth).astype(str)
    mask = truth == NOMINAL_LABEL
    if not mask.any():
        return float("nan")
    return float(np.mean(predicted[mask] != NOMINAL_LABEL))


def identification_accuracy(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of true-fault samples identified as the right fault.

    A missed detection counts as wrong. Dimensionless in [0, 1]. Returns
    ``float('nan')`` when there is no faulted sample.
    """
    predicted = np.asarray(predicted).astype(str)
    truth = np.asarray(truth).astype(str)
    mask = truth != NOMINAL_LABEL
    if not mask.any():
        return float("nan")
    return float(np.mean(predicted[mask] == truth[mask]))


def learned_decisions(
    model: LearnedFaultClassifier, features: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """Labels and consumed confidences of the learned detector at threshold ``tau``.

    Args:
        model: a fitted :class:`safemodetree.learned.LearnedFaultClassifier`.
        features: ``(n, n_channels)`` observation vectors.
        tau: total fault-posterior threshold for declaring a fault,
            dimensionless in [0, 1].

    Returns:
        ``(labels, confidence)``, the second dimensionless in [0, 1].

    Below ``tau`` of total fault posterior the detector declares no fault and
    reports the no-fault posterior as its confidence. At or above it, the
    detector declares the most probable FAULT and reports that fault's
    posterior -- not a renormalised one -- because the policy's question is
    "what is the probability this specific fault is the cause", and the
    unnormalised posterior is that probability.
    """
    proba = model.predict_proba(features)
    classes = model.classes
    fault_columns = [i for i, name in enumerate(classes) if name != NOMINAL_LABEL]
    nominal_column = classes.index(NOMINAL_LABEL) if NOMINAL_LABEL in classes else None
    fault_mass = (
        1.0 - proba[:, nominal_column] if nominal_column is not None else np.ones(len(proba))
    )
    best_fault = np.array(fault_columns)[proba[:, fault_columns].argmax(axis=1)]
    labels = np.empty(proba.shape[0], dtype=object)
    confidence = np.empty(proba.shape[0], dtype=float)
    declare = fault_mass >= tau
    rows = np.arange(proba.shape[0])
    labels[declare] = [classes[int(c)] for c in best_fault[declare]]
    confidence[declare] = proba[rows[declare], best_fault[declare]]
    labels[~declare] = NOMINAL_LABEL
    confidence[~declare] = (
        1.0 - fault_mass[~declare] if nominal_column is not None else 0.0
    )
    return labels, confidence


def _match_false_alarm_rate(
    model: LearnedFaultClassifier, calibration: ObservationSet, target: float
) -> tuple[float, float]:
    """Choose the learned detector's threshold to hit a target false-alarm rate.

    Searches the thresholds at which the calibration-split false-alarm rate
    actually changes, which are the fault-posterior values of the no-fault
    calibration samples. Returns ``(tau, achieved_far_on_calibration)``.
    """
    truth = np.asarray(calibration.labels).astype(str)
    nominal_mass = model.fault_probability(calibration.features)[truth == NOMINAL_LABEL]
    if nominal_mass.size == 0:
        raise ValueError("the calibration split has no no-fault samples to match a rate on")
    candidates = np.unique(
        np.concatenate([[0.0], np.nextafter(nominal_mass, np.inf)])
    )
    best_tau = float(candidates[-1])
    best_gap = math.inf
    best_far = 0.0
    for tau in candidates:
        far = float(np.mean(nominal_mass >= tau))
        gap = abs(far - target)
        if gap < best_gap - 1e-15:
            best_gap, best_tau, best_far = gap, float(tau), far
    return best_tau, best_far


@dataclass(frozen=True)
class DetectorScore:
    """One detector scored on the test split at a stated operating point.

    Attributes:
        name: detector name.
        operating_point: the scalar that fixes the operating point --
            ``margin_sd`` for the baseline, the fault-posterior threshold
            ``tau`` for the learned model.
        false_alarm_rate: fraction of no-fault test samples at which a fault
            was declared, dimensionless.
        detection_rate: fraction of faulted test samples at which some fault
            was declared, dimensionless.
        identification_accuracy: fraction of faulted test samples identified
            as the right fault, dimensionless.
        overall_accuracy: fraction of all test samples labelled correctly.
        calibration: binned calibration of the top-label confidence.
        brier: multi-class Brier score of the full posterior, or None for the
            baseline, which has no posterior over all classes.
    """

    name: str
    operating_point: float
    false_alarm_rate: float
    detection_rate: float
    identification_accuracy: float
    overall_accuracy: float
    calibration: CalibrationResult
    brier: float | None

    def row(self) -> str:
        """One fixed-width line for a comparison table."""
        brier = "-" if self.brier is None else f"{self.brier:.5f}"
        return (
            f"{self.name:<22} {self.operating_point:>8.5f} {self.false_alarm_rate:>9.5f} "
            f"{self.detection_rate:>9.5f} {self.identification_accuracy:>9.5f} "
            f"{self.overall_accuracy:>9.5f} {self.calibration.ece:>8.5f} "
            f"{self.calibration.mce:>8.5f} {brier:>9}"
        )


def _score(
    name: str,
    operating_point: float,
    labels: np.ndarray,
    confidence: np.ndarray,
    truth: np.ndarray,
    n_bins: int,
    proba: np.ndarray | None = None,
    classes: tuple[str, ...] | None = None,
) -> DetectorScore:
    predicted = np.asarray(labels).astype(str)
    truth_str = np.asarray(truth).astype(str)
    correct = predicted == truth_str
    faulted = truth_str != NOMINAL_LABEL
    brier = None
    if proba is not None and classes is not None:
        brier = brier_score(proba, truth_str, classes)
    return DetectorScore(
        name=name,
        operating_point=float(operating_point),
        false_alarm_rate=false_alarm_rate(predicted, truth_str),
        detection_rate=(
            float(np.mean(predicted[faulted] != NOMINAL_LABEL)) if faulted.any() else float("nan")
        ),
        identification_accuracy=identification_accuracy(predicted, truth_str),
        overall_accuracy=float(np.mean(correct)),
        calibration=reliability_curve(confidence, correct, n_bins),
        brier=brier,
    )


def compare_detectors(
    spec: SafeModeSpec,
    *,
    n_per_fault: int = 400,
    noise_scale: float = 4.0,
    seed: int = 63001,
    calibration_method: str = "isotonic",
    n_bins: int = 10,
) -> dict[str, object]:
    """Score the baseline and the learned classifier at an equal false-alarm rate.

    Returns a mapping with keys:

    - ``"baseline"``, ``"learned"``, ``"learned_argmax"``: :class:`DetectorScore`.
      ``learned`` is at the matched false-alarm rate; ``learned_argmax`` is the
      model's own plain argmax operating point, reported so the price of the
      match is visible.
    - ``"tau"``: the matched threshold.
    - ``"target_far"``: the baseline's calibration-split false-alarm rate.
    - ``"ceiling"``: upper bound on achievable accuracy from the declared
      observability model.
    - ``"sizes"``: (train, calibration, test) row counts.
    - ``"noise_scale"``, ``"seed"``.
    """
    observations = make_observations(
        spec, n_per_fault=n_per_fault, noise_scale=noise_scale, seed=seed
    )
    split = split_observations(observations, seed=seed + 1)

    baseline = ThresholdLogicClassifier(spec, margin_sd=0.0)
    cal_labels, _ = baseline.predict_with_confidence(split.calibration.features)
    target_far = false_alarm_rate(cal_labels, split.calibration.labels)

    model = LearnedFaultClassifier(calibration=calibration_method, random_state=seed % 10_000)
    model.fit(split.train.features, split.train.labels)
    tau, _achieved = _match_false_alarm_rate(model, split.calibration, target_far)

    test_truth = split.test.labels
    base_labels, base_conf = baseline.predict_with_confidence(split.test.features)
    baseline_score = _score(
        "threshold logic", 0.0, base_labels, base_conf, test_truth, n_bins
    )

    proba = model.predict_proba(split.test.features)
    matched_labels, matched_conf = learned_decisions(model, split.test.features, tau)
    learned_score = _score(
        "learned (matched FAR)",
        tau,
        matched_labels,
        matched_conf,
        test_truth,
        n_bins,
        proba=proba,
        classes=model.classes,
    )
    argmax_labels, argmax_conf = model.predict_with_confidence(split.test.features)
    argmax_score = _score(
        "learned (argmax)",
        0.0,
        argmax_labels,
        argmax_conf,
        test_truth,
        n_bins,
        proba=proba,
        classes=model.classes,
    )
    return {
        "baseline": baseline_score,
        "learned": learned_score,
        "learned_argmax": argmax_score,
        "tau": tau,
        "target_far": target_far,
        "ceiling": accuracy_ceiling(spec, split.test),
        "sizes": split.sizes(),
        "noise_scale": float(noise_scale),
        "seed": int(seed),
        "model": model,
        "baseline_model": baseline,
        "split": split,
    }


def noise_sweep(
    spec: SafeModeSpec,
    *,
    noise_scales: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0),
    n_per_fault: int = 400,
    seed: int = 63001,
    calibration_method: str = "isotonic",
) -> tuple[dict[str, object], ...]:
    """Run :func:`compare_detectors` at several declared noise scales.

    The comparison is reported as a function of difficulty rather than at one
    value, because at ``noise_scale = 1`` the declared thresholds sit many
    sigma from both class means and the comparison is uninformative.
    """
    return tuple(
        compare_detectors(
            spec,
            n_per_fault=n_per_fault,
            noise_scale=scale,
            seed=seed,
            calibration_method=calibration_method,
        )
        for scale in noise_scales
    )


@dataclass(frozen=True)
class PolicyScore:
    """One policy scored over the faulted and nominal test samples.

    Attributes:
        name: policy name.
        confidence_threshold: the declared routing threshold used.
        n_fault_events: faulted test samples scored.
        n_commanded: faulted events at which a response was commanded.
        n_missed: faulted events at which nothing was commanded (missed
            detections). Not costed; see :mod:`safemodetree.policy`.
        mean_cost_min: mean resolution cost over the commanded faulted events,
            in minutes of mission outage.
        n_misrouted: commanded events where a fault-specific response for the
            WRONG fault was commanded.
        n_unsafe: commanded events after which no declared safe mode remains
            reachable.
        conservative_fraction: share of commanded events routed conservatively.
        false_alarm_cost_min: mean commanded response cost per no-fault test
            sample, in minutes of mission outage.
        n_false_alarms: no-fault test samples at which a response was commanded.
    """

    name: str
    confidence_threshold: float
    n_fault_events: int
    n_commanded: int
    n_missed: int
    mean_cost_min: float
    n_misrouted: int
    n_unsafe: int
    conservative_fraction: float
    false_alarm_cost_min: float
    n_false_alarms: int

    def row(self) -> str:
        """One fixed-width line for a comparison table."""
        return (
            f"{self.name:<22} {self.confidence_threshold:>6.3f} {self.n_commanded:>7} "
            f"{self.n_missed:>7} {self.mean_cost_min:>10.4f} {self.n_misrouted:>7} "
            f"{self.n_unsafe:>7} {self.conservative_fraction:>7.4f} "
            f"{self.false_alarm_cost_min:>10.4f}"
        )


def _score_policy(
    name: str,
    policy: FaultResponsePolicy,
    mode_id: str,
    labels: np.ndarray,
    confidence: np.ndarray,
    truth: np.ndarray,
    *,
    oracle: bool = False,
) -> PolicyScore:
    truth = np.asarray(truth).astype(str)
    labels = np.asarray(labels).astype(str)
    confidence = np.asarray(confidence, dtype=float)
    faulted = truth != NOMINAL_LABEL
    costs: list[float] = []
    n_missed = 0
    n_misrouted = 0
    n_unsafe = 0
    n_conservative = 0
    for believed, conf, true_label in zip(
        labels[faulted], confidence[faulted], truth[faulted], strict=True
    ):
        if oracle:
            cost = policy.oracle_cost(mode_id, true_label)
            costs.append(cost)
            if not math.isfinite(cost):
                n_unsafe += 1
            continue
        decision = policy.decide(mode_id, believed, conf)
        if decision.response_id is None:
            n_missed += 1
            continue
        cost = policy.cost_of(decision, true_label)
        costs.append(cost)
        if decision.routed_conservative:
            n_conservative += 1
        if policy.is_misrouted(decision, true_label):
            n_misrouted += 1
        if not math.isfinite(cost):
            n_unsafe += 1
    fa_costs: list[float] = []
    n_false_alarms = 0
    if not oracle:
        for believed, conf in zip(labels[~faulted], confidence[~faulted], strict=True):
            decision = policy.decide(mode_id, believed, conf)
            if decision.response_id is None:
                fa_costs.append(0.0)
                continue
            n_false_alarms += 1
            fa_costs.append(policy.cost_of(decision, NOMINAL_LABEL))
    return PolicyScore(
        name=name,
        confidence_threshold=policy.confidence_threshold,
        n_fault_events=int(faulted.sum()),
        n_commanded=len(costs),
        n_missed=n_missed,
        mean_cost_min=float(np.mean(costs)) if costs else float("nan"),
        n_misrouted=n_misrouted,
        n_unsafe=n_unsafe,
        conservative_fraction=(n_conservative / len(costs)) if costs else float("nan"),
        false_alarm_cost_min=float(np.mean(fa_costs)) if fa_costs else 0.0,
        n_false_alarms=n_false_alarms,
    )


def compare_policies(
    spec: SafeModeSpec,
    comparison: dict[str, object],
    *,
    mode_id: str | None = None,
    confidence_threshold: float = 0.7,
    assumption: DetectionAssumption | None = None,
    detector: str = "learned",
) -> dict[str, object]:
    """Score four policies on the test split of an existing detector comparison.

    Args:
        spec: the declaration.
        comparison: the mapping returned by :func:`compare_detectors`.
        mode_id: the mode the events are assumed to occur in. Defaults to the
            declaration's initial mode.
        confidence_threshold: the routing threshold for the confidence-routed
            policy, dimensionless and >= 0. The always-conservative arm uses 2,
            because a threshold of exactly 1 leaves an identification whose
            confidence is exactly 1 -- which declared threshold logic emits on
            an exact signature match -- unrouted.
        assumption: detection assumption; default as in
            :class:`safemodetree.policy.FaultResponsePolicy`.
        detector: ``"learned"`` to route on the learned classifier's matched
            operating point, ``"baseline"`` to route on the declared threshold
            logic.

    Returns:
        A mapping with ``"oracle"``, ``"always_specific"``,
        ``"confidence_routed"``, ``"always_conservative"`` (all
        :class:`PolicyScore`), plus ``"cost_of_conservatism_vs_oracle"`` and
        ``"cost_of_conservatism_vs_specific"`` in minutes, and the mode and
        detector used.
    """
    if detector not in ("learned", "baseline"):
        raise ValueError(f"detector must be 'learned' or 'baseline', got {detector!r}")
    mode_id = mode_id or spec.initial_mode
    spec.mode(mode_id)
    split: Split = comparison["split"]  # type: ignore[assignment]
    if detector == "learned":
        model: LearnedFaultClassifier = comparison["model"]  # type: ignore[assignment]
        labels, confidence = learned_decisions(
            model, split.test.features, float(comparison["tau"])  # type: ignore[arg-type]
        )
    else:
        baseline: ThresholdLogicClassifier = comparison["baseline_model"]  # type: ignore[assignment]
        labels, confidence = baseline.predict_with_confidence(split.test.features)
    truth = split.test.labels

    scores: dict[str, object] = {}
    for name, threshold, oracle in (
        ("oracle", 0.0, True),
        ("always_specific", 0.0, False),
        ("confidence_routed", float(confidence_threshold), False),
        ("always_conservative", 2.0, False),
    ):
        policy = FaultResponsePolicy(spec, assumption, confidence_threshold=threshold)
        scores[name] = _score_policy(
            name, policy, mode_id, labels, confidence, truth, oracle=oracle
        )
    routed: PolicyScore = scores["confidence_routed"]  # type: ignore[assignment]
    oracle_score: PolicyScore = scores["oracle"]  # type: ignore[assignment]
    specific: PolicyScore = scores["always_specific"]  # type: ignore[assignment]
    scores["cost_of_conservatism_vs_oracle"] = routed.mean_cost_min - oracle_score.mean_cost_min
    scores["cost_of_conservatism_vs_specific"] = routed.mean_cost_min - specific.mean_cost_min
    scores["misroutes_avoided"] = specific.n_misrouted - routed.n_misrouted
    scores["mode"] = mode_id
    scores["detector"] = detector
    return scores


def conservatism_sweep(
    spec: SafeModeSpec,
    comparison: dict[str, object],
    *,
    thresholds: tuple[float, ...] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0),
    mode_id: str | None = None,
    detector: str = "learned",
) -> tuple[PolicyScore, ...]:
    """The cost / mis-route trade curve over routing thresholds.

    One :class:`PolicyScore` per threshold, so the price of each additional
    avoided mis-route can be read off directly.
    """
    out: list[PolicyScore] = []
    for threshold in thresholds:
        scores = compare_policies(
            spec,
            comparison,
            mode_id=mode_id,
            confidence_threshold=threshold,
            detector=detector,
        )
        routed: PolicyScore = scores["confidence_routed"]  # type: ignore[assignment]
        out.append(
            PolicyScore(
                name=f"threshold {threshold:.2f}",
                confidence_threshold=threshold,
                n_fault_events=routed.n_fault_events,
                n_commanded=routed.n_commanded,
                n_missed=routed.n_missed,
                mean_cost_min=routed.mean_cost_min,
                n_misrouted=routed.n_misrouted,
                n_unsafe=routed.n_unsafe,
                conservative_fraction=routed.conservative_fraction,
                false_alarm_cost_min=routed.false_alarm_cost_min,
                n_false_alarms=routed.n_false_alarms,
            )
        )
    return tuple(out)
