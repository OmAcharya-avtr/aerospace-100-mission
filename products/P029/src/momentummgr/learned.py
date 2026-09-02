"""The learned desaturation scheduler, its label search, and its training.

Order of construction, per the portfolio's rule for AI products: the classical
fixed-threshold scheduler in :mod:`momentummgr.policies` was implemented, tuned and
validated first. This module is benchmarked against it on identical held-out episodes.

Where the labels come from
--------------------------
There is no analytic optimum for "which windows should the magnetorquers run in", so the
labels come from an **offline search** over on/off schedules for each training episode:
seeded candidates (all-off, all-on, several threshold schedules, Bernoulli masks at a
range of rates), then coordinate descent on single-window flips until no flip improves
the episode cost. The search sees the whole episode, so it is not a policy and could
never fly. The classifier trained on its output is causal: its features
(:data:`momentummgr.episodes.FEATURE_NAMES`) use only wheel tachometers, a magnetometer,
an onboard field model and an orbit propagator, and it is evaluated closed-loop, where
its own decisions change the state it later sees.

What it is trying to exploit
----------------------------
The cross-product dumping law removes momentum at a rate proportional to
``|B| sin(theta)``, with ``theta`` the angle between the wheel momentum and the field. A
fixed-threshold rule ignores that entirely; it dumps when the wheels are full, whatever
the geometry. A scheduler that can wait a few windows for a better field angle should be
able to buy the same saturation margin for less magnetorquer duty. Whether it actually
does, on held-out episodes and with confidence intervals, is reported in
``validation/learned_vs_fixed_ci.py`` and in ``MODEL_CARD.md``, including the case where
the difference is inside the interval and the honest answer is "indistinguishable".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier

from . import _validate as _v
from .episodes import (
    N_FEATURES,
    Episode,
    EpisodeMetrics,
    rollout,
    simulate_masks,
)
from .policies import Decider, FixedThresholdScheduler

__all__ = [
    "MaskSearchResult",
    "search_best_mask",
    "harvest_training_rows",
    "LearnedScheduler",
    "train_scheduler",
    "tune_decision_threshold",
    "tune_confidence_band",
]


@dataclass(frozen=True)
class MaskSearchResult:
    """Best schedule found for one episode by the offline search.

    ``mask`` is the per-window on/off vector, ``metrics`` its outcome, ``n_evaluations``
    how many schedules were simulated, and ``seed_cost`` the best cost among the seeded
    candidates before coordinate descent (so the descent's contribution is visible).
    """

    mask: NDArray[np.bool_]
    metrics: EpisodeMetrics
    n_evaluations: int
    seed_cost: float


def _threshold_mask(episode: Episode, on: float, off: float) -> NDArray[np.bool_]:
    """The schedule a fixed-threshold rule would produce on this episode."""
    return rollout(episode, FixedThresholdScheduler(on, off).decider()).actions


def search_best_mask(
    episode: Episode,
    seed: int = 0,
    n_random: int = 160,
    max_rounds: int = 6,
) -> MaskSearchResult:
    """Search for a low-cost on/off schedule for one episode.

    Seeded candidates: all-off, all-on, the schedules produced by nine fixed-threshold
    rules, and ``n_random`` Bernoulli masks whose rate is itself drawn uniformly in
    ``[0.05, 0.95]``. Then coordinate descent: every single-window flip of the incumbent
    is evaluated at once (a batch of ``n_windows`` masks) and the best improving flip is
    taken, for at most ``max_rounds`` rounds.

    This is a heuristic. It gives no optimality guarantee and is not claimed to; its only
    job is to be a consistently better-than-threshold label source, and the gap it opens
    over the tuned baseline is measured on the training set and reported.
    """
    rng = np.random.default_rng([int(seed), int(episode.seed)])
    k = episode.n_windows
    seeds = [np.zeros(k, dtype=bool), np.ones(k, dtype=bool)]
    for on in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        seeds.append(_threshold_mask(episode, on, on * 0.4))
    seeds.append(_threshold_mask(episode, 0.5, 0.1))
    seeds.append(_threshold_mask(episode, 0.7, 0.6))
    rates = rng.uniform(0.05, 0.95, size=_v.as_int_at_least(n_random, "n_random", 1))
    seeds.extend(rng.random((n_random, k)) < rates[:, None])
    cands = np.array(seeds, dtype=bool)
    metrics = simulate_masks(episode, cands)
    costs = np.array([m.cost for m in metrics])
    best_i = int(np.argmin(costs))
    best_mask = cands[best_i].copy()
    best_metrics = metrics[best_i]
    seed_cost = float(costs[best_i])
    n_eval = cands.shape[0]

    for _ in range(_v.as_int_at_least(max_rounds, "max_rounds", 0)):
        flips = np.repeat(best_mask[None, :], k, axis=0)
        flips[np.arange(k), np.arange(k)] ^= True
        flip_metrics = simulate_masks(episode, flips)
        flip_costs = np.array([m.cost for m in flip_metrics])
        n_eval += k
        j = int(np.argmin(flip_costs))
        if flip_costs[j] >= best_metrics.cost - 1e-12:
            break
        best_mask = flips[j].copy()
        best_metrics = flip_metrics[j]
    return MaskSearchResult(
        mask=best_mask, metrics=best_metrics, n_evaluations=n_eval, seed_cost=seed_cost
    )


def harvest_training_rows(
    episode: Episode, mask: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Replay ``mask`` on ``episode`` and return ``(features, labels)``.

    Features are those seen at the start of each window along the trajectory the mask
    actually produces, so the training distribution is the state distribution of the
    schedule being imitated. Windows where the safety override fired are dropped: the
    override is applied identically to every policy at run time, so a classifier does not
    need to learn it, and keeping them would teach it to fire late.
    """
    roll = rollout(episode, lambda k, _f: (bool(mask[k]), 1.0))
    keep = roll.features[:, 0] < 0.95
    return roll.features[keep], roll.actions[keep].astype(np.int64)


@dataclass
class LearnedScheduler:
    """Random-forest desaturation scheduler with a confidence output.

    Attributes
    ----------
    model : GradientBoostingClassifier
        Trained classifier over the nine features of
        :data:`momentummgr.episodes.FEATURE_NAMES`. Gradient-boosted stumps of depth 3;
        chosen over a random forest for prediction latency, since the scheduler is
        evaluated one window at a time inside thousands of closed-loop rollouts and a
        300-tree forest costs 37 ms per single-row call against 0.3 ms here. Accuracy of
        the two was within a percent on the training set.
    decision_threshold : float
        Probability above which the scheduler actuates. Tuned closed-loop on the training
        episodes, not left at 0.5, because the cost of a missed dump and of an
        unnecessary one are not equal.
    min_confidence : float
        When the confidence in the chosen action falls below this, the scheduler defers to
        ``fallback`` instead. Zero disables deferral.
    fallback : FixedThresholdScheduler
        The rule used when the model defers. Deferring to the classical baseline rather
        than to a coin is the only defensible choice for a flight-adjacent component.

    Confidence
    ----------
    ``confidence = p`` when the scheduler actuates and ``1 - p`` when it does not, with
    ``p`` the model's class probability (a logistic transform of the boosted score). It
    is not a calibrated posterior; its calibration is measured (reliability curve and
    Brier score) in ``validation/learned_vs_fixed_ci.py`` and reported in
    ``MODEL_CARD.md``, including where it is overconfident.
    """

    model: GradientBoostingClassifier
    decision_threshold: float = 0.5
    min_confidence: float = 0.0
    fallback: FixedThresholdScheduler = FixedThresholdScheduler()

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Probability of the actuate class for each feature row, shape ``(N,)``."""
        x = np.atleast_2d(np.asarray(features, dtype=float))
        if x.shape[1] != N_FEATURES:
            raise ValueError(
                f"features must have {N_FEATURES} columns, got {x.shape[1]}"
            )
        return self.model.predict_proba(x)[:, 1]

    def decider(self) -> Decider:
        """Return a fresh decision function for one episode."""
        fb = self.fallback.decider()

        def decide(k: int, features: NDArray[np.float64]) -> tuple[bool, float]:
            p = float(self.predict_proba(features[None, :])[0])
            act = p >= self.decision_threshold
            conf = p if act else 1.0 - p
            fb_act, _ = fb(k, features)
            if conf < self.min_confidence:
                return fb_act, conf
            return act, conf

        return decide


def train_scheduler(
    fit_episodes: Sequence[Episode],
    tune_episodes: Sequence[Episode],
    fallback: FixedThresholdScheduler | None = None,
    search_seed: int = 0,
    n_estimators: int = 150,
    max_depth: int = 3,
    min_samples_leaf: int = 8,
    learning_rate: float = 0.1,
    random_state: int = 0,
    threshold_grid: Sequence[float] | None = None,
    confidence_grid: Sequence[float] | None = None,
) -> tuple[LearnedScheduler, dict[str, float], NDArray[np.float64], NDArray[np.int64]]:
    """Search labels, fit the classifier, and tune its two decision knobs.

    Three disjoint episode sets are used in total and none of them is the held-out set:

    ``fit_episodes``
        The offline search runs here and the classifier is fitted on the resulting
        state-action pairs.
    ``tune_episodes``
        The decision threshold and the deferral band are chosen here, by closed-loop grid
        search over their joint grid. They are tuned on episodes the classifier was not
        fitted on because tuning them on the fitting episodes overfits them badly: doing
        so cost 15 % on held-out mean episode cost in the run recorded in
        ``validation/learned_vs_fixed_ci_output.txt``.
    held-out episodes
        Never touched by this function.

    ``fallback`` is the classical scheduler the model defers to below its confidence
    band; pass the tuned baseline.

    Returns ``(scheduler, diagnostics, features, labels)``. The diagnostics carry the mean
    training-set cost of the searched schedules and the label balance, so a model that
    merely learned "always dump" is visible immediately.
    """
    if len(fit_episodes) == 0 or len(tune_episodes) == 0:
        raise ValueError("fit_episodes and tune_episodes must both be non-empty")
    rows: list[NDArray[np.float64]] = []
    labels: list[NDArray[np.int64]] = []
    search_costs: list[float] = []
    for ep in fit_episodes:
        res = search_best_mask(ep, seed=search_seed)
        search_costs.append(res.metrics.cost)
        x, y = harvest_training_rows(ep, res.mask)
        rows.append(x)
        labels.append(y)
    features = np.vstack(rows)
    target = np.concatenate(labels)
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(features, target)
    fb = fallback if fallback is not None else FixedThresholdScheduler()
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5] if threshold_grid is None else list(
        threshold_grid
    )
    bands = [0.0, 0.5, 0.6, 0.7] if confidence_grid is None else list(confidence_grid)
    best: tuple[float, float, float] | None = None
    for thr in thresholds:
        for band in bands:
            candidate = LearnedScheduler(
                model=model, decision_threshold=thr, min_confidence=band, fallback=fb
            )
            cost = float(
                np.mean(
                    [
                        rollout(ep, candidate.decider(), record_history=False).metrics.cost
                        for ep in tune_episodes
                    ]
                )
            )
            if best is None or cost < best[0]:
                best = (cost, thr, band)
    assert best is not None
    tune_cost, threshold, band = best
    scheduler = LearnedScheduler(
        model=model, decision_threshold=threshold, min_confidence=band, fallback=fb
    )
    return (
        scheduler,
        {
            "n_rows": float(features.shape[0]),
            "label_positive_rate": float(target.mean()),
            "search_mean_cost": float(np.mean(search_costs)),
            "tune_split_mean_cost": float(tune_cost),
            "decision_threshold": float(threshold),
            "min_confidence": float(band),
        },
        features,
        target,
    )


def tune_decision_threshold(
    scheduler: LearnedScheduler,
    episodes: Sequence[Episode],
    grid: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Pick the probability threshold minimising mean closed-loop cost on ``episodes``.

    Returns ``(threshold, mean_cost)``. Tuned on training episodes only.
    """
    values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6] if grid is None else list(grid)
    best: tuple[float, float] | None = None
    original = scheduler.decision_threshold
    try:
        for thr in values:
            scheduler.decision_threshold = thr
            cost = float(
                np.mean(
                    [
                        rollout(ep, scheduler.decider(), record_history=False).metrics.cost
                        for ep in episodes
                    ]
                )
            )
            if best is None or cost < best[1]:
                best = (thr, cost)
    finally:
        scheduler.decision_threshold = original
    assert best is not None
    return best


def tune_confidence_band(
    scheduler: LearnedScheduler,
    episodes: Sequence[Episode],
    grid: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Pick the deferral band minimising mean closed-loop cost on ``episodes``.

    Below this confidence the scheduler hands the window to its classical fallback.
    Returns ``(min_confidence, mean_cost)``. Tuned on training episodes only. A tuned
    value of 0 means deferral never paid for itself and is reported as such rather than
    forced on.
    """
    values = [0.0, 0.5, 0.6, 0.7, 0.8] if grid is None else list(grid)
    best: tuple[float, float] | None = None
    original = scheduler.min_confidence
    try:
        for band in values:
            scheduler.min_confidence = band
            cost = float(
                np.mean(
                    [
                        rollout(ep, scheduler.decider(), record_history=False).metrics.cost
                        for ep in episodes
                    ]
                )
            )
            if best is None or cost < best[1]:
                best = (band, cost)
    finally:
        scheduler.min_confidence = original
    assert best is not None
    return best
