"""Campaign search strategies: two classical baselines and the learned one.

All three consume the same interface -- a finite pool of cases plus an oracle
that returns the severity of a pool index -- and all three return the pool
indices they chose, in execution order, so that "severe cases found after b
executions" is comparable between them on the same budget.

The two baselines were implemented and benchmarked before the learned strategy
existed, and they stay in the benchmark whatever the learned strategy scores.

Strategies
----------
``uniform_random``   sample the pool without replacement. The honest reference:
                     anything that cannot beat this is not worth shipping.
``coverage_greedy``  always execute a case whose coverage cell is still
                     uncovered, breaking ties at random; fall back to uniform
                     random once every reachable cell is covered. This is what a
                     careful engineer does by hand.
``kind_mean``        non-learned ablation: rank untried cases by the running
                     mean severity of their fault kind. Present so a reader can
                     see how much of the learned model's score comes from
                     nothing more than "which kind is this".
``learned``          warm up at random, then fit a
                     :class:`~faultinject.prioritizer.CampaignPrioritizer` and
                     execute the untried case with the highest acquisition
                     value, refitting periodically.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .campaign import FaultCase
from .coverage import CoverageTracker
from .prioritizer import CampaignPrioritizer
from .severity import SEVERE_THRESHOLD
from .taxonomy import kinds

Oracle = Callable[[int], float]


@dataclass
class SearchResult:
    """What a strategy executed, in order, and what it found."""

    strategy: str
    order: list[int]
    severities: list[float]
    coverage_curve: list[float] = field(default_factory=list)

    @property
    def n_severe(self) -> int:
        """Number of executed cases at or above the severity threshold."""
        return sum(1 for s in self.severities if s >= SEVERE_THRESHOLD)

    @property
    def found_curve(self) -> list[int]:
        """Cumulative severe count after each execution."""
        out, total = [], 0
        for s in self.severities:
            total += 1 if s >= SEVERE_THRESHOLD else 0
            out.append(total)
        return out

    @property
    def mean_severity(self) -> float:
        return float(np.mean(self.severities)) if self.severities else 0.0


def _check(pool: Sequence[FaultCase], budget: int) -> None:
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if budget > len(pool):
        raise ValueError(f"budget {budget} exceeds pool size {len(pool)}")


def _coverage_curve(pool: Sequence[FaultCase], order: Sequence[int]) -> list[float]:
    tracker = CoverageTracker(kinds())
    out = []
    for i in order:
        tracker.add(pool[i].injection, pool[i].n_steps)
        out.append(tracker.fraction)
    return out


def uniform_random(
    pool: Sequence[FaultCase], oracle: Oracle, budget: int, rng: np.random.Generator
) -> SearchResult:
    """Uniform random search without replacement. Baseline 1."""
    _check(pool, budget)
    order = [int(i) for i in rng.permutation(len(pool))[:budget]]
    sev = [oracle(i) for i in order]
    return SearchResult("uniform_random", order, sev, _coverage_curve(pool, order))


def coverage_greedy(
    pool: Sequence[FaultCase], oracle: Oracle, budget: int, rng: np.random.Generator
) -> SearchResult:
    """Coverage-greedy search: prefer cases in uncovered cells. Baseline 2."""
    _check(pool, budget)
    shuffled = [int(i) for i in rng.permutation(len(pool))]
    cells = {i: pool[i].cell() for i in shuffled}
    covered: set = set()
    order: list[int] = []
    remaining = list(shuffled)
    while len(order) < budget:
        pick = None
        for pos, idx in enumerate(remaining):
            if cells[idx] not in covered:
                pick = pos
                break
        if pick is None:
            pick = 0
        idx = remaining.pop(pick)
        covered.add(cells[idx])
        order.append(idx)
    sev = [oracle(i) for i in order]
    return SearchResult("coverage_greedy", order, sev, _coverage_curve(pool, order))


def kind_mean(
    pool: Sequence[FaultCase],
    oracle: Oracle,
    budget: int,
    rng: np.random.Generator,
    warmup: int = 16,
) -> SearchResult:
    """Non-learned ablation: rank by the running mean severity of the fault kind."""
    _check(pool, budget)
    if warmup >= budget:
        warmup = max(2, budget // 2)
    shuffled = [int(i) for i in rng.permutation(len(pool))]
    order = shuffled[:warmup]
    sev = [oracle(i) for i in order]
    totals: dict[str, list[float]] = {}
    for i, s in zip(order, sev, strict=True):
        totals.setdefault(pool[i].injection.kind.value, []).append(s)
    tried = set(order)
    while len(order) < budget:
        means = {k: float(np.mean(v)) for k, v in totals.items()}
        best_idx, best_score = None, -1.0
        for i in shuffled:
            if i in tried:
                continue
            s = means.get(pool[i].injection.kind.value, 0.5)
            if s > best_score:
                best_idx, best_score = i, s
        assert best_idx is not None
        order.append(best_idx)
        tried.add(best_idx)
        value = oracle(best_idx)
        sev.append(value)
        totals.setdefault(pool[best_idx].injection.kind.value, []).append(value)
    return SearchResult("kind_mean", order, sev, _coverage_curve(pool, order))


def learned(
    pool: Sequence[FaultCase],
    oracle: Oracle,
    budget: int,
    rng: np.random.Generator,
    warmup: int = 32,
    refit_every: int = 8,
    kappa: float = 0.0,
    n_estimators: int = 60,
) -> tuple[SearchResult, CampaignPrioritizer]:
    """Learned prioritiser search.

    Parameters
    ----------
    warmup
        Cases executed uniformly at random before the first fit. Those
        executions count against the budget, as they must.
    refit_every
        Refit after this many further executions.
    kappa
        Acquisition is ``mean + kappa * std`` from the prioritiser's
        uncertainty output. ``0`` is pure exploitation by predicted severity,
        which is what the specification asks the learned strategy to do.
    n_estimators
        Forest size.

    Returns
    -------
    (SearchResult, CampaignPrioritizer)
        The fitted prioritiser is returned so the caller can report its
        uncertainty calibration and feature importances.
    """
    _check(pool, budget)
    if warmup < 2:
        raise ValueError(f"warmup must be >= 2, got {warmup}")
    if warmup >= budget:
        raise ValueError(f"warmup {warmup} must be smaller than budget {budget}")
    if refit_every < 1:
        raise ValueError(f"refit_every must be >= 1, got {refit_every}")

    shuffled = [int(i) for i in rng.permutation(len(pool))]
    order = shuffled[:warmup]
    sev = [oracle(i) for i in order]
    tried = set(order)
    seed = int(rng.integers(0, 2**31 - 1))
    model = CampaignPrioritizer(n_estimators=n_estimators, random_state=seed)
    model.fit([pool[i] for i in order], sev)
    since_fit = 0

    ranked: list[int] = []

    def rerank() -> list[int]:
        untried = [i for i in shuffled if i not in tried]
        pred = model.predict([pool[i] for i in untried])
        scores = pred.acquisition(kappa)
        return [untried[j] for j in np.argsort(-scores, kind="stable")]

    ranked = rerank()
    while len(order) < budget:
        # The model is constant between refits, so the ranking is too: popping
        # from the ranked list is identical to re-running argmax every pick,
        # and avoids refitting-free re-predictions of the whole untried set.
        while ranked and ranked[0] in tried:
            ranked.pop(0)
        if not ranked:
            ranked = rerank()
        best = ranked.pop(0)
        order.append(best)
        tried.add(best)
        sev.append(oracle(best))
        since_fit += 1
        if since_fit >= refit_every and len(order) < budget:
            model.fit([pool[i] for i in order], sev)
            since_fit = 0
            ranked = rerank()

    return SearchResult("learned", order, sev, _coverage_curve(pool, order)), model


STRATEGIES = ("uniform_random", "coverage_greedy", "kind_mean", "learned")
