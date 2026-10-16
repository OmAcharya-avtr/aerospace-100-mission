"""The three strategies being compared, plus a coverage-greedy upper reference.

All four take the same context -- a declared space, the already-executed
scenarios, and a candidate pool -- and return the indices of the scenarios they
would execute next, in order. They differ only in what they use to choose.

1. :class:`UniformRandomStrategy` -- sample without replacement from the pool.
   The null strategy. Any method that cannot beat it is not worth the code.
2. :class:`GreedyMaxMinStrategy` -- farthest-point sampling in the normalised
   embedding: repeatedly take the candidate whose distance to the nearest
   already-selected or already-executed point is largest.

       x_next = argmax_{c in pool} min_{s in selected} ||c - s||_2        (5)

   Source: T. F. Gonzalez, "Clustering to minimize the maximum intercluster
   distance", *Theoretical Computer Science* 38, 293-306, 1985 -- the greedy
   farthest-point rule and its factor-2 approximation guarantee for the k-centre
   objective. The design-of-experiments form of the same criterion is M. E.
   Johnson, L. M. Moore and D. Ylvisaker, "Minimax and maximin distance
   designs", *Journal of Statistical Planning and Inference* 26(2), 131-148,
   1990. Units: distances are in the dimensionless normalised embedding of
   :meth:`ScenarioSpace.normalise`.
3. :class:`LearnedGapFinderStrategy` -- the AI component, :class:`GapFinder`.
4. :class:`CoverageGreedyReference` -- picks the candidate that adds the most
   new t-way tuples. It optimises the benchmark metric directly, so it is
   **not a competitor**: it is the reference line showing how much of the
   achievable gain the three strategies under test leave behind. It needs the
   declared discretisation and the chosen ``t``, which the other three do not.

The greedy max-min rule (5) is deterministic given the context, so its only
run-to-run variation comes from the seeded candidate pool and the seeded
starting campaign.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .gapfinder import GapFinder
from .space import ScenarioSpace

__all__ = [
    "CoverageGreedyReference",
    "GreedyMaxMinStrategy",
    "LearnedGapFinderStrategy",
    "SelectionContext",
    "Strategy",
    "UniformRandomStrategy",
    "default_strategies",
]


@dataclass(frozen=True)
class SelectionContext:
    """Everything a strategy may look at.

    Attributes:
        space: the declared scenario space.
        executed_encoded: ``(n_executed, k)`` integer level indices already run.
        executed_norm: ``(n_executed, d)`` normalised embedding of the same.
        candidate_encoded: ``(n_candidates, k)`` integer level indices on offer.
        candidate_norm: ``(n_candidates, d)`` normalised embedding of the same.
    """

    space: ScenarioSpace
    executed_encoded: np.ndarray
    executed_norm: np.ndarray
    candidate_encoded: np.ndarray
    candidate_norm: np.ndarray

    def __post_init__(self) -> None:
        if self.executed_encoded.shape[0] != self.executed_norm.shape[0]:
            raise ValueError("executed_encoded and executed_norm disagree on row count")
        if self.candidate_encoded.shape[0] != self.candidate_norm.shape[0]:
            raise ValueError("candidate_encoded and candidate_norm disagree on row count")
        if self.candidate_norm.shape[1] != self.executed_norm.shape[1]:
            raise ValueError("executed and candidate embeddings have different widths")

    @property
    def n_candidates(self) -> int:
        """Size of the candidate pool."""
        return int(self.candidate_norm.shape[0])


class Strategy:
    """Base class: choose ``n`` scenarios from the candidate pool."""

    name = "base"

    def select(
        self, context: SelectionContext, n: int, rng: np.random.Generator
    ) -> tuple[int, ...]:
        """Return ``n`` candidate indices, in execution order."""
        raise NotImplementedError

    @staticmethod
    def _check(context: SelectionContext, n: int) -> None:
        if not 1 <= n <= context.n_candidates:
            raise ValueError(
                f"n must satisfy 1 <= n <= {context.n_candidates} (pool size), got {n}"
            )


class UniformRandomStrategy(Strategy):
    """Uniform sampling without replacement from the candidate pool."""

    name = "uniform_random"

    def select(
        self, context: SelectionContext, n: int, rng: np.random.Generator
    ) -> tuple[int, ...]:
        """Draw ``n`` distinct candidate indices uniformly at random."""
        self._check(context, n)
        return tuple(int(i) for i in rng.choice(context.n_candidates, size=n, replace=False))


class GreedyMaxMinStrategy(Strategy):
    """Greedy farthest-point sampling, equation (5)."""

    name = "greedy_maxmin"

    def select(
        self, context: SelectionContext, n: int, rng: np.random.Generator
    ) -> tuple[int, ...]:
        """Return the ``n`` farthest-point picks, in selection order.

        Ties are broken by the lowest candidate index, which makes the result
        reproducible but means a pool with many equidistant candidates is
        resolved by pool order rather than by anything meaningful.
        """
        self._check(context, n)
        pool = context.candidate_norm
        if context.executed_norm.shape[0]:
            min_dist = _min_distance_to(pool, context.executed_norm)
        else:
            min_dist = np.full(pool.shape[0], np.inf)
        chosen: list[int] = []
        for _ in range(n):
            pick = int(np.argmax(min_dist))
            chosen.append(pick)
            min_dist = np.minimum(min_dist, _min_distance_to(pool, pool[pick : pick + 1]))
            min_dist[pick] = -np.inf
        return tuple(chosen)


class LearnedGapFinderStrategy(Strategy):
    """The learned gap finder used as a selection strategy."""

    name = "learned_gapfinder"

    def __init__(
        self,
        bandwidth: str | float = "scott",
        n_bootstrap: int = 0,
        sparsity_quantile: float = 0.10,
        min_separation: float = 0.5,
        sequential: bool = True,
        refit: bool = True,
        name: str | None = None,
    ) -> None:
        """Construct the strategy.

        Args:
            bandwidth: passed through to :class:`GapFinder`.
            n_bootstrap: 0 in the benchmark, where the uncertainty output is
                not used and the bootstrap would multiply the cost by
                ``n_bootstrap``. The examples and the CLI use the default 16.
            sparsity_quantile: passed through to :class:`GapFinder`.
            min_separation: passed through to :class:`GapFinder`.
            sequential: passed through to :meth:`GapFinder.propose`.
            refit: passed through to :meth:`GapFinder.propose`. The three
                combinations of ``sequential`` and ``refit`` are the batch-rule
                ablation in validation/validate_benchmark_output.txt.
            name: overrides the strategy name, so an ablation can put several
                configurations in one benchmark.
        """
        self.bandwidth = bandwidth
        self.n_bootstrap = int(n_bootstrap)
        self.sparsity_quantile = float(sparsity_quantile)
        self.min_separation = float(min_separation)
        self.sequential = bool(sequential)
        self.refit = bool(refit)
        if name is not None:
            self.name = str(name)

    def select(
        self, context: SelectionContext, n: int, rng: np.random.Generator
    ) -> tuple[int, ...]:
        """Fit the density on the executed set and propose ``n`` sparse candidates."""
        self._check(context, n)
        finder = GapFinder(
            bandwidth=self.bandwidth,
            n_bootstrap=self.n_bootstrap,
            sparsity_quantile=self.sparsity_quantile,
            min_separation=self.min_separation,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        finder.fit(context.executed_norm)
        proposals = finder.propose(
            context.candidate_norm, n, sequential=self.sequential, refit=self.refit
        )
        return tuple(p.candidate_index for p in proposals)


class CoverageGreedyReference(Strategy):
    """Upper reference: greedily add the candidate covering the most new t-tuples.

    This optimises the benchmark metric directly and is reported as a ceiling,
    not as one of the three strategies under test. Cost is ``O(n_candidates *
    C(k, t))`` set lookups per step.
    """

    name = "coverage_greedy_reference"

    def __init__(self, t: int = 2) -> None:
        """Args: t: interaction strength to be greedy about."""
        if t < 1:
            raise ValueError("t must be at least 1")
        self.t = int(t)

    def select(
        self, context: SelectionContext, n: int, rng: np.random.Generator
    ) -> tuple[int, ...]:
        """Return the ``n`` greedy coverage picks, in selection order."""
        self._check(context, n)
        k = len(context.space)
        if self.t > k:
            raise ValueError(f"t={self.t} exceeds the parameter count {k}")
        subsets = list(combinations(range(k), self.t))
        covered: set[tuple[int, tuple[int, ...]]] = set()
        for row in context.executed_encoded:
            for s_index, subset in enumerate(subsets):
                covered.add((s_index, tuple(int(row[i]) for i in subset)))
        candidate_tuples = [
            [
                (s_index, tuple(int(row[i]) for i in subset))
                for s_index, subset in enumerate(subsets)
            ]
            for row in context.candidate_encoded
        ]
        chosen: list[int] = []
        taken: set[int] = set()
        for _ in range(n):
            best, best_gain = -1, -1
            for idx, tuples in enumerate(candidate_tuples):
                if idx in taken:
                    continue
                gain = sum(1 for key in tuples if key not in covered)
                if gain > best_gain:
                    best, best_gain = idx, gain
            taken.add(best)
            chosen.append(best)
            covered.update(candidate_tuples[best])
        return tuple(chosen)


def _min_distance_to(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to the nearest reference point.

    Args:
        points: ``(n, d)`` array.
        reference: ``(m, d)`` array, ``m >= 1``.

    Returns:
        ``(n,)`` array of distances in the units of the embedding.
    """
    diff = points[:, None, :] - reference[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)).min(axis=1)


def default_strategies(t: int = 2) -> tuple[Strategy, ...]:
    """The three strategies under test plus the coverage-greedy reference."""
    return (
        UniformRandomStrategy(),
        GreedyMaxMinStrategy(),
        LearnedGapFinderStrategy(),
        CoverageGreedyReference(t=t),
    )
